"""Video decoding, signatures, output orientation and streaming encoding."""

from __future__ import annotations

import json
import math
import os
import subprocess
import tempfile
from fractions import Fraction

import cv2
import numpy as np

from .runtime import ACTIVE_ENCODERS, PROCESS_LOCK


def open_capture(input_file, threads):
    params = []
    if hasattr(cv2, "CAP_PROP_N_THREADS"):
        params = [cv2.CAP_PROP_N_THREADS, threads]
    cap = cv2.VideoCapture(str(input_file), cv2.CAP_FFMPEG, params)
    if not cap.isOpened() and params:
        cap.release()
        cap = cv2.VideoCapture(str(input_file), cv2.CAP_FFMPEG)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Cannot open video with OpenCV/FFmpeg: {input_file}")
    return cap


def probe_video(input_file, ffprobe):
    command = [
        ffprobe,
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=codec_name,width,height,avg_frame_rate,r_frame_rate,nb_frames,"
        "pix_fmt,bits_per_raw_sample,color_transfer,color_primaries,sample_aspect_ratio:"
        "stream_side_data=rotation",
        "-of",
        "json",
        str(input_file),
    ]
    result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    if result.returncode:
        raise RuntimeError(f"ffprobe failed: {result.stderr.strip()[-2000:]}")
    streams = json.loads(result.stdout).get("streams", [])
    if not streams:
        raise RuntimeError("No video stream found.")
    return streams[0]


def frame_rate(metadata, cap):
    for value in (
        metadata.get("avg_frame_rate"),
        metadata.get("r_frame_rate"),
        str(cap.get(cv2.CAP_PROP_FPS)),
    ):
        try:
            rate = Fraction(value)
            if 0 < rate <= 1000:
                return rate.limit_denominator(1000000)
        except (ValueError, TypeError, ZeroDivisionError, OverflowError):
            pass
    raise RuntimeError(
        "Missing or invalid frame rate; cannot retime this video safely."
    )


def expected_frames(metadata, limit):
    try:
        count = int(metadata.get("nb_frames", 0))
    except (ValueError, TypeError):
        count = 0
    return min(count, limit) if count > 0 and limit is not None else count


def check_frame_count(count, expected):
    # nb_frames comes from container metadata, not OpenCV's duration estimate.
    # A mismatch might mean corrupt media or incorrect metadata: do not publish it.
    if expected > 0 and count != expected:
        raise RuntimeError(
            f"Decoded {count} frames; metadata indicates {expected}. "
            "Possible incomplete decode or incorrect container metadata; no MOV was published."
        )


def crop_for_analysis(frame, roi):
    if roi is None:
        return frame
    height, width = frame.shape[:2]
    x, y, w, h = roi
    left, top = int(x * width), int(y * height)
    right = min(width, max(left + 1, math.ceil((x + w) * width)))
    bottom = min(height, max(top + 1, math.ceil((y + h) * height)))
    return frame[top:bottom, left:right]


def signature(frame, args):
    region = crop_for_analysis(frame, args.roi)
    # Preserve the original 160x90/linear/equalized scoring scale by default.
    small = cv2.resize(region, (160, 90), interpolation=cv2.INTER_LINEAR)
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return gray if args.no_equalize else cv2.equalizeHist(gray)


def export_transform(shape, args):
    """Plan export from decoded pixels; never reapply source rotation metadata."""
    height, width = shape[:2]
    direction = args.rotate_portrait if height > width else "off"
    filters = []
    output_height, output_width = height, width
    clockwise_degrees = 0
    if direction != "off":
        filters.append("transpose=clock" if direction == "cw" else "transpose=cclock")
        clockwise_degrees = 90 if direction == "cw" else -90
        output_height, output_width = width, height
    if args.codec != "prores":
        # Match the existing H.264 padding, applied AFTER any rotation.
        filters.append("pad=ceil(iw/2)*2:ceil(ih/2)*2")
        output_height += output_height % 2
        output_width += output_width % 2
    return {
        "portrait_policy": args.rotate_portrait,
        "clockwise_degrees": clockwise_degrees,
        "decoded_width": width,
        "decoded_height": height,
        "output_width": output_width,
        "output_height": output_height,
        "filters": filters,
        "note": "Pixel rotation at export only, after source orientation metadata; analysis and ROI are unchanged.",
    }


class RawEncoder:
    """One FFmpeg encoder with bounded pipe backpressure and file-backed stderr."""

    def __init__(self, path, shape, fps, args):
        self.path = path
        self.shape = shape
        height, width = shape[:2]
        transform = export_transform(shape, args)
        self.error_log = tempfile.TemporaryFile(mode="w+b")
        cmd = [
            args.ffmpeg,
            "-hide_banner",
            "-loglevel",
            "error",
            "-nostdin",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "bgr24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            str(fps),
            "-i",
            "pipe:0",
            "-map",
            "0:v:0",
            "-an",
            "-filter_threads",
            "1",
        ]
        if args.codec == "prores":
            cmd += [
                "-c:v",
                "prores_ks",
                "-profile:v",
                str(args.prores_profile),
                "-pix_fmt",
                "yuv422p10le",
            ]
        else:
            # 4:2:0 encoders need even dimensions; pad, never crop the picture.
            cmd += ["-pix_fmt", "yuv420p"]
            if args.codec == "h264":
                cmd += ["-c:v", "libx264", "-preset", "veryfast", "-crf", str(args.crf)]
            else:
                cmd += [
                    "-c:v",
                    "h264_nvenc",
                    "-preset",
                    "p4",
                    "-rc",
                    "vbr",
                    "-cq",
                    str(args.crf),
                    "-b:v",
                    "0",
                ]
        if transform["filters"]:
            cmd += ["-vf", ",".join(transform["filters"])]
        cmd += ["-threads", str(args.threads), "-f", "mov", str(path)]
        try:
            self.proc = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self.error_log,
            )
        except BaseException:
            self.error_log.close()
            raise
        with PROCESS_LOCK:
            ACTIVE_ENCODERS.add(self.proc)

    def error_text(self):
        self.error_log.flush()
        self.error_log.seek(0, os.SEEK_END)
        size = self.error_log.tell()
        self.error_log.seek(max(0, size - 4000))
        return self.error_log.read().decode("utf-8", errors="replace").strip()

    def write(self, frame):
        if frame.shape != self.shape or frame.dtype != np.uint8:
            raise RuntimeError("Decoded frame format changed during the video.")
        data = memoryview(np.ascontiguousarray(frame)).cast("B")
        try:
            while data:
                written = self.proc.stdin.write(data)
                if not written:
                    raise BrokenPipeError("FFmpeg stdin closed.")
                data = data[written:]
        except (BrokenPipeError, OSError) as exc:
            self.stop()
            raise RuntimeError(
                f"FFmpeg stopped while encoding. {self.error_text()}"
            ) from exc

    def finish(self):
        broken = False
        try:
            self.proc.stdin.close()
        except (BrokenPipeError, OSError):
            broken = True
        code = self.proc.wait()
        if code or broken or not self.path.is_file() or self.path.stat().st_size == 0:
            raise RuntimeError(f"FFmpeg failed (exit {code}). {self.error_text()}")

    def stop(self):
        if self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait()

    def close(self):
        self.stop()
        try:
            if self.proc.stdin and not self.proc.stdin.closed:
                self.proc.stdin.close()
        except (BrokenPipeError, OSError):
            pass
        with PROCESS_LOCK:
            ACTIVE_ENCODERS.discard(self.proc)
        self.error_log.close()
