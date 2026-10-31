#!/usr/bin/env python3
"""Remove long, visually static stretches from a timelapse video."""

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
from typing import List, Optional, Sequence, Tuple


FRAME_SIZE = 64 * 64


def build_segments(
    activity_times: Sequence[float],
    duration: float,
    min_static_duration: float,
    padding: float,
    sample_interval: float = 1.0,
) -> List[Tuple[float, float]]:
    """Return ranges around activity, joining changes separated by short pauses."""
    if not activity_times:
        return [(0.0, duration)]

    segments = []
    start = previous = activity_times[0]
    for timestamp in activity_times[1:]:
        if timestamp - previous > min_static_duration:
            segments.append(
                (max(0.0, start - sample_interval - padding), min(duration, previous + padding))
            )
            start = timestamp
        previous = timestamp
    segments.append(
        (max(0.0, start - sample_interval - padding), min(duration, previous + padding))
    )

    merged = []
    for start, end in segments:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def probe_video(path: str) -> Tuple[float, bool]:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration:stream=codec_type",
            "-of",
            "json",
            path,
        ],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    metadata = json.loads(result.stdout)
    duration = float(metadata["format"]["duration"])
    has_audio = any(stream.get("codec_type") == "audio" for stream in metadata["streams"])
    return duration, has_audio


def find_activity(path: str, sample_interval: float, threshold: float) -> List[float]:
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        path,
        "-vf",
        (
            f"fps={1.0 / sample_interval:.12g},"
            "scale=64:64:force_original_aspect_ratio=decrease,"
            "pad=64:64:(ow-iw)/2:(oh-ih)/2,format=gray"
        ),
        "-f",
        "rawvideo",
        "-pix_fmt",
        "gray",
        "pipe:1",
    ]
    process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    assert process.stdout is not None
    activity_times = []
    previous: Optional[bytes] = None
    frame_index = 0

    while True:
        frame = process.stdout.read(FRAME_SIZE)
        if not frame:
            break
        if len(frame) != FRAME_SIZE:
            process.kill()
            process.wait()
            raise RuntimeError("FFmpeg returned an incomplete video frame")
        if previous is not None:
            difference = sum(abs(left - right) for left, right in zip(frame, previous)) / FRAME_SIZE
            if difference > 0 and difference >= threshold:
                activity_times.append(frame_index * sample_interval)
        previous = frame
        frame_index += 1

    if process.wait() != 0:
        raise RuntimeError("FFmpeg could not decode the input video")
    process.stdout.close()
    return activity_times


def render_video(
    input_path: str,
    output_path: str,
    segments: Sequence[Tuple[float, float]],
    has_audio: bool,
) -> None:
    filters = []
    for index, (start, end) in enumerate(segments):
        filters.append(
            f"[0:v:0]trim=start={start:.6f}:end={end:.6f},"
            f"setpts=PTS-STARTPTS[v{index}]"
        )
        if has_audio:
            filters.append(
                f"[0:a:0]atrim=start={start:.6f}:end={end:.6f},"
                f"asetpts=PTS-STARTPTS[a{index}]"
            )

    inputs = "".join(f"[v{i}][a{i}]" if has_audio else f"[v{i}]" for i in range(len(segments)))
    output_labels = "[outv][outa]" if has_audio else "[outv]"
    filters.append(
        f"{inputs}concat=n={len(segments)}:v=1:a={1 if has_audio else 0}{output_labels}"
    )
    command = [
        "ffmpeg",
        "-v",
        "error",
        "-i",
        input_path,
        "-filter_complex",
        ";".join(filters),
        "-map",
        "[outv]",
    ]
    if has_audio:
        command.extend(["-map", "[outa]", "-c:a", "aac"])
    command.extend(["-c:v", "libx264", "-crf", "18", "-preset", "medium", "-n", output_path])
    subprocess.run(command, check=True)


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(
        description="Remove long static stretches from a timelapse video."
    )
    parser.add_argument("input", help="source video")
    parser.add_argument("output", help="trimmed output video (must not already exist)")
    parser.add_argument(
        "--sample-interval",
        type=float,
        default=1.0,
        help="seconds between sampled frames (default: 1)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=2.0,
        help="minimum average grayscale pixel change (0-255) to count as activity (default: 2)",
    )
    parser.add_argument(
        "--min-static-duration",
        type=float,
        default=8.0,
        help="join activity separated by pauses up to this many seconds (default: 8)",
    )
    parser.add_argument(
        "--padding",
        type=float,
        default=1.5,
        help="seconds to keep around detected activity (default: 1.5)",
    )
    args = parser.parse_args(argv)

    if (
        not math.isfinite(args.sample_interval)
        or not math.isfinite(args.min_static_duration)
        or not math.isfinite(args.padding)
        or args.sample_interval <= 0
        or args.min_static_duration < 0
        or args.padding < 0
    ):
        parser.error("interval must be positive; static duration and padding cannot be negative")
    if not math.isfinite(args.threshold) or args.threshold < 0 or args.threshold > 255:
        parser.error("threshold must be between 0 and 255")
    if os.path.abspath(args.input) == os.path.abspath(args.output):
        parser.error("input and output must be different files")
    if os.path.exists(args.output):
        parser.error(f"output already exists: {args.output}")
    if not os.path.isfile(args.input):
        parser.error(f"input file not found: {args.input}")
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        parser.error("FFmpeg and ffprobe must be installed and available on PATH")

    try:
        duration, has_audio = probe_video(args.input)
        activity_times = find_activity(args.input, args.sample_interval, args.threshold)
        segments = build_segments(
            activity_times,
            duration,
            args.min_static_duration,
            args.padding,
            args.sample_interval,
        )
        if len(segments) == 1 and segments[0] == (0.0, duration):
            print("No removable static stretches detected; output was not created.")
        else:
            render_video(args.input, args.output, segments, has_audio)
            removed = duration - sum(end - start for start, end in segments)
            print(
                f"Saved {args.output}; retained {sum(end - start for start, end in segments):.1f}s "
                f"and removed {removed:.1f}s."
            )
    except (OSError, subprocess.CalledProcessError, KeyError, ValueError, RuntimeError) as error:
        print(f"time-lapser: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
