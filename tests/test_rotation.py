"""Verify decoded orientation, baked pixel rotation and rotation-only exports."""

import contextlib
import io
import json
import subprocess
import tempfile
import threading
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from castlestube_cleanup import activity, cli, pipeline, video


def sample(height=64, width=32):
    frame = np.empty((height, width, 3), np.uint8)
    frame[: height // 2, : width // 2] = 24
    frame[: height // 2, width // 2 :] = 80
    frame[height // 2 :, : width // 2] = 144
    frame[height // 2 :, width // 2 :] = 224
    return frame


def make_source(path, frame, count=8):
    height, width = frame.shape[:2]
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{width}x{height}",
            "-r",
            "60",
            "-i",
            "-",
            "-c:v",
            "libx264",
            "-crf",
            "0",
            "-pix_fmt",
            "yuv444p",
            str(path),
        ],
        input=frame.tobytes() * count,
        check=True,
    )


def decode(path):
    capture = video.open_capture(path, 2)
    frames = []
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frames.append(frame)
    finally:
        capture.release()
    return frames


class RotationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def args(self, source=None, *extra):
        return cli.arguments(
            [
                str(source or self.root / "input.mov"),
                "--codec",
                "h264",
                "--crf",
                "0",
                "--threads",
                "2",
                "--threshold",
                "0",
                "--report",
                *extra,
            ]
        )

    def run_file(self, source, *extra):
        dest = self.root / "output.mov"
        args = self.args(source, *extra)
        with contextlib.redirect_stdout(io.StringIO()):
            status = pipeline.process_file(source, dest, args, threading.Event())
        report_path = (
            dest.with_name("output_analysis.json")
            if args.analyze_only
            else dest.with_suffix(".json")
        )
        return (status, dest, json.loads(report_path.read_text()))

    def test_defaults_landscape_square_and_odd_h264_padding(self):
        args = self.args()
        self.assertEqual(args.rotate_portrait, "cw")
        for shape in [(32, 64, 3), (32, 32, 3)]:
            self.assertEqual(
                video.export_transform(shape, args)["clockwise_degrees"], 0
            )
        t = video.export_transform((65, 33, 3), args)
        self.assertEqual(
            (t["output_width"], t["output_height"], t["clockwise_degrees"]),
            (66, 34, 90),
        )

    def test_encoder_pixels_both_directions_and_codecs(self):
        source = sample()
        for codec in ("h264", "prores"):
            for direction, rotate in [
                ("cw", cv2.ROTATE_90_CLOCKWISE),
                ("ccw", cv2.ROTATE_90_COUNTERCLOCKWISE),
            ]:
                with self.subTest(codec=codec, direction=direction):
                    path = self.root / f"{codec}_{direction}.mov"
                    args = self.args(
                        None, "--codec", codec, "--rotate-portrait", direction
                    )
                    encoder = video.RawEncoder(path, source.shape, Fraction(60), args)
                    try:
                        encoder.write(source)
                        encoder.finish()
                    finally:
                        encoder.close()
                    frames = decode(path)
                    expected = cv2.rotate(source, rotate)
                    self.assertEqual(len(frames), 1)
                    self.assertEqual(frames[0].shape, expected.shape)
                    self.assertLess(
                        np.abs(frames[0].astype(float) - expected).mean(), 3
                    )
                    meta = video.probe_video(path, "ffprobe")
                    self.assertTrue(
                        all(
                            side.get("rotation", 0) == 0
                            for side in meta.get("side_data_list", [])
                        )
                    )

    def test_portrait_with_all_frames_kept_is_still_written(self):
        source = self.root / "portrait.mov"
        make_source(source, sample())
        status, path, report = self.run_file(source)
        self.assertEqual(status, "written")
        self.assertEqual(
            (report["input_frames"], report["selected_frames"], report["output_fps"]),
            (8, 8, "60"),
        )
        self.assertEqual(report["output_transform"]["clockwise_degrees"], 90)
        self.assertEqual(len(decode(path)), 8)
        self.assertEqual(decode(path)[0].shape, (32, 64, 3))

    def test_landscape_and_off_keep_old_unchanged_behavior(self):
        for name, frame, flags in [
            ("landscape", sample(32, 64), []),
            ("portrait_off", sample(), ["--rotate-portrait", "off"]),
        ]:
            with self.subTest(name=name):
                source = self.root / f"{name}.mov"
                make_source(source, frame)
                status, path, report = self.run_file(source, *flags)
                self.assertEqual(status, "unchanged")
                self.assertFalse(path.exists())
                self.assertEqual(report["output_transform"]["clockwise_degrees"], 0)

    def test_analysis_only_plans_rotation_without_writing(self):
        source = self.root / "portrait.mov"
        make_source(source, sample())
        status, path, report = self.run_file(source, "--analyze-only")
        self.assertEqual(status, "analyzed")
        self.assertFalse(path.exists())
        self.assertEqual(report["decoded_shape"], [64, 32, 3])
        self.assertEqual(report["output_transform"]["output_width"], 64)

    def test_display_matrix_is_applied_only_once_before_export_rotation(self):
        raw = self.root / "raw.mov"
        source = self.root / "tagged.mov"
        make_source(raw, sample(32, 64))
        tagged = subprocess.run(
            [
                "ffmpeg",
                "-v",
                "error",
                "-display_rotation",
                "-90",
                "-i",
                str(raw),
                "-c",
                "copy",
                str(source),
            ],
            capture_output=True,
        )
        if tagged.returncode:
            subprocess.run(
                [
                    "ffmpeg",
                    "-v",
                    "error",
                    "-y",
                    "-i",
                    str(raw),
                    "-c",
                    "copy",
                    "-metadata:s:v:0",
                    "rotate=-90",
                    str(source),
                ],
                check=True,
            )
        before = decode(source)[0]
        self.assertEqual(before.shape, (64, 32, 3))
        status, path, report = self.run_file(source)
        self.assertEqual(status, "written")
        expected = cv2.rotate(before, cv2.ROTATE_90_CLOCKWISE)
        actual = decode(path)[0]
        self.assertEqual(actual.shape, expected.shape)
        self.assertLess(np.abs(actual.astype(float) - expected).mean(), 3)
        self.assertEqual(
            report["source_metadata"]["side_data_list"][0]["rotation"], -90
        )

    def test_hand_roi_and_selection_do_not_change_with_rotation(self):
        source = self.root / "portrait.mov"
        make_source(source, sample(), count=12)

        class Detector:
            def __init__(self):
                self.shapes = []

            def present(self, frame, i, fps):
                self.shapes.append(frame.shape)
                return i in (4, 5)

            def close(self):
                pass

        detector = Detector()
        with patch.object(activity, "create_hand_detector", return_value=detector):
            status, path, report = self.run_file(
                source, "--mode", "hands", "--roi", "0", "0", "0.5", "1"
            )
        self.assertEqual(status, "written")
        self.assertEqual(detector.shapes, [(64, 16, 3)] * 12)
        self.assertEqual(report["selected_frames"], 6)
        self.assertEqual(decode(path)[0].shape, (32, 64, 3))

    def test_empty_hand_selection_still_creates_no_video(self):
        source = self.root / "portrait.mov"
        make_source(source, sample())

        class Detector:
            def present(self, *a):
                return False

            def close(self):
                pass

        with patch.object(activity, "create_hand_detector", return_value=Detector()):
            status, path, report = self.run_file(source, "--mode", "hands")
        self.assertEqual(status, "no_selection")
        self.assertFalse(path.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
