"""Small public real-video fixtures supplement the generated-video tests."""

import contextlib
import hashlib
import io
import json
import tempfile
import threading
import unittest
from fractions import Fraction
from pathlib import Path
from unittest.mock import patch

import numpy as np

from castlestube_cleanup import activity, cli, pipeline, video
from tests.test_rotation import decode

DATA = Path(__file__).with_name("data")


class RealClipTests(unittest.TestCase):
    def test_fixture_hashes_decoding_and_annotation_bounds(self):
        manifest = json.loads((DATA / "manifest.json").read_text())
        for clip in manifest["clips"]:
            with self.subTest(clip=clip["file"]):
                source = DATA / clip["file"]
                self.assertEqual(source.stat().st_size, clip["bytes"])
                self.assertEqual(
                    hashlib.sha256(source.read_bytes()).hexdigest(), clip["sha256"]
                )
                metadata = video.probe_video(source, "ffprobe")
                self.assertEqual(
                    Fraction(metadata["avg_frame_rate"]), Fraction(clip["fps"])
                )
                cap = video.open_capture(source, 2)
                count = 0
                try:
                    while True:
                        ok, frame = cap.read()
                        if not ok:
                            break
                        self.assertEqual(
                            frame.shape[:2],
                            (clip["decoded_height"], clip["decoded_width"]),
                        )
                        count += 1
                finally:
                    cap.release()
                self.assertEqual(count, clip["frames"])
                for start, end in clip["hand_present_spans"]:
                    self.assertTrue(0 <= start < end <= count)

    def test_real_portrait_rotation_keeps_all_source_frames_and_fps(self):
        source = DATA / "empty-lego-detail.mp4"
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "rotated.mov"
            args = cli.arguments(
                [
                    str(source),
                    "--threshold",
                    "0",
                    "--codec",
                    "h264",
                    "--threads",
                    "2",
                    "--report",
                ]
            )
            with contextlib.redirect_stdout(io.StringIO()):
                result = pipeline.process_file(
                    source, destination, args, threading.Event()
                )
            self.assertEqual(result, "written")
            metadata = video.probe_video(destination, "ffprobe")
            self.assertEqual((metadata["width"], metadata["height"]), (1920, 1080))
            self.assertEqual(int(metadata["nb_frames"]), 30)
            self.assertEqual(Fraction(metadata["avg_frame_rate"]), 30)
            report = json.loads(destination.with_suffix(".json").read_text())
            self.assertEqual(report["selected_frames"], 30)

    def test_real_building_selection_preserves_context_order_and_fps(self):
        source = DATA / "lego-building.mp4"

        class Detector:
            def present(self, frame, index, fps):
                return index in (0, 1, 2, 3, 4, 8, 9, 10, 11, 18, 19, 26, 27, 28, 29)

            def close(self):
                pass

        # Two context frames on each side merge the first two hand phases.
        expected_indices = [*range(14), *range(16, 22), *range(24, 30)]
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "selected.mov"
            args = cli.arguments(
                [
                    str(source),
                    "--mode",
                    "hands",
                    "--hand-step",
                    "1",
                    "--pre-frames",
                    "2",
                    "--post-frames",
                    "2",
                    "--codec",
                    "h264",
                    "--crf",
                    "0",
                    "--threads",
                    "2",
                    "--report",
                ]
            )
            with (
                patch.object(activity, "create_hand_detector", return_value=Detector()),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                result = pipeline.process_file(
                    source, destination, args, threading.Event()
                )
            self.assertEqual(result, "written")
            metadata = video.probe_video(destination, "ffprobe")
            self.assertEqual(Fraction(metadata["avg_frame_rate"]), 30)
            self.assertEqual((metadata["width"], metadata["height"]), (1920, 1080))
            actual = decode(destination)
            original = decode(source)
            self.assertEqual(len(actual), len(expected_indices))
            for frame, index in zip(actual, expected_indices):
                with self.subTest(source_frame=index):
                    self.assertLess(
                        np.abs(frame.astype(float) - original[index]).mean(), 3
                    )
            report = json.loads(destination.with_suffix(".json").read_text())
            self.assertEqual(report["input_frames"], 30)
            self.assertEqual(report["selected_frames"], 26)
