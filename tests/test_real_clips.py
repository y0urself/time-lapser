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

from castlestube_cleanup import cli, pipeline, video

DATA = Path(__file__).with_name("data")


class RealClipTests(unittest.TestCase):
    def test_fixture_hashes_decoding_and_annotation_bounds(self):
        manifest = json.loads((DATA / "manifest.json").read_text())
        for clip in manifest["clips"]:
            with self.subTest(clip=clip["file"]):
                source = DATA / clip["file"]
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
        source = DATA / "overhead-hands.mp4"
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
