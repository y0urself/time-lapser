"""Protect separated actions and existing exports when processing fails."""

import contextlib
import io
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

from castlestube_cleanup import activity, cli, pipeline
from tests.test_rotation import decode, make_source, sample


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "source.mov"
        self.output = self.root / "result.mov"
        make_source(self.source, sample(32, 64), count=12)
        self.original = self.source.read_bytes()

    def run_file(self, *flags):
        args = cli.arguments(
            [str(self.source), "--codec", "h264", "--threads", "2", *flags]
        )
        with contextlib.redirect_stdout(io.StringIO()):
            return pipeline.process_file(
                self.source, self.output, args, threading.Event()
            )

    def test_identical_separated_hand_actions_each_keep_a_frame(self):
        class Detector:
            def present(self, frame, index, fps):
                return index in (1, 2, 8, 9)

            def close(self):
                pass

        with patch.object(activity, "create_hand_detector", return_value=Detector()):
            self.assertEqual(
                self.run_file(
                    "--mode",
                    "hands-dedupe",
                    "--max-frames",
                    "1",
                    "--pre-frames",
                    "0",
                    "--post-frames",
                    "0",
                ),
                "written",
            )
        self.assertEqual(len(decode(self.output)), 2)
        self.assertEqual(self.source.read_bytes(), self.original)

    def test_encoder_failure_preserves_existing_destination_and_cleans_temporary(self):
        self.output.write_bytes(b"previous successful export")

        class FailingEncoder:
            closed = False

            def __init__(self, path, *args):
                path.write_bytes(b"partial output")

            def write(self, frame):
                pass

            def finish(self):
                raise RuntimeError("synthetic encode failure")

            def close(self):
                FailingEncoder.closed = True

        with patch.object(pipeline, "RawEncoder", FailingEncoder):
            with self.assertRaisesRegex(RuntimeError, "synthetic encode failure"):
                self.run_file("--overwrite", "--max-frames", "1")
        self.assertEqual(self.output.read_bytes(), b"previous successful export")
        self.assertEqual(self.source.read_bytes(), self.original)
        self.assertTrue(FailingEncoder.closed)
        self.assertEqual(list(self.root.glob(".castlestube-*")), [])

    def test_existing_output_is_skipped_before_decoding(self):
        self.output.write_bytes(b"existing")
        with patch.object(
            pipeline, "probe_video", side_effect=AssertionError("should skip")
        ):
            self.assertEqual(self.run_file(), "skipped")
        self.assertEqual(self.output.read_bytes(), b"existing")
