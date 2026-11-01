"""Selection invariants and integration checks for center-first hand dedupe."""

import contextlib
import io
import json
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from castlestube_cleanup import activity, cli, pipeline, selection


def indices(spans):
    return [i for start, end in spans for i in range(start, end)]


class CenterTests(unittest.TestCase):
    def cuts(self, values, minimum=20, protected=(), threshold=8):
        signatures = np.array(values, dtype=np.uint8)[:, None, None]
        forced = np.zeros(len(values), dtype=bool)
        forced[list(protected)] = True
        return selection.center_removals(
            signatures, forced, minimum, threshold, threading.Event()
        )

    def test_static_long_phase_balances_beginning_and_end(self):
        self.assertEqual(self.cuts([50] * 60), [(10, 50)])
        self.assertEqual(self.cuts([50] * 21), [(10, 11)])

    def test_short_phases_are_never_shortened(self):
        for n in (1, 2, 19, 20):
            self.assertEqual(self.cuts([50] * n), [])

    def test_unique_placement_and_state_change_survive(self):
        values = [0] * 29 + [255] + [100] * 30
        removed = set(indices(self.cuts(values)))
        self.assertTrue({0, 28, 29, 30, 59}.isdisjoint(removed))
        self.assertEqual(60 - len(removed), 20)

    def test_distinct_motion_can_exceed_target(self):
        self.assertEqual(self.cuts([0, 255] * 30), [])
        self.assertEqual(self.cuts([5] * 60, threshold=0), [])

    def test_camera_and_context_protection_override_target(self):
        protected = list(range(5, 55))
        removed = set(indices(self.cuts([10] * 60, protected=protected)))
        self.assertTrue(set(protected + [0, 59]).isdisjoint(removed))
        self.assertEqual(60 - len(removed), 52)

    def test_interval_subtraction_handles_crossing_and_empty_cuts(self):
        spans = [(0, 10), (20, 30)]
        self.assertEqual(
            selection.subtract_spans(spans, [(3, 7), (8, 25), (28, 40)]),
            [(0, 3), (7, 8), (25, 28)],
        )
        self.assertEqual(selection.subtract_spans(spans, []), spans)

    def test_cli_defaults_overrides_and_invalid_combinations(self):
        args = cli.arguments(["clip.mov"])
        self.assertEqual(
            (args.pre_frames, args.post_frames, args.segment_frames, args.dedupe_order),
            (2, 2, 20, "forward"),
        )
        args = cli.arguments(
            [
                "clip.mov",
                "--mode",
                "hands-dedupe",
                "--dedupe-order",
                "center",
                "--segment-frames",
                "16",
                "--pre-frames",
                "0",
                "--post-frames",
                "3",
            ]
        )
        self.assertEqual(
            (args.pre_frames, args.post_frames, args.segment_frames), (0, 3, 16)
        )
        for flags in (
            ["--dedupe-order", "center"],
            ["--segment-frames", "1"],
            ["--pre-frames", "-1"],
        ):
            with (
                self.subTest(flags=flags),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                cli.arguments(["clip.mov", *flags])

    def test_cancellation(self):
        stop = threading.Event()
        stop.set()
        with self.assertRaises(InterruptedError):
            selection.center_removals(
                np.zeros((60, 1, 1), np.uint8), np.zeros(60, bool), 20, 8, stop
            )

    def test_two_phases_stay_independent_even_with_joined_context(self):

        class Capture:
            def __init__(self):
                self.i = -1

            def grab(self):
                self.i += 1
                return self.i < 126

            def retrieve(self):
                return (True, np.zeros((30, 50, 3), np.uint8))

            def release(self):
                pass

        args = cli.arguments(
            ["clip.mov", "--mode", "hands-dedupe", "--dedupe-order", "center"]
        )
        phases = [(2, 62), (64, 124)]
        protected = [(0, 2), (62, 64), (124, 126)]
        with patch.object(selection, "open_capture", return_value=Capture()):
            spans, report = selection.plan_center_dedupe(
                Path("clip.mov"),
                [(0, 126)],
                phases,
                protected,
                args,
                threading.Event(),
                lambda *a: None,
            )
        self.assertEqual(spans, [(0, 12), (52, 74), (114, 126)])
        self.assertEqual(report["removed_frames"], 80)
        self.assertEqual(report["shortened_phases"], 2)

    def test_real_export_uses_center_selection_once_and_preserves_fps(self):
        import subprocess

        import cv2

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source, destination = (root / "source.mov", root / "selected.mov")
            pixels = []
            for i in range(80):
                frame = np.full((32, 64, 3), 100, np.uint8)
                for bit in range(7):
                    frame[:8, bit * 8 : (bit + 1) * 8] = 255 if i & 1 << bit else 0
                pixels.append(frame.tobytes())
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
                    "64x32",
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
                    str(source),
                ],
                input=b"".join(pixels),
                check=True,
            )

            class Detector:
                def present(self, frame, i, fps):
                    return 10 <= i < 70

                def close(self):
                    pass

            args = cli.arguments(
                [
                    str(source),
                    "--mode",
                    "hands-dedupe",
                    "--dedupe-order",
                    "center",
                    "--roi",
                    "0",
                    "0.5",
                    "1",
                    "0.5",
                    "--codec",
                    "h264",
                    "--crf",
                    "0",
                    "--report",
                    "--threads",
                    "2",
                ]
            )
            with patch.object(
                activity, "create_hand_detector", return_value=Detector()
            ):
                self.assertEqual(
                    pipeline.process_file(source, destination, args, threading.Event()),
                    "written",
                )
            report = json.loads(destination.with_suffix(".json").read_text())
            self.assertEqual(
                (report["selected_frames"], report["output_fps"]), (24, "60")
            )
            self.assertEqual(report["center_dedupe"]["removed_frames"], 40)
            capture = cv2.VideoCapture(str(destination))
            actual = []
            while True:
                ok, frame = capture.read()
                if not ok:
                    break
                actual.append(
                    sum(
                        1 << bit
                        for bit in range(7)
                        if frame[4, bit * 8 + 4].mean() > 128
                    )
                )
            capture.release()
            self.assertEqual(actual, list(range(8, 20)) + list(range(60, 72)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
