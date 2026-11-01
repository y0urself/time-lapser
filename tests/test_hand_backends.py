"""Regression checks for backend switching, optional dependencies and frame safety."""

import contextlib
import io
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from castlestube_cleanup import activity, cli, hands


class BackendTests(unittest.TestCase):
    def test_legacy_defaults_and_override(self):
        a = cli.arguments(["clip.mov", "--mode", "hands"])
        self.assertEqual(
            (a.hand_detector, a.hand_confidence, a.hand_model.name),
            ("mediapipe", 0.5, "hand_landmarker.task"),
        )
        self.assertEqual(
            cli.arguments(["clip.mov", "--hand-confidence", ".7"]).hand_confidence, 0.7
        )

    def test_yolo_defaults_and_explicit_zero(self):
        a = cli.arguments(["clip.mov", "--mode", "hands", "--hand-detector", "yolo"])
        self.assertEqual(
            (a.hand_confidence, a.hand_model.name, a.hand_device),
            (0.1, "YOLOv10x_hands.pt", "cpu"),
        )
        a = cli.arguments(
            [
                "clip.mov",
                "--mode",
                "hands",
                "--hand-detector",
                "yolo",
                "--hand-confidence",
                "0",
                "--hand-model",
                "custom.pt",
                "--hand-device",
                "0",
            ]
        )
        self.assertEqual(
            (a.hand_confidence, a.hand_model.name, a.hand_device),
            (0.0, "custom.pt", "0"),
        )

    def test_invalid_settings_fail_before_model_loading(self):
        for flags in [
            ["--hand-confidence", "nan"],
            ["--hand-confidence", "inf"],
            ["--hand-confidence", "-0.1"],
            ["--hand-detector", "yolo", "--hand-model", "wrong.task"],
            ["--hand-model", "wrong.pt"],
            ["--hand-detector", "yolo", "--hand-device", ""],
        ]:
            with (
                self.subTest(flags=flags),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                cli.arguments(["clip.mov", "--mode", "hands", *flags])

    def test_missing_yolo_weights_never_trigger_download_or_import(self):
        with patch.object(
            hands, "import_yolo", side_effect=AssertionError("must not import")
        ):
            with self.assertRaisesRegex(ValueError, "local hand-specific"):
                hands.YoloHandDetector(Path("/does-not-exist/hand.pt"))

    def test_backend_switch_preserves_camera_and_context_union(self):

        class Capture:
            def __init__(self):
                self.index = -1
                self.released = False

            def get(self, k):
                return 14

            def grab(self):
                self.index += 1
                return self.index < 14

            def retrieve(self):
                return (True, np.zeros((30, 50, 3), dtype=np.uint8))

            def release(self):
                self.released = True

        class Detector:
            def __init__(self, *args):
                self.closed = False
                self.shapes = []
                self.stats = {}

            def present(self, frame, i, fps):
                self.shapes.append(frame.shape)
                return i in [2, 3, 8]

            def close(self):
                self.closed = True

        class Camera:
            def __init__(self, *args):
                self.i = -1
                self.stats = {}

            def moving(self, frame):
                self.i += 1
                return self.i == 12

        for backend, cls in [
            ("mediapipe", "HandDetector"),
            ("yolo", "YoloHandDetector"),
        ]:
            with self.subTest(backend=backend):
                args = cli.arguments(
                    [
                        "clip.mov",
                        "--mode",
                        "hands",
                        "--hand-detector",
                        backend,
                        "--keep-camera-motion",
                        "--roi",
                        "0",
                        "0",
                        "0.5",
                        "1",
                    ]
                )
                cap = Capture()
                detector = Detector()
                with (
                    patch.object(activity, "open_capture", return_value=cap),
                    patch.object(hands, cls, return_value=detector) as factory,
                    patch.object(activity, "CameraMotionDetector", Camera),
                ):
                    spans, protected, hand_stats, camera = activity.analyze_activity(
                        Path("clip.mov"),
                        args,
                        30.0,
                        None,
                        threading.Event(),
                        lambda *a: None,
                    )
                self.assertEqual(spans, [(0, 14)])
                self.assertEqual(protected, [(0, 2), (4, 8), (9, 14)])
                self.assertEqual(hand_stats["phase_spans"], [(2, 4), (8, 9)])
                self.assertEqual(hand_stats["positives"], 3)
                self.assertEqual(hand_stats["detector"], backend)
                self.assertEqual(detector.shapes, [(30, 25, 3)] * 14)
                self.assertTrue(detector.closed and cap.released)
                factory.assert_called_once()

    def test_dedupe_does_not_load_either_hand_dependency(self):
        args = cli.arguments(
            [
                "clip.mov",
                "--mode",
                "dedupe",
                "--hand-detector",
                "yolo",
                "--keep-camera-motion",
            ]
        )

        class Capture:
            def __init__(self):
                self.i = 0

            def get(self, k):
                return 1

            def grab(self):
                self.i += 1
                return self.i == 1

            def retrieve(self):
                return (True, np.zeros((100, 100, 3), dtype=np.uint8))

            def release(self):
                pass

        with (
            patch.object(activity, "open_capture", return_value=Capture()),
            patch.object(
                activity,
                "create_hand_detector",
                side_effect=AssertionError("unselected hand dependency"),
            ),
        ):
            _, _, hands, _ = activity.analyze_activity(
                Path("clip.mov"), args, 30.0, None, threading.Event(), lambda *a: None
            )
        self.assertIsNone(hands)


if __name__ == "__main__":
    unittest.main(verbosity=2)
