import contextlib
import io
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from castlestube_cleanup import cli, hands


class HandAreaTests(unittest.TestCase):
    def detector(self, box_frames, cut=0.005):
        boxes = iter(box_frames)

        class FakeModel:
            task = "detect"
            names = {0: "no_gesture"}

            def predict(self, **kw):
                return [
                    SimpleNamespace(boxes=SimpleNamespace(xyxyn=np.array(next(boxes))))
                ]

        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        p = Path(temporary.name) / "hand.pt"
        p.touch()
        with patch.object(
            hands, "import_yolo", return_value=lambda *a, **k: FakeModel()
        ):
            return hands.YoloHandDetector(p, min_area=cut)

    def test_small_only_rejected_mixed_real_hand_preserved(self):
        small = [0.1, 0.1, 0.2, 0.12]
        real = [0.1, 0.1, 0.6, 0.6]
        d = self.detector([[small], [small, real], []])
        self.assertEqual(
            [d.present(None, i, 30) for i in range(3)], [False, True, False]
        )
        self.assertEqual(
            d.stats,
            {
                "candidate_frames": 2,
                "candidate_boxes": 3,
                "rejected_small_boxes": 2,
                "frames_rejected_by_size": 1,
            },
        )
        d.close()
        self.assertEqual(d.stats["frames_rejected_by_size"], 1)

    def test_visible_area_at_border_and_inclusive_boundary(self):
        d = self.detector([[[-1, 0, 0.001, 1]], [[0, 0, 0.1, 0.05]]])
        self.assertFalse(d.present(None, 0, 30))
        self.assertTrue(d.present(None, 1, 30))

    def test_zero_restores_tiny_detections(self):
        d = self.detector([[[0.1, 0.1, 0.11, 0.11]]], 0)
        self.assertTrue(d.present(None, 0, 30))

    def test_invalid_area_and_defaults(self):
        self.assertEqual(
            cli.arguments(["clip.mov", "--hand-detector", "yolo"]).hand_min_area, 0.005
        )
        for value in ["nan", "inf", "-0.1", "1.01"]:
            with (
                self.subTest(value=value),
                contextlib.redirect_stderr(io.StringIO()),
                self.assertRaises(SystemExit),
            ):
                cli.arguments(["clip.mov", "--hand-min-area", value])


if __name__ == "__main__":
    unittest.main(verbosity=2)
