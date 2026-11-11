"""The optional model evaluator must compare predictions with independent labels."""

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tests import evaluate_models


class EvaluationTests(unittest.TestCase):
    def test_confusion_counts_and_model_hash(self):
        class Detector:
            def __init__(self, positive_clip):
                self.positive_clip = positive_clip

            def present(self, frame, index, fps):
                if self.positive_clip:
                    # Miss positive frame 0 and falsely flag empty frame 5.
                    return index in (
                        1,
                        2,
                        3,
                        4,
                        5,
                        8,
                        9,
                        10,
                        11,
                        18,
                        19,
                        26,
                        27,
                        28,
                        29,
                    )
                return index == 6

            def close(self):
                pass

        with tempfile.TemporaryDirectory() as tmp:
            model = Path(tmp) / "fake.task"
            model.write_bytes(b"fake weights")
            output = io.StringIO()
            with (
                patch.object(
                    evaluate_models.hands,
                    "create_hand_detector",
                    side_effect=[Detector(True), Detector(False)],
                ),
                contextlib.redirect_stdout(output),
            ):
                self.assertEqual(evaluate_models.main(["--hand-model", str(model)]), 0)
            report = json.loads(output.getvalue())
            for key, expected in {"tp": 14, "fp": 2, "fn": 1, "tn": 43}.items():
                self.assertEqual(report["total"][key], expected)
            self.assertAlmostEqual(report["total"]["recall"], 14 / 15)
            self.assertAlmostEqual(report["total"]["precision"], 14 / 16)
            self.assertEqual(report["model_sha256"], evaluate_models.digest(model))
