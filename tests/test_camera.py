"""Global movement must be distinguished from static or local scene changes."""

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from castlestube_cleanup.camera import CameraMotionDetector


class CameraTests(unittest.TestCase):
    def setUp(self):
        cv2.setRNGSeed(0)
        rng = np.random.default_rng(42)
        gray = rng.integers(25, 180, (240, 320), dtype=np.uint8)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        self.frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    def shifted(self):
        return cv2.warpAffine(
            self.frame,
            np.float32([[1, 0, 8], [0, 1, 4]]),
            (320, 240),
            borderMode=cv2.BORDER_REFLECT,
        )

    def test_global_translation_is_preserved(self):
        detector = CameraMotionDetector()
        self.assertFalse(detector.moving(self.frame))
        self.assertTrue(detector.moving(self.shifted()))

    def test_stationary_and_exposure_changes_are_not_camera_motion(self):
        for changed in (
            self.frame.copy(),
            cv2.add(self.frame, np.full_like(self.frame, 20)),
        ):
            with self.subTest(exposure=not np.array_equal(self.frame, changed)):
                detector = CameraMotionDetector()
                detector.moving(self.frame)
                self.assertFalse(detector.moving(changed))

    def test_local_object_motion_is_not_global_motion(self):
        changed = self.frame.copy()
        changed[80:140, 100:180] = self.shifted()[80:140, 100:180]
        detector = CameraMotionDetector()
        detector.moving(self.frame)
        self.assertFalse(detector.moving(changed))

    def test_orb_fallback_detects_translation_when_flow_is_unavailable(self):
        detector = CameraMotionDetector()
        detector.moving(self.frame)
        with patch.object(detector, "_flow", return_value=None):
            self.assertTrue(detector.moving(self.shifted()))
        self.assertEqual(detector.stats["orb_pairs"], 1)

    def test_textureless_frame_pair_is_unclassified(self):
        detector = CameraMotionDetector()
        frame = np.zeros_like(self.frame)
        detector.moving(frame)
        self.assertFalse(detector.moving(frame))
        self.assertEqual(detector.stats["unclassified_pairs"], 1)
