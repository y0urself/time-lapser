"""Global camera-motion estimation on consecutive source frames."""

from __future__ import annotations

import math
from collections import Counter

import cv2
import numpy as np


class CameraMotionDetector:
    """Detect broadly supported camera motion between consecutive small images.

    Unknown pairs are not classified as motion. A model-positive pair includes
    both source endpoints; the caller handles interval padding and preservation.
    """

    def __init__(self, width=480, threshold=0.002):
        if width < 96 or not math.isfinite(threshold) or not 0 < threshold < 1:
            raise ValueError(
                "Camera width must be at least 96 and threshold strictly between 0 and 1."
            )
        self.width = width
        self.threshold = threshold
        self.previous = None
        self.stats = {
            "comparisons": 0,
            "detected_pairs": 0,
            "stationary_pairs": 0,
            "unclassified_pairs": 0,
            "flow_pairs": 0,
            "orb_pairs": 0,
            "orb_fallbacks": 0,
            "max_displacement_fraction": 0.0,
        }
        self.orb = cv2.ORB_create(nfeatures=1200, edgeThreshold=15, fastThreshold=12)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING)

    @staticmethod
    def _cells(points, shape):
        height, width = shape
        x = np.clip((points[:, 0] * 4 / width).astype(int), 0, 3)
        y = np.clip((points[:, 1] * 4 / height).astype(int), 0, 3)
        return x, y

    @classmethod
    def _spread(cls, points, shape):
        height, width = shape
        x, y = cls._cells(points, shape)
        return (
            len(np.unique(y * 4 + x)) >= 8
            and len(np.unique(x)) >= 3
            and len(np.unique(y)) >= 3
            and np.ptp(points[:, 0]) >= width * 0.5
            and np.ptp(points[:, 1]) >= height * 0.5
            and cv2.contourArea(cv2.convexHull(points.astype(np.float32)))
            >= width * height * 0.20
        )

    @staticmethod
    def _corners(gray):
        # Balance the image so a textured moving hand cannot supply all points.
        height, width = gray.shape
        points = []
        for row in range(4):
            top, bottom = row * height // 4, (row + 1) * height // 4
            for column in range(4):
                left, right = column * width // 4, (column + 1) * width // 4
                tile = gray[top:bottom, left:right]
                if min(tile.shape) < 7:
                    continue
                corners = cv2.goodFeaturesToTrack(
                    tile,
                    maxCorners=16,
                    qualityLevel=0.015,
                    minDistance=5,
                    blockSize=7,
                )
                if corners is not None:
                    points.append(corners.reshape(-1, 2) + (left, top))
        return np.concatenate(points).astype(np.float32) if points else None

    def _fit(self, source, target, shape):
        if len(source) < 24:
            return None
        height, width = shape
        diagonal = math.hypot(width, height)
        matrix, mask = cv2.estimateAffinePartial2D(
            source,
            target,
            method=cv2.RANSAC,
            ransacReprojThreshold=max(1.0, diagonal * 0.003),
            maxIters=1500,
            confidence=0.99,
            refineIters=10,
        )
        if matrix is None or mask is None or not np.isfinite(matrix).all():
            return None
        inliers = mask.reshape(-1).astype(bool)
        if inliers.sum() < 18 or inliers.mean() < 0.65:
            return None
        if not (
            self._spread(source[inliers], shape)
            and self._spread(target[inliers], shape)
        ):
            return None
        scale = math.hypot(matrix[0, 0], matrix[1, 0])
        if not 0.5 <= scale <= 2.0:
            return None
        # Measure motion throughout the image, including rotation and zoom.
        xx, yy = np.meshgrid(
            np.linspace(0.1, 0.9, 5) * width, np.linspace(0.1, 0.9, 5) * height
        )
        grid = np.column_stack((xx.ravel(), yy.ravel()))
        transformed = grid @ matrix[:, :2].T + matrix[:, 2]
        displacement = (
            float(np.median(np.linalg.norm(transformed - grid, axis=1))) / diagonal
        )
        return (
            displacement
            if math.isfinite(displacement) and displacement <= 0.5
            else None
        )

    def _flow(self, previous, current, points):
        options = dict(
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        target, forward_ok, errors = cv2.calcOpticalFlowPyrLK(
            previous, current, points.reshape(-1, 1, 2), None, **options
        )
        if target is None or forward_ok is None:
            return None
        target = target.reshape(-1, 2)
        height, width = current.shape
        valid = (
            forward_ok.reshape(-1).astype(bool)
            & np.isfinite(target).all(axis=1)
            & (target[:, 0] >= 0)
            & (target[:, 0] < width)
            & (target[:, 1] >= 0)
            & (target[:, 1] < height)
        )
        if errors is not None:
            valid &= np.isfinite(errors.reshape(-1)) & (errors.reshape(-1) <= 35)
        source = points[valid]
        target = target[valid]
        if len(source) < 24:
            return None
        backward, backward_ok, _ = cv2.calcOpticalFlowPyrLK(
            current, previous, target.reshape(-1, 1, 2), None, **options
        )
        if backward is None or backward_ok is None:
            return None
        backward = backward.reshape(-1, 2)
        valid = (
            backward_ok.reshape(-1).astype(bool)
            & np.isfinite(backward).all(axis=1)
            & (np.linalg.norm(backward - source, axis=1) <= 1.0)
        )
        if np.count_nonzero(valid) < max(24, math.ceil(len(points) * 0.35)):
            return None
        return self._fit(source[valid], target[valid], current.shape)

    def _orb(self, previous, current):
        source_keys, source_desc = self.orb.detectAndCompute(previous, None)
        target_keys, target_desc = self.orb.detectAndCompute(current, None)
        if (
            source_desc is None
            or target_desc is None
            or min(len(source_desc), len(target_desc)) < 24
        ):
            return None
        forward = self.matcher.knnMatch(source_desc, target_desc, k=2)
        backward = self.matcher.knnMatch(target_desc, source_desc, k=2)
        reverse = {
            pair[0].queryIdx: pair[0].trainIdx
            for pair in backward
            if len(pair) == 2 and pair[0].distance < 0.75 * pair[1].distance
        }
        matches = [
            pair[0]
            for pair in forward
            if len(pair) == 2
            and pair[0].distance < 0.75 * pair[1].distance
            and reverse.get(pair[0].trainIdx) == pair[0].queryIdx
        ]
        # Strong matches from one object must not overwhelm the rest of the frame.
        counts = Counter()
        selected = []
        height, width = previous.shape
        for match in sorted(matches, key=lambda item: item.distance):
            x, y = source_keys[match.queryIdx].pt
            cell = (min(3, int(x * 4 / width)), min(3, int(y * 4 / height)))
            if counts[cell] < 16:
                selected.append(match)
                counts[cell] += 1
        if len(selected) < 24:
            return None
        source = np.float32([source_keys[match.queryIdx].pt for match in selected])
        target = np.float32([target_keys[match.trainIdx].pt for match in selected])
        return self._fit(source, target, current.shape)

    def moving(self, bgr_frame):
        height, width = bgr_frame.shape[:2]
        if max(height, width) > self.width:
            ratio = self.width / max(height, width)
            bgr_frame = cv2.resize(
                bgr_frame,
                (max(1, round(width * ratio)), max(1, round(height * ratio))),
                interpolation=cv2.INTER_AREA,
            )
        current = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2GRAY)
        previous, self.previous = self.previous, current
        if previous is None:
            return False
        self.stats["comparisons"] += 1
        result = None
        if previous.shape == current.shape and min(current.shape) >= 32:
            points = self._corners(previous)
            if (
                points is not None
                and len(points) >= 24
                and self._spread(points, previous.shape)
            ):
                result = self._flow(previous, current, points)
                if result is None:
                    self.stats["orb_fallbacks"] += 1
                    result = self._orb(previous, current)
                    if result is not None:
                        self.stats["orb_pairs"] += 1
                else:
                    self.stats["flow_pairs"] += 1
        if result is None:
            self.stats["unclassified_pairs"] += 1
            return False
        self.stats["max_displacement_fraction"] = max(
            self.stats["max_displacement_fraction"], result
        )
        moving = result >= self.threshold
        self.stats["detected_pairs" if moving else "stationary_pairs"] += 1
        return moving
