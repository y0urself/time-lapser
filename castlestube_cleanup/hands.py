"""Optional MediaPipe and YOLO hand-presence backends."""

from __future__ import annotations

import math
import os
from pathlib import Path

import cv2

HAND_DEFAULTS = {
    "mediapipe": ("hand_landmarker.task", 0.5),
    "yolo": ("YOLOv10x_hands.pt", 0.1),
}


class HandDetector:
    """Run MediaPipe on sampled frames without retaining video frames."""

    def __init__(self, model_path: Path, confidence: float = 0.5, width: int = 640):
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_python
            from mediapipe.tasks.python import vision as mp_vision
        except ImportError as exc:
            raise RuntimeError(
                "Hand detection requires MediaPipe. Install it with: "
                "uv sync --extra mediapipe"
            ) from exc

        if width < 1:
            raise ValueError("The hand analysis width must be positive.")
        if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("Hand confidence must be between 0 and 1.")

        self._mp = mp
        self._width = width
        self._last_timestamp_ms = -1
        # Reading bytes in Python avoids native Windows path-encoding problems.
        model_data = Path(model_path).read_bytes()
        options = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_buffer=model_data),
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=2,
            min_hand_detection_confidence=confidence,
            min_hand_presence_confidence=confidence,
            min_tracking_confidence=0.5,
        )
        self._landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def present(self, bgr_frame, source_frame_index: int, fps: float) -> bool:
        if not math.isfinite(fps) or fps <= 0:
            raise ValueError("Hand analysis requires a positive frame rate.")
        height, width = bgr_frame.shape[:2]
        longest = max(height, width)
        if longest > self._width:
            scale = self._width / longest
            bgr_frame = cv2.resize(
                bgr_frame,
                (max(1, round(width * scale)), max(1, round(height * scale))),
                interpolation=cv2.INTER_AREA,
            )
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=rgb)
        timestamp_ms = max(
            self._last_timestamp_ms + 1,
            round(source_frame_index * 1000.0 / fps),
        )
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        self._last_timestamp_ms = timestamp_ms
        return bool(result.hand_landmarks)

    def close(self):
        landmarker = self._landmarker
        if landmarker is not None:
            self._landmarker = None
            landmarker.close()


def import_yolo():
    # Set before importing Ultralytics, which reads these flags on import.
    # Do not change the user's persistent Ultralytics settings.
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    try:
        from ultralytics import YOLO
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "YOLO hand detection requires Ultralytics/PyTorch. Install with: "
            "uv sync --extra yolo"
        ) from exc
    return YOLO


class YoloHandDetector:
    """Hand presence from a local HaGRID checkpoint; one model per video worker."""

    def __init__(
        self, model_path: Path, confidence=0.1, width=640, device="cpu", min_area=0.005
    ):
        model_path = Path(model_path).expanduser().resolve()
        if not model_path.is_file() or model_path.suffix.lower() != ".pt":
            raise ValueError(
                f"YOLO needs a local hand-specific .pt model: {model_path}"
            )
        if width < 1 or not math.isfinite(confidence) or not 0 <= confidence <= 1:
            raise ValueError(
                "Hand width must be positive; confidence must be between 0 and 1."
            )
        if not math.isfinite(min_area) or not 0 <= min_area <= 1:
            raise ValueError(
                "Minimum hand area must be a finite fraction between 0 and 1."
            )
        self._confidence = confidence
        self._width = width
        self._device = device
        self._min_area = min_area
        self.stats = {
            "candidate_frames": 0,
            "candidate_boxes": 0,
            "rejected_small_boxes": 0,
            "frames_rejected_by_size": 0,
        }
        self._model = import_yolo()(str(model_path), task="detect")
        if self._model.task != "detect":
            raise ValueError(
                "YOLO needs a hand detection checkpoint, not a pose/classification model."
            )
        # Avoid silently counting people or other COCO objects as hands.
        names = self._model.names
        names = names.values() if isinstance(names, dict) else names
        if "person" in {str(name).lower() for name in names}:
            raise ValueError(
                "This looks like a general object detector. Use a HaGRID hand model."
            )

    def present(self, bgr_frame, source_frame_index: int, fps: float) -> bool:
        # Pass original BGR pixels (or ROI) directly. Ultralytics handles resize,
        # letterboxing and channel order; pre-resizing would change our benchmark.
        results = self._model.predict(
            source=bgr_frame,
            imgsz=self._width,
            conf=self._confidence,
            device=self._device,
            verbose=False,
            save=False,
            save_txt=False,
            save_crop=False,
            show=False,
            stream=False,
            augment=False,
        )
        candidates = accepted = 0
        for result in results:
            if result.boxes is None:
                continue
            # Normalized coordinates use the original analysis image/ROI. Clip
            # to its visible area so padding cannot make a tiny edge box large.
            for x1, y1, x2, y2 in result.boxes.xyxyn.tolist():
                candidates += 1
                if not all(math.isfinite(v) for v in (x1, y1, x2, y2)):
                    continue
                area = max(0.0, min(1.0, x2) - max(0.0, x1)) * max(
                    0.0, min(1.0, y2) - max(0.0, y1)
                )
                accepted += area >= self._min_area
        self.stats["candidate_frames"] += bool(candidates)
        self.stats["candidate_boxes"] += candidates
        self.stats["rejected_small_boxes"] += candidates - accepted
        self.stats["frames_rejected_by_size"] += candidates > 0 and accepted == 0
        return accepted > 0

    def close(self):
        # Predictor caches only the current frame; release it with the model.
        self._model = None


def create_hand_detector(args):
    if args.hand_detector == "yolo":
        return YoloHandDetector(
            args.hand_model,
            args.hand_confidence,
            args.hand_width,
            args.hand_device,
            args.hand_min_area,
        )
    return HandDetector(args.hand_model, args.hand_confidence, args.hand_width)
