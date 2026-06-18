from __future__ import annotations

from pathlib import Path

import numpy as np
from .types import ViolationEvent, ViolationType


class HelmetDetector:
    """Detects helmet absence on two-wheeler riders.

    Uses YOLO classifier trained on helmet/no-helmet crops.
    Falls back to heuristic if no classifier provided.
    """

    def __init__(self, classifier_path: str | Path | None = None):
        self.classifier_path = classifier_path
        self._classifier = None

    def _load_classifier(self):
        if self._classifier is None and self.classifier_path is not None:
            from ultralytics import YOLO
            self._classifier = YOLO(str(self.classifier_path))
        return self._classifier

    def detect(self, detections, frame: np.ndarray, frame_idx: int, signal_state: str) -> list[ViolationEvent]:
        violations = []
        h, w = frame.shape[:2]

        if not hasattr(detections, "class_id") or detections.class_id is None:
            return violations

        for i in range(len(detections)):
            cls_id = int(detections.class_id[i])
            if cls_id != 3:
                continue

            bbox = detections.xyxy[i].astype(int)
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))
            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            has_helmet = self._classify_helmet(crop)
            if not has_helmet:
                conf = float(detections.confidence[i]) if detections.confidence is not None else 0.7
                tid = int(detections.tracker_id[i]) if hasattr(detections, "tracker_id") and detections.tracker_id is not None else -1
                violations.append(
                    ViolationEvent(
                        type=ViolationType.NO_HELMET,
                        track_id=tid,
                        frame=frame_idx,
                        confidence=conf,
                        bbox=tuple(bbox),
                    )
                )
        return violations

    def _classify_helmet(self, crop: np.ndarray) -> bool:
        model = self._load_classifier()
        if model is None:
            return self._heuristic_helmet(crop)

        result = model(crop, verbose=False)[0]
        label = result.probs.top1
        conf = result.probs.top1conf
        return label == 1 and conf > 0.6

    def _heuristic_helmet(self, crop: np.ndarray) -> bool:
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV) if len(crop.shape) == 3 else crop
        if len(crop.shape) == 2:
            return True

        h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
        bright_mask = (v > 200) & (s < 50)
        bright_ratio = np.sum(bright_mask) / max(bright_mask.size, 1)

        skin_mask = ((h < 25) | (h > 165)) & (s > 40) & (v > 80)
        skin_ratio = np.sum(skin_mask) / max(skin_mask.size, 1)

        if bright_ratio > 0.3:
            return True
        if skin_ratio > 0.4:
            return False
        return True


import cv2
