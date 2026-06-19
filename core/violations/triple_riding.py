from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2

from .types import ViolationEvent, ViolationType

TRIPLE_CLASSES = {0: "normal_riders", 1: "triple_riding"}


class TripleRidingDetector:
    """Detects 3+ riders on a two-wheeler using ML classification.

    Uses a YOLO classifier trained on two-wheeler crops with multiple riders.
    Falls back to heuristic (HSV skin blob detection) if no model is available.
    """

    def __init__(self, classifier_path: str | Path | None = "models/fluxo_triple_riding_v1.pt"):
        self.classifier_path = classifier_path
        self._classifier = None

    def _load_classifier(self):
        if self._classifier is None and self.classifier_path is not None:
            if Path(self.classifier_path).exists():
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
            if cls_id not in (0,):
                continue

            det_conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
            if det_conf < 0.7:
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

            is_triple = self._classify_triple(crop)

            if is_triple:
                tid = int(detections.tracker_id[i]) if hasattr(detections, "tracker_id") and detections.tracker_id is not None else -1
                violations.append(
                    ViolationEvent(
                        type=ViolationType.TRIPLE_RIDING,
                        track_id=tid,
                        frame=frame_idx,
                        confidence=det_conf,
                        bbox=tuple(bbox),
                    )
                )
        return violations

    def _classify_triple(self, crop: np.ndarray) -> bool:
        model = self._load_classifier()
        if model is None:
            return self._heuristic_triple(crop)

        result = model(crop, verbose=False)[0]
        label = result.probs.top1
        conf = result.probs.top1conf

        return label == 1 and conf > 0.6

    def _heuristic_triple(self, crop: np.ndarray) -> bool:
        if len(crop.shape) < 3:
            return False

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        skin_mask = ((hsv[:, :, 0] < 25) | (hsv[:, :, 0] > 165)) & (hsv[:, :, 1] > 40) & (hsv[:, :, 2] > 80)
        skin_u8 = skin_mask.astype(np.uint8) * 255

        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(skin_u8, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        head_count = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 80:
                continue
            rect = cv2.boundingRect(cnt)
            if rect[2] > 0 and rect[3] > 0:
                ratio = rect[3] / rect[2]
                if 0.5 < ratio < 2.5 and area > 120:
                    head_count += 1

        return head_count >= 3
