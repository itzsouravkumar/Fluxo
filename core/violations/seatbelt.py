from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2

from .types import ViolationEvent, ViolationType

SEATBELT_CLASSES = {0: "with_seatbelt", 1: "without_seatbelt"}


class SeatbeltDetector:
    """Detects seatbelt absence using ML classification.

    Uses a YOLO classifier trained on driver crops.
    Falls back to heuristic (diagonal edge scan) if no model is available.
    """

    def __init__(self, classifier_path: str | Path | None = "models/fluxo_seatbelt_v1.pt"):
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
            if cls_id != 0:
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

            driver_region = self._extract_driver_region(crop)
            has_seatbelt = self._classify_seatbelt(driver_region if driver_region is not None else crop)

            if not has_seatbelt:
                tid = int(detections.tracker_id[i]) if hasattr(detections, "tracker_id") and detections.tracker_id is not None else -1
                violations.append(
                    ViolationEvent(
                        type=ViolationType.NO_SEATBELT,
                        track_id=tid,
                        frame=frame_idx,
                        confidence=det_conf,
                        bbox=tuple(bbox),
                    )
                )
        return violations

    def _extract_driver_region(self, vehicle_crop: np.ndarray) -> np.ndarray | None:
        h, w = vehicle_crop.shape[:2]
        if h < 10 or w < 10:
            return None
        driver_h = max(int(h * 0.6), 1)
        cx = w // 2
        half_w = max(w // 3, 1)
        x1 = max(0, cx - half_w)
        x2 = min(w, cx + half_w)
        return vehicle_crop[0:driver_h, x1:x2]

    def _classify_seatbelt(self, crop: np.ndarray) -> bool:
        model = self._load_classifier()
        if model is None:
            return self._heuristic_seatbelt(crop)

        result = model(crop, verbose=False)[0]
        label = result.probs.top1
        conf = result.probs.top1conf

        return label == 0 and conf > 0.6

    def _heuristic_seatbelt(self, crop: np.ndarray) -> bool:
        if len(crop.shape) < 3:
            return True
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        edge = cv2.Canny(gray, 50, 150)

        best_count = 0
        for angle in range(-30, 31, 10):
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            rotated = cv2.warpAffine(edge, M, (w, h))
            for row in range(int(h * 0.3), int(h * 0.8), max(h // 20, 1)):
                line_count = np.sum(rotated[row, :] > 0)
                best_count = max(best_count, line_count)

        edge_ratio = best_count / max(w, 1)
        return edge_ratio < 0.3
