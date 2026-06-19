from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2

from .types import ViolationEvent, ViolationType

MOBILE_CLASSES = {0: "no_phone", 1: "using_phone"}


class MobilePhoneDetector:
    """Detects mobile phone usage by drivers using ML classification.

    Uses a YOLO classifier trained on driver upper-body crops.
    Falls back to heuristic (skin + edge density) if no model is available.
    """

    def __init__(self, classifier_path: str | Path | None = "models/fluxo_mobile_phone_v1.pt"):
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

            upper_body = self._extract_upper_body(crop)
            uses_phone = self._classify_phone(upper_body if upper_body is not None else crop)

            if uses_phone:
                tid = int(detections.tracker_id[i]) if hasattr(detections, "tracker_id") and detections.tracker_id is not None else -1
                violations.append(
                    ViolationEvent(
                        type=ViolationType.MOBILE_PHONE,
                        track_id=tid,
                        frame=frame_idx,
                        confidence=det_conf,
                        bbox=tuple(bbox),
                    )
                )
        return violations

    def _extract_upper_body(self, vehicle_crop: np.ndarray) -> np.ndarray | None:
        h, w = vehicle_crop.shape[:2]
        if h < 10 or w < 10:
            return None
        upper_h = max(int(h * 0.5), 1)
        cx = w // 2
        half_w = max(w // 2, 1)
        x1 = max(0, cx - half_w)
        x2 = min(w, cx + half_w)
        return vehicle_crop[0:upper_h, x1:x2]

    def _classify_phone(self, crop: np.ndarray) -> bool:
        model = self._load_classifier()
        if model is None:
            return self._heuristic_phone(crop)

        result = model(crop, verbose=False)[0]
        label = result.probs.top1
        conf = result.probs.top1conf

        return label == 1 and conf > 0.6

    def _heuristic_phone(self, crop: np.ndarray) -> bool:
        if len(crop.shape) < 3:
            return False
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        skin_mask = ((h_ch < 25) | (h_ch > 165)) & (s_ch > 40) & (v_ch > 80)
        skin_ratio = np.sum(skin_mask) / max(skin_mask.size, 1)

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges > 0)

        has_hand_near_face = skin_ratio > 0.25 and edge_density > 0.08
        return has_hand_near_face
