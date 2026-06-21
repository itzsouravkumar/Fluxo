from __future__ import annotations

from pathlib import Path
from collections import defaultdict

import numpy as np
import cv2

from .types import ViolationEvent, ViolationType

MOBILE_CLASSES = {0: "no_phone", 1: "using_phone"}

# Phone usage is relevant for both two-wheelers and car drivers
PHONE_VEHICLE_CLASSES = {0, 2}  # two_wheeler + light_motor_vehicle


class MobilePhoneDetector:
    """Detects mobile phone usage by drivers using ML classification.

    Uses a YOLO classifier trained on driver upper-body crops.
    Falls back to improved heuristic with temporal consistency if no model.

    Checks both two-wheelers (class 0) and cars (class 2).
    """

    # Minimum consecutive frames a phone-like signal must persist
    MIN_TEMPORAL_CONSISTENCY = 3

    def __init__(self, classifier_path: str | Path | None = "models/fluxo_mobile_phone_v1.pt"):
        self.classifier_path = classifier_path
        self._classifier = None
        self._phone_streak: dict[int, int] = defaultdict(int)

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

        active_tids = set()

        for i in range(len(detections)):
            cls_id = int(detections.class_id[i])
            # Check both two-wheelers and cars
            if cls_id not in PHONE_VEHICLE_CLASSES:
                continue

            det_conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
            if det_conf < 0.7:
                continue

            tid = int(detections.tracker_id[i]) if hasattr(detections, "tracker_id") and detections.tracker_id is not None else -1
            if tid >= 0:
                active_tids.add(tid)

            bbox = detections.xyxy[i].astype(int)
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))

            bw, bh = x2 - x1, y2 - y1
            if bw < 40 or bh < 40:
                continue

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            upper_body = self._extract_upper_body(crop)
            uses_phone = self._classify_phone(upper_body if upper_body is not None else crop)

            if uses_phone:
                if tid >= 0:
                    self._phone_streak[tid] += 1
                else:
                    self._phone_streak[-1] = self.MIN_TEMPORAL_CONSISTENCY

                # Only fire after temporal consistency (reduces FP on momentary gestures)
                if self._phone_streak.get(tid, 0) >= self.MIN_TEMPORAL_CONSISTENCY:
                    violations.append(
                        ViolationEvent(
                            type=ViolationType.MOBILE_PHONE,
                            track_id=tid,
                            frame=frame_idx,
                            confidence=det_conf * 0.85,
                            bbox=tuple(bbox),
                        )
                    )
            else:
                if tid >= 0:
                    # Decay streak on non-detection, don't reset instantly
                    self._phone_streak[tid] = max(0, self._phone_streak.get(tid, 0) - 1)

        # Cleanup stale tracks
        stale = [tid for tid in self._phone_streak if tid not in active_tids and tid >= 0]
        for tid in stale:
            del self._phone_streak[tid]

        return violations

    def _extract_upper_body(self, vehicle_crop: np.ndarray) -> np.ndarray | None:
        h, w = vehicle_crop.shape[:2]
        if h < 20 or w < 20:
            return None
        upper_h = max(int(h * 0.5), 1)
        return vehicle_crop[0:upper_h, :]

    def _classify_phone(self, crop: np.ndarray) -> bool:
        model = self._load_classifier()
        if model is None:
            return self._heuristic_phone(crop)

        result = model(crop, verbose=False)[0]
        label = result.probs.top1
        conf = result.probs.top1conf

        return label == 1 and conf > 0.6

    def _heuristic_phone(self, crop: np.ndarray) -> bool:
        """Improved heuristic with multiple signals to reduce false positives.

        Looks for the specific pattern of a phone held near the ear/face:
        1. Skin-colored region near the head area (hand+face)
        2. A small rectangular dark object within that region (phone body)
        3. The dark object must be near the edge of the face region

        This is much more specific than just skin_ratio + edge_density.
        """
        if len(crop.shape) < 3 or crop.shape[0] < 20 or crop.shape[1] < 20:
            return False

        h, w = crop.shape[:2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # 1. Detect skin regions
        skin_mask = ((h_ch < 25) | (h_ch > 165)) & (s_ch > 30) & (v_ch > 60)
        skin_ratio = np.sum(skin_mask) / max(skin_mask.size, 1)

        # Must have some skin, but not too much (a person, not a skin-colored wall)
        if skin_ratio < 0.10 or skin_ratio > 0.60:
            return False

        # 2. Detect dark rectangular objects (phone body)
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        # Phones are typically dark objects
        dark_mask = gray < 60
        dark_ratio = np.sum(dark_mask) / max(dark_mask.size, 1)

        # Need a small dark region (phone), not too large
        if dark_ratio < 0.03 or dark_ratio > 0.25:
            return False

        # 3. Check if dark object is near skin (hand holding phone near face)
        # Dilate skin mask and check overlap with dark mask
        kernel = np.ones((7, 7), np.uint8)
        skin_dilated = cv2.dilate(skin_mask.astype(np.uint8), kernel, iterations=2)
        overlap = np.sum((skin_dilated > 0) & dark_mask) / max(np.sum(dark_mask), 1)

        # Phone must be adjacent to skin (held by hand)
        if overlap < 0.3:
            return False

        # 4. Check that the dark object has rectangular-ish contours
        dark_u8 = dark_mask.astype(np.uint8) * 255
        contours, _ = cv2.findContours(dark_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        phone_like_objects = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < 200:
                continue
            rect = cv2.boundingRect(cnt)
            rw, rh = rect[2], rect[3]
            if rw < 5 or rh < 5:
                continue
            aspect = max(rw, rh) / min(rw, rh)
            # Phones are roughly 2:1 aspect ratio
            if 1.3 < aspect < 4.0:
                phone_like_objects += 1

        return phone_like_objects >= 1
