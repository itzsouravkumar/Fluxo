from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2

from .types import ViolationEvent, ViolationType

SEATBELT_CLASSES = {0: "with_seatbelt", 1: "without_seatbelt"}

# Seatbelts are relevant for cars/LMVs, not two-wheelers
SEATBELT_VEHICLE_CLASSES = {2}  # light_motor_vehicle only


class SeatbeltDetector:
    """Detects seatbelt absence on car/LMV drivers using ML classification.

    Uses a YOLO classifier trained on driver crops.
    Falls back to heuristic (diagonal edge scan) if no model is available.

    Only checks light motor vehicles (class 2) — seatbelts don't apply
    to two-wheelers, buses, or heavy vehicles.
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
            # Only check cars/LMVs for seatbelts
            if cls_id not in SEATBELT_VEHICLE_CLASSES:
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

            bw, bh = x2 - x1, y2 - y1
            # Need a reasonably sized crop to analyze
            if bw < 60 or bh < 60:
                continue

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
                        confidence=det_conf * 0.85,  # discount since heuristic is uncertain
                        bbox=tuple(bbox),
                    )
                )
        return violations

    def _extract_driver_region(self, vehicle_crop: np.ndarray) -> np.ndarray | None:
        """Extract the left-front driver area of a car crop.

        For a front-facing camera, the driver is typically in the
        upper-left quadrant (Indian roads = right-hand drive, driver on right side
        of the car from camera's perspective).
        """
        h, w = vehicle_crop.shape[:2]
        if h < 30 or w < 30:
            return None
        # Upper 60% vertically, right 60% horizontally (right-hand drive)
        driver_h = max(int(h * 0.6), 1)
        driver_w_start = max(int(w * 0.4), 0)
        return vehicle_crop[0:driver_h, driver_w_start:w]

    def _classify_seatbelt(self, crop: np.ndarray) -> bool:
        model = self._load_classifier()
        if model is None:
            return self._heuristic_seatbelt(crop)

        result = model(crop, verbose=False)[0]
        label = result.probs.top1
        conf = result.probs.top1conf

        return label == 0 and conf > 0.6

    def _heuristic_seatbelt(self, crop: np.ndarray) -> bool:
        """Improved heuristic: look for a diagonal dark band across the chest region.

        Seatbelts appear as a continuous dark diagonal stripe from shoulder to hip.
        We look for consistent diagonal edge responses in the expected seatbelt
        angle range (30-60 degrees from horizontal).
        """
        if len(crop.shape) < 3:
            return True  # can't analyze, assume present to avoid FP

        h, w = crop.shape[:2]
        if h < 20 or w < 20:
            return True

        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

        # Focus on chest region (middle 40% vertically)
        chest_start = int(h * 0.2)
        chest_end = int(h * 0.7)
        chest = gray[chest_start:chest_end, :]
        if chest.size == 0:
            return True

        # Use Hough line detection to find diagonal lines
        edges = cv2.Canny(chest, 50, 150)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=20,
                                minLineLength=min(w, h) // 4,
                                maxLineGap=5)

        if lines is None:
            return True  # no strong lines = assume seatbelt present (conservative)

        # Check for lines in the seatbelt angle range (30-60 degrees)
        diagonal_line_count = 0
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1:
                continue
            angle = abs(np.degrees(np.arctan2(abs(y2 - y1), abs(x2 - x1))))
            if 25 <= angle <= 65:
                # Check if the line spans a reasonable width
                line_len = ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5
                if line_len > min(w, h) * 0.2:
                    diagonal_line_count += 1

        # Seatbelt present if we find at least one strong diagonal line
        return diagonal_line_count >= 1
