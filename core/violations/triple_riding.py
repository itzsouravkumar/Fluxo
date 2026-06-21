from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2

from .types import ViolationEvent, ViolationType

TRIPLE_CLASSES = {0: "normal_riders", 1: "triple_riding"}


class TripleRidingDetector:
    """Detects 3+ riders on a two-wheeler using ML classification.

    Uses a YOLO classifier trained on two-wheeler crops with multiple riders.
    Falls back to improved heuristic (vertical head blob analysis with
    spatial constraints) if no model is available.
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

            bw, bh = x2 - x1, y2 - y1
            # Triple riding creates tall bounding boxes; skip small/wide ones
            if bw < 40 or bh < 60:
                continue

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
                        confidence=det_conf * 0.80,
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
        """Improved heuristic: detect vertically stacked head-like blobs.

        Triple riding creates a characteristic vertical stacking pattern:
        multiple head-shaped blobs arranged roughly along the vertical center.

        Improvements over the old heuristic:
        1. Tighter skin color range to avoid matching bags, ads, etc.
        2. Head blobs must be roughly circular (aspect ratio 0.6-1.6)
        3. Blobs must be in the upper 70% (rider heads, not wheels)
        4. Blobs must be roughly vertically aligned (within central 60% horizontally)
        5. Minimum blob size scaled to crop dimensions
        """
        if len(crop.shape) < 3:
            return False

        h, w = crop.shape[:2]
        if h < 60 or w < 30:
            return False

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)

        # Tighter skin detection: use both hue ranges for different skin tones
        # but require higher saturation and value to avoid matching brown/beige objects
        skin_mask = (
            ((hsv[:, :, 0] < 20) | (hsv[:, :, 0] > 170)) &
            (hsv[:, :, 1] > 50) &
            (hsv[:, :, 2] > 80) &
            (hsv[:, :, 2] < 230)  # exclude very bright (white objects)
        )
        skin_u8 = skin_mask.astype(np.uint8) * 255

        # Morphological cleanup
        kernel = np.ones((5, 5), np.uint8)
        closed = cv2.morphologyEx(skin_u8, cv2.MORPH_CLOSE, kernel)
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)

        contours, _ = cv2.findContours(opened, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Minimum head-like area scales with crop size
        min_head_area = max(150, (h * w) * 0.005)
        max_head_area = (h * w) * 0.15  # A single head shouldn't be >15% of crop

        head_centers = []
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < min_head_area or area > max_head_area:
                continue

            rect = cv2.boundingRect(cnt)
            rx, ry, rw, rh = rect
            if rw < 5 or rh < 5:
                continue

            # Head-like aspect ratio (roughly circular)
            aspect = rh / rw
            if aspect < 0.5 or aspect > 2.0:
                continue

            # Must be in the upper 70% of the crop (rider area, not wheels)
            center_y = ry + rh // 2
            if center_y > h * 0.70:
                continue

            # Must be roughly centered horizontally (within central 70%)
            center_x = rx + rw // 2
            if center_x < w * 0.15 or center_x > w * 0.85:
                continue

            head_centers.append((center_x, center_y, area))

        # Need 3+ head-like blobs that are vertically distributed
        if len(head_centers) < 3:
            return False

        # Verify vertical distribution: sort by Y and check spacing
        head_centers.sort(key=lambda c: c[1])
        # Heads should be spread across at least 30% of the crop height
        y_spread = head_centers[-1][1] - head_centers[0][1]
        if y_spread < h * 0.15:
            return False  # All clustered together, probably not separate riders

        return True
