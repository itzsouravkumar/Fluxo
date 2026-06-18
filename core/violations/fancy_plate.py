from __future__ import annotations

import re

import cv2
import numpy as np

from .types import ViolationEvent, ViolationType

INDIAN_PLATE_REGEX = re.compile(
    r"^[A-Z]{2}[\s\-]?\d{1,2}[\s\-]?[A-Z]{1,3}[\s\-]?\d{1,4}$"
)


class FancyPlateDetector:
    """Detects non-standard, modified, or obscured number plates.

    Flags plates where:
    - YOLO26 localises a plate region but OCR confidence is low
    - OCR output doesn't match any valid Indian plate format
    - Plate region is absent or covered (cloth, tape)

    This is a novel violation class — the TR-TRVD paper (2024) explicitly
    noted that fancy number plate detection has received limited attention.
    """

    def __init__(self, ocr_confidence_threshold: float = 0.3):
        self.ocr_confidence_threshold = ocr_confidence_threshold

    def detect(
        self,
        detections,
        frame: np.ndarray,
        frame_idx: int,
        signal_state: str = "GREEN",
        ocr_results: dict[int, tuple[str | None, float]] | None = None,
    ) -> list[ViolationEvent]:
        violations = []
        h, w = frame.shape[:2]

        if not hasattr(detections, "class_id") or detections.class_id is None:
            return violations

        if ocr_results is None:
            ocr_results = {}

        for i in range(len(detections)):
            cls_id = int(detections.class_id[i])
            if cls_id not in (0, 2):
                continue

            tid = int(detections.tracker_id[i]) if hasattr(detections, "tracker_id") and detections.tracker_id is not None else -1
            if tid < 0:
                continue

            plate_text, ocr_conf = ocr_results.get(tid, (None, 0.0))

            if plate_text is None:
                bbox = detections.xyxy[i].astype(int)
                x1, y1, x2, y2 = bbox
                x1 = max(0, min(x1, w - 1))
                x2 = max(0, min(x2, w))
                y1 = max(0, min(y1, h - 1))
                y2 = max(0, min(y2, h))
                crop = frame[y1:y2, x1:x2]
                if crop.size > 0 and self._is_plate_region_visible(crop):
                    violations.append(
                        ViolationEvent(
                            type=ViolationType.FANCY_PLATE,
                            track_id=tid,
                            frame=frame_idx,
                            confidence=0.6,
                            bbox=tuple(bbox),
                        )
                    )
            else:
                is_valid = bool(INDIAN_PLATE_REGEX.match(plate_text.strip().upper()))
                if not is_valid and ocr_conf > self.ocr_confidence_threshold:
                    bbox = detections.xyxy[i].astype(int)
                    violations.append(
                        ViolationEvent(
                            type=ViolationType.FANCY_PLATE,
                            track_id=tid,
                            frame=frame_idx,
                            confidence=min(ocr_conf, 0.9),
                            plate_number=plate_text,
                            bbox=tuple(bbox),
                        )
                    )

        return violations

    def _is_plate_region_visible(self, vehicle_crop: np.ndarray) -> bool:
        h, w = vehicle_crop.shape[:2]
        plate_region = vehicle_crop[int(h * 0.5):, int(w * 0.15):int(w * 0.85)]
        if plate_region.size == 0:
            return False

        gray = cv2.cvtColor(plate_region, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_ratio = np.sum(edges > 0) / max(edges.size, 1)
        return edge_ratio > 0.05
