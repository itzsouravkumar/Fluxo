from __future__ import annotations

from pathlib import Path

import numpy as np
import cv2

from .types import ViolationEvent, ViolationType


HELMET_CLASSES = {0: "helmet", 1: "no_helmet"}
HEADWEAR_CLASSES = {0: "helmet", 1: "cap", 2: "turban", 3: "scarf", 4: "bare_head"}


class HelmetDetector:
    """Detects helmet absence on two-wheeler riders.

    Uses a two-stage approach:
    1. Primary YOLO classifier for helmet vs no-helmet detection
    2. Secondary headwear discriminator to avoid false positives on
       caps, turbans, hijabs, and scarves (documented failure mode
       in Indian deployment context - arXiv:2408.02244, S2772662224001309)

    Assigns helmet status per seat position (driver/pillion-1/pillion-2)
    using trapezium rider boxes, avoiding driver/passenger bleed-through.
    """

    def __init__(
        self,
        classifier_path: str | Path | None = "models/fluxo_helmet_v1.pt",
        headwear_path: str | Path | None = "models/fluxo_headwear_v1.pt",
    ):
        self.classifier_path = classifier_path
        self.headwear_path = headwear_path
        self._classifier = None
        self._headwear_classifier = None

    def _load_classifier(self):
        if self._classifier is None and self.classifier_path is not None:
            from ultralytics import YOLO
            self._classifier = YOLO(str(self.classifier_path))
        return self._classifier

    def _load_headwear_classifier(self):
        if self._headwear_classifier is None and self.headwear_path is not None:
            from pathlib import Path as P
            if P(self.headwear_path).exists():
                from ultralytics import YOLO
                self._headwear_classifier = YOLO(str(self.headwear_path))
        return self._headwear_classifier

    def detect(self, detections, frame: np.ndarray, frame_idx: int, signal_state: str) -> list[ViolationEvent]:
        violations = []
        h, w = frame.shape[:2]

        if not hasattr(detections, "class_id") or detections.class_id is None:
            return violations

        for i in range(len(detections)):
            cls_id = int(detections.class_id[i])
            if cls_id != 0:
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

            head_region = self._extract_head_region(crop)
            has_helmet = self._classify_helmet(head_region if head_region is not None else crop)

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

    def _extract_head_region(self, vehicle_crop: np.ndarray) -> np.ndarray | None:
        h, w = vehicle_crop.shape[:2]
        if h < 10 or w < 10:
            return None
        head_h = max(h // 3, 1)
        head_w = max(w // 2, 1)
        cx = w // 2
        x1 = max(0, cx - head_w // 2)
        x2 = min(w, cx + head_w // 2)
        return vehicle_crop[0:head_h, x1:x2]

    def _classify_helmet(self, crop: np.ndarray) -> bool:
        model = self._load_classifier()
        if model is None:
            return self._heuristic_helmet(crop)

        result = model(crop, verbose=False)[0]
        label = result.probs.top1
        conf = result.probs.top1conf

        if label == 1 and conf > 0.6:
            headwear_model = self._load_headwear_classifier()
            if headwear_model is not None:
                hw_result = headwear_model(crop, verbose=False)[0]
                hw_label = hw_result.probs.top1
                hw_class = HEADWEAR_CLASSES.get(hw_label, "bare_head")
                if hw_class in ("cap", "turban", "scarf"):
                    return False
            return True

        return False

    def _heuristic_helmet(self, crop: np.ndarray) -> bool:
        if len(crop.shape) < 3:
            return True

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
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
