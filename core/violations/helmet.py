from __future__ import annotations

from pathlib import Path
import numpy as np
from .detector import ViolationEvent, ViolationType


class HelmetDetector:
    """Two-stage helmet absence detection on two-wheeler riders."""

    def __init__(self, classifier_path: str | Path | None = None):
        self.classifier_path = classifier_path
        self._classifier = None

    def _load_classifier(self):
        if self._classifier is None and self.classifier_path is not None:
            from ultralytics import YOLO
            self._classifier = YOLO(str(self.classifier_path))
        return self._classifier

    def detect(self, tracks, frame: np.ndarray, signal_state: str) -> list[ViolationEvent]:
        violations = []
        model = self._load_classifier()
        if model is None:
            return violations

        for track in tracks:
            class_id = getattr(track, "class_id", None)
            if class_id != 0:
                continue

            crop = self._crop_rider(frame, track)
            if crop is None:
                continue

            result = model(crop, verbose=False)[0]
            label = result.probs.top1
            conf = result.probs.top1conf

            if label == 0 and conf > 0.75:
                violations.append(
                    ViolationEvent(
                        type=ViolationType.NO_HELMET,
                        track_id=getattr(track, "id", -1),
                        frame=0,
                        confidence=float(conf),
                    )
                )
        return violations

    def _crop_rider(self, frame: np.ndarray, track) -> np.ndarray | None:
        bbox = getattr(track, "bbox", None)
        if bbox is None:
            return None
        x1, y1, x2, y2 = map(int, bbox)
        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None
        return crop
