from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Detection:
    bbox: np.ndarray
    class_id: int
    confidence: float
    track_id: int | None = None


class FluxoDetector:
    """YOLO26 inference wrapper for Indian traffic vehicle detection.

    Uses YOLO26 (NMS-free design) for single-pass unified detection.
    One detection pass drives all violation classifiers via shared track IDs.
    """

    CLASSES = {
        0: "two_wheeler",
        1: "auto_rickshaw",
        2: "light_motor_vehicle",
        3: "bus",
        4: "heavy_vehicle",
        5: "pedestrian",
        6: "emergency_vehicle",
    }

    PCE_MAP = {
        0: 0.25,
        1: 0.5,
        2: 1.0,
        3: 3.0,
        4: 3.5,
        5: 0.0,
        6: 0.0,
    }

    def __init__(self, model_path: str | Path = "yolo26n.pt", conf: float = 0.4, iou: float = 0.5):
        self.model_path = model_path
        self.conf = conf
        self.iou = iou
        self._model = None

    def _load_model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(str(self.model_path))
        return self._model

    def detect(self, frame: np.ndarray) -> list[Detection]:
        model = self._load_model()
        results = model(frame, conf=self.conf, iou=self.iou, verbose=False)[0]
        detections = []
        if results.boxes is not None:
            for box in results.boxes:
                detections.append(
                    Detection(
                        bbox=box.xyxy[0].cpu().numpy(),
                        class_id=int(box.cls[0].item()),
                        confidence=float(box.conf[0].item()),
                    )
                )
        return detections
