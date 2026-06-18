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

    def detect_with_enhancement(
        self,
        frame: np.ndarray,
        enhancer=None,
        tile_detector=None,
    ) -> tuple[list[Detection], dict]:
        """Run detection with quality-aware enhancement pipeline.

        Auto-detects low-quality footage and applies super-resolution
        + sharpening. Optionally uses tile-based detection for far/small objects.
        Returns detections plus quality metadata.
        """
        from .enhancement import FrameEnhancer, TileDetector

        if enhancer is None:
            enhancer = FrameEnhancer()
        if tile_detector is None:
            tile_detector = TileDetector(tile_size=640, overlap=0.2)

        enhanced, quality = enhancer.enhance(frame)

        if quality.get("needs_enhancement"):
            def _detect_fn(tile, conf):
                model = self._load_model()
                results = model(tile, conf=conf, iou=self.iou, verbose=False)[0]
                dets = []
                if results.boxes is not None:
                    for box in results.boxes:
                        dets.append({
                            "bbox": box.xyxy[0].cpu().numpy(),
                            "class_id": int(box.cls[0].item()),
                            "confidence": float(box.conf[0].item()),
                        })
                return dets

            tile_dets = tile_detector.detect_with_tiles(enhanced, _detect_fn, conf=self.conf)
            detections = [
                Detection(
                    bbox=d["bbox"],
                    class_id=d["class_id"],
                    confidence=d["confidence"],
                )
                for d in tile_dets
            ]
        else:
            detections = self.detect(enhanced)

        return detections, quality
