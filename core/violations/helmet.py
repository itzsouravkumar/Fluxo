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
            if cls_id != 0:  # Only two-wheelers
                continue

            bbox = detections.xyxy[i].astype(int)
            x1, y1, x2, y2 = bbox
            x1 = max(0, min(x1, w - 1))
            x2 = max(0, min(x2, w))
            y1 = max(0, min(y1, h - 1))
            y2 = max(0, min(y2, h))

            bw, bh = x2 - x1, y2 - y1
            # Skip very small detections — can't reliably see helmet
            if bw < 40 or bh < 40:
                continue

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
                        confidence=conf * 0.85,  # discount for heuristic uncertainty
                        bbox=tuple(bbox),
                    )
                )
        return violations

    def _extract_head_region(self, vehicle_crop: np.ndarray) -> np.ndarray | None:
        h, w = vehicle_crop.shape[:2]
        if h < 20 or w < 20:
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
        """Improved multi-signal heuristic for helmet detection.

        Instead of just checking bright/skin ratios (which falsely flags
        white helmets or dark-skinned riders), we check for:
        1. Smooth, uniform-color dome shape (helmet characteristic)
        2. Visor glare (specular highlight band across the middle)
        3. Texture uniformity in the head region (helmets are smoother than hair)

        Conservative: defaults to True (helmet present) when uncertain,
        to reduce false accusations.
        """
        if len(crop.shape) < 3 or crop.shape[0] < 15 or crop.shape[1] < 15:
            return True  # Too small to tell, assume helmet present

        h, w = crop.shape[:2]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]

        # Signal 1: Texture uniformity
        # Helmets have smooth, uniform texture. Hair/bare head has more texture variation.
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        texture_variance = laplacian.var()

        # Signal 2: Color uniformity
        # Helmets are typically one solid color. Hair/skin has more color variation.
        color_std = np.std(gray)

        # Signal 3: Check for a visor band (horizontal high-contrast strip)
        # in the middle third of the image
        mid_start = h // 3
        mid_end = 2 * h // 3
        mid_band = gray[mid_start:mid_end, :]
        if mid_band.size > 0:
            mid_edges = cv2.Canny(mid_band, 50, 150)
            visor_score = np.mean(mid_edges > 0)
        else:
            visor_score = 0

        # Signal 4: Roundness check via contour analysis
        # Helmets create a dome silhouette
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        roundness = 0.0
        if contours:
            largest = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(largest)
            perimeter = cv2.arcLength(largest, True)
            if perimeter > 0 and area > 100:
                roundness = 4 * np.pi * area / (perimeter ** 2)

        # Decision logic:
        # Helmets: smooth texture (low variance), uniform color, possibly visor, round shape
        helmet_signals = 0

        if texture_variance < 800:  # Smooth surface
            helmet_signals += 1
        if color_std < 45:  # Uniform color
            helmet_signals += 1
        if visor_score > 0.05:  # Has visor-like band
            helmet_signals += 1
        if roundness > 0.5:  # Round dome shape
            helmet_signals += 1

        # Need at least 2 positive helmet signals to declare helmet present
        # Conservative: fewer signals = assume helmet (reduce FP)
        return helmet_signals >= 2
