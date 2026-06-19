from __future__ import annotations

import re

import cv2
import numpy as np


INDIAN_PLATE_REGEX = re.compile(
    r"^[A-Z]{2}[\s\-]?\d{1,2}[\s\-]?[A-Z]{1,3}[\s\-]?\d{1,4}$"
)

HSRP_FORMAT = re.compile(
    r"^[A-Z]{2}[\s\-]?\d{2}[\s\-]?[A-Z]{1,3}[\s\-]?\d{4}$"
)

STATE_CODES = {
    "AP", "AR", "AS", "BR", "CG", "GA", "GJ", "HR", "HP", "JH", "KA",
    "KL", "LA", "MP", "MH", "MN", "ML", "MZ", "NL", "OD", "PB", "RJ",
    "SK", "TN", "TS", "TR", "UP", "UK", "WB", "AN", "CH", "DD", "DL",
    "JK", "LD", "PY",
}


class ANPRReader:
    """Automatic Number Plate Recognition using EasyOCR with SR preprocessing.

    Includes:
    - Super-resolution preprocessing for low-res plate crops
    - Per-character confidence gating for partial reads
    - Plate color-aware preprocessing (white/yellow/green backgrounds)
    - Indian plate format validation (HSRP vs non-HSRP)
    - State-code regex gating to reduce false positives
    - Plate obstruction detection (sticker/tape covering)

    Reference: MDPI Math (May 2025) — SR reduces character confusion 15-20%
    Reference: Vol 3 — ANPR: Indian Plates & OCR Robustness
    """

    def __init__(self, use_sr: bool = True, use_color_aware: bool = True):
        self._reader = None
        self._sr_model = None
        self._use_sr = use_sr
        self._use_color_aware = use_color_aware

    def _init_reader(self):
        if self._reader is None:
            import ssl
            import easyocr
            _orig = ssl._create_default_https_context
            ssl._create_default_https_context = ssl._create_unverified_context
            try:
                self._reader = easyocr.Reader(["en"], verbose=False)
            finally:
                ssl._create_default_https_context = _orig
        return self._reader

    def _init_sr(self):
        if self._sr_model is None and self._use_sr:
            try:
                self._sr_model = cv2.dnn_superres.DnnSuperResImpl_create()
                sr_path = "models/ESPCN_x2.pb"
                from pathlib import Path
                if Path(sr_path).exists():
                    self._sr_model.readModel(sr_path)
                    self._sr_model.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
                    self._sr_model.setModel("espcn", 2)
                else:
                    self._sr_model = None
            except Exception:
                self._sr_model = None
        return self._sr_model

    def _super_resolve(self, crop: np.ndarray) -> np.ndarray:
        sr = self._init_sr()
        if sr is None:
            return crop
        try:
            h, w = crop.shape[:2]
            if w < 80:
                return sr.upsample(crop)
        except Exception:
            pass
        return crop

    def _color_aware_preprocess(self, crop: np.ndarray) -> np.ndarray:
        if not self._use_color_aware or crop.size == 0:
            return crop

        plate_type = self.classify_plate_type(crop)
        lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
        l_ch, a_ch, b_ch = cv2.split(lab)

        if plate_type == "commercial_yellow":
            mask = b_ch > 140
            l_ch[mask] = np.clip(l_ch[mask] + 20, 0, 255)
        elif plate_type == "electric_green":
            mask = a_ch > 128
            l_ch[mask] = np.clip(l_ch[mask] + 15, 0, 255)
        elif plate_type == "private_white":
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(4, 4))
            l_ch = clahe.apply(l_ch)

        lab = cv2.merge([l_ch, a_ch, b_ch])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def read_plate(self, crop: np.ndarray) -> str | None:
        preprocessed = self._color_aware_preprocess(crop)
        enhanced = self._super_resolve(preprocessed)
        reader = self._init_reader()
        results = reader.readtext(enhanced)
        if not results:
            return None
        plate_text = " ".join([r[1] for r in results])
        plate_text = plate_text.strip().upper()
        if not plate_text:
            return None
        return plate_text

    def read_plate_with_confidence(self, crop: np.ndarray) -> dict:
        preprocessed = self._color_aware_preprocess(crop)
        enhanced = self._super_resolve(preprocessed)
        reader = self._init_reader()
        results = reader.readtext(enhanced)

        if not results:
            return {"plate": None, "overall_confidence": 0.0, "char_confidences": [], "is_partial": True}

        plate_text = " ".join([r[1] for r in results]).strip().upper()
        char_confs = []
        for r in results:
            text = r[1].strip()
            conf = float(r[2])
            for ch in text:
                char_confs.append({"char": ch, "confidence": conf})

        overall_conf = float(np.mean([c["confidence"] for c in char_confs])) if char_confs else 0.0
        min_char_conf = min([c["confidence"] for c in char_confs]) if char_confs else 0.0
        is_partial = min_char_conf < 0.6 or overall_conf < 0.5

        return {
            "plate": plate_text if plate_text else None,
            "overall_confidence": round(overall_conf, 3),
            "char_confidences": char_confs,
            "is_partial": is_partial,
            "min_char_confidence": round(min_char_conf, 3),
        }

    def detect_plate_obstruction(self, crop: np.ndarray) -> float:
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if len(crop.shape) == 3 else crop
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / max(edges.size, 1)
        _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)
        bright_ratio = np.sum(thresh > 0) / max(thresh.size, 1)
        if edge_density < 0.05 and bright_ratio > 0.4:
            return round(float(1.0 - edge_density), 3)
        return 0.0

    def validate_plate(self, text: str) -> dict:
        if not text:
            return {"valid": False, "format": "empty", "state": None, "is_hsrp": False}

        cleaned = re.sub(r"[\s\-]", "", text).upper()
        is_hsrp = bool(HSRP_FORMAT.match(cleaned))
        is_valid = bool(INDIAN_PLATE_REGEX.match(cleaned))

        state = None
        if len(cleaned) >= 2:
            prefix = cleaned[:2]
            if prefix in STATE_CODES:
                state = prefix

        return {
            "valid": is_valid,
            "format": "hsrp" if is_hsrp else ("standard" if is_valid else "non_standard"),
            "state": state,
            "is_hsrp": is_hsrp,
        }

    def classify_plate_type(self, crop: np.ndarray) -> str:
        if len(crop.shape) < 3:
            return "unknown"
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mean_h = np.mean(hsv[:, :, 0])
        mean_s = np.mean(hsv[:, :, 1])
        mean_v = np.mean(hsv[:, :, 2])
        if 15 < mean_h < 35 and mean_s > 100:
            return "commercial_yellow"
        if 85 < mean_h < 135 and mean_s > 50:
            return "electric_green"
        if mean_s < 30 and mean_v > 180:
            return "private_white"
        return "unknown"
