from __future__ import annotations

import re

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
    - Indian plate format validation (HSRP vs non-HSRP)
    - State-code regex gating to reduce false positives
    """

    def __init__(self, use_sr: bool = True):
        self._reader = None
        self._sr_model = None
        self._use_sr = use_sr

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
                import cv2
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
            import cv2
            h, w = crop.shape[:2]
            if w < 80:
                result = sr.upsample(crop)
                return result
        except Exception:
            pass
        return crop

    def read_plate(self, crop: np.ndarray) -> str | None:
        enhanced = self._super_resolve(crop)
        reader = self._init_reader()
        results = reader.readtext(enhanced)
        if not results:
            return None
        plate_text = " ".join([r[1] for r in results])
        plate_text = plate_text.strip().upper()
        if not plate_text:
            return None
        return plate_text

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
        import cv2
        if len(crop.shape) < 3:
            return "unknown"

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        mean_h = np.mean(hsv[:, :, 0])
        mean_s = np.mean(hsv[:, :, 1])
        mean_v = np.mean(hsv[:, :, 2])

        if mean_h > 15 and mean_h < 35 and mean_s > 100:
            return "commercial_yellow"
        if mean_h > 85 and mean_h < 135 and mean_s > 50:
            return "electric_green"
        if mean_s < 30 and mean_v > 180:
            return "private_white"
        return "unknown"
