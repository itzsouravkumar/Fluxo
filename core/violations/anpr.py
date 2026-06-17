from __future__ import annotations

import numpy as np


class ANPRReader:
    """Automatic Number Plate Recognition using EasyOCR."""

    def __init__(self):
        self._reader = None

    def _init_reader(self):
        if self._reader is None:
            import easyocr
            self._reader = easyocr.Reader(["en"])
        return self._reader

    def read_plate(self, crop: np.ndarray) -> str | None:
        reader = self._init_reader()
        results = reader.readtext(crop)
        if not results:
            return None
        plate_text = " ".join([r[1] for r in results])
        return plate_text.strip() if plate_text.strip() else None
