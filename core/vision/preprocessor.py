from __future__ import annotations

import cv2
import numpy as np

from .weather import WeatherPreprocessor


class FramePreprocessor:
    """CLAHE preprocessing for night/rain modes + weather-aware preprocessing + resize.

    Integrates:
    - CLAHE for night conditions
    - Dark-channel dehazing for fog (He et al., TPAMI 2011)
    - Rain streak removal
    - HDR tone mapping for headlight glare
    - Plate color-aware contrast enhancement
    """

    def __init__(
        self,
        clip_limit: float = 2.0,
        tile_grid_size: tuple[int, int] = (8, 8),
        target_size: tuple[int, int] = (640, 640),
        enable_weather: bool = True,
    ):
        self.clip_limit = clip_limit
        self.tile_grid_size = tile_grid_size
        self.target_size = target_size
        self.clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
        self._weather = WeatherPreprocessor() if enable_weather else None

    def apply_clahe(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        lightness, a_ch, b_ch = cv2.split(lab)
        lightness = self.clahe.apply(lightness)
        lab = cv2.merge([lightness, a_ch, b_ch])
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def resize(self, frame: np.ndarray) -> np.ndarray:
        return cv2.resize(frame, self.target_size, interpolation=cv2.INTER_LINEAR)

    def preprocess(self, frame: np.ndarray, night_mode: bool = False) -> tuple[np.ndarray, dict]:
        metadata = {"night_mode": night_mode, "weather": {}}

        if night_mode:
            frame = self.apply_clahe(frame)
            metadata["clahe_applied"] = True

        if self._weather is not None:
            frame, weather_meta = self._weather.preprocess(frame, night_mode=night_mode)
            metadata["weather"] = weather_meta

        frame = self.resize(frame)
        return frame, metadata
