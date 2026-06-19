from __future__ import annotations

import cv2
import numpy as np


class WeatherPreprocessor:
    """Handles fog, rain, and monsoon conditions for CCTV footage.

    Implements:
    - Dark-channel prior dehazing for fog (He et al., TPAMI 2011)
    - Rain streak detection and removal
    - Adherent raindrop detection on lens covers
    - Night-mode HDR tone mapping for headlight glare

    Reference: ODD-Net (Nature Sci Rep, Dec 2024) for dehazing.
    Reference: Li et al., IEEE TIP 2021 for online rain/snow removal.
    """

    def __init__(
        self,
        fog_dark_channel_size: int = 15,
        fog_omega: float = 0.95,
        fog_t0: float = 0.1,
        rain_streak_threshold: float = 25.0,
        raindrop_coverage_threshold: float = 0.20,
        glare_saturation_threshold: float = 0.80,
    ):
        self.fog_dark_channel_size = fog_dark_channel_size
        self.fog_omega = fog_omega
        self.fog_t0 = fog_t0
        self.rain_streak_threshold = rain_streak_threshold
        self.raindrop_coverage_threshold = raindrop_coverage_threshold
        self.glare_saturation_threshold = glare_saturation_threshold
        self._prev_frame = None

    def estimate_visibility(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        contrast = np.std(gray) / 128.0
        visibility = min(1.0, (laplacian_var / 200.0) * 0.5 + contrast * 0.5)
        return round(visibility, 3)

    def detect_fog(self, frame: np.ndarray) -> bool:
        dark_channel = self._dark_channel(frame)
        mean_dark = np.mean(dark_channel)
        return mean_dark < 0.15

    def dehaze(self, frame: np.ndarray) -> np.ndarray:
        dark_channel = self._dark_channel(frame)
        h, w = dark_channel.shape
        num_pixels = h * w
        num_bright = int(max(num_pixels * 0.001, 1))

        flat = dark_channel.ravel()
        indices = np.argpartition(flat, -num_bright)[-num_bright:]

        atmospheric = np.zeros(3, dtype=np.float64)
        for c in range(3):
            channel = frame[:, :, c].ravel().astype(np.float64)
            atmospheric[c] = np.mean(channel[indices])
        atmospheric = np.clip(atmospheric, 128, 255)

        norm = np.zeros_like(dark_channel, dtype=np.float64)
        for c in range(3):
            norm += dark_channel.astype(np.float64) / max(atmospheric[c], 1.0)
        norm /= 3.0

        transmission = 1.0 - self.fog_omega * norm
        transmission = np.clip(transmission, self.fog_t0, 1.0)

        result = np.zeros_like(frame, dtype=np.float64)
        for c in range(3):
            result[:, :, c] = (
                frame[:, :, c].astype(np.float64) - atmospheric[c]
            ) / np.maximum(transmission, self.fog_t0) + atmospheric[c]

        return np.clip(result, 0, 255).astype(np.uint8)

    def detect_rain_streaks(self, frame: np.ndarray) -> float:
        if self._prev_frame is None:
            self._prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
            return 0.0

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        diff = cv2.absdiff(self._prev_frame, gray)
        _, thresh = cv2.threshold(diff, self.rain_streak_threshold, 255, cv2.THRESH_BINARY)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 5))
        streaks = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

        coverage = np.sum(streaks > 0) / max(streaks.size, 1)
        self._prev_frame = gray
        return round(float(coverage), 4)

    def detect_raindrops_on_lens(self, frame: np.ndarray) -> float:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        blurred = cv2.GaussianBlur(gray, (15, 15), 0)
        diff = cv2.absdiff(gray, blurred)
        _, thresh = cv2.threshold(diff, 30, 255, cv2.THRESH_BINARY)
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        drop_area = 0
        h, w = gray.shape
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 100:
                drop_area += area

        coverage = drop_area / max(h * w, 1)
        return round(float(coverage), 4)

    def detect_headlight_glare(self, plate_crop: np.ndarray) -> float:
        if plate_crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY) if len(plate_crop.shape) == 3 else plate_crop
        saturation_ratio = np.sum(gray > 240) / max(gray.size, 1)
        return round(float(saturation_ratio), 4)

    def tone_map_hdr(self, frame: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l_ch = lab[:, :, 0].astype(np.float64)
        p1 = np.percentile(l_ch, 1)
        p99 = np.percentile(l_ch, 99)
        if p99 - p1 < 10:
            return frame
        l_norm = np.clip((l_ch - p1) / max(p99 - p1, 1.0), 0, 1)
        l_mapped = (l_norm * 255).astype(np.uint8)
        lab[:, :, 0] = l_mapped
        return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    def _dark_channel(self, frame: np.ndarray) -> np.ndarray:
        min_channel = np.min(frame, axis=2)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT,
            (self.fog_dark_channel_size, self.fog_dark_channel_size),
        )
        return cv2.erode(min_channel, kernel)

    def preprocess(self, frame: np.ndarray, night_mode: bool = False) -> tuple[np.ndarray, dict]:
        metadata = {"fog_detected": False, "rain_streak_coverage": 0.0,
                     "raindrop_coverage": 0.0, "glare_ratio": 0.0, "visibility": 1.0}

        metadata["visibility"] = self.estimate_visibility(frame)

        if self.detect_fog(frame):
            frame = self.dehaze(frame)
            metadata["fog_detected"] = True

        rain = self.detect_rain_streaks(frame)
        metadata["rain_streak_coverage"] = rain
        if rain > 0.15:
            frame = self._remove_rain_streaks(frame)

        drops = self.detect_raindrops_on_lens(frame)
        metadata["raindrop_coverage"] = drops
        if drops > self.raindrop_coverage_threshold:
            metadata["degraded"] = True

        if night_mode:
            frame = self.tone_map_hdr(frame)

        return frame, metadata

    def _remove_rain_streaks(self, frame: np.ndarray) -> np.ndarray:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 7))
        opened = cv2.morphologyEx(frame, cv2.MORPH_OPEN, kernel)
        return cv2.addWeighted(frame, 0.7, opened, 0.3, 0)
