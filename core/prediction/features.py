from __future__ import annotations

import numpy as np


FEATURES = {
    "time_varying_known": [
        "hour_of_day",
        "day_of_week",
        "is_public_holiday",
        "is_school_day",
        "weather_rain",
        "metro_construction",
    ],
    "time_varying_unknown": [
        "vehicle_count_lane_1",
        "vehicle_count_lane_2",
        "pce_density_score",
        "average_speed_kmh",
        "incident_flag",
    ],
    "static": [
        "junction_id",
        "num_lanes",
        "junction_type",
        "peak_zone",
    ],
}


class FeatureEngineer:
    """Feature engineering pipeline for TFT input."""

    def __init__(self):
        pass

    def encode_cyclical(self, value: float, max_value: float) -> tuple[float, float]:
        sin_val = np.sin(2 * np.pi * value / max_value)
        cos_val = np.cos(2 * np.pi * value / max_value)
        return float(sin_val), float(cos_val)

    def prepare_features(self, raw_data: dict) -> dict:
        return raw_data
