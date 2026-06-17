from __future__ import annotations

from .model import FluxoTFTModel


class TFTInference:
    """Real-time prediction service."""

    def __init__(self, model: FluxoTFTModel):
        self.model = model

    async def forecast(self, junction_id: str, features: dict | None = None) -> dict:
        return {
            "junction_id": junction_id,
            "forecast": [
                {"horizon_min": 15, "density_score": 0.5, "ci_low": 0.4, "ci_high": 0.6},
                {"horizon_min": 30, "density_score": 0.5, "ci_low": 0.35, "ci_high": 0.65},
            ],
            "congestion_level": "MODERATE",
        }
