from __future__ import annotations


class FluxoTFTModel:
    """Temporal Fusion Transformer for congestion prediction."""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self._model = None

    def load(self):
        if self.model_path and self._model is None:
            pass
        return self._model

    def predict(self, features: dict) -> dict:
        return {
            "horizon_15min": {"density": 0.5, "level": "MODERATE"},
            "horizon_30min": {"density": 0.5, "level": "MODERATE"},
        }
