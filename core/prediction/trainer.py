from __future__ import annotations


class TFTTrainer:
    """Training loop for TFT forecaster."""

    def __init__(self, model, config: dict | None = None):
        self.model = model
        self.config = config or {}

    def train(self, data_path: str, epochs: int = 50, horizon: int = 30):
        pass

    def validate(self, val_data):
        pass
