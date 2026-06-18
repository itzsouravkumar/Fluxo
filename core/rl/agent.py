from __future__ import annotations

import logging

log = logging.getLogger(__name__)


class RLAgent:
    """Inference wrapper for trained PPO agent."""

    PHASE_DURATIONS = [15, 20, 25, 30, 40, 50, 60, 90, 120]

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self._model = None

    def load(self):
        if self.model_path and self._model is None:
            try:
                from stable_baselines3 import PPO
                self._model = PPO.load(self.model_path)
                log.info(f"Loaded RL model from {self.model_path}")
            except Exception as e:
                log.warning(f"Failed to load RL model: {e}")
        return self._model

    def recommend(self, state: list[float]) -> dict:
        model = self.load()
        if model is None:
            return {
                "phase": "N-S",
                "duration_s": 30,
                "confidence": 0.0,
                "reason": "no_model_loaded",
            }

        try:
            import numpy as np
            obs = np.array(state, dtype=np.float32)
            action, _ = model.predict(obs, deterministic=True)
            duration = self.PHASE_DURATIONS[int(action)]

            phase_names = ["N-S", "E-W", "N-S", "E-W"]
            phase_idx = int(action) % 4

            return {
                "phase": phase_names[phase_idx],
                "duration_s": duration,
                "confidence": 0.85,
                "reason": "rl_agent_recommendation",
            }
        except Exception as e:
            log.warning(f"RL prediction failed: {e}")
            return {
                "phase": "N-S",
                "duration_s": 30,
                "confidence": 0.0,
                "reason": f"prediction_error: {e}",
            }
