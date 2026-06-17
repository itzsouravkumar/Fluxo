from __future__ import annotations


class RLAgent:
    """Inference wrapper for trained PPO agent."""

    def __init__(self, model_path: str | None = None):
        self.model_path = model_path
        self._model = None

    def load(self):
        if self.model_path and self._model is None:
            from stable_baselines3 import PPO
            self._model = PPO.load(self.model_path)
        return self._model

    def recommend(self, state: list[float]) -> dict:
        model = self.load()
        if model is None:
            return {"phase": "N-S", "duration_s": 30, "confidence": 0.0, "reason": "no_model_loaded"}

        action, _ = model.predict(state, deterministic=True)
        duration = [15, 20, 25, 30, 40, 50, 60, 90, 120][action]
        return {
            "phase": "N-S",
            "duration_s": duration,
            "confidence": 0.85,
            "reason": "rl_agent_recommendation",
        }
