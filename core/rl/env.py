from __future__ import annotations

import gymnasium as gym
import numpy as np


class FluxoSignalEnv(gym.Env):
    """RL environment for adaptive signal control.

    State: [queue_N, queue_S, queue_E, queue_W, elapsed_green, hour_sin, hour_cos, predicted_density_15min]
    Action: phase_duration ∈ {15, 20, 25, 30, 40, 50, 60, 90, 120} seconds
    """

    ACTION空間 = [15, 20, 25, 30, 40, 50, 60, 90, 120]
    PHASE_MIN = 15
    PHASE_MAX = 120
    STARVATION_THRESHOLD = 90

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None):
        super().__init__()
        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(8,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(self.ACTION空间))
        self.render_mode = render_mode

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs = np.zeros(8, dtype=np.float32)
        return obs, {}

    def step(self, action):
        obs = np.zeros(8, dtype=np.float32)
        reward = 0.0
        terminated = False
        truncated = False
        info = {}
        return obs, reward, terminated, truncated, info

    def render(self):
        if self.render_mode == "human":
            pass
