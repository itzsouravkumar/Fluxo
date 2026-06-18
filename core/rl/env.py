from __future__ import annotations

import gymnasium as gym
import numpy as np

from .reward import compute_reward
from .constraints import SafetyConstraints


class FluxoSignalEnv(gym.Env):
    """RL environment for adaptive signal control.

    State: [queue_N, queue_S, queue_E, queue_W, elapsed_green, hour_sin, hour_cos, predicted_density_15min]
    Action: phase_duration index -> {15, 20, 25, 30, 40, 50, 60, 90, 120} seconds
    """

    PHASE_DURATIONS = [15, 20, 25, 30, 40, 50, 60, 90, 120]
    MIN_GREEN = 15
    MAX_GREEN = 120

    metadata = {"render_modes": ["human"]}

    def __init__(self, render_mode=None, num_lanes=4, max_steps=3600):
        super().__init__()
        self.num_lanes = num_lanes
        self.max_steps = max_steps
        obs_size = num_lanes + 3  # queues + elapsed + hour_sin/cos + predicted_density

        self.observation_space = gym.spaces.Box(low=0.0, high=1.0, shape=(obs_size,), dtype=np.float32)
        self.action_space = gym.spaces.Discrete(len(self.PHASE_DURATIONS))
        self.render_mode = render_mode
        self.constraints = SafetyConstraints()

        self._step_count = 0
        self._queues = np.zeros(num_lanes, dtype=np.float32)
        self._elapsed_green = 0.0
        self._current_phase = 0
        self._total_wait = 0.0
        self._starvation_count = 0

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._step_count = 0
        self._queues = self.np_random.uniform(0.1, 0.5, size=self.num_lanes).astype(np.float32)
        self._elapsed_green = 0.0
        self._current_phase = self.np_random.integers(0, 4)
        self._total_wait = 0.0
        self._starvation_count = 0

        obs = self._get_obs()
        return obs, {}

    def step(self, action):
        duration_s = float(self.PHASE_DURATIONS[action])
        duration_s = self.constraints.apply(duration_s)

        queue_before = self._queues.copy()

        arrivals = self.np_random.poisson(0.3, size=self.num_lanes).astype(np.float32) * 0.05
        self._queues = np.clip(self._queues + arrivals - 0.1, 0.0, 1.0)

        active_lanes = [self._current_phase, (self._current_phase + 2) % 4]
        for lane in active_lanes:
            served = min(self._queues[lane], duration_s / 600.0)
            self._queues[lane] = max(0.0, self._queues[lane] - served)

        wait = float(np.sum(self._queues * 60.0))
        self._total_wait += wait
        self._starvation_count += sum(1 for q in self._queues if q > 0.8)

        lane_states = [
            {"queue_length": float(q), "wait_time": float(q * 60.0), "vehicles_passed_last_phase": max(0, int((a - q) * 100))}
            for q, a in zip(queue_before, arrivals)
        ]

        reward = compute_reward(lane_states)

        self._current_phase = (self._current_phase + 1) % 4
        self._elapsed_green = 0.0
        self._step_count += 1

        terminated = self._step_count >= self.max_steps
        truncated = False

        if self._starvation_count > 10:
            reward -= 500
            terminated = True

        obs = self._get_obs()
        info = {
            "queues": self._queues.tolist(),
            "phase": self._current_phase,
            "total_wait": self._total_wait,
            "starvation_events": self._starvation_count,
        }

        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        hour_frac = (self._step_count % 3600) / 3600.0
        hour_sin = np.sin(2 * np.pi * hour_frac)
        hour_cos = np.cos(2 * np.pi * hour_frac)
        predicted_density = float(np.mean(self._queues))

        obs = np.concatenate([
            self._queues,
            [self._elapsed_green / self.MAX_GREEN, hour_sin, hour_cos, predicted_density],
        ]).astype(np.float32)

        return np.clip(obs, 0.0, 1.0)

    def render(self):
        if self.render_mode == "human":
            print(f"Step {self._step_count} | Phase {self._current_phase} | Queues: {[f'{q:.2f}' for q in self._queues]}")
