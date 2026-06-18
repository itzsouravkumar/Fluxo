"""Tests for FLUXO RL environment."""

import pytest

gymnasium = pytest.importorskip("gymnasium", reason="gymnasium not installed")


def test_env_init():
    from core.rl.env import FluxoSignalEnv
    env = FluxoSignalEnv()
    assert env.observation_space.shape == (8,)
    assert env.action_space.n == 9


def test_env_reset():
    from core.rl.env import FluxoSignalEnv
    env = FluxoSignalEnv()
    obs, info = env.reset()
    assert obs.shape == (8,)
    assert isinstance(info, dict)


def test_env_step():
    from core.rl.env import FluxoSignalEnv
    env = FluxoSignalEnv()
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)
    assert obs.shape == (8,)
    assert isinstance(reward, float)
