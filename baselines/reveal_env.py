"""Information-revelation wrapper: reveal the hidden team with probability p.

Wraps a DYNAMIC-mode :class:`~env_510k.env.FiveTenKEnv`. On every observation
the team bit-vector (teammates' identities) is included with probability
``reveal_prob``; otherwise it is all-zero. p=0 corresponds to DYNAMIC (fully
hidden), p=1 to OBVIOUS (fully revealed), and intermediate p gives a graded
information dial — the axis used for the information-revelation ablation
(§6.3) and the connection to IIGC.
"""
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces


class RevealEnv(gym.Wrapper):
    """Adds a 4-dim teammate-mask to the observation with probability ``p``."""

    def __init__(self, env, reveal_prob: float = 0.5):
        super().__init__(env)
        self.reveal_prob = float(np.clip(reveal_prob, 0.0, 1.0))
        state_dim = int(env.observation_space.shape[0])
        self.observation_space = spaces.Box(0.0, 1.0, (state_dim + 4,), dtype=np.float32)

    def _agent_id(self) -> int:
        unwrapped = self.env.unwrapped
        return getattr(unwrapped, "agent_id", 0)

    def _team_bits(self) -> np.ndarray:
        bits = np.zeros(4, dtype=np.float32)
        game = self.env.unwrapped.game
        if game is not None and game.red_a_team is not None \
                and np.random.random() < self.reveal_prob:
            me = self._agent_id()
            for i in range(4):
                if i != me and i in game.red_a_team:
                    bits[i] = 1.0
        return bits

    def _wrap(self, obs, info):
        obs = np.asarray(obs, dtype=np.float32)
        return np.concatenate([obs, self._team_bits()]), info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._wrap(obs, info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        wrapped, info = self._wrap(obs, info)
        return wrapped, reward, terminated, truncated, info