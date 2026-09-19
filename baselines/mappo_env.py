"""MAPPO wrapper env: Dict observation {local, global} for a centralized critic.

Ported from the sibling IIGC repo (``AAAI2027-510k-clear/src/env/mappo_env.py``)
with imports adapted to the installed ``env_510k`` package.
"""
import numpy as np
from gymnasium import spaces, Wrapper

from env_510k.env import FiveTenKEnv
from env_510k.obs_utils import obs_for_player


class MAPPOEnv(Wrapper):
    """Wraps FiveTenKEnv to provide a Dict observation.

    local:  agent's own observation (112-dim, +4 in OBVIOUS)
    global: all four players' observations concatenated (4x local)
    """

    def __init__(self, mode: str = "single", num_players: int = 4):
        base_env = FiveTenKEnv(mode=mode, num_players=num_players)
        super().__init__(base_env)
        local_dim = int(base_env.observation_space.shape[0])
        self.observation_space = spaces.Dict({
            "local": spaces.Box(0, 1, (local_dim,), np.float32),
            "global": spaces.Box(0, 1, (local_dim * 4,), np.float32),
        })
        self.action_space = base_env.action_space

    def _dict_obs(self):
        game = self.env.game
        if game is None:
            local = np.zeros(self.env.observation_space.shape[0], np.float32)
            global_ = np.zeros(self.env.observation_space.shape[0] * 4, np.float32)
        else:
            local = obs_for_player(game, self.env.agent_id).astype(np.float32)
            global_ = np.concatenate(
                [obs_for_player(game, i) for i in range(game.num_players)]
            ).astype(np.float32)
        return {"local": local, "global": global_}

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return self._dict_obs(), info

    def step(self, action):
        obs, reward, done, truncated, info = self.env.step(action)
        return self._dict_obs(), reward, done, truncated, info

    def _get_action_mask(self):
        return self.env._get_action_mask()

    def _get_info(self):
        return self.env._get_info()

    def set_policy_bot(self, model):
        self.env.set_policy_bot(model)

    def set_rule_bot(self):
        self.env.set_rule_bot()
