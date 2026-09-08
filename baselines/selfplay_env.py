"""Self-play env: one shared policy controls all seats (IPPO w/ parameter sharing).

The raw :class:`~env_510k.game.Game` is driven directly: on every ``step`` the
policy acts for whichever player is currently active, so a single shared
policy learns the full game from all positions. The observation is the active
player's own observation; the reward is terminal-only (deferred credit), as in
the single-agent environment.
"""
from typing import Optional

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from env_510k.card import card_to_id
from env_510k.game import Game, GameMode
from env_510k.obs_utils import obs_for_player, action_mask_for_player
from env_510k.env import MAX_ACTIONS


class SelfPlayEnv(gym.Env):
    """Gymnasium env where the 'agent' is whichever player is on turn."""

    metadata = {"render_modes": ["ansi"], "render_fps": 10}

    def __init__(self, mode: str = "single", num_players: int = 4,
                 render_mode: Optional[str] = None):
        super().__init__()
        self._mode = mode
        self.num_players = num_players
        self.render_mode = render_mode
        self.include_jokers = num_players == 3
        self.n_cards = 54 if self.include_jokers else 52

        obs_dim = self.n_cards * 2 + 1 + 4 + 1 + 1 + 1
        if mode == "obvious":
            obs_dim += 4
        self.observation_space = spaces.Box(0, 1, (obs_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(MAX_ACTIONS)

        self.game: Optional[Game] = None

    def _game_mode(self) -> GameMode:
        return GameMode("obvious" if self._mode == "obvious" else self._mode)

    def _active_obs(self) -> np.ndarray:
        return obs_for_player(self.game, self.game.current_player).astype(np.float32)

    def _get_action_mask(self) -> np.ndarray:
        if self.game is None:
            mask = np.zeros(MAX_ACTIONS, dtype=np.int64)
            mask[0] = 1
            return mask
        return action_mask_for_player(self.game, self.game.current_player)

    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        import random as _random
        if seed is not None:
            _random.seed(seed)
            np.random.seed(seed)
        self.game = Game(mode=self._game_mode(), num_players=self.num_players,
                         include_jokers=self.include_jokers)
        return self._active_obs(), {"action_mask": self._get_action_mask()}

    def step(self, action: int):
        if self.game is None or self.game.is_over:
            return self._active_obs(), 0.0, True, False, {"action_mask": self._get_action_mask()}
        pid = self.game.current_player
        patterns = self.game.get_valid_actions(pid)
        taken = False
        if action == 0 and self.game.can_pass(pid):
            taken = self.game.pass_turn(pid)
        else:
            idx = action - 1
            if 0 <= idx < len(patterns):
                taken = self.game.play_cards(pid, patterns[idx].cards)
        if not taken:
            if patterns:
                import random as _random
                self.game.play_cards(pid, _random.choice(patterns).cards)
            elif self.game.can_pass(pid):
                self.game.pass_turn(pid)
        done = self.game.is_over
        from env_510k.scorer import Scorer
        reward = Scorer(self.game).compute_rewards().get(pid, 0.0) if done else 0.0
        return (self._active_obs(), reward, done, False,
                {"action_mask": self._get_action_mask()})

    def render(self):
        if not self.game:
            return "Game not started"
        lines = [f"Turn: P{self.game.current_player}"]
        for i, p in enumerate(self.game.players):
            lines.append(f"P{i}: {len(p.hand)} cards")
        return "\n".join(lines)