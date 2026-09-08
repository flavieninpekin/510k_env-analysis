"""Masked LSTM policy: sb3-contrib RecurrentActorCriticPolicy + action masking.

``RecurrentPPO`` (unlike ``MaskablePPO``) has no masking mechanism and its
``collect_rollouts`` never passes ``action_masks``. To make a masked recurrent
baseline work with an unmodified ``RecurrentPPO``, we embed the action mask
into the observation via :class:`MaskEmbedWrapper` (obs = state ++ mask), and
:class:`MaskedLstmActorCriticPolicy` extracts the trailing ``mask_dim``
columns and applies them to the action distribution inside both ``forward``
(rollout / prediction) and ``evaluate_actions`` (training).
"""
import numpy as np
import torch as th
import gymnasium as gym
from gymnasium import spaces

from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.maskable.distributions import make_masked_proba_distribution
from sb3_contrib.common.recurrent.type_aliases import RNNStates


class MaskEmbedWrapper(gym.Wrapper):
    """Appends the action mask to the observation so a policy can self-mask."""

    def __init__(self, env):
        super().__init__(env)
        n = int(env.unwrapped.action_space.n)
        state_dim = int(env.unwrapped.observation_space.shape[0])
        self.mask_dim = n
        self.observation_space = spaces.Box(0.0, 1.0, (state_dim + n,), dtype=np.float32)

    def _embed(self, obs, info):
        mask = np.asarray(info["action_mask"], dtype=np.float32)
        return np.concatenate([np.asarray(obs, dtype=np.float32), mask]), info

    def reset(self, seed=None, options=None):
        obs, info = self.env.reset(seed=seed, options=options)
        return self._embed(obs, info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        embedded, info = self._embed(obs, info)
        return embedded, reward, terminated, truncated, info


class MaskedLstmActorCriticPolicy(RecurrentActorCriticPolicy):
    def __init__(self, observation_space, action_space, lr_schedule,
                 mask_dim: int = 300, **kwargs):
        # We must know the mask dim before building the distribution; the base
        # __init__ creates self.action_dist, so we stash it via an attribute
        # that the base uses later.
        self.mask_dim = mask_dim
        super().__init__(observation_space, action_space, lr_schedule, **kwargs)
        # Replace the (unmasked) action distribution with a maskable one.
        self.action_dist = make_masked_proba_distribution(self.action_space)

    def _extract_mask(self, obs: th.Tensor) -> th.Tensor | None:
        if self.mask_dim and obs.shape[-1] >= self.mask_dim:
            return obs[:, -self.mask_dim:]
        return None

    def _get_action_dist_from_latent(self, latent_pi: th.Tensor):
        action_logits = self.action_net(latent_pi)
        return self.action_dist.proba_distribution(action_logits=action_logits)

    def get_distribution(self, obs, lstm_states, episode_starts):
        """Same as the base, but applies the action mask (needed because
        ``RecurrentPPO``'s ``_predict`` calls ``get_distribution``, not ``forward``)."""
        features = self.extract_features(obs, self.pi_features_extractor)
        latent_pi, lstm_states = self._process_sequence(
            features, lstm_states, episode_starts, self.lstm_actor)
        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        distribution = self._get_action_dist_from_latent(latent_pi)
        mask = self._extract_mask(obs)
        if mask is not None:
            distribution.apply_masking(mask)
        return distribution, lstm_states

    def forward(self, obs, lstm_states: RNNStates, episode_starts: th.Tensor,
                deterministic: bool = False):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features

        latent_pi, lstm_states_pi = self._process_sequence(
            pi_features, lstm_states.pi, episode_starts, self.lstm_actor)
        if self.lstm_critic is not None:
            latent_vf, lstm_states_vf = self._process_sequence(
                vf_features, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
            lstm_states_vf = (lstm_states_pi[0].detach(), lstm_states_pi[1].detach())
        else:
            latent_vf = self.critic(vf_features)
            lstm_states_vf = lstm_states_pi

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        values = self.value_net(latent_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)
        mask = self._extract_mask(obs)
        if mask is not None:
            distribution.apply_masking(mask)
        actions = distribution.get_actions(deterministic=deterministic)
        log_prob = distribution.log_prob(actions)
        actions = actions.reshape((-1, *self.action_space.shape))
        return actions, values, log_prob, RNNStates(lstm_states_pi, lstm_states_vf)

    def evaluate_actions(self, obs: th.Tensor, actions: th.Tensor,
                         lstm_states: RNNStates, episode_starts: th.Tensor):
        features = self.extract_features(obs)
        if self.share_features_extractor:
            pi_features = vf_features = features
        else:
            pi_features, vf_features = features

        latent_pi, _ = self._process_sequence(
            pi_features, lstm_states.pi, episode_starts, self.lstm_actor)
        if self.lstm_critic is not None:
            latent_vf, _ = self._process_sequence(
                vf_features, lstm_states.vf, episode_starts, self.lstm_critic)
        elif self.shared_lstm:
            latent_vf = latent_pi.detach()
        else:
            latent_vf = self.critic(vf_features)

        latent_pi = self.mlp_extractor.forward_actor(latent_pi)
        latent_vf = self.mlp_extractor.forward_critic(latent_vf)
        values = self.value_net(latent_vf)

        distribution = self._get_action_dist_from_latent(latent_pi)
        mask = self._extract_mask(obs)
        if mask is not None:
            distribution.apply_masking(mask)
        log_prob = distribution.log_prob(actions)
        entropy = distribution.entropy()
        return values, log_prob, entropy