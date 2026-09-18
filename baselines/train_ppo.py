"""Train a (masked) PPO baseline on 510K. Resumable + chunked.

Policies:
  - ``--policy mlp`` : ``MaskablePPO`` (sb3-contrib)
  - ``--policy lstm``: ``RecurrentPPO`` + :class:`MaskedLstmActorCriticPolicy`

Usage::

    python -m baselines.train_ppo --mode dynamic --seed 42 --total 1000000
    python -m baselines.train_ppo --mode dynamic --seed 42 --chunk 100000   # resumes
"""
import argparse
import json
import os
import sys
from dataclasses import replace
from typing import Optional

import numpy as np
from sb3_contrib import MaskablePPO, RecurrentPPO
from sb3_contrib.common.wrappers import ActionMasker

from env_510k import FiveTenKEnv
from .config import PPOConfig
from .masked_lstm_policy import MaskedLstmActorCriticPolicy, MaskEmbedWrapper
from .selfplay_env import SelfPlayEnv
from .reveal_env import RevealEnv


def mask_fn(env):
    return env.unwrapped._get_action_mask()


def maybe_reveal(cfg: PPOConfig, env):
    if cfg.reveal is not None:
        return RevealEnv(env, reveal_prob=cfg.reveal)
    return env


def make_env(cfg: PPOConfig):
    if cfg.policy == "ippo":
        env = SelfPlayEnv(mode=cfg.mode, num_players=4)
        return maybe_reveal(cfg, ActionMasker(env, mask_fn))
    env = FiveTenKEnv(mode=cfg.mode, num_players=4)
    if cfg.opponent == "rule":
        env.set_rule_bot()
    if is_lstm(cfg):
        return maybe_reveal(cfg, MaskEmbedWrapper(env))
    return maybe_reveal(cfg, ActionMasker(env, mask_fn))


def is_lstm(cfg: PPOConfig) -> bool:
    return cfg.policy == "lstm"


def make_model(cfg: PPOConfig, env):
    if is_lstm(cfg):
        return RecurrentPPO(
            MaskedLstmActorCriticPolicy,
            env,
            learning_rate=cfg.learning_rate,
            n_steps=cfg.n_steps,
            batch_size=cfg.batch_size,
            n_epochs=cfg.n_epochs,
            gamma=cfg.gamma,
            gae_lambda=cfg.gae_lambda,
            clip_range=cfg.clip_range,
            ent_coef=cfg.ent_coef,
            vf_coef=cfg.vf_coef,
            policy_kwargs=dict(
                net_arch=dict(pi=list(cfg.net_arch), vf=list(cfg.net_arch)),
                lstm_hidden_size=cfg.lstm_hidden_size,
                enable_critic_lstm=True,
                mask_dim=300,
            ),
            seed=cfg.seed,
            verbose=0,
        )
    return MaskablePPO(
        "MlpPolicy",
        env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        clip_range=cfg.clip_range,
        ent_coef=cfg.ent_coef,
        vf_coef=cfg.vf_coef,
        policy_kwargs=dict(net_arch=dict(pi=list(cfg.net_arch), vf=list(cfg.net_arch))),
        seed=cfg.seed,
        verbose=0,
    )


def load_model(cfg: PPOConfig, ckpt: str):
    env = make_env(cfg)
    if is_lstm(cfg):
        return RecurrentPPO.load(ckpt, env=env)
    return MaskablePPO.load(ckpt, env=env)


def checkpoint_path(cfg: PPOConfig) -> str:
    return os.path.join(cfg.out_dir, f"{cfg.tag()}_{cfg.mode}_s{cfg.seed}.zip")


def done_path(cfg: PPOConfig) -> str:
    return os.path.join(cfg.out_dir, f"{cfg.tag()}_{cfg.mode}_s{cfg.seed}.done")


def train_chunk(cfg: PPOConfig):
    os.makedirs(cfg.out_dir, exist_ok=True)
    ckpt = checkpoint_path(cfg)
    done = done_path(cfg)

    if os.path.exists(done):
        print(f"already done: {done}", flush=True)
        return 0

    if os.path.exists(ckpt):
        model = load_model(cfg, ckpt)
        n_done = int(model.num_timesteps)
    else:
        model = make_model(cfg, make_env(cfg))
        n_done = 0

    steps = min(cfg.chunk_size, cfg.total_timesteps - n_done)
    if steps <= 0:
        with open(done, "w") as f:
            f.write("done\n")
        print(f"already at total steps ({n_done}); marking done", flush=True)
        return 0

    print(f"[{cfg.mode} s{cfg.seed}] {cfg.tag()} training {steps} steps "
          f"(at {n_done}/{cfg.total_timesteps})", flush=True)
    model.learn(total_timesteps=steps, reset_num_timesteps=False)
    model.save(ckpt)

    n_done = int(model.num_timesteps)
    if n_done >= cfg.total_timesteps:
        with open(done, "w") as f:
            f.write("done\n")
        print(f"[{cfg.mode} s{cfg.seed}] FINISHED {n_done} steps", flush=True)
    else:
        print(f"[{cfg.mode} s{cfg.seed}] checkpoint at {n_done} steps", flush=True)
    return 0


def make_eval_env(cfg: PPOConfig):
    """Evaluation env: the trained policy always plays player 0 vs fixed bots."""
    env = FiveTenKEnv(mode=cfg.mode, num_players=4)
    if cfg.opponent == "rule":
        env.set_rule_bot()
    if is_lstm(cfg):
        return maybe_reveal(cfg, MaskEmbedWrapper(env))
    return maybe_reveal(cfg, ActionMasker(env, mask_fn))


def quick_eval(cfg: PPOConfig, n_episodes: int = 100,
               eval_opponent: Optional[str] = None) -> dict:
    ckpt = checkpoint_path(cfg)
    if not os.path.exists(ckpt):
        return {}
    model = load_model(cfg, ckpt)
    env_cfg = replace(cfg, opponent=eval_opponent) if eval_opponent else cfg
    env = make_eval_env(env_cfg)  # policy always controls P0

    rewards, lengths, wins, illegal = [], [], 0, 0
    for ep in range(n_episodes):
        obs, info = env.reset(seed=1000 + ep)
        done, tot, steps = False, 0.0, 0
        state, ep_start = None, np.array([True])
        while not done:
            if is_lstm(cfg):
                a, state = model.predict(obs, state=state, episode_start=ep_start,
                                         deterministic=True)
                ep_start = np.array([False])
            else:
                a, _ = model.predict(obs, action_masks=info["action_mask"], deterministic=True)
            if info["action_mask"][int(a)] == 0:
                illegal += 1
            obs, r, done, tr, info = env.step(int(a))
            tot += r
            steps += 1
        rewards.append(tot)
        lengths.append(steps)
        if tot > 0:
            wins += 1
    n_actions = sum(lengths)
    return {
        "win_rate": wins / n_episodes,
        "mean_reward": float(np.mean(rewards)),
        "std_reward": float(np.std(rewards)),
        "mean_len": float(np.mean(lengths)),
        "illegal_action_rate": illegal / n_actions if n_actions else 0.0,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="single")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", default="mlp", choices=["mlp", "lstm", "ippo"])
    ap.add_argument("--opponent", default="rule", choices=["random", "rule"])
    ap.add_argument("--total", type=int, default=1_000_000)
    ap.add_argument("--chunk", type=int, default=100_000)
    ap.add_argument("--n-steps", type=int, default=2048)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--n-epochs", type=int, default=4)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--reveal", type=float, default=None,
                    help="info-reveal probability for DYNAMIC mode (0..1); None=off")
    ap.add_argument("--ent-coef", type=float, default=0.01)
    ap.add_argument("--net-arch", nargs="+", type=int, default=[256, 256])
    ap.add_argument("--lstm-hidden", type=int, default=256)
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--eval-episodes", type=int, default=100)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    cfg = PPOConfig(
        mode=args.mode, seed=args.seed, policy=args.policy, opponent=args.opponent,
        total_timesteps=args.total, chunk_size=args.chunk, n_steps=args.n_steps,
        batch_size=args.batch_size, n_epochs=args.n_epochs, learning_rate=args.lr,
        ent_coef=args.ent_coef, net_arch=tuple(args.net_arch),
        lstm_hidden_size=args.lstm_hidden, tag_suffix=args.tag_suffix,
        out_dir=args.out, eval_episodes=args.eval_episodes, reveal=args.reveal,
    )

    if args.eval_only:
        print(json.dumps(quick_eval(cfg, cfg.eval_episodes), indent=2))
        return 0

    rc = train_chunk(cfg)
    res = quick_eval(cfg, cfg.eval_episodes)
    n_steps = 0
    if os.path.exists(checkpoint_path(cfg)):
        n_steps = int(load_model(cfg, checkpoint_path(cfg)).num_timesteps)
    eval_path = os.path.join(cfg.out_dir, f"{cfg.tag()}_{cfg.mode}_s{cfg.seed}.eval.json")
    with open(eval_path, "w") as f:
        json.dump({"seed": cfg.seed, "mode": cfg.mode, "opponent": cfg.opponent,
                   "policy": cfg.policy, "reveal": cfg.reveal,
                   "num_timesteps": n_steps, **res}, f, indent=2)
    # append to learning-curve history
    hist_path = os.path.join(cfg.out_dir, f"{cfg.tag()}_{cfg.mode}_s{cfg.seed}.history.jsonl")
    with open(hist_path, "a") as f:
        f.write(json.dumps({"num_timesteps": n_steps,
                            "win_rate": res.get("win_rate"),
                            "mean_reward": res.get("mean_reward")}) + "\n")
    return rc


if __name__ == "__main__":
    sys.exit(main())