"""MAPPO (centralized critic, self-play) training on 510K. Chunked + resumable.

Same conventions as :mod:`baselines.train_ppo` so results sit next to the other
baselines: one chunk per invocation, checkpoint resumes, a ``.done`` marker
finishes, and each chunk writes an eval JSON plus a history JSONL.

Usage::

    python -m baselines.train_mappo --mode dynamic --seed 0 --total 1000000 --chunk 50000
    python -m baselines.train_mappo --mode dynamic --seed 0 --eval-only
"""
import argparse
import json
import os
import sys

import numpy as np
from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker

from .mappo_env import MAPPOEnv
from .mappo_policy import MAPPOCentralizedCriticPolicy


def mask_fn(env):
    return env.unwrapped._get_action_mask()


def make_env(mode: str, selfplay_model=None):
    env = ActionMasker(MAPPOEnv(mode=mode, num_players=4), mask_fn)
    if selfplay_model is not None:
        env.unwrapped.set_policy_bot(selfplay_model)
    return env


def make_model(mode: str, seed: int, env):
    model = MaskablePPO(
        MAPPOCentralizedCriticPolicy,
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        policy_kwargs=dict(net_arch=[]),
        seed=seed,
        verbose=0,
    )
    env.unwrapped.set_policy_bot(model)
    return model


def checkpoint_path(mode: str, seed: int, out: str) -> str:
    return os.path.join(out, f"mappo_{mode}_s{seed}.zip")


def done_path(mode: str, seed: int, out: str) -> str:
    return os.path.join(out, f"mappo_{mode}_s{seed}.done")


def quick_eval(model, mode: str, n_episodes: int = 100) -> dict:
    env = make_env(mode)          # fixed rule-bot opponents, no self-play
    rewards, lengths, wins, illegal = [], [], 0, 0
    for ep in range(n_episodes):
        obs, info = env.reset(seed=1000 + ep)
        done, tot, steps = False, 0.0, 0
        while not done:
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
    ap.add_argument("--mode", default="dynamic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--total", type=int, default=1_000_000)
    ap.add_argument("--chunk", type=int, default=50_000)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--eval-episodes", type=int, default=100)
    ap.add_argument("--eval-only", action="store_true")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    ckpt = checkpoint_path(args.mode, args.seed, args.out)
    done = done_path(args.mode, args.seed, args.out)

    if args.eval_only:
        if not os.path.exists(ckpt):
            print("{}")
            return 0
        env_tmp = make_env(args.mode)
        model = MaskablePPO.load(ckpt, env=env_tmp)
        print(json.dumps(quick_eval(model, args.mode, args.eval_episodes), indent=2))
        return 0

    if os.path.exists(done):
        print(f"already done: {done}", flush=True)
    else:
        if os.path.exists(ckpt):
            env = make_env(args.mode)
            model = MaskablePPO.load(ckpt, env=env)
            env.unwrapped.set_policy_bot(model)
            n_done = int(model.num_timesteps)
        else:
            env = make_env(args.mode)
            model = make_model(args.mode, args.seed, env)
            n_done = 0

        steps = min(args.chunk, args.total - n_done)
        model.learn(total_timesteps=steps, reset_num_timesteps=False)
        model.save(ckpt)
        n_done = int(model.num_timesteps)
        if n_done >= args.total:
            with open(done, "w") as f:
                f.write("done\n")
            print(f"[{args.mode} s{args.seed}] FINISHED {n_done} steps", flush=True)
        else:
            print(f"[{args.mode} s{args.seed}] checkpoint at {n_done} steps", flush=True)

    env_eval = make_env(args.mode)
    model = MaskablePPO.load(ckpt, env=env_eval)
    res = quick_eval(model, args.mode, args.eval_episodes)
    n_steps = int(model.num_timesteps)
    with open(os.path.join(args.out, f"mappo_{args.mode}_s{args.seed}.eval.json"), "w") as f:
        json.dump({"seed": args.seed, "mode": args.mode, "opponent": "rule",
                   "policy": "mappo", "num_timesteps": n_steps, **res}, f, indent=2)
    with open(os.path.join(args.out, f"mappo_{args.mode}_s{args.seed}.history.jsonl"), "a") as f:
        f.write(json.dumps({"num_timesteps": n_steps,
                            "win_rate": res.get("win_rate"),
                            "mean_reward": res.get("mean_reward")}) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
