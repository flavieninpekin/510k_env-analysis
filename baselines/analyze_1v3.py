"""Evaluate a DYNAMIC-mode checkpoint, splitting results by red-team size (2v2 vs 1v3/3v1).

This verifies that the "hidden cooperation" results do not depend on the deal's
team composition (76% 2v2 / 24% 1v3 in the full deal distribution).
"""
import argparse
import json
import sys

import numpy as np

from env_510k import FiveTenKEnv
from .config import PPOConfig
from .train_ppo import load_model, checkpoint_path, is_lstm
from .masked_lstm_policy import MaskEmbedWrapper
from sb3_contrib.common.wrappers import ActionMasker


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="dynamic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", default="mlp", choices=["mlp", "lstm", "ippo"])
    ap.add_argument("--out", default="runs")
    ap.add_argument("--n-episodes", type=int, default=300)
    args = ap.parse_args()

    cfg = PPOConfig(mode=args.mode, seed=args.seed, policy=args.policy, out_dir=args.out)
    ckpt = checkpoint_path(cfg)
    model = load_model(cfg, ckpt)

    env = FiveTenKEnv(mode=args.mode, num_players=4)
    env.set_rule_bot()
    if is_lstm(cfg):
        env = MaskEmbedWrapper(env)

    buckets = {"2v2": [], "1v3": []}
    for ep in range(args.n_episodes):
        obs, info = env.reset(seed=5000 + ep)
        game = env.unwrapped.game
        if game.red_a_team is None:
            continue
        n_red = len(game.red_a_team)
        bucket = "2v2" if n_red == 2 else "1v3"
        done, tot = False, 0.0
        state, ep_start = None, np.array([True])
        while not done:
            if is_lstm(cfg):
                a, state = model.predict(obs, state=state, episode_start=ep_start,
                                         deterministic=True)
                ep_start = np.array([False])
            else:
                a, _ = model.predict(obs, action_masks=info["action_mask"], deterministic=True)
            obs, r, done, tr, info = env.step(int(a))
            tot += r
        buckets[bucket].append(tot)

    out = {}
    for b, rs in buckets.items():
        if rs:
            out[b] = {
                "n": len(rs),
                "win_rate": float(np.mean(np.array(rs) > 0)),
                "mean_reward": float(np.mean(rs)),
                "std_reward": float(np.std(rs)),
            }
            print(f"{b}: n={len(rs)} win_rate={out[b]['win_rate']:.3f} "
                  f"reward={out[b]['mean_reward']:.1f}±{out[b]['std_reward']:.1f}")
    print(json.dumps(out))
    return 0


if __name__ == "__main__":
    sys.exit(main())