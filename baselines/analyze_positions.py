"""Per-agent signal metrics beyond coarse team win rate (PLAN.md §5-6).

Evaluates a trained checkpoint as player 0 against fixed bots and reports, per
episode: win, whether the agent finished, its finish position, its own 510K
score, the teammate's score, and the own-share of score variance (the reward
dilution of `notes/theory.md` Prop D). For DYNAMIC/OBVIOUS the results are also
split by red-team size (2v2 vs 1v3).

Usage::

    python -m baselines.analyze_positions --mode dynamic --seed 0 --policy mlp --out runs
"""
import argparse
import json
import os

import numpy as np

from env_510k import FiveTenKEnv
from .config import PPOConfig
from .train_ppo import load_model, checkpoint_path, is_lstm, make_eval_env


def _teammates(cfg: PPOConfig, game) -> set:
    if game.red_a_team is not None:
        return set(game.red_a_team)
    if cfg.mode == "static":
        return {0, 2}
    return {0}


def _bucket_stats(rows: list) -> dict:
    if not rows:
        return {}
    wins = np.array([r["win"] for r in rows], dtype=float)
    finished = np.array([r["finish_pos"] is not None for r in rows], dtype=float)
    pos = np.array([r["finish_pos"] for r in rows if r["finish_pos"] is not None],
                   dtype=float)
    own = np.array([r["own_score"] for r in rows], dtype=float)
    mate = np.array([r["teammate_score"] for r in rows], dtype=float)
    rew = np.array([r["reward"] for r in rows], dtype=float)
    out = {
        "n": len(rows),
        "win_rate": float(wins.mean()),
        "finish_rate": float(finished.mean()),
        "mean_finish_pos": float(pos.mean()) if pos.size else None,
        "mean_reward": float(rew.mean()),
        "std_reward": float(rew.std()),
        "mean_own_score": float(own.mean()),
        "mean_teammate_score": float(mate.mean()),
    }
    if own.std() + mate.std() > 0:
        out["own_std_share"] = float(own.std() / (own.std() + mate.std()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="dynamic")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--policy", default="mlp", choices=["mlp", "lstm"])
    ap.add_argument("--opponent", default="rule", choices=["random", "rule"])
    ap.add_argument("--reveal", type=float, default=None)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--n-episodes", type=int, default=300)
    args = ap.parse_args()

    cfg = PPOConfig(mode=args.mode, seed=args.seed, policy=args.policy,
                    opponent=args.opponent, out_dir=args.out, reveal=args.reveal)
    ckpt = checkpoint_path(cfg)
    if not os.path.exists(ckpt):
        raise SystemExit(f"missing checkpoint: {ckpt}")
    model = load_model(cfg, ckpt)
    env = make_eval_env(cfg)  # policy controls player 0

    buckets = {"overall": []}
    if args.mode in ("dynamic", "obvious"):
        buckets["2v2"] = []
        buckets["1v3"] = []

    for ep in range(args.n_episodes):
        obs, info = env.reset(seed=5000 + ep)
        done, tot = False, 0.0
        state, ep_start = None, np.array([True])
        while not done:
            if is_lstm(cfg):
                a, state = model.predict(obs, state=state, episode_start=ep_start,
                                         deterministic=True)
                ep_start = np.array([False])
            else:
                a, _ = model.predict(obs, action_masks=info["action_mask"],
                                     deterministic=True)
            obs, r, done, _, info = env.step(int(a))
            tot += r

        game = env.unwrapped.game
        teammates = _teammates(cfg, game) - {0}
        teammate_score = sum(float(game.player_510k_scores[i]) for i in teammates)
        finish_pos = (game.finish_order.index(0) + 1) if 0 in game.finish_order else None
        row = {
            "win": tot > 0,
            "reward": float(tot),
            "finish_pos": finish_pos,
            "own_score": float(game.player_510k_scores[0]),
            "teammate_score": teammate_score,
        }
        buckets["overall"].append(row)
        if game.red_a_team is not None:
            key = "2v2" if len(game.red_a_team) == 2 else "1v3"
            if key in buckets:
                buckets[key].append(row)

    res = {b: _bucket_stats(rows) for b, rows in buckets.items() if rows}
    out_path = os.path.join(args.out, f"{cfg.tag()}_{cfg.mode}_s{cfg.seed}.positions.json")
    with open(out_path, "w") as f:
        json.dump(res, f, indent=2)
    for b, s in res.items():
        print(f"{b}: n={s['n']} win={s['win_rate']:.3f} finish={s['finish_rate']:.3f} "
              f"pos={s['mean_finish_pos']} own={s['mean_own_score']:.1f} "
              f"mate={s['mean_teammate_score']:.1f} "
              f"own_share={s.get('own_std_share', float('nan')):.3f}")
    print(f"saved {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
