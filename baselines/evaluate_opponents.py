"""Cross-opponent evaluation: evaluate checkpoints against a different bot.

Training used a fixed opponent (`--train-opponent`, default rule). This script
re-evaluates the *same* checkpoints against another bot (`--eval-opponent`,
default random) and writes a separate results directory, so the capability
profile can be checked for opponent robustness (PLAN.md §2).

Usage::

    python -m baselines.evaluate_opponents --policy mlp --seeds 0 1 2 3 4
    python -m baselines.evaluate_opponents --policy ippo --seeds 0 1 2
"""
import argparse
import json
import os

from .config import PPOConfig, MODES
from .train_ppo import quick_eval, checkpoint_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=MODES)
    ap.add_argument("--policy", default="mlp", choices=["mlp", "lstm", "ippo"])
    ap.add_argument("--train-opponent", default="rule", choices=["random", "rule"])
    ap.add_argument("--eval-opponent", default="random", choices=["random", "rule"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-episodes", type=int, default=100)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--out-dir", default=None,
                    help="default: <out>/opp_<eval-opponent>")
    args = ap.parse_args()

    out_dir = args.out_dir or os.path.join(args.out, f"opp_{args.eval_opponent}")
    os.makedirs(out_dir, exist_ok=True)

    n_done = 0
    for mode in args.modes:
        for seed in args.seeds:
            cfg = PPOConfig(mode=mode, seed=seed, policy=args.policy,
                            opponent=args.train_opponent, out_dir=args.out)
            ckpt = checkpoint_path(cfg)
            if not os.path.exists(ckpt):
                print(f"skip {mode} s{seed} (no checkpoint)")
                continue
            res = quick_eval(cfg, args.n_episodes, eval_opponent=args.eval_opponent)
            res.update({
                "seed": seed, "mode": mode, "policy": args.policy,
                "opponent": args.eval_opponent,
                "train_opponent": args.train_opponent,
                "checkpoint": os.path.basename(ckpt),
            })
            path = os.path.join(out_dir, f"{cfg.tag()}_{mode}_s{seed}.eval.json")
            with open(path, "w") as f:
                json.dump(res, f, indent=2)
            n_done += 1
            print(f"{mode:8} s{seed}: win={res['win_rate']:.3f} "
                  f"rew={res['mean_reward']:.1f} len={res['mean_len']:.1f}")
    print(f"done: {n_done} evals -> {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
