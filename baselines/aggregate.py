"""Aggregate baseline eval results into a paper-ready table.

Reads ``<out>/*.eval.json`` files and prints a markdown/JSON table of
mean ± std over seeds for each (mode, policy, opponent) cell.
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np


def collect(out_dir: str) -> dict:
    cells = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(out_dir, "*.eval.json"))):
        with open(path) as f:
            d = json.load(f)
        key = (d["mode"], d["policy"], d["opponent"])
        cells[key].append(d)
    return cells


def fmt(x, s):
    return f"{x:.3f}±{s:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args()

    cells = collect(args.out)
    modes = ["single", "static", "dynamic", "obvious"]
    policies = ["mlp", "lstm", "ippo"]

    rows = []
    for mode in modes:
        for pol in policies:
            for opp in ["rule"]:
                key = (mode, pol, opp)
                if key not in cells:
                    continue
                d = cells[key]
                seeds = [x["seed"] for x in d]
                wr = np.array([x["win_rate"] for x in d])
                rw = np.array([x["mean_reward"] for x in d])
                il = np.array([x["illegal_action_rate"] for x in d])
                rows.append({
                    "mode": mode, "policy": pol, "opponent": opp,
                    "n_seeds": len(d), "seeds": seeds,
                    "win_rate_mean": float(wr.mean()), "win_rate_std": float(wr.std()),
                    "reward_mean": float(rw.mean()), "reward_std": float(rw.std()),
                    "illegal_mean": float(il.mean()),
                })

    if args.json:
        print(json.dumps(rows, indent=2))
        return

    print("| mode | policy | opponent | seeds | win_rate | mean_reward | illegal |")
    print("|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['mode']} | {r['policy']} | {r['opponent']} | {r['n_seeds']} "
              f"| {fmt(r['win_rate_mean'], r['win_rate_std'])} "
              f"| {r['reward_mean']:.1f}±{r['reward_std']:.1f} "
              f"| {r['illegal_mean']:.3f} |")


if __name__ == "__main__":
    main()