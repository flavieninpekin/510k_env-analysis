"""Aggregate baseline eval results into a paper-ready table.

Reads ``<out>/*.eval.json`` files and prints a markdown/JSON table of
mean ± std over seeds for each (mode, policy, opponent) cell.
"""
import argparse
import glob
import json
import os
import re
from collections import defaultdict

import numpy as np


def collect(out_dir: str) -> dict:
    cells = defaultdict(list)
    for path in sorted(glob.glob(os.path.join(out_dir, "*.eval.json"))):
        with open(path) as f:
            d = json.load(f)
        reveal = d.get("reveal")
        if reveal is None:
            # older runs predate the ``reveal`` field: recover it from the tag
            m = re.search(r"_r([0-9.]+)_", os.path.basename(path))
            if m:
                reveal = float(m.group(1))
        key = (d["mode"], d["policy"], d["opponent"], reveal)
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
            for opp in ["rule", "random"]:
                for reveal in sorted(
                    {k[3] for k in cells if k[0] == mode and k[1] == pol and k[2] == opp},
                    key=lambda r: (r is not None, r),
                ):
                    key = (mode, pol, opp, reveal)
                    if key not in cells:
                        continue
                    d = cells[key]
                    seeds = [x["seed"] for x in d]
                    wr = np.array([x["win_rate"] for x in d])
                    rw = np.array([x["mean_reward"] for x in d])
                    il = np.array([x["illegal_action_rate"] for x in d])
                    rows.append({
                        "mode": mode, "policy": pol, "opponent": opp,
                        "reveal": reveal,
                        "n_seeds": len(d), "seeds": seeds,
                        "win_rate_mean": float(wr.mean()), "win_rate_std": float(wr.std()),
                        "reward_mean": float(rw.mean()), "reward_std": float(rw.std()),
                        "illegal_mean": float(il.mean()),
                    })

    if args.json:
        print(json.dumps(rows, indent=2))
        return

    print("| mode | policy | opponent | reveal | seeds | win_rate | mean_reward | illegal |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        rev = "—" if r["reveal"] is None else f"{r['reveal']:g}"
        print(f"| {r['mode']} | {r['policy']} | {r['opponent']} | {rev} | {r['n_seeds']} "
              f"| {fmt(r['win_rate_mean'], r['win_rate_std'])} "
              f"| {r['reward_mean']:.1f}±{r['reward_std']:.1f} "
              f"| {r['illegal_mean']:.3f} |")


if __name__ == "__main__":
    main()