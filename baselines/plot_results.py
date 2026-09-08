"""Generate paper figures from collected results.

- ``--heatmap``: capability profile (mode x policy win rate)
- ``--reveal``: information-revelation curve (win rate vs reveal p)
- ``--curves``: learning curves from ``*.history.jsonl``
"""
import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def load_eval(out_dir):
    cells = {}
    for path in sorted(glob.glob(os.path.join(out_dir, "*.eval.json"))):
        with open(path) as f:
            d = json.load(f)
        cells[(d["mode"], d["policy"], d.get("reveal"))] = d
    return cells


def heatmap(out_dir, save):
    cells = load_eval(out_dir)
    modes = ["single", "static", "dynamic", "obvious"]
    policies = ["mlp", "lstm", "ippo"]
    W = np.full((len(modes), len(policies)), np.nan)
    for (m, p, r), d in cells.items():
        if r is not None:
            continue
        if m in modes and p in policies:
            i, j = modes.index(m), policies.index(p)
            W[i, j] = d["win_rate"]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    im = ax.imshow(W, cmap="RdYlGn", vmin=0.2, vmax=0.9, aspect="auto")
    ax.set_xticks(range(len(policies))); ax.set_xticklabels(["MLP", "LSTM", "IPPO"])
    ax.set_yticks(range(len(modes))); ax.set_yticklabels([m.upper() for m in modes])
    for i in range(len(modes)):
        for j in range(len(policies)):
            v = W[i, j]
            ax.text(j, i, f"{v:.3f}" if not np.isnan(v) else "—", ha="center", va="center",
                    color="black", fontsize=11)
    ax.set_title("Win rate vs rule bot (player 0)")
    fig.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(save, dpi=200)
    print(f"saved {save}")


def reveal_curve(out_dir, save):
    cells = load_eval(out_dir)
    pts = []
    for path in sorted(glob.glob(os.path.join(out_dir, "*.eval.json"))):
        name = os.path.basename(path)
        with open(path) as f:
            d = json.load(f)
        if d.get("policy") != "mlp":
            continue
        if d.get("mode") == "obvious":
            pts.append((1.0, d["win_rate"]))
        elif d.get("mode") == "dynamic":
            # reveal comes from the tag in the filename (older runs predate the
            # ``reveal`` field in the json), else default to p=0 (plain DYNAMIC)
            r = d.get("reveal")
            if r is None:
                import re
                m = re.search(r"_r([0-9.]+)_", name)
                r = float(m.group(1)) if m else 0.0
            pts.append((r, d["win_rate"]))
    xs = sorted(set(x for x, _ in pts))
    means, stds = [], []
    for x in xs:
        ys = [y for xx, y in pts if abs(xx - x) < 1e-9]
        means.append(np.mean(ys)); stds.append(np.std(ys))
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.errorbar(xs, means, yerr=stds, fmt="-o", capsize=4)
    ax.set_xlabel("Team-reveal probability p")
    ax.set_ylabel("Win rate (vs rule bot)")
    ax.set_title("Information-revelation ablation (MLP PPO, DYNAMIC)")
    ax.set_ylim(0.7, 0.9)
    fig.tight_layout()
    fig.savefig(save, dpi=200)
    print(f"saved {save}")


def curves(out_dir, save):
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for path in sorted(glob.glob(os.path.join(out_dir, "*.history.jsonl"))):
        name = os.path.basename(path).replace("ppo_mlp_rule_", "").replace(".history.jsonl", "")
        steps, wins = [], []
        for line in open(path):
            d = json.loads(line)
            steps.append(d["num_timesteps"]); wins.append(d["win_rate"])
        if steps:
            ax.plot(steps, wins, marker="o", ms=3, label=name)
    ax.set_xlabel("Timesteps"); ax.set_ylabel("Win rate")
    ax.set_title("Learning curves")
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(save, dpi=200)
    print(f"saved {save}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs")
    ap.add_argument("--figdir", default="paper/figs")
    ap.add_argument("--heatmap", action="store_true")
    ap.add_argument("--reveal", action="store_true")
    ap.add_argument("--curves", action="store_true")
    ap.add_argument("--all", action="store_true")
    args = ap.parse_args()
    os.makedirs(args.figdir, exist_ok=True)
    if args.all or args.heatmap:
        heatmap(args.out, os.path.join(args.figdir, "capability_heatmap.png"))
    if args.all or args.reveal:
        reveal_curve(args.out, os.path.join(args.figdir, "reveal_curve.png"))
    if args.all or args.curves:
        curves(args.out, os.path.join(args.figdir, "learning_curves.png"))


if __name__ == "__main__":
    main()