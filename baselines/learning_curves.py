"""Learning-curve analysis for the masked-PPO matrix.

Reads ``ppo_mlp_rule_<mode>_s<seed>.history.jsonl`` (one entry per 100k-step
chunk) and reports per-mode mean +/- 95% CI, plateau slopes, and the
SINGLE-vs-chance test. Writes a 2x2 figure to ``paper/figs/``.

Usage::

    python -m baselines.learning_curves --out runs
"""
import argparse
import glob
import json
import os

import numpy as np
from scipy import stats

MODES = ["single", "static", "dynamic", "obvious"]


def load_history(out_dir, mode):
    runs = []
    for path in sorted(glob.glob(os.path.join(out_dir, f"ppo_mlp_rule_{mode}_s*.history.jsonl"))):
        with open(path) as f:
            entries = [json.loads(line) for line in f if line.strip()]
        runs.append(entries)
    return runs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs")
    ap.add_argument("--figdir", default="paper/figs")
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(9, 5.6), sharex=True, sharey=True)
    print("| chunk | " + " | ".join(m.upper() for m in MODES) + " |")
    print("|---|---|---|---|---|")

    curves = {}
    for ax, mode in zip(axes.ravel(), MODES):
        runs = load_history(args.out, mode)
        n = min(len(r) for r in runs)
        steps = [e["num_timesteps"] for e in runs[0]]
        W = np.array([[e["win_rate"] for e in r[:n]] for r in runs])
        mean, ci = W.mean(0), stats.t.ppf(0.975, W.shape[0] - 1) * W.std(0, ddof=1) / np.sqrt(W.shape[0])
        curves[mode] = (steps, W)
        x = np.arange(n)
        ax.plot(x, mean, marker="o", ms=3)
        ax.fill_between(x, mean - ci, mean + ci, alpha=0.25)
        ax.axhline(0.5, color="gray", ls=":", lw=1)
        ax.set_title(mode.upper(), fontsize=10)
        ax.set_ylim(0.2, 1.0)

        # plateau slope over the last 3 chunks
        slopes = np.array([np.polyfit(x[-3:], [e["win_rate"] for e in r[:n]][-3:], 1)[0]
                           for r in runs])
        t, p = stats.ttest_1samp(slopes, 0.0) if slopes.size > 1 else (float("nan"), float("nan"))
        print(f"PLATEAU {mode}: last-3 slope {slopes.mean():+.4f}/chunk "
              f"(t={t:.2f}, p={p:.3f})")

    print("")
    print("| chunk | " + " | ".join(m.upper() for m in MODES) + " |")
    print("|---|---|---|---|---|")
    for i in range(10):
        row = [str(curves[m][0][i] // 1000) + "k" if i < len(curves[m][0]) else "—" for m in MODES]
        vals = []
        for m in MODES:
            _, W = curves[m]
            if i < W.shape[1]:
                vals.append(f"{W[:, i].mean():.3f}")
            else:
                vals.append("—")
        print(f"| {i+1} | " + " | ".join(vals) + " |")

    # key tests
    print("")
    for mode in MODES:
        _, W = curves[mode]
        final = W[:, -1]
        if mode == "single":
            t, p = stats.ttest_1samp(final, 0.5)
            print(f"TEST single final vs chance 0.5: {final.mean():.3f}, t={t:.2f}, p={p:.4f}")
        if mode in ("dynamic", "obvious"):
            t, p = stats.ttest_rel(final, W[:, 2])
            print(f"TEST {mode} final vs 300k: {final.mean():.3f} vs {W[:, 2].mean():.3f}, "
                  f"paired t={t:.2f}, p={p:.4f}")

    fig.tight_layout()
    os.makedirs(args.figdir, exist_ok=True)
    save = os.path.join(args.figdir, "learning_curves_modes.png")
    fig.savefig(save, dpi=200)
    print(f"saved {save}")


if __name__ == "__main__":
    main()
