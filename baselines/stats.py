"""Statistical analysis for the 510K benchmark paper.

Reads the eval JSONs (rule-bot runs in ``runs/``, random-bot runs in
``runs/opp_random/``) and the cooperation-conditioning JSONs, and writes a
markdown report to ``runs/stats_report.md`` with:

  * mean +/- 95% CI per (mode, policy)
  * MLP vs IPPO (Welch t-test) per mode
  * DYNAMIC vs OBVIOUS
  * information-reveal curve: per-seed slope (win ~ p), one-sample t-test, and
    a paired p=0 vs p=1 comparison
  * rule- vs random-bot evaluation (paired by seed)
  * cooperation asymmetry: two-proportion z-test (partner vs opponent) and a
    one-sample t-test of the asymmetry across seeds

Usage::

    python -m baselines.stats --out runs
"""
import argparse
import glob
import json
import os
from collections import defaultdict

import numpy as np
from scipy import stats

MODES = ["single", "static", "dynamic", "obvious"]


def load_evals(out_dir, sub=""):
    rows = []
    for path in sorted(glob.glob(os.path.join(out_dir, sub, "*.eval.json"))):
        with open(path) as f:
            rows.append(json.load(f))
    return rows


def ci95(x):
    x = np.asarray(x, float)
    n = x.size
    if n < 2:
        return float("nan")
    return stats.t.ppf(0.975, n - 1) * x.std(ddof=1) / np.sqrt(n)


def cell(rows, mode, policy, opponent="rule", reveal=None):
    xs = [d["win_rate"] for d in rows
          if d["mode"] == mode and d["policy"] == policy
          and d["opponent"] == opponent and d.get("reveal") == reveal]
    return np.array(xs, float)


def fmt(x, c):
    return f"{x:.3f} ± {c:.3f}"


def two_prop_z(p1, n1, p2, n2):
    if min(n1, n2) == 0:
        return float("nan"), float("nan")
    p = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    if se == 0:
        return float("nan"), float("nan")
    z = (p1 - p2) / se
    return z, 2 * stats.norm.sf(abs(z))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()
    lines = ["# 510K statistical report", ""]

    rule = load_evals(args.out)
    rand = load_evals(args.out, "opp_random")

    # 1. capability profile
    lines += ["## 1 Capability profile (win rate, mean ± 95% CI)", ""]
    lines += ["| mode | MLP (5) | recurrent e05 (2) | IPPO (3) |", "|---|---|---|---|"]
    for mode in MODES:
        cells = {}
        for pol in ["mlp", "ippo"]:
            xs = cell(rule, mode, pol)
            cells[pol] = fmt(xs.mean(), ci95(xs)) if xs.size else "—"
        lines.append(f"| {mode} | {cells['mlp']} | (see §6.4) | {cells['ippo']} |")
    lines.append("")

    # 2. MLP vs IPPO
    lines += ["## 2 MLP vs IPPO (Welch t-test)", ""]
    for mode in MODES:
        a, b = cell(rule, mode, "mlp"), cell(rule, mode, "ippo")
        if a.size > 1 and b.size > 1:
            t, p = stats.ttest_ind(a, b, equal_var=False)
            lines.append(f"- {mode}: MLP {a.mean():.3f} vs IPPO {b.mean():.3f} "
                         f"(diff {a.mean()-b.mean():+.3f}), t={t:.2f}, p={p:.4f}")
    lines.append("")

    # 3. DYNAMIC vs OBVIOUS
    lines += ["## 3 DYNAMIC vs OBVIOUS (MLP, paired by seed)", ""]
    dyn, obv = cell(rule, "dynamic", "mlp"), cell(rule, "obvious", "mlp")
    if dyn.size == obv.size and dyn.size > 1:
        t, p = stats.ttest_rel(dyn, obv)
        lines.append(f"- MLP: DYNAMIC {dyn.mean():.3f} vs OBVIOUS {obv.mean():.3f} "
                     f"(diff {dyn.mean()-obv.mean():+.3f}), paired t={t:.2f}, p={p:.4f}")
    lines.append("")

    # 4. reveal curve
    lines += ["## 4 Information-reveal curve (MLP)", ""]
    p_dyn = cell(rule, "dynamic", "mlp", reveal=None)
    p_obv = cell(rule, "obvious", "mlp")
    p_mid = {p: np.sort(cell(rule, "dynamic", "mlp", reveal=p))
             for p in (0.25, 0.5, 0.75)}
    slopes = []
    for i in range(min(p_dyn.size, p_obv.size, *(v.size for v in p_mid.values()))):
        xs = [0.0, 0.25, 0.5, 0.75, 1.0]
        ys = [p_dyn[i], p_mid[0.25][i], p_mid[0.5][i], p_mid[0.75][i], p_obv[i]]
        slopes.append(np.polyfit(xs, ys, 1)[0])
    slopes = np.array(slopes)
    if slopes.size > 1:
        t, p = stats.ttest_1samp(slopes, 0.0)
        lines.append(f"- per-seed slope (win ~ p): mean {slopes.mean():+.4f}, "
                     f"one-sample t={t:.2f}, p={p:.4f} (n={slopes.size})")
    if p_dyn.size == p_obv.size and p_dyn.size > 1:
        t, p = stats.ttest_rel(p_dyn, p_obv)
        lines.append(f"- paired p=0 vs p=1: {p_dyn.mean():.3f} vs {p_obv.mean():.3f} "
                     f"(diff {p_dyn.mean()-p_obv.mean():+.3f}), t={t:.2f}, p={p:.4f}")
    lines.append("")

    # 5. rule vs random (paired by seed; only seeds present in both)
    lines += ["## 5 Rule- vs random-bot evaluation (paired by seed)", ""]
    for pol in ["mlp", "ippo"]:
        for mode in MODES:
            pairs = []
            for d in rule:
                if d["mode"] != mode or d["policy"] != pol or d["opponent"] != "rule":
                    continue
                if d.get("reveal") is not None:
                    continue
                for e in rand:
                    if (e["mode"] == mode and e["policy"] == pol
                            and e["seed"] == d["seed"]):
                        pairs.append((d["win_rate"], e["win_rate"]))
            if len(pairs) > 1:
                a = np.array([x for x, _ in pairs])
                b = np.array([y for _, y in pairs])
                t, p = stats.ttest_rel(a, b)
                lines.append(f"- {pol} {mode}: rule {a.mean():.3f} vs random "
                             f"{b.mean():.3f}, paired t={t:.2f}, p={p:.4f} (n={len(pairs)})")
    lines.append("")

    # 6. cooperation asymmetry
    lines += ["## 6 Cooperation asymmetry (deferral)", ""]
    for name in ["mlp", "lstm_e05", "lstm_small"]:
        path = os.path.join(args.out, f"team_conditioning_{'mlp_rule' if name=='mlp' else 'lstm_rule_'+name.split('_')[1]}.json")
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        lines.append(f"### {name}")
        for mode, seeds in data.items():
            zs, asyms = [], []
            for s in seeds:
                if s.get("pass_asymmetry") is None:
                    continue
                asyms.append(s["pass_asymmetry"])
                z, p = two_prop_z(s["pass_rate_partner_leads"], s["n_partner_leads"],
                                  s["pass_rate_opponent_leads"], s["n_opponent_leads"])
                zs.append((z, p))
            if asyms:
                t, p1 = stats.ttest_1samp(np.array(asyms), 0.0) if len(asyms) > 1 else (float("nan"), float("nan"))
                zline = ", ".join(f"z={z:.1f},p={p:.1e}" for z, p in zs)
                lines.append(f"- {mode}: asym mean {np.mean(asyms):+.3f} "
                             f"(one-sample t={t:.2f}, p={p1:.4f}, n={len(asyms)}); "
                             f"per-seed z: {zline}")
        lines.append("")

    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, "stats_report.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
