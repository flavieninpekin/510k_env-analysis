# 510K-Analysis: Experiment Plan

> Status legend: ✅ done (preliminary) · 🔜 planned · ⏸ blocked (needs healthy machine)
> Base commands assume `cd <repo>` and `pip install -e .[all]` (env) + the
> analysis repo dependencies (torch, sb3_contrib, pettingzoo, matplotlib).

## 0. Validation (done)

- ✅ Env test suite: 82/82 pass (`python -m pytest tests/ -q`)
- ✅ `gymnasium.utils.env_checker` on all 4 modes
- ✅ PettingZoo `api_test` (AEC)
- ✅ Masking compliance: illegal-action rate = 0.0 for all trained policies

## 1. Core capability profile (primary §6.1)

Full matrix: 4 modes × 3 policies × 5 seeds × 1M steps.

| mode | mlp | lstm | ippo |
|---|---|---|---|
| single | ✅ 0.510 (3s, 300k) | ✅ 0.437 (3s, 200k) | ✅ 0.363 (3s, 200k) |
| static | ✅ 0.750 (3s, 300k) | 🔜 | 🔜 |
| dynamic | ✅ 0.847 (3s, 300k) | ✅ 0.850 (3s, 200k) | ✅ 0.820 (3s, 200k) |
| obvious | ✅ 0.793 (3s, 300k) | 🔜 | 🔜 |

```bash
python -m baselines.run_matrix --modes single static dynamic obvious \
    --policy mlp --seeds 0 1 2 3 4 --total 1000000 --out runs
python -m baselines.run_matrix --modes single static dynamic obvious \
    --policy lstm --seeds 0 1 2 --total 1000000 --out runs
python -m baselines.run_matrix --modes single static dynamic obvious \
    --policy ippo --seeds 0 1 2 --total 1000000 --out runs
```

⏸ **Blocked on a healthy machine** (this dev machine has random native
crashes; the runner is crash-tolerant but slow).

## 2. Random-bot evaluation (adds §6.1/§5 evidence)

Evaluate all checkpoints against random bots (in addition to rule bots).

```bash
python -m baselines.train_ppo --mode dynamic --seed 0 --policy mlp --opponent random --eval-only --out runs
```

🔜

## 3. Learning curves (§6.4)

Per-chunk eval history already accumulates into `*.history.jsonl`.
Generate figures once the 1M matrix is complete.

```bash
python -m baselines.plot_results --out runs --curves
```

🔜

## 4. Information-revelation ablation (§6.2)

| p | status |
|---|---|
| 0.00 (DYNAMIC) | ✅ 0.847 (3s) |
| 0.25 | ✅ 0.850 (3s, 200k) |
| 0.50 | ✅ 0.845 (3s, 200k) |
| 0.75 | ✅ 0.827 (3s, 200k) |
| 1.00 (OBVIOUS) | ✅ 0.793 (3s, 300k) |

Extended: 1M steps + random-bot eval + **which p is exploitable**.

```bash
for r in 0.25 0.5 0.75; do
  python -m baselines.run_matrix --modes dynamic --policy mlp \
    --reveal $r --seeds 0 1 2 3 4 --total 1000000 --out runs
done
```

🔜 extended

## 5. 1v3 vs 2v2 robustness (§6.3)

- ✅ Preliminary: 2v2 ≈ 0.84-0.86, 1v3 ≈ 0.86-0.91 across 3 seeds.
- 🔜 On final checkpoints + report finish-position split.

```bash
python -m baselines.analyze_1v3 --mode dynamic --seed 0 --policy mlp --out runs
```

## 6. Agent-own-signal metrics (finish position, score contribution)

Win rate vs rule bots is coarse in team modes. Add:
- finish position of the controlled player,
- the controlled player's own 510K score contribution,
- leaderboard-style pairwise tournaments (agent vs agent) across seeds.

🔜 (new analysis script)

## 7. Reward-shaping stability (future work, §7)

The immediate 510K score is a natural dense reward competing with the terminal
finish objective:
- R0: terminal-only (current).
- R1: + per-trick score shaping (naive, non-potential).
- R2: + potential-based shaping (Ng et al. 1999).
- Metric: does R1 induce reward hacking vs R0/R2? (Skalse et al. 2022 framing)

🔜 (needs a shaped-reward env variant)

## 8. Centralized critics (future work, §7)

MAPPO / QMIX on the PettingZoo AEC env; compare against IPPO.

🔜 (port from sibling `AAAI2027-510k-clear/src/train`)

## 9. Search / planning baseline (future work, §7)

A heuristic-search agent (e.g., depth-limited best-response) as a strong
non-learned baseline.

🔜

## 10. 3-player mode (§3.1)

Include jokers (54 cards); obs is 116-dim. Run the mlp matrix on `3p`.

🔜

## 11. Trajectory dataset release (§3.4)

Collect expert/rule-bot trajectories as `*.npz`/`.jsonl`; release with schema
docs.

🔜

## 12. Human-play interface

CLI already exists (`play-510k`). Add a human-vs-bot eval harness for the
paper's human-comparison table.

🔜

## Priority order

1. §6.1 full matrix (unblocks §6.2/§6.4) — **needs healthy machine**
2. §2 random-bot eval (cheap, adds robustness)
3. §5 finish-position / score-contribution metrics (cheap, strengthens analysis)
4. §7 reward-shaping study (high novelty, medium cost)
5. §8 MAPPO/QMIX, §9 search, §10 3p, §11 trajectory release

## How to run on a clean machine

```bash
git clone https://github.com/flavieninpekin/510k_env.git        # env
git clone https://github.com/flavieninpekin/510k_env-analysis.git # analysis
pip install -e ./510k_env[all]
pip install -r ./510k_env-analysis/requirements.txt
cd 510k_env-analysis && python -m baselines.run_matrix --modes single static dynamic obvious \
    --policy mlp --seeds 0 1 2 3 4 --total 1000000 --out runs
```