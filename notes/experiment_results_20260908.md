# 510K Baseline Experiment Results (preliminary)

> Date: 2026-09-08. Preliminary results from the crash-tolerant pipeline.
> All numbers: masked PPO, evaluated as **player 0 vs rule bots**, 100 seeded
> games, mean ± std over seeds. **illegal-action rate = 0.0 for all runs.**

## 1. Capability profile (win rate vs rule bot)

Training: MLP = 300k steps; LSTM / IPPO = 200k steps; 3 seeds.

| mode | MLP PPO | LSTM PPO | IPPO (shared) |
|---|---|---|---|
| SINGLE | 0.510 ± 0.022 | 0.437 ± 0.038 | 0.363 ± 0.017 |
| STATIC | 0.750 ± 0.016 | — | — |
| DYNAMIC | 0.847 ± 0.026 | 0.850 ± 0.000 | 0.820 ± 0.033 |
| OBVIOUS | 0.793 ± 0.019 | — | — |

Reading:
- SINGLE is the hardest for every method (win rate ≈ chance 0.5, no teammate).
- Team modes inflate win rate because rule-bot teammates share the outcome.
- SINGLE method ordering MLP > LSTM > IPPO consistent with 200-300k budget.

## 2. Information-revelation ablation (MLP PPO, DYNAMIC)

`RevealEnv` reveals the team with probability p per decision; p=0 ≡ DYNAMIC,
p=1 ≡ OBVIOUS. Training 200k steps, 3 seeds.

| reveal p | win rate | reward |
|---|---|---|
| 0.00 | 0.847 ± 0.026 | 77.2 |
| 0.25 | 0.850 ± 0.008 | 72.7 |
| 0.50 | 0.845 ± 0.009 | 73.5 |
| 0.75 | 0.827 ± 0.019 | 74.8 |
| 1.00 | 0.793 ± 0.019 | 67.4 |

Revealing teammate identity does **not** help a short-trained PPO against rule
bots (flat / slightly decreasing). Consistent with IIGC's *deceptive
stability*: extra information does not translate into better policies at short
horizons. Longer training + stronger opponents needed (§7 future work).

## 3. 1v3 vs 2v2 robustness (MLP PPO, DYNAMIC)

300 seeded eval games per seed, split by red-team size (76% 2v2 / 24% 1v3).

| seed | 2v2 win rate | 1v3 win rate |
|---|---|---|
| 0 | 0.859 | 0.909 |
| 1 | 0.842 | 0.864 |
| 2 | 0.850 | 0.909 |

Results are insensitive to team composition → the "embrace 1v3" design choice
does not drive the conclusions.

## 4. Reproduce

```bash
# full baseline matrix (resumable, crash-tolerant)
python -m baselines.run_matrix --modes single static dynamic obvious \
    --policy mlp --seeds 0 1 2 3 4 --total 1000000 --out runs
python -m baselines.run_matrix --modes single dynamic \
    --policy lstm --seeds 0 1 2 --total 1000000 --out runs
python -m baselines.run_matrix --modes single dynamic \
    --policy ippo --seeds 0 1 2 --total 1000000 --out runs

# info-reveal ablation
for r in 0.25 0.5 0.75; do
  python -m baselines.run_matrix --modes dynamic --policy mlp \
    --reveal $r --seeds 0 1 2 --total 1000000 --out runs
done

# analysis
python -m baselines.aggregate --out runs
python -m baselines.plot_results --out runs --all     # -> paper/figs/
python -m baselines.analyze_1v3 --mode dynamic --seed 0 --policy mlp --out runs
```

## 5. Caveats

- Preliminary: short training (200-300k), 3 seeds, rule-bot opponents only.
- Machine flaky (random native crashes) — runner restarts crashed chunks; a
  clean machine is recommended for the final 1M-step matrix.
- Win rate vs rule bots is a coarse metric; finish position, agent-score
  contribution, and random-bot evals should be added.