# 510K-Env-Analysis

Reproduction kit for the 510K benchmark paper
(*510K: A Card-Game Testbed for Dynamic Cooperation, Partial Observability, and
Information Revelation in Multi-Agent RL*).

The environment itself lives in
[`flavieninpekin/510k_env`](https://github.com/flavieninpekin/510k_env); this
repo contains everything needed to reproduce the paper's experiments and
figures.

## Layout

```
baselines/    training + evaluation + analysis code (crash-tolerant runner)
experiments/  experiment plan / todo
notes/        positioning analysis and result records
paper/        manuscript draft + figures
runs/         per-run results (*.eval.json, *.history.jsonl)
```

## Setup

```bash
pip install -e ../510k_env[all]      # or: pip install git+https://github.com/flavieninpekin/510k_env
pip install -r requirements.txt
```

## Reproduce

```bash
# core capability profile (4 modes x 3 policies x seeds x 1M steps)
python -m baselines.run_matrix --modes single static dynamic obvious \
    --policy mlp --seeds 0 1 2 3 4 --total 1000000 --out runs
python -m baselines.run_matrix --modes single static dynamic obvious \
    --policy lstm --seeds 0 1 2 --total 1000000 --out runs
python -m baselines.run_matrix --modes single static dynamic obvious \
    --policy ippo --seeds 0 1 2 --total 1000000 --out runs

# info-reveal ablation
for r in 0.25 0.5 0.75; do
  python -m baselines.run_matrix --modes dynamic --policy mlp \
    --reveal $r --seeds 0 1 2 --total 1000000 --out runs
done

# 1v3 robustness
python -m baselines.analyze_1v3 --mode dynamic --seed 0 --policy mlp --out runs

# tables + figures
python -m baselines.aggregate --out runs
python -m baselines.plot_results --out runs --all   # -> paper/figs/
```

See `experiments/PLAN.md` for the full experiment list and status.

## Notes on the runtime

The development machine exhibits random native crashes (0xC0000005 /
0x80000003) in sustained Python workloads. Training is therefore chunked into
short subprocesses (`run_matrix.py`) that checkpoint and restart on failure;
long runs are robust but slower. A clean machine is recommended for the final
matrix.