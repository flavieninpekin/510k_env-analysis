# 510K Baselines

Reference implementations for the 510K benchmark paper. Three policies and a
crash-tolerant experiment orchestrator.

## Policies

| `--policy` | Implementation | Interface |
|---|---|---|
| `mlp`  | `sb3_contrib.MaskablePPO` (MLP) | single-agent: controls player 0, others auto-played (random or rule bot) |
| `lstm` | `RecurrentPPO` + `MaskedLstmActorCriticPolicy` | single-agent, recurrent/memory |
| `ippo` | `MaskablePPO` on a self-play env (shared policy, all seats) | multi-agent, parameter-sharing IPPO |

All policies use action masking; the observation space is the normalized
112-dim (state) vector. Eval always plays the trained policy as **player 0
against fixed bots**, so results across policies are comparable.

## Usage

Train one chunk (resumable; repeat or use the runner):

```bash
python -m baselines.train_ppo --mode dynamic --seed 42 --policy mlp --total 1000000
python -m baselines.train_ppo --mode dynamic --seed 42 --policy mlp --chunk 100000   # resumes
python -m baselines.train_ppo --mode dynamic --seed 42 --policy mlp --eval-only       # evaluate
```

Evaluate a trained checkpoint:

```bash
python -m baselines.train_ppo --mode dynamic --seed 42 --policy mlp --eval-only
```

Full experiment matrix with crash tolerance (each chunk in a subprocess,
restarts from the last checkpoint on failure):

```bash
python -m baselines.run_matrix --modes single static dynamic obvious \
    --policy mlp --seeds 0 1 2 3 4 --total 1000000 --out runs
python -m baselines.run_matrix --only dynamic:42 --policy lstm --total 1000000 --out runs
```

## Outputs

Each task writes into `--out`:

- `<tag>_<mode>_s<seed>.zip` — checkpoint (resumable)
- `<tag>_<mode>_s<seed>.done` — completion marker
- `<tag>_<mode>_s<seed>.eval.json` — eval metrics after the latest chunk:
  `win_rate`, `mean_reward`, `std_reward`, `mean_len`, `illegal_action_rate`

Tags: `ppo_mlp_rule`, `ppo_mlp_random`, `ppo_lstm_rule`, ..., `ippo`.

## Crash tolerance

The machine this was developed on exhibits random native crashes (0xC0000005 /
0x80000003) in sustained Python workloads. `run_matrix.py` therefore runs each
training chunk as a short subprocess and restarts from the last checkpoint on
any non-zero exit, which makes long training runs robust despite the flaky
runtime.