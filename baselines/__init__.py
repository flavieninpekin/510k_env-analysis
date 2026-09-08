"""Baseline training stack for the 510K benchmark.

- ``train_ppo.py``: single-agent (or recurrent) masked PPO. Resumable and
  chunked: each invocation trains ``chunk_size`` steps and exits; a ``.done``
  marker is written when ``total_timesteps`` is reached.
- ``run_matrix.py``: crash-tolerant orchestrator. Spawns one subprocess per
  (mode, seed) chunk, restarts from the last checkpoint on failure, and marks
  tasks complete via the ``.done`` marker.

The chunked + resumable + subprocess design makes long training runs robust to
the flaky native crashes observed on this machine.
"""