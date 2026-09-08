"""Crash-tolerant experiment orchestrator.

Each (mode, seed, policy, opponent) task is trained by repeatedly invoking
``baselines.train_ppo`` in a subprocess; every invocation trains one chunk and
exits cleanly, resuming from the last checkpoint. If a subprocess dies (native
crash / OOM / killed), the orchestrator simply restarts the same task, which
resumes from its checkpoint. A ``.done`` marker marks completion.
"""
import argparse
import itertools
import os
import subprocess
import sys
import time

from .config import PPOConfig, MODES


def task_done(cfg: PPOConfig) -> bool:
    return os.path.exists(
        os.path.join(cfg.out_dir, f"{cfg.tag()}_{cfg.mode}_s{cfg.seed}.done")
    )


def run_task(cfg: PPOConfig, max_retries: int = 20, python: str = sys.executable):
    os.makedirs(cfg.out_dir, exist_ok=True)
    n_attempts = 0
    while n_attempts < max_retries:
        if task_done(cfg):
            print(f"[done] {cfg.mode} s{cfg.seed} {cfg.tag()}", flush=True)
            return True
        n_attempts += 1
        cmd = [
            python, "-m", "baselines.train_ppo",
            "--mode", cfg.mode, "--seed", str(cfg.seed),
            "--policy", cfg.policy, "--opponent", cfg.opponent,
            "--total", str(cfg.total_timesteps), "--chunk", str(cfg.chunk_size),
            "--n-steps", str(cfg.n_steps), "--batch-size", str(cfg.batch_size),
            "--n-epochs", str(cfg.n_epochs), "--lr", str(cfg.learning_rate),
            "--out", cfg.out_dir,
        ]
        if cfg.reveal is not None:
            cmd += ["--reveal", str(cfg.reveal)]
        t0 = time.time()
        r = subprocess.run(cmd, capture_output=True, text=True)
        dt = time.time() - t0
        if r.returncode == 0:
            print(f"[chunk ok] {cfg.mode} s{cfg.seed} {cfg.tag()} ({dt:.0f}s)", flush=True)
        else:
            print(f"[CRASH] {cfg.mode} s{cfg.seed} {cfg.tag()} attempt {n_attempts} "
                  f"(rc={r.returncode}, {dt:.0f}s) -> resume", flush=True)
    print(f"[FAILED] {cfg.mode} s{cfg.seed} {cfg.tag()} after {max_retries} attempts", flush=True)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=MODES)
    ap.add_argument("--policy", default="mlp", choices=["mlp", "lstm", "ippo"])
    ap.add_argument("--opponent", default="rule", choices=["random", "rule"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--total", type=int, default=1_000_000)
    ap.add_argument("--chunk", type=int, default=100_000)
    ap.add_argument("--out", default="runs")
    ap.add_argument("--reveal", type=float, default=None,
                    help="info-reveal probability for DYNAMIC mode (0..1)")
    ap.add_argument("--max-retries", type=int, default=20)
    ap.add_argument("--only", type=str, default=None,
                    help="comma list of 'mode:seed' to run; overrides --modes/--seeds")
    args = ap.parse_args()

    if args.only:
        tasks = []
        for spec in args.only.split(","):
            m, _, s = spec.partition(":")
            tasks.append(PPOConfig(mode=m, seed=int(s), policy=args.policy,
                                   opponent=args.opponent, total_timesteps=args.total,
                                   chunk_size=args.chunk, out_dir=args.out, reveal=args.reveal))
    else:
        tasks = [
            PPOConfig(mode=m, seed=s, policy=args.policy, opponent=args.opponent,
                      total_timesteps=args.total, chunk_size=args.chunk,
                      out_dir=args.out, reveal=args.reveal)
            for m in args.modes for s in args.seeds
        ]

    print(f"{len(tasks)} tasks", flush=True)
    ok = 0
    for cfg in tasks:
        if run_task(cfg, max_retries=args.max_retries):
            ok += 1
    print(f"done: {ok}/{len(tasks)}", flush=True)
    return 0 if ok == len(tasks) else 1


if __name__ == "__main__":
    sys.exit(main())