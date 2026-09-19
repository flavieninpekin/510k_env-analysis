"""Model-level coupling probe: does a trained policy condition on the hidden team?

This is the strategy-level counterpart to ``coupling_probes.py`` (which measures
the environment's structural couplings with fixed bots). Here we ask whether a
*learned* policy exploits the cooperative structure:

  * cooperation inference (C2): does the agent defer to the true partner more
    than to an opponent, i.e. does its behaviour depend on the latent team?
  * information utilization (C4): compare a DYNAMIC policy (team hidden) with an
    OBVIOUS policy (team revealed). If the revealed policy shows a stronger
    defer asymmetry but not better outcomes, the information is *available but
    not profitably used*.

For every following decision (a leader exists and the agent may pass) we record
whether the agent passed, whether the leader was its true partner, and its side
size (1 = solo, 2 = pair, 3 = trio).

Usage::

    python -m baselines.analyze_team_conditioning --modes static dynamic obvious \
        --seeds 0 1 2 --n-episodes 300 --out runs
"""
import argparse
import json
import os

import numpy as np

from .config import PPOConfig
from .train_ppo import load_model, checkpoint_path, is_lstm, make_eval_env


def _load_mappo(cfg: PPOConfig):
    from sb3_contrib import MaskablePPO
    from .train_mappo import make_env, checkpoint_path as mappo_ckpt
    path = mappo_ckpt(cfg.mode, cfg.seed, cfg.out_dir)
    env = make_env(cfg.mode)
    env.unwrapped.set_rule_bot()
    model = MaskablePPO.load(path, env=env)
    return model, env, path


def team_map(mode: str, game):
    if mode == "static":
        return {0: 0, 1: 1, 2: 0, 3: 1}
    red = game.red_a_team
    if red is None:
        return None
    return {i: (0 if i in red else 1) for i in range(4)}


def decide(model, cfg, obs, info, state=None, ep_start=None):
    if is_lstm(cfg):
        a, state = model.predict(obs, state=state, episode_start=ep_start,
                                 deterministic=True)
        return int(a), state
    a, _ = model.predict(obs, action_masks=info["action_mask"], deterministic=True)
    return int(a), None


def run_episodes(cfg: PPOConfig, n_episodes: int) -> list:
    if cfg.policy == "mappo":
        model, env, _ = _load_mappo(cfg)
    else:
        model = load_model(cfg, checkpoint_path(cfg))
        env = make_eval_env(cfg)
    rows = []
    for ep in range(n_episodes):
        obs, info = env.reset(seed=7000 + ep)
        done = False
        state, ep_start = None, np.array([True])
        while not done:
            game = env.unwrapped.game
            leader = game.last_trick.player if game.last_trick is not None else None
            can_pass = game.can_pass(0)
            tm = team_map(cfg.mode, game)
            partners = set()
            team_size = 1
            if tm is not None:
                partners = {i for i in range(4) if i != 0 and tm[i] == tm[0]}
                team_size = len(partners) + 1

            a, state = decide(model, cfg, obs, info, state, ep_start)
            ep_start = np.array([False])
            passed = int(a == 0 and can_pass)

            if leader is not None and leader != 0 and can_pass:
                rows.append({
                    "team_size": int(team_size),
                    "partner_leads": int(leader in partners),
                    "passed": passed,
                })
            obs, r, done, _, info = env.step(a)
    return rows


def summarise(rows: list) -> dict:
    if not rows:
        return {}
    passed = np.array([r["passed"] for r in rows], float)
    pl = np.array([r["partner_leads"] for r in rows], bool)
    ts = np.array([r["team_size"] for r in rows], int)

    def rate(mask):
        return float(passed[mask].mean()) if mask.sum() else None

    out = {
        "n_decisions": len(rows),
        "n_partner_leads": int(pl.sum()),
        "n_opponent_leads": int((~pl).sum()),
        "pass_rate_partner_leads": rate(pl),
        "pass_rate_opponent_leads": rate(~pl),
        "pass_asymmetry": (rate(pl) - rate(~pl)) if (pl.sum() and (~pl).sum()) else None,
        "by_team_size": {},
    }
    for s in sorted(set(ts.tolist())):
        m = ts == s
        ms = m & pl
        mo = m & (~pl)
        out["by_team_size"][str(s)] = {
            "n": int(m.sum()),
            "pass_rate": rate(m),
            "pass_rate_partner_leads": rate(ms) if ms.sum() else None,
            "pass_rate_opponent_leads": rate(mo) if mo.sum() else None,
            "pass_asymmetry": (rate(ms) - rate(mo)) if (ms.sum() and mo.sum()) else None,
        }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["static", "dynamic", "obvious"])
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--policy", default="mlp", choices=["mlp", "lstm", "mappo"])
    ap.add_argument("--opponent", default="rule", choices=["random", "rule"])
    ap.add_argument("--tag-suffix", default="")
    ap.add_argument("--n-episodes", type=int, default=300)
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    result = {}
    for mode in args.modes:
        per_seed = []
        for seed in args.seeds:
            cfg = PPOConfig(mode=mode, seed=seed, policy=args.policy,
                            opponent=args.opponent, out_dir=args.out,
                            tag_suffix=args.tag_suffix)
            if args.policy == "mappo":
                from .train_mappo import checkpoint_path as mappo_ckpt
                exists = os.path.exists(mappo_ckpt(mode, seed, args.out))
            else:
                exists = os.path.exists(checkpoint_path(cfg))
            if not exists:
                print(f"skip {mode} s{seed} (no checkpoint)")
                continue
            per_seed.append(summarise(run_episodes(cfg, args.n_episodes)))
        if per_seed:
            result[mode] = per_seed

    suffix = f"_{args.tag_suffix}" if args.tag_suffix else ""
    path = os.path.join(args.out,
                        f"team_conditioning_{args.policy}_{args.opponent}{suffix}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)

    print(f"{'mode':<9}{'s':>3}{'n':>7}{'pLead':>8}{'pOpp':>8}{'asym':>8}")
    for mode, seeds in result.items():
        for s, r in enumerate(seeds):
            print(f"{mode:<9}{s:>3}{r.get('n_decisions', 0):>7}"
                  f"{r.get('pass_rate_partner_leads') or 0:>8.3f}"
                  f"{r.get('pass_rate_opponent_leads') or 0:>8.3f}"
                  f"{r.get('pass_asymmetry') or 0:>8.3f}")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
