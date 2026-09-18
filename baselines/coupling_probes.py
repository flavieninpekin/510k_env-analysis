"""Structural coupling probes for 510K (no training required).

Drives all four seats with a fixed bot and measures the *entanglement* between
the benchmark's challenge axes. These are the cheap, deterministic signatures
behind the paper's central claim that 510K's axes interact rather than stack.

Couplings probed
----------------
C1 objective entanglement  : does the immediate-score objective trade off with
                             the terminal finish objective?
C3 objective x cooperation : does the value of the scoring objective (slope of
                             reward on own score) change with team structure?
C2 latent cooperation      : how does the reward->observation map depend on the
                             hidden team (deal-time composition statistics)?

Usage::

    python -m baselines.coupling_probes --n-episodes 5000 --bot rule
"""
import argparse
import json
import os
import random

import numpy as np
from scipy import stats

from env_510k.game import Game, GameMode
from env_510k.scorer import Scorer
from env_510k.bots.rule_bot import choose_action as rule_choose

MODES = ["single", "static", "dynamic", "obvious"]


def _play_one(mode: str, bot: str, seed: int) -> dict:
    random.seed(seed)
    np.random.seed(seed)
    game = Game(mode=GameMode(mode), num_players=4, include_jokers=False)

    guard = 0
    while not game.is_over and guard < 5000:
        guard += 1
        pid = game.current_player
        actions = game.get_valid_actions(pid)
        if not actions:
            if game.can_pass(pid):
                game.pass_turn(pid)
            else:
                break
            continue
        chosen = rule_choose(game, pid, actions) if bot == "rule" else random.choice(actions)
        if chosen is None:
            if game.can_pass(pid):
                game.pass_turn(pid)
            else:
                game.play_cards(pid, actions[0].cards)
        else:
            game.play_cards(pid, chosen.cards)

    rewards = Scorer(game).compute_rewards()
    order = list(game.finish_order)
    pos = {p: (order.index(p) + 1) for p in order}
    teammates = {}
    for i in range(4):
        if game.red_a_team is not None:
            teammates[i] = {j for j in game.red_a_team if j != i}
        elif mode == "static":
            teammates[i] = {j for j in range(4) if j != i and (i - j) % 2 == 0}
        else:
            teammates[i] = set()

    return {
        "finish_pos": [pos.get(i, 5) for i in range(4)],
        "score": [float(game.player_510k_scores[i]) for i in range(4)],
        "reward": [float(rewards.get(i, 0.0)) for i in range(4)],
        "mate_score": [sum(float(game.player_510k_scores[j]) for j in teammates[i])
                       for i in range(4)],
        "red_size": len(game.red_a_team) if game.red_a_team is not None else 0,
        "tricks": sum(1 for a in game.actions_log if a.get("action") == "trick_end"),
    }


def _spearman(x, y) -> float:
    x, y = np.asarray(x, float), np.asarray(y, float)
    if x.size < 3 or np.unique(x).size < 2 or np.unique(y).size < 2:
        return float("nan")
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    rx -= rx.mean(); ry -= ry.mean()
    denom = np.sqrt((rx ** 2).sum() * (ry ** 2).sum())
    return float((rx * ry).sum() / denom) if denom else float("nan")


def _ci95(x):
    x = np.asarray(x, float)
    return stats.t.ppf(0.975, x.size - 1) * x.std(ddof=1) / np.sqrt(x.size)


def _r2(score, mate, reward):
    X = np.column_stack([np.ones_like(score), score, mate])
    beta, *_ = np.linalg.lstsq(X, reward, rcond=None)
    pred = X @ beta
    return float(1 - ((reward - pred) ** 2).sum() / ((reward - reward.mean()) ** 2).sum())


def probe_mode(mode: str, bot: str, n: int, rng: np.random.Generator) -> dict:
    eps = [_play_one(mode, bot, seed) for seed in range(n)]

    score = np.array([s for e in eps for s in e["score"]])
    reward = np.array([r for e in eps for r in e["reward"]])
    mate = np.array([m for e in eps for m in e["mate_score"]])

    # per-episode C1 signatures
    top_first = np.array([
        int(e["score"][int(np.argmax(e["score"]))] > 0
            and e["finish_pos"][int(np.argmax(e["score"]))] == 1)
        for e in eps
    ], float)
    rhos = np.array([_spearman(e["score"], e["finish_pos"]) for e in eps], float)
    rhos = rhos[~np.isnan(rhos)]

    rho_t, rho_p = stats.ttest_1samp(rhos, 0.0)
    binom_p = stats.binomtest(int(top_first.sum()), top_first.size, 0.25,
                              alternative="greater").pvalue

    # C3: regression reward ~ own score + correlation
    lr = stats.linregress(score, reward)
    r_own, p_own = stats.pearsonr(score, reward)

    # C2: R^2 of reward on own+teammate, with a bootstrap CI over episodes
    boots = []
    for _ in range(500):
        idx = rng.integers(0, len(eps), len(eps))
        s = np.array([x for i in idx for x in eps[i]["score"]])
        r = np.array([x for i in idx for x in eps[i]["reward"]])
        m = np.array([x for i in idx for x in eps[i]["mate_score"]])
        boots.append(_r2(s, m, r))
    r2 = _r2(score, mate, reward)
    r2_lo, r2_hi = np.percentile(boots, [2.5, 97.5])

    return {
        "mode": mode, "bot": bot, "n": n,
        "p_top_scorer_finishes_first": float(top_first.mean()),
        "top_scorer_first_binom_p_vs_0.25": float(binom_p),
        "score_finish_rank_corr": float(rhos.mean()),
        "score_finish_rank_corr_ci95": float(_ci95(rhos)),
        "score_finish_rank_corr_t": float(rho_t),
        "score_finish_rank_corr_p": float(rho_p),
        "reward_on_own_score_slope": float(lr.slope),
        "reward_on_own_score_slope_se": float(lr.stderr),
        "reward_on_own_score_slope_p": float(lr.pvalue),
        "corr_own_score_reward": float(r_own),
        "corr_own_score_reward_p": float(p_own),
        "r2_reward_own_plus_mate": r2,
        "r2_ci95": [float(r2_lo), float(r2_hi)],
        "mean_tricks": float(np.mean([e["tricks"] for e in eps])),
        "red_size_hist": {int(k): int(v) for k, v in
                          zip(*np.unique([e["red_size"] for e in eps], return_counts=True))},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-episodes", type=int, default=5000)
    ap.add_argument("--bot", default="rule", choices=["rule", "random"])
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    rng = np.random.default_rng(0)
    rows = [probe_mode(m, args.bot, args.n_episodes, rng) for m in MODES]
    os.makedirs(args.out, exist_ok=True)
    path = os.path.join(args.out, f"coupling_probes_{args.bot}.json")
    with open(path, "w") as f:
        json.dump(rows, f, indent=2)

    print(f"bot={args.bot}  n={args.n_episodes}/mode")
    print(f"{'mode':<8} {'rho':>8} {'rho_ci':>8} {'rho_p':>9} {'top1st':>7} "
          f"{'binom_p':>9} {'corr':>6} {'slope':>7} {'R2':>6} {'R2_ci':>13}")
    for r in rows:
        rho_ci = r["score_finish_rank_corr_ci95"]
        r2_lo, r2_hi = r["r2_ci95"]
        print(f"{r['mode']:<8} {r['score_finish_rank_corr']:>8.3f} "
              f"{rho_ci:>8.3f} {r['score_finish_rank_corr_p']:>9.1e} "
              f"{r['p_top_scorer_finishes_first']:>7.3f} "
              f"{r['top_scorer_first_binom_p_vs_0.25']:>9.1e} "
              f"{r['corr_own_score_reward']:>6.3f} "
              f"{r['reward_on_own_score_slope']:>7.3f} "
              f"{r['r2_reward_own_plus_mate']:>6.3f} "
              f"[{r2_lo:.3f},{r2_hi:.3f}]")
    print(f"saved {path}")


if __name__ == "__main__":
    main()
