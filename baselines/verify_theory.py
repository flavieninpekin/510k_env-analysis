"""Verify the theory propositions in notes/theory.md.

- A: team-size prior (hypergeometric)
- B: membership probabilities
- C: reveal-dial mutual information (analytic vs channel sampling)
- D: reward dilution in team modes

Usage: python -m baselines.verify_theory
"""
import random
from collections import Counter
from math import comb

import numpy as np

from env_510k.game import Game, GameMode


def get_prior(n_games: int = 4000):
    teams = Counter()
    team_sizes = Counter()
    agent_in_red = 0
    n = 0
    for seed in range(n_games):
        random.seed(seed)
        g = Game(mode=GameMode.DYNAMIC)
        rt = g.red_a_team
        if rt is None:
            continue
        n += 1
        teams[frozenset(i for i in range(1, 4) if i in rt)] += 1
        team_sizes[len(rt)] += 1
        if 0 in rt:
            agent_in_red += 1
    return teams, n, team_sizes, agent_in_red


def analytic_mi(teams, n, p):
    p_empty = teams[frozenset()] / n
    q = np.array([c / n for c in teams.values()])
    H_T = float(-np.sum(q * np.log2(q)))
    nonempty = [c / n for t, c in teams.items() if len(t) > 0]
    a = (1 - p) + p * p_empty

    def lg(x):
        return np.log2(x) if x > 0 else 0.0

    H_R = -(a * lg(a)) - sum(p * qq * lg(p * qq) for qq in nonempty)
    h_p = -(p * lg(p) + (1 - p) * lg(1 - p))
    H_R_given_T = (1 - p_empty) * h_p
    return H_R - H_R_given_T, H_T, p_empty


def sampled_mi(p, n_games: int = 6000):
    joint = Counter()
    n = 0
    for seed in range(n_games):
        random.seed(seed + 2026)
        g = Game(mode=GameMode.DYNAMIC)
        rt = g.red_a_team
        if rt is None:
            continue
        t = frozenset(i for i in range(1, 4) if i in rt)
        r = t if random.random() < p else frozenset()
        joint[(t, r)] += 1
        n += 1
    pt, pr = Counter(), Counter()
    for (t, r), c in joint.items():
        pt[t] += c
        pr[r] += c
    mi = 0.0
    for (t, r), c in joint.items():
        if c:
            mi += (c / n) * np.log2((c / n) / ((pt[t] / n) * (pr[r] / n)))
    return mi


def main():
    C5213 = comb(52, 13)
    C5011 = comb(50, 11)
    print(f"[A] analytic P(size 2) = {1 - 4 * C5011 / C5213:.4f}  "
          f"P(size 1) = {4 * C5011 / C5213:.4f}")
    print(f"[B] analytic P(agent in red) = {1 - comb(50, 13) / C5213:.4f}  "
          f"P(solo red) = {C5011 / C5213:.4f}")

    teams, n, team_sizes, agent_in_red = get_prior()
    print(f"[A-emp] size1 = {team_sizes[1] / n:.4f}  size2 = {team_sizes[2] / n:.4f}  "
          f"(analytic 0.2353 / 0.7647)")
    solo = teams[frozenset()] / n
    print(f"[B-emp] P(agent in red) ~ {agent_in_red / n:.4f}  P(solo red) ~ {solo:.4f}")

    for p in [0.25, 0.5, 0.75, 1.0]:
        mi, H_T, p_empty = analytic_mi(teams, n, p)
        ms = sampled_mi(p)
        print(f"[C] p={p:.2f}: analytic MI={mi:.4f}  sampled={ms:.4f}  "
              f"p*H(T)={p * H_T:.4f}  ratio={mi / (p * H_T):.3f}  (p_empty={p_empty:.4f})")

    # D: reward dilution in STATIC
    own, tm = [], []
    for seed in range(1500):
        random.seed(seed + 777)
        g = Game(mode=GameMode.STATIC)
        for _ in range(1000):
            pid = g.current_player
            acts = g.get_valid_actions(pid)
            if acts:
                g.play_cards(pid, random.choice(acts).cards)
            elif g.can_pass(pid):
                g.pass_turn(pid)
            else:
                break
            if g.is_over:
                break
        if g.is_over:
            own.append(g.player_510k_scores[0])
            tm.append(g.player_510k_scores[2])
    own, tm = np.array(own), np.array(tm)
    print(f"[D] std(own)={own.std():.2f}  std(teammate)={tm.std():.2f}  "
          f"own std share={own.std() / (own.std() + tm.std()):.3f}")


if __name__ == "__main__":
    main()