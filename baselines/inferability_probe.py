"""Inferability probe: is the hidden team relation decodable from public play?

The behavioural probe (`analyze_team_conditioning.py`) shows trained policies do
not condition on the true partner. A natural objection is that the relation may
be un-inferable from what the agent can see. This script tests the opposite by
simulating games with fixed bots and training a linear probe to predict the
*true* relationship (which opponents are teammates) from features the agent
has access to:

  * ``obs``  -- single-step observation only (what a memoryless policy sees),
  * ``hist`` -- observation + public history: which players have revealed red
               aces, plus who deferred to / overtook whom.

Key quantities:
  * AUC of each probe for partner identity;
  * the subset of decisions where the relation is *logically determined* by
    public information (own red aces + revealed red aces), where a memory-based
    agent could know it with certainty.

A high ``hist`` AUC (and near-certain accuracy on the determined subset) with a
chance-level ``obs`` AUC establishes that the relation is inferable from
public play but invisible to a memoryless policy.

Usage::

    python -m baselines.inferability_probe --modes dynamic obvious --n-games 1200
"""
import argparse
import json
import os
import random

import numpy as np
import torch
import torch.nn as nn

from env_510k.card import Rank
from env_510k.game import Game, GameMode
from env_510k.bots.rule_bot import choose_action as rule_choose


def teammates_of(game, mode, p=0):
    if mode == "static":
        return {2} if p == 0 else ({0} if p == 2 else set())
    red = game.red_a_team
    if red is None:
        return set()
    return (red - {p}) if p in red else {i for i in range(4) if i not in red and i != p}


def _own_red(game, p=0):
    n = 0
    for c in game.players[p].hand:
        if c.rank == Rank.ACE and c.is_red:
            n += 1
    return n


def state_features(game, own_red):
    f = [1.0 if own_red == 0 else 0.0, 1.0 if own_red == 1 else 0.0,
         1.0 if own_red == 2 else 0.0]
    for i in range(4):
        f.append(len(game.players[i].hand) / 13.0)
    for i in range(4):
        f.append(game.player_510k_scores[i] / 150.0)
    f.append(game.pass_count / 3.0)
    f.append(game.trick_pending_score / 100.0)
    lt = game.last_trick
    leader = lt.player if lt is not None else -1
    for i in range(4):
        f.append(1.0 if leader == i else 0.0)
    f.append(1.0 if game.can_pass(0) else 0.0)
    return f


def history_extra(red_played):
    """red_played[j] = number of red aces player j has revealed by playing."""
    f = []
    for j in range(1, 4):
        f.append(min(red_played[j], 2) / 2.0)
    f.append(1.0 if any(red_played[j] >= 1 for j in (1, 2, 3)) else 0.0)
    return f


def pair_features(follow, defer, overtake):
    f = []
    for j in range(1, 4):
        n0, nj = follow[0, j], follow[j, 0]
        d0 = defer[0, j] / n0 if n0 > 0 else 0.0
        dj = defer[j, 0] / nj if nj > 0 else 0.0
        f += [min(n0 / 5.0, 1.0), min(nj / 5.0, 1.0),
              d0, dj, min(d0, dj), abs(d0 - dj)]
    return f


def determined_partner(own_red, red_played):
    """Is the teammate set logically determined by public information?"""
    n_red_revealed = sum(1 for j in (1, 2, 3) if red_played[j] >= 1)
    if own_red == 2:
        return True                      # solo: no teammate
    if own_red == 1:
        return n_red_revealed >= 1       # the other red ace has been revealed
    if own_red == 0:
        if any(red_played[j] >= 2 for j in (1, 2, 3)):
            return True                  # one opponent held both red aces (solo red)
        return n_red_revealed >= 2       # both red holders identified
    return False


def simulate(mode, n_games, seed0=0):
    Xo, Xh, Y, D = [], [], [], []
    for g in range(n_games):
        random.seed(seed0 + g)
        np.random.seed(seed0 + g)
        game = Game(mode=GameMode(mode), num_players=4, include_jokers=False)
        follow = np.zeros((4, 4)); defer = np.zeros((4, 4)); overtake = np.zeros((4, 4))
        red_played = np.zeros(4)
        steps = 0
        while not game.is_over and steps < 3000:
            steps += 1
            pid = game.current_player
            actions = game.get_valid_actions(pid)
            if not actions:
                if game.can_pass(pid):
                    game.pass_turn(pid)
                else:
                    break
                continue

            leader = game.last_trick.player if game.last_trick is not None else -1
            if pid == 0:
                own_red = _own_red(game, 0)
                mates = teammates_of(game, mode, 0)
                Y.append([1.0 if j in mates else 0.0 for j in (1, 2, 3)])
                Xo.append(state_features(game, own_red))
                Xh.append(state_features(game, own_red)
                          + history_extra(red_played) + pair_features(follow, defer, overtake))
                D.append(1.0 if determined_partner(own_red, red_played) else 0.0)

            played_cards = []
            chosen = rule_choose(game, pid, actions)
            passed = False
            if chosen is None:
                if game.can_pass(pid):
                    game.pass_turn(pid); passed = True
                else:
                    played_cards = actions[0].cards
                    game.play_cards(pid, played_cards)
            else:
                played_cards = chosen.cards
                game.play_cards(pid, played_cards)

            for c in played_cards:
                if c.rank == Rank.ACE and c.is_red:
                    red_played[pid] += 1
            if leader is not None and leader != pid:
                follow[pid, leader] += 1
                if passed:
                    defer[pid, leader] += 1
                else:
                    overtake[pid, leader] += 1
    return (np.array(Xo, np.float32), np.array(Xh, np.float32),
            np.array(Y, np.float32), np.array(D, np.float32))


def auc(y, s):
    y, s = np.asarray(y), np.asarray(s)
    pos, neg = s[y > 0.5], s[y <= 0.5]
    if pos.size == 0 or neg.size == 0:
        return float("nan")
    order = np.argsort(np.concatenate([pos, neg]))
    ranks = np.empty_like(order, dtype=float)
    ranks[order] = np.arange(1, pos.size + neg.size + 1)
    return float((ranks[:pos.size].sum() - pos.size * (pos.size + 1) / 2) / (pos.size * neg.size))


def fit_predict(Xtr, Ytr, Xte, epochs=400, lr=0.05, seed=0):
    torch.manual_seed(seed)
    mu, sd = Xtr.mean(0), Xtr.std(0) + 1e-6
    Xtr, Xte = (Xtr - mu) / sd, (Xte - mu) / sd
    model = nn.Linear(Xtr.shape[1], 3)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    lossf = nn.BCEWithLogitsLoss()
    xt, yt = torch.tensor(Xtr), torch.tensor(Ytr)
    for _ in range(epochs):
        opt.zero_grad(); loss = lossf(model(xt), yt); loss.backward(); opt.step()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(Xte))).numpy()


def evaluate(name, Xo, Xh, Y, D, test_frac=0.25, seed=0):
    n = Xo.shape[0]
    n_test = int(n * test_frac)
    tr, te = slice(0, n - n_test), slice(n - n_test, n)
    po = fit_predict(Xo[tr], Y[tr], Xo[te], seed=seed)
    ph = fit_predict(Xh[tr], Y[tr], Xh[te], seed=seed)
    out = {"n_samples": int(n), "determined_frac": float(D.mean()),
           "prevalence": [float(p) for p in Y[te].mean(0)]}
    dt = D[te] > 0.5
    for tag, pred in (("obs", po), ("hist", ph)):
        out[f"{tag}_auc"] = [auc(Y[te][:, k], pred[:, k]) for k in range(3)]
        out[f"{tag}_auc_mean"] = float(np.nanmean(out[f"{tag}_auc"]))
        if dt.sum() > 10:
            acc = []
            for k in range(3):
                acc.append(float(((pred[dt, k] > 0.5) == (Y[te][dt, k] > 0.5)).mean()))
            out[f"{tag}_acc_determined"] = acc
            out[f"{tag}_acc_determined_mean"] = float(np.mean(acc))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["dynamic", "obvious"])
    ap.add_argument("--n-games", type=int, default=1200)
    ap.add_argument("--seed0", type=int, default=0)
    ap.add_argument("--save-npz", default=None,
                    help="save one shard to <prefix>_<mode>_s<seed0>.npz")
    ap.add_argument("--aggregate", default=None,
                    help="glob (with {mode}) of shard npz files: train and evaluate")
    ap.add_argument("--out", default="runs")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)

    if args.aggregate:
        import glob as _glob
        report = {}
        for mode in args.modes:
            files = sorted(_glob.glob(args.aggregate.format(mode=mode)))
            if not files:
                print(f"no shards for {mode}")
                continue
            Xo = np.concatenate([np.load(f)["Xo"] for f in files])
            Xh = np.concatenate([np.load(f)["Xh"] for f in files])
            Y = np.concatenate([np.load(f)["Y"] for f in files])
            D = np.concatenate([np.load(f)["D"] for f in files])
            res = evaluate(mode, Xo, Xh, Y, D)
            report[mode] = res
            print(f"{mode}: n={res['n_samples']} determined={res['determined_frac']:.3f} "
                  f"obs_AUC={res['obs_auc_mean']:.3f} hist_AUC={res['hist_auc_mean']:.3f} "
                  f"obs_acc_det={res.get('obs_acc_determined_mean', float('nan')):.3f} "
                  f"hist_acc_det={res.get('hist_acc_determined_mean', float('nan')):.3f}")
        path = os.path.join(args.out, "inferability_probe.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"saved {path}")
        return

    report = {}
    for mode in args.modes:
        Xo, Xh, Y, D = simulate(mode, args.n_games, seed0=args.seed0)
        if args.save_npz:
            np.savez(f"{args.save_npz}_{mode}_s{args.seed0}.npz",
                     Xo=Xo, Xh=Xh, Y=Y, D=D)
            print(f"{mode} s{args.seed0}: saved {Xo.shape[0]} samples")
            continue
        res = evaluate(mode, Xo, Xh, Y, D)
        report[mode] = res
        print(f"{mode}: n={res['n_samples']} determined={res['determined_frac']:.3f} "
              f"obs_AUC={res['obs_auc_mean']:.3f} hist_AUC={res['hist_auc_mean']:.3f} "
              f"obs_acc_det={res.get('obs_acc_determined_mean', float('nan')):.3f} "
              f"hist_acc_det={res.get('hist_acc_determined_mean', float('nan')):.3f}")

    if report:
        path = os.path.join(args.out, "inferability_probe.json")
        with open(path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"saved {path}")


if __name__ == "__main__":
    main()
