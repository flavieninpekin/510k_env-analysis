# 510K: A Card-Game Testbed for Dynamic Cooperation, Partial Observability, and Information Revelation in Multi-Agent RL

**Draft v0.2** — ICLR 2027 benchmark-style submission. Preliminary results; the
full 1M-step experiment matrix is running (see `experiments/PLAN.md`).

---

## Abstract

Existing multi-agent game benchmarks typically isolate individual challenges:
immediate scoring, terminal competitive objectives, fixed cooperation, or
partial observability. We present **510K**, a four-player shedding card game
that *couples* these challenges in a single decision process and additionally
makes the **cooperative relationship itself a hidden latent variable**: team
membership — and even team size (2v2 vs 1v3) — is determined at deal time by a
private red-A distribution and must be inferred from observed play. 510K ships
with four controllable modes (SINGLE, STATIC, DYNAMIC, OBVIOUS) that ablate the
cooperation/information axes, a Gymnasium single-agent and PettingZoo AEC
interface with built-in action masking, and a prior scientific result
(Information-Induced Gradient Contraction, IIGC) already discovered on it. We
report a baseline suite (masked PPO, recurrent masked PPO, independent PPO) and
a preliminary capability profile: SINGLE is hardest for all methods (win rate
≈ chance), team modes inflate win rate, and revealing teammate identity does
*not* help a short-trained PPO against rule-bot opponents — a direct echo of
IIGC's *deceptive stability*. 510K is positioned as a testbed in which
individually-studied challenges are jointly controlled and ablatable.

---

## 1 Introduction

A central question in multi-agent RL (MARL) is how agents should behave when
they face multiple, interacting, and partially hidden objectives. Card games
contain this structure naturally, but most existing benchmarks isolate each
difficulty:

- **Dou Dizhu** (DouZero; Zha et al., ICML 2021) couples competition with
  *fixed* cooperation (landlord vs. two peasants) and sparse terminal reward.
- **Hanabi** (Bard et al., AIJ 2020) isolates cooperative imperfect-information
  reasoning but has no competition and no scoring trade-off.
- **Mahjong** (Suphx; Li et al., 2020) has scoring and partial observability
  but no team structure.
- **Big Two** (2026) studies short-term vs. long-term trade-offs but has no
  scoring or teams.

**510K** combines these axes *and* adds one that none of the above has: **the
team assignment is hidden and inferred**. "Who should I help?" becomes a
latent-variable inference problem rather than a given. Moreover, the axes are
not merely stacked — they interact: a single play of a scoring card changes
the immediate score, the future pattern space, the information revealed about
relationships, and the race to finish first.

Our contributions:

1. **A complete, tested environment.** Gymnasium single-agent and PettingZoo
   AEC interfaces with built-in action masking, four controllable modes, and a
   deterministic rule bot (Section 3).
2. **A challenge decomposition** showing how scoring, terminal competition,
   partial observability, and dynamic cooperation interact, and a
   controlled-variants design in which each axis is independently ablatable
   (Section 4).
3. **A baseline suite and evaluation protocol.** Masked PPO, recurrent masked
   PPO, and independent (shared-policy) PPO, evaluated identically as player 0
   against fixed bots, with masking-compliance as a correctness check
   (Section 5).
4. **A preliminary capability profile.** SINGLE is hardest for every method
   (win rate ≈ chance), team modes inflate win rate, and information
   revelation does not help a short-trained PPO against rule bots — echoing the
   IIGC *deceptive-stability* finding (Section 6).

---

## 2 Related Work

### 2.1 Card-game RL environments and toolkits

- **RLCard** (Zha et al., 2019/2020) is a toolkit over many card games
  (Blackjack, Hold'em, UNO, Dou Dizhu, Mahjong) with easy interfaces and
  action abstraction. Each included game isolates a subset of challenges; none
  exposes a hidden/dynamic team assignment.
- **OpenSpiel** (Lanctot et al., 2019) provides game-theoretic tooling over
  50+ games, oriented toward solving extensive-form games rather than RL under
  sparse rewards.
- **PyTAG** (2023) covers 20+ tabletop games with a common API but shallow
  per-game depth.

### 2.2 Shedding-type and scoring card games

- **DouZero** (Zha et al., ICML 2021) established that a shedding-type game
  with a *fixed* landlord role and sparse terminal reward is a serious
  benchmark; it describes Dou Dizhu as "a task with long horizons and sparse
  reward ... the only time a nonzero reward is incurred is at the end of a
  game." 510K shares this structure and adds **hidden, dynamically-determined
  teams** and an **immediate-scoring objective** that competes with the
  terminal finish objective. **DouZero+** (2022) adds opponent modeling.
- **Big Two** (2026) emphasizes choosing long-term strategic actions over
  locally optimal ones.
- **Tichu, Hearts, Spades** have scoring and fixed, known partnerships; they
  lack hidden/inferred relationships.

### 2.3 Cooperative and imperfect-information benchmarks

- **Hanabi Challenge** (Bard et al., AIJ 2020) is the canonical cooperative
  imperfect-information benchmark; agents cannot see their own cards and must
  reason about others' beliefs. It has no competition and no scoring.
- **Overcooked-AI** (Carroll et al., NeurIPS 2019) probes ad-hoc human–AI
  coordination under full observability.
- **Melting Pot** (Agapiou et al., 2022) studies mixed-motive social
  interaction at a general substrate level, but not card-game structure.

### 2.4 Multi-objective and hidden-role settings

- **MOSMAC** (AAMAS 2025) argues MARL lacks benchmarks for long-horizon
  multi-objective tasks; 510K is a card-game instance with immediate (score)
  and terminal (finish) objectives.
- **Diplomacy/Cicero** (Silver et al., Science 2022) and hidden-role LLM
  studies (Avalon/Werewolf) show community interest in *relationship
  inference*, but lack a small, ground-truth, trainable RL testbed — a gap
  510K fills.

### 2.5 Prior results on 510K

- **Information-Induced Gradient Contraction (IIGC)** (prior work, AAAI2027):
  under hidden team assignment, policy-gradient updates from different
  relationships cancel in expectation, producing *deceptive stability* (the
  agent's policy appears to stop learning without converging). This paper is
  deliberately distinct: IIGC studies a *phenomenon*; here we characterize the
  *environment* as a benchmark.

### 2.6 Positioning

| Challenge | 510K | DouDizhu | Hanabi | Mahjong | Big2 | Tichu | Overcooked | SMAC |
|---|---|---|---|---|---|---|---|---|
| Immediate scoring | ● | ○ | ○ | ● | ○ | ● | ○ | ○ |
| Terminal competition | ● | ● | ○ | ● | ● | ● | ○ | ● |
| Partial observability | ● | ● | ● | ● | ● | ◐ | ○ | ◐ |
| Team cooperation | ● hidden | ● fixed | ● all | ○ | ○ | ● fixed | ● | ◐ |
| **Hidden/dynamic teams** | **●** | ○ | ○ | ○ | ○ | ○ | ○ | ○ |
| Controlled ablation | ● | ○ | ○ | ○ | ○ | ○ | ◐ | ◐ |

● strong / present, ◐ partial, ○ absent. Only 510K couples scoring × terminal
competition × partial observability × **hidden dynamic teams** × controllable
ablation in one game.

---

## 3 The 510K Environment

### 3.1 Rules

510K is a shedding-type card game played with a standard 52-card deck (4
players) or 54 cards including jokers (3 players). Each player is dealt 13 (or
18) cards. On a turn a player must either play a combination that beats the
current trick or pass; when all other active players pass, the trick ends and
its accumulated **score** is awarded to the trick leader, who opens the next
trick.

**Immediate scoring.** Cards score when collected in a won trick: 5 = 5 points,
10 = 10 points, K = 10 points. Each trick thus carries an immediate-score
signal that competes with the game-level objective of emptying one's hand
first.

**Terminal objective.** The game ends when (SINGLE) the first player empties
their hand, or (team modes) when both members of a team finish. Final reward
combines collected 510K points with finish-position bonuses.

**Patterns.** 16 playable combination types: singles, pairs, triples,
three-with-kicker, three-with-pair, straights, consecutive pairs, airplanes
(with/without kickers), bombs (four-of-a-kind), the eponymous **510K** (5-10-K,
suited and non-suited), joker bomb, and (in team modes) red-A singles/pairs.

### 3.2 Team mechanics: hidden dynamic cooperation

- **STATIC**: fixed teams {0,2} vs {1,3}; identities known.
- **DYNAMIC**: teams are determined at deal time by the red-A distribution:
  any player holding a red ace (or all four aces) joins the "red" team. Team
  membership is **hidden** from observations and must be inferred from others'
  play.
- **OBVIOUS**: same deal-time teams as DYNAMIC, but the observation includes a
  4-bit team mask — a direct ablation of *teammate-identity visibility*.

The hidden relationship can be a 2v2 partnership or a **1v3/3v1 solo**
configuration: over 5,000 deals the red team has size 2 in 76% and size 1 in
24% of games. We treat this as a feature: the latent variable to infer is not
only *who* is my partner but *whether I even have one*.

### 3.3 Interfaces

- **Gymnasium single-agent** (`FiveTenKEnv`, registered as `510K-v0`):
  controls player 0; other players are auto-played by a random or rule bot.
  Observation is a normalized 112-dim vector (own-hand one-hot + last-trick
  cards/type + hand sizes + current player + pass count + own score; +4 team
  bits in OBVIOUS). 300-discrete action space with built-in masking. Passes
  `gymnasium.utils.env_checker` in all four modes.
- **PettingZoo AEC** (`FiveTenKMultiEnv`): controls all players; observation is
  a `{"observation", "action_mask"}` dict per agent; passes the official
  `api_test`. We intentionally provide the AEC (not Parallel) API: 510K is
  strictly turn-based with irregular turn order (players skip finished
  opponents; a trick's leader reopens the next trick), so the Parallel
  convention "all agents act every cycle" does not map to legal play.
- **Action masking** is built in, and the legal action set is compact: the
  largest hand produces ≈75 valid patterns, far below the 300 action slots.

### 3.4 Reproducibility

Seeded `reset`, deterministic rules, `pip install env-510k`, 82 passing tests,
and a published reference implementation (`baselines/`). A trajectory release
and Docker configuration are planned (see `experiments/PLAN.md`).

---

## 4 Challenge Decomposition and Controlled Variants

The central design claim: 510K does not *stack* challenges; they **interact and
alter each other's value**. Playing a 5/10/K card:

- changes the immediate score (collected points),
- changes the remaining hand and future pattern space,
- may reveal or hide information about team membership,
- affects who can finish first (terminal objective).

This yields a difficulty hierarchy: immediate reward → situation value →
teammate/opponent intent → future state space → game outcome.

**Controlled variants** (in code; the paper's experimental backbone):

| Axis | SINGLE | STATIC | DYNAMIC | OBVIOUS | Info-reveal (0→100%) |
|---|---|---|---|---|---|
| Scoring | ✓ | ✓ | ✓ | ✓ | ✓ |
| Terminal competition | ✓ | ✓ | ✓ | ✓ | ✓ |
| Teams | none | fixed, known | hidden | hidden, revealed in obs | DYNAMIC + revealed fraction |
| Partial observability | ✓ | ✓ | ✓ | ✓ | graded |

The information-reveal axis is realized by `RevealEnv`, which includes the
team mask with probability `p` per decision (p=0 ≡ DYNAMIC, p=1 ≡ OBVIOUS).

This lets us ask: *which algorithm fails when a specific challenge combination
is introduced?*

---

## 5 Experimental Setup

**Baselines** (all action-masked; `baselines/`):

- **Masked PPO (MLP)** — `sb3_contrib.MaskablePPO`; controls player 0 against
  fixed auto-played opponents (random or rule bot).
- **Recurrent PPO (LSTM)** — `RecurrentPPO` + a custom `MaskedLstmActorCriticPolicy`
  (sb3-contrib ships no masked LSTM policy; we embed the mask in the
  observation and apply it inside the policy).
- **Independent PPO (IPPO)** — a shared-policy self-play baseline
  (`SelfPlayEnv`): one policy controls all seats (parameter-sharing IPPO).

**Evaluation protocol.** Every policy is evaluated identically: the trained
policy plays **player 0 against fixed opponents** (random bot, rule bot) for
seeded games. Metrics: win rate, mean total reward (mean±std), mean episode
length, and **illegal-action rate** (masking compliance).

**Protocol.** Modes SINGLE/STATIC/DYNAMIC/OBVIOUS; ≥5 seeds; mean ± 95% CI.
Capability-profile analysis: which method collapses under (hidden team ×
partial observability). 1v3 robustness: main results on all deals; a 2v2-only
split verifies conclusions do not depend on the filtering choice.

**Crash tolerance.** Training is chunked into subprocesses (`run_matrix.py`);
each chunk saves a checkpoint and is restarted on failure, so long runs are
robust even on a flaky runtime.

**Compute budget.** Measured (single GPU, CUDA): MaskablePPO ≈ 466 env-steps/s.
4 modes × 3 baselines × 5 seeds × 1M steps ≈ 36 GPU-hours; LSTM ~2× slower,
IPPO ~1.5× — budget ≤2M steps for those.

---

## 6 Results

Preliminary results (200-300k env steps, 3 seeds, evaluated as player 0 against
rule bots; the full 1M-step matrix is running). All policies have
**illegal-action rate 0.0** — masking is respected exactly.

### 6.1 Capability profile across modes and methods

Win rate (fraction of episodes with positive team reward), mean over 3 seeds:

| mode | MLP PPO | LSTM PPO | IPPO (shared) |
|---|---|---|---|
| SINGLE | 0.510 ± 0.022 | 0.437 ± 0.038 | 0.363 ± 0.017 |
| STATIC | 0.750 ± 0.016 | — | — |
| DYNAMIC | 0.847 ± 0.026 | 0.850 ± 0.000 | 0.820 ± 0.033 |
| OBVIOUS | 0.793 ± 0.019 | — | — |

Reading:

- **SINGLE is the hardest setting** for every method: no teammate, win rate
  ≈ chance (0.5) for MLP PPO and below for LSTM/IPPO (comparatively
  under-trained at 200k steps).
- **Team modes inflate win rate** because a teammate's play contributes to the
  team outcome; 0.75-0.85 in STATIC/DYNAMIC/OBVIOUS does not mean the task is
  easy, it means wins are shared.
- **Method ordering in SINGLE** (MLP > LSTM > IPPO) is consistent with
  sample-efficiency expectations at a 200-300k-step budget.

### 6.2 Information-revelation ablation

Masked PPO in DYNAMIC with the team revealed with probability `p` per decision
(`RevealEnv`):

| reveal p | win rate | reward |
|---|---|---|
| 0.00 | 0.847 ± 0.026 | 77.2 |
| 0.25 | 0.850 ± 0.008 | 72.7 |
| 0.50 | 0.845 ± 0.009 | 73.5 |
| 0.75 | 0.827 ± 0.019 | 74.8 |
| 1.00 | 0.793 ± 0.019 | 67.4 |

Against rule-bot opponents, **revealing teammate identity does not help a
short-trained PPO** (flat to slightly decreasing). This mirrors IIGC's
*deceptive stability*: at short horizons, extra information does not translate
into better policies. Longer training and stronger evaluation opponents are
needed to probe whether the information is eventually exploited (§7).

### 6.3 Hidden-relationship robustness (1v3 vs 2v2)

DYNAMIC deals split by red-team size (76% 2v2 / 24% 1v3), MLP PPO:

| seed | 2v2 win rate | 1v3 win rate |
|---|---|---|
| 0 | 0.859 | 0.909 |
| 1 | 0.842 | 0.864 |
| 2 | 0.850 | 0.909 |

Results are **insensitive to team composition**, so conclusions do not depend
on the (intentionally unfiltered) 1v3 deals. The latent relationship to infer
is genuinely *both* team identity and team size.

### 6.4 Learning curves

Learning curves are accumulated per chunk into `*.history.jsonl`
(`paper/figs/learning_curves.png`); the full set will be reported with the
1M-step matrix.

### 6.5 Connection to IIGC

Under hidden teams (DYNAMIC), short-trained PPO exhibits *deceptive stability*:
its evaluation behavior changes little whether or not the team is revealed
(§6.2), and the OBVIOUS ablation (known team) does not yield better policies at
short horizons. This benchmark paper supplies the environment characterization
behind that phenomenon: the four modes, the info-reveal dial, and the baseline
suite make the phenomenon reproducible and the mechanism studyable.

---

## 7 Discussion, Limitations, and Future Work

**Positioning.** 510K is not claimed to be the "hardest card game"; it is
claimed to be a *controllable combination* of individually-studied but rarely
jointly-controlled challenges — a unified testbed rather than a SOTA chase.

**The hidden relationship is a learnable latent.** Unlike social benchmarks
with opaque relationship structure, 510K has ground-truth deal-time teams, so
"did the agent infer the relationship" is measurable (e.g., via behavior
conditioned on the true team).

**Limitations.**

- *Preliminary results.* Current numbers use 200-300k steps and 3 seeds vs rule
  bots; the 1M-step, 5-seed, random- and rule-bot matrix is running.
- *Win rate vs rule bots is coarse.* Team modes share the win with rule-bot
  teammates; finish position and the agent's own score contribution are better
  per-agent signals (planned).
- *The reveal curve is flat at short horizons.* Whether information is
  exploited with more compute / stronger opponents is open.

**Future work.**

- Full 1M-step matrix, random-bot evals, 5 seeds, learning curves, and
  statistical tests.
- **Reward-shaping stability**: the immediate 510K score is a natural dense
  reward that competes with the terminal finish objective — a controlled
  reward-hacking study (Ng et al. 1999; Skalse et al. 2022).
- MAPPO/QMIX (centralized critics), search/planning baselines.
- Human-play interface and LLM-agent evaluation.
- 3-player mode and a trajectory dataset release.

---

## 8 Conclusion

510K couples immediate scoring, terminal competition, partial observability,
and — uniquely — *hidden dynamic cooperation* in a single, controllable card
game. It ships as a clean Gymnasium + PettingZoo environment with action
masking and four ablative modes, plus a reference baseline suite and a prior
phenomenon (IIGC) discovered on it. Preliminary results show SINGLE is hardest
for all methods, team modes inflate win rate, and information revelation does
not help short-trained PPO against rule bots — echoing IIGC's deceptive
stability. 510K is offered to the community as a testbed for the *interaction*
of RL challenges rather than another "hard game".

---

## Reproducibility Statement

- **Environment**: `pip install git+https://github.com/flavieninpekin/510k_env`
  (Gymnasium + PettingZoo AEC, `gymnasium.register('510K-v0')`); 82 tests.
- **Baselines & analysis**: `https://github.com/flavieninpekin/510k_env-analysis`
  (`baselines/`), with fixed seeds, chunked checkpoints, and per-chunk eval
  history.
- **Compute**: one GPU; measured throughput 466 env-steps/s for MaskablePPO.
- **Data**: per-run `*.eval.json` and `*.history.jsonl` committed to the
  analysis repo.

---

## References

- Agapiou, C., et al. *Melting Pot 2.0*. 2022.
- Bard, N., Foerster, J., Chandar, S., et al. *The Hanabi Challenge: A New Frontier for AI Research*. Artificial Intelligence 280, 2020.
- Carroll, M., et al. *On the Utility of Learning about Humans for Human-AI Coordination*. NeurIPS 2019.
- Lanctot, M., et al. *OpenSpiel: A Framework for Reinforcement Learning in Games*. arXiv:1908.09453, 2019.
- Li, J., et al. *Suphx: Mastering Mahjong with Deep Reinforcement Learning*. arXiv:2003.13590, 2020.
- Ng, A., Harada, D., Russell, S. *Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping*. ICML 1999.
- Samvelyan, M., et al. *The StarCraft Multi-Agent Challenge*. AAMAS 2019.
- Silver, D., et al. *Mastering the Game of Strategy with Self-Play and Multi-Agent Reinforcement Learning* (Cicero). Science 2022.
- Skalse, J., Howe, N., Krasheninnikov, D., Krueger, D. *Defining and Characterizing Reward Hacking*. NeurIPS 2022.
- Terry, J., et al. *PettingZoo: Gym for Multi-Agent Reinforcement Learning*. NeurIPS 2021 (Datasets & Benchmarks).
- Zha, D., et al. *RLCard: A Toolkit for Reinforcement Learning in Card Games*. AAAI 2020.
- Zha, D., et al. *DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning*. ICML 2021.
- *Big Two: Self-Play Reinforcement Learning under Imperfect Information*. 2026.
- *MOSMAC: A Multi-agent RL Benchmark on Sequential Multi-objective Tasks*. AAMAS 2025.
- IIGC (prior work): *Information-Induced Gradient Contraction in Partially-Observable MARL*. AAAI 2027.