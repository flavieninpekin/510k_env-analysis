# 510K: A Card-Game Testbed for Dynamic Cooperation, Partial Observability, and Information Revelation in Multi-Agent RL

**Draft v0.4** — ICLR 2027 benchmark-style submission. The masked-PPO 1M ×
5-seed matrix, the full information-reveal curve, and IPPO are complete;
statistical tests are integrated; a LaTeX manuscript (`paper/main.tex`,
`paper/main.pdf`) is available. A masked-LSTM collapse is flagged in §7.

---

## Abstract

Existing multi-agent game benchmarks typically isolate individual challenges:
immediate scoring, terminal competitive objectives, fixed cooperation, or
partial observability. We present **510K**, a four-player shedding card game
that *couples* these challenges in a single decision process: the axes are not
merely stacked but alter one another's value, and the cooperative relationship
itself is a hidden latent variable — team membership — and even team size (2v2
vs 1v3) — is determined at deal time by a private red-A distribution and must be
inferred from observed play. 510K ships with four controllable modes (SINGLE,
STATIC, DYNAMIC, OBVIOUS) that ablate the cooperation/information axes, a
Gymnasium single-agent and PettingZoo AEC interface with built-in action
masking, and a prior scientific result (Information-Induced Gradient
Contraction, IIGC) already discovered on it. We characterize the coupling with
three measurements: (i) structural probes show the immediate-score and
terminal-finish objectives are in genuine tension, and that the hidden team
leaves a large non-attributable share of reward variance; (ii) a 1M-step,
5-seed masked-PPO suite shows SINGLE is hardest for every method (win rate ≈
chance), while team-mode win rates remain inflated by teammate and
team-selection effects; (iii) a behavioural probe finds policies learn to
cooperate with a *fixed, known* partner (deferral asymmetry +0.38) but fail to
infer a *hidden* partner (+0.00, null) and do not exploit one that is
*explicitly revealed* (−0.04, with no performance gain) — a direct echo of
IIGC's *deceptive stability*. 510K is positioned as a testbed in which
individually-studied challenges are jointly controlled and ablatable.

---

## 1 Introduction

A central question in multi-agent reinforcement learning (MARL) is how agents
should behave when several objectives interact and part of the task structure is
hidden. Card games are a natural home for this question, yet existing benchmarks
tend to isolate the difficulties one at a time. Dou Dizhu (DouZero; Zha et al.,
ICML 2021) couples competition with *fixed* cooperation and a sparse terminal
reward. Hanabi (Bard et al., AIJ 2020) is purely cooperative, with neither
competition nor an immediate scoring trade-off. Mahjong (Suphx; Li et al., 2020)
has scoring and partial observability but no teams. Tichu, Hearts and Spades
have scoring and partnerships, but the partnerships are *fixed and known*. The
picture that emerges is fragmented: we can measure how an algorithm handles
scoring, or hidden information, or cooperation, but not how it handles them
*together*.

This gap matters because in realistic tasks these aspects are not independent:
acting on one changes the value of the others. An agent that makes a test pass
now may damage the architecture later; an agent interacting with others must
infer who is aligned with it and whether that alignment persists. Benchmarks
that stack independent switches cannot expose such interactions, because their
optimal values decompose.

We introduce **510K**, a four-player shedding card game designed so that they
do not. 510K couples (i) *immediate scoring* (5/10/K cards collected in won
tricks), (ii) a *terminal competitive* objective (finish first), (iii) *partial
observability*, and (iv) *dynamic cooperation*: team membership is determined
at deal time by the private red-A distribution and is hidden, so "who should I
help?" becomes a latent-variable inference problem rather than a given.
Crucially, the axes are not stacked but entangled — a single play changes the
immediate score, the future pattern space, the information revealed about
relationships, and the race to finish. We make this precise by formalizing
coupling as *non-separability* and by giving each coupling a measurable
signature rather than a rhetorical one (Section 4, Section 6.3).

The environment ships with four controllable modes (SINGLE, STATIC, DYNAMIC,
OBVIOUS) that ablate the cooperation and information axes, a graded
information-reveal dial, Gymnasium and PettingZoo AEC interfaces with built-in
action masking, and a reference baseline suite. On it we report a capability
profile and a behavioural result: policies learn to cooperate with a fixed,
known partner but neither infer a hidden partner nor exploit an explicitly
revealed one — an information-utilization failure that mirrors the previously
reported IIGC phenomenon (AAAI 2027).

Our contributions:

1. **A complete, tested environment.** Gymnasium single-agent and PettingZoo
   AEC interfaces with built-in action masking, four controllable modes, a
   deterministic rule bot, and 82 passing tests.
2. **A challenge decomposition and controlled variants** in which scoring,
   terminal competition, partial observability, and dynamic cooperation are
   independently ablatable.
3. **A coupling characterization.** We formalize "coupled, not stacked" as
   non-separability and give measurable structural signatures for the
   objective–objective, objective–cooperation, and cooperation–observability
   couplings.
4. **A baseline suite and evaluation protocol.** Masked PPO, recurrent masked
   PPO, and independent (shared-policy) PPO, evaluated identically as player 0
   against fixed bots, with masking compliance as a correctness check and 95%
   confidence intervals plus significance tests for all comparisons.
5. **A capability profile and a cooperation-inference result.** At 1M steps and
   5 seeds SINGLE is hardest for every method, team-mode win rates are inflated
   by teammate and team-selection effects, and a behavioural probe shows
   policies cooperate with a *fixed, known* partner but neither infer nor
   exploit a *hidden/revealed* one.

---

## 2 Related Work

### 2.1 Card-game RL toolkits

- **RLCard** (Zha et al., AAAI 2020) provides many card games behind a common
  interface and is the most widely used toolkit in this space; each included
  game, however, isolates a subset of challenges, and none exposes a hidden or
  dynamically determined team assignment.
- **OpenSpiel** (Lanctot et al., 2019) offers game-theoretic tooling over 50+
  games, oriented toward solving extensive-form games rather than RL under
  sparse rewards.
- **PyTAG** (2023) covers 20+ tabletop games with a modern API but shallow
  per-game depth.
- These toolkits make card games easy to run; they do not provide a controlled
  instrument for studying *interactions* between challenges.

### 2.2 Shedding-type and scoring card games

- **DouZero** (Zha et al., ICML 2021) established that a shedding-type game with
  a fixed landlord role and sparse terminal reward is a serious benchmark:
  "long horizons and sparse reward ... the only time a nonzero reward is
  incurred is at the end of a game." 510K keeps that long-horizon, sparse
  structure and adds **hidden, deal-time-determined teams** and an
  **immediate-scoring** objective that competes with the terminal finish
  objective.
- **Tichu, Hearts, Spades** have scoring and partnerships, but those
  partnerships are fixed and known, so the relationship itself is never
  inferred.

### 2.3 Cooperative and imperfect-information benchmarks

- **Hanabi** (Bard et al., AIJ 2020) is the canonical cooperative
  imperfect-information benchmark — agents cannot see their own cards — but it
  has no competition and no scoring trade-off.
- **Overcooked-AI** (Carroll et al., NeurIPS 2019) probes ad-hoc human–AI
  coordination under full observability.
- **Melting Pot** (Agapiou et al., 2022) studies mixed-motive social
  interaction at a general substrate level, not within a compact, fully
  specified decision problem.
- None of these supplies a small, ground-truth, trainable instance in which the
  relationship between agents must be inferred.

### 2.4 Multi-objective and hidden-role settings

- Multi-objective RL and planning methods tackle trade-offs between conflicting
  objectives (Hayes et al., AAMAS 2022); 510K is a card-game instance with
  exactly two such objectives (immediate score and terminal finish) whose
  tension is measurable.
- **Diplomacy/Cicero** (Silver et al., Science 2022) and hidden-role LLM studies
  (Avalon, Werewolf) demonstrate strong interest in relationship inference at a
  much larger scale, but lack a lightweight RL testbed with ground-truth team
  assignments. 510K fills that gap: the latent relationship is known to the
  simulator, so "did the agent infer it?" is directly measurable.

### 2.5 Prior results on 510K

- **IIGC** (AAAI 2027) studies a *phenomenon* on this environment: under hidden
  team assignment, policy-gradient updates from different relationships cancel
  in expectation, producing *deceptive stability* (the policy appears to stop
  learning without converging). The present paper is deliberately complementary:
  IIGC asks *why* a learner stalls; we characterize the *environment* and
  provide the controlled ablations and behavioural probes that make the
  phenomenon reproducible and the mechanism studyable.

### 2.6 Positioning

| Challenge | 510K | DouDizhu | Hanabi | Mahjong | Tichu | Overcooked | SMAC |
|---|---|---|---|---|---|---|---|
| Immediate scoring | ● | ○ | ○ | ● | ● | ○ | ○ |
| Terminal competition | ● | ● | ○ | ● | ● | ○ | ● |
| Partial observability | ● | ● | ● | ● | ◐ | ○ | ◐ |
| Team cooperation | ● hidden | ● fixed | ● all | ○ | ● fixed | ● | ◐ |
| **Hidden/dynamic teams** | **●** | ○ | ○ | ○ | ○ | ○ | ○ |
| Controlled ablation | ● | ○ | ○ | ○ | ○ | ◐ | ◐ |

● strong / present, ◐ partial, ○ absent. Only 510K couples immediate scoring ×
terminal competition × partial observability × **hidden dynamic teams** ×
controllable ablation in one game. The claim is about this combination and its
controllability, not about being the hardest card game: each individual
challenge is studied elsewhere, and our contribution is to make them jointly
controllable and measurable.

---

## 3 The 510K Environment

### 3.1 Rules

510K is a shedding-type card game played with a standard 52-card deck (4
players) or 54 cards including jokers (3 players). Each player is dealt 13 (or
18) cards. On a turn a player must either play a combination that beats the
current trick or pass; when all other active players pass, the trick ends and
its accumulated **score** is awarded to the trick leader, who opens the next
trick. The result is a deliberately mixed reward structure: a dense, immediately
observable score signal layered on a sparse terminal objective.

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
configuration: the red team has size 2 with probability
`1 − 4·C(50,11)/C(52,13) ≈ 0.765` and size 1 otherwise — a closed-form
hypergeometric fact verified exactly (§6.8). We treat the solo case as a
feature: the latent variable to infer is not only *who* is my partner but
*whether I even have one*. Because the team is hidden, the reward function
itself is latent: the same observation–action pair can map to different returns
depending on the deal.

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

### 4.1 Coupling is non-separability, not stacking

A benchmark *stacks* challenges when a difficulty (or the optimal value)
decomposes additively across axes, so each can be studied in isolation. 510K is
designed so that it does not. Let the axes be scoring (S), terminal competition
(T), partial observability (P) and hidden cooperation (C). The couplings we
deliberately instantiate are:

- **C1 objective entanglement (S×T).** A 5/10/K play moves the immediate score
  and the race to finish *in tension*: securing points means keeping control of
  the trick (delaying one's finish), while shedding fast concedes points.
  Signature: rank correlation between a player's collected score and finish
  position.
- **C2 cooperation × observability (C×P).** In DYNAMIC the reward is indexed by
  the latent team `T`; two histories that look identical can induce different
  reward mappings. Partial observability is therefore *constitutive of* the
  cooperation structure, not additive noise.
- **C3 objective × cooperation (S×C).** The same scoring objective has a
  different payoff depending on team structure: in SINGLE the agent's score
  enters its reward directly; in team modes it is aggregated with a teammate's
  and multiplied by the win/lose factor.
- **C4 information × cooperation × state.** The value of revealing the team is
  not a constant: it depends on whether the agent has a partner at all (1v3 vs
  2v2) and on the current terminal race. `RevealEnv` turns this into a graded
  dial.

Because every axis is a code-level switch and the full state is available as
ground truth, each coupling has a *measurable* signature (Section 6.3) rather
than a rhetorical one — which is what distinguishes a controllable instrument
from a merely complicated game.

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
  observation and apply it inside the policy). Its evaluation collapses in
  DYNAMIC/OBVIOUS; see §6.4 and §7.
- **Independent PPO (IPPO)** — a shared-policy self-play baseline
  (`SelfPlayEnv`): one policy controls all seats (parameter-sharing IPPO).

**Evaluation protocol.** Every policy is evaluated identically: the trained
policy plays **player 0 against fixed opponents** (random bot, rule bot) for
seeded games. Metrics: win rate, mean total reward (mean ± 95% CI), mean episode
length, and **illegal-action rate** (masking compliance).

**Protocol.** All runs are 1M steps in modes SINGLE/STATIC/DYNAMIC/OBVIOUS:
masked PPO with 5 seeds, IPPO with 3, recurrent PPO with 2. Capability-profile
analysis: which method collapses under (hidden team × partial observability).
1v3 robustness: main results on all deals; a 2v2-only split verifies conclusions
do not depend on the filtering choice.

**Coupling and cooperation-inference probes.** Structural coupling signatures
are measured with fixed bots (rule / random) driving all four seats
(`coupling_probes.py`): a score–finish rank correlation (C1) and a
reward-attribution decomposition (C2/C3). Strategy-level behaviour is probed
with trained policies (`analyze_team_conditioning.py`): on every following
decision we record whether the agent passes and whether the leader is its
*true* partner, yielding a deferral asymmetry
`asym = P(pass | partner leads) − P(pass | opponent leads)`; `asym > 0` is a
behavioural signature of learned cooperation. `analyze_positions.py` provides
per-agent metrics (finish position, own/teammate score) that avoid the team
win-rate dilution (Prop D).

**Statistical analysis.** Cells are reported as mean ± 95% CI (t-based,
ddof=1). We use Welch t-tests for algorithm comparisons (MLP vs IPPO), paired
t-tests for within-checkpoint comparisons (DYNAMIC vs OBVIOUS; rule- vs
random-bot evaluation), a per-seed linear slope of win rate on reveal
probability `p` with a one-sample t-test, and two-proportion z-tests for the
deferral asymmetries. Full report: `runs/stats_report.md` (`baselines/stats.py`).

**Crash tolerance.** Training is chunked into subprocesses (`run_matrix.py`);
each chunk saves a checkpoint and is restarted on failure, so long runs are
robust even on a flaky runtime.

**Compute budget.** Measured (single GPU, CUDA): MaskablePPO ≈ 500 env-steps/s,
i.e. ≈40 min per 1M-step run. The 4-mode × 5-seed masked-PPO matrix therefore
costs ≈13 GPU-hours; LSTM ≈2× slower, IPPO ≈1.5×.

---

## 6 Results

Core results use masked PPO trained for 1M steps with 5 seeds and evaluated as
player 0 against rule bots. All policies have **illegal-action rate 0.0** —
masking is respected exactly.

### 6.1 Capability profile across modes and methods

Win rate (fraction of episodes with positive team reward), mean ± 95% CI (all
1M steps). Recurrent is the `ent_coef`=0.05 variant (2 seeds, no CI).

| mode | MLP PPO (5 seeds) | recurrent PPO (2)† | IPPO (3) |
|---|---|---|---|
| SINGLE | 0.516 ± 0.029 | 0.540 | 0.300 ± 0.114 |
| STATIC | 0.774 ± 0.031 | 0.735 | 0.737 ± 0.094 |
| DYNAMIC | 0.862 ± 0.016 | 0.850† | 0.810 ± 0.025 |
| OBVIOUS | 0.822 ± 0.006 | 0.780† | 0.807 ± 0.080 |

†Higher-entropy recurrent variant (`ent_coef`=0.05), the healthier of two
masked-LSTM variants tested. Its DYNAMIC/OBVIOUS checkpoints remain
seed-degenerate (identical across seeds) while SINGLE/STATIC are seed-diverse,
so the recurrent column is reported with a caveat rather than as a clean
baseline; see §6.4 and §7.

Reading:

- **SINGLE is the hardest setting** for every method: no teammate, MLP win rate
  ≈ chance (0.516 ± 0.029, unchanged from 300k to 1M), and IPPO collapses to
  0.300 ± 0.114 — *below* chance.
- **MLP > IPPO in all four modes**, with the largest gap in SINGLE (+0.216);
  the gap is significant in SINGLE (Welch p=0.007) and DYNAMIC (p=0.001) but not
  in STATIC (p=0.22) or OBVIOUS (p=0.50), where team-mode dilution dominates.
  This is the expected ordering once the per-agent signal is cleanest (Prop D).
- **Team-mode win rates are inflated by two effects** that make cross-mode
  comparison unsafe: (i) a teammate's play contributes to the team outcome; and
  (ii) DYNAMIC/OBVIOUS teams are *selected* by holding a red ace, so the agent's
  team tends to hold stronger cards. We therefore treat per-agent and
  behavioural metrics (Sections 6.3-6.4), not team win rate, as the primary
  evidence; Prop D gives the formal dilution argument.
- **DYNAMIC is significantly above OBVIOUS** (0.862 vs 0.822, paired
  t=8.94, p=0.0009), but the difference is small and opponent-dependent: it
  reverses against random bots (0.948 vs 0.958, §6.1). We therefore treat the
  behavioural result (§6.4) as the robust one.

**Opponent robustness.** Re-evaluating the same rule-trained checkpoints
against random bots (`evaluate_opponents.py`) preserves the profile's ordering:
SINGLE stays hardest for both methods and MLP > IPPO in every mode.

| mode | MLP rule | MLP random | IPPO rule | IPPO random |
|---|---|---|---|---|
| SINGLE | 0.516 | 0.764 | 0.300 | 0.450 |
| STATIC | 0.774 | 0.950 | 0.737 | 0.867 |
| DYNAMIC | 0.862 | 0.948 | 0.810 | 0.907 |
| OBVIOUS | 0.822 | 0.958 | 0.807 | 0.950 |

Team modes saturate near ceiling against random bots (0.87-0.96), so the random
bot is too weak to discriminate there; the rule bot is the more informative
opponent, and the random-bot column is reported as a robustness check rather
than as a second capability profile.

### 6.2 Information-revelation ablation

Masked PPO with the team revealed with probability `p` per decision
(`RevealEnv`), all at 1M steps × 5 seeds:

| reveal p | win rate | reward |
|---|---|---|
| 0.00 (DYNAMIC) | 0.862 ± 0.015 | 78.0 ± 5.0 |
| 0.25 | 0.838 ± 0.020 | 79.3 ± 4.3 |
| 0.50 | 0.838 ± 0.019 | 76.9 ± 3.4 |
| 0.75 | 0.844 ± 0.029 | 77.7 ± 3.4 |
| 1.00 (OBVIOUS) | 0.822 ± 0.005 | 69.6 ± 0.1 |

Against rule-bot opponents the curve is flat to slightly decreasing — never
better than the fully hidden `p=0` baseline. A per-seed linear fit gives a mean
slope of −0.030 per unit `p` (one-sample t=−6.12, p=0.0036, n=5), and the paired
`p=0` vs `p=1` difference is +0.040 (p=0.0009): at 1M steps, revealing the team
significantly *hurts* rather than helps. Prop C shows the dial is a
nearly-linear information channel (≥91% of the ideal `p·H(T)`), so this is a
genuine *information-utilization* failure, not an unturned dial. Against random
bots the `p=0`/`p=1` difference reverses and is within noise (§6.1), so the
opponent-robust claim is the behavioural one (§6.4), not the win-rate curve.

### 6.3 Structural coupling signatures (C1-C3)

Fixed-bot probes (`coupling_probes.py`) quantify the couplings of Section 4.1
over 500 games/mode. `top1st` = fraction of games whose top scorer finishes
first (4-player chance 0.25, binomial test); `rho` = within-game rank
correlation between a player's score and finish position (mean ± 95% CI over
games, one-sample t-test); `dR/d(own)` = OLS slope of reward on own score;
`R²` = reward variance explained by own+teammate scores (bootstrap 95% CI).

Rule bot:

| mode | top1st | rho (95% CI) | dR/d(own) | corr(own,R) | R² (95% CI) |
|---|---|---|---|---|---|
| SINGLE | 0.596 | −0.271 ± 0.043 | 1.51 | 0.893 | 0.797 [0.78, 0.82] |
| STATIC | 0.592 | −0.546 ± 0.038 | 2.34 | 0.587 | 0.844 [0.82, 0.87] |
| DYNAMIC | 0.604 | −0.565 ± 0.036 | 2.16 | 0.569 | 0.337 [0.30, 0.37] |
| OBVIOUS | 0.374 | −0.364 ± 0.046 | 2.23 | 0.588 | 0.356 [0.32, 0.39] |

Random bot:

| mode | top1st | rho (95% CI) | dR/d(own) | corr(own,R) | R² (95% CI) |
|---|---|---|---|---|---|
| SINGLE | 0.406 | −0.151 ± 0.047 | 1.29 | 0.799 | 0.638 [0.61, 0.66] |
| STATIC | 0.384 | −0.252 ± 0.048 | 1.80 | 0.476 | 0.669 [0.63, 0.71] |
| DYNAMIC | 0.392 | −0.271 ± 0.049 | 1.64 | 0.442 | 0.200 [0.17, 0.23] |
| OBVIOUS | 0.392 | −0.271 ± 0.049 | 1.64 | 0.442 | 0.200 [0.17, 0.24] |

Under both bots, every `rho` is significantly negative and every `top1st`
significantly above 0.25 (all p<1e-9).

Reading:

- **C1 holds:** `rho < 0` under both bots and all modes — collecting points and
  finishing first genuinely trade off; the top scorer finishes first well above
  chance but far from always.
- **C3 holds:** `corr(own score, reward)` drops from ≈0.8 in SINGLE to ≈0.4-0.6
  in team modes — the same scoring action carries a different incentive
  depending on the cooperative structure.
- **C2 holds:** in the hidden-team modes only 0.20-0.36 of reward variance is
  attributable to own+teammate scores, versus 0.64-0.84 in SINGLE/STATIC. The
  SINGLE and DYNAMIC bootstrap CIs do not overlap, so the latent relationship
  injects reward variance the agent cannot attribute without inference.

### 6.4 Can a policy condition on the hidden team? (C2/C4)

The strategy-level counterpart to Section 6.3. On every following decision we
measure the deferral asymmetry
`asym = P(pass | true partner leads) − P(pass | opponent leads)`; `asym > 0`
means the policy behaves differently toward its true partner. Masked PPO, 1M
steps, vs rule bots (MLP: 5 seeds, 1000 games/seed; recurrent variant: 2 seeds,
500 games/seed):

| mode | MLP (5 seeds) | recurrent e05 (2) |
|---|---|---|
| STATIC | **+0.379** (p<1e-4) | **+0.375** |
| DYNAMIC | **+0.003** (p=0.21) | **−0.043** |
| OBVIOUS | **−0.037** (p<1e-4) | **−0.066** |

p-values are one-sample t-tests of the per-seed asymmetries against zero; the
per-seed two-proportion z-tests (partner vs opponent pass rates, n≈16k
decisions/seed) carry the same significance.

Reading:

- **A fixed, known partner is learned.** In STATIC the policy defers heavily to
  seat 2 (`asym ≈ +0.38`), so the architecture and algorithm *can* learn
  cooperation.
- **A hidden partner is not inferred.** In DYNAMIC `asym = +0.003`
  (p=0.21, i.e. a null): the policy's behaviour does not depend on the true
  partnership at all.
- **Merely revealing the partner does not help.** In OBVIOUS, where the team is
  in the observation, `asym` is significantly *negative* (−0.037, p<1e-4) and
  the win rate is *lower* than DYNAMIC. Four of the five OBVIOUS seeds produce
  identical decision sequences on the shared evaluation deals — a signature of a
  policy that has collapsed to ignoring the team bits.
- This chain — cooperate-when-told, fail-when-hidden, ignore-when-revealed — is
  only observable because 510K exposes all three cooperation regimes with
  ground-truth teams in one environment; it is the behavioural face of C2/C4 and
  the mechanism behind the information-utilization failure of §6.2.

**Recurrent corroboration.** A natural objection is that the MLP is memoryless,
so it *cannot* infer the hidden relationship. We therefore trained two
masked-LSTM variants (higher entropy; smaller net + lower lr) for 1M steps,
2 seeds. The healthiest (`ent_coef`=0.05) is seed-diverse in SINGLE
(0.540 ± 0.030, above the MLP) and STATIC and learns fixed-partner cooperation
(asym +0.37/+0.38, matching the MLP), yet it still shows no cooperation under
hidden or revealed teams (DYNAMIC asym −0.043, seed-identical; OBVIOUS
−0.07/−0.06). The cooperation-inference failure is therefore not simply an
artifact of a memoryless policy. Caveat: the DYNAMIC checkpoints remain
seed-degenerate, so we treat this as corroborating rather than definitive
evidence (`notes/baseline_anomaly_lstm.md`).

### 6.5 Hidden-relationship robustness (1v3 vs 2v2)

DYNAMIC deals split by red-team size (76% 2v2 / 24% 1v3), MLP PPO:

| seed | 2v2 win rate | 1v3 win rate |
|---|---|---|
| 0 | 0.820 | 0.897 |
| 1 | 0.852 | 0.931 |
| 2 | 0.854 | 0.853 |
| 3 | 0.846 | 0.871 |
| 4 | 0.828 | 0.905 |

Averaged over the 5 seeds: 2v2 0.840, 1v3 0.891 (deal split 384/116 ≈
76.8%/23.2%, matching Prop A). Results are **insensitive to team composition**,
so conclusions do not depend on the (intentionally unfiltered) 1v3 deals. The
latent relationship to infer is genuinely *both* team identity and team size.

### 6.6 Learning curves

Per-chunk evaluation history (one point per 100k steps, `*.history.jsonl`;
figure `learning_curves_modes.png`) shows that every mode plateaus early: the
mean slope over the last three chunks is not different from zero in any mode
(|slope| ≤ 0.004/chunk, all p>0.24, n=5 seeds).

| chunk (k steps) | SINGLE | STATIC | DYNAMIC | OBVIOUS |
|---|---|---|---|---|
| 100 | 0.442 | 0.754 | 0.866 | 0.780 |
| 300 | 0.494 | 0.762 | 0.836 | 0.796 |
| 500 | 0.514 | 0.764 | 0.832 | 0.806 |
| 700 | 0.518 | 0.768 | 0.876 | 0.810 |
| 900 | 0.506 | 0.774 | 0.844 | 0.824 |
| 1000 | 0.516 | 0.774 | 0.862 | 0.822 |

SINGLE never leaves chance (final 0.516 vs 0.5, p=0.195). DYNAMIC and OBVIOUS
are statistically flat by 300k already (final vs 300k, paired p=0.13 and 0.07),
so the information-utilization failure of §6.2 is not a "not yet trained"
artifact: the policies have converged and still do not exploit the reveal dial.

### 6.7 Connection to IIGC

Under hidden teams (DYNAMIC), PPO exhibits *deceptive stability*: at 1M steps
its evaluation performance is unchanged whether or not the team is revealed
(§6.2), and the behavioural probe shows it never conditions on the true partner
(§6.4) — while the OBVIOUS ablation (known team) does not help and slightly
hurts. This benchmark paper supplies the environment characterization behind
that phenomenon: the four modes, the info-reveal dial, and the baseline suite
make the phenomenon reproducible and the mechanism studyable.

### 6.8 Theoretical grounding

Four analytical results (proofs and verification in `notes/theory.md`) tie the
environment's *design* and the *observed* results together:

1. **Team-size prior (design).** The red team has size 2 with probability
   `1 − 4·C(50,11)/C(52,13) ≈ 0.765` and size 1 with the remainder — a derived
   hypergeometric fact (verified exactly over 20k deals), not an empirical
   artifact. The "embrace 1v3" design choice therefore rests on a closed-form
   prior, and the latent relationship includes both identity and size.

2. **Membership probabilities.** `P(agent ∈ red) ≈ 0.441`, and
   `P(agent is the solo red player) ≈ 0.059`.

3. **The reveal dial is a nearly-linear information channel.** `RevealEnv`
   implements `R = T·Z`, `Z~Bernoulli(p)`, whose mutual information is
   `MI(p) = p·H(T) − δ(p)`, with `δ(p) = 0` iff no deal is 1v3. Measured:
   `MI(0.25)=0.62, MI(0.5)=1.26, MI(0.75)=1.91` (≥91% of the ideal `p·H(T)`).
   Hence the §6.2 curve is an information axis that *was* swept: at 1M the agent
   still fails to exploit the (large) provided information, and §6.4 shows it
   does not condition on the team even when revealed. The small deficit δ(p) is
   itself a signature of the hidden solo-red state.

4. **Reward dilution explains coarse win rates.** In team modes the Scorer
   aggregates two players' scores; under fixed policies the teammate's play
   contributes ~half the reward variance (measured own/std share ≈ 0.51).
   This is why SINGLE is the cleanest probe (§6.1 orders methods there) and why
   win rate vs rule bots is a threshold on a heavily diluted signal (§6.2).

---

## 7 Discussion, Limitations, and Future Work

**Positioning.** 510K is not claimed to be the "hardest card game"; it is
claimed to be a *controllable combination* of individually-studied but rarely
jointly-controlled challenges — a unified testbed rather than a SOTA chase.

**The hidden relationship is a learnable latent.** Unlike social benchmarks
with opaque relationship structure, 510K has ground-truth deal-time teams, so
"did the agent infer the relationship" is measurable; Section 6.4 realizes this
with a behavioural deferral asymmetry and finds that current PPO does *not*
condition on the true partner.

**Limitations.**

- *Coverage.* The masked-PPO matrix and the full reveal curve are complete at
  1M × 5 seeds, and IPPO is complete at 1M × 3 seeds in all four modes; the
  recurrent baselines cover all modes at 2 seeds but remain partly degenerate
  (§6.4). Opponents are rule/random bots (no self-play or search).
- *Recurrent baseline collapse.* Masked-LSTM seeds collapse to a seed- and
  state-independent "play-lowest-valid" policy; an eval-path bug (the LSTM state
  was never carried at evaluation) was found and fixed, and higher-entropy /
  smaller-net variants rescued seed diversity in SINGLE and STATIC but not in
  DYNAMIC/OBVIOUS. The recurrent column is therefore reported with a caveat, not
  used as a clean baseline (`notes/baseline_anomaly_lstm.md`).
- *Win rate vs rule bots is coarse.* Team modes share the win with rule-bot
  teammates and select stronger teams; per-agent metrics
  (`analyze_positions.py`) are the intended primary signals.
- *The information-utilization failure is a negative result.* At 1M, revealing
  the team still does not help, but whether a stronger opponent or a different
  algorithm eventually exploits the information is open.

**Future work.**

- LSTM/IPPO at 1M, random-bot evals, statistical tests, and learning curves.
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
phenomenon (IIGC) discovered on it. We formalize the coupling as
non-separability and give measurable structural and behavioural signatures: the
scoring and finish objectives trade off, the hidden team injects
non-attributable reward variance, and a 1M-step policy learns to cooperate with
a fixed known partner (+0.38 deferral asymmetry) but neither infers a hidden
one (0.00) nor exploits an explicitly revealed one (−0.04, with no performance
gain) — echoing IIGC's deceptive stability. 510K is offered to the community as
a testbed for the *interaction* of RL challenges rather than another "hard
game".

---

## Reproducibility Statement

- **Environment**: `pip install git+https://github.com/flavieninpekin/510k_env`
  (Gymnasium + PettingZoo AEC, `gymnasium.register('510K-v0')`); 82 tests.
- **Baselines & analysis**: `https://github.com/flavieninpekin/510k_env-analysis`
  (`baselines/`), with fixed seeds, chunked checkpoints, and per-chunk eval
  history.
- **Compute**: one GPU; measured throughput ≈500 env-steps/s for MaskablePPO
  (≈40 min per 1M-step run; the 4-mode × 5-seed matrix ≈13 GPU-hours).
- **Data**: per-run `*.eval.json` and `*.history.jsonl`, plus the coupling /
  positions / team-conditioning analyses (`coupling_probes_*.json`,
  `*.positions.json`, `team_conditioning_*.json`) and the statistical report
  (`runs/stats_report.md`, `baselines/stats.py`), committed to the analysis
  repo.

---

## References

- Agapiou, C., et al. *Melting Pot 2.0*. arXiv:2211.13746, 2022.
- Balla, M., Long, G. E. M., Jeurissen, D., Goodman, J., Gaina, R. D., Perez-Liebana, D. *PyTAG: Challenges and Opportunities for Reinforcement Learning in Tabletop Games*. IEEE CoG 2023.
- Bard, N., Foerster, J., Chandar, S., et al. *The Hanabi Challenge: A New Frontier for AI Research*. Artificial Intelligence 280, 2020.
- Carroll, M., et al. *On the Utility of Learning about Humans for Human-AI Coordination*. NeurIPS 2019.
- Hayes, C. F., Rădulescu, R., Bargiacchi, E., et al. *A Practical Guide to Multi-Objective Reinforcement Learning and Planning*. Autonomous Agents and Multi-Agent Systems 36(26), 2022.
- Lanctot, M., et al. *OpenSpiel: A Framework for Reinforcement Learning in Games*. arXiv:1908.09453, 2019.
- Li, J., et al. *Suphx: Mastering Mahjong with Deep Reinforcement Learning*. arXiv:2003.13590, 2020.
- Meta Fundamental AI Research Diplomacy Team (FAIR), Bakhtin, A., Brown, N., Dinan, E., et al. *Human-level Play in the Game of Diplomacy by Combining Language Models with Strategic Reasoning*. Science 378(6624):1067–1074, 2022.
- Ng, A. Y., Harada, D., Russell, S. *Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping*. ICML 1999.
- Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., Dormann, N. *Stable-Baselines3: Reliable Reinforcement Learning Implementations*. JMLR 22(268):1–8, 2021.
- Samvelyan, M., et al. *The StarCraft Multi-Agent Challenge*. AAMAS 2019.
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. *Proximal Policy Optimization Algorithms*. arXiv:1707.06347, 2017.
- Skalse, J., Howe, N. H. R., Krasheninnikov, D., Krueger, D. *Defining and Characterizing Reward Hacking*. NeurIPS 2022.
- Terry, J. K., et al. *PettingZoo: Gym for Multi-Agent Reinforcement Learning*. NeurIPS 2021 (Datasets & Benchmarks).
- Towers, M., et al. *Gymnasium: A Standard Interface for Reinforcement Learning Environments*. arXiv:2407.17032, 2024.
- Zha, D., Lai, K.-H., Cao, Y., et al. *RLCard: A Toolkit for Reinforcement Learning in Card Games*. AAAI-20 Workshop on Reinforcement Learning in Games, 2020.
- Zha, D., Xie, J., Ma, W., et al. *DouZero: Mastering DouDizhu with Self-Play Deep Reinforcement Learning*. ICML 2021.
- IIGC (prior work): *Information-Induced Gradient Contraction in Partially-Observable MARL*. AAAI 2027.