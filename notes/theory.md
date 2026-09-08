# 510K: Theory Notes

Analytical results on the environment's structure. All propositions are
verified numerically (the machine's flaky runtime required retried runs; final
numbers below come from successful runs).

Notation: 4 players; the red team is the set of players holding ≥1 red ace
(A♥/A♦) — the "black ace counts as red if you hold all four aces" rule is
*redundant* (holding all four aces implies holding a red ace). The agent is
player 0. Let `T ⊆ {1,2,3}` be the set of the agent's teammates.

## Proposition A — team-size prior

**Statement.** Over the deal distribution,
`P(|red team| = 2) = 1 − 4·C(50,11)/C(52,13) ≈ 0.7647`
and `P(|red team| = 1) = 4·C(50,11)/C(52,13) ≈ 0.2353`.

**Proof sketch.** The red team has size 1 iff both red aces land in the same
hand. Fix a player: P(both red aces in that hand) = C(50,11)/C(52,13)
(choose the other 11 cards from the 50 non-red-aces). By symmetry over 4
players, P(size 1) = 4·C(50,11)/C(52,13) ≈ 0.2353.

**Verification** (20,000 deals): empirical 0.7647 / 0.2353 — exact match.

**Consequence.** The "embrace 1v3" design choice (never filter deals) rests on
a *derived*, not empirical, prior: 23.5% of DYNAMIC games are 1v3/3v1. The
latent relationship to infer includes both identity and team *size*.

## Proposition B — agent membership probabilities

**Statement.** `P(agent ∈ red) = 1 − C(50,13)/C(52,13) ≈ 0.4412`;
`P(agent is the only red player) = C(50,11)/C(52,13) ≈ 0.0588`.

**Proof.** The agent's hand is a uniform 13-subset. P(no red ace in the
agent's hand) = C(50,13)/C(52,13). P(agent holds both red aces) =
C(50,11)/C(52,13). Holds-both is exactly "agent is solo red".

**Verification:** empirical 0.447 / 0.059 (6,000 deals).

## Proposition C — the reveal dial is a (nearly) linear information channel

Let `Z ~ Bernoulli(p)`; `RevealEnv` returns `R = T·Z` (team bits with prob p,
else zero-vector). The mutual information between the true teammate set and the
revealed observation is

```
MI(p) = p·H(T) − δ(p),
δ(p) = a·log a − p·p_empty·log p_empty − (1−p_empty)(1−p)·log(1−p),  (log base 2)
a = (1−p) + p·p_empty,   p_empty = P(T = ∅) = C(50,11)/C(52,13) ≈ 0.0588.
```

**Derivation.** `H(R|T) = (1−p_empty)·h(p)` (the empty team has entropy 0
given itself), and `H(R) = h(p) + p·H(T)` only when `p_empty = 0`; otherwise
the "team not revealed" output ∅ collides with the "solo red, revealed" output
∅, so `MI(p) < p·H(T)`. The deficit `δ(p) ≥ 0` is exactly this ambiguity, and
`δ(p) = 0` when `p_empty = 0` (e.g., in a 2v2-only variant).

**Verification** (analytic vs 6,000-deal channel sampling):

| p | MI analytic | MI sampled | ratio to p·H(T) |
|---|---|---|---|
| 0.25 | 0.624 | 0.637 | 0.912 |
| 0.50 | 1.258 | 1.266 | 0.919 |
| 0.75 | 1.915 | 1.931 | 0.933 |
| 1.00 | 2.736 | 2.734 | 1.000 |

**Consequences.**
1. The §6.2 reveal ablation sweeps an *information* axis that is nearly linear
   in p (≥91% of the ideal p·H(T) for all p). A flat performance-vs-p curve
   therefore genuinely means "the information is not being exploited", not
   "the dial was not turned".
2. The small deficit (~7-9%) is itself a signature of the 1v3 deals: the
   solo-red state is informationally hidden even when "revealed". This gives a
   *measurable* cost of the embrace-1v3 design.

## Proposition D — reward dilution explains why team-mode win rate is coarse

**Statement.** In STATIC/DYNAMIC/OBVIOUS, the Scorer's team reward aggregates
the 510K scores of *two* players (the agent and one teammate). Under identical
random policies, the agent's own score and its teammate's score contribute
approximately equally to the score variance, so roughly **half of the reward
variation the agent experiences is produced by a player it does not control**.

**Verification** (1,500 finished STATIC random games): std(own score) = 16.6,
std(teammate score) = 16.3; own share of (own+teammate) std ≈ 0.51.

**Consequence.** This is a structural explanation for two observations:
1. **SINGLE is the cleanest probe** (§6.1): there, the agent's reward is a
   function of its *own* score and finish position only, so method differences
   are not diluted by teammate noise. This is why SINGLE cleanly orders
   MLP > LSTM > IPPO while team modes all sit at 0.75-0.85.
2. **Win rate vs rule bots is insensitive to p** (§6.2): the metric is a
   threshold on a heavily diluted reward; the information gained by revealing
   the team (Prop C) changes the agent's own play, but that is a second-order
   effect compared to teammate-driven variance.

## Open directions

- Relate δ(p) (Prop C) to a *learnability* bound: how much reveal is needed
  before a sample-efficient learner can extract the team signal (link to IIGC's
  κ and the flat §6.2 curve).
- Tighten Prop D to a signal-to-noise theorem: with teammate policy fixed, the
  agent's reward SNR is bounded by 1/(1 + var(teammate)/var(own)), motivating
  agent-own-signal metrics (finish position, own score) for benchmarking.