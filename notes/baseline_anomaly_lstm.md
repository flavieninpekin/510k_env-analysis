# Baseline anomaly: masked-LSTM PPO collapses (investigation)

> 2026-09-16. Found while running the 1M matrices. Status: **diagnosed**.

## Symptom

The three LSTM seeds produce **byte-identical evaluation metrics**:

| mode | seed 0 | seed 1 | seed 2 |
|---|---|---|---|
| dynamic | 0.85 / 74.65 / 19.28 | 0.85 / 74.65 / 19.28 | 0.85 / 74.65 / 19.28 |
| single | 0.49 / 17.40 / 14.42 | 0.49 / 17.40 / 14.42 | 0.49 / 17.40 / 14.42 |

(`win_rate` / `mean_reward` / `mean_len`, 1M steps.)

## Diagnosis (two separate issues)

### 1. Eval-path bug (real; fixed)

`model.predict(obs, deterministic=True)` — as used in `quick_eval` and all
analysis scripts — calls sb3-contrib's
`RecurrentActorCriticPolicy.predict`, which, when `state`/`episode_start` are
not passed, **initializes the LSTM hidden state to zeros on every call** and
sets `episode_start=False`. Recurrence was therefore disabled at evaluation
(the trained memory was never used or reset).

Fixed in `baselines/train_ppo.py::quick_eval`,
`baselines/analyze_1v3.py`, `baselines/analyze_positions.py`,
`baselines/analyze_team_conditioning.py`: carry `state`, pass
`episode_start=np.array([True])` at each `reset`.

**Impact: none on the numbers.** Re-evaluating all six LSTM checkpoints with the
corrected state carry gives the exact same metrics — the policy's behaviour does
not depend on the recurrent state either.

### 2. Root cause: the policy collapses to a degenerate greedy rule

* Checkpoints differ (MD5s differ; parameter `|mean|` = 0.043524 / 0.043540 /
  0.043646, norms 78.01 / 78.03 / 78.21) → not a save collision.
* Yet deterministic action sequences on shared deals are identical across seeds,
  and identical whether the LSTM state is zeroed each step or carried correctly
  (verified live, not from cached JSON).
* Direct logit inspection (`diag_lstm3`): for a sample DYNAMIC state, the
  unmasked logits have **argmax = action 1 (the lowest valid pattern) for all
  three seeds**, with action 2 second; cross-seed logit differences are
  0.5–0.97 but do not change the ranking. The policy has converged to an
  essentially observation- and seed-independent "play the lowest valid pattern"
  rule (a rule-bot-like heuristic), making the recurrent memory irrelevant.

## Consequence for the paper

- LSTM numbers must **not** be reported as a normal baseline; `±0.000` is an
  artifact of collapse, not stability. The `dynamic 0.850` is *not* evidence for
  or against the paper's claims; only MLP/IPPO numbers should carry them.
- MLP and IPPO are unaffected (per-seed variation is real: MLP std
  0.004–0.022, IPPO dynamic 0.008).
- The collapse is itself an interesting **baseline failure mode**: a masked
  recurrent PPO on a long-horizon, partially-observable, dynamically-cooperative
  game degenerates to a memoryless greedy heuristic. Candidate to report (with
  the caveat that it may be implementation-specific).
- Fixed eval path is a strict improvement for any future recurrent runs.

## Variant experiment (2026-09-18): can hyperparameters rescue the collapse?

Two variants, 1M steps, DYNAMIC seed 0 first, then the cross-seed check, all vs
rule bots:

* `e05`: `ent_coef` 0.01 -> 0.05
* `small`: net 128x128, LSTM 128, lr 1e-4

**Win rate (complete 1M runs, per-variant tags)**

| variant | seeds | SINGLE | STATIC | DYNAMIC | OBVIOUS |
|---|---|---|---|---|---|
| e05 | 2 | 0.570 / 0.510 | 0.730 / 0.740 | 0.850 / 0.850 | 0.780 / 0.780 |
| small | 2 | 0.410 / 0.490 | 0.740 / 0.740 | 0.850 / 0.850 | 0.780 / 0.780 |
| original | 3 | 0.490 x3 | - | 0.850 x3 | - |

**Deferral asymmetry** (`P(pass | true partner leads) - P(pass | opponent leads)`)

| variant | STATIC | DYNAMIC | OBVIOUS |
|---|---|---|---|
| e05 s0 / s1 | +0.367 / +0.383 | -0.043 / -0.043 | -0.071 / -0.062 |
| small s0 / s1 | +0.383 / +0.383 | -0.043 / -0.043 | -0.071 / -0.071 |

## Conclusion

1. **STATIC cooperation is learned** by every variant (+0.37~+0.38), matching
   the MLP (+0.38): fixed, known partnership is easy.
2. **DYNAMIC does not recover.** All variants and seeds produce identical
   DYNAMIC win rate (0.85/0.85) and identical asymmetry (-0.043); the hidden
   team remains un-inferred. Raising entropy / shrinking the net does not fix it.
3. **e05 is partially healthier**: its SINGLE (0.540±0.030, beats MLP 0.516) and
   STATIC OBVIOUS asymmetries are seed-diverse, so it is not fully collapsed.
   Yet even e05 is anti-cooperative in OBVIOUS (-0.07/-0.06) and seed-identical
   in DYNAMIC.
4. **Interpretation for the paper**: the cooperation-inference failure is not a
   memoryless/architecture artifact - a seed-diverse recurrent variant that
   learns fixed-partner cooperation *still* fails on hidden/revealed teams. It
   is, however, confounded with the DYNAMIC degeneracy, so report it as
   corroborating evidence, not as the primary claim.
5. **Cost note**: LSTM aggregate throughput is ~330 env-steps/s regardless of
   concurrency (GPU-bound); the full 32-run matrix would take ~27h, so the batch
   was trimmed to 2 seeds/variant after the DYNAMIC answer was already fixed.

## Remaining follow-ups (optional)

1. Fix the custom `MaskedLstmActorCriticPolicy` (or use a history-augmented MLP
   observation) so the DYNAMIC policy is non-degenerate, then re-test inference.
2. Compare against plain (unmasked) `RecurrentPPO` + valid-action wrapper.
3. Report the DYNAMIC collapse itself as an observed baseline failure mode
   (with the implementation-specific caveat).

