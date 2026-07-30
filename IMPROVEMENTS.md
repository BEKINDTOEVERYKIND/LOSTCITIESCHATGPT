# Correctness and strength work

This repository was seeded from
`BEKINDTOEVERYKIND/LostCities@4df68f7b2cbda7bd9ee160618693f6436b29e9ec`.
The upstream repository was not modified.

## Why the old policy factorization was restrictive

The old policy used:

```text
logit(action, draw) = play_or_discard[action] + draw_source[draw]
```

Both pieces appear in the move logit, but there is no interaction between
them. For two actions `A` and `B` and draw sources `deck` and `Yellow`:

```text
logit(A, Yellow) - logit(A, deck) = draw[Yellow] - draw[deck]
logit(B, Yellow) - logit(B, deck) = draw[Yellow] - draw[deck]
```

The relative draw preference is therefore forced to be identical for every
card and disposition in the same state. Version 6 retains those shared terms
and adds a learned complete-move residual:

```text
logit(action, draw) =
    play_or_discard[action] + draw_source[draw] + interaction[action, draw]
```

This preserves the useful statistical sharing while permitting combinations
such as “play this card and take Yellow” to differ from “discard this card and
take Yellow.”

## Implemented

### Search and engine correctness

- Fully initialize every MCTS node, including the root `visits` count. The
  former implementation read uninitialized stack memory inside PUCT.
- At a real final-round terminal, MCTS now includes the carried cumulative
  score and uses the checkpoint's hybrid match return. It can no longer call a
  won match a loss merely because the last round was lost narrowly.
- Harden weighted sampling so zero, negative, NaN, or overflowed weights cannot
  select a zero-probability action or produce an invalid index.
- Treat identical wager copies correctly when public knowledge is updated.
  Playing any same-suit wager now removes one publicly known wager guarantee,
  rather than leaking which unobservable physical copy moved.
- Emit one legal action for identical same-suit wagers. This removes duplicate
  policy mass and prevents equivalent moves from occupying multiple search
  candidate slots.
- Keep the optional dead-discard pruning heuristic off by default because it is
  not true state dominance.
- Add rollout objective mode 2: round margin in rounds 0/1, then
  `0.05 * final match margin + 50 * result` only in actual round index 2.
  Candidate choice, uncertainty gates, sampled confirmation, reported Q, and
  expert targets all use the same objective.

### Network representation

- Added a 720-combination policy residual.
- Added semantic ordered features for every buried discard-pile depth. States
  with the same discarded set, pile count, and top but different reveal order
  no longer look identical.
- Increased the feature dimension from 556 to 666.
- Added v6 model files with explicit v3/v4/v5 migration. Legacy feature rows
  retain their original indices; new rows and absent heads initialize to zero.
  The shipped v3 and v4 models produce exactly their previous outputs after
  migration.
- Updated the independent NumPy referee for v6 features and policy logits.
- Added exact wager-parameter projection across input, policy, interaction, and
  belief rows. New networks start tied, and both trainers sum and tie the
  corresponding gradients before Adam so the symmetry cannot drift.
- Added policy averaging over 5, 10, 20, or all 120 exact suit relabellings.
  The 20-way affine group is the measured default.

### Training correctness

- Expert generation now executes the move actually returned by rollout/MCTS
  gating. Search statistics remain a separate training target.
- `--sample-plies` now means plies within each round, rather than only the
  beginning of round one.
- Fixed PPO with `--temp != 1`: collection and optimization now use the same
  tempered policy and correct chain-rule gradient.
- Replaced silent full-buffer truncation with unbiased reservoir sampling and
  report retained/generated counts.
- Excluded publicly known opponent cards from belief BCE.
- Use fixed configurable evaluation seeds across checkpoints.
- Preserve exact cumulative match totals in training, analysis, mining,
  qpair, and the Python referee. Independent ±320 clipping previously changed
  the actual score lead.
- Canonicalize legacy physical-wager targets when samples are loaded, summing
  duplicate target mass instead of dropping it.
- Add exact random suit relabelling to the imitation/expert trainer with
  `--suit-augment`, plus `--gen-sym` for symmetry-ensemble teachers.
- Validate `--rounds` before fixed-size trajectory buffers are used, and add
  `--margin-weight` so rollout mode-2 values can match the finishing return.
- Keep search Q as a policy target only. Lambda returns now bootstrap from the
  network value instead of mixing skipped-search network values, round-margin
  Q, and final-round hybrid Q in one trajectory.

### Evaluation and tooling

- Derive match RNG streams from stable pair/agent identifiers. Results are now
  identical across thread counts, and one agent no longer consumes the other
  agent's random stream.
- Compute both margin and match-score uncertainty over complete mirrored
  deal-pairs, using sample variance across pairs.
- Corrected qpair's displayed value head, which was scaled by 50 twice.
- Analyzer belief dumps can now contain every uncertain card and the
  card-count prior. The evaluator reports within-state AUC, Brier score, log
  loss, top-hand-size recall, calibration error, and baseline lift.
- Build all maintained tools from `make`; add cross-module regression tests
  and GCC/Clang plus sanitizer CI.
- Updated working defaults to `data/champion.bin`; interactive policy defaults
  use its 20-way suit ensemble.

Configured one- and two-round diagnostic matches retain their existing round
semantics intentionally. Only real round index 2 is treated as the deciding
round by the match-trained policy/search options.

## Champion selection

The first audit established that inherited `c8.bin` consistently beat the file
previously named `best.bin`:

| seed | pairs | margin/match | match score |
| ---: | ---: | ---: | ---: |
| 94004 | 2,000 | +1.80 ± 0.77 SE | 51.3% ± 0.6% SE |
| 95005 | 5,000 | +2.03 ± 0.51 SE | 51.2% ± 0.4% SE |

An exploratory v6 PPO continuation from `c8.bin` learned the new parameters
and produced positive point margins, but it did not improve the primary match
win objective consistently:

| independent seed | pairs | margin/match | match score |
| ---: | ---: | ---: | ---: |
| 100010 | 5,000 | +1.10 ± 0.42 SE | 50.7% ± 0.4% SE |
| 110011 | 10,000 | +0.26 ± 0.30 SE | 50.23% ± 0.25% SE |
| 120012 | 20,000 | +0.48 ± 0.21 SE | 49.98% ± 0.18% SE |

It was therefore not promoted.

The new `data/champion.bin` is a v6-compatible, exactly wager-symmetric
projection of `c8.bin`. Projection alone was positive on three independent
holdouts:

| pairs | margin/match | match score |
| ---: | ---: | ---: |
| 5,000 | +1.32 ± 0.47 SE | 50.9% ± 0.4% SE |
| 10,000 | +1.44 ± 0.33 SE | 50.7% ± 0.3% SE |
| 20,000 | +1.05 ± 0.24 SE | 50.5% ± 0.2% SE |

At play time, averaging the champion's policy over the 20-element exact
affine suit group produces the main strength gain. The complete practical
configuration beat raw `c8.bin` directly on two fresh holdouts:

| seed | pairs | margin/match | match score |
| ---: | ---: | ---: | ---: |
| 180018 | 2,000 | **+21.24 ± 1.02 SE** | **61.4% ± 0.7% SE** |
| 190019 | 5,000 | **+19.81 ± 0.64 SE** | **60.5% ± 0.5% SE** |

The combined 7,000-pair result is approximately +20.22 points and 60.7%
match score.

Five-way averaging alone scored +15.41 ± 0.66 and 58.4% ± 0.5% over 5,000
pairs. Ten-way then beat five-way by +1.56 ± 0.80, and 20-way beat ten-way by
+1.79 ± 0.75 in 2,000-pair screens. Full 120-way averaging was only
+0.70 ± 1.34 / 51.6% ± 1.2% over 20-way in 500 pairs and costs roughly six
times more, so 20-way is the default.

Three policy-only attempts to distil a 141,500-position 20-way teacher dataset
back into one network were all weaker over 2,000-pair holdouts:

| learning rate | margin/match | match score |
| ---: | ---: | ---: |
| 3e-6 | -3.09 ± 0.85 SE | 48.81% ± 0.66% SE |
| 1e-5 | -6.95 ± 0.95 SE | 46.22% ± 0.71% SE |
| 3e-5 | -17.50 ± 0.99 SE | 40.65% ± 0.71% SE |

They were rejected rather than presenting lower imitation loss as playing
strength.

Rollout hybrid objective mode 2 is implemented but remains opt-in. At the full
96-world configuration it scored +1.11 ± 3.65 points and 51.0% ± 3.1% over
100 mirrored pairs against margin mode 0. Two cheaper independent screens
were directionally similar, but the full result is much too noisy to count as
a strength claim; a 1,500–2,000-pair locked comparison is still needed.

## Decision-audit hardening

The interactive match audit now treats rollout as a post-hoc measurement
instrument, not as a source of authoritative labels:

- The actor, deals, and evaluator have independent deterministic RNG streams,
  so changing audit compute cannot change the recorded match.
- Only the shortest top-policy prefix covering 99.5% mass is considered,
  subject to a two-move minimum and four-move cap. Near-zero draw variants
  are never forced into the audit.
- Hidden hands use the uniform card-count prior. The learned hand model is
  displayed separately and cannot bias Q.
- Fast continuation evaluation no longer conflates suit randomization with
  full policy-action sampling. Each downstream decision draws one member of
  the 20-way suit group and takes its greedy action. Repeated worlds therefore
  approximate the champion cheaply without evaluating a high-entropy player.
- Root candidates are never removed by the dead-discard focus used inside
  continuation worlds. A separate guard can block an independently supported
  but structurally questionable discard without concealing it from the audit.
- Every candidate shares the same hidden worlds. The JSON reports Q's own
  standard error separately from the paired difference and its SE versus the
  policy leader.
- Adaptive stopping uses a 3.5-SE family-wise guard. Every challenger is
  tested directly against the policy leader, so one biased numerical leader
  cannot hide a smaller real correction. Every discovery that qualifies must
  then repeat on 1,000 fresh hidden worlds at 99% confidence. Otherwise the
  result is explicitly inconclusive or failed confirmation.

The second human-reviewed match exposed the key distinction between variance
and continuation-policy bias. Simply increasing the world count made several
bad conclusions more certain. The corrected method instead (1) spends worlds
only on the top four policy moves, (2) preserves a greedy continuation actor,
(3) uses independent candidate-wise confirmation, and (4) keeps root and
continuation pruning separate. At the locked positions, the green-start
overrides at plies 21/23 disappear; low-prior green distractions at 25/29/31
never enter the shortlist; and B10 at ply 36 is independently confirmed over
Y10 (`+2.42 ± 0.34` discovery, `+1.48 ± 0.36` confirmation for the locked
seed). Two additional seeds independently selected B10 as well.

The corrected method was then tested as an actual player rather than inferred
from individual positions. The locked configuration uses the raw 20-way policy
before round ply 20, then 512 uniform hidden worlds over at most four
policy-ranked moves, a 3.5-SE/two-point primary gate, and 512 freshly seeded
random-symmetry-greedy confirmation worlds:

```text
rolloutu:data/champion.bin:512:4:0.02:0:1:20:0:0:0:0:3.5:2:2:20:0:0:20:1:0:512:1
```

Against `policy:data/champion.bin:0:20` on fresh seed 950005 it scored
**+14.70 ± 1.53 SE points per match** and **56.5% ± 1.5% SE match score**
over 200 mirrored pairs (225 wins, 173 losses, 2 draws). Approximate 95%
intervals are +11.70 to +17.70 points and 53.6% to 59.4%. This is the basis
for making rollout an actual late-round decision-maker. No early-round search
is deployed: exploratory phase screens did not justify it, and the reviewed
failures showed that increasing world count cannot repair a biased
continuation policy.

The original UI critique is preserved as fixed-state semantic regressions:

| reviewed position | full/primary audit result |
| --- | ---: |
| ply 3, Bx then deck vs W2 | deck **+5.51 ± 1.81** paired SE |
| ply 4, Bx then deck vs W2 | deck **+5.34 ± 1.79** |
| ply 8, B3 then deck vs W2 | deck **+3.11 ± 1.73** |
| ply 10, Wx then deck vs W2 | **inconclusive**, W2 +0.19 ± 1.17 |
| ply 12, W4 then deck vs R2 | deck **+5.62 ± 1.57** |
| ply 16, discard Y2 vs play W7 | discard Y2 **+1.72 ± 0.70** |
| ply 20, discard W3 vs play W7 | discard W3 **+2.77 ± 0.57** |
| ply 20, discard W3 vs discard Wx | **inconclusive**, W3 +0.70 ± 0.68 |

`make audit-test` reruns these slower checks. The ordinary runtime suite also
locks exact policy-prefix selection and prevents the old forced W2 variants.

The hand diagnostic was separately calibrated on 12 unfiltered calibration
matches and 20 untouched validation matches. Twenty-way suit averaging with
`alpha=1.15` improved held-out Brier score from 0.17538 (uniform) to 0.16194
and log loss from 0.53272 to 0.49622. Its marginals and sampler now describe
the same exact-K joint distribution. It remains labelled experimental and is
not used by the decision audit.

## Conservative correction distillation

`tools/robust_distill.c` is a separate, fail-closed path for turning only
independently confirmed search disagreements into policy training targets.
The generated dataset is tied to the exact frozen-network hash and records the
world counts, phase window, symmetry method, statistical gates, and lower
confidence bound for every correction. Loading validates the complete hostile
`State` payload, legal moves, label math, counts, file length, and every
record's declared ply/deck phase before calling engine or feature code.

Training changes only the previously zero card/action × draw-source residual
head. It anchors the full legal distribution with KL, uses pairwise
lower-confidence-bound targets for confirmed corrections, applies
deterministic suit augmentation, and verifies that every pre-residual model
byte is unchanged. Atomic, no-clobber output and canonical path checks prevent
overwriting the frozen network or record file by alias.

The locked generation run used seed 2026073003, 512 primary plus 512 fresh
confirmation worlds, and produced 426 states: 28 confirmed corrections and
398 KL anchors. The correction lower-confidence bounds averaged 1.785 points
(range 0.059–9.310). Candidate checkpoints are promoted only by independent
paired match play, never by training loss or correction accuracy.

That promotion rule was applied rather than relaxed. Two finalists advanced
from a 2,000-pair screen to a locked 20,000-pair holdout on fresh seed
2026080201:

| residual candidate | margin/match | exact match score | 2.24-SE score lower bound |
| --- | ---: | ---: | ---: |
| safe | +0.08 ± 0.08 SE | 50.04375% | 49.81975% |
| balanced-a | +0.06 ± 0.11 SE | 50.06375% | 49.83975% |

Neither score lower bound cleared 50%, so neither model was promoted. The
unchanged champion plus the validated late-round search wrapper remains the
strongest agent. This negative result is also useful: the confirmed
corrections are a sounder training signal than the old noisy audit labels, but
this small residual head did not turn them into a measurable net strength
gain.

## Remaining high-value work

1. Give earlier-round MCTS one consistent utility. Final-round terminals are
   now exact, but rounds 0/1 still mix match-trained network leaves with
   current-round terminal margins. A separate round-margin value head or an
   explicit continuation through future deals is required.
2. Replace current-round rollout utility in early rounds with either remaining
   match simulation or a learned round-end continuation table.
3. Add frozen-champion/opponent-population PPO. The existing `--ref` is
   evaluation-only; self-play never anchors against a fixed opponent.
4. Train v6 over several independent seeds, select on fixed validation deals,
   and report only once on a locked final set.
5. Re-measure playing strength for learned-world search. The calibrated
   fixed-cardinality posterior now beats the card-count prior on held-out
   Brier score and log loss, but the decision audit deliberately remains
   uniform until a locked match-strength ablation shows that learned worlds
   improve choices.
6. Measure and report the 300-ply cap rate. Consider an explicit repetition or
   adjudication rule for evaluation.
7. Replace raw native C-struct persistence with a canonical endian-stable,
   checksummed parameter format.

## Validation

The current snapshot passes:

```text
make
make test
make audit-test  # slower semantic probes; optional in the fast CI loop
python3 tools/referee.py --selftest data/champion.bin 424242 --dumpfeat bin/dumpfeat
python3 tools/verify_transcript.py data/game.txt
```

The tests include randomized card conservation, unique semantic moves, suit
permutation round trips and ensemble equivariance, wager knowledge and
parameter/gradient tying, pile-order distinguishability, v4-to-v6 migration,
interaction-head isolation, model round trips, hybrid final-round utility,
robust sampling, fixed-cardinality marginal/sampler agreement, exact
policy-prefix selection, raw-policy phase gating, and one-versus-four-thread
identity for both ordinary and rollout matches. The slow suite locks the
reviewed W2, R2, Y2/W7, W3/W7, G5, B10/Y10, and discard-guard positions.
