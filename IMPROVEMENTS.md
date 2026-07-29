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
- Harden weighted sampling so zero, negative, NaN, or overflowed weights cannot
  select a zero-probability action or produce an invalid index.
- Treat identical wager copies correctly when public knowledge is updated.
  Playing any same-suit wager now removes one publicly known wager guarantee,
  rather than leaking which unobservable physical copy moved.
- Keep the optional dead-discard pruning heuristic off by default because it is
  not true state dominance.

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
- Updated working defaults to `data/c8.bin`.

Configured one- and two-round diagnostic matches retain their existing round
semantics intentionally. Only real round index 2 is treated as the deciding
round by the match-trained policy/search options.

## Champion selection

The inherited `c8.bin` checkpoint consistently beat the file previously named
`best.bin` under the corrected runner:

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

It was therefore not promoted. `c8.bin` remains the champion until a
multi-seed v6 run wins on a locked match-score holdout.

## Remaining high-value work

1. Canonicalize or aggregate identical wager actions throughout policy
   training, serialization, and replay. The public-knowledge leak is fixed,
   but equivalent physical moves still split policy capacity.
2. Give MCTS one consistent full-match utility. Network leaves represent a
   match-trained objective while current-round terminal nodes still use round
   margin.
3. Replace current-round rollout utility in early rounds with either remaining
   match simulation or a learned round-end continuation table.
4. Train v6 over several independent seeds, select on fixed validation deals,
   and report only once on a locked final set.
5. Calibrate the belief head on held-out games, then evaluate a
   cardinality-aware joint hand posterior rather than converting independently
   trained marginal logits into a Plackett-Luce subset. On the current
   all-card seed-424 analysis it ranks cards above chance but loses to the
   card-count prior on Brier score and log loss.
6. Measure and report the 300-ply cap rate. Consider an explicit repetition or
   adjudication rule for evaluation.
7. Replace raw native C-struct persistence with a canonical endian-stable,
   checksummed parameter format.

## Validation

The current snapshot passes:

```text
make
make test
python3 tools/referee.py --selftest data/c8.bin 424242 --dumpfeat bin/dumpfeat
python3 tools/verify_transcript.py data/game.txt
```

The tests include randomized card-conservation play, wager knowledge,
pile-order distinguishability, v4-to-v6 migration, interaction-head isolation,
model round trips, robust sampling, and one-versus-four-thread match identity.
