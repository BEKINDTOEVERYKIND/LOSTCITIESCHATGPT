# Locked match-value objective-3 v2 campaign

Status: definition source pending its unique seal. This document specifies a
fresh, one-shot campaign. It makes no strength claim, does not promote an
actor, and does not authorize use of any result until the complete evidence
has been independently downloaded and verified.

The machine-readable authority is
`data/experiments/locked_match_value_objective3_v2_plan.json`. If prose and
that canonical JSON ever disagree, the campaign fails closed.

## Question being tested

The maintained actor searches from zero-based ply 14 and values an early-round
rollout by current-round margin. Objective 3 instead finishes the current
round normally and uses a controller-bound Bellman table for the value of the
remaining rounds. This v2 campaign asks one deliberately narrow strength
question: does the raw or isotonic-projected objective-3 table make the frozen
800+800 controller stronger when that search is allowed at every ply?

Only two challengers exist:

| Name | Table | `ply_lo` | Objective |
|---|---|---:|---:|
| `RAW_ALL_PLY` | `data/models/match_value_objective3_v2_raw.lcmv` | 0 | 3 |
| `PROJECTED_ALL_PLY` | `data/models/match_value_objective3_v2_projected.lcmv` | 0 | 3 |

There is no ply-14 challenger and no result-dependent shortlist. The
development panel chooses between only these two frozen actors. Objective 3
changes both the round-boundary continuation value and the final-round leaf
utility, so a pass establishes the strength of the complete actor; it does
not isolate a causal effect of the table alone.

The unchanged baseline is:

```text
rolloutu:data/champion.bin:800:5:0.02:0:1:14:0:0:0:0:3.5:2:4:20:0:0:20:1:0:800:1:0:0:0:0:0:0:3:1:0:0:0:0:0:0:1
```

Each candidate is constructed mechanically. It changes `rolloutu` to
`rolloutu2`, binds the same champion as the continuation network, changes
only `ply_lo` from 14 to 0 and objective from 0 to 3, materializes the five
otherwise default tail fields, and appends its table path. Every unrelated
controller field stays textually identical. The champion SHA-256 remains
`af2b2c237d21f5ec15acbcba2fde3e45864a6e44af4ddb1ff6f3756fd687f417`.

## One paired table build

Before any efficacy match, one builder invocation creates both table variants
from the same transition histograms:

```sh
./bin/build_match_value \
  --model data/champion.bin \
  --out data/models/match_value_objective3_v2_projected.lcmv \
  --raw-out data/models/match_value_objective3_v2_raw.lcmv \
  --samples 16000 \
  --threads 8 \
  --seed 202610200001 \
  --playout-symmetries 20
```

This is 16,000 samples for each of 301 policy-visible carried leads at each
of two remaining round indices: 9,632,000 round simulations. It contains 40
complete 20-by-20 player-role cycles. Counter-based generation is independent
of worker assignment. The raw and projected artifacts must bind the same
controller ABI, controller words, model fingerprint, build profile, transition
histograms, role schedule, and build invocation. A no-clobber pre-efficacy
manifest hashes the tables, parsed metadata, compiler and runner identity,
source, binaries, actors, commands, seeds, gates, and shard matrices before
the first efficacy match begins.

The locked compiler is GCC 13.3.0 as reported by
`gcc -dumpfullversion -dumpversion`, on `ubuntu-24.04`, with:

```text
-O3 -march=x86-64-v3 -ffast-math -funroll-loops -Wall -Wextra -std=c11
```

and `-lm -pthread`. The expected match-value build-profile word is
`0030d23b`.

## Diagnostic-audit firewall

The completed exact-17 commented-ply audit is bound at commit
`42c89f554a92269ce6051a2808d4fb495530c37e`. Its canonical JSON, Markdown,
authoritative result, and reconstructed evidence ZIP are hash-bound by the
plan. They remain diagnostic only.

None of the 17 states, candidate moves, hidden worlds, estimates, or verdicts
may be used as a table transition, training example, generated probe,
development or gate seed, variant-selection input, threshold-selection input,
stopping signal, or promotion input. In particular, the audit may motivate a
general method but cannot choose raw versus projected, alter an actor, expand
a shortlist, or weaken a criterion.

## Frozen staged evidence

All panels are reciprocal three-round arenas against the unchanged baseline.
The candidate occupies agent A in one orientation and agent B in the other.
Identical mirrored pair indices are used across both development variants.
Every expected raw row, immutable shard, sidecar, hash, source/model/actor
binding, pair range, and exact sufficient statistic must validate, and every
stage requires zero capped rounds.

| Stage | Candidates | Pairs/orientation | Sharding | Candidate-first seed | Baseline-first seed |
|---|---:|---:|---:|---:|---:|
| Development | 2 | 1,000 | 100 pairs, starts 0..900 | 202610200101 | 202610200102 |
| Safety | selected one | 200 | 20 pairs, starts 0..180 | 202610210101 | 202610210102 |
| Reserved final | selected one | 2,500 | 100 pairs, starts 0..2400 | 202610220101 | 202610220102 |

Development contains 4,000 mirrored pairs and 40 raw shards. A variant is
eligible only when its combined equal-weight score is at least 0.5, combined
margin is strictly positive, each orientation score is at least 0.475, all
inputs are exactly valid, and caps are zero. Among eligible variants, selection
maximizes the exact combined score numerator, then the exact combined margin
numerator, then uses the fixed tie order `PROJECTED_ALL_PLY`,
`RAW_ALL_PLY`. If neither is eligible, there is no challenger: safety and
final are skipped and the maintained actor is retained.

Safety uses a disjoint 400-pair, 20-shard panel. Its gate is exactly the same
noninferiority-plus-positive-margin rule: combined score at least 0.5,
combined margin strictly above zero, each orientation at least 0.475, exact
validity, and zero caps. Only an exact pass activates the already reserved
final panel.

The final contains 5,000 mirrored pairs and 50 raw shards. With
`z = 1.645`, promotion requires all of:

- combined pair-clustered, orientation-stratified match-score LCB strictly
  above 0.5;
- combined pair-clustered, orientation-stratified margin LCB strictly above
  zero;
- each reciprocal orientation's match-score point estimate strictly above
  0.5;
- exact validity and zero capped rounds.

The bound is a 95% one-sided lower confidence bound, equivalently the lower
endpoint of a 90% two-sided interval. Equality does not pass either final LCB
or either final orientation rule. The margin point estimate alone is not a
substitute for the locked margin LCB.

## Seeds, barriers, and one-shot topology

Production seeds use only the `20261020`, `20261021`, and `20261022`
namespaces enumerated above. Smoke tests use only `20261029`. Smoke work can
never consume a production seed. A production seed touched by a failed,
cancelled, timed-out, or completed attempt is permanently retired; the locked
attempt is never rerun.

The definition is frozen by exactly three commits before execution:

1. `D` adds only the five new definition files: workflow, helper, tests, plan,
   and this document, on parent `42c89f554a92269ce6051a2808d4fb495530c37e`.
2. Its sole child `L` adds only
   `data/experiments/locked_match_value_objective3_v2_definition_lock.json`,
   binding `D`, its tree, every definition file, and inherited dependencies.
3. Its sole child `E` adds only
   `data/experiments/locked_match_value_objective3_v2_execution.json` and is
   pushed once, non-forced, on the first attempt.

There is no manual dispatch, retry, rerun, cancellation continuation, optional
stopping, adaptive top-up, partial-result inspection, candidate substitution,
or result-dependent table rebuild. Within each stage, all raw shards and
sidecars must finish and validate before any reciprocal merge, selection,
gate, or downstream condition is computed. Repository tests that inspect Git
history must run from a real checkout before the source is transported; the
transported evaluator then uses hash and topology manifests rather than
pretending a `git archive` contains `.git`.

The predecessor `.github/workflows/match-value-variant.yml` is immutable and
inert. Its workflow, plan, execution template, and helper hashes are frozen in
the v2 plan. Its execution path has zero commits in all Git history and is
absent. It must never be backfilled or launched.

## Evidence and promotion

The workflow uploads one terminal artifact named
`match-value-objective3-v2-complete-evidence`. It includes build provenance,
the table pair, actor and pre-efficacy manifests, all expected raw stage
evidence, complete reciprocal results, selection and gate decisions, explicit
skipped-stage records, terminal recommendation, and SHA-256 manifests. The
workflow never writes the branch, changes a checkpoint, installs a table, or
edits `final_actor_result.json`.

After the run, independent verification must reproduce every integrity check,
exact statistic, development eligibility and selection decision, safety gate,
and both final confidence bounds from the complete evidence. Only if every
locked gate passes may a later persistence commit add the already-canonical
selected table, record
`data/experiments/match_value_objective3_v2_result.json`, and mechanically
update `data/experiments/final_actor_result.json` to bind the unchanged
checkpoint plus that table. Otherwise the 800+800 actor remains maintained,
the authoritative failure result is still persisted, and neither generated
table becomes a maintained asset.

No criterion may be relaxed because of runtime, cost, a close result, a
favored audit ply, or any observed outcome.

## Relationship to policy-frequency-dependent overrides

This campaign does not guess policy-gap thresholds. Allowing objective-3
search at every ply first removes the known mismatch between early-round
search utility and the three-round match objective. Its result measures the
current override rule after that objective correction; it is not a theorem
about every possible policy-gap rule. A failure retains the maintained actor
for this campaign, but does not preclude a separately sealed gap-calibration
candidate: frequency-aware costs could suppress precisely the harmful
low-prior overrides that an unconditional all-ply rule permits.

Policy-frequency-dependent override costs therefore remain a separate,
freshly sealed calibration campaign regardless of this campaign's outcome.
The principled form is a transitive policy cost based on log-frequency
differences, fitted over predeclared ply bands with very large generated
samples, disjoint validation and final arenas, and the same fail-closed
criteria. The 17 commented plies remain held out even then. No numeric
gap-by-ply threshold should be inferred from those 17 cases or chosen before
the calibration data and selection rule are frozen.

## Required regression coverage

The focused suite must cover canonical plan constants; exact actor transforms;
audit hashes and diagnostic-only isolation; predecessor history zero; `D/L/E`
topology; push-only launch; checkout-before-archive ordering; execution and
source guards; single-build table-pair identity; role balance and no-clobber
behavior; every production/smoke seed and shard range; complete-stage
barriers; development eligibility, exact ranking, and fixed tie handling;
safety equality boundaries; strict score and margin final LCB boundaries;
orientation, cap, and raw-validity failures; terminal evidence completeness;
and pass/fail promotion behavior. The full repository test suite must pass
before `D` is frozen.
