# Locked policy-frequency search cost v1

This campaign tests one general controller change: search must pay a learned
cost for departing from policy, and that cost grows continuously with the
policy-frequency gap. It does not contain position patches, commented-ply
labels, a candidate-zero bonus, or a relaxed arena criterion.

The deployable root score is

```text
Q(m) + lambda_A(nply) log P_A(m) + lambda_D(nply) log P_D(m | A).
```

`P_A` is the normalized suit-symmetrized probability of the semantic
played-or-discarded card core, summed across every legal draw source. `P_D` is
the conditional probability of the complete move's draw source inside that
core. Physical wager identities are collapsed in both terms. The scalar score
is transitive: there is no pair-specific exemption by which a 1% play can skip
the 4% play above it.

At a real root, candidate zero is the literal complete-move policy argmax. A
nonzero move must be the unique adjusted leader. It must also have a strictly
positive raw paired advantage over candidate zero, every higher joint
semantic complete-move prior, every higher `P_A` rival, and every same-core
higher-`P_D` rival. It must clear those rivals independently on the primary
800-world panel at 3.5 SE and a fresh 800-world panel at 2.58 SE, with both
panels selecting the same move. The intrinsic exact one-card solver runs
before this rule. The rule is never recursively charged at continuation
nodes.

The sole new actor family is `rolloutu5`; all earlier actor families retain
their byte-identical parser semantics. Every deployed artifact freezes source
seed `202611140101` and the exact binary64 floor `0x1p-150`. Its controller
tuple is fixed at root/playout symmetries `20/20`, playout sample/prune `4/1`,
exact-terminal/no-belief `1/1`, primary/fresh worlds `800/800`, root width and
semantic-core count `5/3`, minimum candidates `1`, `ply_hi=0`, discard guard
`1`, root pruning `0`, and override `k/min=3.5/0`; only the preregistered
floor and onset configuration vary.

Preflight requires the runner's exact `Python 3.12.3`, downloads the frozen
NumPy 2.3.5 CPython-3.12 manylinux wheel under the checked-in hash-required
requirements file, verifies SHA-256
`0d8163f43acde9a73c2a33605353a4f1bc4798745a8b1d73183b28e5b435ae28`,
and seals that wheel into the source-free transport. Only the TRAIN, SELECT,
TEST, and terminal independent-verification reducers install it, locally, with
`--no-index --no-deps`; every job
verifies the exact Python version and sealed wheel binding without trusting a
runner-provided NumPy package. Every job also verifies the frozen deterministic
numeric environment: one BLAS/OpenMP/MKL thread, OpenBLAS `Haswell`, AVX-512
NumPy dispatch disabled, and `PYTHONHASHSEED=0`. This makes the terminal
byte-exact calibration/selection replay independent of runner CPU variation.

The source-free transport also retains the canonical Objective-3 disposition
and every file in its authoritative evidence manifest under their original
repository-relative names. Terminal verification reopens that complete tree,
rechecks every SHA-256, actor/model/table binding, and gate disposition, and
requires it to reproduce the prerequisite snapshot embedded in the execution
addendum. The retained Objective-3 transport must be the exact preflight
`SHA256SUMS.txt` member set: interpreter caches (including `__pycache__` and
`.pyc`) are forbidden. If the completed upstream terminal artifact contains
such downstream-created caches, the authoritative persistence step may only
clean and canonically repack the already-completed evidence tree and regenerate
its result/evidence manifest; it may not rerun any efficacy shard or gate.

## Frozen data design

Three disjoint exact policy-20 source reservoirs are generated before any
search or truth label:

- TRAIN: 65,536 matches, seed `202611100101`.
- SELECT: 32,768 matches, seed `202611100201`.
- TEST: 32,768 matches, seed `202611100301`.

Discovery is monolithic per split unless a separately reviewed merger can
prove global SHA-256 priority minima and an ordered state-chain commitment.
Concatenating local shards is invalid.

All 17 user-commented information-view suit orbits are rejected before the
census denominator, policy classification, reservoir priority, allocation,
search, or truth. The four canonical exact-17 v3 artifacts and their hashes
and both canonical exclusion manifests (text and JSON) are bound in the plan
and execution. The states, alternatives, evaluations,
and audit outcomes are forbidden from training and selection.

Before the pre-efficacy barrier opens, the frozen source-free native collector
scans every retained row in all three reservoirs. It requires the exact
174-byte information-view encoding and source bounds, revalidates the native
view, recomputes the state and suit-orbit hashes, rejects the exact-17 set,
and recomputes the policy masks and cell identity. Each split produces a
sealed canonical proof; terminal verification reruns the same bound binary on
the full retained reservoir and requires a byte-identical proof.

TRAIN has exactly 13,824 states: 3 rounds × 24 fixed within-round ply strata ×
6 prior-ratio bands × 2 pair types × 16. Each state contributes one
prior-oriented pair, one primary row, one independent fresh row, and one
independent truth row. Source matches are globally distinct across TRAIN
cells. The truth panel uses 512 exact policy-20 full-remaining-match worlds.
If objective-3 v2 promoted, the target is full-match hybrid value; otherwise
it is current-round margin. That choice is fixed mechanically from the
terminal prerequisite before discovery.

SELECT and TEST each have exactly 9,216 complete vectors: 64 vectors in every
round × ply-stratum × 1%-frontier absent/present base cell. A state is assigned
to `J = SHA256(split seed, state hash) mod K`, where `K` is its master width;
the 64 allocations are divided as evenly as possible among census-feasible
candidate-slot cells. Every selected record evaluates its complete vector.
Natural discovery mass is restored only by preregistered post-stratification.
The maintained actor-selected move is appended to truth support when it lies
outside the new union.

Allocation membership is frozen before an outcome-blind scheduling
permutation. TRAIN is quota-rank-major over a diagonal ordering of all 864
cells; SELECT and TEST are quota-rank-major over all 144 base cells with three
ply bands interleaved. Fixed evaluator slices therefore mix rounds and game
phases without changing a selected unit, topping up a cell, or using a label.

The master shortlist is enumerated once at a 1% aggregate semantic-core floor.
Candidate zero is always first, and every 1%-eligible core among the true top
three aggregate semantic-action cores is retained. If candidate zero's core
ranks outside those three, it is an additional mandatory baseline comparator;
it never displaces rank three and is not shortlist widening. The fixed
maximum width remains five, so that case leaves room for at most one safe
conditional-draw alternative (otherwise at most two). The 2% list is a
no-refill mask of that exact master. Any state that fails bit-exact nestedness
fails before efficacy.

## Calibration and threshold meaning

The two nonnegative schedules use anchors
`0,4,8,12,16,24,32,40,48,64`, linear interpolation, and a clamped 64-ply
tail. Five-fold source-match-grouped cross-validation uses seed
`202611140101`, standardized Huber loss (delta 1.345), SE floor 0.25, and the
fixed smoothness grid `0,1e-4,1e-3,1e-2,1e-1,1,10,100`. The conventional
one-SE rule chooses the greatest smoothness whose mean source-match-grouped CV
loss is no greater than the minimum mean plus the source-match-cluster SE of
the minimum-loss model. Model-adequacy comparisons use the same rule inside
nested grouped CV so their outer folds remain untouched by smoothness
selection. Within-cell fold assignment carries its deterministic remainder
into the next fixed cell, keeping global source counts balanced to within one.
Every robust fit uses a frozen 500-iteration IRLS cap, a 20,000-iteration
coordinate-solver cap, and tolerance `1e-10`. Every successful fit's
convergence and iteration count is persisted; exhausting either fixed cap is
a fail-closed calibration error.
The nested round-specific, cell-saturated, zero-cost, pair-type, and
early/mid/late adequacy results are persisted preregistered diagnostics, not
extra promotion gates. Promotion remains governed only by the frozen TEST,
safety, and final criteria.

The TRAIN TSV is converted once by `policy_cost_campaign.py train-input` into
both pair-observation JSONL and the exact sealed calibration allocation JSON;
calibration is invoked with `--require-campaign-design` and
`--campaign-allocation-manifest` pointing to that companion JSON. The raw
evaluation header, allocation TSV SHA-256, raw discovery SHA-256, reservoir
SHA-256, and every selected allocation row are checked before either artifact
is emitted.

For two moves differing only in semantic-core probability, the search-EV
advantage required of the lower-prior move is

```text
lambda_A(nply) log(P_A(high) / P_A(low)).
```

Thus the learned multiplier is shared, but the actual threshold automatically
distinguishes 55/45, 95/4, 95/1, and 4/1. Equivalent `lambda_D` thresholds
apply inside a semantic core. The evidence materializes those four examples,
all six ratio-band lower bounds, every anchor, and every integer ply through
299; no threshold is chosen by looking at a commented position.

## Selection, TEST, and arena gates

Exactly twelve configurations are fixed: floor `{1%,2%}` crossed with earliest
search ply `{14,12,10,8,4,0}`. SELECT uses a 20,000-replicate source-match
cluster max-t bootstrap with seed `202611150101`. An earlier onset must beat its
immediate later same-floor parent with a strictly positive simultaneous
incremental LCB. The 1% floor must similarly beat the same-onset 2% parent.
On statistical ties, prefer later search and then 2%.

TEST opens once for that sole frozen configuration. It uses independent
P800/F800/T1024 evidence and requires, at `z=1.645`, discovery-post-stratified
hybrid-gain LCB > 0 and match-score-gain LCB > 0, nonnegative hybrid point
estimates in the 1%-frontier subset and each round, exact counts/hashes, and
zero caps. There is no second TEST or top-up.

Only a passing TEST reaches the unchanged 200-pair-per-orientation safety
gate: combined score ≥ 0.5, combined margin > 0, each orientation ≥ 0.475,
exact validity, and zero caps. Only passing safety reaches the unchanged
2,500-pair-per-orientation final gate: at `z=1.645`, combined score LCB > 0.5,
combined margin LCB > 0, each orientation point estimate > 0.5, exact validity,
and zero caps. Failure anywhere retains the prior maintained actor.

## One-shot execution

The workflow has only a push trigger for the previously absent execution
addendum. It rejects forced pushes, retries, merge commits, pre-existing
addenda, branch drift, and any launch commit changing another path. Preflight
compiles and tests once; later jobs receive only a SHA-256-bound transport and
never check out source. There is no manual dispatch, cancellation-on-new-push,
optional stopping, shortlist widening, seed substitution, repository write,
or automatic retry.

The checked-in execution and pre-efficacy JSON files are inert templates. The
real addendum can be generated only after the terminal objective-3 result is
persisted, by `tools/policy_cost_campaign.py prepare-execution`. The workflow
produces a recommendation and complete evidence archive; promotion is a
separate independent verification and persistence step.
