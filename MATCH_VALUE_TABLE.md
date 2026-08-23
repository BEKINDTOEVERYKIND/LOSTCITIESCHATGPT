# Controller-bound match-value tables

Status: implemented as an opt-in rollout objective. It is deliberately not
enabled in `LC_CHAMPION_AGENT_SPEC`, and no strength claim or promotion is
attached to this change. The builder and tests use development seeds, not any
locked efficacy seed.

## Why this exists

Ordinary rollout ends at the end of the current round. Historically, rounds
zero and one were then scored by current-round margin, while the learned value
target and the actual competition are based on the completed three-round
match. That can rank an early-round action by points even when the carried
score changes how the remaining rounds should be played.

Simulating two additional rounds inside every candidate/world would be the
most literal fix, but it multiplies already-expensive online search. A fresh
round retains no hand, deck, or discard history. For a frozen continuation
controller, the expected remaining match return therefore depends only on:

- the next round index;
- the cumulative score lead from the evaluated player's perspective; and
- whether that player starts the next round.

The table stores exactly that small Markov state. A rollout still simulates the
current round normally, then replaces the early-round margin proxy with one
constant-time lookup. Final-round leaves use the exact finishing utility

```text
U(final lead) = 0.05 * final lead + 50 * sign(final lead)
```

which has the theoretical range `[-227, +227]` for three Lost Cities rounds.

## Offline Bellman construction

`bin/build_match_value` estimates a round-margin transition kernel for the
frozen continuation controller. The network sees cumulative lead clipped to
`[-150,+150]`, so the builder estimates one distribution for each of those
301 policy-visible leads. It separately covers real round indices one and two.

For the non-starting player, the last-round table is

```text
V2_not_start(d) = E[U(d + round_margin)]
```

Player-swap symmetry gives the other orientation exactly:

```text
V2_start(d) = -V2_not_start(-d)
```

The same transition kernel then backs up the middle round, accounting for the
alternating starter:

```text
V1_not_start(d) = E[V2_start(d + round_margin)]
V1_start(d)     = -V1_not_start(-d)
```

There are two intentionally distinct outputs. The raw table is the Monte Carlo
policy-value estimate above. It is not guaranteed monotone: although an
optimal player cannot be hurt by an extra carried point, this fixed neural
controller sees the lead and can change its behavior badly at the next value.
The projected table applies an equal-weight pool-adjacent-violators fit as a
structural regularizer toward optimal game value. It is not described as exact
policy evaluation. The artifact flags which variant it contains, and both
artifacts record the largest projection adjustment in each backed-up table.

Deals and player-role mappings are counter-keyed by `(seed, round, sample)`.
They do not depend on worker assignment, so changing `--threads` produces the
same artifact byte for byte. The same sample is reused across all visible
leads as common-random-numbers variance reduction. With 20 suit mappings, one
complete independent player-role product is 400 samples. The production
parser accepts only an integer number of complete role cycles; partial-cycle
artifacts remain useful as fast development fixtures but are explicitly
marked `role_balance=incomplete` and fail closed as playing actors.

Example build (9,632,000 complete round simulations):

```sh
make bin/build_match_value
./bin/build_match_value \
  --model data/champion.bin \
  --out data/champion-o0-16000.lcmv \
  --raw-out data/champion-o0-16000-raw.lcmv \
  --samples 16000 \
  --threads 8 \
  --seed 7331001
```

One transition generation produces both variants; `--raw-out` does not pay for
another 9.6 million round simulations. The paired files differ in checked
projection metadata and payload checksum. If publishing the raw half fails,
the builder removes the projected half that invocation just created rather
than leaving a misleading mismatched pair.

The initial builder intentionally evaluates one controller family: separately
stratified fixed roles (`playout_sample=4`), greedy actions, exact deck-one
completion, no recursive late replan, and the configured dead-discard focus.
The output binds the complete controller settings, a fingerprint of every
network parameter bit, a manually bumped controller-semantics ABI, and a
compiler/math/SIMD build profile. The builder uses the same production
`CFLAGS` as arena; a fast-math artifact therefore fails validation in a strict
floating-point binary, and vice versa. Any semantic change to playout policy
requires bumping `MATCH_VALUE_CONTROLLER_ABI` before rebuilding.

## Opt-in runtime use

Rollout objective `3` requires a match-value artifact in zero-based rollout
tail field 41 (the 42nd tail field). Tail field 40 remains
`action_ranker_min`; non-ranker actors must supply a zero placeholder. All
intervening fields must be present. For the maintained controller settings, an
example is:

```text
rolloutu2:data/champion.bin:data/champion.bin:512:5:0.02:0:1:14:0:0:3:0:3.5:2:4:20:0:0:20:1:0:512:1:0:0:0:0:0:0:3:1:0:0:0:0:0:0:1:0:0:0:1:0:data/champion-o0-16000.lcmv
```

Parsing rejects a corrupt artifact, a non-balanced role schedule, a different
continuation-network fingerprint, or any controller-setting mismatch. The
binary format is no-clobber, fixed-width little-endian binary64, checksummed,
range checked, and zero-sum checked. Projected artifacts are additionally
monotonicity checked; raw artifacts deliberately are not. Loaded tables are
fully scanned once; each hot-path lookup repeats cheap controller and
selected-entry checks.

Objective 3 rejects `rollout3`/`rolloutu3` actors: their third network would
evaluate a confirmation suffix with a table bound to a different continuation
controller. A `rollout4`/`rolloutu4` direct action ranker may coexist because
it only vetoes the already evaluated root pair and never supplies a table-valued
continuation.

The horizon is exactly three rounds. The shared match runner and direct
simulators (`showgame`, `analyze`, and RL opponent generation) reject table
actors when asked for one- or two-round play; the one-round interactive
`play` tool rejects them outright, as does the independent-round `probe`
collector whose labels are round margins. Silently valuing nonexistent future
rounds would be worse than refusing the run. Mutable `selfrollout`/training
generators are also rejected because a table bound to the initial network bits
becomes stale immediately after an optimizer update.

Legacy actors retain a null table pointer, preserve objectives 0–2, and do not
consume any additional RNG. The maintained actor remains unchanged until a
separate, predeclared validation campaign shows a reliable match-win gain.

Every confidence and practical-effect calculation now uses table/hybrid units
under objective 3. In the frozen maintained actor, however, `override_min` is
not a live decision parameter: `semantic_cand=0`, there are no draw/planner
extras, `policy_prefix_mode=3`, and every generated candidate is trusted. The
low-prior qualifier therefore never reads that threshold. Actors differing
only in `override_min` serialize byte for byte identically in this
configuration. The preregistered panel holds it fixed at 2 rather than wasting
matches on an unreachable parameter. If a future controller enables untrusted
extras, its match-utility calibration must be declared and tuned in a new
experiment.

At `deck_left=1`, objective 3 directly evaluates every legal deck-ending
action in table units. It does not assume the frozen controller's
maximum-margin action is still best; a focused regression proves a
nonmonotone raw table can select a lower-margin but higher-table-value action.

The table changes choices only where rollout evaluation actually runs. With
the maintained `ply_lo=14`, ordinary plies 0–13 still use the policy baseline;
the intrinsic exact deck-one branch is the exception. Development evaluation
must therefore include both an all-ply diagnostic actor and the maintained
ply-14 phase rather than claiming this fixes every early move by itself.

## Interpretation and remaining work

This is a one-step policy-improvement objective over a frozen controller. The
raw round transition table evaluates that frozen policy; the projected variant
adds an explicit optimal-value structural prior. An actor using either table
may make improved root choices. If one is promoted, rebuilding and iterating
the controller/value pair is the principled policy-iteration path.

The method fixes cross-round utility consistency; it does not by itself fix a
bad within-round continuation policy or insufficient hidden-world sampling.
A direct full-suffix Monte Carlo evaluator is still valuable as a slower oracle
for selected states and for measuring table approximation error.

MCTS is not integrated in this first implementation. Its leaf utility remains
unchanged and must not be described as table-aware. The next safe step is a
separate MCTS adapter plus parity tests against rollout at completed-round
leaves, followed by an efficacy plan whose seeds are locked before results are
observed.

The development panel is deliberately blocked until the current world-count
campaign chooses its winner. Before building, a prebuild manifest must freeze
the then-current remote commit, winner model and hash, compiler profile, and
the deterministic table command and seed. Generation then supplies the table
hashes and projection diagnostics; before any efficacy match, a second
execution addendum must freeze those hashes, rollout worlds, complete actors,
matrix, commands, and development seeds. This two-step record avoids the
circular requirement to know an artifact hash before building the artifact.
The preregistered panel tests the real 2x2 interaction: raw versus projected
values crossed with all-ply versus ply-14 use. Each candidate is compared with
the unchanged baseline in reciprocal agent-order blocks so parser or harness
position cannot masquerade as strength. A mechanical screen and larger
development confirmation record every mirrored-pair row. Only a passing
development result permits the already reserved, disjoint locked final seeds
to be activated by a separate committed execution addendum.

Objective 3 also changes final-round leaves from pure margin to the declared
hybrid match utility. The panel therefore measures the strength of the whole
actor, not isolated causality for the Bellman continuation table. An
objective-2 control could answer that narrower attribution question in a
future development study; it is intentionally outside this promotion path.

## Verification

```sh
make match-value-test
```

The focused suite covers alternating starters, zero-sum lookup, exact final
utility, theoretical ranges, controller binding, corruption/trailing bytes,
no-clobber persistence, role-cycle promotion rejection, deterministic output
across thread counts, raw/projected paired generation, three-round-only
enforcement in every match simulator, mutable-training rejection, direct
raw-table deck-one selection, and end-to-end balanced tables loaded by rollout.
