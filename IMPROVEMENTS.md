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
- Center MCTS leaves and first-play urgency with the antisymmetric critic
  `0.5 * (V(player) - V(opponent))`. The opponent view is evaluated only after
  a root determinization, never against the real hidden hand, so common value
  bias is removed without leaking information.
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
- Added an exact visible-hand scheduler. With at most 16 deck cards it
  enumerates playable subsets of the current hand within the guaranteed
  remaining turn budget, normalizes equivalent play orders, and prefers a
  first play that leaves lower unseen cards insertable. The root correction is
  deliberately conservative: it requires either a 12-point reduction in
  blocked lower-card options or at least two points of visible-hand regret
  from spending a turn on a policy move outside every best guaranteed plan.
  It never reads a hidden card.
- Added a narrower late draw-source planner. It keeps the policy-selected card
  and play/discard action fixed, evaluates public pile tops exactly, and
  averages deck draws over the mover's complete information-set support. Root
  play and rollout continuations have separate thresholds, so a root-only
  strength result cannot silently alter the search world model. With four or
  fewer deck cards the root-only repair gained **+2.318 ± 0.222 points per
  match** and **51.138% ± 0.259% match score** over 2,000 fresh policy-only
  mirrored pairs.
  The full rollout actor was then tested on 30 external deal-pairs. Root-only
  repair changed no match outcome and added `+0.65 ± 1.44` points versus the
  unchanged actor on those identical deals, so it remains opt-in. Applying
  the repair inside every continuation was rejected: on 15 identical
  deal-pairs it lost `−11.37 ± 7.76` points and `−16.67 ± 9.34` match-score
  percentage points. The deployed continuation model is therefore unchanged.
- Solve the complete one-card-deck decision exactly. Every legal semantic
  play/discard action is paired with the deck draw, scored under the same
  round/final-match objective as rollout, and compared without sampling. The
  solver runs both at the real root and at the end of every simulated
  continuation, so earlier stalling and ending decisions inherit optimal
  final-turn play instead of exploiting a policy-network error. A controlled
  policy-action terminal mode keeps real roots identical while preserving the
  policy's simulated final action, allowing a direct propagation ablation. On
  20 precommitted mirrored pairs, full propagation scored `+5.62 ± 6.74`
  points and `52.5% ± 5.7%` match score against that control. This small screen
  is positive but inconclusive rather than promotion-grade evidence. This is
  exact-leaf propagation, not complete backward solution: live play remains
  policy-only before round ply 14, a singleton shortlist performs no sampled
  comparison, and ordinary deck-two/deck-three rollout intermediates use the
  champion continuation before their exact one-card leaf. Every evaluated
  candidate inherits the solved ending, but not every preceding stall state is
  itself solved.
- A separate 20-pair whole-actor screen against the raw exact-20 policy scored
  `+17.45 ± 5.80` points per game and `60.0% ± 5.8%` match score. This is a
  small, wide exploratory comparison of the complete maintained actor. It is
  not an isolated exact-terminal or bounded-late-resolver ablation and is not
  presented as component-level promotion evidence. Exact specifications,
  seeds, aggregate results, and the limits of the retained provenance are in
  [`data/experiments/locked_strength_screens.json`](data/experiments/locked_strength_screens.json).
- Detect complete repeated continuation states once the deck has at most three
  cards. The active detector hashes only the current mover's sanitized
  information set: own hand, known-card guarantees, and the complete public
  position. It never branches progress behavior on the sampled opponent hand
  or future deck order. Interchangeable wager IDs are collapsed and the ply
  counter is deliberately omitted so a public pile shuttle cannot evade
  detection. On repetition the maintained greedy network is conditioned on
  drawing from the deck, so its learned card/action×draw interaction—not the
  unrestricted complete-move winner—chooses the action. Optional
  sampled/planned actors and the propagation-control ablation retain their
  selected semantic action. A deliberately pathological eight-ply pile
  shuttle is a regression fixture. For a long non-repeating pile walk, the
  last `deck_left` engine-fuse slots reserve one real deck draw apiece under
  the same conditional policy. Any continuation that somehow remains
  unfinished is counted, makes the search unresolved, and is forbidden from
  authorizing a policy override. All work/failure counters are 64-bit, so an
  extreme audit cannot wrap the fail-closed unfinished-leaf check through zero.
- Added focused semantic rollout challengers rather than an exhaustive legal
  move scan: useful face-up pile draws attached to one of the top three
  card/disposition actions, and one one-sided isolated wager discard when the
  opponent cannot score it directly. The discard is not presumed safe because
  the opponent may still pick it up to stall. Outside the ordinary rollout
  window it is compared alone with the baseline using a 16,384-world primary
  cap plus 16,384 fresh confirmation worlds; ordinary broad early rollout
  remains disabled.
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
  gating. Its rollout target is also selection-aligned: a rejected fresh-panel
  proposal becomes a one-hot fallback target, while a confirmed two-panel
  choice may retain averaged soft Q values. Search can no longer play one move
  while training toward the move its own consensus check rejected.
- `--sample-plies` now means plies within each round, rather than only the
  beginning of round one.
- Fixed PPO with `--temp != 1`: collection and optimization now use the same
  tempered policy and correct chain-rule gradient.
- Replaced silent full-buffer truncation with unbiased reservoir sampling and
  report retained/generated counts.
- Excluded publicly known opponent cards from belief supervision and replaced
  independent-card BCE with the exact fixed-cardinality joint likelihood used
  by deployment sampling.  Its gradient is posterior marginal minus the true
  K-card label, and only mover-perspective states after ply zero are trained.
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
- Add frozen-opponent population generation to PPO. Opponent actions remain
  useful for value and belief supervision but are explicitly masked out of
  PPO, entropy, and anchor-KL losses. Adjacent games reuse the same three deals
  with learner seats swapped, cancelling deal and starter exposure exactly;
  the seat counts and learner score are reported every iteration.
- Add full-legal-action `KL(anchor || live)` regularization and a `--v6-only`
  warm-up. The latter can learn only ordered-pile input rows and complete-move
  interactions while restoring every inherited parameter byte-for-byte after
  each optimizer step. Policy gradients compensate for the smaller actor
  fraction in opponent games without changing the historical self-play scale.
- Add default-off, trajectory-coherent PPO suit augmentation with
  `--trajectory-symmetries`. A seed/global-match-id selects one exact-group
  mapping that stays fixed across every round and ply. States, chosen actions,
  old probabilities, value targets and belief labels remain in that orientation;
  only the selected action is inverse-mapped for the canonical engine. Frozen
  opponent decisions remain explicitly excluded from PPO gradients.
- Couple the two perspective value gradients into the same antisymmetric
  critic used by search. Complete self-play states provide both legal views;
  paired perspective rows are half-weighted so they sum to one centralized
  squared loss, and the common value bias receives exactly zero gradient.
- Add `--belief-only` calibration. It backpropagates exact-K likelihood only
  into `wbel`/`bbel` and uses an Adam range update whose weight decay cannot
  touch trunk, policy, value or interaction parameters.

### Evaluation and tooling

- Derive match RNG streams from stable pair/agent identifiers. Results are now
  identical across thread counts, and one agent no longer consumes the other
  agent's random stream.
- Compute both margin and match-score uncertainty over complete mirrored
  deal-pairs, using sample variance across pairs.
- Add absolute pair ranges and atomic per-pair arena evidence. Expensive
  comparisons can now be split without repeating deals: the deal and both
  actor streams remain keyed by the original block seed plus absolute pair
  index. `tools/merge_arena.py` validates a complete nonoverlapping range,
  rejects cap-terminated games and provenance drift, recomputes exact
  pair-clustered statistics from integer rows, and combines reciprocal blocks
  only after inverting the second orientation. Allocation/thread failures and
  incomplete workers fail before an evidence footer can be published. The
  final reciprocal command reopens, hashes, and exactly remerges every
  recorded raw shard before it can label a promotion; self-consistent edited
  summaries are therefore insufficient. Its critical z value, directional
  margin requirement, and per-orientation requirement are explicit result
  fields rather than an undocumented interpretation step.
- Corrected qpair's displayed value head, which was scaled by 50 twice.
- Analyzer belief dumps can now contain every uncertain card and the
  card-count prior. The evaluator reports within-state AUC, Brier score, log
  loss, top-hand-size recall, calibration error, and baseline lift.
- Added `bin/belief_eval`, a deterministic held-out evaluator for the actual
  exact-K joint posterior. Its frozen exact-policy trajectories do not consult
  beliefs. The trajectory actor is a separately loaded checkpoint, so a
  full-model belief candidate cannot change which states it is scored on. A
  reusable information-view boundary removes hidden opponent cards and future
  deck order before every network call. It reports joint NLL
  per state/card, Brier, tie-correct within-state AUC and top-K recall against
  analytic uniform card-count baselines. Its finite-cluster sandwich
  uncertainty targets the same pooled state/card/pair-weighted estimators that
  are printed, clustered by full match. Hidden-assignment invariance,
  frozen-actor separation, and malformed CLI inputs are regression tested.
- Learned-world rollout exposes a bounded `belief_alpha` calibration control;
  analyzer output records both the world model and exact alpha.
- Analyzer and replay UI now distinguish raw-policy, visible-hand, last-deck,
  and information-set draw baselines, including separate root/continuation
  thresholds. A repaired draw source is therefore visible rather than silently
  attributed to the network policy.
- Add opt-in role-separated coherent continuation panels. Each player's
  continuation policy receives an independently stratified suit orientation
  fixed for the complete sampled world, covering the product group without
  extra network forwards. Reviewed positions were mixed, so this remains an
  experiment and the measured champion default is unchanged.
- Added perspective-scrubbed actor-history inference for offline review.
  `tools/history_belief.py` passes only the observer's original hand, public
  action prefix, and that observer's own deck draws to a deterministic
  rejection worker. Hidden opponent cards and future draws never enter its
  wire format. The v1 wrapper rejects stochastic, planner-adjusted, semantic,
  or rollout-search prefixes that its raw-policy likelihood cannot model, and
  hashes both checkpoints before accepting the recorded actor attribution.
- Build all maintained tools from `make`; add cross-module regression tests
  and CI jobs for GCC, Clang, AddressSanitizer, and UndefinedBehaviorSanitizer.
  Those workflow definitions are distinct from local validation: no local
  Clang or LeakSanitizer run is claimed here.
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
times more.

That exploratory result was not used as a deployment claim. A fresh reciprocal
5,000+5,000-pair policy qualification passed its compute gate, after which the
unchanged 120-root candidate faced the complete maintained rollout actor on
fresh seeds. The authoritative 1,000+1,000-pair reciprocal actor panel scored
`49.7125% ± 0.6540% SE`, with orientation scores `48.95%` and `50.475%`, a
`-0.3955 ± 0.8091` point margin, and W/L/D `1978/2001/21`. Its required
1.695-SE lower bound was `48.6039%`; the combined bound, first orientation, and
positive-margin conditions all failed. Root 20 therefore remains the default.
The two-sided 95% score interval still includes parity, so the conclusion is
"not validated," not "proved weaker." All 20 raw shards, timings, hashes, the
exact evaluator, and a hardened remerge are archived under
[`data/experiments/root120_stage2`](data/experiments/root120_stage2).

The generated champion retains exact `+0` values in the 110 v6 pile-order
input rows and the complete 720-way interaction head because its source is a
legacy checkpoint. Rollout inference now proves those regions directly from
the loaded parameter bits once per decision and skips only the corresponding
zero multiply-adds. It fails closed for any nonzero bit pattern, negative
zero, NaN, nonfinite activation, owner mismatch, or invalid plan, and a plan
is valid only while its network remains immutable. The generic v6 path is
unchanged and remains the oracle. Runtime regression tests require byte-exact
agreement across 1/5/10/20/120-way policy evaluation, explicit and sampled
suit permutations, full rollout results and diagnostics, confirmation panels,
recursive late replanning, and post-call RNG state. This optimization is not
counted as a strength improvement. A balanced same-machine four-block
crossover on the maintained 120-way-versus-20-way rollout workload produced a
`1.439x` geometric-mean wall-time speedup (`1.381x`-`1.495x`) with all 32 game
rows and completion records byte-identical. Two blocks rebuilt from the final
small-policy-dispatch revision retained byte-identical results at `1.448x`
geometric-mean speedup. The dispatch threshold also removes the proof-scan
overhead from five- and ten-way policy-only actors. Exact provenance is in
[`data/experiments/evalplan_performance.json`](data/experiments/evalplan_performance.json).

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

That comparison is now precommitted for the maintained 512+512-world actor in
[`locked_objective_mode2_plan.json`](data/experiments/locked_objective_mode2_plan.json).
Its explicitly excluded 50-pair runtime block took 2,307.3 seconds and returned
50.0% ± 4.0% / -2.16 ± 4.87 points; the complete result and command are in
[`objective_mode2_development_runtime.json`](data/experiments/objective_mode2_development_runtime.json).
The score is not promotion evidence. It establishes that the two reciprocal
1,000-pair blocks require roughly 12.8 hours when run concurrently on two
four-thread groups, and the production default remains mode 0 until that locked
gate is actually completed.

## Decision-audit hardening

The interactive match audit treats rollout as a post-hoc measurement
instrument, not as a source of authoritative labels. Its exact default is:

```text
rolloutu:data/champion.bin:2048:5:0.01:0:1:14:0:0:0:0:3.5:2:2:20:0:0:20:1:0:2048:1:0:0:0:0:0:0:2:1:0:0:2:1:0:3:1:0:0:1
```

- The actor, deals, and evaluator have independent deterministic RNG streams,
  so audit compute cannot change the recorded match.
- For the first 14 actions of each round the audit reports the exact policy
  and explicitly skips rollout. Search begins at zero-based `round_ply=14`,
  the 15th displayed action. A 2,048-world opening probe still produced a confident low-prior
  ply-8 override, confirming that extra worlds do not repair the long-horizon
  continuation bias. This matches the live phase boundary selected by direct
  match play.
- From that 15th action the audit first groups complete moves by semantic
  card/play-discard action, then admits at most three distinct top-policy
  action cores with at least 1% aggregate prior. Its five total slots can also
  retain two policy-supported draw alternatives for those cores. It still does
  not scan every legal move or enable the broad planner/semantic research
  tails.
- Uniform-world outer panels at deck two or three now sample ordered
  mover-view assignments without replacement, independently of whether the
  recursive resolver is enabled. The primary, trusted-prefix-confirmation, and
  challenger-confirmation domains each use their own deterministic order. A
  2,048-world request exhausts one 90-assignment deck-two census and up to 990
  deck-three assignments; public opponent-card knowledge can reduce either
  support. Duplicate assignments no longer inflate the apparent sample size.
  The maintained live actor requests 512 outer worlds (90 at deck two, 512
  unique assignments from at-most-990 support at deck three); the deep audit
  requests 2,048 and exhausts both supports. The maintained audit's historical
  recursive-replan fields are both zero.
- At an actual audit root with two or three deck cards, a separate bounded late
  resolver exhausts the complete ordered mover-view support: at most 90 worlds
  at deck two and 990 at deck three, with public knowledge reducing either
  count. This support is independent of `deck2_replan_worlds`. The explicit
  `bounded_late_root=1` audit flag enables the panel while both recursive fields
  stay zero. Its separate `bounded_late_min=1` requires a greater-than-one-point
  gain in both horizons without changing ordinary prefix confirmation. The
  maintained actor leaves the flag and recursive fields at zero.
- Root candidates remain policy-focused, but candidate zero is always the
  literal global complete-move policy argmax even if its semantic core would
  otherwise fall outside the shortlist. If a later state must progress through
  the deck, its action cores are ranked by the policy conditional on a deck
  draw. The evaluator does not choose an unrestricted argmax and then replace
  its draw source.
- The bounded panel carries each ordered particle through the stall sequence,
  solves every deck-one leaf exactly, and improves only later information sets
  belonging to the current root player. Nonterminal opponent information nodes
  retain the frozen champion policy. It separately solves H=2 and H=4; a root
  override requires the same complete move to win both horizons and to exceed
  the configured practical-gain threshold in each. Once this bounded panel
  completes, it is the authoritative gate at that audit root: an accepted
  challenger is selected, while disagreement or insufficient gain retains the
  literal policy candidate immediately. Retention is conservative, not a proof
  that the policy is optimal. Only a panel unavailable before completion falls
  back to the ordinary policy-focused evaluator; it does not enable the
  historical recursive method.
- The locked deck-three ply-42 probe is stable on G8+deck at H=2 and H=4 but is
  rejected because its H=2 gain is only `+0.036`, below the one-point practical
  gate. The locked deck-two ply-43 probe accepts playing B10 to the blue
  expedition followed by the Yellow-pile draw at `+20.622` over the literal
  policy baseline in both horizons. These are finite restricted-model
  regressions, not claims of equilibrium play.
- The older recursively redeterminized evaluator remains available only to
  low-level experimental specs. It replans from the current mover's sanitized
  information set, with a separate world/depth budget, information-set-safe
  seeds, and top policy semantic cores. The maintained audit does not enable
  it: a bounded-panel failure resumes the ordinary evaluator, and a completed
  rejection returns candidate zero. Its recursive calls/worlds/evaluations,
  root calls/worlds, caps, fallbacks, cache hits, cycles, depth, and stall-chain
  counters must all remain zero in the tracked audit.
- In those low-level recursive experiments, the actual selected path uses the
  configured suit ensemble and up to three semantic cores. Hypothetical
  descendants use one deterministic information-state-keyed group member and
  their top core, with a strict shared work budget and low-world fallback. This
  path stays regression-tested but no longer affects the maintained audit.
- An all-depth exact-20 variant was attempted on both locked late probes, but
  each run exceeded 25 minutes without completing and was stopped. It was
  rejected as computationally impractical; no probe value or playing outcome
  is claimed from those incomplete runs.
- In low-level recursive experiments, each panel has a separate deterministic
  public-information domain and cache. State/path/budget keys, mover-view node
  seeds, low-world fallback, and explicit cycle closure prevent private-root
  leakage or a compute limit from manufacturing a deck draw. No such cache or
  redeterminization is active in the maintained audit.
- The JSON/viewer exposes the bounded resolver's ordered support, H=2/H=4
  values, stability, practical-gate result, frozen-opponent work, and whether it
  was permitted to select. It separately exposes first-panel calls/worlds,
  total replans/worlds, candidate evaluations, budget-cap hits, low-world
  fallbacks, transposition hits, recursive cycle closures, maximum recursive
  depth, and maximum stall chain for the historical evaluator. The compatibility
  name remains `deck2_replan`, although both methods cover deck two and deck
  three. Those historical counters are asserted to be zero in the maintained
  audit. The bounded resolver remains audit-only: freezing the opponent and
  using the champion policy inside upstream playouts makes it dynamically
  inconsistent with re-rooted self-play. Making a completed root panel
  authoritative removes contradictory fall-through, but does not repair that
  upstream inconsistency. It is neither an equilibrium claim nor a promoted
  actor component.
- The tracked viewer now displays a deterministic match/build identifier. Its
  generator validates and stages the standalone JSON and embedded payload
  together, rolling back a partial install, and the undefined diagnostics
  accumulator that caused a runtime stop was removed. These changes make a
  stale or mismatched viewer artifact diagnosable instead of looking like the
  newly generated match.
- The primary panel gives all candidates the same requested set of up to 2,048
  uniform hidden worlds; at a late root, the unique support census above is the
  cap. Each downstream decision draws one member of the 20-way suit group and
  takes its greedy move, preserving a strong low-cost continuation instead of
  sampling high-entropy policy actions. The separately tested recursive method
  may use a full ensemble on selected-path nodes and one deterministic group
  member on hypothetical descendants, but that method is disabled here.
- If the primary ordinary-prefix leader differs from the raw policy, a fresh
  panel requests up to 2,048 worlds and assigns balanced suit mappings that
  remain fixed for each complete hidden-world trajectory; at a late root, its
  own without-replacement support is the cap. The proposed move must
  independently beat candidate zero by both two paired standard errors and one
  objective point. The fresh panel may rank a statistically close alternative
  improvement first: requiring the identical argmax made two good stalls
  cancel each other and restore a clearly inferior deck-ending baseline. This
  remains a conservative evidence threshold, not proof that the shared
  continuation model is unbiased.
- Hidden hands use the uniform card-count prior. The learned hand estimate is
  displayed separately and cannot bias Q.
- Every reported Q has its own standard error plus a paired difference and
  paired SE against the policy reference. Root candidates are not removed by
  continuation-only discard focus. These SEs describe outer hidden-world
  variation conditional on the deterministic searched policy; they neither
  count a reused inner decision as fresh evidence nor include uncertainty over
  alternative inner-search seeds.
- Purposefully added low-prior component challengers, when explicitly enabled,
  retain the older 3.5-SE, practical-gap, and fresh-confirmation gates. The
  optional 16,384-world one-sided-wager probe and visible-hand planner are not
  part of the default actor or default audit.

The second human-reviewed match exposed the key distinction between variance
and continuation-policy bias: simply increasing the world count made several
bad conclusions more certain. The revised method therefore spends worlds only
on distinct top-policy action cores, keeps policy actions greedy, and separates
the cheap and coherent continuation panels. A proposed correction must clear
the fresh paired evidence and practical-effect gates against candidate zero;
exact leader agreement is required only when those gates are disabled. On the
earlier locked three-slot 2,048-world audit, neither Green-5 action is
admitted at plies 29 or 31; ply 31 selects the clean R6 discard. At ply 36 the
three candidates are exactly Y10/B10/W10 and B10 beats Y10 by `+2.27 ± 0.24`
in discovery and `+1.95 ± 0.21` on the fresh panel. These are executable
regressions in `make audit-test`, not claims that every continuation value is
unbiased, and they do not silently change the live actor.

### Pre-consensus rollout and component history

The preceding ply-20 actor used the raw 20-way policy before round ply 20,
then 512 uniform hidden worlds over at most four policy-ranked moves with a
3.5-SE/two-point gate and fresh random-symmetry-greedy confirmation:

```text
rolloutu:data/champion.bin:512:4:0.02:0:1:20:0:0:0:0:3.5:2:2:20:0:0:20:1:0:512:1
```

Against `policy:data/champion.bin:0:20` on fresh seed 950005 it scored
**+14.70 ± 1.53 SE points per match** and **56.5% ± 1.5% SE match score**
over 200 mirrored pairs (225 wins, 173 losses, 2 draws). Approximate 95%
intervals are +11.70 to +17.70 points and 53.6% to 59.4%. This established
that rollout can be a real decision-maker, but the actor has since been
superseded by the directly tested ply-14 two-panel consensus method.

The scheduler was also isolated from rollout on an independent 2,000-pair
test. `policy:data/champion.bin:0:20:16:12` beat the raw exact-20 policy by
**+2.38 ± 0.33 SE points per match** with **51.9% ± 0.4% SE match score**.
This validates the conservative visible-hand layer, but it does not imply an
additive gain after late rollout has already corrected many of the same
positions.

The pre-consensus component screen compared three actors:

- A: the preceding ply-20 rollout above;
- B: A plus the planner-only `:16:12:0` tail;
- C: A plus the planner and semantic `:16:12:1` tail.

On the same 40-pair seed against raw policy, A scored +22.04 ± 4.25 points and
58.8% ± 3.5%, while B scored +16.18 ± 5.34 and 57.5% ± 4.2%. A fresh direct
20-pair B-versus-A check was +1.60 ± 5.34 and 55.0% ± 5.0% for B: compatible
with either a modest gain or loss, and nowhere near a promotion result.
Direct C versus A over 40 pairs was -6.06 ± 4.36 and 46.2% ± 3.8%. This did
not establish planner or semantic additions as beneficial to the preceding
actor, much less to the newer consensus actor. They remain explicit component
tools and are not hidden inside live play or the default audit.

The legacy semantic-probe configuration remains available as:

```text
rolloutu:data/champion.bin:512:4:0.02:0:1:20:0:0:0:0:3.5:2:2:20:0:0:20:1:0:512:1:16:12:1
```

Its focused semantic additions are validated below on frozen positions, not
as a global match-strength claim.

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

A belief-head-only continuation then improved again on one 100-match
evaluation set and a separate 100-match alpha-calibration set without changing
any trunk, policy, value, or interaction byte. On evaluation seed
`208020712`, joint NLL/state improved from `13.20308` to `13.03848`, Brier
from `0.164236` to `0.162103`, within-state AUC from `0.658982` to `0.673527`,
and top-K recall from `0.470341` to `0.480762`. On calibration seed
`208020713`, which selected `alpha=1.15`, it improved NLL/state from `13.002282`
to `12.823294`, Brier from `0.162092` to `0.159679`, AUC from `0.670997` to
`0.683571`, and recall from `0.484400` to `0.494815`.

That inference win did not justify deployment. Used by itself for live hidden
worlds against the uniform-world champion on fresh seed `208020714`, the
candidate scored only `40.0% ± 6.9%` over 20 mirrored pairs (W/L/D 16/24/0),
with a noisy `+1.57 ± 6.92` point margin. The candidate was rejected: posterior
quality is a component metric, while match wins remain the promotion gate.
Combining that belief head (`alpha=1.15`) with the three-core shortlist and a
near-greedy `0.03` fresh-panel temperature also failed on locked seed
`208020716`: `−6.78 ± 7.40` points and exactly `50.0% ± 5.1%` match score over
20 mirrored pairs (W/L/D 20/20/0). It is not deployed.

The uniform-world three-slot/three-core shortlist (`root_width=3`) was also
tested without the rejected belief head. Its exploratory seed `208020715`
scored `+20.23 ± 7.29` points and `65.0% ± 5.3%` match score over 20 mirrored
pairs (W/L/D 26/14/0). That screen was not treated as promotion evidence. On
the precommitted independent seed `208020717`, 40 further pairs produced only
`+1.04 ± 5.37` points and `51.2% ± 3.8%` match score (W/L/D 41/39/0). The
method therefore remains the focused post-game audit shortlist; the
maintained five-complete-move live actor and checkpoint are unchanged.

### Historical frozen-showcase review

That earlier review did not regenerate or screen for a friendlier game. The
original random seed `5726968372613385`, deals, and all 144 historical actions
were retained unchanged at the time. The results below remain experiment
history; the current tracked viewer contains a separately precommitted random
self-play game under the promoted actor.

At ply 4, the old current-state belief head incorrectly ranked Y4 above the
high yellows. Perspective-scrubbed actor-history inference proposed 100,000
deals and accepted 4,433 whose visible prefix was plausible under the frozen
actor. Its opponent-hand marginals were Y4 21.93%, Y9 22.94%, and Y10 22.96%.
This restores the expected near-flat yellow ordering without leaking the
opponent's true hand or future deck draws.

Ply 7 received a deeper three-family continuation audit. The pooled
exact-policy estimates for G5 and W5 were 29.49 ± 0.38 and 29.60 ± 0.39;
W5's paired difference over G5 was only +0.11 ± 0.35. Y4 was decisively worse
at 7.63 ± 0.37, or -21.86 ± 0.40 against G5. The defensible answer is
G5-or-W5, not the premature Y4 commitment.

Two one-sided wager cases motivated the urgent semantic path:

| position | focused correction | primary audit | fresh confirmation |
| --- | --- | ---: | ---: |
| ply 59 | discard Bx, take R | **+5.16 ± 0.40** (16,384 worlds) | **+4.77 ± 0.40** (16,384) |
| ply 61 | discard Bx, take R | **+4.26 ± 0.39** (16,384 worlds) | **+4.04 ± 0.40** (16,384) |

The useful draw is chosen together with the wager discard, rather than forcing
the deck variant or evaluating every draw source. The move still must pass
both 16,384-world probe panels because the opponent can pick up the
wager to stall. At ply 96, an exact
one-card-deck rule selects G9 then deck over the historical G9 then White:
both score the same immediately, but the pile draw can only grant the
opponent an extra optional turn.

These corrections do not mean rollout is globally fixed. Review also found
positions where a biased continuation attached high confidence to a poor
ordering. Optional exact visible-hand corrections remain useful component
probes, and unrecognized low-prior moves can still require a dedicated
`qpair` probe or future policy training. The promoted actor searches from ply
14 but reduces orientation instability by requiring its cheap and coherent
panels to agree; agreement still cannot cure a continuation bias shared by
both panels.

In particular, ply 17 remains an honest historical miss and is now inside the
live search window: when forced to compare the reviewed alternatives over
4,096 worlds, that continuation
favored play R8 and put discard B5 then take R at -2.20 ± 0.73. At ply 25,
taking Bx now enters the focused shortlist, but the locked uniform-world audit
still put it at -5.52 ± 0.58. Behavior-conditioned hidden worlds improve the
latter estimate materially without yet producing a stable override. Those
cases are retained as belief/continuation-model targets rather than encoded as
position-specific rules.

## Conservative correction distillation

`tools/robust_distill.c` is a separate, fail-closed path for turning only
independently confirmed search disagreements into policy training targets.
The generated dataset is tied to the exact frozen-network hash and records the
world counts, phase window, symmetry method, statistical gates, and lower
confidence bound for every correction. Loading validates the complete hostile
`State` payload, legal moves, label math, counts, file length, and every
record's declared ply/deck phase before calling engine or feature code.

Training changes only previously zero v6 capacity: always the card/action ×
draw-source residual head and, when `--train-pile-order` is explicit, the
appended ordered-pile input rows. It anchors the full legal distribution with
KL, uses pairwise lower-confidence-bound targets for confirmed corrections,
applies deterministic suit augmentation, and verifies that every legacy model
byte is unchanged. Policy KL uses exact double-precision log-softmax rather
than probability floors. Every epoch and final candidate must satisfy policy,
value, and belief trust regions over all 120 suit permutations of every record;
any non-finite activation, logit, or metric fails closed. Atomic, no-clobber
output and canonical path checks prevent overwriting the frozen network or
record file by alias. Multiple record shards may be combined only when their
model hash and complete label-generation provenance match.

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
unchanged champion plus the promoted ply-14 two-panel consensus wrapper remains
the strongest agent. This negative result is also useful: the confirmed
corrections are a sounder training signal than the old noisy audit labels, but
this small residual head did not turn them into a measurable net strength
gain.

## Coherent rollout consensus and v6 adapter experiment

Review of a second unfiltered match found a deeper continuation problem. Fast
mode `2` selected a fresh suit relabelling at every downstream decision, so a
single sampled world could be played by a sequence of mutually inconsistent
network orientations. Mode `3` now assigns one mapping to the complete
hidden-world trajectory and stratifies worlds across the requested symmetry
group. It repaired several reviewed orderings, but regressed others, so it is
not treated as an automatic replacement for every primary comparison.

The safer playing method uses the cheap mode-2 panel for discovery, then
rechecks any nonbaseline leader from the ordinary top-policy prefix on fresh
worlds whose suit mappings are balanced and fixed for each trajectory. The
move changes only when both panels select the same leader. Purposefully added
low-prior challengers remain subject to the older practical, SE, and fresh-
confirmation gates. Bounded pile-draw variants were also implemented for the
top one or two distinct card/disposition actions, but their 28-pair screen
faded to approximately neutral and they remain disabled by default.

On a locked 100-pair external-reference set (200 matches, seed 20310801), the
ply-14, top-five consensus actor scored `+10.08 ± 3.98` points per match and
`57.25% ± 3.02%` match score (`114/85/1`). The approximate 95% intervals for
both point margin and match score exclude parity. In a separate locked direct
test against the preceding ply-20 actor (100 pairs/200 matches, seed 20320801),
it scored `+3.60 ± 3.72` points and `55.0% ± 2.9%` match score (`109/89/2`).
Both direct-test intervals include parity, so this is positive but not
independently conclusive evidence; the promotion rests on the consistent
direction across both tests plus the reproduced continuation-method repair,
not on a claim that the direct 100-pair margin alone is significant.

The reviewed ply-29 Green start illustrates why panel agreement is still not
a proof of optimality. The raw mode-2 continuation preferred play G5 over
discard R6 by `+7.91 ± 1.22`; coherent mode 3 gave `+6.72 ± 1.12`, and exact
20-way continuation gave `+5.85 ± 0.85`. Replacing the downstream raw policy
with an eight-world, top-three receding-horizon search for both players reduced
the pooled difference to `+0.42 ± 1.15`. A stronger independent checkpoint
still weakly preferred G5, so R6 is not established as correct; what is
established is that the original audit overstated its certainty by rewarding a
root move for pushing its own weak continuation into a different behavioural
basin. `qpair -A` retains this expensive search-aware continuation for such
contentious positions. A selective top-three, 16-world continuation re-search
did repair ply 23 and halve the ply-29 bias, but still failed ply 31, cost
roughly 5–8× more, and scored about `−2.51 ± 1.39` points with a 46.8% match
score over 300 paired single rounds. It was rejected rather than hidden inside
live play.

Opponent match logs were then used only to *propose* information states. The
new `tools/mine_duel_states.py` replay-validated all 300 matches and 42,823
plies before emitting 317 validated states; no opponent score became a training
label. Our frozen evaluator independently confirmed 66 of those proposals.
Combined with our own probes and self-play, adapter training used 534 records:
71 corrections, 463 KL anchors, and 61 corrections containing buried-pile
order. Version-3 correction shards bind the fixed-world continuation method,
exact-five confirmation, model hash, and complete generation provenance.

`robust_distill` can now train the zero-initialized interaction head and,
optionally, only the appended ordered-pile input rows. It fails closed on
incompatible shard provenance, non-finite parameters, insufficient pile-order
evidence, excessive mean or worst-state KL, or any changed legacy byte. The
resulting adapter repeated a small point-margin gain:

| holdout | mirrored pairs | margin/match |
| --- | ---: | ---: |
| initial screen | 3,000 | +0.15 ± 0.26 SE |
| independent seed 995102 | 10,000 | +0.24 ± 0.13 SE |
| independent seed 995202 | 10,000 | +0.24 ± 0.12 SE |

Across the two large holdouts it gained about `+0.24 ± 0.09` points but only
about 50.07% match score. Since match wins are the objective, the adapter was
not promoted and `data/champion.bin` remains unchanged.

## External-teacher residual and population-training experiments

A later residual-only teacher used 183 high-confidence search disagreements
from the first 100 mirrored pairs of an independently implemented opponent.
The opponent's match outcomes were never labels: each target was its searched
move only when its own paired world estimate cleared both a practical gap and
a 3.5-SE lower bound. Only the otherwise-zero v6 card/action × draw-source
head was updated; every inherited checkpoint byte stayed frozen.

This produced a real exact-policy gain on two fresh arena seeds:

| holdout | mirrored pairs | margin/match | match score |
| --- | ---: | ---: | ---: |
| seed 208020702 | 2,000 | **+2.14 ± 0.59 SE** | **51.14%** |
| seed 208020703 | 5,000 | **+1.54 ± 0.38 SE** | **51.09%** |

But replacing the checkpoint throughout the deployed rollout actor was
harmful. On the same 30 external deal-pairs used for a frozen-actor baseline,
the residual actor scored 41.67% versus 56.67% for the unchanged actor. The
same-deal difference was `−13.13 ± 7.26` points and `−15.00 ± 5.95`
match-score percentage points. This cleanly separates a stronger root policy
from a worse continuation model: a component win is not a deployment win.
The all-residual actor was rejected. A second deployment kept the residual
only at the root and froze the proven checkpoint for every continuation. It
also failed the match objective: over 50 fresh mirrored pairs (seed
208020704), it scored **46.0% ± 3.8% SE** (W-L-D 45-53-2), despite a noisy
`+0.56 ± 4.40` point margin. The root-only residual was rejected too.

The PPO trainer now supports balanced frozen-opponent population games,
full-action anchor KL, and v6-only warm-up so that this class of experiment can
be rerun safely. Small multi-seed candidates were neutral or negative, so no
PPO checkpoint was promoted in this cycle. The infrastructure is retained;
the failed checkpoints are not.

## Remaining high-value work

1. Give earlier-round MCTS one consistent utility. Final-round terminals are
   now exact, but rounds 0/1 still mix match-trained network leaves with
   current-round terminal margins. A separate round-margin value head or an
   explicit continuation through future deals is required.
2. Replace current-round rollout utility in early rounds with either remaining
   match simulation or a learned round-end continuation table.
3. Train v6 with the frozen-opponent/KL path over several independent seeds,
   select on fixed validation deals,
   and report only once on a locked final set.
4. Revisit learned-world search only together with a better continuation
   policy or a separately validated root method. The corrected posterior beats
   the card-count prior on held-out metrics, but a direct flat-rollout ablation
   lost match score; the decision audit therefore remains uniform.
5. Measure and report the 300-ply cap rate. Consider an explicit repetition or
   adjudication rule for evaluation.
6. Replace raw native C-struct persistence with a canonical endian-stable,
   checksummed parameter format.
7. Make complete PPO checkpoint generation invariant to `--threads`. Match
   evaluation and trajectory suit-map selection are thread-stable, but PPO's
   worker-local gameplay/reservoir streams and floating gradient reduction
   still make fixed-seed training depend on the configured worker count; use
   the same count when comparing training runs.

## Validation

The current snapshot passes:

```text
make
make test
make audit-test  # slower semantic probes; optional in the fast CI loop
make history-belief-test
make belief-eval-test
python3 tools/referee.py --selftest data/champion.bin 424242 --dumpfeat bin/dumpfeat
python3 tools/verify_transcript.py data/game.txt
```

The tests include randomized card conservation, unique semantic moves, suit
permutation round trips and ensemble equivariance, wager knowledge and
parameter/gradient tying, pile-order distinguishability, v4-to-v6 migration,
interaction-head isolation, model round trips, hybrid final-round utility,
robust sampling, fixed-cardinality marginal/sampler agreement, exact
joint truth scoring, hidden-state input scrubbing, exact policy-prefix
selection, visible-hand planning and regret guards, raw-policy phase gating,
exact all-action-core final-turn solving at the root and inside continuations,
semantic late-cycle equality, engine-fuse deck reserve, unique
without-replacement late outer panels with replanning off, full deck-two and
sampled deck-three recursive panels, globally allocated pile slots, exhaustive
90/990 bounded-root support independent of the recursive cap, retention of the
literal global policy argmax, deck-conditional forced-progress ranking, exact
deck-one leaves, one-sided frozen-opponent continuation, H=2/H=4 consensus and
practical-gain gating,
information-set-safe node seeds and cycle progress, bounded multi-stall
recursion, path-safe transposition reuse, policy fallback on budget/low-world
limits, rejection of unfinished ply-cap leaves,
perspective-scrubbed history inference, role-separated coherent continuation,
opponent-population actor masking and seat balance, v6-only byte preservation,
and one-versus-four-thread identity
for both ordinary and rollout matches. The
regression suites lock the reviewed W2, R2, Y2/W7, W3/W7, G5, ply-36 B10/Y10,
the rejected `+0.036` ply-42 candidate, the accepted `+20.622` ply-43 bounded
consensus, one-sided wager plies 59/61, final-deck ply 96, and discard-guard
positions.
