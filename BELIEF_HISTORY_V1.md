# Locked belief-history v1 accuracy campaign

This campaign answers one narrow question: can a standalone causal history
model assign more accurate probabilities to the opponent's hidden **exact
K-card hand** after observing their actions?

It is deliberately not a playing-strength experiment.  The maintained actor,
its policy/value network, its uniform rollout worlds, and the live policy-cost
campaign remain untouched.  Passing this campaign may publish a best belief
artifact, but cannot promote an actor.  Any later runtime integration is a new
candidate and must pass the unchanged reciprocal safety and final match gates.

## Fixed phase-one scale

| Split | Source trajectories (up to three rounds) | Shards |
|---|---:|---:|
| TRAIN, base head-only control | 262,144 | 4 ordered chunks |
| TRAIN, history residual branch | 65,536 | 1 |
| TRAIN, matched head-only branch | 65,536 | 1 |
| TEST | 65,536 | 16 |

Every causal mover state with a nontrivial exact-K label is retained. State
counts are data-dependent and are never fabricated, label-selected, topped
up, or dropped after evaluation. Every prefix from one source match remains
in one split. If the engine cap fires, valid observed prefixes are retained,
the capped round is recorded, and no later round is fabricated. TEST uses
sixteen fixed 4,096-source-trajectory shards.

The history-aware generator is the expensive stage. Preflight must demonstrate
at least 3.5 three-round trajectories per second on 100 source-free smoke
matches before any production root is touched. Every history or control job is
therefore capped at 65,536 matches and six hours. The 262,144-match base
control is four ordered resumable chunks with disjoint match ranges; its
optimizer state and exact prior checkpoint are bound across each handoff.

The candidate is a standalone `LCBHM1` causal action-history residual over a
262,144-match refit of the existing head. Starting from that identical frozen
base, a separate matched head-only control consumes the same 65,536 source
trajectories, public prefixes, exact-K labels, order, symmetries, exclusions,
and pre-truth source-manifest bytes as the residual branch. Both controls must
remain byte-identical to `champion.bin` outside `wbel`/`bbel` and preserve
wager tying. Every control and candidate command uses the development-fixed
`--base-alpha 1.0` explicitly. Thus the primary comparison gives both branches
the same additional labels; the 262,144-match base remains diagnostic only.
The verdict is scoped to predictive gain from the fixed action-history branch
over an equally trained state-only branch on this frozen self-play source
policy. It does not claim calibration to human opponents or isolate any one
individual action feature.

The history residual uses learning rate `0.01` and L2 `1e-7`. This setting
was selected before definition freeze on the burned development-only roots
`202608290508`/`202608290509`: 1,000 training matches followed by 2,000 shared
held-out matches. The documented broad-then-narrow sweep reached its best
all-state joint NLL at
`0.01` (`12.922342591`, Brier `0.157755341`); both smaller and larger neighbors
were worse. Those development labels and every model inspected in that sweep
are forbidden from the untouched TEST decision.

The matched head-only continuation uses learning rate `0.00015` and L2
`1e-7`. The modern control trainer was separately bracketed on the same burned
1,000/2,000-match development split at rates `0.00005`, `0.0001`, `0.00015`,
`0.0003`, and `0.001`. The interior winner `0.00015` reached all-state NLL
`13.074490098` and Brier `0.159653965`, improving on `0.0001` in all-state and
post-action NLL and Brier; every paired one-sided normal 99% lower bound for
those four development gains was positive. This sweep tunes the current
matched continuation, rather than transferring the earlier RL trainer's
`0.0003` setting. The separate 262,144-match resumable base refit remains at
its conservative `0.0001` rate; the short continuation sweep is not presented
as evidence about that much longer optimizer horizon.

The actual incumbent is also scored on every TEST state: `champion.bin` at
its maintained belief alpha 1.15. History is retained only if it passes the
complete proper-score bundle directly against both the matched head and the
incumbent. Otherwise the matched head is retained only if it directly beats
the incumbent; otherwise the incumbent remains best. No comparison is
inferred by transitivity. This chooses only a belief-accuracy artifact, never
a playing actor.

## Locked accuracy verdict

The untouched TEST panel has one candidate and one look.  The primary metric
is the strictly proper true-hand joint NLL. No arbitrary effect-size floor is
imposed: development evidence did not support one. For every ordered
replacement comparison, the point gain and its one-sided 99%
source-match-cluster lower bound must both be strictly above zero for
all-state joint NLL, post-opponent-action joint NLL, and all-state per-card
Brier. A single-step max-standardized-error bootstrap critical value provides
nominal asymptotic 99% familywise coverage across all nine direct selection
components; exact finite-sample coverage is not claimed, and marginal
percentile bounds are descriptive only. A zero original cluster standard
error cannot support a strict improvement and fails its bundle. Per-card Brier improvement,
exact-cardinality, opening uniformity, hidden-information invariance, suit
equivariance, wager tying, counts, and hashes are separate mandatory gates.
There is no SELECT split, fitted calibrator, or pre-TEST efficacy look. Model
bytes and hyperparameters freeze immediately after TRAIN; per-card Brier is a
separate calibration-sensitive proper-score TEST gate.

The residual uses the frozen 20-way base posterior at each state, reconstructed
through float log-weights and checked for marginal equivalence within `2e-6`.
Its sparse AdaGrad update applies the declared L2 term only to coordinates
active in that training state; it is not represented as lazy global weight
decay. This optimizer behavior is part of the frozen training contract.
Head-only
training uses one deterministic scheduled suit symmetry per state for runtime
feasibility, while every TEST head is evaluated with exact 20-way symmetry.
Those objectives are deliberately disclosed as different; equality of the
65,536 branch source manifests proves equal causal examples, not identical
augmentation objectives.

The seven canonical artifacts defining the 17 user-reviewed positions are
cryptographically bound. The source-free runtime contains only their 17-hash
manifest, never the reviewed states or comments. Before any history-training
or evaluation label is read, a firewall-local fieldwise projector constructs
the current mover information view without reading hidden deck bytes or the
opponent private hand, hashes it under all 120 suit relabellings, and rejects a
matching orbit. The chronological trace is deliberately not part of that key,
so every history converging to a reviewed current-view orbit is rejected. Only after a
frozen terminal verdict may a separate diagnostic addendum evaluate those
positions; that audit can never train, tune, select, or promote this model.

## One-shot execution

The definition is inert because
`data/experiments/locked_belief_history_v1_execution.json` is absent.  The
separate `belief-history-v1-definition.yml` workflow validates a clean,
detached tracked worktree of the inert parent with the complete GCC test suite and Clang
ASAN/UBSAN before launch preparation. It also exercises `prepare-execution`
and `guard-execution` against that worktree, then removes the temporary binding.
Both workflows are push-only. The definition workflow watches the exact inert
source/plan inventory only on `agent/correctness-and-policy-upgrade`; it has no
pull-request trigger because a cumulative PR diff would also match after the
later addendum-only push. The campaign workflow watches only the exact future
execution path, which is deliberately excluded from the definition workflow's
push filter. A launch must be a unique direct-child commit adding only the
canonical execution binding, must remain attempt 1, and cannot be manually
dispatched, retried, topped up, or reuse a touched root. Complete evidence is
independently replayed before an accuracy artifact can be accepted.

The authoritative machine-readable contracts are
[`locked_belief_history_v1_plan.json`](data/experiments/locked_belief_history_v1_plan.json)
and
[`belief_history_v1_exclusions.json`](data/experiments/belief_history_v1_exclusions.json).
