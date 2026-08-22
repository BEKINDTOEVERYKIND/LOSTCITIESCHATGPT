# Lost Cities AI

A from-scratch Lost Cities engine, neural network, and self-play training
pipeline for the full competitive game — three-round matches with cumulative
scoring — written in C with no external dependencies.

This repository began as a correctness-and-strength continuation of
[`BEKINDTOEVERYKIND/LostCities`](https://github.com/BEKINDTOEVERYKIND/LostCities)
at commit `4df68f7b2cbda7bd9ee160618693f6436b29e9ec`. The original repository is
unchanged.

## Current status

The maintained playing agent uses the exact 20-way champion policy for the
first 14 actions of each round. Beginning at zero-based `round_ply=14` (the
15th displayed action), it evaluates at most five policy-ranked moves with at
least 2% prior on 512 shared uniform hidden worlds. If that primary panel
prefers a move other than the raw policy leader, a fresh 512-world panel uses
balanced suit mappings that remain fixed for each complete trajectory. The
move changes only when both panels select the same leader; disagreement falls
back to the policy. This is the exact deployed specification:

```text
rolloutu:data/champion.bin:512:5:0.02:0:1:14:0:0:0:0:3.5:2:2:20:0:0:20:1:0:512:1:0:0:0:0:0:0:2:1:0:0:0:0:0:0:1
```

Two locked tests supported the promotion (each mirrored pair is two complete
three-round matches with seats swapped):

| comparison | seed | mirrored pairs | margin/match | match score | W/L/D |
| --- | ---: | ---: | ---: | ---: | ---: |
| consensus actor vs preceding ply-20 actor | 20320801 | 100 | **+3.60 ± 3.72 SE** | **55.0% ± 2.9% SE** | 109/89/2 |
| consensus actor vs independent external opponent | 20310801 | 100 | **+10.08 ± 3.98 SE** | **57.25% ± 3.02% SE** | 114/85/1 |

Both approximate 95% intervals for the direct test include parity; its
match-score result is favorable but not independently conclusive. The
independent external test is stronger corroboration: approximate 95%
intervals are +2.28 to +17.88 points and 51.33% to 63.17% match score. This
evidence, plus the method's repair of independently reproduced continuation
bias, is why it replaces the preceding actor. The network checkpoint itself
is unchanged.

The last turn is now solved as a complete decision rather than patched only
after the policy chooses a card. With one deck card remaining, every legal
play/discard action is scored under the active round or final-match objective
and paired with the deck draw that ends the round. The same exact solver runs
inside every rollout continuation. Consequently, whenever a decision with two
or more deck cards is actually compared by rollout, each candidate is evaluated
against optimal final-turn play; that comparison cannot prefer an earlier
stall by assuming that either player will blunder on the easiest turn.

That propagation has a deliberately narrower claim than full backward
optimality. The live actor still uses the raw champion policy before round ply
14, and a one-candidate shortlist needs no sampled comparison. Within an
ordinary deck-two/deck-three rollout, intermediate choices remain champion
policy choices before the exact one-card leaf. Thus every candidate that is
actually rolled out is scored with an optimal final turn, but the actor does
not yet solve every earlier stall information set. The bounded audit below was
kept out of live play precisely because using a stronger late rule only after
re-rooting would make those earlier rollout assumptions inconsistent.

A precommitted 20-pair exploratory screen isolated that propagation effect.
Both actors solved real one-card roots; the control merely preserved the
ordinary policy's action inside simulated one-card leaves. Exact propagated
play scored `+5.62 ± 6.74` points per match and `52.5% ± 5.7%` match score
(seed `20803011`). The direction is encouraging but the interval includes
parity, so this is supporting evidence for the exact method, not a claimed
independent strength promotion.

A separate 20-pair whole-actor screen against the raw exact-20 policy scored
`+17.45 ± 5.80` points per game and `60.0% ± 5.8%` match score. This is a
small exploratory result with wide uncertainty. It bundles the entire
maintained actor and is not an isolated exact-tail or bounded-resolver
ablation, so it is reported as a screen rather than promotion evidence for any
one component. The exact recorded specifications, seeds, aggregate results,
and provenance limitations for both screens are in
[`data/experiments/locked_strength_screens.json`](data/experiments/locked_strength_screens.json).

Late continuations also detect genuine repeated decision states at deck counts
three or lower. The active cycle identity is built from the current mover's
sanitized information set—own hand, known cards, and complete public
position—not the sampled opponent hand or future deck order. Interchangeable
wager IDs are collapsed and the ply counter is omitted so a literal public
pile shuttle cannot evade detection. On repetition, the maintained greedy
network policy is conditioned on the now-required deck draw, including its
learned card/action×draw interaction. This is stronger than choosing an
unrestricted move and swapping only its draw source. Optional sampled/planned
actors and the propagation-control ablation retain their selected semantic
action. A distinct non-repeating walk can also approach the engine fuse, so
the final `deck_left` slots use the same conditional deck policy. Pile shuttles
are therefore completed by the game rule rather than scored as if the
artificial 300-ply fuse were a terminal. Any impossible residual cap is still
counted, marks the comparison unresolved, and cannot override the deployed
policy.

For historical context, the preceding ply-20 actor scored **+14.70 ± 1.53 SE
points per match** and **56.5% ± 1.5% SE match score** against policy alone on
a fresh 200-pair holdout. That result remains evidence that rollout can improve
late play, but it is not the current actor's direct promotion test.

Independently, the optional visible-hand scheduler historically scored **+2.38 ± 0.33 SE
points per match** and **51.9% ± 0.4% SE match score** against the raw
exact-20 policy over 2,000 fresh mirrored pairs. It and the narrowly targeted
semantic candidates remain available as component tools, but predecessor-era
combined screens did not establish an additive gain over that rollout actor.
Planner-only was +1.60 ± 5.34 points in a direct 20-pair screen against the
preceding actor; the full planner+semantic tail was -6.06 ± 4.36 over 40 pairs.
They are therefore not silently enabled for live play. `play`, `showgame`,
`analyze`, and the replay generator use the measured configuration above;
`analyze` separately labels its higher-compute review recommendations.

The earlier three-slot/three-core shortlist (`root_width=3`) received the same
direct promotion test. An exploratory 20-pair screen was promising (+20.23 ±
7.29 points, 65.0% ± 5.3% match score), but the precommitted independent seed
did not reproduce a clear match-win gain: over 40 mirrored pairs it scored
only +1.04 ± 5.37 points and 51.2% ± 3.8% (W/L/D 41/39/0) against the
maintained actor. The screen is selection-biased and the independent result is
inconclusive, so it was not promoted to live play. The post-game audit now
retains those three top policy action cores but has five total slots, allowing
two policy-supported draw alternatives. At a real audit root with two or three
deck cards, a new bounded resolver exhausts the mover's full ordered
information-set support: at most 90 worlds at deck two and 990 at deck three.
That census is independent of the older recursive method. The maintained audit
sets `bounded_late_root=1` while leaving both `deck2_replan_*` fields at zero,
so no recursively redeterminized late panel can multiply its cost or overturn
its result. Its independent `bounded_late_min=1` requires more than one
objective point of gain in both horizons; it does not borrow or alter the
ordinary prefix-confirmation gate.

Candidate zero is always the literal global complete-move policy argmax, even
when the semantic shortlist would otherwise omit it. Whenever a continuation
is forced to draw from the deck, card/actions are ranked under that
deck-conditional policy rather than by replacing the draw source of an
unrestricted argmax. The same particles are carried forward through stalls,
one-card leaves are solved exactly, later information sets for the root player
may be improved, and nonterminal opponent choices remain frozen to the champion
policy. The resolver separately solves bounded horizons H=2 and H=4 and can
recommend a change only when both horizons choose the same complete root move
and both clear the configured practical-gain gate. A completed panel is the
authoritative gate for that audit root: it either authorizes that challenger
or immediately retains literal candidate zero. It does not pass a rejected
challenger to the older recursive evaluator, where a different approximation
could reverse the conservative decision. Only a panel that is unavailable
before completing falls back to the ordinary policy-focused rollout evaluator;
the disabled recursive method is not invoked. Retaining candidate zero is
deliberately conservative, not a proof that the policy move is optimal.

The locked deck-three ply-42 probe is deliberately conservative: both horizons
stably prefer G8+deck, but its H=2 gain is only `+0.036`, so the completed
panel authoritatively retains the policy baseline. At deck two on ply 43, both
horizons accept B10 followed by the
Yellow-pile draw—placing B10 on the blue expedition—with `+20.622` over the
policy baseline at each horizon. These
are restricted-model diagnostics, not an equilibrium solution. The opponent is
frozen, and upstream ordinary playouts still use the champion continuation
policy rather than re-rooting this panel at every prior turn. That dynamic
inconsistency is why the resolver is audit-only and has not been promoted to
live self-play.

The tracked [interactive match viewer](web/viewer.html) embeds the precommitted
random seed's self-play match under the maintained actor, with no result
screening or seed replacement: seed `209430960825253`, 145 plies, final score
P1 190–P2 175. The same seed was replayed only while repairing a
continuation-validity defect exposed by its diagnostics. The actor, deals, and
independent post-game audit use separate deterministic RNG streams. The viewer
keeps the move that was actually played
separate from the higher-compute review and shows the focused shortlist,
paired uncertainty, coherent-panel result, exact-terminal leaf count,
recursive late-search work, budget/cycle fallbacks, search depth, and the
experimental hand estimate at every move. Its tracked payload now includes a
match/build identifier, and the previously undefined diagnostics accumulator
that could stop the page at runtime has been removed; showcase generation also
installs the validated JSON and embedded viewer together, which makes stale or
partially updated match artifacts visible instead of silently presenting them
as the new match.

Its underlying network, `data/champion.bin`, combines an exactly
wager-symmetric projection of the strongest inherited checkpoint with
probability averaging over 20 exact suit relabellings. The policy alone
decisively beat the old raw `c8.bin` policy on two fresh holdouts:

| holdout | paired matches | margin/match | match score |
| --- | ---: | ---: | ---: |
| seed 180018 | 2,000 | **+21.24 ± 1.02 SE** | **61.4% ± 0.7% SE** |
| seed 190019 | 5,000 | **+19.81 ± 0.64 SE** | **60.5% ± 0.5% SE** |

Combined across 7,000 mirrored pairs, the direct result is approximately
+20.22 points per match and a 60.7% match score.

The wager projection alone measured approximately +1.20 ± 0.18 points and
50.62% over 35,000 pairs. The 20-way suit ensemble supplies most of the
remaining gain by averaging away arbitrary suit-slot preferences. A positive
policy-only qualification sent exhaustive 120-way root averaging to a locked
2,000+2,000-game full-actor holdout. It scored `49.7125% ± 0.6540% SE` with a
`-0.3955 ± 0.8091` point margin; its predeclared lower bound was `48.6039%`.
It therefore failed promotion, and the maintained actor remains at 20 root
symmetries. Every pair row, shard digest, evaluator binary, and the hardened
remerge are retained in
[`data/experiments/root120_stage2`](data/experiments/root120_stage2). This is a
failure to validate 120, not proof that it is intrinsically weaker.

No newly trained checkpoint cleared the promotion bar. In particular, three
attempts to distil the 20-way ensemble back into one fast network all became
weaker as imitation progressed. A later conservative residual-only experiment
also failed its predeclared locked gate: over 20,000 fresh mirrored pairs, its
two finalists scored 50.04375% and 50.06375% against the unchanged champion,
with lower confidence bounds below 50%. Those negative candidates are not
shipped.

Version-6 networks can learn full card/action × draw-source interactions and
the complete public order of each discard pile. Loading a legacy v3/v4/v5 model
zero-initializes only these additions and preserves its old outputs exactly.
`make` deterministically regenerates `data/champion.bin` from the tracked
`data/c8.bin`; its expected SHA-256 is
`af2b2c237d21f5ec15acbcba2fde3e45864a6e44af4ddb1ff6f3756fd687f417`.

The inherited checkpoint has exact positive-zero parameters in all 110
appended pile-order input rows and throughout the 720-way card/action × draw
interaction head. At each rollout decision the runtime derives an evaluation
plan from those actual parameter bits. It omits only work whose complete
parameter region is proven to be `+0`; any learned value, negative zero, NaN,
nonfinite activation, wrong network owner, or unrecognized flag value falls
back to the complete v6 path. Plans are derived by the runtime and kept only
for one decision while their network is immutable. The shortcut is not keyed
to a filename or model version and does
not change features, logits, probabilities, suit averaging, candidate
selection, RNG use, diagnostics, or moves. Regression tests compare the
ordinary and planned paths byte for byte, including full rollout panels. This
is a runtime optimization, not a playing-strength result.

On the fixed 120-way-versus-20-way maintained rollout workload, a balanced
four-block same-machine crossover reduced wall time by a geometric mean of
`1.439x` (block range `1.381x`-`1.495x`) while all 32 game rows and completion
records remained byte-identical. The final small-policy dispatch refinement
was then rechecked on two rollout blocks at `1.448x` geometric-mean speedup and
on a 3,000-pair five-way policy workload with identical raw results. Exact
commands, binary/model hashes, timings, and payload hashes are retained in
[`data/experiments/evalplan_performance.json`](data/experiments/evalplan_performance.json).

See [`IMPROVEMENTS.md`](IMPROVEMENTS.md) for implemented fixes, validation,
and remaining research work.

Lost Cities (Reiner Knizia) is a two-player imperfect-information card game.
Five suits of twelve cards (three wagers and the numbers 2-10). Each turn you
play a card to one of your own expeditions or discard it, then draw from the
deck or from the top of any discard pile (never the pile you just discarded
to). Expeditions must ascend; wagers must come before numbers. An expedition
scores `(sum of numbers - 20) x (1 + wagers)`, plus 20 if it holds eight or
more cards, and only if you opened it at all. A round ends when the deck runs
out; a competitive match is three rounds, totals win, and the first player
alternates by round.

## Layout

```
src/lc.[ch]         rules engine: state, move generation, scoring, match context
src/features.[ch]   information-set encoding (666 inputs, sparse + dense)
src/net.[ch]        three-headed network, forward/backward, Adam, save/load
src/heuristic.[ch]  hand-crafted projection evaluation (baseline, bootstrap)
src/planner.[ch]    exact scheduling of cards already visible in one hand
src/search.[ch]     determinized MCTS with network priors and values
src/rollout.c       rollout policy improvement over belief-sampled worlds
src/agent.[ch]      move-selection policies; belief-weighted determinization
src/match.[ch]      paired-deal match runner (single rounds or full matches)
src/spec.[ch]       agent command-line specs
tools/rl.c          PPO self-play trainer over full matches  <- the main trainer
tools/train.c       imitation / expert-iteration trainer (+ dataset dump/load)
tools/arena.c       head-to-head matches with error bars (-r 3 for full matches)
tools/ladder.c      round robin with fitted Elo
tools/analyze.c     per-ply JSON dump: state, values, policy, search, beliefs
tools/qpair.c       paired rollout Q for any named moves at a replayed position
tools/belief_eval.c frozen held-out exact-K belief metrics and card-count prior
tools/history_belief.py actor-aware offline hand posterior from scrubbed history
tools/planarena.c   isolated mirrored evaluation of the visible-hand scheduler
tools/robust_distill.c conservative confirmed-correction residual trainer
tools/mine_duel_states.py replay-validated external-corpus state proposer
tools/referee.py    numpy port of engine+net, verified to ~1e-6 against the C
tools/dumpfeat.c    parity reference dumper for the referee
tools/verify_transcript.py  independent replay/audit of printed transcripts
tools/showgame.c    replayable match transcripts, re-scored independently
tools/play.c        play against the agent in a terminal
web/viewer.html     self-contained analysis console (published as an artifact)
tests/test_engine.c rule, information, and match invariant tests
tests/test_runtime.c search, migration, feature, RNG regression tests
```

Build and test: `make && make test`.

### Full-strength competition handoff

The complete runtime source set is `Makefile`, `data/c8.bin`,
`tools/symmetrize.c`, and every `.c`/`.h` file in `src/`. Add
`tools/arena.c` for the supplied paired match executable; `tools/play.c` and
`tools/showgame.c` are optional interfaces. Run:

```sh
make data/champion.bin bin/arena
sha256sum data/champion.bin
```

The expected checkpoint hash is
`af2b2c237d21f5ec15acbcba2fde3e45864a6e44af4ddb1ff6f3756fd687f417`.
`data/champion.bin` is deliberately generated and gitignored; using the
tracked `data/c8.bin` directly skips the wager projection and is materially
weaker. Full strength requires the generated checkpoint and the exact
`LC_CHAMPION_AGENT_SPEC` in `src/spec.h`. Model paths in agent specs are
relative to the process working directory unless the match harness substitutes
an absolute path.

## What the agent knows and how it decides

**Information tracking.** The state records, beyond the public board:

* the deck count (a direct network input, with endgame flags),
* *known cards*: every card taken from a discard pile is drawn face up, so
  until it is played again the opponent provably holds it. The engine tracks
  this both ways, the encoder exposes both planes ("cards I know they hold",
  "cards of mine they know about"), and the world-sampler treats known cards
  as certainties, never as unknowns.

**Learned opponent inference.** A third network head scores every card whose
location the player cannot pin down. Those scores are converted into one
coherent fixed-cardinality distribution: a hand with exactly K unknown cards
has probability proportional to the product of its card weights. Dynamic
programming gives analytic inclusion probabilities that sum exactly to K and
an exact sampler from the same joint distribution. The diagnostic viewer
averages belief logits over 20 suit relabellings and applies a held-out
self-play calibration temperature (`alpha=1.15`). Opening positions are a hard
uniform-card-count invariant, before either opponent has supplied behavioural
evidence.

This learned posterior remains experimental. The authoritative decision audit
samples hidden hands from the uniform card-count prior, so questionable belief
calibration cannot distort its Q comparisons. Learned-world rollout and MCTS
remain available for research. A post-fix live-actor screen using only the
new learned worlds scored `+1.57 ± 6.92` points but just `40.0% ± 6.9%` match
score over 20 fresh mirrored pairs (W/L/D 16/24/0). Better inference metrics
therefore did not translate into better play, and uniform worlds remain the
deployed default.

For offline review, `tools/history_belief.py` also provides an actor-aware
posterior. It reconstructs only the observer's original hand, public actions,
and that observer's own deck draws, then rejection-samples deals whose visible
action prefix is plausible under the frozen actor. The perspective-scrubbed
wire format never contains the opponent's hidden cards or future draws. On the
preserved showcase at ply 4, 100,000 proposals produced 4,433 accepted worlds:
Y4 was estimated at 21.93%, Y9 at 22.94%, and Y10 at 22.96%. That fixes the
obvious old inversion in which Y4 led while Y9/Y10 fell off the displayed
list, without pretending the three yellow ranks should be far apart. This is
an offline actor-conditioned diagnostic, not an input to the playing actor.
Version
1 accepts only a provably deterministic raw-policy prefix and rejects
temperature sampling, rollout decisions, or planner/semantic-enabled actors
rather than silently assigning them the wrong likelihood. It also verifies
that the inference checkpoint is byte-identical to the recorded actor model.

**Match play, not just round play.** The network's inputs include the round
number and the cumulative score difference. Training is staged: early PPO
rewards `margin + win bonus` (the dense margin signal teaches point play),
and the finishing phase switches to `0.05 x margin + 50 x match result`
(--mw / --winbonus in tools/rl.c) so that winning is nearly all that matters.
Being 40 up in round three genuinely changes what the policy optimises:
protect the win rather than maximise expectation, and gamble when behind.
Only actual round index 2 switches to the explicit win-dominated rollout
objective; one- and two-round diagnostic configurations intentionally retain
their historical semantics.

**Exact game symmetries.** The three wagers in a suit are one observable card
type, despite having separate physical IDs in the 60-card engine. Legal move
generation now emits one semantic wager action; checkpoint parameters and
training gradients for the three copies are tied exactly. The five suit names
are also strategically interchangeable. At inference, `:5`, `:10`, `:20`,
or `:120` after a policy spec averages probabilities over increasingly large
exact suit-permutation groups. The default 20-element affine group is
2-transitive and captures nearly all of the measured benefit at one-sixth the
cost of all 120 permutations. These are rules-preserving transformations, not
heuristic augmentations. The 20-way mode is invariant to that affine subgroup,
while only the 120-way mode is invariant to every possible suit permutation.
MCTS averages only at the root. Rollout has a separate
`playout_symmetries` setting. It can either average that group exactly at
every continuation decision, cheaply draw a new group member at each decision,
or assign one balanced group member to an entire hidden-world trajectory.
Policy-action sampling is a separate diagnostic mode; it is no longer
accidentally coupled to random symmetry sampling. The maintained actor and
audit use the cheap per-decision mode for the primary panel and the coherent
trajectory-fixed mode for an independent confirmation panel whenever the
primary panel proposes changing the policy move. The bounded actual-root panel
ranks its candidates with the complete configured suit ensemble. In optional
low-level recursive experiments, selected-path late nodes do the same, while
hypothetical descendants use one cheaper group member chosen deterministically
from their current mover information state and panel domain, never from an
outer hidden-world RNG position.

**Stalling.** Drawing a useless card from a pile to deny the opponent a turn
of deck progress is in the action space, and nothing hand-crafted decides it:
the policy learns from match outcomes when a stall is worth more than the
tempo it gives away. The engine's only concession is a 300-ply safety cap per
round (real rules allow unbounded mutual pile-recycling), which sane play
never approaches.

## Why the architecture looks like this

The value-only approach fails measurably in this game: candidate moves differ
by one or two points while a finished round's margin has a standard deviation
near 60, so no value function learnable from outcomes can rank moves by
one-ply lookahead — a near-perfect distillation of the hand-crafted
evaluation (4.6 pts RMS) still lost by 71 points a game to the evaluation it
copied, and search built on that value function was no stronger than its own
prior. What works is predicting decisions directly. The policy retains shared
card-and-disposition and draw-source terms, then adds a full 720-way
interaction residual. Consequently, whether drawing from Yellow is preferred
over the deck can depend on which card is played or discarded. The value head
serves as a PPO baseline. In complete self-play states and sampled search
worlds it is projected to the antisymmetric critic
`0.5 × (V(player) − V(opponent))`, removing common head bias and guaranteeing
opposite values for the two seats. A real hidden root is never evaluated from
the opponent's perspective; centering happens only after a legal
determinization. Search is principally done by *rollouts*: play a
policy-ranked shortlist to the end of the round in shared hidden worlds.
Pairing every candidate on the same worlds sharply reduces uncertainty in the
move-to-move difference. The audit reports the standard error of each Q mean
separately from the paired standard error of each difference.

Training is: imitate the heuristic for a sane start (it knows nothing about
match context or beliefs), then PPO over full three-round matches with the
belief head learning on the side. The trunk is 666 -> 512 -> 256.

## Historical upstream results

All numbers are 3-round paired matches (each triple of deals played twice with
seats swapped) unless stated. Margins are total match points; "wins" are match
wins with draws counting half. The table and discussion below are retained as
upstream experiment history; unlike the current-status table above, many used
the older leg-level win-rate error estimate. In this section, “champion” and
“shipped model” refer to the inherited upstream `c8.bin`, not the new
symmetry-ensemble configuration.

| comparison | margin/match | match wins |
| --- | ---: | ---: |
| win-training continuations vs each predecessor | 63.6% / 61.5% / 57.9% | (500/400/400 pairs) |
| shipped champion vs heuristic | **+173.7 ± 3.6** | **98.0%** (300 pairs) |
| margin-trained champion vs imitation start | +204.8 ± 4.0 | 96.6% (400 pairs) |
| rollout search vs raw policy (margin-trained) | +26.5 ± 4.8 | 63.3% (60 pairs) |
| belief-sampled vs uniform worlds | −3.7 ± 5.4 | 45.8% (60 pairs) |

The shipped model is the *win-trained* one. Training ends with a phase whose
return is `0.05 x margin + 50 x match result`, so winning dominates: a 5%
chance to steal the match outranks a certain narrow loss even at a terrible
expected margin, exactly as competitive play demands. That phase converted
margin into wins -- against the heuristic it gives back ~15 points of margin
relative to the margin-trained champion while winning matches it previously
lost, and it beats that champion head-to-head 63.6% of the time.

The training trajectory (evaluated vs the frozen imitation start every 3
iterations): the PPO run climbs from parity to a peak of ~+205/match around
iteration 51, then over-optimises into stall-heavy play and falls back to
+66 by iteration 130. The shipped model is the peak checkpoint, selected by a
5-way 300-pair tournament and confirmed head-to-head against its neighbours
(+5.5 ± 1.9 over iteration 45, +2.9 ± 1.8 over iteration 48). Checkpoint
selection matters: the *last* iterate of a PPO run is not the best one.

The belief-sampling ablation is a null result: the rollout agent is no
stronger (and no weaker, within noise) when its imagined worlds come from the
learned posterior instead of uniform sampling. The likely reason is that the
policy driving the playouts shares its trunk with the belief head, so the same
inference already shapes every playout decision; making the sampled hands more
realistic adds little on top.

**Belief quality:** the historical analysis contained only the top 14
predictions from one game, so it was selection-biased. The replacement
calibrated its temperature on 12 unscreened matches and evaluated every
uncertain card on 20 separate matches. Twenty-way averaging with
`alpha=1.15` improved Brier score from 0.17538 for the card-count prior to
0.16194 and log loss from 0.53272 to 0.49622. That is useful evidence, but
still a small validation set and not a demonstrated playing-strength gain.
The UI therefore labels the hand estimate experimental, and authoritative
rollout continues to use uniform fixed-card-count worlds.

`bin/belief_eval` makes that validation reproducible against the corrected
joint distribution. It generates deterministic, fixed-seed games with a
separately loaded frozen exact-symmetry policy argmax (`--actor-net`, default
`data/champion.bin`), so changing the evaluated checkpoint cannot change the
scored states; beliefs never select an action. Before
either policy or belief inference, an explicit information-view boundary
removes the opponent's unknown hand and future deck order. Referee truth is
used only afterward to score the exact unknown K-card subset. The report gives
joint NLL per state and uncertain card, Brier score, tie-correct within-state
AUC, tie-correct top-K recall, exact uniform card-count baselines, and
finite-cluster sandwich standard errors for those pooled estimators, clustered
by complete three-round match. On the default 20-match
validation seed, the maintained wager-projected champion scores joint NLL/state `13.0302`
versus `14.0770` for uniform, Brier `0.16353` versus `0.17721`, within-state
AUC `0.6638`, and top-K recall `0.4739` versus `0.3369`. These are inference
metrics, not evidence that using learned worlds improves match play.

## When to search, and when the policy alone is enough

Early low-compute experiments established that policy confidence alone is not
a reliable phase gate. The maintained agent instead uses a round-ply cutoff
selected by direct match play: rollout is off for zero-based plies 0–13 (the
first 14 displayed actions) and on beginning with the 15th action.
At each searched position it evaluates at most five policy moves with at least
2% prior on 512 shared uniform hidden worlds. Ordinary prefix moves are not
accepted through a noisy significance threshold: if the primary numerical
leader differs from the raw policy, a fresh 512-world panel with balanced,
trajectory-fixed suit mappings must select the same leader. Otherwise the
actor falls back to the raw policy. The older 3.5-SE/two-point gates remain
available only for purposefully added low-prior research challengers, none of
which are enabled in live play.

The optional research tail `:16:12:1` adds two focused,
information-set-respecting layers. With
at most 16 deck cards, an exact subset scheduler computes the best guaranteed finish
from the current visible hand and may reorder commuting first plays only when
it preserves at least 12 points of still-unseen lower-card options. If the
policy's first play is not part of any best guaranteed schedule, a
conservative regret guard replaces it only when spending that turn loses at
least two already-secured points. Separately, semantic mode considers a small
number of purposeful pile draws attached to one of the top three
card/disposition actions, plus a one-sided isolated wager discard when the
opponent cannot score it directly. That discard is not assumed safe: the
opponent can still pick it up to stall. In post-hoc review outside the
ordinary search window it
was evaluated only against the deployed baseline, not alongside the preceding
actor's ordinary shortlist,
and receives a primary cap plus a fresh confirmation batch of 16,384 worlds.
Only support from both batches can supersede the generic discard guard. This
does not evaluate the Cartesian product of legal plays and draw sources. A
one-candidate shortlist still skips continuation forwards entirely. This
tail remains an audit/component configuration until it beats the maintained
actor in a new locked direct test.

`draw_root_deck_max` and `draw_playout_deck_max` independently enable a
narrower draw-source repair at the deployed root and inside rollout
continuations. The policy still chooses exactly one card and play/discard
action; only legal draw sources attached to that action are compared. A pile
top is public and scheduled exactly. The deck alternative is averaged over
every card that can be next from the mover's information set, never the hidden
top card in a determinized world. At `draw_root_deck_max=4` this focused
late-tempo planner beat the frozen exact-20 policy by **+2.32 ± 0.22 points per
match** and **51.14% ± 0.26% match score** over 2,000 fresh mirrored pairs.
It also remained positive with the independently trained teacher-residual
checkpoint: **+1.99 ± 0.33 points per match** over 1,000 pairs. Separate root
and continuation thresholds prevent a root-only result from silently changing
the rollout world model. That distinction mattered in full-actor testing:
root-only repair changed no win/loss outcome over 30 external deal-pairs and
added `+0.65 ± 1.44` points on those same deals, while continuation repair lost
`−11.37 ± 7.76` points and `−16.67 ± 9.34` match-score percentage points over
15 pairs. Both controls remain off in the maintained actor unless a future
locked match-score test clears the promotion bar.

The current consensus actor and its preceding rollout actor were each locked
before their corresponding holdouts:

| comparison | mirrored pairs | margin/match | match score |
| --- | ---: | ---: | ---: |
| consensus ply-14 actor vs preceding ply-20 actor, seed 20320801 | 100 | **+3.60 ± 3.72 SE** | **55.0% ± 2.9% SE** |
| consensus ply-14 actor vs external opponent, seed 20310801 | 100 | **+10.08 ± 3.98 SE** | **57.25% ± 3.02% SE** |
| preceding ply-20 actor vs policy alone, seed 950005 | 200 | **+14.70 ± 1.53 SE** | **56.5% ± 1.5% SE** |

This is why rollout remains an actual decision-maker late in each round, not
merely a UI diagnostic. Broad search remains off in the early game: phase
screens did not justify its compute there, the horizon is longest there, and
the human-reviewed failures showed that more worlds can make a biased
continuation estimate more confidently wrong. The post-hoc audit's
one-sided-wager trigger is a specific public-information probe, not a general
reversal of that result or an unmeasured live-play override.

**The candidate floor cuts both ways.** Ordinary candidates come from the
policy, and moves below a 2% prior are pruned. Semantic mode adds only the few
rule-derived variants described above; it never opens every legal move.
Without one of those public-information signals, a near-certain policy can
still leave search with one candidate and no way to overrule it. A
replayed position made this concrete: the policy put 100% on a discard, and a
paired re-evaluation (tools/qpair.c, 4000 shared worlds) showed a wager it
had written off was better -- +2.9 ± 0.6 with the net that played the game
(robust to sampled playouts and to search-driven continuations). The leak
family recurs, smaller, in the inherited champion: in the analogous position
of the embedded game its written-off wager play measures +0.6 to +1.0 over
the 100%-prior discard under three estimators -- real, but below what the
older 96-world play-time search could resolve, which made it a training target
rather than a blanket search override. But *forcing*
the floor open is worse than the disease: full rollout with `min_cand` 3
scored only **42.8% ± 3.5%** (-10.8 ± 4.2/match, 100 pairs) against the
baseline. In that older setup, a 96-world Q difference carried ±2-4 points of
paired noise; most
true gaps between a near-certain policy move and its alternatives are
smaller than that, and taking the argmax of several noisy estimates
systematically flatters the winner. So `min_cand` selects among noise.
`eval_cand` can report extra policy-ranked diagnostics without making them
eligible, but the maintained viewer leaves it off: analysis compute is
reserved for the top-policy prefix. Human questions about a written-off move
belong in the locked `qpair` probe suite, not in every ply of the UI.

**Where the search earns its keep: late, not early.** Earlier exploratory
phase screens (using the older low-world method) showed the same qualitative
pattern:

| search window (plies of each round) | margin/match | match wins |
| --- | ---: | ---: |
| everywhere (150 pairs) | +10.6 ± 3.0 | 51.5% ± 2.9% |
| only plies >= 14 (150 pairs) | **+11.4 ± 2.4** | **56.2% ± 2.9%** |
| only plies < 14 (200 pairs) | +4.6 ± 2.4 | 53.4% ± 2.5% |
| only plies < 14, forced 3 candidates (200 pairs) | -4.7 ± 3.1 | 50.1% ± 2.5% |

Those runs are historical evidence, not the production strength claim. Their
wide intervals motivated the larger world count, independent confirmation,
corrected continuation policy, and the current two-panel deployment after the
first 14 actions of each round.
Early-round moves are often near-equivalent in true value, so there is little
for rollout to find; late-round positions diverge more sharply and have much
shorter, more accurately evaluated horizons.

**The search reports its own noise and reference.** Every reported Q carries
its own standard error. `delta_vs_baseline` is paired against candidate zero;
`delta_vs_reference` is paired against the move that an optional low-prior
challenger must actually beat. The default deep audit reports policy evidence
but deliberately skips rollout for the first 14 actions of each round; search
starts at zero-based `round_ply=14`, the 15th displayed action. A 2,048-world ply-8 probe
still produced a confident low-prior override, confirming that more worlds do
not cure the long opening horizon. From that 15th action it requests up to
2,048 primary worlds on at most three distinct top-policy card/play-discard cores with at
least 1% aggregate prior, plus up to two policy-supported alternative draws in
a five-slot budget.

At a uniform-world root with two or three deck cards, the outer primary,
trusted-prefix-confirmation, and challenger-confirmation panels draw ordered
information-set assignments without replacement. This applies even when the
recursive resolver is disabled. A 2,048-world request therefore becomes one
complete 90-assignment census at deck two and at most 990 unique assignments
at deck three; publicly known opponent cards can make either support smaller.
Separate deterministic panel domains choose their own unique order. This outer
census prevents duplicate worlds from masquerading as additional evidence.

Concretely, the maintained live actor requests 512 outer worlds: deck two is
still the complete 90-assignment census, while deck three uses 512 unique
assignments from support of at most 990. The deep audit requests 2,048, so it
exhausts both supports. Its historical recursive-replan fields are both zero.

At an actual audit root with two or three deck cards, the bounded late resolver
exhausts the complete ordered support:
at most 90 mover-view assignments at deck two and 990 at deck three, with
publicly known cards reducing those counts. The independent
`bounded_late_root=1` flag enables this panel while recursive replanning remains
off. `bounded_late_min` is a separate practical-gain floor and defaults to one
point.

The resolver groups moves into semantic card/play-discard cores and globally
assigns the remaining slots to the strongest supported pile variants. It also
retains the literal global complete-move policy argmax as candidate zero,
regardless of the core ranking. Where progress through the deck is compulsory,
the action ranking is recomputed conditional on a deck draw; it never takes an
unrestricted policy argmax and merely swaps its draw source. Particles remain
coupled across stalls, the root player's later information sets receive the
same bounded policy improvement, opponent nonterminal information sets use the
frozen champion policy, and every deck-one leaf evaluates all legal final
card/actions exactly.

Two separately bounded solutions, H=2 and H=4, must identify the same
complete root move. A change is authorized only if that move also beats the
literal policy baseline by more than the configured practical threshold in
both horizons. The locked deck-three ply-42 panel therefore retains policy even
though G8+deck is stable: its H=2 gain is only `+0.036`. The deck-two ply-43
panel accepts playing B10 to the blue expedition followed by the Yellow-pile
draw, which gains `+20.622` in both horizons. This is a one-sided
restricted-model policy improvement, not an equilibrium computation or a
promoted playing rule.

The older recursive evaluator remains available only to low-level experimental
specifications. It can redeterminize at later two- or three-card states under a
separate world/depth budget, but the maintained audit does not enable it. If the
bounded actual-root panel is unavailable before completing, the ordinary
policy-focused evaluator resumes instead. A completed rejection returns the
literal policy baseline immediately.

That experimental recursive method is deliberately bounded rather than
disguised as exact game-tree search. Its actual selected path uses the complete
configured suit ensemble and up to three semantic cores; hypothetical
descendants use one deterministic, information-state-keyed group member and
their top core. These details no longer describe the maintained audit, whose
recursive counters must remain zero.

An all-depth exact-20 variant was also attempted on both locked late probes.
Each run exceeded 25 minutes without completing and was stopped, so it was
rejected as impractical rather than reported as a measured playing result.

The old single-frontier ply-43 assertion was invalid: its exact local arithmetic
was B10 `-21.511`, Y10 followed by immediate progress `-1.511`, and Y10 on the
one-stall frontier `-18.933`. The replacement regression does not reuse that
truncated comparison. It exhausts all 90 ordered worlds and locks the finite
H=2/H=4 consensus above, including the B10-to-blue-expedition/Yellow-pile-draw
continuation. The resolver still remains experimental and audit-only because
frozen opponent responses and the champion continuation policy used by prior
plies make the combined evaluator dynamically inconsistent. Making its completed
root panel authoritative prevents a second approximate evaluator from
contradicting its gate; it does not remove that upstream inconsistency or
establish optimal play.

For low-level recursive experiments, strict candidate-trajectory budgets,
information-set-safe state keys, panel-local caches, and cycle closure prevent
private-root leakage or a compute limit from becoming a game rule. Those paths
remain covered by focused tests but are not part of the maintained audit.

The analyzer and viewer separately report the bounded root resolver's support,
H=2/H=4 values, stability, practical-gate result, frozen-opponent work, and
whether it authoritatively retained policy or selected a challenger. Historical
recursive counters remain in the schema for low-level experiments and are
asserted to be zero in the maintained audit. Both the live actor and maintained
audit keep the two `deck2_replan_*` fields at zero; only the audit enables
`bounded_late_root`.
Reported Q standard errors measure outer-world variation conditional on this
deterministic searched policy. They do not treat repeated inner searches as
independent samples or include uncertainty from choosing another inner-search
seed.

When the primary leader differs from the policy, a fresh panel requests up to
2,048 worlds with balanced, trajectory-fixed suit mappings; a late root instead
uses the unique support described above. It must confirm that proposed move
over candidate zero by both two paired standard errors and at least one
objective point. It need not reproduce an exact argmax among statistically
tied improvements; otherwise two good stalls can cancel each other and restore
a clearly bad deck-ending baseline. This rejects weak evidence without
confusing leader jitter with failed confirmation. In the final round the dump also reports
each candidate's exact match-score fraction over the playouts. Selecting by that win
fraction is available (`win_q`) but off by default, because it measured no
better than margin selection -- 50.4% ± 0.8% match wins pooled over 2,000
head-to-head pairs (a 300-pair run at 48.0% ± 2.0% and a 1,700-pair
confirmation at 50.8% ± 0.9%), while costing 1.3 ± 0.6 points of margin:
decided finals tie on win%, and the historical 96-world test made close-final
win fractions noisy binomial estimates. The win-trained policy already carries the
clutch behaviour into every playout. The same lesson as the candidate
floor, from the other direction: at fixed compute, the statistically
efficient objective beats the theoretically right one.

**Historical frozen-showcase checks.** The earlier detailed review used its
original random
trajectory, not a regenerated match. At ply 7, three independent high-world
continuation families agreed that play G5 and play W5 are effectively tied:
their pooled exact-policy estimates were 29.49 ± 0.38 and 29.60 ± 0.39,
with W5 only +0.11 ± 0.35 over G5. Premature Y4 was decisively worse at
7.63 ± 0.37, or -21.86 ± 0.40 against G5. The correct conclusion is that the
original G5 is defensible, W5 is equally plausible, and Y4 is not.

The historical focused one-sided-wager trigger corrected two later reviewed
positions without opening the whole move list. Under that 16,384-world audit,
discard Bx then take R beat the deployed baseline at ply 59 by
+5.16 ± 0.40 in the primary audit and +4.77 ± 0.40 in fresh confirmation.
At ply 61 it measured +4.26 ± 0.39 and +4.04 ± 0.40 respectively. The earlier
frozen artifact retained a directionally matching 4,096-world ply-59 analysis;
these values are preserved here as experiment history rather than presented as
part of the current top-policy-only viewer.
With one deck card left at ply 96, the exact rule
chooses the same play followed by a deck draw: ending the round weakly
dominates taking a pile and gifting the opponent another turn. These are
specific validated corrections, not evidence that every rollout estimate is
now reliable.

Two reviewed blind spots remain explicit. At ply 17, a 4,096-world comparison
favored play R8; discard B5 then take R measured -2.20 ± 0.73 against it under
that continuation model. Ply 17 is now inside the live search window. Requiring
two different continuation panels to agree reduces orientation instability,
but it cannot prove per-position optimality or eliminate a bias shared by both
panels. At ply 25, the optional semantic generator admitted taking Bx, but
uniform-world rollout still rejected it (-5.52 ± 0.58 in the locked probe).
Conditioning hidden worlds on the
opponent's public behavior moves that estimate substantially, but not yet
enough for a stable playing rule; it remains a belief/continuation-training
target rather than a hand-authored correction.

**Dead-discard pruning was tested upstream** and measured strength-neutral
over 300 pairs. It is now off by default: the replacement can cover a
different discard pile or expose a different buried card later, so this is a
search-focus heuristic rather than mathematical dominance. It remains
available as an explicit agent-spec option for controlled experiments.

**Expert iteration** (tools/train.c --gen selfrollout, Q-softmax targets
over searched-plus-advisory candidates): twelve iterations from the
champion moved every targeted confidently-wrong prior -- four probe
positions went from 0% prior on the better move to 25-36% -- and flipped
the sequencing watch-probe toward optimal ordering. It did not, however,
produce a stronger agent: 48.2% ± 2.9% search-vs-search against the
champion; the blanket soft targets give up more sharpness than the fixed
leaks return. The refined recipe (corrections only at statistically
significant search-policy disagreements, KL-anchored elsewhere) is the
open training direction.

**Historically, the significance-gated override was a measured gain** -- the discipline
blanket forcing lacked. Advisory candidates (eval_cand) may take the move
only when they lead the best policy-plausible candidate by more than
`override_k` paired standard errors, the statistical signature of a
confidently-wrong prior rather than noise. A/B at k=3 with four evaluated
candidates: **+6.35 ± 1.88 per match, 52.5% ± 2.0%** over 300 pairs
against the previous maximum-strength config -- the first strength
improvement since the shipped champion, at ~1.7x search compute.

Expert review of an override-enabled game then exposed two further gates
that purposefully added low-prior challengers need. (1) `override_min` points:
the SE gate is
world-count-dependent in the wrong direction -- more worlds shrink noise
but sharpen *bias*, so at 512 worlds a 3-SE gate fired on ~1-point stall-
and discard-flavoured playout bias; in the reviewed game every override
gap over 4 points was one the reviewer endorsed and every graded blunder
was under 2.5. (2) Sampled confirmation: the surviving gap must also hold
at half the floor under stochastically-sampled continuations, because
deterministic playouts repeat knife-edge downstream decisions across all
paired worlds -- one position produced a +5.0 ± 0.14 argmax gap for
discarding over a free scoring play that sampling collapsed to +0.6.
Regenerating the reviewed game under the full gates removed every
reviewer-graded blunder while keeping the overrides the reviewer agreed
with.

The old `playout_sample=1` mode sampled full policy actions and random suit
symmetries together. That is retained only as a robustness ablation. Mode `2`
separates the useful cheap symmetry approximation from action sampling: each
downstream decision draws one requested symmetry and takes its greedy move.
Mode `3` instead fixes a balanced symmetry mapping for the complete sampled
trajectory. The maintained method uses mode 2 for discovery and independent
mode 3 for consensus, without changing the evaluated player into a much weaker
high-entropy policy. When the optional recursive late evaluator is enabled,
its first and actually selected-path deck-two/deck-three shortlists use the
full requested ensemble. Hypothetical top-one descendants use their
information-state-keyed single group member; the surrounding ordinary
continuation still follows the selected bounded mode.

The maintained default is the measured two-panel ply-14 rollout actor printed
at the top of this README. The scheduler and semantic challenger generator
remain available in explicitly labelled component tools, but are enabled in
neither live play nor the default top-policy-only post-hoc audit. Objective
mode `2` remains available for research and uses
round margin in rounds 0/1 plus
the champion's `0.05 × final match margin + 50 × result` return only in real
round index 2. The measured production actor uses objective mode `0`;
changing the objective would require another locked strength test. The
last-round-only win objective is intentionally preserved. The maintained
high-compute post-hoc audit spec is printed into every analysis artifact by
`tools/analyze`.

The production-setting objective-mode comparison is precommitted in
[`data/experiments/locked_objective_mode2_plan.json`](data/experiments/locked_objective_mode2_plan.json).
An excluded 50-pair runtime block and its exact command are recorded separately
in [`data/experiments/objective_mode2_development_runtime.json`](data/experiments/objective_mode2_development_runtime.json);
it is operational provenance, not strength evidence, and does not change the
default actor.

## Reproducing

```
# 1. imitation start (heuristic plays the rounds; ~15 min on 4 cores)
./bin/train --gen heur --gen-switch 99 --rounds 3 --iters 4 --games 2500 \
            --steps 15000 --batch 512 --lr 1e-3 --tau 0.5 --suit-augment \
            --out data/m0.bin

# 2. PPO over full matches with belief learning (~2 h on 4 cores)
./bin/rl --init data/m0.bin --ref policy:data/m0.bin --rounds 3 --winbonus 15 \
         --iters 130 --games 900 --epochs 1 --lr 2.5e-4 --ent 0.003 --out data/m1.bin

# 3. finishing phase: win-dominated reward (~40 min)
./bin/rl --init <peak of step 2> --ref policy:<same> --rounds 3 \
         --winbonus 50 --mw 0.05 --iters 80 --games 900 --epochs 1 \
         --lr 1.5e-4 --ent 0.002 --lambda 0.9 --out data/w1.bin
# then select the checkpoint by MATCH WIN RATE over a 500-pair validation

# Optional conservative v6 research warm-up against a frozen champion.
# Use an even game count; this is a training capability, not a promoted model.
./bin/rl --init data/champion.bin --gen-opponent policy:data/champion.bin:0:20 \
         --opponent-mix 0.5 --anchor data/champion.bin --kl 0.5 --v6-only \
         --games 1000 --rounds 3 --out data/v6-population-candidate.bin

# Optional, default-off PPO suit augmentation.  One exact-group mapping is
# fixed for each complete match; learner actions are mapped back before the
# canonical engine applies them.  Valid group sizes: 1, 5, 10, 20, 120.
./bin/rl --init data/champion.bin --trajectory-symmetries 20 \
         --games 1000 --rounds 3 --out data/trajectory-augmented.bin

# Optional belief calibration with every trunk/policy/value byte frozen.
# This mode uses the same exact-cardinality hand likelihood as deployment.
./bin/rl --init data/champion.bin --belief-only --bw 1 \
         --trajectory-symmetries 20 --games 1000 --rounds 3 \
         --out data/belief-head-only.bin

# Optional continuation-role PPO.  The immutable champion plays plies 0..13,
# then its exact production shortlist supplies a gradient-free ply-14 root.
# PPO learns only later decisions in uniformly sampled hidden worlds.
./bin/rl --init data/champion.bin --continuation-start 14 \
         --continuation-root data/champion.bin \
         --anchor data/champion.bin --kl 0.5 --lambda 1 \
         --rounds 3 --games 1000 \
         --out data/continuation-candidate.bin
```

These PPO capabilities are opt-in research tools and are disabled by default.
`--belief-only` requires positive `--bw`, cannot be combined with `--v6-only`
or anchor-KL optimization, and applies weight decay only inside the belief
head. Continuation mode accepts only the maintained handoff at ply 14. It uses
an independently loaded, required `--continuation-root` checkpoint, so resuming
the learner cannot silently replace the frozen root actor. The root is loaded
byte-for-byte without an implicit wager projection. Before training, canonical
path and existing-file identity checks reject `--out` and every generated
`.itN` checkpoint if it could overwrite the root, learner initialization, or
anchor through a direct path, symlink, or hard link. It then uses
the shared production flat-policy admission (width five, 2% floor, minimum
one). When at least one nonbaseline move is admitted, candidate zero is chosen
half the time and the other half is divided uniformly over those challengers;
a singleton shortlist necessarily uses candidate zero. The selected root and exact
deck-one solution receive no policy row; both seats' downstream learner moves
do. Downstream behavior and PPO condition on production's maintained
dead-discard mask, including after suit augmentation: masked moves have zero
behavior, PPO, and entropy mass, while the full-legal anchor KL can still
protect their logits. The shared semantic late-cycle tracker and cap-reserve
rule force the conditionally best deck draw without an actor row, ensuring
completion through the real deck rule; an unfinished capped round is rejected.
Continuation mode defaults to and requires `--lambda 1` exactly. Its target is
therefore always the production audit's mode-0 completed-round margin, with the
value head serving only as an action-independent advantage baseline. Belief
loss and standalone-policy evaluation default off because the eventual
checkpoint must be qualified specifically in the dual-network rollout
continuation role. Match evaluation and trajectory suit-map selection are
thread-stable, but complete PPO training is not yet checkpoint-identical when
`--threads` changes; keep the worker count fixed across training comparisons.

Dual actors use `rollout2:ROOT:CONT:...` or
`rolloutu2:ROOT:CONT:...`; the remaining positional tail is unchanged from the
corresponding one-network spec. `ROOT` supplies the real decision's policy,
value, belief, priors, and shortlist. `CONT` supplies every policy decision
after a candidate is applied. Using the same checkpoint in both positions is
an exact backward-compatible spelling of the historical actor.

## Playing, analysing, measuring

```
./bin/play                                         # strongest validated actor
./bin/showgame -r 3                                # same actor, self-play
python3 tools/verify_transcript.py <transcript>    # independent rules audit
./bin/analyze -r 3 > data/analysis.json
./bin/belief_eval                                   # fixed held-out exact-K report
./bin/belief_eval --net candidate.bin --json       # machine-readable comparison
./bin/belief_eval --net candidate.bin --actor-net data/champion.bin
python3 tools/make_showcase.py --seed SEED --output /path/to/showcase.json \
        --embed-viewer web/viewer.html
./bin/arena -a policy:data/champion.bin:0:20 -b heur -n 300 -r 3
# Restartable evidence: keep one seed and split its absolute pair range.
./bin/arena -a AGENT_A -b AGENT_B -n 125 -r 3 -s BLOCK_SEED \
    --pair-start 0 --raw-pairs results/a-000.jsonl --raw-only \
    --provenance PLAN_SHA256,ARENA_SHA256,MODEL_SHA256
./bin/arena -a AGENT_A -b AGENT_B -n 125 -r 3 -s BLOCK_SEED \
    --pair-start 125 --raw-pairs results/a-125.jsonl --raw-only \
    --provenance PLAN_SHA256,ARENA_SHA256,MODEL_SHA256
python3 tools/merge_arena.py block --expect-start 0 --expect-pairs 250 \
    --output results/a.json results/a-000.jsonl results/a-125.jsonl
# Combine two complete, independently seeded A/B and B/A blocks.
python3 tools/merge_arena.py reciprocal --first results/a.json \
    --second results/b.json --output results/reciprocal.json
python3 tools/referee.py match NETA NETB --pairs 400 --rounds 3
# what was move X worth at ply N of an analysed game? (paired, with SE)
./bin/qpair -n data/champion.bin -s SEED -f moves.txt -p N -w 4000 \
            -U -y 20 -c "Y2 d deck" -c "W4 p deck"
make audit-test   # slow locked checks for the reviewed UI positions
make belief-eval-test
```

Raw arena artifacts contain every mirrored pair's integer scores, plies, and
cap count. Shards of one block must use the same seed and disjoint absolute
`--pair-start` ranges; changing the seed creates a different block rather than
continuing one. The merger rejects gaps, overlaps, incomplete files, metadata
drift, cap-terminated rounds, summary tampering, and overlapping reciprocal
RNG domains. The reciprocal step reopens, hashes, and exactly remerges every
recorded raw shard, so even an internally consistent edited summary cannot
produce an authoritative promotion flag. It recomputes pair-clustered
uncertainty from the raw rows instead of averaging rounded shard error bars;
precommitted tests can also record a nondefault critical z and require positive
point margin explicitly. Completed files and merged artifacts are published
with no-clobber writes so reruns cannot silently replace evidence.

The analysis console replays a match ply by ply: board, both hands (marked
where publicly known), the policy distribution, actual playing-search
decision, and independent post-hoc Q values for the focused top-policy
shortlist. The default audit does not score every legal move or add
rule-derived exceptions. The experimental fixed-cardinality and actor-history hand
estimates sit next to omniscient truth only for retrospective calibration,
along with the value trajectory. The UI keeps the actual playing choice
separate from the independent audit, marks the selection reference, and
distinguishes a highest sampled mean from two-panel numerical consensus and
from a fully gated low-prior correction. It also identifies whether the
baseline came from raw policy, exact hand scheduling, last-deck dominance, or
information-set draw repair. Recorded deck draws are explicitly
marked as future information unavailable at decision time.

Agent specs include `random`, `heur`,
`policy:PATH[:temperature[:symmetries[:plan_deck_max[:plan_block_gap[:draw_root_deck_max]]]]]`,
`rolloutu:...` (uniform-world
belief ablation), `mcts:PATH[...]`, and `net:PATH`. The complete rollout tail
is `worlds:candidates:floor:gate:min_candidates:ply_lo:ply_hi:eval_candidates:`
`objective:prune:override_k:override_min:sample:symmetries:policy_mass:`
`batch_worlds:playout_symmetries:discard_guard:deck_max:confirm_worlds:`
`playout_prune:plan_deck_max:plan_block_gap:semantic_candidates:`
`confirm_exact5:draw_variant_cores:draw_variant_deck_max:policy_prefix_mode:`
`belief_alpha:draw_root_deck_max:draw_playout_deck_max:prefix_confirm_k:`
`prefix_confirm_min:confirm_temp:action_core_count:exact_terminal:`
`deck2_replan_worlds:deck2_replan_cores:bounded_late_root:bounded_late_min`;
objective is
`0` for round margin, `1` for pure final-round match result, or `2` for the
champion hybrid. Continuation mode `0` is exact-group greedy, `1` is the
random-group/full-policy-sampling ablation, `2` is per-decision random-group
greedy, and `3` fixes one group member for an entire hidden-world trajectory.
Mode `4` fixes independently stratified group members for the two players in
each trajectory, removing arbitrary cross-player orientation correlation at
the same forward-pass cost.
`confirm_exact5` changes generic challenger confirmation, not the separate
trusted-prefix consensus panel. `draw_variant_cores` admits pile-draw variants
only for the top one or two distinct card/disposition actions and never grows
the root list beyond eight; `draw_variant_deck_max=0` means no phase limit.
`policy_prefix_mode=0` gates every override, mode `1` trusts the numerical
leader within the ordinary policy prefix, and mode `2` uses a fresh balanced
fixed-world panel. With paired evidence thresholds disabled it requires the
same leader. Low-prior semantic or draw
variants remain statistically gated in modes 1/2. Mode `3` applies the same
consensus rule with separate coherent player orientations. `belief_alpha`
scales the learned fixed-cardinality posterior (`0` is uniform); `rolloutu`
remains uniform regardless of that field.
The optional `prefix_confirm_k` and `prefix_confirm_min` fields strengthen
modes 2/3: the fresh panel's proposed move must beat candidate zero by both the
configured number of paired standard errors and the configured objective-unit
floor. It may rank another statistically close improvement first; the question
is whether the proposed correction independently beats the deployed baseline.
Both thresholds must be positive to enable the gate; leaving both at zero
preserves the earlier exact-leader consensus behavior.
`confirm_temp>0` makes only fresh confirmation continuations near-greedy:
actions are sampled from the shortest policy prefix covering 99.5% mass with
stateless move-keyed common Gumbel noise. A value of `0` preserves exact
argmax. `action_core_count=1..5` enables a hierarchical ordinary shortlist:
distinct card/play-discard actions are ranked by aggregate policy mass, one
complete move per core is retained, and any room left in the hard five-move
budget receives at most one information-set-safe draw alternative per core
whose complete-move prior clears the configured policy floor. Candidate zero
is always the unmodified deployed baseline. In this mode reported policy
coverage is the aggregate mass of every draw source in the retained cores,
not merely the representative complete moves. Both controls are opt-in and
leave all historical specs equivalent.

`exact_terminal=1` enables the all-action-core one-card solver. The two
historically named `deck2_replan_*` fields configure recursive deck-two/deck-three
descendant panels and set their number of top semantic cores. Their world value
remains the historical recursive cap. `bounded_late_root=1` instead requires
both recursive fields to be zero and independently enables the exhaustive
actual-root panel. Its ordered support remains at most 90/990. A completed root
panel is authoritative; an unavailable panel alone falls back to the ordinary
policy-focused evaluator. `bounded_late_min` controls its two-horizon practical
gain requirement independently of `prefix_confirm_min`.

Exact one-card solving is enabled in live play. Recursive replanning and the
bounded actual-root resolver remain disabled there; they are audit-only. On
seven frozen, human-reviewed
positions with 4,096 shared worlds, three action cores removed duplicate
draw-source crowding and the criticized Green-5 choices at plies 29 and 31,
while preserving the Blue-10 correction at ply 36 and admitting stronger
late-round alternatives at plies 62 and 64. It did not repair ply 23; four or
five cores reintroduced the bad Green-5 candidates. Confirmation temperatures
from 0.2 through 1.0 changed none of those tested leaders. The focused core
method is now the default post-game audit: its locked 2,048-world checks admit
neither G5 move at plies 29/31, choose R6 at ply 31, and confirm B10 over Y10
on both panels at ply 36. In live-actor testing, however, the independent
40-pair confirmation of the five-slot/three-core configuration was only
`+2.20 ± 5.91` points and `50.6% ± 4.4%` match score. That is compatible with
a gain but far from promotion evidence, so the maintained playing spec still
leaves both fields at zero.
Supported symmetry modes are `1`, `5`, `10`, `20`, and `120`.

All matches are paired: every deal (all three of them, in match mode) is
played twice with the seats swapped, so deal luck cancels.

## Honest limits

* There is no public Lost Cities benchmark bot or human game corpus to
  measure against offline; strength claims are relative (baselines, earlier
  stages, ablations), not against known human experts.
* The belief head conditions on the current information set, which carries
  most but not all behavioural evidence (the exact order of past actions is
  not encoded).
* Training is pure self-play after the imitation start; margins over
  qualitatively different opponents argue against self-overfitting, but a
  genuinely alien style could still find something.
* tools/blunders.py tallies outcome-level events -- expeditions that finished
  negative, wagered expeditions that finished deep negative, discards the
  opponent took at once. These are *style statistics*, not error rates: under
  optimal play every one of them is non-zero (a good gamble that fails still
  shows up in the tally), and with no optimal-play reference there is no
  "correct" value to compare against. They are only useful for watching style
  drift between versions (e.g. the agent hands the opponent far fewer
  immediately-useful discards than the heuristic, 2.5 vs 17.9 per match), and
  say nothing about whether any individual count is too high.
* Win-focused continuation training converged after three rounds: a fourth
  continuation stayed flat at 50-52% against its predecessor through 63
  iterations and was abandoned. Further gains likely need a bigger change
  (deeper search at training time, a larger trunk, or an opponent pool)
  rather than more of the same recipe.
