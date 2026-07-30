# Lost Cities AI

A from-scratch Lost Cities engine, neural network, and self-play training
pipeline for the full competitive game — three-round matches with cumulative
scoring — written in C with no external dependencies.

This repository began as a correctness-and-strength continuation of
[`BEKINDTOEVERYKIND/LostCities`](https://github.com/BEKINDTOEVERYKIND/LostCities)
at commit `4df68f7b2cbda7bd9ee160618693f6436b29e9ec`. The original repository is
unchanged.

## Current status

The maintained playing agent uses the exact 20-way champion policy and a
focused uniform-world rollout over a policy-ranked shortlist from round ply 20:

```text
rolloutu:data/champion.bin:512:4:0.02:0:1:20:0:0:0:0:3.5:2:2:20:0:0:20:1:0:512:1
```

This locked late-round rollout scored
**+14.70 ± 1.53 SE points per match** and **56.5% ± 1.5% SE match score**
against policy alone on a fresh 200-pair holdout (400 complete three-round
matches). The exact one-card-deck weak-dominance rule is included directly:
after the same card action, drawing the last deck card ends the round instead
of gifting the opponent another optional turn.

Independently, the optional visible-hand scheduler scored **+2.38 ± 0.33 SE
points per match** and **51.9% ± 0.4% SE match score** against the raw
exact-20 policy over 2,000 fresh mirrored pairs. It and the narrowly targeted
semantic candidates are available to the post-hoc evaluator, but combined
screens did not establish an additive gain over the locked rollout actor.
Planner-only was +1.60 ± 5.34 points in a direct 20-pair screen against the
locked actor; the full planner+semantic tail was -6.06 ± 4.36 over 40 pairs.
They are therefore not silently enabled for live play. `play`, `showgame`,
`analyze`, and the replay generator use the measured configuration above;
`analyze` separately labels its higher-compute review recommendations.

The [interactive match observatory](https://lost-cities-ai-match.juliandelphiki.chatgpt.site)
retains one unscreened random self-play match generated once under this actor
(seed 5726968372613385, with no retry or result selection).
Its deals and 144-move trajectory have not been changed to make the new method
look better. The offline decision review and hand estimates were upgraded
against those same frozen positions. The viewer keeps the historical action
separate from the current post-game audit and shows the focused shortlist,
paired uncertainty, independent confirmation, guard status, and experimental
hand estimate at every move.

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
remaining gain by averaging away arbitrary suit-slot preferences. Exhaustive
120-way averaging was only +0.70 ± 1.34 over the 20-way mode in a 500-pair
screen while costing about six times as much, so it was not made the default.

No newly trained checkpoint cleared the promotion bar. In particular, three
attempts to distil the 20-way ensemble back into one fast network all became
weaker as imitation progressed. A later conservative residual-only experiment
also failed its predeclared locked gate: over 20,000 fresh mirrored pairs, its
two finalists scored 50.04375% and 50.06375% against the unchanged champion,
with lower confidence bounds below 50%. Those negative candidates are not
shipped.

Version-6 networks can learn full card/action × draw-source interactions and
the complete public order of each discard pile. Loading a legacy v3/v4 model
zero-initializes only these additions and preserves its old outputs exactly.
`make` deterministically regenerates `data/champion.bin` from the tracked
`data/c8.bin`; its expected SHA-256 is
`af2b2c237d21f5ec15acbcba2fde3e45864a6e44af4ddb1ff6f3756fd687f417`.

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
tools/history_belief.py actor-aware offline hand posterior from scrubbed history
tools/planarena.c   isolated mirrored evaluation of the visible-hand scheduler
tools/robust_distill.c conservative confirmed-correction residual trainer
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
remain available for research, but their playing-strength measurements must be
rerun after the fixed-cardinality sampler change.

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
every continuation decision, or cheaply draw one group member and take its
greedy move. Policy-action sampling is a separate diagnostic mode; it is no
longer accidentally coupled to random symmetry sampling. The maintained
audit uses the exact 20-way actor at the root and random-member, greedy
20-way continuations, which costs one forward pass per downstream decision
without evaluating a high-entropy player.

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
serves as a PPO baseline where its errors cancel, and search is principally
done by *rollouts*: play a policy-ranked shortlist to the end of the round in
shared hidden worlds. Pairing every candidate on the same worlds sharply
reduces uncertainty in the move-to-move difference. The audit reports the
standard error of each Q mean separately from the paired standard error of
each difference.

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

## When to search, and when the policy alone is enough

Early low-compute experiments established that policy confidence alone is not
a reliable phase gate. The maintained agent instead uses a round-ply cutoff
selected by direct match play: ordinary rollout is off for plies 0–19 and on
thereafter. At each searched position it starts from policy moves with at
least 2% prior, with a hard cap of four. It spends 512 shared hidden worlds on
the primary comparison. A challenger must clear a family-wise 3.5-SE test and
a two-point practical floor, then repeat on 512 independently seeded worlds
before it may replace the deployed baseline.

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
is evaluated only against the deployed baseline, not alongside the top four,
and receives a primary cap plus a fresh confirmation batch of 16,384 worlds.
Only support from both batches can supersede the generic discard guard. This
does not evaluate the Cartesian product of legal plays and draw sources. A
one-candidate shortlist still skips continuation forwards entirely. This
tail remains an audit/component configuration until it beats the maintained
actor in a new locked direct test.

The maintained late-round rollout configuration was locked before its fresh
holdout:

| comparison | mirrored pairs | margin/match | match score |
| --- | ---: | ---: | ---: |
| rollout from round ply 20 vs policy alone, seed 950005 | 200 | **+14.70 ± 1.53 SE** | **56.5% ± 1.5% SE** |

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
the 100%-prior discard under three estimators -- real, but below what a
96-world play-time search can resolve, which makes it a training target,
not a search target. But *forcing*
the floor open is worse than the disease: full rollout with `min_cand` 3
scored only **42.8% ± 3.5%** (-10.8 ± 4.2/match, 100 pairs) against the
baseline. A 96-world Q difference carries ±2-4 points of paired noise, most
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
corrected continuation policy, and stricter round-ply-20 deployment above.
Early-round moves are often near-equivalent in true value, so there is little
for rollout to find; late-round positions diverge more sharply and have much
shorter, more accurately evaluated horizons.

**The search reports its own noise.** Every reported Q carries its own
standard error plus the paired-difference standard error against the policy
leader. A newly generated deep audit uses 2,048 primary worlds and runs a
separate 2,048-world confirmation only for a qualifying challenger; the frozen
showcase retains its original 1,000-world audit. The playing agent
uses 512 plus an optional 512. An urgent one-sided wager correction compares
only that challenger with the deployed baseline, using up to 16,384 primary
worlds and exactly 16,384 fresh confirmation worlds. In the final round the
dump also reports each candidate's match win fraction over
the playouts (the last round decides the match exactly, so point EV stops
being the objective there). Selecting by that win
fraction is available (`win_q`) but off by default, because it measured no
better than margin selection -- 50.4% ± 0.8% match wins pooled over 2,000
head-to-head pairs (a 300-pair run at 48.0% ± 2.0% and a 1,700-pair
confirmation at 50.8% ± 0.9%), while costing 1.3 ± 0.6 points of margin:
decided finals tie on win%, close finals make a 96-world win fraction a
noisy binomial estimate, and the win-trained policy already carries the
clutch behaviour into every playout. The same lesson as the candidate
floor, from the other direction: at fixed compute, the statistically
efficient objective beats the theoretically right one.

**Frozen showcase checks.** The detailed review uses the original random
trajectory, not a regenerated match. At ply 7, three independent high-world
continuation families agreed that play G5 and play W5 are effectively tied:
their pooled exact-policy estimates were 29.49 ± 0.38 and 29.60 ± 0.39,
with W5 only +0.11 ± 0.35 over G5. Premature Y4 was decisively worse at
7.63 ± 0.37, or -21.86 ± 0.40 against G5. The correct conclusion is that the
original G5 is defensible, W5 is equally plausible, and Y4 is not.

The focused one-sided-wager trigger corrects two later reviewed positions
without opening the whole move list. Under the production 16,384-world path,
discard Bx then take R beat the deployed baseline at ply 59 by
+5.16 ± 0.40 in the primary audit and +4.77 ± 0.40 in fresh confirmation.
At ply 61 it measured +4.26 ± 0.39 and +4.04 ± 0.40 respectively. The frozen
viewer also retains the earlier, directionally matching 4,096-world ply-59
analysis rather than rewriting the historical artifact.
With one deck card left at ply 96, the exact rule
chooses the same play followed by a deck draw: ending the round weakly
dominates taking a pile and gifting the opponent another turn. These are
specific validated corrections, not evidence that every rollout estimate is
now reliable.

Two reviewed blind spots remain explicit. At ply 17, forcing a 4,096-world
comparison still favored play R8; discard B5 then take R measured
-2.20 ± 0.73 against it under the current continuation model. Broad rollout
therefore stays off before its validated late-round window instead of turning
that biased estimate into a confident override. At ply 25, the semantic
generator now admits taking Bx, but uniform-world rollout still rejected it
(-5.52 ± 0.58 in the locked audit). Conditioning hidden worlds on the
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

**The significance-gated override is a measured gain** -- the discipline
blanket forcing lacked. Advisory candidates (eval_cand) may take the move
only when they lead the best policy-plausible candidate by more than
`override_k` paired standard errors, the statistical signature of a
confidently-wrong prior rather than noise. A/B at k=3 with four evaluated
candidates: **+6.35 ± 1.88 per match, 52.5% ± 2.0%** over 300 pairs
against the previous maximum-strength config -- the first strength
improvement since the shipped champion, at ~1.7x search compute.

Expert review of an override-enabled game then exposed two further gates
the SE test needs. (1) `override_min` points (now 2 in the maintained agent):
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
separates the useful cheap symmetry approximation from action sampling:
each downstream decision draws one requested symmetry and takes its greedy
move. Independent confirmation uses this mode as well, so it perturbs
symmetry-sensitive discontinuities without changing the evaluated player
into a much weaker high-entropy policy.

The maintained default is the measured round-ply-20 rollout actor printed at
the top of this README. The scheduler and semantic challenger generator remain
enabled in the explicitly labelled post-hoc evaluator and component tools,
not in live play. Objective mode `2` remains available for research and uses
round margin in rounds 0/1 plus
the champion's `0.05 × final match margin + 50 × result` return only in real
round index 2. The measured production actor uses objective mode `0`;
changing the objective would require another locked strength test. The
last-round-only win objective is intentionally preserved. The maintained
high-compute post-hoc audit spec is printed into every analysis artifact by
`tools/analyze`.

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
```

## Playing, analysing, measuring

```
./bin/play                                         # strongest validated actor
./bin/showgame -r 3                                # same actor, self-play
python3 tools/verify_transcript.py <transcript>    # independent rules audit
./bin/analyze -r 3 > data/analysis.json
python3 tools/make_showcase.py --seed SEED --output /path/to/showcase.json
./bin/arena -a policy:data/champion.bin:0:20 -b heur -n 300 -r 3
python3 tools/referee.py match NETA NETB --pairs 400 --rounds 3
# what was move X worth at ply N of an analysed game? (paired, with SE)
./bin/qpair -n data/champion.bin -s SEED -f moves.txt -p N -w 4000 \
            -U -y 20 -c "Y2 d deck" -c "W4 p deck"
make audit-test   # slow locked checks for the reviewed UI positions
```

The analysis console replays a match ply by ply: board, both hands (marked
where publicly known), the policy distribution, actual playing-search
decision, and independent post-hoc Q values for the focused top-policy
shortlist plus any rule-derived semantic challenger. It does not score every
legal move. The experimental fixed-cardinality and actor-history hand
estimates sit next to omniscient truth only for retrospective calibration,
along with the value trajectory. The UI distinguishes the numerical
leader from a supported policy correction. A correction must clear a
family-wise discovery test and then repeat on 2,048 freshly seeded worlds;
unresolved gaps never receive an “audit pick” label. Recorded deck draws are
explicitly marked as future information unavailable at decision time.

Agent specs include `random`, `heur`,
`policy:PATH[:temperature[:symmetries]]`, `rolloutu:...` (uniform-world
belief ablation), `mcts:PATH[...]`, and `net:PATH`. The complete rollout tail
is `worlds:candidates:floor:gate:min_candidates:ply_lo:ply_hi:eval_candidates:`
`objective:prune:override_k:override_min:sample:symmetries:policy_mass:`
`batch_worlds:playout_symmetries:discard_guard:deck_max:confirm_worlds:`
`playout_prune:plan_deck_max:plan_block_gap:semantic_candidates`; objective is
`0` for round margin, `1` for pure final-round match result, or `2` for the
champion hybrid. Continuation mode `0` is exact-group greedy, `1` is the
random-group/full-policy-sampling ablation, and `2` is random-group greedy.
The maintained actor omits the optional tail. The post-hoc review tail is
`:16:12:1`; `:0:0:0` disables all three additions explicitly.
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
