# Lost Cities AI

A from-scratch Lost Cities engine, neural network, and self-play training
pipeline for the full competitive game — three-round matches with cumulative
scoring — written in C with no external dependencies.

This repository began as a correctness-and-strength continuation of
[`BEKINDTOEVERYKIND/LostCities`](https://github.com/BEKINDTOEVERYKIND/LostCities)
at commit `4df68f7b2cbda7bd9ee160618693f6436b29e9ec`. The original repository is
unchanged.

## Current status

Use `data/champion.bin`, normally through
`policy:data/champion.bin:0:20`. It combines an exactly wager-symmetric
projection of the strongest inherited checkpoint with probability averaging
over 20 exact suit relabellings. It decisively beat the old raw `c8.bin`
policy on two fresh holdouts:

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
weaker as imitation progressed. Those negative candidates are not shipped.

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
`playout_symmetries` setting and can apply an exact ensemble at every
continuation decision. The static viewer uses the exact 20-way actor at the
root and a deterministic five-rotation continuation to control offline audit
cost; the locked human-reviewed probes use full 20-way continuations.

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
predictions from one game, so it was selection-biased. A new all-card
evaluation of `c8.bin` (seed 424) reports pooled AUC 0.703 and mean
within-state AUC 0.622, but worse Brier score and log loss than the simple
card-count prior, with ECE 0.098. The head contains ranking information but is
overconfident; coupled with the null belief-sampling strength ablation, this
remains a research component rather than an established source of strength.

## When to search, and when the policy alone is enough

Instrumented over 6,697 self-play decisions (tools/searchcmp.c): when the
policy's top move already carries >= 0.95 probability -- 59% of all decisions
-- rollout search disagrees with it only 3-7% of the time, for a mean gain of
0.1-0.2 points; below 0.95 confidence, disagreement is 39-81% and the mean
gain per decision is 1-5 points. Confidence is the dominant variable: the
pattern barely moves across rounds, deck phase, or match closeness (low-
confidence late-deck decisions have the largest tail, up to ~5 points).

The rollout agent therefore takes a gate parameter -- skip the search when the
policy's confidence is already >= the gate (`rollout:NET:worlds:cands:floor:gate`).
The maintained audit also uses the shortest cumulative policy-mass prefix
(minimum two, maximum eight moves), never forces negligible draw variants into
the list, and spends its paired worlds only on that shortlist.
Measured (3-round paired matches vs the raw policy):

| configuration | margin/match | match wins | speed |
| --- | ---: | ---: | ---: |
| full rollout, 96 worlds | **+30.0 ± 5.1** | **69.5%** (50 pairs) | 0.8 matches-games/s |
| gate 0.85 (searches ~23% of plies) | +17.1 ± 3.9 | 57.5% (50 pairs) | 1.9 |
| gate 0.95 (searches ~41% of plies) | +14.3 ± 4.6 | 60.5% (50 pairs) | 1.4 |

Head-to-head, the 0.95 gate loses -3.6 ± 4.2 per match to the ungated search
(43.8% over 40 pairs). The lesson cuts both ways: per *decision* the
high-confidence searches look worthless, but there are ~40 of them per match
per side and their 0.15-point slivers add up to most of the gap -- so gating
is a compute trade, not a free lunch.

**The candidate floor cuts both ways.** Candidates come from the policy, and
moves below a 2% prior are pruned -- so when the policy is *certain*, the
"search" has one candidate and can only confirm it, never overrule it. A
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

**Where the search earns its keep: late, not early** (all vs the raw policy,
3-round paired matches):

| search window (plies of each round) | margin/match | match wins |
| --- | ---: | ---: |
| everywhere (150 pairs) | +10.6 ± 3.0 | 51.5% ± 2.9% |
| only plies >= 14 (150 pairs) | **+11.4 ± 2.4** | **56.2% ± 2.9%** |
| only plies < 14 (200 pairs) | +4.6 ± 2.4 | 53.4% ± 2.5% |
| only plies < 14, forced 3 candidates (200 pairs) | -4.7 ± 3.1 | 50.1% ± 2.5% |

Restricting the search to the mid/late round loses nothing -- it matches or
beats searching everywhere while skipping ~30% of the searched plies, and
the direct head-to-head confirms it: late-only vs full search is a dead
heat, +0.6 ± 3.0/match, 49.7% ± 2.9% (150 pairs). Early search contributes
little, and *aggressive* early search (forced candidates) contributes
nothing at all. The mechanism shows up clearly at the opening
ply of the embedded game: three different first moves measure within ±0.5
points of each other at 8000 worlds under three different estimators.
Early-round moves are often near-equivalent in true value, so there is
little for a rollout to find, and its noise can only hurt; late-round
positions diverge sharply and have short, accurately-evaluated horizons.
(An earlier 50-pair run put full search at +30.0 ± 5.1 / 69.5%; the
run-to-run spread between that and the 150-pair number above is itself a
caution about small evaluation batches.)

**The search reports its own noise.** Every reported Q carries the standard
error of its paired difference against the chosen move -- a gap under ~2 of
those is sampling noise, which at 96 worlds means most gaps under ~4-8
points; the analysis dump uses 512 worlds to make the displayed numbers
meaningful. In the final round the dump also reports each candidate's match
win fraction over the playouts (the last round decides the match exactly,
so point EV stops being the objective there). Selecting by that win
fraction is available (`win_q`) but off by default, because it measured no
better than margin selection -- 50.4% ± 0.8% match wins pooled over 2,000
head-to-head pairs (a 300-pair run at 48.0% ± 2.0% and a 1,700-pair
confirmation at 50.8% ± 0.9%), while costing 1.3 ± 0.6 points of margin:
decided finals tie on win%, close finals make a 96-world win fraction a
noisy binomial estimate, and the win-trained policy already carries the
clutch behaviour into every playout. The same lesson as the candidate
floor, from the other direction: at fixed compute, the statistically
efficient objective beats the theoretically right one.

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
the SE test needs. (1) `override_min` points (default 4): the SE gate is
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

Sampled playouts (playout_sample, spec field 14) A/B'd against argmax at
the full config: 49.4% ± 2.0% over 300 pairs -- a tie.  Unbiased
continuations cost nothing in strength, so analysis and training labels
use them; match play keeps argmax with the sampled confirmation gate.

The strongest newly validated default is the 20-way policy ensemble:
`policy:data/champion.bin:0:20`. Rollout remains available for slower
analysis and play. Objective mode `2` uses round margin in rounds 0/1 and the
champion's `0.05 × final match margin + 50 × result` return only in real round
index 2. A high-compute analysis spec is
`rollout:data/champion.bin:512:5:0.02:0:1:0:0:4:2:1:3:4:0:20`.

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
./bin/play                                         # champion, 20-way ensemble
./bin/showgame -a policy:data/champion.bin:0:20 -r 3
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
where publicly known), the policy distribution, post-hoc Q values for only the
top-policy moves, the calibrated fixed-cardinality hand estimate next to
omniscient truth, and the value trajectory. It distinguishes the numerical
leader from a statistically resolved best move; unresolved gaps never receive
an “audit pick” label. Recorded deck draws are explicitly marked as future
information unavailable at decision time.

Agent specs include `random`, `heur`,
`policy:PATH[:temperature[:symmetries]]`, `rolloutu:...` (uniform-world
belief ablation), `mcts:PATH[...]`, and `net:PATH`. The complete rollout tail
is `worlds:candidates:floor:gate:min_candidates:ply_lo:ply_hi:eval_candidates:`
`objective:prune:override_k:override_min:sample:symmetries:policy_mass:`
`batch_worlds:playout_symmetries`; objective is
`0` for round margin, `1` for pure final-round match result, or `2` for the
champion hybrid. Supported symmetry modes are `1`, `5`, `10`, `20`, and
`120`.

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
