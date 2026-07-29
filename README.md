# Lost Cities AI

A from-scratch Lost Cities engine, neural network, and self-play training
pipeline for the full competitive game — three-round matches with cumulative
scoring — written in C with no external dependencies.

This repository began as a correctness-and-strength continuation of
[`BEKINDTOEVERYKIND/LostCities`](https://github.com/BEKINDTOEVERYKIND/LostCities)
at commit `4df68f7b2cbda7bd9ee160618693f6436b29e9ec`. The original repository is
unchanged.

## Current status

Use `data/c8.bin` as the inherited champion. The upstream repository still
named an older checkpoint `best.bin`, but two independent evaluations with
the corrected pair-clustered statistics found `c8.bin` stronger:

| holdout | paired matches | margin/match | match score |
| --- | ---: | ---: | ---: |
| seed 94004 | 2,000 | +1.80 ± 0.77 SE | 51.3% ± 0.6% SE |
| seed 95005 | 5,000 | +2.03 ± 0.51 SE | 51.2% ± 0.4% SE |

Version-6 networks can additionally learn full card/action × draw-source
interactions and the complete public order of each discard pile. Loading a
legacy v3/v4 model zero-initializes only these additions and preserves its old
outputs exactly. A small exploratory v6 fine-tune improved point margin but
did not clear the promotion bar on match wins, so it is deliberately not
shipped as a new champion.

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

**Learned opponent inference.** A third network head predicts, for every card
whose location the player cannot pin down, the probability that the opponent
holds it. It is trained on self-play states where the true opponent hand is
known — every position is a free supervised example — so it learns behavioural
inference from the same data that trains the policy: an opponent who opened
Yellow with a wager is Yellow-heavy, while their expeditions, discards, known
cards, and the exact current discard-pile order all constrain what they can
hold. The determinized search then
samples opponent hands from this posterior (Gumbel-top-k over the belief
logits) instead of uniformly, so every rollout plays against plausible
opponents rather than random ones. The same inference reaches the raw policy
implicitly: the policy/value heads share the trunk with the belief head and
see all the same signals.

**Match play, not just round play.** The network's inputs include the round
number and the cumulative score difference. Training is staged: early PPO
rewards `margin + win bonus` (the dense margin signal teaches point play),
and the finishing phase switches to `0.05 x margin + 50 x match result`
(--mw / --winbonus in tools/rl.c) so that winning is nearly all that matters.
Being 40 up in round three genuinely changes what the policy optimises:
protect the win rather than maximise expectation, and gamble when behind.

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
done by *rollouts*: play each candidate move out to the end of the round with
the policy in belief-sampled worlds, sharing worlds across candidates so the
comparison is paired.

Training is: imitate the heuristic for a sane start (it knows nothing about
match context or beliefs), then PPO over full three-round matches with the
belief head learning on the side. The trunk is 666 -> 512 -> 256.

## Historical upstream results

All numbers are 3-round paired matches (each triple of deals played twice with
seats swapped) unless stated. Margins are total match points; "wins" are match
wins with draws counting half. The table and discussion below are retained as
upstream experiment history; unlike the current-status table above, many used
the older leg-level win-rate error estimate.

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
family recurs, smaller, in the current champion: in the analogous position
of the embedded game its written-off wager play measures +0.6 to +1.0 over
the 100%-prior discard under three estimators -- real, but below what a
96-world play-time search can resolve, which makes it a training target,
not a search target. But *forcing*
the floor open is worse than the disease: full rollout with `min_cand` 3
scored only **42.8% ± 3.5%** (-10.8 ± 4.2/match, 100 pairs) against the
baseline. A 96-world Q difference carries ±2-4 points of paired noise, most
true gaps between a near-certain policy move and its alternatives are
smaller than that, and taking the argmax of several noisy estimates
systematically flatters the winner. So `min_cand` selects among noise, while
`eval_cand` (the analysis setting) evaluates and *reports* extra candidates
without letting them be selected -- the viewer shows what written-off moves
were worth at zero strength cost.

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

Recommended settings: **maximum strength**
`rollout:NET:96:5:0.02:0:1:14:0:4:0:1:3` (search from ply 14, four
candidates evaluated, dominated discards pruned, 3-SE advisory override);
**gate 0.85 for real-time play**; raw policy for bulk generation.
Analysis uses `rollout:NET:512:5:0.02:0:1:0:0:4:0:1:3` -- the same
selection rules at 512 worlds, searched at every ply for display.

## Reproducing

```
# 1. imitation start (heuristic plays the rounds; ~15 min on 4 cores)
./bin/train --gen heur --gen-switch 99 --rounds 3 --iters 4 --games 2500 \
            --steps 15000 --batch 512 --lr 1e-3 --tau 0.5 --out data/m0.bin

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
./bin/play -a rollout:data/c8.bin:128:4            # play against the agent
./bin/showgame -a policy:data/c8.bin -r 3          # full match transcript
python3 tools/verify_transcript.py <transcript>    # independent rules audit
./bin/analyze -a rollout:data/c8.bin:512:5:0.02:0:1:0:0:4 -r 3 > data/analysis.json
./bin/arena -a policy:data/c8.bin -b heur -n 300 -r 3
python3 tools/referee.py match NETA NETB --pairs 400 --rounds 3
# what was move X worth at ply N of an analysed game? (paired, with SE)
./bin/qpair -n data/c8.bin -s SEED -f moves.txt -p N -w 4000 \
            -c "Y2 d deck" -c "W4 p deck"
```

The analysis console (`web/viewer.html`, embedded game included) replays a
match ply by ply: board, both hands (marked where publicly known), the policy
distribution, rollout Q values per candidate, the network's belief about the
opponent's hidden hand next to the omniscient truth, and the value trajectory
across all three rounds.

Agent specs: `random`, `heur`, `policy:PATH[:temp]`,
`rollout:PATH[:worlds[:cands[:floor[:gate[:minc[:plylo[:plyhi[:evalc[:winq]]]]]]]]]`
(strongest), `rolloutu:...` (uniform-world
ablation of the belief sampler), `mcts:PATH[...]`, `net:PATH` (kept as the
negative result it is).

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
