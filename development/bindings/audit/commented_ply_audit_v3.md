# Explicit commented-ply audit

Attempt: `v3` (fresh recovery from failed v2; v2 rerun forbidden)  
Actor: `rolloutu:data/champion.bin:800:5:0.02:0:1:14:0:0:0:0:3.5:2:4:20:0:0:20:1:0:800:1:0:0:0:0:0:0:3:1:0:0:0:0:0:0:1`  
Evaluation network: `data/champion.bin`  
Evidence: 17/17 explicit displayed plies; exactly 1024 paired current/future worlds per case, except 2214615196 / 10 at 2,048; exact policy-20 continuation through the full remaining match.

> This is diagnostic evidence only. The reviewed moves are excluded from training and are neither safety gates nor promotion gates. The locked validation criteria are unchanged.

## Per-ply evidence

| Ply | Actor's actual move | Reviewed context | Paired evidence |
|---|---|---|---|
| 2214615196 / 3 | `Bx p deck` | Low W2 pickup should not rival the deck. | Bx p W vs Bx p deck: match-score -1.03±1.67 pp (95% [-4.30, +2.24]); final margin -4.49±2.06; hybrid -1.25±1.74; inconclusive |
| 2214615196 / 4 | `Bx p deck` | Low W2 pickup should not rival the deck. | Bx p W vs Bx p deck: match-score -2.15±1.77 pp (95% [-5.61, +1.31]); final margin -3.81±2.16; hybrid -2.34±1.85; inconclusive |
| 2214615196 / 8 | `B3 p deck` | Low W2 pickup should not rival the deck. | B3 p W vs B3 p deck: match-score -6.05±1.68 pp (95% [-9.34, -2.77]); final margin -7.96±2.08; hybrid -6.45±1.75; reference ahead |
| 2214615196 / 10 | `Wx p deck` | The W2 pickup was called overrated; prior evidence was inconclusive and must be reported honestly. | Wx p W vs Wx p deck: match-score +0.12±1.16 pp (95% [-2.14, +2.39]); final margin -1.99±1.44; hybrid +0.02±1.21; inconclusive |
| 2214615196 / 12 | `W4 p deck` | Prefer the deck to the low R2 pickup. | W4 p R vs W4 p deck: match-score -4.54±1.53 pp (95% [-7.54, -1.55]); final margin -9.06±1.85; hybrid -4.99±1.59; reference ahead |
| 2214615196 / 13 | `Yx p deck` | Audit the exact-cardinality belief rather than an independent-card approximation. | Yx d deck vs Yx p deck: match-score +0.59±1.44 pp (95% [-2.23, +3.40]); final margin -0.17±1.86; hybrid +0.58±1.50; inconclusive; belief overlay: fixed-K Y9=25.46% vs 20.00% prior; top B10=35.40% (held); marginal sum=8.000000 for K=8 |
| 2214615196 / 16 | `Yx d deck` | Preserve White options instead of committing W7 early. | W7 p deck vs Y2 d deck: match-score -3.37±0.97 pp (95% [-5.26, -1.48]); final margin -2.72±0.99; hybrid -3.51±0.99; reference ahead; Yx d deck vs Y2 d deck: match-score -1.56±0.81 pp (95% [-3.16, +0.03]); final margin -0.33±0.88; hybrid -1.58±0.84; inconclusive |
| 2214615196 / 20 | `G4 p deck` | W7 was the concern; W3 and the wager discard were both reviewed, with no forced ordering between close discards. | W7 p deck vs W3 d deck: match-score -2.25±0.94 pp (95% [-4.09, -0.40]); final margin -3.36±0.88; hybrid -2.41±0.97; reference ahead; Wx d deck vs W3 d deck: match-score -0.20±0.92 pp (95% [-2.00, +1.61]); final margin +0.09±1.03; hybrid -0.19±0.95; inconclusive; G4 p deck vs W3 d deck: match-score -0.10±1.16 pp (95% [-2.37, +2.17]); final margin -2.23±1.20; hybrid -0.21±1.19; inconclusive |
| 5726968372613385 / 14 | `R4 d deck` | Evaluate both suggested alternatives against the recorded R4 discard; do not force-rank G7 versus B3. | G7 p deck vs R4 d deck: match-score +0.29±1.00 pp (95% [-1.67, +2.26]); final margin -0.88±1.20; hybrid +0.25±1.04; inconclusive; B3 d deck vs R4 d deck: match-score +0.24±1.21 pp (95% [-2.12, +2.61]); final margin -1.02±1.37; hybrid +0.19±1.25; inconclusive |
| 5726968372613385 / 15 | `W4 d deck` | Preserve W4 and discard B5. | W4 d deck vs B5 d deck: match-score +1.61±1.52 pp (95% [-1.37, +4.60]); final margin +3.52±1.89; hybrid +1.79±1.59; inconclusive |
| 5726968372613385 / 17 | `R8 p deck` | Discard B5 and take R4 rather than prematurely play R8. | R8 p R vs B5 d R: match-score -0.39±1.38 pp (95% [-3.10, +2.32]); final margin -0.72±1.84; hybrid -0.43±1.45; inconclusive; R8 p deck vs B5 d R: match-score +3.47±1.48 pp (95% [+0.56, +6.37]); final margin +3.89±1.78; hybrid +3.66±1.54; alternative ahead |
| 5726968372613385 / 32 | `Bx p deck` | Compare the third Blue wager with safe ten plays. | Bx p deck vs W10 p deck: match-score +0.29±0.44 pp (95% [-0.57, +1.16]); final margin -0.88±0.51; hybrid +0.25±0.46; inconclusive; R10 p deck vs W10 p deck: match-score +2.78±0.61 pp (95% [+1.58, +3.99]); final margin +4.04±0.74; hybrid +2.99±0.64; alternative ahead |
| 725402798 / 21 | `Bx d deck` | Avoid overvaluing G5; audit the clean Blue-wager discard with both relevant draw sources. | Bx d G vs Bx d deck: match-score -1.71±1.43 pp (95% [-4.52, +1.10]); final margin -0.62±1.52; hybrid -1.74±1.48; inconclusive; G5 p deck vs Bx d deck: match-score +0.39±1.39 pp (95% [-2.34, +3.12]); final margin -1.10±1.48; hybrid +0.34±1.44; inconclusive |
| 725402798 / 22 | `R7 p deck` | Discard R2 rather than commit R7 early. | R7 p deck vs R2 d deck: match-score -0.73±0.52 pp (95% [-1.75, +0.28]); final margin -1.25±0.54; hybrid -0.80±0.53; inconclusive |
| 725402798 / 23 | `Wx d deck` | Avoid overvaluing G5 when the White wager can be discarded. | G5 p deck vs Wx d deck: match-score -0.10±1.22 pp (95% [-2.49, +2.29]); final margin +2.50±1.37; hybrid +0.03±1.26; inconclusive |
| 725402798 / 25 | `Y4 p deck` | The face-up Blue wager pickup must be evaluated even when it falls outside the policy prefix. | Y4 p deck vs Y4 p B: match-score +2.73±1.73 pp (95% [-0.66, +6.13]); final margin +5.72±2.30; hybrid +3.02±1.82; inconclusive |
| 95647345759839 / 44 | `W10 p deck` | End the round through the last deck card instead of gifting another turn for a Green-wager pickup. | W10 p G vs W10 p deck: match-score +0.05±0.88 pp (95% [-1.67, +1.77]); final margin +1.21±0.68; hybrid +0.11±0.89; inconclusive |

## Aggregate diagnostic signals

- The actor selected a reviewed move on 8/16 action-review plies. This is descriptive alignment, not an accuracy score.

- Among reference-relative alternatives, the 95% intervals show 4 reference-ahead, 1 alternative-ahead, and 16 inconclusive comparisons.

- Actor-selected moves outside the nominated support were also graded: 0 reference-ahead, 1 actor-move-ahead, and 1 inconclusive comparisons.

- 7 reviewed candidates had policy prior below 2%; they were still evaluated because this audit is not restricted to the deployed shortlist.

- The p13 policy-neutral action panel has 0 reference-ahead, 0 alternative-ahead, and 1 inconclusive alternatives.

- Counterfactual cap hits: 0. The evaluator aborts the whole case on the first cap rather than reporting a truncated value.

## General-improvement implications

- Learn signed paired action advantages from independent natural states, retaining wins, losses, and ties. Keep these reviewed plies as a held-out diagnostic so improvements must generalize.

- Model card/action and draw-source value jointly. The repeated low-pile, face-up-wager, and final-deck cases are one interaction family, not a list of moves to hard-code.

- Improve option-value and commitment timing across suits (early sevens, G5, the third wager) through broader counterfactual data, not per-position patches.

- Keep action quality and belief calibration separate. The p13 posterior is an exact fixed-cardinality diagnostic whose marginals must sum to the opponent's unknown hand size.

## Provenance

- Attempt: `v3`
- Failed predecessor execution SHA-256: `de385af4ec2e7ac004f72b1a928bc86a461e39b91c2ca05c28b452921eb6951b`
- Audit definition SHA-256: `12ff900ef6b69a78b3384ba26bb5b2e6a2d2fe3e2d2135e291ed2bcb4d8cde83`
- Actor spec SHA-256: `51d1427d17774483bddd0b9d54c661346eef26323b8c1335fcc481210afd9c80`
- Repository HEAD: `681ccfa825b00b6ce5fc5a5d1356a17473b64aa4`
