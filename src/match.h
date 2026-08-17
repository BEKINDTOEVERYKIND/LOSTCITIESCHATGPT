/* match.h -- paired-deal match runner shared by the arena and the trainer. */
#ifndef MATCH_H
#define MATCH_H

#include "agent.h"

typedef struct {
    int pairs, games;
    double margin, margin_se;   /* per game, from agent a's point of view */
    double winrate, winrate_se;
    double points_a, points_b, plies;
    double wins, losses, draws;
    uint64_t capped_rounds;
} MatchResult;

/* Exact outcome for one mirrored deal pair.  Leg zero seats A first.  Leg one
 * seats B first, but scores are stored in canonical A/B order in both legs so
 * shards can be concatenated and recomputed without rounding. */
typedef struct {
    uint64_t index;
    int score_a[2], score_b[2];
    int plies[2];
    int capped_rounds[2];
} MatchPairResult;

/* rounds = 1 gives single-deal games; rounds = MATCH_ROUNDS gives the full
 * competitive format, cumulative totals, alternating first player, margins
 * and winrate reported per match. */
void match_run_r(const Agent *a, const Agent *b, int pairs, int nthread,
                 uint64_t seed, int rounds, MatchResult *out);
int match_run_range_r(const Agent *a, const Agent *b, uint64_t pair_start,
                      int pairs, int nthread, uint64_t seed, int rounds,
                      MatchPairResult *pair_out, MatchResult *out);
void match_run(const Agent *a, const Agent *b, int pairs, int nthread,
               uint64_t seed, MatchResult *out);

#endif
