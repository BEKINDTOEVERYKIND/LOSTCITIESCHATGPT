/* rollout.c -- policy improvement by playing candidate moves out.
 *
 * The policy network is sharp, so a turn usually has two to four moves worth
 * considering.  For each of them we sample a world consistent with what the
 * mover knows -- the opponent's hand and the deck order -- and play the game to
 * the end with the same policy driving both seats, then compare the final
 * margins.
 *
 * Two properties make this work where the value network did not:
 *
 *  - The estimate comes from real finished games, so it never inherits the
 *    value head's inability to separate moves that differ by a point or two.
 *  - Each sampled world is shared by every candidate, and the playouts are
 *    deterministic given the world, so the *difference* between candidates is
 *    measured on identical futures.  That pairing is what makes a few hundred
 *    samples enough to resolve small differences.
 *
 * Rollouts also avoid the strategy fusion that spoils determinized tree search:
 * inside a sampled world each side still chooses from its own information set,
 * because the policy only ever sees the features of the player to move.
 */
#include "search.h"
#include "agent.h"
#include "heuristic.h"
#include <math.h>
#include <stdlib.h>

#define MAX_CAND 8

/* Rank the legal moves of s for the player to move.  With a network that is
 * the policy head; without one it is the hand-crafted evaluation, which gives
 * the classical "heuristic + perfect-information Monte Carlo" baseline. */
static int rank_moves(const Net *net, const State *s, Move *mv, float *score)
{
    if (net) return policy_probs(net, s, mv, score, NULL);
    int n = lc_moves(s, mv);
    for (int i = 0; i < n; i++) score[i] = heur_move_value_det(s, mv[i]);
    return n;
}

/* Play s out to the end of the round, returning the round margin for player
 * p.  In the final round of a match the round's end decides the match, so
 * *winpts gets the match result (1 win, 0.5 draw, 0 loss) from the carried
 * cumulative totals; in earlier rounds it gets -1 (margin is the only
 * available objective there, and it doubles as the natural proxy).
 * srng != NULL samples the policy instead of argmaxing it: deterministic
 * playouts repeat every knife-edge downstream decision identically across
 * paired worlds, which can manufacture large fake Q gaps with tiny paired
 * errors; sampling breaks that correlation. */
static int playout(const Net *net, State *s, int p, int prune, Rng *srng,
                   double *winpts)
{
    Move mv[MAX_MOVES];
    float score[MAX_MOVES];
    while (!s->over) {
        int n = rank_moves(net, s, mv, score);
        if (n <= 0) break;
        uint64_t dead = prune ? (lc_dead_cards(s) & s->hand[s->turn]) : 0;
        int best = -1;
        if (srng && net) {
            float w[MAX_MOVES];
            float tot = 0.0f;
            for (int i = 0; i < n; i++) {
                w[i] = (dead && lc_discard_dominated(s, mv[i], dead)) ? 0.0f : score[i];
                tot += w[i];
            }
            if (tot > 0.0f) best = sample_index(w, n, srng);
        }
        if (best < 0) {
            for (int i = 0; i < n; i++) {
                if (dead && lc_discard_dominated(s, mv[i], dead)) continue;
                if (best < 0 || score[i] > score[best]) best = i;
            }
        }
        if (best < 0) best = 0;
        lc_apply(s, mv[best]);
    }
    int sp = lc_score(s, p), so = lc_score(s, p ^ 1);
    if (winpts) {
        if (s->round == MATCH_ROUNDS - 1) {
            int tp = s->cum[p] + sp, to = s->cum[p ^ 1] + so;
            *winpts = tp > to ? 1.0 : (tp == to ? 0.5 : 0.0);
        } else *winpts = -1.0;
    }
    return sp - so;
}

Move rollout_move(const struct Agent *a, const State *st, Rng *rng,
                  float *out_value, SearchStats *stats)
{
    Move mv[MAX_MOVES];
    float prob[MAX_MOVES];
    float value = 0.0f;
    int n;
    if (a->net) {
        n = policy_probs(a->net, st, mv, prob, &value);
    } else {
        DrawSamples ds;
        draw_samples_init(st, st->turn, rng, 6, &ds);
        n = lc_moves(st, mv);
        for (int i = 0; i < n; i++) prob[i] = move_value_heur(st, mv[i], &ds);
    }
    /* Optional dead-discard search-focus heuristic.  It is deliberately not
     * strict dominance because burying and baiting effects can matter. */
    if (a->prune_dom && n > 1) {
        uint64_t dead = lc_dead_cards(st);
        if (dead & st->hand[st->turn]) {
            int k = 0;
            for (int i = 0; i < n; i++) {
                if (lc_discard_dominated(st, mv[i], dead)) continue;
                mv[k] = mv[i];
                prob[k] = prob[i];
                k++;
            }
            if (k > 0) n = k;
        }
    }
    if (n <= 1) {
        if (out_value) *out_value = value;
        if (stats) {
            stats->n = n;
            if (n == 1) {
                stats->mv[0] = mv[0]; stats->visits[0] = 1; stats->q[0] = value;
                stats->se[0] = 0.0; stats->qw[0] = -1.0;
            }
            stats->value = value;
        }
        return mv[0];
    }

    /* ply window: outside it the raw policy plays (see agent.h) */
    if ((a->ply_lo > 0 && st->nply < a->ply_lo) ||
        (a->ply_hi > 0 && st->nply >= a->ply_hi)) {
        int top = 0;
        for (int i = 1; i < n; i++) if (prob[i] > prob[top]) top = i;
        if (out_value) *out_value = value;
        if (stats) {
            stats->n = 1;
            stats->mv[0] = mv[top];
            stats->visits[0] = 0;
            stats->q[0] = value;
            stats->se[0] = 0.0; stats->qw[0] = -1.0;
            stats->value = value;
        }
        return mv[top];
    }

    /* confidence gate: when the policy is already near-certain, searching can
     * only confirm it or override it with noise -- return the policy move and
     * spend the compute where decisions are actually contested */
    if (a->gate > 0.0f) {
        int top = 0;
        for (int i = 1; i < n; i++) if (prob[i] > prob[top]) top = i;
        if (prob[top] >= a->gate) {
            if (out_value) *out_value = value;
            if (stats) {
                stats->n = 1;
                stats->mv[0] = mv[top];
                stats->visits[0] = 0;
                stats->q[0] = value;
                stats->se[0] = 0.0; stats->qw[0] = -1.0;
                stats->value = value;
            }
            return mv[top];
        }
    }

    /* candidates: the most likely moves, cut off once the policy stops caring */
    int order[MAX_MOVES];
    for (int i = 0; i < n; i++) order[i] = i;
    int ncand = a->root_width < MAX_CAND ? a->root_width : MAX_CAND;
    if (ncand > n) ncand = n;
    for (int i = 0; i < ncand; i++) {
        int best = i;
        for (int j = i + 1; j < n; j++) if (prob[order[j]] > prob[order[best]]) best = j;
        int t = order[i]; order[i] = order[best]; order[best] = t;
    }
    int nsorted = ncand;                     /* prefix of order[] that is sorted */
    if (a->net) {
        float floor_p = a->cand_floor > 0.0f ? a->cand_floor : 0.02f;
        int keep = a->min_cand > 1 ? a->min_cand : 1;
        if (keep > ncand) keep = ncand;
        while (ncand > keep && prob[order[ncand - 1]] < floor_p) ncand--;
    }
    /* advisory candidates: evaluated and reported but never selected, so an
     * analysis dump can show what the search thinks of moves the policy has
     * written off without letting that opinion change the game (forcing the
     * floor open for *selection* measured 42.8% vs the baseline) */
    int neval = ncand;
    if (a->eval_cand > neval) {
        neval = a->eval_cand < nsorted ? a->eval_cand : nsorted;
    }
    /* draw-variant expansion: "same action, another draw" shares every
     * sampled world and costs almost nothing extra, yet those variants are
     * the alternatives an analyst asks about most -- make sure the two
     * top-prior actions have all their legal draw sources on the board */
    if (a->eval_cand > 0) {
        for (int t = 0; t < 2 && t < neval; t++) {
            Move top = mv[order[t]];
            for (int i = 0; i < n && neval < MAX_CAND; i++) {
                if (mv[i].card != top.card || mv[i].discard != top.discard) continue;
                int seen = 0;
                for (int c = 0; c < neval; c++) if (order[c] == i) { seen = 1; break; }
                if (!seen) order[neval++] = i;
            }
        }
    }

    double sum[MAX_CAND], sumw[MAX_CAND];
    for (int i = 0; i < neval; i++) { sum[i] = 0.0; sumw[i] = 0.0; }
    const int p = st->turn;
    int reps = a->dets > 0 ? a->dets : 1;
    int lastround = st->round == MATCH_ROUNDS - 1;
    double *val = (double *)malloc(sizeof(double) * (size_t)neval * (size_t)reps);

    for (int d = 0; d < reps; d++) {
        State world;
        determinize_b(st, p, rng, a->no_belief ? NULL : a->net, &world);
        uint64_t wseed = 0x9E3779B97F4A7C15ULL * (uint64_t)(d + 1) ^ rng->s[0];
        for (int c = 0; c < neval; c++) {
            State s = world;                 /* same world for every candidate */
            lc_apply(&s, mv[order[c]]);
            double w;
            Rng pr;
            if (a->playout_sample) rng_seed(&pr, wseed);   /* same seed per world */
            int m = playout(a->net, &s, p, a->prune_dom,
                            a->playout_sample ? &pr : NULL, &w);
            if (val) val[(size_t)c * reps + d] = m;
            sum[c] += m;
            if (w >= 0.0) sumw[c] += w;
        }
    }

    /* In the final round the playouts decide the match, so pick by match
     * wins with margin as the tiebreak -- a 5% shot at stealing the match
     * outranks a certain narrow loss regardless of expected points.  In
     * earlier rounds margin is all a round-end playout can know. */
    int usew = lastround && a->win_q;
    int best = 0;
    for (int c = 1; c < ncand; c++) {
        if (usew ? (sumw[c] > sumw[best] ||
                    (sumw[c] == sumw[best] && sum[c] > sum[best]))
                 : (sum[c] > sum[best])) best = c;
    }
    /* significance-gated override: an advisory candidate may take the move
     * only when its lead over the eligible best exceeds override_k paired
     * standard errors AND override_min points.  The SE gate rejects noise
     * (what blanket forcing lacked: it overrode on any gap and lost 42.8%);
     * the points gate rejects playout bias, which more worlds sharpen
     * rather than shrink.  The reference is the eligible best, fixed, and
     * the highest-Q qualifier wins -- chaining comparisons through interim
     * winners made the outcome depend on candidate order. */
    if (a->override_k > 0.0f && val && reps > 1) {
        int elig = best;
        for (int c = ncand; c < neval; c++) {
            double dm = (sum[c] - sum[elig]) / reps;
            if (dm <= 0.0) continue;
            double v2 = 0.0;
            for (int d = 0; d < reps; d++) {
                double x = val[(size_t)c * reps + d] - val[(size_t)elig * reps + d] - dm;
                v2 += x * x;
            }
            double sed = sqrt(v2 / (reps - 1) / reps);
            if (dm > a->override_k * sed && dm > a->override_min &&
                sum[c] > sum[best]) best = c;
        }
        /* sampled confirmation: a qualifying gap must survive stochastic
         * continuations at half the floor, or it was determinism bias --
         * measured concretely: a +5.0 +- 0.14 argmax gap that collapsed to
         * +0.6 under sampling, from one knife-edge downstream decision
         * repeating across every paired world */
        if (best != elig) {
            double ds = 0.0;
            for (int d = 0; d < reps; d++) {
                State world;
                determinize_b(st, p, rng, a->no_belief ? NULL : a->net, &world);
                uint64_t wseed = 0x9E3779B97F4A7C15ULL * (uint64_t)(d + 1) ^ rng->s[0];
                Rng r1, r2;
                rng_seed(&r1, wseed);
                rng_seed(&r2, wseed);
                State sa = world, sb = world;
                lc_apply(&sa, mv[order[best]]);
                lc_apply(&sb, mv[order[elig]]);
                ds += playout(a->net, &sa, p, a->prune_dom, &r1, NULL)
                    - playout(a->net, &sb, p, a->prune_dom, &r2, NULL);
            }
            if (ds / reps < 0.5 * a->override_min) best = elig;
        }
    }
    float bestq = (float)(sum[best] / reps);
    if (stats) {
        stats->n = neval;
        for (int c = 0; c < neval; c++) {
            stats->mv[c] = mv[order[c]];
            stats->visits[c] = reps;
            stats->q[c] = sum[c] / reps;
            stats->qw[c] = lastround ? sumw[c] / reps : -1.0;
            double v = 0.0;
            if (val && reps > 1) {
                double mean = (sum[c] - (c == best ? 0.0 : sum[best])) / reps;
                for (int d = 0; d < reps; d++) {
                    double x = val[(size_t)c * reps + d]
                             - (c == best ? 0.0 : val[(size_t)best * reps + d]);
                    v += (x - mean) * (x - mean);
                }
                v = sqrt(v / (reps - 1) / reps);
            }
            stats->se[c] = v;
        }
        stats->value = bestq;
    }
    free(val);
    if (out_value) *out_value = bestq;
    return mv[order[best]];
}
