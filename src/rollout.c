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
#include <string.h>

#define MAX_CAND 8

/* Rank the legal moves of s for the player to move.  With a network that is
 * the policy head; without one it is the hand-crafted evaluation, which gives
 * the classical "heuristic + perfect-information Monte Carlo" baseline. */
static int rank_moves(const Net *net, const State *s, Move *mv, float *score,
                      int symmetries)
{
    if (net) return policy_probs_sym(net, s, mv, score, NULL, symmetries);
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
                   int symmetries, double *winpts)
{
    Move mv[MAX_MOVES];
    float score[MAX_MOVES];
    while (!s->over) {
        int n = rank_moves(net, s, mv, score, symmetries);
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

/* Objective used to compare completed playouts.  Modes:
 *   0: round margin (historical default)
 *   1: pure match result in real round index 2
 *   2: 0.05 * final match margin + 50 * signed match result
 * Rounds 0 and 1 always retain margin semantics, independent of mode.  Mode 2
 * matches the strongest checkpoint's finishing reward while preserving the
 * intentional last-round-only switch to match-winning play. */
double rollout_terminal_objective(const State *terminal, int p, int mode)
{
    int round_margin = lc_score(terminal, p) - lc_score(terminal, p ^ 1);
    if (terminal->round != MATCH_ROUNDS - 1 || mode <= 0)
        return (double)round_margin;

    int total_margin = (int)terminal->cum[p] - (int)terminal->cum[p ^ 1]
                     + round_margin;
    int result = (total_margin > 0) - (total_margin < 0);
    if (mode == 1)
        return 50.0 * (double)result;
    return 0.05 * (double)total_margin + 50.0 * (double)result;
}

Move rollout_move(const struct Agent *a, const State *st, Rng *rng,
                  float *out_value, SearchStats *stats)
{
    if (stats) {
        memset(stats, 0, sizeof *stats);
        for (int i = 0; i < MAX_MOVES; i++) stats->qw[i] = -1.0;
    }
    Move mv[MAX_MOVES];
    float prob[MAX_MOVES];
    float value = 0.0f;
    int n;
    if (a->net) {
        n = policy_probs_sym(a->net, st, mv, prob, &value,
                             a->symmetries);
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
            stats->nlegal = n;
            stats->worlds = 0;
            stats->max_worlds = a->dets;
            stats->resolved = n == 1;
            stats->raw_best = 0;
            stats->policy_mass = n == 1 ? 1.0 : 0.0;
            if (n == 1) {
                stats->mv[0] = mv[0]; stats->visits[0] = 0; stats->q[0] = value;
                stats->se[0] = 0.0; stats->prior[0] = 1.0;
            }
            stats->value = value;
        }
        Move none = { 0, 0, 0 };
        return n == 1 ? mv[0] : none;
    }

    /* ply window: outside it the raw policy plays (see agent.h) */
    if ((a->ply_lo > 0 && st->nply < a->ply_lo) ||
        (a->ply_hi > 0 && st->nply >= a->ply_hi)) {
        int top = 0;
        for (int i = 1; i < n; i++) if (prob[i] > prob[top]) top = i;
        if (out_value) *out_value = value;
        if (stats) {
            stats->n = 1;
            stats->nlegal = n;
            stats->worlds = 0;
                stats->max_worlds = a->dets;
                stats->resolved = 0;
            stats->raw_best = 0;
            stats->policy_mass = prob[top];
            stats->mv[0] = mv[top];
            stats->visits[0] = 0;
            stats->q[0] = value;
            stats->se[0] = 0.0; stats->prior[0] = prob[top];
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
                stats->nlegal = n;
                stats->worlds = 0;
                stats->max_worlds = a->dets;
            stats->resolved = 0;
                stats->raw_best = 0;
                stats->policy_mass = prob[top];
                stats->mv[0] = mv[top];
                stats->visits[0] = 0;
                stats->q[0] = value;
                stats->se[0] = 0.0; stats->prior[0] = prob[top];
                stats->value = value;
            }
            return mv[top];
        }
    }

    /* Candidates are a policy-guided prefix.  Analysis compute belongs on
     * moves the champion genuinely considers, not forced variants with
     * effectively zero prior.  A cumulative-mass target is more stable than a
     * fixed width when the policy ranges from near-certain to genuinely broad. */
    int order[MAX_MOVES];
    for (int i = 0; i < n; i++) order[i] = i;
    int maxcand = a->root_width < MAX_CAND ? a->root_width : MAX_CAND;
    if (maxcand < 1) maxcand = 1;
    if (maxcand > n) maxcand = n;
    int nsorted = maxcand;
    if (a->eval_cand > nsorted) {
        nsorted = a->eval_cand < MAX_CAND ? a->eval_cand : MAX_CAND;
        if (nsorted > n) nsorted = n;
    }
    for (int i = 0; i < nsorted; i++) {
        int best = i;
        for (int j = i + 1; j < n; j++) if (prob[order[j]] > prob[order[best]]) best = j;
        int t = order[i]; order[i] = order[best]; order[best] = t;
    }
    int keep = a->min_cand > 1 ? a->min_cand : 1;
    if (keep > maxcand) keep = maxcand;
    int ncand = maxcand;
    if (a->net && a->cand_mass > 0.0f) {
        ncand = 0;
        double mass = 0.0;
        while (ncand < maxcand &&
               (ncand < keep || mass < (double)a->cand_mass)) {
            mass += prob[order[ncand]];
            ncand++;
        }
    } else if (a->net) {
        float floor_p = a->cand_floor > 0.0f ? a->cand_floor : 0.02f;
        while (ncand > keep && prob[order[ncand - 1]] < floor_p) ncand--;
    }
    /* Optional advisory candidates are simply the next policy-ranked moves.
     * They are diagnostic only and never displace the eligible prefix. */
    int neval = ncand;
    if (a->eval_cand > neval) {
        neval = a->eval_cand < nsorted ? a->eval_cand : nsorted;
    }

    double sum[MAX_CAND], sumw[MAX_CAND], sumobj[MAX_CAND];
    for (int i = 0; i < neval; i++) {
        sum[i] = 0.0;
        sumw[i] = 0.0;
        sumobj[i] = 0.0;
    }
    const int p = st->turn;
    int cap = a->dets > 0 ? a->dets : 1;
    int batch = a->batch_dets > 0 ? a->batch_dets : cap;
    if (batch > cap) batch = cap;
    int lastround = st->round == MATCH_ROUNDS - 1;
    double *val = (double *)malloc(sizeof(double) * (size_t)neval * (size_t)cap);
    if (!val) {
        if (out_value) *out_value = value;
        return mv[order[0]];
    }

    /* The root belief distribution is constant across all worlds.  Preparing
     * it once removes hundreds of duplicate network forwards. */
    BeliefDist belief;
    int have_belief = a->net && !a->no_belief &&
                      belief_dist_init(a->net, st, p, a->symmetries,
                                       1.0f, &belief);

    int reps = 0;
    int resolved = 0;
    int rawbest = 0;
    /* Ordinary 1.96-SE intervals are not valid after repeatedly checking up
     * to eight leaders at several batch boundaries.  3.5 is a conservative
     * family-wise guard for as many as 16 looks across eight candidates. */
    double resolve_z = a->override_k > 3.5f ? a->override_k : 3.5;
    int cont_sym = a->playout_symmetries > 0 ? a->playout_symmetries : 1;

    for (int d = 0; d < cap; d++) {
        State world;
        if (have_belief) belief_dist_sample(st, p, rng, &belief, &world);
        else determinize(st, p, rng, &world);
        uint64_t wseed = 0x9E3779B97F4A7C15ULL * (uint64_t)(d + 1) ^ rng->s[0];
        for (int c = 0; c < neval; c++) {
            State s = world;                 /* same world for every candidate */
            lc_apply(&s, mv[order[c]]);
            double w;
            Rng pr;
            if (a->playout_sample) rng_seed(&pr, wseed);   /* same seed per world */
            int m = playout(a->net, &s, p, a->prune_dom,
                            a->playout_sample ? &pr : NULL,
                            cont_sym, &w);
            double obj = rollout_terminal_objective(&s, p, a->win_q);
            if (val) val[(size_t)c * cap + d] = obj;
            sum[c] += m;
            if (w >= 0.0) sumw[c] += w;
            sumobj[c] += obj;
        }
        reps = d + 1;

        /* Sequential paired evaluation: stop once the numerical leader clears
         * every alternative's two-sided confidence bar.  Ambiguous decisions
         * receive more worlds up to cap; obvious ones release the compute. */
        if (reps % batch == 0 || reps == cap) {
            rawbest = 0;
            for (int c = 1; c < neval; c++)
                if (sumobj[c] > sumobj[rawbest]) rawbest = c;
            resolved = reps > 1;
            for (int c = 0; c < neval && resolved; c++) {
                if (c == rawbest) continue;
                double dm = (sumobj[rawbest] - sumobj[c]) / reps;
                double v2 = 0.0;
                for (int j = 0; j < reps; j++) {
                    double x = val[(size_t)rawbest * cap + j]
                             - val[(size_t)c * cap + j] - dm;
                    v2 += x * x;
                }
                double sed = sqrt(v2 / (reps - 1) / reps);
                if (!(dm > resolve_z * sed)) resolved = 0;
            }
            if (a->batch_dets > 0 && resolved) break;
        }
    }

    /* The raw leader is descriptive.  Move selection is deliberately more
     * conservative: only eligible candidates can win, and a challenger must
     * clear the configured paired-error and practical-effect gates. */
    int eligible_best = 0;
    for (int c = 1; c < ncand; c++) {
        if (sumobj[c] > sumobj[eligible_best] ||
            (sumobj[c] == sumobj[eligible_best] && sum[c] > sum[eligible_best]))
            eligible_best = c;
    }
    /* override_k == 0 preserves the historical rollout agent: take its
     * numerical leader.  Positive values opt into the conservative mode used
     * by the audit and by any rollout player that must resist noisy changes. */
    int best = a->override_k <= 0.0f ? eligible_best : 0;
    if (eligible_best != 0 && eligible_best == rawbest && resolved &&
        a->override_k > 0.0f && reps > 1) {
        double dm = (sumobj[eligible_best] - sumobj[0]) / reps;
        double v2 = 0.0;
        for (int d = 0; d < reps; d++) {
            double x = val[(size_t)eligible_best * cap + d]
                     - val[d] - dm;
            v2 += x * x;
        }
        double sed = sqrt(v2 / (reps - 1) / reps);
        if (dm > a->override_k * sed && dm > a->override_min)
            best = eligible_best;
    }
    float bestq = (float)(sumobj[best] / reps);
    if (stats) {
        stats->n = neval;
        stats->nlegal = n;
        stats->worlds = reps;
        stats->max_worlds = cap;
        stats->resolved = resolved;
        stats->raw_best = rawbest;
        stats->policy_mass = 0.0;
        for (int c = 0; c < ncand; c++)
            stats->policy_mass += prob[order[c]];
        for (int c = 0; c < neval; c++) {
            stats->mv[c] = mv[order[c]];
            stats->visits[c] = reps;
            stats->q[c] = sumobj[c] / reps;
            stats->qw[c] = lastround ? sumw[c] / reps : -1.0;
            stats->prior[c] = prob[order[c]];
            double qv = 0.0, dv = 0.0;
            double dm = (sumobj[c] - sumobj[0]) / reps;
            if (reps > 1) {
                for (int d = 0; d < reps; d++) {
                    double qx = val[(size_t)c * cap + d] - stats->q[c];
                    double dx = val[(size_t)c * cap + d] - val[d] - dm;
                    qv += qx * qx;
                    dv += dx * dx;
                }
                qv = sqrt(qv / (reps - 1) / reps);
                dv = sqrt(dv / (reps - 1) / reps);
            }
            stats->se[c] = qv;
            stats->delta[c] = dm;
            stats->dse[c] = c == 0 ? 0.0 : dv;
        }
        stats->value = bestq;
    }
    free(val);
    /* Keep out_value on one stable scale across searched and skipped moves:
     * it is always the policy-network continuation value (ensemble-averaged
     * when enabled). SearchStats.value/q carry the rollout objective. */
    if (out_value) *out_value = value;
    return mv[order[best]];
}
