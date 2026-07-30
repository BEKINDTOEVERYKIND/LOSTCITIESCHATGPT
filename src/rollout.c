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
                      int symmetries, Rng *symrng)
{
    if (net) {
        if (symrng && symmetries > 1)
            return policy_probs_random_sym(net, s, mv, score, symrng,
                                           symmetries);
        return policy_probs_sym(net, s, mv, score, NULL, symmetries);
    }
    int n = lc_moves(s, mv);
    for (int i = 0; i < n; i++) score[i] = heur_move_value_det(s, mv[i]);
    return n;
}

/* Play s out to the end of the round, returning the round margin for player
 * p.  In the final round of a match the round's end decides the match, so
 * *winpts gets the match result (1 win, 0.5 draw, 0 loss) from the carried
 * cumulative totals; in earlier rounds it gets -1 (margin is the only
 * available objective there, and it doubles as the natural proxy).
 * symrng != NULL draws one suit-group member at each downstream decision,
 * avoiding a 20-forward exact average while preserving a greedy actor.
 * sample_actions is a separate robustness ablation: it samples from that
 * member's full policy rather than taking its best move.  Conflating these
 * two sources of randomness made the old fast mode evaluate a much weaker,
 * high-entropy continuation policy instead of approximating the champion. */
static int playout(const Net *net, State *s, int p, int prune, Rng *symrng,
                   int sample_actions, int symmetries, double *winpts)
{
    Move mv[MAX_MOVES];
    float score[MAX_MOVES];
    while (!s->over) {
        int n = rank_moves(net, s, mv, score, symmetries, symrng);
        if (n <= 0) break;
        uint64_t dead = prune ? (lc_dead_cards(s) & s->hand[s->turn]) : 0;
        int best = -1;
        if (sample_actions && symrng && net) {
            float w[MAX_MOVES];
            float tot = 0.0f;
            for (int i = 0; i < n; i++) {
                w[i] = (dead && lc_discard_dominated(s, mv[i], dead)) ? 0.0f : score[i];
                tot += w[i];
            }
            if (tot > 0.0f) best = sample_index(w, n, symrng);
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
    const int nlegal = n;
    if (n <= 1) {
        if (out_value) *out_value = value;
        if (stats) {
            stats->n = n;
            stats->nlegal = nlegal;
            stats->worlds = 0;
            stats->max_worlds = a->dets;
            stats->resolved = n == 1;
            stats->skip_reason = SEARCH_SKIP_FORCED;
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
    int outside_ply =
        (a->ply_lo > 0 && st->nply < a->ply_lo) ||
        (a->ply_hi > 0 && st->nply >= a->ply_hi);
    int outside_deck = a->deck_max > 0 && st->deck_left > a->deck_max;
    if (outside_ply || outside_deck) {
        int top = 0;
        for (int i = 1; i < n; i++) if (prob[i] > prob[top]) top = i;
        if (out_value) *out_value = value;
        if (stats) {
            stats->n = 1;
            stats->nlegal = nlegal;
            stats->worlds = 0;
            stats->max_worlds = a->dets;
            stats->resolved = 0;
            stats->skip_reason = outside_ply ? SEARCH_SKIP_PLY_WINDOW
                                             : SEARCH_SKIP_DECK_PHASE;
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
                stats->nlegal = nlegal;
                stats->worlds = 0;
                stats->max_worlds = a->dets;
                stats->resolved = 0;
                stats->skip_reason = SEARCH_SKIP_POLICY_CONFIDENCE;
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

    /* Optional dead-discard search-focus heuristic.  Apply it only after the
     * raw-policy phase and confidence gates: a late-only search must return
     * the untouched champion early, and confidence means the champion's
     * actual probability rather than a renormalized conditional one. */
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
            if (k > 0) {
                n = k;
                if (a->net) {
                    double kept = 0.0;
                    for (int i = 0; i < n; i++) kept += prob[i];
                    if (kept > 0.0)
                        for (int i = 0; i < n; i++)
                            prob[i] = (float)(prob[i] / kept);
                }
            }
        }
    }
    if (n == 1) {
        if (out_value) *out_value = value;
        if (stats) {
            stats->n = 1;
            stats->nlegal = nlegal;
            stats->worlds = 0;
            stats->max_worlds = a->dets;
            stats->resolved = 1;
            stats->skip_reason = SEARCH_SKIP_ROOT_FOCUS;
            stats->raw_best = 0;
            stats->policy_mass = 1.0;
            stats->mv[0] = mv[0];
            stats->visits[0] = 0;
            stats->q[0] = value;
            stats->se[0] = 0.0;
            stats->prior[0] = 1.0;
            stats->value = value;
        }
        return mv[0];
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
    /* A mass-based shortlist can legitimately contain only the policy leader.
     * With neither an eligible challenger nor requested advisory rows, paired
     * rollouts cannot change or teach anything, so spending hundreds of
     * continuation worlds would be pure waste.  Keep the ordinary path when
     * eval_cand requested advisory Q targets even though they cannot play.
     *
     * Consume only the hidden-world draws that the old one-candidate path
     * consumed.  Continuation playouts use forked RNGs, so this preserves the
     * actor's later decision stream exactly while removing nearly all of the
     * wasted forwards. */
    if (ncand == 1 && neval == 1) {
        const int p = st->turn;
        int cap = a->dets > 0 ? a->dets : 1;
        int reps = cap;
        if (a->batch_dets > 0) {
            reps = a->batch_dets < 2 ? 2 : a->batch_dets;
            if (reps > cap) reps = cap;
        }
        BeliefDist singleton_belief;
        int singleton_have_belief =
            a->net && !a->no_belief &&
            belief_dist_init(a->net, st, p, a->symmetries, 1.0f,
                             &singleton_belief);
        for (int d = 0; d < reps; d++) {
            State ignored;
            if (singleton_have_belief)
                belief_dist_sample(st, p, rng, &singleton_belief, &ignored);
            else
                determinize(st, p, rng, &ignored);
        }
        if (out_value) *out_value = value;
        if (stats) {
            stats->n = 1;
            stats->nlegal = nlegal;
            stats->worlds = 0;
            stats->max_worlds = a->dets;
            stats->resolved = 0;
            stats->skip_reason = SEARCH_SKIP_POLICY_CONFIDENCE;
            stats->raw_best = 0;
            stats->policy_mass = prob[order[0]];
            stats->mv[0] = mv[order[0]];
            stats->visits[0] = 0;
            stats->q[0] = value;
            stats->se[0] = 0.0;
            stats->prior[0] = prob[order[0]];
            stats->value = value;
        }
        return mv[order[0]];
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
    /* Fork confirmation randomness before the adaptive primary loop.  Its
     * worlds must not depend on which batch boundary happened to stop the
     * discovery estimate, and running a confirmation must not perturb the
     * agent's future decision RNG stream. */
    uint64_t confirm_seed_base =
        rng->s[0] ^ rotl64(rng->s[1], 13) ^
        rotl64(rng->s[2], 29) ^ rotl64(rng->s[3], 47) ^
        UINT64_C(0xA0761D6478BD642F);

    int reps = 0;
    int resolved = 0;
    int rawbest = 0;
    /* Ordinary 1.96-SE intervals are not valid after repeatedly checking up
     * to eight leaders at several batch boundaries.  3.5 is a conservative
     * family-wise guard for as many as 16 looks across eight candidates. */
    double resolve_z = a->override_k > 3.5f ? a->override_k : 3.5;
    int cont_sym = a->playout_symmetries > 0 ? a->playout_symmetries : 1;
    int cont_prune =
        a->playout_prune < 0 ? a->prune_dom : a->playout_prune != 0;
    int random_cont_sym = a->playout_sample > 0;
    int sample_cont_actions = a->playout_sample == 1;

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
            if (random_cont_sym) rng_seed(&pr, wseed); /* same seed per world */
            int m = playout(a->net, &s, p, cont_prune,
                            random_cont_sym ? &pr : NULL,
                            sample_cont_actions, cont_sym, &w);
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
     * conservative: only eligible candidates can win, and every challenger is
     * tested directly against candidate zero, the policy leader. */
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
    int confirmed = 0, confirm_worlds = 0;
    int qual[MAX_CAND], nqual = 0;
    double csum[MAX_CAND] = { 0 }, csum2[MAX_CAND] = { 0 };
    double cdelta[MAX_CAND] = { 0 }, cdse[MAX_CAND] = { 0 };
    uint8_t pqualified[MAX_CAND] = { 0 };
    uint8_t csupported[MAX_CAND] = { 0 };
    uint8_t guard_rejected[MAX_CAND] = { 0 };

    if (a->override_k > 0.0f && reps > 1) {
        /* Qualify each challenger independently.  Requiring the raw numerical
         * leader to survive made a biased leader hide a smaller real
         * correction: when that leader failed confirmation the old code fell
         * all the way back to policy. */
        for (int c = 1; c < ncand; c++) {
            double dm = (sumobj[c] - sumobj[0]) / reps;
            double v2 = 0.0;
            for (int d = 0; d < reps; d++) {
                double x = val[(size_t)c * cap + d] - val[d] - dm;
                v2 += x * x;
            }
            double sed = sqrt(v2 / (reps - 1) / reps);
            if (dm > resolve_z * sed && dm > a->override_min) {
                pqualified[c] = 1;
                qual[nqual++] = c;
            }
        }

        if (nqual > 0) {
            /* An exact-ensemble continuation can repeat one downstream policy
             * discontinuity in every paired world, producing a large mean
             * with a misleadingly tiny SE.  Validate all primary qualifiers
             * on one fresh, fixed-size batch of hidden worlds.  Each decision
             * uses a random member of the requested suit group but remains
             * greedy: this perturbs the discontinuity without changing the
             * target into a weak high-entropy policy.  Candidate zero is
             * played out once per world, so no confirmation compute is spent
             * on candidates that failed the primary screen. */
            confirm_worlds =
                a->confirm_dets > 0 ? a->confirm_dets : cap;
            if (confirm_worlds < 2) confirm_worlds = 2;
            Rng confirm_rng;
            rng_seed(&confirm_rng, confirm_seed_base);
            for (int d = 0; d < confirm_worlds; d++) {
                State world;
                if (have_belief)
                    belief_dist_sample(st, p, &confirm_rng, &belief, &world);
                else
                    determinize(st, p, &confirm_rng, &world);
                uint64_t wseed =
                    UINT64_C(0xD1B54A32D192ED03) * (uint64_t)(d + 1)
                    ^ confirm_rng.s[0];
                State baseline = world;
                lc_apply(&baseline, mv[order[0]]);
                Rng brng;
                rng_seed(&brng, wseed);
                (void)playout(a->net, &baseline, p, cont_prune,
                              &brng, 0, cont_sym, NULL);
                double bobj =
                    rollout_terminal_objective(&baseline, p, a->win_q);

                for (int q = 0; q < nqual; q++) {
                    int c = qual[q];
                    State challenger = world;
                    lc_apply(&challenger, mv[order[c]]);
                    Rng crng;
                    rng_seed(&crng, wseed);
                    (void)playout(a->net, &challenger, p, cont_prune,
                                  &crng, 0, cont_sym, NULL);
                    double x =
                        rollout_terminal_objective(&challenger, p, a->win_q)
                        - bobj;
                    csum[c] += x;
                    csum2[c] += x * x;
                }
            }

            for (int q = 0; q < nqual; q++) {
                int c = qual[q];
                cdelta[c] = csum[c] / confirm_worlds;
                if (confirm_worlds > 1) {
                    double cv =
                        csum2[c] - csum[c] * csum[c] / confirm_worlds;
                    if (cv < 0.0) cv = 0.0;
                    cdse[c] =
                        sqrt(cv / (confirm_worlds - 1) / confirm_worlds);
                }
                /* The discovery pass already paid the family-wise 3.5-SE
                 * guard.  This independent two-move check asks for 99%
                 * confirmation and at least half the practical-effect floor. */
                if (cdelta[c] > 2.58 * cdse[c] &&
                    cdelta[c] > 0.5 * a->override_min) {
                    /* Report the evaluator's independent statistical support
                     * separately from the optional strategic safety rail.
                     * Consumers can then distinguish "search supports this"
                     * from "the guard declined to execute it". */
                    csupported[c] = 1;
                    uint64_t dead = lc_dead_cards(st);
                    int baseline_guarded =
                        lc_discard_dominated(st, mv[order[0]], dead);
                    if (a->discard_guard && !baseline_guarded &&
                        lc_discard_dominated(st, mv[order[c]], dead)) {
                        guard_rejected[c] = 1;
                        continue;
                    }
                    if (best == 0 || cdelta[c] > cdelta[best] ||
                        (cdelta[c] == cdelta[best] &&
                         sumobj[c] > sumobj[best]))
                        best = c;
                }
            }
            confirmed = best != 0;
        }
    }
    float bestq = (float)(sumobj[best] / reps);
    if (stats) {
        stats->n = neval;
        stats->nlegal = nlegal;
        stats->worlds = reps;
        stats->max_worlds = cap;
        stats->resolved = resolved;
        stats->raw_best = rawbest;
        stats->confirmed = confirmed;
        stats->confirm_worlds = confirm_worlds;
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
            stats->cdelta[c] = cdelta[c];
            stats->cdse[c] = cdse[c];
            stats->pqualified[c] = pqualified[c];
            stats->csupported[c] = csupported[c];
            stats->guard_rejected[c] = guard_rejected[c];
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
