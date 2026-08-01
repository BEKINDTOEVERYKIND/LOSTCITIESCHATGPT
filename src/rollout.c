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
#include "planner.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MAX_CAND 8
#define URGENT_SEMANTIC_WORLDS 16384

/* Rank the legal moves of s for the player to move.  With a network that is
 * the policy head; without one it is the hand-crafted evaluation, which gives
 * the classical "heuristic + perfect-information Monte Carlo" baseline. */
static int rank_moves(const Net *net, const State *s, Move *mv, float *score,
                      int symmetries, Rng *symrng,
                      const uint8_t fixed_perm[NSUIT])
{
    if (net) {
        if (fixed_perm)
            return policy_probs_perm(net, s, mv, score, NULL, fixed_perm);
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
 * symrng without fixed_perm draws one suit-group member at each downstream
 * decision.  fixed_perm instead retains one member for the full sampled
 * world, avoiding a temporally inconsistent change of network orientation.
 * sample_actions is a separate robustness ablation: it samples from that
 * member's full policy rather than taking its best move.  Conflating these
 * two sources of randomness made the old fast mode evaluate a much weaker,
 * high-entropy continuation policy instead of approximating the champion. */
static int playout(const Net *net, State *s, int p, int prune, Rng *symrng,
                   const uint8_t fixed_perm[NSUIT], int sample_actions,
                   int symmetries, int plan_deck_max, int plan_block_gap,
                   double *winpts)
{
    Move mv[MAX_MOVES];
    float score[MAX_MOVES];
    while (!s->over) {
        int n = rank_moves(net, s, mv, score, symmetries, symrng,
                           fixed_perm);
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
        /* Continuation values need a horizon-aware lower bound, not another
         * myopic policy step.  Once the deck is short, follow an optimal
         * schedule of cards already visible in the current player's hand.
         * This is information-set safe and normalizes commuting play orders;
         * the production actor remains more conservative at the root. */
        if (!sample_actions && net && plan_deck_max > 0 &&
            plan_block_gap > 0 && s->deck_left <= plan_deck_max) {
            int order[MAX_MOVES];
            for (int i = 0; i < n; i++) order[i] = i;
            int keep = n < 8 ? n : 8;
            for (int i = 0; i < keep; i++) {
                int top = i;
                for (int j = i + 1; j < n; j++)
                    if (score[order[j]] > score[order[top]]) top = j;
                int tmp = order[i]; order[i] = order[top]; order[top] = tmp;
            }
            int planned = hand_plan_choose(
                s, s->turn, mv, score, order, keep,
                (s->deck_left + 1) / 2);
            if (planned >= 0) best = planned;
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

static int move_equal(Move a, Move b)
{
    return MOVE_PACK(a) == MOVE_PACK(b);
}

static int find_move(const Move *mv, int n, Move target)
{
    for (int i = 0; i < n; i++)
        if (move_equal(mv[i], target)) return i;
    return -1;
}

static int append_unique(int *order, int *count, int limit, int index)
{
    if (index < 0) return 0;
    for (int i = 0; i < *count; i++)
        if (order[i] == index) return 0;
    if (*count >= limit) return 0;
    order[(*count)++] = index;
    return 1;
}

static int same_semantic_action(Move a, Move b)
{
    if (a.discard != b.discard) return 0;
    if (CARD_IS_WAGER(a.card) && CARD_IS_WAGER(b.card))
        return CARD_SUIT(a.card) == CARD_SUIT(b.card);
    return a.card == b.card;
}

/* Add only pile-draw variants of the highest-prior distinct play/discard
 * actions.  The complete-move policy factorization can make one global draw
 * preference suppress every pile alternative at once; ranking action cores
 * first preserves the policy's card decision while allowing rollout to value
 * late-round tempo.  First guarantee one best pile option per requested core,
 * then fill the audit's remaining slots by prior.  The caller adds planner
 * and semantic challengers first so this optional expansion cannot crowd out
 * the more targeted corrections. */
static int add_top_action_pile_variants(
    const Move *mv, const float *prob, int n,
    const int *ranked, int nsorted, int wanted_cores,
    int *order, int *ncand)
{
    if (wanted_cores < 1) return 0;
    if (wanted_cores > 2) wanted_cores = 2;
    int core[2], ncore = 0;
    for (int r = 0; r < nsorted && ncore < wanted_cores; r++) {
        int distinct = 1;
        for (int c = 0; c < ncore; c++)
            if (same_semantic_action(mv[ranked[r]], mv[core[c]])) {
                distinct = 0;
                break;
            }
        if (distinct) core[ncore++] = ranked[r];
    }

    int added = 0;
    for (int c = 0; c < ncore && *ncand < MAX_CAND; c++) {
        int best = -1;
        for (int i = 0; i < n; i++) {
            if (mv[i].draw == 0 ||
                !same_semantic_action(mv[i], mv[core[c]]))
                continue;
            int present = 0;
            for (int k = 0; k < *ncand; k++)
                if (order[k] == i) { present = 1; break; }
            if (!present && (best < 0 || prob[i] > prob[best])) best = i;
        }
        added += append_unique(order, ncand, MAX_CAND, best);
    }

    while (*ncand < MAX_CAND) {
        int best = -1;
        for (int i = 0; i < n; i++) {
            if (mv[i].draw == 0) continue;
            int selected_core = 0;
            for (int c = 0; c < ncore; c++)
                if (same_semantic_action(mv[i], mv[core[c]])) {
                    selected_core = 1;
                    break;
                }
            if (!selected_core) continue;
            int present = 0;
            for (int k = 0; k < *ncand; k++)
                if (order[k] == i) { present = 1; break; }
            if (!present && (best < 0 || prob[i] > prob[best])) best = i;
        }
        if (best < 0) break;
        added += append_unique(order, ncand, MAX_CAND, best);
    }
    return added;
}

static int top_action_group(const Move *mv, const float *prob, int n,
                            int index, int limit)
{
    double candidate = 0.0;
    for (int i = 0; i < n; i++)
        if (same_semantic_action(mv[i], mv[index]))
            candidate += prob[i];
    int better = 0;
    for (int i = 0; i < n; i++) {
        int first = 1;
        for (int j = 0; j < i; j++)
            if (same_semantic_action(mv[j], mv[i])) {
                first = 0;
                break;
            }
        if (!first) continue;
        double action_prob = 0.0;
        for (int j = 0; j < n; j++)
            if (same_semantic_action(mv[j], mv[i]))
                action_prob += prob[j];
        if (action_prob > candidate) better++;
    }
    return better < limit;
}

/* With one deck card left, drawing it after the same play/discard ends the
 * round immediately.  Taking a pile instead merely gives the opponent an
 * extra optional scoring turn, so the deck variant weakly dominates. */
static int last_deck_equivalent(const State *st, const Move *mv, int n,
                                int policy_top)
{
    if (st->deck_left != 1 || mv[policy_top].draw == 0) return -1;
    for (int i = 0; i < n; i++)
        if (mv[i].card == mv[policy_top].card &&
            mv[i].discard == mv[policy_top].discard && mv[i].draw == 0)
            return i;
    return -1;
}

/* A wager that is the only card of its suit in our hand, before we have
 * started that suit, is a high-value semantic discard candidate once the
 * opponent has played a number and therefore cannot score the wager.  It is
 * not literally cost-free—the opponent could pick it up to stall—so it still
 * has to beat the baseline in both large stochastic comparisons.  This is
 * deliberately much narrower than evaluating every legal discard. */
static int isolated_one_sided_wager_discard(
    const State *st, const Move *mv, const float *prob, int n)
{
    const int p = st->turn, o = p ^ 1;
    int best = -1, best_score = -1000000;
    for (int i = 0; i < n; i++) {
        int c = mv[i].card, s = CARD_SUIT(c);
        if (!mv[i].discard || !CARD_IS_WAGER(c) ||
            st->exp_n[p][s] != 0 || st->exp_top[o][s] == 0)
            continue;
        int suit_cards = 0;
        uint64_t hand = st->hand[p];
        while (hand) {
            int h = __builtin_ctzll(hand);
            hand &= hand - 1;
            if (CARD_SUIT(h) == s) suit_cards++;
        }
        if (suit_cards != 1) continue;

        /* Compare the known post-move hands without ever reading a hidden
         * deck card.  This lets a one-sided wager discard pair naturally with a
         * useful face-up pickup (p59/p61) instead of forcing the deck variant
         * and spending a second semantic slot on the Cartesian combination. */
        State after = *st;
        if (mv[i].draw == 0) {
            after.hand[p] &= ~(1ULL << c);
            if (after.hand_n[p] > 0) after.hand_n[p]--;
            if (after.deck_left > 0) after.deck_left--;
        } else {
            lc_apply(&after, mv[i]);
        }
        HandPlan plan;
        hand_plan_build(&after, p, after.deck_left / 2, &plan);
        if (best < 0 || plan.score > best_score ||
            (plan.score == best_score &&
             mv[i].draw == 0 && mv[best].draw != 0) ||
            (plan.score == best_score &&
             mv[i].draw == mv[best].draw && prob[i] > prob[best])) {
            best = i;
            best_score = plan.score;
        }
    }
    return best;
}

/* Add one purposeful pile pickup, not one move per draw source.  Number cards
 * must improve the exact visible-hand schedule (for example R4 before held
 * R8/R9).  A wager may instead be denied when public knowledge proves that
 * the opponent has already picked up or committed to that suit. */
static int useful_pile_pickup(
    const State *st, const Move *mv, const float *prob, int n,
    int discard)
{
    const int p = st->turn, o = p ^ 1;
    int best = -1, best_gain = -1000000, best_score = -1000000;
    for (int s = 0; s < NSUIT; s++) {
        if (st->pile_n[s] == 0) continue;
        int pickup = st->pile[s][st->pile_n[s] - 1];
        int wager = CARD_IS_WAGER(pickup);
        if (wager && st->exp_top[p][s] != 0) continue;
        uint64_t suit_mask =
            ((UINT64_C(1) << NRANK) - 1) << (s * NRANK);
        uint64_t wager_mask =
            ((UINT64_C(1) << WAGERS_PER_SUIT) - 1) << (s * NRANK);
        int public_signal =
            (st->known[o] & wager_mask) != 0 || st->exp_n[o][s] != 0;
        int own_numbers =
            __builtin_popcountll(st->hand[p] & suit_mask & ~wager_mask);
        if (wager && !public_signal && own_numbers < 2) continue;

        for (int i = 0; i < n; i++) {
            if (mv[i].draw != s + 1 || mv[i].discard != discard ||
                !top_action_group(mv, prob, n, i, 3))
                continue;
            State after = *st;
            lc_apply(&after, mv[i]);
            HandPlan plan;
            hand_plan_build(&after, p, after.deck_left / 2, &plan);
            int uses_pickup = (plan.used_cards >> pickup) & 1ULL;
            State without = after;
            without.hand[p] &= ~(1ULL << pickup);
            if (without.hand_n[p] > 0) without.hand_n[p]--;
            HandPlan no_pickup;
            hand_plan_build(&without, p, after.deck_left / 2, &no_pickup);
            int gain = plan.score - no_pickup.score;
            if ((!wager && (!uses_pickup || gain <= 0)) ||
                (wager && !uses_pickup && !public_signal))
                continue;
            if (best < 0 || gain > best_gain ||
                (gain == best_gain && plan.score > best_score) ||
                (gain == best_gain && plan.score == best_score &&
                 prob[i] > prob[best])) {
                best = i;
                best_gain = gain;
                best_score = plan.score;
            }
        }
    }
    return best;
}

/* Recheck a proposed ordinary policy-prefix override on fresh worlds with a
 * coherent continuation actor.  Suit mappings are balanced (counts differ by
 * at most one), fixed for each complete trajectory, and shared by every root
 * candidate.  The proposal survives only when the independent panel selects
 * the same numerical leader; disagreement falls back to the raw policy. */
static int confirm_trusted_prefix(
    const Agent *a, const State *st, int p,
    const Move *mv, const int *order, int ntrusted, int proposed,
    int cont_prune, int cont_sym, const BeliefDist *belief, int have_belief,
    uint64_t seed, int cap, int *worlds_out,
    double *q_out, double *se_out)
{
    *worlds_out = 0;
    if (proposed <= 0 || ntrusted <= 1) return proposed;
    int worlds = a->confirm_dets > 0 ? a->confirm_dets : cap;
    if (worlds < 2) worlds = 2;

    uint8_t perms[120][NSUIT];
    int nperm = suit_permutations(cont_sym, perms);
    Rng perm_rng;
    rng_seed(&perm_rng, seed ^ UINT64_C(0x8CB92BA72F3D8DD7));
    int offset = (int)rng_below(&perm_rng, (uint32_t)nperm);
    Rng world_rng;
    rng_seed(&world_rng, seed ^ UINT64_C(0xDB4F0B9175AE2165));

    double total[MAX_CAND] = { 0 };
    double total2[MAX_CAND] = { 0 };
    double margin[MAX_CAND] = { 0 };
    for (int d = 0; d < worlds; d++) {
        State world;
        if (have_belief)
            belief_dist_sample(st, p, &world_rng, belief, &world);
        else
            determinize(st, p, &world_rng, &world);
        const uint8_t *perm = perms[(offset + d) % nperm];
        for (int c = 0; c < ntrusted; c++) {
            State s = world;
            lc_apply(&s, mv[order[c]]);
            int m = playout(a->net, &s, p, cont_prune, NULL, perm, 0,
                            cont_sym, a->plan_deck_max,
                            a->plan_block_gap, NULL);
            double obj = rollout_terminal_objective(&s, p, a->win_q);
            total[c] += obj;
            total2[c] += obj * obj;
            margin[c] += m;
        }
    }
    *worlds_out = worlds;
    for (int c = 0; c < ntrusted; c++) {
        q_out[c] = total[c] / worlds;
        double centered = total2[c] - total[c] * total[c] / worlds;
        if (centered < 0.0) centered = 0.0;
        se_out[c] = sqrt(centered / (worlds - 1) / worlds);
    }
    int leader = 0;
    for (int c = 1; c < ntrusted; c++)
        if (total[c] > total[leader] ||
            (total[c] == total[leader] && margin[c] > margin[leader]))
            leader = c;
    return leader == proposed ? proposed : 0;
}

Move rollout_move(const struct Agent *a, const State *st, Rng *rng,
                  float *out_value, SearchStats *stats)
{
    if (stats) {
        memset(stats, 0, sizeof *stats);
        stats->policy_top = -1;
        stats->metric_kind = SEARCH_METRIC_NETWORK_VALUE;
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

    int policy_top_index = 0;
    for (int i = 1; i < n; i++)
        if (prob[i] > prob[policy_top_index]) policy_top_index = i;
    Move policy_top_move = mv[policy_top_index];
    int baseline_index =
        last_deck_equivalent(st, mv, n, policy_top_index);
    int deck_end_baseline =
        baseline_index >= 0 && baseline_index != policy_top_index;
    int baseline_from_planner = 0;
    if (baseline_index < 0 && a->net && a->plan_deck_max > 0 &&
        a->plan_block_gap > 0 && st->deck_left <= a->plan_deck_max) {
        int planned_order[MAX_MOVES];
        for (int i = 0; i < n; i++) planned_order[i] = i;
        int planned_keep = n < 8 ? n : 8;
        for (int i = 0; i < planned_keep; i++) {
            int top = i;
            for (int j = i + 1; j < n; j++)
                if (prob[planned_order[j]] > prob[planned_order[top]])
                    top = j;
            int tmp = planned_order[i];
            planned_order[i] = planned_order[top];
            planned_order[top] = tmp;
        }
        baseline_index = hand_plan_conservative_choose(
            st, st->turn, mv, prob, planned_order, planned_keep,
            (st->deck_left + 1) / 2, a->plan_block_gap);
        baseline_from_planner =
            baseline_index >= 0 && baseline_index != policy_top_index;
    }
    if (baseline_index < 0) baseline_index = policy_top_index;
    int changed_baseline = baseline_from_planner || deck_end_baseline;
    Move baseline_move = mv[baseline_index];
    int safe_discard_index = a->semantic_cand
        ? isolated_one_sided_wager_discard(st, mv, prob, n) : -1;
    int pile_play_index = a->semantic_cand
        ? useful_pile_pickup(st, mv, prob, n, 0) : -1;
    int pile_discard_index = a->semantic_cand
        ? useful_pile_pickup(st, mv, prob, n, 1) : -1;
    const int have_safe_discard = safe_discard_index >= 0;
    const int have_pile_play = pile_play_index >= 0;
    const int have_pile_discard = pile_discard_index >= 0;
    const Move safe_discard_move =
        have_safe_discard ? mv[safe_discard_index] : policy_top_move;
    const Move pile_play_move =
        have_pile_play ? mv[pile_play_index] : policy_top_move;
    const Move pile_discard_move =
        have_pile_discard ? mv[pile_discard_index] : policy_top_move;

    /* A validated visible-hand correction is the deployed baseline, not
     * another noisy rollout candidate.  Continuing to simulate here allowed
     * the same myopic scheduler error to "confirm" itself and undo the exact
     * correction.  Likewise, with one deck card left the dominance rule is
     * exact.  Report both the corrected baseline and raw policy, but spend no
     * hidden worlds. */
    if (changed_baseline) {
        int turns = (st->deck_left + 1) / 2;
        HandPlan plan;
        hand_plan_build(st, st->turn, turns, &plan);
        int selected_score = baseline_from_planner
            ? hand_plan_score_after_play(
                  st, st->turn, baseline_move.card, turns)
            : plan.score;
        int policy_score = baseline_from_planner
            ? hand_plan_score_after_play(
                  st, st->turn, policy_top_move.card, turns)
            : plan.score;
        if (out_value) *out_value = value;
        if (stats) {
            stats->n = 2;
            stats->nlegal = nlegal;
            stats->worlds = 0;
            stats->max_worlds = a->dets;
            stats->resolved = 1;
            stats->skip_reason = baseline_from_planner
                ? SEARCH_SKIP_VISIBLE_PLAN : SEARCH_SKIP_LAST_DECK;
            stats->raw_best = 0;
            stats->policy_top = 1;
            stats->planned_baseline = baseline_from_planner;
            stats->deck_end_baseline = deck_end_baseline;
            stats->metric_kind = baseline_from_planner
                ? SEARCH_METRIC_VISIBLE_PLAN
                : SEARCH_METRIC_LAST_DECK_RULE;
            stats->planner_turns = baseline_from_planner ? turns : 0;
            stats->planner_score = baseline_from_planner ? plan.score : 0;
            stats->planner_policy_score =
                baseline_from_planner ? policy_score : 0;
            stats->planner_regret =
                baseline_from_planner ? plan.score - policy_score : 0;
            stats->planner_policy_block = baseline_from_planner
                ? hand_plan_block_cost(st, st->turn, policy_top_move.card) : 0;
            stats->planner_selected_block = baseline_from_planner
                ? hand_plan_block_cost(st, st->turn, baseline_move.card) : 0;
            stats->policy_mass =
                prob[baseline_index] + prob[policy_top_index];
            stats->mv[0] = baseline_move;
            stats->mv[1] = policy_top_move;
            stats->q[0] = baseline_from_planner ? selected_score : value;
            stats->q[1] = baseline_from_planner ? policy_score : value;
            stats->delta[0] = 0.0;
            stats->delta[1] = baseline_from_planner
                ? (double)(policy_score - selected_score) : 0.0;
            stats->prior[0] = prob[baseline_index];
            stats->prior[1] = prob[policy_top_index];
            stats->value = value;
        }
        return baseline_move;
    }

    /* Outside the rollout window, the validated scheduler remains part of
     * the policy actor.  Only the Monte Carlo comparison is phase-gated,
     * except for the narrow one-sided-wager signal: that public constraint is
     * how an otherwise low-prior correction at an early round ply earns a
     * focused comparison. */
    int outside_ply =
        (a->ply_lo > 0 && st->nply < a->ply_lo) ||
        (a->ply_hi > 0 && st->nply >= a->ply_hi);
    int outside_deck = a->deck_max > 0 && st->deck_left > a->deck_max;
    int urgent_semantic =
        (outside_ply || outside_deck) && safe_discard_index >= 0 &&
        !move_equal(safe_discard_move, baseline_move);
    if ((outside_ply || outside_deck) && !urgent_semantic) {
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
            stats->policy_mass = prob[baseline_index];
            stats->policy_top = changed_baseline ? -1 : 0;
            stats->planned_baseline = baseline_from_planner;
            stats->deck_end_baseline = deck_end_baseline;
            stats->mv[0] = baseline_move;
            stats->visits[0] = 0;
            stats->q[0] = value;
            stats->se[0] = 0.0;
            stats->prior[0] = prob[baseline_index];
            stats->value = value;
        }
        return baseline_move;
    }

    /* confidence gate: when the policy is already near-certain, searching can
     * only confirm it or override it with noise -- return the policy move and
     * spend the compute where decisions are actually contested */
    if (a->gate > 0.0f) {
        if (prob[policy_top_index] >= a->gate) {
            if (out_value) *out_value = value;
            if (stats) {
                stats->n = 1;
                stats->nlegal = nlegal;
                stats->worlds = 0;
                stats->max_worlds = a->dets;
                stats->resolved = 0;
                stats->skip_reason = SEARCH_SKIP_POLICY_CONFIDENCE;
                stats->raw_best = 0;
                stats->policy_mass = prob[baseline_index];
                stats->policy_top = changed_baseline ? -1 : 0;
                stats->planned_baseline = baseline_from_planner;
                stats->deck_end_baseline = deck_end_baseline;
                stats->mv[0] = baseline_move;
                stats->visits[0] = 0;
                stats->q[0] = value;
                stats->se[0] = 0.0;
                stats->prior[0] = prob[baseline_index];
                stats->value = value;
            }
            return baseline_move;
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
    /* Root-focus pruning compacts mv[] and prob[].  Semantic candidates were
     * identified before the phase gate, so remap their move identities rather
     * than retaining now-stale array indices. */
    safe_discard_index =
        have_safe_discard ? find_move(mv, n, safe_discard_move) : -1;
    pile_play_index =
        have_pile_play ? find_move(mv, n, pile_play_move) : -1;
    pile_discard_index =
        have_pile_discard ? find_move(mv, n, pile_discard_move) : -1;
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
    if ((a->semantic_cand || a->plan_deck_max > 0 ||
         a->draw_variant_cores > 0) && nsorted < MAX_CAND) {
        nsorted = n < MAX_CAND ? n : MAX_CAND;
    }
    for (int i = 0; i < nsorted; i++) {
        int best = i;
        for (int j = i + 1; j < n; j++) if (prob[order[j]] > prob[order[best]]) best = j;
        int t = order[i]; order[i] = order[best]; order[best] = t;
    }
    int ranked[MAX_CAND];
    for (int i = 0; i < nsorted; i++) ranked[i] = order[i];
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

    int current_baseline = -1, current_policy_top = -1;
    for (int i = 0; i < n; i++) {
        if (move_equal(mv[i], baseline_move)) current_baseline = i;
        if (move_equal(mv[i], policy_top_move)) current_policy_top = i;
    }
    if (current_baseline < 0) {
        current_baseline = current_policy_top >= 0
            ? current_policy_top : order[0];
        baseline_from_planner = 0;
        deck_end_baseline = 0;
        changed_baseline = 0;
    }
    append_unique(order, &ncand, MAX_CAND, current_baseline);
    int trusted_candidates = ncand;

    /* One exact hand-scheduling challenger is enough.  It comes from the
     * policy's top eight, so this remains a focused audit rather than a scan
     * of every legal move. */
    if (a->net && a->plan_deck_max > 0 &&
        st->deck_left <= a->plan_deck_max) {
        int planned = hand_plan_choose(
            st, st->turn, mv, prob, ranked, nsorted,
            (st->deck_left + 1) / 2);
        append_unique(order, &ncand, MAX_CAND, planned);
    }
    int semantic_added = 0;
    if (a->semantic_cand) {
        semantic_added +=
            append_unique(order, &ncand, MAX_CAND, safe_discard_index);
        semantic_added +=
            append_unique(order, &ncand, MAX_CAND, pile_play_index);
        semantic_added +=
            append_unique(order, &ncand, MAX_CAND, pile_discard_index);
    }
    int draw_variant_added = 0;
    if (a->net && a->draw_variant_cores > 0 &&
        (a->draw_variant_deck_max <= 0 ||
         st->deck_left <= a->draw_variant_deck_max)) {
        draw_variant_added = add_top_action_pile_variants(
            mv, prob, n, ranked, nsorted, a->draw_variant_cores,
            order, &ncand);
    }
    /*
     * An early one-sided-wager trigger is deliberately a two-move test.  Its
     * purpose is to rescue one low-prior semantic blind spot, not to turn the
     * entire early policy prefix into a 16,384-world audit.  This keeps the
     * expensive path at baseline versus wager-discard only.
     */
    if (urgent_semantic) {
        order[0] = current_baseline;
        ncand = 1;
        semantic_added =
            append_unique(order, &ncand, MAX_CAND, safe_discard_index);
        draw_variant_added = 0;
        trusted_candidates = 1;
    }

    /* Candidate zero is the deployed actor's baseline: ordinarily the raw
     * policy, but occasionally the validated scheduler or exact deck=1 rule.
     * Keep the untouched policy leader in the list and report its true index. */
    int baseline_pos = -1;
    for (int i = 0; i < ncand; i++)
        if (order[i] == current_baseline) baseline_pos = i;
    if (baseline_pos > 0) {
        int tmp = order[0]; order[0] = order[baseline_pos];
        order[baseline_pos] = tmp;
    }

    /* Optional advisory rows are the next policy-ranked moves.  They remain
     * diagnostic-only; semantic additions above are eligible because each is
     * a single rule-derived challenger. */
    int neval = ncand;
    int advisory_target = urgent_semantic ? 0 :
        (a->eval_cand < nsorted ? a->eval_cand : nsorted);
    for (int i = 0; i < advisory_target; i++)
        append_unique(order, &neval, MAX_CAND, ranked[i]);

    int policy_pos = -1;
    for (int i = 0; i < neval; i++)
        if (order[i] == current_policy_top) policy_pos = i;
    if (current_policy_top >= 0 && policy_pos < 0) {
        append_unique(order, &neval, MAX_CAND, current_policy_top);
        for (int i = 0; i < neval; i++)
            if (order[i] == current_policy_top) policy_pos = i;
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
            stats->policy_top = policy_pos == 0 ? 0 : -1;
            stats->planned_baseline = baseline_from_planner;
            stats->deck_end_baseline = deck_end_baseline;
            stats->semantic_candidates = semantic_added;
            stats->draw_variant_candidates = draw_variant_added;
            stats->trusted_candidates = trusted_candidates;
            stats->selection_reference = 0;
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
    if (urgent_semantic && cap < URGENT_SEMANTIC_WORLDS)
        cap = URGENT_SEMANTIC_WORLDS;
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
    int fixed_world_sym = a->playout_sample == 3 && cont_sym > 1;
    uint8_t cont_perms[120][NSUIT];
    int ncont_perms = fixed_world_sym
        ? suit_permutations(cont_sym, cont_perms) : 0;
    /* Mode 3 is a coherent sampled member of the suit ensemble.  Stratify the
     * panel instead of independently redrawing that member in every world:
     * over any complete group cycle each relabelling receives exactly the
     * same number of hidden worlds, while the offset remains seed-dependent.
     * This removes a needless source of audit variance without changing the
     * hidden-card distribution or spending worlds on extra root moves. */
    int fixed_perm_offset = fixed_world_sym
        ? (int)(confirm_seed_base % (uint64_t)ncont_perms) : 0;

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
            const uint8_t *fixed_perm = NULL;
            if (fixed_world_sym) {
                fixed_perm =
                    cont_perms[(fixed_perm_offset + d) % ncont_perms];
            }
            int m = playout(a->net, &s, p, cont_prune,
                            random_cont_sym ? &pr : NULL,
                            fixed_perm, sample_cont_actions, cont_sym,
                            a->plan_deck_max, a->plan_block_gap, &w);
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
     * tested directly against candidate zero, the deployed policy/planner
     * baseline. */
    int eligible_best = 0;
    for (int c = 1; c < ncand; c++) {
        if (sumobj[c] > sumobj[eligible_best] ||
            (sumobj[c] == sumobj[eligible_best] && sum[c] > sum[eligible_best]))
            eligible_best = c;
    }
    /* The optional two-tier mode trusts only the ordinary policy-floor prefix
     * as a direct numerical comparison.  Purposefully admitted low-prior pile
     * and semantic variants must still clear both statistical gates. */
    int reference = 0;
    if (a->policy_prefix_mode > 0) {
        for (int c = 1; c < trusted_candidates; c++)
            if (sumobj[c] > sumobj[reference] ||
                (sumobj[c] == sumobj[reference] && sum[c] > sum[reference]))
                reference = c;
    }
    int proposed_reference = reference;
    int prefix_confirm_worlds = 0;
    int prefix_confirmed = 0;
    double prefix_q[MAX_CAND] = { 0 };
    double prefix_se[MAX_CAND] = { 0 };
    if (a->policy_prefix_mode == 2 && proposed_reference != 0) {
        reference = confirm_trusted_prefix(
            a, st, p, mv, order, trusted_candidates, proposed_reference,
            cont_prune, cont_sym, &belief, have_belief,
            confirm_seed_base ^ UINT64_C(0xE7037ED1A0B428DB),
            cap, &prefix_confirm_worlds, prefix_q, prefix_se);
        prefix_confirmed = reference == proposed_reference;
    }
    /* override_k == 0 preserves the fully ungated historical rollout agent
     * only when prefix selection is disabled.  Once a prefix mode is enabled,
     * its numerical or fresh-panel consensus result owns the trusted-prefix
     * decision even when the low-prior challenger gate is off. */
    int best = a->override_k <= 0.0f && a->policy_prefix_mode == 0
        ? eligible_best : reference;
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
         * all the way back to the baseline. */
        for (int c = 0; c < ncand; c++) {
            if (c == reference ||
                (a->policy_prefix_mode > 0 && c < trusted_candidates))
                continue;
            double dm = (sumobj[c] - sumobj[reference]) / reps;
            double v2 = 0.0;
            for (int d = 0; d < reps; d++) {
                double x = val[(size_t)c * cap + d]
                         - val[(size_t)reference * cap + d] - dm;
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
            if (urgent_semantic &&
                confirm_worlds < URGENT_SEMANTIC_WORLDS)
                confirm_worlds = URGENT_SEMANTIC_WORLDS;
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
                lc_apply(&baseline, mv[order[reference]]);
                Rng brng;
                rng_seed(&brng, wseed);
                const uint8_t *confirm_perm = NULL;
                if (fixed_world_sym && !a->confirm_exact5) {
                    int pk = (int)rng_below(&brng, (uint32_t)ncont_perms);
                    confirm_perm = cont_perms[pk];
                }
                (void)playout(a->net, &baseline, p, cont_prune,
                              a->confirm_exact5 || confirm_perm ? NULL : &brng,
                              confirm_perm, 0,
                              a->confirm_exact5 ? 5 : cont_sym,
                              a->plan_deck_max, a->plan_block_gap, NULL);
                double bobj =
                    rollout_terminal_objective(&baseline, p, a->win_q);

                for (int q = 0; q < nqual; q++) {
                    int c = qual[q];
                    State challenger = world;
                    lc_apply(&challenger, mv[order[c]]);
                    Rng crng;
                    rng_seed(&crng, wseed);
                    if (confirm_perm)
                        (void)rng_below(&crng, (uint32_t)ncont_perms);
                    (void)playout(a->net, &challenger, p, cont_prune,
                                  a->confirm_exact5 || confirm_perm
                                      ? NULL : &crng,
                                  confirm_perm, 0,
                                  a->confirm_exact5 ? 5 : cont_sym,
                                  a->plan_deck_max, a->plan_block_gap, NULL);
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
                        lc_discard_dominated(st, mv[order[reference]], dead);
                    /*
                     * The generic discard guard predates the semantic
                     * shortlist and can reject a one-sided wager even after
                     * the focused primary and independent confirmation both
                     * support it.  In this narrow public-information case the
                     * opponent cannot score the wager, while any remaining
                     * stall cost is already represented in both 16,384-world
                     * continuation batches.  Let that evidence supersede the
                     * coarse generic guard.  The maintained urgent path uses
                     * 16,384 primary and 16,384 fresh worlds because the
                     * reviewed ply-61 signal was not stable at 4,096.
                     */
                    int confirmed_one_sided_wager =
                        have_safe_discard &&
                        move_equal(mv[order[c]], safe_discard_move);
                    if (a->discard_guard && !baseline_guarded &&
                        !confirmed_one_sided_wager &&
                        lc_discard_dominated(st, mv[order[c]], dead)) {
                        guard_rejected[c] = 1;
                        continue;
                    }
                    if (best == reference || cdelta[c] > cdelta[best] ||
                        (cdelta[c] == cdelta[best] &&
                         sumobj[c] > sumobj[best]))
                        best = c;
                }
            }
            confirmed = best != reference;
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
        stats->policy_top = policy_pos;
        stats->planned_baseline = baseline_from_planner;
        stats->deck_end_baseline = deck_end_baseline;
        stats->semantic_candidates = semantic_added;
        stats->draw_variant_candidates = draw_variant_added;
        stats->trusted_candidates = trusted_candidates;
        stats->prefix_proposed = proposed_reference;
        stats->selection_reference = reference;
        stats->trusted_prefix_override =
            a->policy_prefix_mode > 0 && reference != 0;
        stats->prefix_confirmed = prefix_confirmed;
        stats->prefix_confirm_worlds = prefix_confirm_worlds;
        stats->metric_kind = SEARCH_METRIC_ROLLOUT;
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
            double qv = 0.0, dv = 0.0, rdv = 0.0;
            double dm = (sumobj[c] - sumobj[0]) / reps;
            double rdm =
                (sumobj[c] - sumobj[reference]) / reps;
            if (reps > 1) {
                for (int d = 0; d < reps; d++) {
                    double qx = val[(size_t)c * cap + d] - stats->q[c];
                    double dx = val[(size_t)c * cap + d] - val[d] - dm;
                    double rdx = val[(size_t)c * cap + d]
                               - val[(size_t)reference * cap + d] - rdm;
                    qv += qx * qx;
                    dv += dx * dx;
                    rdv += rdx * rdx;
                }
                qv = sqrt(qv / (reps - 1) / reps);
                dv = sqrt(dv / (reps - 1) / reps);
                rdv = sqrt(rdv / (reps - 1) / reps);
            }
            stats->se[c] = qv;
            stats->delta[c] = dm;
            stats->dse[c] = c == 0 ? 0.0 : dv;
            stats->rdelta[c] = rdm;
            stats->rdse[c] = c == reference ? 0.0 : rdv;
            stats->cdelta[c] = cdelta[c];
            stats->cdse[c] = cdse[c];
            stats->prefix_q[c] = prefix_q[c];
            stats->prefix_se[c] = prefix_se[c];
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
