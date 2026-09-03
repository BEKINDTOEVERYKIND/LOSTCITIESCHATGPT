/* continuation_arena -- fast paired screen for continuation checkpoints.
 *
 * This is deliberately NOT a promotion arena.  It measures only the policy
 * role reached below a production-shaped ply-14 rollout root:
 *
 *   - a frozen root champion creates genuine match context and plays an exact
 *     20-way, information-safe greedy prefix;
 *   - the champion's literal policy baseline and the exact production
 *     width-5/floor-.02 prefix determine one gradient-free root candidate;
 *   - the mover's information state is uniformly determinized, that same
 *     candidate is applied, and the resulting complete world is cloned;
 *   - candidate and baseline continuation checkpoints play the two clones
 *     with their controller seats swapped.
 *
 * A fixed suit mapping belongs to each player role and is reused in both
 * controller-swap legs.  The legacy default preserves the original adjacent
 * seat-balanced screen.  The opt-in independent schedule matches deployed
 * rollout mode 4: mappings are root-player-relative and every complete 20x20
 * block covers the ordered product exactly once.  All gameplay decisions see
 * agent_information_view(), never the referee's hidden hand/deck.  The exact
 * one-card solver and production cap-reserve deck forcing are preserved.
 *
 * The result is a cheap iteration screen.  A checkpoint that passes here must
 * still face the locked full-actor arena before it can be promoted.
 */
#include "../src/lc.h"
#include "../src/agent.h"
#include "../src/net.h"
#include "../src/search.h"
#include "../src/match.h"

#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

enum {
    ROOT_PLY = 14,
    ROOT_SYMMETRIES = 20,
    ROOT_WIDTH = 5,
    ROLE_MAPPING_LEGACY = 0,
    ROLE_MAPPING_SHARED = 1,
    ROLE_MAPPING_INDEPENDENT = 2
};

static const float ROOT_FLOOR = 0.02f;

typedef struct {
    uint64_t index;
    int round;
    int root_player;
    int admitted;
    int picked;
    uint16_t root_move;
    int cumulative_before[2];
    int player_mapping[2];
    int candidate_seat[2];
    int score[2][2];
    int margin[2]; /* candidate round margin (legacy field name internally) */
    double objective_target[2];
    int final_match_margin[2]; /* defined only for real round index 2 */
    int final_match_result[2]; /* signed -1/0/1, round 2 only */
    double hybrid_target[2];   /* exact mode-2 target, round 2 only */
    int tail_plies[2];
    int capped[2];
    int exact_moves[2];
    int cap_forces[2];
    int cycle_forces[2];
} ContinuationPair;

typedef struct {
    const Net *root;
    const Net *candidate;
    const Net *baseline;
    const NetEvalPlan *root_plan;
    const NetEvalPlan *candidate_plan;
    const NetEvalPlan *baseline_plan;
    ContinuationPair *rows;
    uint64_t pair_start;
    uint64_t seed;
    int pairs;
    int target_round;
    int objective;
    int role_mapping_mode;
    int thread;
    int nthread;
    atomic_int *failed;
} Job;

typedef struct {
    uint64_t pairs;
    double round_margin;
    double round_margin_se;
    double objective_target;
    double objective_target_se;
    double final_match_margin;
    double final_match_margin_se;
    double match_score;
    double match_score_se;
    uint64_t match_wins, match_losses, match_draws;
} RoundSummary;

typedef struct {
    double margin;
    double margin_se;
    double objective;
    double objective_se;
    double score;
    double score_se;
    double candidate_points;
    double baseline_points;
    uint64_t wins, losses, draws;
    uint64_t caps, exact_moves, cap_forces, cycle_forces;
    uint64_t baseline_roots, challenger_roots, singleton_roots;
    RoundSummary round[MATCH_ROUNDS];
} Summary;

static uint64_t mix64(uint64_t x)
{
    x ^= x >> 30;
    x *= UINT64_C(0xBF58476D1CE4E5B9);
    x ^= x >> 27;
    x *= UINT64_C(0x94D049BB133111EB);
    return x ^ (x >> 31);
}

static int parse_u64(const char *s, uint64_t *out)
{
    if (!s || !*s || !out) return 0;
    uint64_t value = 0;
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        if (*p < '0' || *p > '9') return 0;
        uint64_t digit = (uint64_t)(*p - '0');
        if (value > (UINT64_MAX - digit) / UINT64_C(10)) return 0;
        value = value * UINT64_C(10) + digit;
    }
    *out = value;
    return 1;
}

static int parse_int_range(const char *s, int low, int high, int *out)
{
    uint64_t value;
    if (low < 0 || high < low || !parse_u64(s, &value) ||
        value < (uint64_t)low || value > (uint64_t)high)
        return 0;
    *out = (int)value;
    return 1;
}

static const char *role_mapping_name(int mode)
{
    if (mode == ROLE_MAPPING_SHARED) return "shared";
    if (mode == ROLE_MAPPING_INDEPENDENT) return "independent";
    return "legacy-seat-balanced";
}

static int assign_player_mappings(
    uint64_t seed, uint64_t trajectory, int root_player, int mode,
    uint8_t group[120][NSUIT], int player_mapping[2],
    uint8_t perm[2][NSUIT])
{
    if (root_player < 0 || root_player > 1 ||
        (mode != ROLE_MAPPING_LEGACY && mode != ROLE_MAPPING_SHARED &&
         mode != ROLE_MAPPING_INDEPENDENT))
        return 0;
    uint64_t schedule_seed = mix64(
        seed ^ UINT64_C(0xA4093822299F31D0));
    int fixed_offset = (int)(schedule_seed % ROOT_SYMMETRIES);
    int index[2];
    if (mode == ROLE_MAPPING_LEGACY) {
        for (int seat = 0; seat < 2; seat++)
            index[seat] = (fixed_offset
                         + (int)((trajectory % UINT64_C(10)) * UINT64_C(2))
                         + seat) % ROOT_SYMMETRIES;
    } else if (mode == ROLE_MAPPING_SHARED) {
        int shared = (fixed_offset
                    + (int)(trajectory % ROOT_SYMMETRIES))
                   % ROOT_SYMMETRIES;
        index[0] = index[1] = shared;
    } else {
        int other_offset =
            (int)(rotl64(schedule_seed, 31) % ROOT_SYMMETRIES);
        int fixed = (fixed_offset
                   + (int)(trajectory % ROOT_SYMMETRIES))
                  % ROOT_SYMMETRIES;
        int other =
            (other_offset
             + (int)((trajectory / ROOT_SYMMETRIES) % ROOT_SYMMETRIES)
             + (int)(trajectory % ROOT_SYMMETRIES))
            % ROOT_SYMMETRIES;
        index[root_player] = fixed;
        index[root_player ^ 1] = other;
    }
    for (int seat = 0; seat < 2; seat++) {
        player_mapping[seat] = index[seat];
        memcpy(perm[seat], group[index[seat]], NSUIT);
    }
    return 1;
}

static void shuffled_deck(uint64_t seed, uint64_t pair, int round,
                          uint8_t deck[NCARD])
{
    uint64_t key = seed ^ UINT64_C(0x243F6A8885A308D3)
        ^ (pair + UINT64_C(1)) * UINT64_C(0x9E3779B97F4A7C15)
        ^ (uint64_t)(round + 1) * UINT64_C(0xD1B54A32D192ED03);
    Rng rng;
    rng_seed(&rng, mix64(key));
    for (int i = 0; i < NCARD; i++) deck[i] = (uint8_t)i;
    for (int i = NCARD - 1; i > 0; i--) {
        uint32_t j = rng_below(&rng, (uint32_t)i + 1);
        uint8_t tmp = deck[i];
        deck[i] = deck[j];
        deck[j] = tmp;
    }
}

/* Return a move index ranked only from the mover's information state. */
static int greedy_index(const Net *net, const NetEvalPlan *plan,
                        const State *complete, const uint8_t *fixed_perm,
                        int exact20, int prune, int objective,
                        Move *mv, float *prob, int *exact,
                        RolloutLateCycleHistory *cycle_history,
                        int *cap_force, int *cycle_force)
{
    State view;
    agent_information_view(complete, complete->turn, &view);
    int n;
    if (complete->deck_left == 1) {
        n = lc_moves(&view, mv);
        int selected = rollout_exact_terminal_choice(
            &view, mv, NULL, n, objective, NULL);
        if (selected >= 0) {
            if (exact) (*exact)++;
            return selected;
        }
    }
    if (exact20)
        n = policy_probs_sym_plan(
            net, &view, mv, prob, NULL, ROOT_SYMMETRIES, plan);
    else
        n = policy_probs_perm_plan(
            net, &view, mv, prob, NULL, fixed_perm, plan);
    if (n <= 0) return -1;
    uint64_t dead = prune
        ? (lc_dead_cards(&view) & view.hand[view.turn]) : 0;
    int best = -1;
    for (int i = 0; i < n; i++) {
        if (dead && lc_discard_dominated(&view, mv[i], dead)) continue;
        if (best < 0 || prob[i] > prob[best]) best = i;
    }

    int force_cap =
        (int)complete->nply + (int)complete->deck_left >= LC_MAX_PLIES;
    int force_cycle = cycle_history
        ? rollout_late_cycle_repeated(cycle_history, complete) : 0;
    if (force_cap || force_cycle) {
        int previous = best;
        int forced = rollout_policy_deck_choice(&view, mv, prob, n, dead);
        if (forced < 0) return -1;
        if (previous >= 0 && mv[previous].draw != 0) {
            if (force_cap && cap_force) (*cap_force)++;
            else if (force_cycle && cycle_force) (*cycle_force)++;
        }
        best = forced;
    }
    if (best < 0) best = 0;
    return best;
}

static int play_champion_round(const Net *root, const NetEvalPlan *plan,
                               State *st, int objective)
{
    Move mv[MAX_MOVES];
    float prob[MAX_MOVES];
    while (!st->over) {
        int best = greedy_index(
            root, plan, st, NULL, 1, 0, objective, mv, prob, NULL,
            NULL, NULL, NULL);
        if (best < 0) return 0;
        lc_apply(st, mv[best]);
    }
    return st->deck_left == 0;
}

static int play_tail(const State *start, int candidate_seat,
                     const Net *candidate, const NetEvalPlan *candidate_plan,
                     const Net *baseline, const NetEvalPlan *baseline_plan,
                     uint8_t perm[2][NSUIT], ContinuationPair *row,
                     int leg, int objective)
{
    State st = *start;
    Move mv[MAX_MOVES];
    float prob[MAX_MOVES];
    int exact = 0, cap_force = 0, cycle_force = 0;
    int first_ply = st.nply;
    RolloutLateCycleHistory cycle_history;
    rollout_late_cycle_init(&cycle_history);
    while (!st.over) {
        int p = st.turn;
        const Net *net = p == candidate_seat ? candidate : baseline;
        const NetEvalPlan *plan = p == candidate_seat
            ? candidate_plan : baseline_plan;
        int best = greedy_index(
            net, plan, &st, perm[p], 0, 1, objective,
            mv, prob, &exact,
            &cycle_history, &cap_force, &cycle_force);
        if (best < 0) return 0;
        lc_apply(&st, mv[best]);
    }
    row->candidate_seat[leg] = candidate_seat;
    row->score[leg][0] = lc_score(&st, 0);
    row->score[leg][1] = lc_score(&st, 1);
    row->margin[leg] = row->score[leg][candidate_seat]
                     - row->score[leg][candidate_seat ^ 1];
    row->objective_target[leg] = rollout_terminal_objective(
        &st, candidate_seat, objective);
    if (st.round == MATCH_ROUNDS - 1) {
        row->final_match_margin[leg] =
            (int)st.cum[candidate_seat]
          - (int)st.cum[candidate_seat ^ 1] + row->margin[leg];
        row->final_match_result[leg] =
            (row->final_match_margin[leg] > 0)
          - (row->final_match_margin[leg] < 0);
        row->hybrid_target[leg] = rollout_terminal_objective(
            &st, candidate_seat, 2);
    }
    row->tail_plies[leg] = (int)st.nply - first_ply;
    row->capped[leg] = st.deck_left > 0;
    row->exact_moves[leg] = exact;
    row->cap_forces[leg] = cap_force;
    row->cycle_forces[leg] = cycle_force;
    return 1;
}

static int evaluate_pair(const Job *j, uint64_t absolute,
                         ContinuationPair *row)
{
    memset(row, 0, sizeof(*row));
    row->index = absolute;
    int target = j->target_round >= 0
        ? j->target_round : (int)(absolute % MATCH_ROUNDS);
    row->round = target;

    int cum[2] = { 0, 0 };
    for (int round = 0; round < target; round++) {
        uint8_t deck[NCARD];
        shuffled_deck(j->seed, absolute, round, deck);
        State context;
        lc_deal_from_deck(&context, deck);
        context.round = (uint8_t)round;
        context.turn = (uint8_t)(round & 1);
        context.cum[0] = (int16_t)cum[0];
        context.cum[1] = (int16_t)cum[1];
        if (!play_champion_round(
                j->root, j->root_plan, &context, j->objective))
            return 0;
        cum[0] += lc_score(&context, 0);
        cum[1] += lc_score(&context, 1);
    }

    uint8_t deck[NCARD];
    shuffled_deck(j->seed, absolute, target, deck);
    State real;
    lc_deal_from_deck(&real, deck);
    real.round = (uint8_t)target;
    real.turn = (uint8_t)(target & 1);
    real.cum[0] = (int16_t)cum[0];
    real.cum[1] = (int16_t)cum[1];
    row->cumulative_before[0] = cum[0];
    row->cumulative_before[1] = cum[1];

    Move mv[MAX_MOVES];
    float prob[MAX_MOVES];
    while (!real.over && real.nply < ROOT_PLY) {
        int best = greedy_index(
            j->root, j->root_plan, &real, NULL, 1, 0, j->objective,
            mv, prob, NULL, NULL, NULL, NULL);
        if (best < 0) return 0;
        lc_apply(&real, mv[best]);
    }
    if (real.over || real.nply != ROOT_PLY) return 0;

    const int p = real.turn;
    row->root_player = p;
    State root_view;
    agent_information_view(&real, p, &root_view);
    int n = policy_probs_sym_plan(
        j->root, &root_view, mv, prob, NULL,
        ROOT_SYMMETRIES, j->root_plan);
    if (n <= 0) return 0;
    int baseline = 0;
    for (int i = 1; i < n; i++)
        if (prob[i] > prob[baseline]) baseline = i;
    int order[ROLLOUT_MAX_CANDIDATES];
    int admitted = rollout_policy_prefix_indices(
        mv, prob, n, baseline, ROOT_WIDTH, ROOT_FLOOR, 0.0f, 1, order);
    if (admitted <= 0 || order[0] != baseline) return 0;
    row->admitted = admitted;

    int picked = 0;
    if ((absolute & UINT64_C(1)) && admitted > 1)
        picked = 1 + (int)((absolute / UINT64_C(2)) %
                           (uint64_t)(admitted - 1));
    row->picked = picked;
    Move root_move = mv[order[picked]];
    row->root_move = MOVE_PACK(root_move);

    Rng det_rng;
    uint64_t det_key = j->seed ^ UINT64_C(0x13198A2E03707344)
        ^ (absolute + UINT64_C(1)) * UINT64_C(0x94D049BB133111EB);
    rng_seed(&det_rng, mix64(det_key));
    State post;
    determinize(&root_view, p, &det_rng, &post);
    lc_apply(&post, root_move);

    uint8_t group[120][NSUIT];
    int ngroup = suit_permutations(ROOT_SYMMETRIES, group);
    if (ngroup != ROOT_SYMMETRIES) return 0;
    uint8_t perm[2][NSUIT];
    if (!assign_player_mappings(
            j->seed, absolute, p, j->role_mapping_mode, group,
            row->player_mapping, perm))
        return 0;

    return play_tail(&post, p, j->candidate, j->candidate_plan,
                     j->baseline, j->baseline_plan, perm, row, 0,
                     j->objective)
        && play_tail(&post, p ^ 1, j->candidate, j->candidate_plan,
                     j->baseline, j->baseline_plan, perm, row, 1,
                     j->objective);
}

static void *worker(void *arg)
{
    Job *j = (Job *)arg;
    for (int i = j->thread; i < j->pairs; i += j->nthread) {
        if (atomic_load_explicit(j->failed, memory_order_relaxed)) break;
        uint64_t absolute = j->pair_start + (uint64_t)i;
        if (!evaluate_pair(j, absolute, &j->rows[i])) {
            atomic_store_explicit(j->failed, 1, memory_order_relaxed);
            break;
        }
    }
    return NULL;
}

static double result_score(int margin)
{
    return margin > 0 ? 1.0 : (margin == 0 ? 0.5 : 0.0);
}

static void summarize(const ContinuationPair *rows, int pairs, Summary *out)
{
    memset(out, 0, sizeof(*out));
    double margin_sum = 0.0, margin_sumsq = 0.0;
    double configured_sum = 0.0, configured_sumsq = 0.0;
    double score_sum = 0.0, score_sumsq = 0.0;
    double candidate_points = 0.0, baseline_points = 0.0;
    double round_margin_sum[MATCH_ROUNDS] = { 0 };
    double round_margin_sumsq[MATCH_ROUNDS] = { 0 };
    double objective_sum[MATCH_ROUNDS] = { 0 };
    double objective_sumsq[MATCH_ROUNDS] = { 0 };
    double final_margin_sum[MATCH_ROUNDS] = { 0 };
    double final_margin_sumsq[MATCH_ROUNDS] = { 0 };
    double match_score_sum[MATCH_ROUNDS] = { 0 };
    double match_score_sumsq[MATCH_ROUNDS] = { 0 };
    for (int i = 0; i < pairs; i++) {
        int round = rows[i].round;
        if (round < 0 || round >= MATCH_ROUNDS) continue;
        RoundSummary *rs = &out->round[round];
        rs->pairs++;
        double pair_margin = (double)rows[i].margin[0] + rows[i].margin[1];
        double pair_objective = rows[i].objective_target[0]
                              + rows[i].objective_target[1];
        double pair_score = 0.0;
        double pair_final_margin = 0.0;
        double pair_match_score = 0.0;
        for (int leg = 0; leg < 2; leg++) {
            int seat = rows[i].candidate_seat[leg];
            candidate_points += rows[i].score[leg][seat];
            baseline_points += rows[i].score[leg][seat ^ 1];
            double s = result_score(rows[i].margin[leg]);
            pair_score += s;
            if (s == 1.0) out->wins++;
            else if (s == 0.0) out->losses++;
            else out->draws++;
            if (round == MATCH_ROUNDS - 1) {
                int result = rows[i].final_match_result[leg];
                double ms = result > 0 ? 1.0 : (result == 0 ? 0.5 : 0.0);
                pair_final_margin += rows[i].final_match_margin[leg];
                pair_match_score += ms;
                if (result > 0) rs->match_wins++;
                else if (result < 0) rs->match_losses++;
                else rs->match_draws++;
            }
            out->caps += (uint64_t)rows[i].capped[leg];
            out->exact_moves += (uint64_t)rows[i].exact_moves[leg];
            out->cap_forces += (uint64_t)rows[i].cap_forces[leg];
            out->cycle_forces += (uint64_t)rows[i].cycle_forces[leg];
        }
        margin_sum += pair_margin;
        margin_sumsq += pair_margin * pair_margin;
        configured_sum += pair_objective;
        configured_sumsq += pair_objective * pair_objective;
        score_sum += pair_score;
        score_sumsq += pair_score * pair_score;
        round_margin_sum[round] += pair_margin;
        round_margin_sumsq[round] += pair_margin * pair_margin;
        objective_sum[round] += pair_objective;
        objective_sumsq[round] += pair_objective * pair_objective;
        if (round == MATCH_ROUNDS - 1) {
            final_margin_sum[round] += pair_final_margin;
            final_margin_sumsq[round] +=
                pair_final_margin * pair_final_margin;
            match_score_sum[round] += pair_match_score;
            match_score_sumsq[round] += pair_match_score * pair_match_score;
        }
        if (rows[i].picked == 0) {
            out->baseline_roots++;
            if (rows[i].admitted == 1) out->singleton_roots++;
        } else out->challenger_roots++;
    }
    out->margin = margin_sum / (2.0 * pairs);
    out->objective = configured_sum / (2.0 * pairs);
    out->score = score_sum / (2.0 * pairs);
    out->candidate_points = candidate_points / (2.0 * pairs);
    out->baseline_points = baseline_points / (2.0 * pairs);
    if (pairs > 1) {
        double mv = (margin_sumsq - margin_sum * margin_sum / pairs)
                  / (pairs - 1);
        double sv = (score_sumsq - score_sum * score_sum / pairs)
                  / (pairs - 1);
        double ov =
            (configured_sumsq - configured_sum * configured_sum / pairs)
            / (pairs - 1);
        if (mv < 0.0) mv = 0.0;
        if (sv < 0.0) sv = 0.0;
        if (ov < 0.0) ov = 0.0;
        out->margin_se = sqrt(mv / pairs) / 2.0;
        out->objective_se = sqrt(ov / pairs) / 2.0;
        out->score_se = sqrt(sv / pairs) / 2.0;
    }
    for (int round = 0; round < MATCH_ROUNDS; round++) {
        RoundSummary *rs = &out->round[round];
        if (rs->pairs == 0) continue;
        double n = (double)rs->pairs;
        rs->round_margin = round_margin_sum[round] / (2.0 * n);
        rs->objective_target = objective_sum[round] / (2.0 * n);
        if (round == MATCH_ROUNDS - 1) {
            rs->final_match_margin = final_margin_sum[round] / (2.0 * n);
            rs->match_score = match_score_sum[round] / (2.0 * n);
        }
        if (rs->pairs > 1) {
            double rmv =
                (round_margin_sumsq[round]
                 - round_margin_sum[round] * round_margin_sum[round] / n)
                / (n - 1.0);
            double ov =
                (objective_sumsq[round]
                 - objective_sum[round] * objective_sum[round] / n)
                / (n - 1.0);
            if (rmv < 0.0) rmv = 0.0;
            if (ov < 0.0) ov = 0.0;
            rs->round_margin_se = sqrt(rmv / n) / 2.0;
            rs->objective_target_se = sqrt(ov / n) / 2.0;
            if (round == MATCH_ROUNDS - 1) {
                double fmv =
                    (final_margin_sumsq[round]
                     - final_margin_sum[round] *
                       final_margin_sum[round] / n)
                    / (n - 1.0);
                double msv =
                    (match_score_sumsq[round]
                     - match_score_sum[round] * match_score_sum[round] / n)
                    / (n - 1.0);
                if (fmv < 0.0) fmv = 0.0;
                if (msv < 0.0) msv = 0.0;
                rs->final_match_margin_se = sqrt(fmv / n) / 2.0;
                rs->match_score_se = sqrt(msv / n) / 2.0;
            }
        }
    }
}

static void json_string(FILE *f, const char *s)
{
    fputc('"', f);
    for (const unsigned char *p = (const unsigned char *)s; *p; p++) {
        switch (*p) {
        case '"': fputs("\\\"", f); break;
        case '\\': fputs("\\\\", f); break;
        case '\b': fputs("\\b", f); break;
        case '\f': fputs("\\f", f); break;
        case '\n': fputs("\\n", f); break;
        case '\r': fputs("\\r", f); break;
        case '\t': fputs("\\t", f); break;
        default:
            if (*p < 0x20) fprintf(f, "\\u%04x", *p);
            else fputc(*p, f);
        }
    }
    fputc('"', f);
}

static void write_summary_json(FILE *f, const Summary *summary,
                               int objective, int target_round,
                               const char *record)
{
    fputs("{\"record\":", f);
    json_string(f, record);
    fprintf(f, ",\"continuation_objective\":%d,"
               "\"configured_objective_aggregate_comparable\":%s,"
               "\"configured_objective_per_leg\":",
            objective,
            objective == 2 && target_round < 0 ? "false" : "true");
    if (objective == 2 && target_round < 0)
        fputs("null,\"configured_objective_pair_clustered_se\":null,",
              f);
    else
        fprintf(f, "%.17g,\"configured_objective_pair_clustered_se\":"
                   "%.17g,", summary->objective, summary->objective_se);
    fputs("\"rounds\":[", f);
    for (int round = 0; round < MATCH_ROUNDS; round++) {
        const RoundSummary *rs = &summary->round[round];
        if (round) fputc(',', f);
        fprintf(f,
                "{\"round\":%d,\"pairs\":%llu,"
                "\"selection_semantics\":",
                round, (unsigned long long)rs->pairs);
        json_string(f, round == MATCH_ROUNDS - 1 && objective == 2
                         ? "final_match_hybrid"
                         : "round_margin");
        fprintf(f,
                ",\"round_margin_per_leg\":%.17g,"
                "\"round_margin_pair_clustered_se\":%.17g,"
                "\"configured_objective_per_leg\":%.17g,"
                "\"configured_objective_pair_clustered_se\":%.17g",
                rs->round_margin, rs->round_margin_se,
                rs->objective_target, rs->objective_target_se);
        if (round == MATCH_ROUNDS - 1) {
            fprintf(f,
                    ",\"final_match_margin_per_leg\":%.17g,"
                    "\"final_match_margin_pair_clustered_se\":%.17g,"
                    "\"match_score\":%.17g,"
                    "\"match_score_pair_clustered_se\":%.17g,"
                    "\"match_wins\":%llu,\"match_losses\":%llu,"
                    "\"match_draws\":%llu",
                    rs->final_match_margin, rs->final_match_margin_se,
                    rs->match_score, rs->match_score_se,
                    (unsigned long long)rs->match_wins,
                    (unsigned long long)rs->match_losses,
                    (unsigned long long)rs->match_draws);
        }
        fputc('}', f);
    }
    fputs("]}", f);
}

static int write_raw(const char *path, const ContinuationPair *rows,
                     int pairs, uint64_t pair_start, uint64_t seed,
                     int target_round, int objective, int role_mapping_mode,
                     const Summary *summary, const char *root_path,
                     const char *candidate_path, const char *baseline_path,
                     const char *provenance)
{
    size_t npath = strlen(path);
    if (npath > SIZE_MAX - 64) return 0;
    char *tmp = malloc(npath + 64);
    if (!tmp) return 0;
    snprintf(tmp, npath + 64, "%s.tmp.%ld", path, (long)getpid());
    int fd = open(tmp, O_WRONLY | O_CREAT | O_EXCL, 0666);
    if (fd < 0) {
        fprintf(stderr, "%s: cannot create temporary raw file: %s\n",
                tmp, strerror(errno));
        free(tmp);
        return 0;
    }
    FILE *f = fdopen(fd, "w");
    if (!f) {
        close(fd); unlink(tmp); free(tmp);
        return 0;
    }
    fputs("{\"record\":\"meta\",\"schema\":2,"
          "\"evidence_scope\":\"candidate_screen_only_not_promotion\","
          "\"seed\":", f);
    fprintf(f, "\"%llu\",\"pair_start\":\"%llu\",\"pair_count\":%d,"
               "\"target_round\":",
            (unsigned long long)seed, (unsigned long long)pair_start, pairs);
    if (target_round < 0) fputs("\"cycle_0_1_2\"", f);
    else fprintf(f, "%d", target_round);
    fprintf(f, ",\"continuation_objective\":%d,"
               "\"round_0_1_semantics\":\"round_margin\","
               "\"round_2_mode_2_semantics\":"
               "\"0.05*final_match_margin+50*signed_match_result\","
               "\"role_mapping_mode\":",
            objective);
    json_string(f, role_mapping_name(role_mapping_mode));
    fputs(",\"root_checkpoint\":", f); json_string(f, root_path);
    fputs(",\"candidate_checkpoint\":", f); json_string(f, candidate_path);
    fputs(",\"baseline_checkpoint\":", f); json_string(f, baseline_path);
    fputs(",\"root_ply\":14,\"root_symmetries\":20,"
          "\"root_width\":5,\"root_floor\":0.02,\"root_min\":1,"
          "\"root_mix\":\"alternating_absolute_index_baseline_nonbaseline_with_singleton_fallback\","
          "\"world_model\":\"uniform_mover_information_set\","
          "\"continuation_policy\":\"greedy_fixed_player_mapping_affine20\","
          "\"late_cycle\":\"production_semantic_information_tracker_deck_le_3\","
          "\"pairing\":\"identical_post_root_world_controller_seat_swap\","
          "\"provenance\":", f);
    json_string(f, provenance);
    fputs("}\n", f);

    for (int i = 0; i < pairs; i++) {
        const ContinuationPair *r = &rows[i];
        fprintf(f,
            "{\"record\":\"pair\",\"index\":\"%llu\","
            "\"round\":%d,\"root_player\":%d,\"admitted\":%d,"
            "\"picked\":%d,\"root_move\":%u,"
            "\"cum_before\":[%d,%d],"
            "\"cumulative_before\":[%d,%d],"
            "\"player_mapping\":[%d,%d],"
            "\"root_role_mapping\":%d,"
            "\"opponent_role_mapping\":%d,"
            "\"candidate_seat\":[%d,%d],"
            "\"score_by_seat\":[[%d,%d],[%d,%d]],"
            "\"candidate_margin\":[%d,%d],"
            "\"candidate_round_margin\":[%d,%d],"
            "\"candidate_objective_target\":[%.17g,%.17g],",
            (unsigned long long)r->index, r->round, r->root_player,
            r->admitted, r->picked, (unsigned)r->root_move,
            r->cumulative_before[0], r->cumulative_before[1],
            r->cumulative_before[0], r->cumulative_before[1],
            r->player_mapping[0], r->player_mapping[1],
            r->player_mapping[r->root_player],
            r->player_mapping[r->root_player ^ 1],
            r->candidate_seat[0], r->candidate_seat[1],
            r->score[0][0], r->score[0][1],
            r->score[1][0], r->score[1][1],
            r->margin[0], r->margin[1],
            r->margin[0], r->margin[1],
            r->objective_target[0], r->objective_target[1]);
        if (r->round == MATCH_ROUNDS - 1) {
            fprintf(f,
                    "\"candidate_final_match_margin\":[%d,%d],"
                    "\"candidate_final_match_result\":[%d,%d],"
                    "\"candidate_hybrid_target\":[%.17g,%.17g],",
                    r->final_match_margin[0], r->final_match_margin[1],
                    r->final_match_result[0], r->final_match_result[1],
                    r->hybrid_target[0], r->hybrid_target[1]);
        } else {
            fputs("\"candidate_final_match_margin\":null,"
                  "\"candidate_final_match_result\":null,"
                  "\"candidate_hybrid_target\":null,", f);
        }
        fprintf(f,
            "\"tail_plies\":[%d,%d],"
            "\"capped\":[%d,%d],\"exact_moves\":[%d,%d],"
            "\"cap_forces\":[%d,%d],\"cycle_forces\":[%d,%d]}\n",
            r->tail_plies[0], r->tail_plies[1],
            r->capped[0], r->capped[1],
            r->exact_moves[0], r->exact_moves[1],
            r->cap_forces[0], r->cap_forces[1],
            r->cycle_forces[0], r->cycle_forces[1]);
    }
    write_summary_json(f, summary, objective, target_round, "summary");
    fputc('\n', f);
    fprintf(f, "{\"record\":\"complete\",\"pairs\":%d}\n", pairs);
    int failed = ferror(f) || fflush(f) != 0 || fsync(fd) != 0;
    if (fclose(f) != 0) failed = 1;
    if (!failed && link(tmp, path) != 0) {
        fprintf(stderr, "%s: refusing to replace raw result: %s\n",
                path, strerror(errno));
        failed = 1;
    }
    unlink(tmp);
    free(tmp);
    return !failed;
}

static void usage(const char *argv0)
{
    fprintf(stderr,
        "usage: %s -a CANDIDATE [-r ROOT] [-b BASELINE] [-n pairs] "
        "[-t threads] [-s seed]\n"
        "          [--pair-start N] [--target-round 0|1|2] [-q]\n"
        "          [--continuation-objective 0|2]\n"
        "          [--continuation-role-mappings legacy|shared|independent]\n"
        "          [--raw-pairs FILE --provenance ID [--raw-only]]\n"
        "  Fast continuation-role candidate screen only; never promotion "
        "evidence.\n"
        "  ROOT and BASELINE default to data/champion.bin.  Without "
        "--target-round,\n"
        "  absolute pair indices cycle through genuine round-0/1/2 match "
        "contexts.  Objective 2 preserves round margin in rounds 0/1 and\n"
        "  uses the final-match hybrid only in round 2.  Independent role\n"
        "  mappings follow deployed mode-4 ordered-product scheduling.\n",
        argv0);
}

int main(int argc, char **argv)
{
    const char *root_path = "data/champion.bin";
    const char *baseline_path = "data/champion.bin";
    const char *candidate_path = NULL;
    const char *raw_path = NULL;
    const char *provenance = NULL;
    int pairs = 500, nthread = 4, target_round = -1;
    int objective = 0, role_mapping_mode = ROLE_MAPPING_LEGACY;
    int quiet = 0, raw_only = 0;
    uint64_t seed = UINT64_C(20260822), pair_start = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-a") && i + 1 < argc)
            candidate_path = argv[++i];
        else if (!strcmp(argv[i], "-r") && i + 1 < argc)
            root_path = argv[++i];
        else if (!strcmp(argv[i], "-b") && i + 1 < argc)
            baseline_path = argv[++i];
        else if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            if (!parse_int_range(argv[++i], 1, MATCH_MAX_PAIRS, &pairs)) {
                fprintf(stderr, "invalid pair count\n"); return 1;
            }
        } else if (!strcmp(argv[i], "-t") && i + 1 < argc) {
            if (!parse_int_range(argv[++i], 1, 1024, &nthread)) {
                fprintf(stderr, "invalid thread count\n"); return 1;
            }
        } else if (!strcmp(argv[i], "-s") && i + 1 < argc) {
            if (!parse_u64(argv[++i], &seed)) {
                fprintf(stderr, "invalid seed\n"); return 1;
            }
        } else if (!strcmp(argv[i], "--pair-start") && i + 1 < argc) {
            if (!parse_u64(argv[++i], &pair_start)) {
                fprintf(stderr, "invalid pair start\n"); return 1;
            }
        } else if (!strcmp(argv[i], "--target-round") && i + 1 < argc) {
            if (!parse_int_range(argv[++i], 0, MATCH_ROUNDS - 1,
                                 &target_round)) {
                fprintf(stderr, "invalid target round\n"); return 1;
            }
        } else if (!strcmp(argv[i], "--continuation-objective")) {
            if (++i >= argc ||
                (strcmp(argv[i], "0") != 0 && strcmp(argv[i], "2") != 0)) {
                fprintf(stderr,
                        "--continuation-objective must be exactly 0 or 2\n");
                return 1;
            }
            objective = argv[i][0] - '0';
        } else if (!strcmp(argv[i], "--continuation-role-mappings")) {
            if (++i >= argc) {
                fprintf(stderr,
                        "--continuation-role-mappings must be exactly "
                        "legacy, shared, or independent\n");
                return 1;
            }
            if (!strcmp(argv[i], "legacy"))
                role_mapping_mode = ROLE_MAPPING_LEGACY;
            else if (!strcmp(argv[i], "shared"))
                role_mapping_mode = ROLE_MAPPING_SHARED;
            else if (!strcmp(argv[i], "independent"))
                role_mapping_mode = ROLE_MAPPING_INDEPENDENT;
            else {
                fprintf(stderr,
                        "--continuation-role-mappings must be exactly "
                        "legacy, shared, or independent\n");
                return 1;
            }
        } else if (!strcmp(argv[i], "--raw-pairs") && i + 1 < argc)
            raw_path = argv[++i];
        else if (!strcmp(argv[i], "--provenance") && i + 1 < argc)
            provenance = argv[++i];
        else if (!strcmp(argv[i], "--raw-only")) raw_only = 1;
        else if (!strcmp(argv[i], "-q")) quiet = 1;
        else { usage(argv[0]); return 1; }
    }
    if (!candidate_path) { usage(argv[0]); return 1; }
    if (nthread > pairs) nthread = pairs;
    if (pair_start > UINT64_MAX - ((uint64_t)pairs - UINT64_C(1))) {
        fprintf(stderr, "pair range overflows\n"); return 1;
    }
    if (raw_only && !raw_path) {
        fprintf(stderr, "--raw-only requires --raw-pairs\n"); return 1;
    }
    if (raw_path && (!*raw_path || !provenance || !*provenance)) {
        fprintf(stderr, "--raw-pairs requires a nonempty path and provenance\n");
        return 1;
    }

    Net *root = malloc(sizeof(*root));
    Net *candidate = malloc(sizeof(*candidate));
    Net *baseline = malloc(sizeof(*baseline));
    if (!root || !candidate || !baseline) {
        fprintf(stderr, "cannot allocate checkpoints\n");
        free(root); free(candidate); free(baseline);
        return 1;
    }
    if (net_load(root, root_path) != 0 ||
        net_load(candidate, candidate_path) != 0 ||
        net_load(baseline, baseline_path) != 0) {
        fprintf(stderr, "cannot load root/candidate/baseline checkpoint\n");
        free(root); free(candidate); free(baseline);
        return 1;
    }
    NetEvalPlan root_plan, candidate_plan, baseline_plan;
    net_eval_plan_init(root, &root_plan);
    net_eval_plan_init(candidate, &candidate_plan);
    net_eval_plan_init(baseline, &baseline_plan);

    ContinuationPair *rows = calloc((size_t)pairs, sizeof(*rows));
    Job *jobs = calloc((size_t)nthread, sizeof(*jobs));
    pthread_t *threads = calloc((size_t)nthread, sizeof(*threads));
    if (!rows || !jobs || !threads) {
        fprintf(stderr, "cannot allocate evaluator work\n");
        free(rows); free(jobs); free(threads);
        free(root); free(candidate); free(baseline);
        return 1;
    }
    atomic_int failed;
    atomic_init(&failed, 0);
    int created = 0;
    for (int t = 0; t < nthread; t++) {
        jobs[t] = (Job){
            .root = root, .candidate = candidate, .baseline = baseline,
            .root_plan = &root_plan,
            .candidate_plan = &candidate_plan,
            .baseline_plan = &baseline_plan,
            .rows = rows, .pair_start = pair_start, .seed = seed,
            .pairs = pairs, .target_round = target_round,
            .objective = objective,
            .role_mapping_mode = role_mapping_mode,
            .thread = t, .nthread = nthread, .failed = &failed
        };
        if (pthread_create(&threads[t], NULL, worker, &jobs[t]) != 0) {
            atomic_store_explicit(&failed, 1, memory_order_relaxed);
            break;
        }
        created++;
    }
    for (int t = 0; t < created; t++)
        if (pthread_join(threads[t], NULL) != 0)
            atomic_store_explicit(&failed, 1, memory_order_relaxed);
    if (created != nthread || atomic_load_explicit(&failed,
                                                   memory_order_relaxed)) {
        fprintf(stderr, "continuation screen failed; no result published\n");
        free(rows); free(jobs); free(threads);
        free(root); free(candidate); free(baseline);
        return 1;
    }

    Summary summary;
    summarize(rows, pairs, &summary);
    if (raw_path && !write_raw(
            raw_path, rows, pairs, pair_start, seed, target_round,
            objective, role_mapping_mode, &summary,
            root_path, candidate_path, baseline_path, provenance)) {
        free(rows); free(jobs); free(threads);
        free(root); free(candidate); free(baseline);
        return 1;
    }

    if (!raw_only) {
        if (quiet) {
            printf("%.6f %.6f %.6f %.6f %llu %llu %llu %llu\n",
                   summary.margin, summary.margin_se,
                   summary.score, summary.score_se,
                   (unsigned long long)summary.wins,
                   (unsigned long long)summary.losses,
                   (unsigned long long)summary.draws,
                   (unsigned long long)summary.caps);
        } else {
            puts("CONTINUATION CANDIDATE SCREEN -- NOT PROMOTION EVIDENCE");
            printf("candidate %s\nbaseline  %s\nroot      %s\n",
                   candidate_path, baseline_path, root_path);
            printf("  %d paired tails (%d legs), post-root controller swap\n",
                   pairs, 2 * pairs);
            printf("  objective mode %d; role mappings %s\n",
                   objective, role_mapping_name(role_mapping_mode));
            printf("  round margin/leg %+.3f  (pair-clustered SE %.3f)\n",
                   summary.margin, summary.margin_se);
            if (objective == 2 && target_round < 0)
                puts("  configured objective aggregate omitted: round "
                     "margin and final-match hybrid are different units");
            else
                printf("  configured objective/leg %+.3f "
                       "(pair-clustered SE %.3f)\n",
                       summary.objective, summary.objective_se);
            if (objective == 0) {
                printf("  W/L/D %llu/%llu/%llu   round score %.2f%% "
                       "(pair-clustered SE %.2f%%)\n",
                       (unsigned long long)summary.wins,
                       (unsigned long long)summary.losses,
                       (unsigned long long)summary.draws,
                       100.0 * summary.score, 100.0 * summary.score_se);
            } else {
                for (int round = 0; round < MATCH_ROUNDS; round++) {
                    const RoundSummary *rs = &summary.round[round];
                    if (rs->pairs == 0) continue;
                    if (round < MATCH_ROUNDS - 1) {
                        printf("  round %d: %llu pairs, round margin/leg "
                               "%+.3f (SE %.3f)\n",
                               round, (unsigned long long)rs->pairs,
                               rs->round_margin, rs->round_margin_se);
                    } else {
                        printf("  round 2: %llu pairs, hybrid target/leg "
                               "%+.3f (SE %.3f), final match margin/leg "
                               "%+.3f (SE %.3f)\n",
                               (unsigned long long)rs->pairs,
                               rs->objective_target,
                               rs->objective_target_se,
                               rs->final_match_margin,
                               rs->final_match_margin_se);
                        printf("           match W/L/D %llu/%llu/%llu, "
                               "score %.2f%% (SE %.2f%%)\n",
                               (unsigned long long)rs->match_wins,
                               (unsigned long long)rs->match_losses,
                               (unsigned long long)rs->match_draws,
                               100.0 * rs->match_score,
                               100.0 * rs->match_score_se);
                    }
                }
            }
            printf("  round points/leg %.2f vs %.2f\n",
                   summary.candidate_points, summary.baseline_points);
            printf("  roots baseline/challenger/singleton %llu/%llu/%llu\n",
                   (unsigned long long)summary.baseline_roots,
                   (unsigned long long)summary.challenger_roots,
                   (unsigned long long)summary.singleton_roots);
            printf("  exact deck-one moves %llu; cap-reserve forces %llu; "
                   "late-cycle forces %llu; cap-terminated tails %llu\n",
                   (unsigned long long)summary.exact_moves,
                   (unsigned long long)summary.cap_forces,
                   (unsigned long long)summary.cycle_forces,
                   (unsigned long long)summary.caps);
        }
    }

    free(rows); free(jobs); free(threads);
    free(root); free(candidate); free(baseline);
    return 0;
}
