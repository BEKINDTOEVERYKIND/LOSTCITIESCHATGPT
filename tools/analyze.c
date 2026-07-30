/* analyze -- play one self-play match and dump a per-ply JSON analysis.
 *
 * A match is -r ROUNDS rounds (default MATCH_ROUNDS): fresh deal per round,
 * st.round / st.cum carrying the match context, and the first player
 * alternating by round, exactly like the reference loop in src/match.c.
 *
 * For every ply, before the chosen move is applied, the dump records the full
 * public state plus the mover's hand(s), the round and cumulative totals, the
 * publicly known cards in each hand, the value head from both perspectives,
 * the policy distribution over legal moves, and the rollout search
 * statistics.  The actor and evaluator have independent RNG streams:
 * rollout_move is post-hoc only and its return value never changes the match.
 *
 * Output is a single JSON object on stdout; redirect to a file.
 */
#define _POSIX_C_SOURCE 200809L /* open_memstream under -std=c11 */
#include "../src/lc.h"
#include "../src/agent.h"
#include "../src/search.h"
#include <math.h>
#include "../src/spec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char SUIT_CH[NSUIT + 1] = "YBWGR";

/* ---- tiny JSON helpers: fp is either stdout or a memstream -------------- */

static void j_string(FILE *fp, const char *s)
{
    fputc('"', fp);
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        switch (c) {
        case '"':  fputs("\\\"", fp); break;
        case '\\': fputs("\\\\", fp); break;
        case '\b': fputs("\\b", fp); break;
        case '\f': fputs("\\f", fp); break;
        case '\n': fputs("\\n", fp); break;
        case '\r': fputs("\\r", fp); break;
        case '\t': fputs("\\t", fp); break;
        default:
            if (c < 0x20) fprintf(fp, "\\u%04x", c);
            else fputc(c, fp);
        }
    }
    fputc('"', fp);
}

static void j_card(FILE *fp, int c)
{
    char b[8];
    lc_card_name(c, b);
    j_string(fp, b);
}

/* ["Y7","Bx",...] */
static void j_card_arr(FILE *fp, const uint8_t *cards, int n)
{
    fputc('[', fp);
    for (int i = 0; i < n; i++) {
        if (i) fputc(',', fp);
        j_card(fp, cards[i]);
    }
    fputc(']', fp);
}

static const char *act_str(Move m) { return m.discard ? "discard" : "play"; }

static void draw_str(Move m, char *b)
{
    if (m.draw == 0) strcpy(b, "deck");
    else { b[0] = SUIT_CH[m.draw - 1]; b[1] = 0; }
}

/* hand sorted by card id */
static void j_hand(FILE *fp, const State *st, int p)
{
    uint8_t cards[HAND_SIZE];
    int n = lc_hand_cards(st, p, cards);
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (cards[j] < cards[i]) { uint8_t t = cards[i]; cards[i] = cards[j]; cards[j] = t; }
    j_card_arr(fp, cards, n);
}

/* [[cards p0 is publicly known to hold],[same for p1]], sorted by card id
 * (bit-order iteration of st->known is ascending id order already) */
static void j_known(FILE *fp, const State *st)
{
    fputc('[', fp);
    for (int p = 0; p < 2; p++) {
        if (p) fputc(',', fp);
        uint8_t cards[HAND_SIZE];
        int n = 0;
        uint64_t k = st->known[p];
        while (k) { cards[n++] = (uint8_t)__builtin_ctzll(k); k &= k - 1; }
        j_card_arr(fp, cards, n);
    }
    fputc(']', fp);
}

/* one player's five expeditions in play order (card-id order per suit:
 * wagers first, then ascending numbers -- which is the only legal order) */
static void j_exps(FILE *fp, const State *st, int p)
{
    fputc('[', fp);
    for (int s = 0; s < NSUIT; s++) {
        if (s) fputc(',', fp);
        uint8_t cards[NRANK];
        int n = 0;
        for (int c = s * NRANK; c < (s + 1) * NRANK; c++)
            if ((st->played[p] >> c) & 1ULL) cards[n++] = (uint8_t)c;
        j_card_arr(fp, cards, n);
    }
    fputc(']', fp);
}

/* the five discard piles bottom-to-top */
static void j_piles(FILE *fp, const State *st)
{
    fputc('[', fp);
    for (int s = 0; s < NSUIT; s++) {
        if (s) fputc(',', fp);
        j_card_arr(fp, st->pile[s], st->pile_n[s]);
    }
    fputc(']', fp);
}

/* {"card":"R2","act":"play","draw":"deck"  -- shared prefix of policy/search
 * entries and the move object; the caller closes the brace */
static void j_move_open(FILE *fp, Move m)
{
    char d[8];
    draw_str(m, d);
    fprintf(fp, "{\"card\":");
    j_card(fp, m.card);
    fprintf(fp, ",\"act\":\"%s\",\"draw\":\"%s\"", act_str(m), d);
}

static int move_eq(Move a, Move b)
{
    return a.card == b.card && a.discard == b.discard && a.draw == b.draw;
}

static int search_guard_blocked(const SearchStats *ss)
{
    for (int i = 0; i < ss->n; i++)
        if (ss->guard_rejected[i]) return 1;
    return 0;
}

static const char *search_status(const SearchStats *ss, Move recommended)
{
    if (ss->worlds == 0) {
        if (ss->skip_reason == SEARCH_SKIP_FORCED)
            return "forced_move";
        if (ss->skip_reason == SEARCH_SKIP_PLY_WINDOW)
            return "skipped_ply_window";
        if (ss->skip_reason == SEARCH_SKIP_DECK_PHASE)
            return "skipped_deck_phase";
        if (ss->skip_reason == SEARCH_SKIP_ROOT_FOCUS)
            return "reduced_by_root_focus";
        if (ss->skip_reason == SEARCH_SKIP_VISIBLE_PLAN)
            return "selected_by_visible_plan";
        if (ss->skip_reason == SEARCH_SKIP_LAST_DECK)
            return "selected_by_last_deck_rule";
        return "skipped_policy_confidence";
    }
    if (ss->confirmed && ss->n > 0 &&
        !move_eq(ss->mv[0], recommended))
        return "supported_baseline_override";
    if (search_guard_blocked(ss))
        return "blocked_by_discard_guard";
    if (ss->confirm_worlds > 0)
        return "failed_stochastic_confirmation";
    if (!ss->resolved)
        return "inconclusive";
    if (ss->raw_best >= 0 && ss->raw_best < ss->n &&
        move_eq(ss->mv[ss->raw_best], recommended))
        return "resolved";
    return "resolved_below_action_threshold";
}

static const char *search_metric(const SearchStats *ss)
{
    switch (ss->metric_kind) {
    case SEARCH_METRIC_NETWORK_VALUE: return "network_state_value";
    case SEARCH_METRIC_VISIBLE_PLAN: return "visible_hand_guarantee";
    case SEARCH_METRIC_LAST_DECK_RULE: return "last_deck_rule";
    default: return "rollout_objective";
    }
}

static void j_search_rows(FILE *fp, const SearchStats *ss, Move played,
                          Move recommended)
{
    int order[MAX_MOVES];
    for (int i = 0; i < ss->n; i++) order[i] = i;
    for (int i = 0; i < ss->n; i++)
        for (int j = i + 1; j < ss->n; j++)
            if (ss->q[order[j]] > ss->q[order[i]]) {
                int t = order[i]; order[i] = order[j]; order[j] = t;
            }

    fputc('[', fp);
    for (int i = 0; i < ss->n; i++) {
        if (i) fputc(',', fp);
        int k = order[i];
        j_move_open(fp, ss->mv[k]);
        fprintf(fp, ",\"q\":%.3f,\"q_se\":%.3f,"
                    "\"delta_vs_baseline\":%.3f,\"delta_se\":%.3f,"
                    "\"policy_prob\":%.6f,\"visits\":%.0f,"
                    "\"played\":%s,\"baseline\":%s,"
                    "\"policy_top\":%s,\"retained\":%s,"
                    "\"highest_mean\":%s,\"confirmed_best\":%s,"
                    "\"primary_pass\":%s,"
                    "\"confirmation_delta\":%.3f,"
                    "\"confirmation_se\":%.3f,"
                    "\"confirmation_pass\":%s,"
                    "\"guard_rejected\":%s",
                ss->q[k], ss->se[k], ss->delta[k], ss->dse[k],
                ss->prior[k], ss->visits[k],
                move_eq(ss->mv[k], played) ? "true" : "false",
                k == 0 ? "true" : "false",
                k == ss->policy_top ? "true" : "false",
                move_eq(ss->mv[k], recommended) ? "true" : "false",
                k == ss->raw_best ? "true" : "false",
                ss->csupported[k] && move_eq(ss->mv[k], recommended)
                    ? "true" : "false",
                ss->pqualified[k] ? "true" : "false",
                ss->cdelta[k], ss->cdse[k],
                ss->csupported[k] ? "true" : "false",
                ss->guard_rejected[k] ? "true" : "false");
        if (ss->qw[k] >= 0.0) fprintf(fp, ",\"qw\":%.3f", ss->qw[k]);
        fputc('}', fp);
    }
    fputc(']', fp);
}

static uint64_t mix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

int main(int argc, char **argv)
{
    const char *actor_spec = LC_CHAMPION_AGENT_SPEC;
    const char *eval_spec =
        "rolloutu:data/champion.bin:2048:4:0.01:0.995:2:20:0:0:2:0:3.5:1:2:20:0.995:0:20:1:0:2048:1:16:12:1";
    uint64_t seed = 1;
    int rounds = MATCH_ROUNDS;
    float belief_alpha = 1.15f;
    int belief_symmetries = 20;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-a") && i + 1 < argc) actor_spec = argv[++i];
        else if (!strcmp(argv[i], "-e") && i + 1 < argc) eval_spec = argv[++i];
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "-r") && i + 1 < argc) rounds = atoi(argv[++i]);
        else if (!strcmp(argv[i], "--belief-alpha") && i + 1 < argc)
            belief_alpha = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--belief-symmetries") && i + 1 < argc)
            belief_symmetries = atoi(argv[++i]);
        else {
            fprintf(stderr, "usage: %s [-a ACTOR] [-e EVALUATOR] [-s seed] "
                            "[-r rounds] [--belief-alpha A] "
                            "[--belief-symmetries N]\n", argv[0]);
            return 1;
        }
    }
    if (rounds < 1) rounds = 1;
    if (rounds > MATCH_ROUNDS) rounds = MATCH_ROUNDS;

    Agent actor, evaluator;
    spec_parse(actor_spec, &actor);
    spec_parse(eval_spec, &evaluator);
    if (!actor.net ||
        (actor.kind != AG_POLICY && actor.kind != AG_ROLLOUT)) {
        fprintf(stderr, "analyze: actor must be a network policy or rollout "
                        "spec (got '%s')\n", actor_spec);
        return 1;
    }
    if (evaluator.kind != AG_ROLLOUT || !evaluator.net) {
        fprintf(stderr, "analyze: evaluator must be a network rollout spec\n");
        return 1;
    }

    Rng deal_rng, actor_rng;
    rng_seed(&deal_rng, seed);
    rng_seed(&actor_rng, mix64(seed ^ 0xA17C0AULL));

    /* the ply array is streamed into memory while the match is played, because
     * meta (which needs the final scores) comes first in the output */
    char *plybuf = NULL;
    size_t plylen = 0;
    FILE *pf = open_memstream(&plybuf, &plylen);
    if (!pf) { fprintf(stderr, "analyze: open_memstream failed\n"); return 1; }

    char start_hands[2][256];
    int cum[2] = { 0, 0 };
    int round_scores[MATCH_ROUNDS][2];
    int ply = 0;

    for (int rd = 0; rd < rounds; rd++) {
    State st;
    lc_deal(&st, &deal_rng);
    st.round = (uint8_t)rd;
    st.cum[0] = (int16_t)cum[0];
    st.cum[1] = (int16_t)cum[1];
    st.turn = (uint8_t)(rd & 1);

    if (rd == 0) {
        for (int p = 0; p < 2; p++) {
            char *hb = NULL;
            size_t hl = 0;
            FILE *hf = open_memstream(&hb, &hl);
            j_hand(hf, &st, p);
            fclose(hf);
            snprintf(start_hands[p], sizeof start_hands[p], "%s", hb);
            free(hb);
        }
    }

    while (!st.over) {
        int p = st.turn;
        ply++;
        if (ply > 1) fputc(',', pf);
        fprintf(pf, "{\"n\":%d,\"round_ply\":%u,\"player\":%d,"
                    "\"round\":%d,\"cum\":[%d,%d],\"deck_left\":%d,",
                ply, (unsigned)st.nply, p, rd, cum[0], cum[1], st.deck_left);

        fprintf(pf, "\"known\":");
        j_known(pf, &st);
        fprintf(pf, ",\"hands\":[");
        j_hand(pf, &st, 0);
        fputc(',', pf);
        j_hand(pf, &st, 1);
        fprintf(pf, "],\"exps\":[");
        j_exps(pf, &st, 0);
        fputc(',', pf);
        j_exps(pf, &st, 1);
        fprintf(pf, "],\"piles\":");
        j_piles(pf, &st);

        /* value head from each perspective, in points */
        Features feat;
        float v[2];
        for (int q = 0; q < 2; q++) {
            feat_extract(&st, q, &feat);
            v[q] = net_value(actor.net, &feat) * VAL_SCALE;
        }
        fprintf(pf, ",\"values\":[%.1f,%.1f]", v[0], v[1]);

        /* Coherent fixed-cardinality belief diagnostic.  These are analytic
         * inclusion marginals of one joint K-card distribution, so they sum
         * exactly to the number of unknown opponent cards.  The production
         * audit uses uniform worlds until this learned ranking clears locked
         * validation; the diagnostic remains visible and honestly labelled. */
        {
            int mp = st.turn, mo = mp ^ 1;
            BeliefDist bd;
            float effective_alpha = st.nply == 0 ? 0.0f : belief_alpha;
            if (!belief_dist_init(actor.net, &st, mp, belief_symmetries,
                                  effective_alpha, &bd)) {
                fprintf(stderr, "analyze: belief distribution failed at ply %d\n", ply);
                return 1;
            }
            int bord[NCARD];
            for (int i = 0; i < bd.n; i++) bord[i] = i;
            for (int i = 0; i < bd.n; i++)
                for (int j2 = i + 1; j2 < bd.n; j2++)
                    if (bd.marginal[bord[j2]] > bd.marginal[bord[i]]) {
                        int t = bord[i]; bord[i] = bord[j2]; bord[j2] = t;
                    }
            float prior = bd.n > 0 ? (float)bd.need / (float)bd.n : 0.0f;
            double msum = 0.0;
            for (int i = 0; i < bd.n; i++) msum += bd.marginal[i];
            fprintf(pf, ",\"belief\":{\"persp\":%d,\"unknown_hand\":%d,"
                        "\"unknown_pool\":%d,\"prior\":%.6f,"
                        "\"method\":\"fixed_cardinality\","
                        "\"alpha\":%.3f,\"symmetries\":%d,"
                        "\"used_by_rollout\":false,\"marginal_sum\":%.6f,"
                        "\"cards\":[",
                    mp, bd.need, bd.n, prior, effective_alpha,
                    belief_symmetries, msum);
            int bkeep = bd.n < 14 ? bd.n : 14;
            for (int i = 0; i < bkeep; i++) {
                int ci = bord[i];
                if (i) fputc(',', pf);
                fprintf(pf, "{\"card\":");
                j_card(pf, bd.card[ci]);
                fprintf(pf, ",\"p\":%.3f,\"held\":%s}",
                        bd.marginal[ci],
                        ((st.hand[mo] >> bd.card[ci]) & 1ULL) ? "true" : "false");
            }
            fprintf(pf, "],\"all_cards\":[");
            for (int i = 0; i < bd.n; i++) {
                int ci = bord[i];
                if (i) fputc(',', pf);
                fprintf(pf, "{\"card\":");
                j_card(pf, bd.card[ci]);
                fprintf(pf, ",\"p\":%.6f,\"held\":%s}",
                        bd.marginal[ci], ((st.hand[mo] >> bd.card[ci]) & 1ULL)
                                ? "true" : "false");
            }
            fprintf(pf, "]}");
        }

        /* policy head over all legal moves, best first, capped at 10 */
        Move pmv[MAX_MOVES];
        float prob[MAX_MOVES], pv;
        int nleg = policy_probs_sym(actor.net, &st, pmv, prob, &pv,
                                    actor.symmetries);
        int ord[MAX_MOVES];
        for (int i = 0; i < nleg; i++) ord[i] = i;
        for (int i = 0; i < nleg; i++)
            for (int j = i + 1; j < nleg; j++)
                if (prob[ord[j]] > prob[ord[i]]) { int t = ord[i]; ord[i] = ord[j]; ord[j] = t; }
        fprintf(pf, ",\"nlegal\":%d,\"policy\":[", nleg);
        int keep = nleg < 10 ? nleg : 10;
        for (int i = 0; i < keep; i++) {
            if (i) fputc(',', pf);
            j_move_open(pf, pmv[ord[i]]);
            fprintf(pf, ",\"prob\":%.3f}", prob[ord[i]]);
        }
        fputc(']', pf);

        /* The actor chooses first.  Preserve its own rollout statistics when
         * search is part of the playing agent; those explain the real move and
         * are distinct from the deeper, stateless post-hoc audit below. */
        SearchStats actor_ss;
        memset(&actor_ss, 0, sizeof actor_ss);
        Move played;
        if (actor.kind == AG_ROLLOUT)
            played = rollout_move(&actor, &st, &actor_rng, NULL, &actor_ss);
        else
            played = agent_move(&actor, &st, &actor_rng);

        /* A stateless evaluator seed makes the post-hoc audit reproducible
         * without consuming deal/actor randomness. */
        SearchStats ss;
        float sval = 0.0f;
        Rng eval_rng;
        rng_seed(&eval_rng, mix64(seed ^ 0xE7A100ULL
                                 ^ ((uint64_t)rd << 48)
                                 ^ (uint64_t)ply));
        Move recommended = rollout_move(&evaluator, &st, &eval_rng, &sval, &ss);
        const char *analysis_status = search_status(&ss, recommended);

        fprintf(pf, ",\"search\":");
        j_search_rows(pf, &ss, played, recommended);

        Move actor_policy_move = pmv[ord[0]];
        Move actor_baseline_move =
            actor.kind == AG_ROLLOUT && actor_ss.n > 0
                ? actor_ss.mv[0] : actor_policy_move;
        fprintf(pf, ",\"actor_decision\":{\"method\":\"%s\","
                    "\"used_to_choose\":%s,\"searched\":%s,"
                    "\"status\":\"%s\",\"worlds\":%d,\"max_worlds\":%d,"
                    "\"overrode_policy\":%s,\"root_width\":%d,"
                    "\"policy_mass\":%.6f,\"target_policy_mass\":%.6f,"
                    "\"policy_floor\":%.6f,\"min_candidates\":%d,"
                    "\"continuation_policy\":\"%s\","
                    "\"continuation_symmetries\":%d,"
                    "\"evaluation_kind\":\"%s\","
                    "\"baseline_source\":\"%s\","
                    "\"planner\":{\"deck_max\":%d,\"block_gap\":%d,"
                    "\"turns\":%d,\"guaranteed_score\":%d,"
                    "\"policy_score\":%d,\"policy_regret\":%d,"
                    "\"policy_block_cost\":%d,\"selected_block_cost\":%d},"
                    "\"semantic_candidates\":%d,"
                    "\"confirmation\":{\"required\":%s,\"passed\":%s,"
                    "\"worlds\":%d,\"configured_worlds\":%d},"
                    "\"policy_move\":",
                actor.kind == AG_ROLLOUT
                    ? "late_round_rollout" : "policy_argmax",
                actor.kind == AG_ROLLOUT ? "true" : "false",
                actor.kind == AG_ROLLOUT && actor_ss.worlds > 0
                    ? "true" : "false",
                actor.kind == AG_ROLLOUT
                    ? search_status(&actor_ss, played) : "policy_argmax",
                actor.kind == AG_ROLLOUT ? actor_ss.worlds : 0,
                actor.kind == AG_ROLLOUT ? actor_ss.max_worlds : 0,
                !move_eq(actor_policy_move, played) ? "true" : "false",
                actor.kind == AG_ROLLOUT ? actor.root_width : 1,
                actor.kind == AG_ROLLOUT ? actor_ss.policy_mass
                                         : prob[ord[0]],
                actor.kind == AG_ROLLOUT ? actor.cand_mass : 1.0f,
                actor.kind == AG_ROLLOUT ? actor.cand_floor : 0.0f,
                actor.kind == AG_ROLLOUT ? actor.min_cand : 1,
                actor.kind == AG_ROLLOUT
                    ? (actor.playout_sample == 1
                        ? "sampled_policy"
                        : (actor.playout_sample == 2
                            ? "random_symmetry_argmax"
                            : "exact_ensemble_argmax"))
                    : "none",
                actor.kind == AG_ROLLOUT ? actor.playout_symmetries : 0,
                actor.kind == AG_ROLLOUT
                    ? search_metric(&actor_ss) : "network_state_value",
                actor.kind == AG_ROLLOUT && actor_ss.planned_baseline
                    ? "visible_hand_scheduler"
                    : (actor.kind == AG_ROLLOUT &&
                       actor_ss.deck_end_baseline
                        ? "last_deck_dominance" : "network_policy"),
                actor.kind == AG_ROLLOUT ? actor.plan_deck_max : 0,
                actor.kind == AG_ROLLOUT ? actor.plan_block_gap : 0,
                actor.kind == AG_ROLLOUT ? actor_ss.planner_turns : 0,
                actor.kind == AG_ROLLOUT ? actor_ss.planner_score : 0,
                actor.kind == AG_ROLLOUT ? actor_ss.planner_policy_score : 0,
                actor.kind == AG_ROLLOUT ? actor_ss.planner_regret : 0,
                actor.kind == AG_ROLLOUT ? actor_ss.planner_policy_block : 0,
                actor.kind == AG_ROLLOUT ? actor_ss.planner_selected_block : 0,
                actor.kind == AG_ROLLOUT
                    ? actor_ss.semantic_candidates : 0,
                actor.kind == AG_ROLLOUT && actor.override_k > 0.0f
                    ? "true" : "false",
                actor.kind == AG_ROLLOUT && actor_ss.confirmed
                    ? "true" : "false",
                actor.kind == AG_ROLLOUT ? actor_ss.confirm_worlds : 0,
                actor.kind == AG_ROLLOUT ? actor.confirm_dets : 0);
        j_move_open(pf, actor_policy_move);
        fprintf(pf, "},\"baseline_move\":");
        j_move_open(pf, actor_baseline_move);
        fprintf(pf, "},\"selected\":");
        j_move_open(pf, played);
        fprintf(pf, "},\"candidates\":");
        if (actor.kind == AG_ROLLOUT)
            j_search_rows(pf, &actor_ss, played, played);
        else
            fputs("[]", pf);
        fputc('}', pf);

        fprintf(pf, ",\"actor_value\":%.3f,"
                    "\"analysis\":{\"kind\":\"posthoc_rollout\","
                    "\"objective\":\"%s\",\"worlds\":%d,\"max_worlds\":%d,"
                    "\"used_to_choose\":false,\"searched\":%s,"
                    "\"status\":\"%s\",\"resolved\":%s,"
                    "\"confidence_guard_se\":%.3f,"
                    "\"practical_threshold\":%.3f,\"policy_mass\":%.6f,"
                    "\"target_policy_mass\":%.6f,\"shortlist_capped\":%s,"
                    "\"audited_moves\":%d,\"omitted_moves\":%d,"
                    "\"world_model\":\"%s\","
                    "\"continuation_policy\":\"%s\","
                    "\"continuation_symmetries\":%d,"
                    "\"continuation_symmetry_mode\":\"%s\","
                    "\"root_dead_discard_focus\":%s,"
                    "\"continuation_dead_discard_focus\":%s,"
                    "\"challenger_discard_guard\":%s,"
                    "\"deck_max\":%d,"
                    "\"evaluation_kind\":\"%s\","
                    "\"planner\":{\"deck_max\":%d,\"block_gap\":%d,"
                    "\"turns\":%d,\"guaranteed_score\":%d,"
                    "\"policy_score\":%d,\"policy_regret\":%d,"
                    "\"policy_block_cost\":%d,\"selected_block_cost\":%d},"
                    "\"semantic_candidates\":%d,"
                    "\"baseline_source\":\"%s\","
                    "\"confirmation\":{\"required\":%s,"
                    "\"passed\":%s,\"worlds\":%d,\"configured_worlds\":%d,"
                    "\"continuation\":\"random_symmetry_argmax\"}}",
                pv,
                rd == MATCH_ROUNDS - 1 && evaluator.win_q == 2
                    ? "final_hybrid"
                    : (rd == MATCH_ROUNDS - 1 && evaluator.win_q == 1
                        ? "final_result" : "round_margin"),
                ss.worlds, ss.max_worlds, ss.worlds > 0 ? "true" : "false",
                analysis_status, ss.resolved ? "true" : "false",
                evaluator.override_k > 3.5f ? evaluator.override_k : 3.5f,
                evaluator.override_min, ss.policy_mass, evaluator.cand_mass,
                ss.worlds > 0 && evaluator.cand_mass > 0.0f &&
                        ss.policy_mass + 1e-6 < evaluator.cand_mass &&
                        ss.n >= evaluator.root_width
                    ? "true" : "false",
                ss.n, nleg - ss.n,
                evaluator.no_belief ? "uniform_card_count"
                                    : "learned_fixed_cardinality",
                evaluator.playout_sample == 1
                    ? "sampled_policy"
                    : (evaluator.playout_sample == 2
                        ? "random_symmetry_argmax"
                        : "exact_ensemble_argmax"),
                evaluator.playout_symmetries,
                evaluator.playout_sample > 0 &&
                        evaluator.playout_symmetries > 1
                    ? "random_group_member_per_decision"
                    : "exact_average",
                evaluator.prune_dom ? "true" : "false",
                (evaluator.playout_prune < 0
                     ? evaluator.prune_dom : evaluator.playout_prune)
                    ? "true" : "false",
                evaluator.discard_guard ? "true" : "false",
                evaluator.deck_max,
                search_metric(&ss),
                evaluator.plan_deck_max,
                evaluator.plan_block_gap,
                ss.planner_turns,
                ss.planner_score,
                ss.planner_policy_score,
                ss.planner_regret,
                ss.planner_policy_block,
                ss.planner_selected_block,
                ss.semantic_candidates,
                ss.planned_baseline
                    ? "visible_hand_scheduler"
                    : (ss.deck_end_baseline
                        ? "last_deck_dominance" : "network_policy"),
                evaluator.override_k > 0.0f ? "true" : "false",
                ss.confirmed ? "true" : "false",
                ss.confirm_worlds, evaluator.confirm_dets);

        /* the card that will be drawn: read before lc_apply */
        int drawn = played.draw == 0 ? st.deck[st.deck_pos]
                                : st.pile[played.draw - 1][st.pile_n[played.draw - 1] - 1];
        fprintf(pf, ",\"move\":");
        j_move_open(pf, played);
        fprintf(pf, ",\"drawn\":");
        j_card(pf, drawn);
        fprintf(pf, "}}");

        lc_apply(&st, played);
    }

    round_scores[rd][0] = lc_score(&st, 0);
    round_scores[rd][1] = lc_score(&st, 1);
    cum[0] += round_scores[rd][0];
    cum[1] += round_scores[rd][1];
    }   /* rounds */
    fclose(pf);

    printf("{\"meta\":{\"actor\":");
    j_string(stdout, actor_spec);
    printf(",\"evaluator\":");
    j_string(stdout, eval_spec);
    printf(",\"belief_alpha\":%.3f,\"belief_symmetries\":%d,"
           "\"seed\":%llu,\"plies\":%d,\"rounds\":%d,\"round_scores\":[",
           belief_alpha, belief_symmetries,
           (unsigned long long)seed, ply, rounds);
    for (int rd = 0; rd < rounds; rd++)
        printf("%s[%d,%d]", rd ? "," : "", round_scores[rd][0], round_scores[rd][1]);
    printf("],\"final\":[%d,%d],\"generated\":\"analyze\"},\n", cum[0], cum[1]);
    printf("\"start_hands\":[%s,%s],\n", start_hands[0], start_hands[1]);
    printf("\"plies\":[%s]}\n", plybuf);
    free(plybuf);
    return 0;
}
