/* flagged_ply_probe -- one fixed-state, two-actor policy-focused audit.
 *
 * Candidate construction is intentionally narrow: take each root model's
 * top three complete semantic policy moves, merge only genuinely identical
 * moves (including draw source, while ignoring physical wager IDs), and cap
 * the deterministic union at five.  Only those admitted complete moves
 * receive rollout worlds.  This is an audit of what the policies seriously
 * consider, not an exhaustive legal-move oracle.
 */
#include "../src/agent.h"
#include "../src/lc.h"
#include "../src/net.h"
#include "../src/search.h"
#include "../src/spec.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#define AUDIT_TOP_MOVES 3
#define AUDIT_CANDIDATES 5

typedef struct {
    Move action;
    double mass;
    int representative;
    int first;
} PolicyCore;

typedef struct {
    Move move[MAX_MOVES];
    float probability[MAX_MOVES];
    int nmove;
    PolicyCore core[MAX_MOVES];
    int ncore;
    int core_order[MAX_MOVES];
    int move_order[MAX_MOVES];
} PolicyView;

typedef struct {
    Move move;
    int source_rank[2];
    float source_probability[2];
} Candidate;

static int same_card(int a, int b)
{
    return a == b || (CARD_IS_WAGER(a) && CARD_IS_WAGER(b) &&
                      CARD_SUIT(a) == CARD_SUIT(b));
}

static int same_action(Move a, Move b)
{
    return a.discard == b.discard && same_card(a.card, b.card);
}

static int same_complete_move(Move a, Move b)
{
    return a.draw == b.draw && same_action(a, b);
}

static uint32_t semantic_pack(Move move)
{
    if (CARD_IS_WAGER(move.card))
        move.card = (uint8_t)CARD_MAKE(CARD_SUIT(move.card), 0);
    return MOVE_PACK(move);
}

/* Lowest unused physical ID for a semantic card name. */
static int name_id_free(const char *name, uint64_t used)
{
    char shown[8];
    for (int c = 0; c < NCARD; c++) {
        lc_card_name(c, shown);
        if (!strcasecmp(shown, name) && !((used >> c) & UINT64_C(1)))
            return c;
    }
    return -1;
}

/* State format is tools/statedump.py's deliberately plain text contract. */
static int load_state(const char *path, State *st)
{
    FILE *file = fopen(path, "r");
    if (!file) return 0;
    memset(st, 0, sizeof *st);
    uint64_t used = 0;
    char line[512];
    while (fgets(line, sizeof line, file)) {
        char *token = strtok(line, " \t\r\n");
        if (!token) continue;
        if (!strcmp(token, "turn")) {
            token = strtok(NULL, " \t\r\n");
            if (!token) goto bad;
            st->turn = (uint8_t)atoi(token);
        } else if (!strcmp(token, "round")) {
            token = strtok(NULL, " \t\r\n");
            if (!token) goto bad;
            st->round = (uint8_t)atoi(token);
        } else if (!strcmp(token, "nply")) {
            token = strtok(NULL, " \t\r\n");
            if (!token) goto bad;
            st->nply = (uint16_t)atoi(token);
        } else if (!strcmp(token, "deck_left")) {
            token = strtok(NULL, " \t\r\n");
            if (!token) goto bad;
            st->deck_left = (uint8_t)atoi(token);
        } else if (!strcmp(token, "cum")) {
            char *a = strtok(NULL, " \t\r\n");
            char *b = strtok(NULL, " \t\r\n");
            if (!a || !b) goto bad;
            st->cum[0] = (int16_t)atoi(a);
            st->cum[1] = (int16_t)atoi(b);
        } else if (!strncmp(token, "hand", 4) &&
                   token[4] >= '0' && token[4] <= '1') {
            int player = token[4] - '0';
            while ((token = strtok(NULL, " \t\r\n"))) {
                int card = name_id_free(token, used);
                if (card < 0) goto bad;
                used |= UINT64_C(1) << card;
                st->hand[player] |= UINT64_C(1) << card;
                st->hand_n[player]++;
            }
        } else if (!strncmp(token, "known", 5) &&
                   token[5] >= '0' && token[5] <= '1') {
            int player = token[5] - '0';
            while ((token = strtok(NULL, " \t\r\n"))) {
                char shown[8];
                int found = -1;
                for (int c = 0; c < NCARD; c++) {
                    lc_card_name(c, shown);
                    if (!strcasecmp(shown, token) &&
                        ((st->hand[player] >> c) & UINT64_C(1)) &&
                        !((st->known[player] >> c) & UINT64_C(1))) {
                        found = c;
                        break;
                    }
                }
                if (found < 0) goto bad;
                st->known[player] |= UINT64_C(1) << found;
            }
        } else if (!strcmp(token, "exp")) {
            char *pword = strtok(NULL, " \t\r\n");
            char *sword = strtok(NULL, " \t\r\n");
            if (!pword || !sword) goto bad;
            int player = atoi(pword), suit = atoi(sword);
            if (player < 0 || player > 1 || suit < 0 || suit >= NSUIT)
                goto bad;
            while ((token = strtok(NULL, " \t\r\n"))) {
                int card = name_id_free(token, used);
                if (card < 0 || CARD_SUIT(card) != suit) goto bad;
                used |= UINT64_C(1) << card;
                st->played[player] |= UINT64_C(1) << card;
                st->exp_n[player][suit]++;
                if (CARD_IS_WAGER(card)) {
                    st->exp_wager[player][suit]++;
                } else {
                    int value = CARD_VALUE(card);
                    if (value > st->exp_top[player][suit])
                        st->exp_top[player][suit] = (uint8_t)value;
                    st->exp_sum[player][suit] =
                        (uint8_t)(st->exp_sum[player][suit] + value);
                }
            }
        } else if (!strcmp(token, "pile")) {
            char *sword = strtok(NULL, " \t\r\n");
            if (!sword) goto bad;
            int suit = atoi(sword);
            if (suit < 0 || suit >= NSUIT) goto bad;
            while ((token = strtok(NULL, " \t\r\n"))) {
                int card = name_id_free(token, used);
                if (card < 0 || CARD_SUIT(card) != suit ||
                    st->pile_n[suit] >= NRANK)
                    goto bad;
                used |= UINT64_C(1) << card;
                st->pile[suit][st->pile_n[suit]++] = (uint8_t)card;
                st->discarded |= UINT64_C(1) << card;
            }
        } else {
            goto bad;
        }
    }
    fclose(file);
    return st->turn <= 1 && st->round < MATCH_ROUNDS &&
           st->hand_n[0] == HAND_SIZE && st->hand_n[1] == HAND_SIZE &&
           st->deck_left <= NCARD;
bad:
    fclose(file);
    return 0;
}

static int build_policy_view(const Agent *actor, const State *st,
                             PolicyView *view)
{
    memset(view, 0, sizeof *view);
    if (!actor || !actor->net) return 0;
    view->nmove = policy_probs_sym(
        actor->net, st, view->move, view->probability, NULL,
        actor->symmetries);
    if (view->nmove <= 0) return 0;
    for (int i = 0; i < view->nmove; i++) {
        int core = -1;
        for (int c = 0; c < view->ncore; c++)
            if (same_action(view->move[i], view->core[c].action)) {
                core = c;
                break;
            }
        if (core < 0) {
            core = view->ncore++;
            view->core[core].action = view->move[i];
            view->core[core].representative = i;
            view->core[core].first = i;
        }
        float probability = isfinite(view->probability[i]) &&
                            view->probability[i] > 0.0f
            ? view->probability[i] : 0.0f;
        view->core[core].mass += probability;
        int representative = view->core[core].representative;
        if (probability > view->probability[representative] ||
            (probability == view->probability[representative] &&
             semantic_pack(view->move[i]) <
                 semantic_pack(view->move[representative])))
            view->core[core].representative = i;
    }
    for (int i = 0; i < view->ncore; i++) view->core_order[i] = i;
    for (int i = 0; i < view->ncore; i++) {
        int best = i;
        for (int j = i + 1; j < view->ncore; j++) {
            PolicyCore *a = &view->core[view->core_order[j]];
            PolicyCore *b = &view->core[view->core_order[best]];
            float apr = view->probability[a->representative];
            float bpr = view->probability[b->representative];
            if (a->mass > b->mass ||
                (a->mass == b->mass && apr > bpr) ||
                (a->mass == b->mass && apr == bpr &&
                 semantic_pack(view->move[a->representative]) <
                     semantic_pack(view->move[b->representative])))
                best = j;
        }
        int swap = view->core_order[i];
        view->core_order[i] = view->core_order[best];
        view->core_order[best] = swap;
    }
    for (int i = 0; i < view->nmove; i++) view->move_order[i] = i;
    for (int i = 0; i < view->nmove; i++) {
        int best = i;
        for (int j = i + 1; j < view->nmove; j++) {
            int a = view->move_order[j], b = view->move_order[best];
            if (view->probability[a] > view->probability[b] ||
                (view->probability[a] == view->probability[b] &&
                 semantic_pack(view->move[a]) < semantic_pack(view->move[b])))
                best = j;
        }
        int swap = view->move_order[i];
        view->move_order[i] = view->move_order[best];
        view->move_order[best] = swap;
    }
    return 1;
}

static int add_candidate(Candidate *candidate, int *n,
                         Move move, int actor, int rank, float probability)
{
    for (int i = 0; i < *n; i++)
        if (same_complete_move(candidate[i].move, move)) {
            candidate[i].source_rank[actor] = rank;
            candidate[i].source_probability[actor] = probability;
            return 0;
        }
    if (*n >= 2 * AUDIT_TOP_MOVES) return 0;
    Candidate *out = &candidate[(*n)++];
    memset(out, 0, sizeof *out);
    out->move = move;
    out->source_rank[actor] = rank;
    out->source_probability[actor] = probability;
    return 1;
}

static int candidate_score(const Candidate *candidate)
{
    int score = 0;
    for (int actor = 0; actor < 2; actor++)
        if (candidate->source_rank[actor] > 0)
            score += AUDIT_TOP_MOVES + 1 - candidate->source_rank[actor];
    return score;
}

static int candidate_before(const Candidate *a, const Candidate *b)
{
    int ascore = candidate_score(a), bscore = candidate_score(b);
    if (ascore != bscore) return ascore > bscore;
    int asources = (a->source_rank[0] > 0) + (a->source_rank[1] > 0);
    int bsources = (b->source_rank[0] > 0) + (b->source_rank[1] > 0);
    if (asources != bsources) return asources > bsources;
    int amin = 99, bmin = 99;
    float aprob = 0.0f, bprob = 0.0f;
    for (int actor = 0; actor < 2; actor++) {
        if (a->source_rank[actor] > 0 && a->source_rank[actor] < amin)
            amin = a->source_rank[actor];
        if (b->source_rank[actor] > 0 && b->source_rank[actor] < bmin)
            bmin = b->source_rank[actor];
        if (a->source_probability[actor] > aprob)
            aprob = a->source_probability[actor];
        if (b->source_probability[actor] > bprob)
            bprob = b->source_probability[actor];
    }
    if (amin != bmin) return amin < bmin;
    if (aprob != bprob) return aprob > bprob;
    return semantic_pack(a->move) < semantic_pack(b->move);
}

static int find_policy_move(const PolicyView *view, Move move)
{
    for (int i = 0; i < view->nmove; i++)
        if (same_complete_move(view->move[i], move)) return i;
    return -1;
}

static int find_policy_core(const PolicyView *view, Move move)
{
    for (int c = 0; c < view->ncore; c++)
        if (same_action(view->core[c].action, move)) return c;
    return -1;
}

static int move_rank(const PolicyView *view, int index)
{
    for (int rank = 0; rank < view->nmove; rank++)
        if (view->move_order[rank] == index) return rank + 1;
    return 0;
}

static int core_rank(const PolicyView *view, int index)
{
    for (int rank = 0; rank < view->ncore; rank++)
        if (view->core_order[rank] == index) return rank + 1;
    return 0;
}

static void json_string(const char *text)
{
    putchar('"');
    for (const unsigned char *p = (const unsigned char *)text; *p; p++) {
        if (*p == '"' || *p == '\\') printf("\\%c", *p);
        else if (*p == '\n') fputs("\\n", stdout);
        else if (*p < 0x20) printf("\\u%04x", *p);
        else putchar(*p);
    }
    putchar('"');
}

static void move_text(Move move, char text[24])
{
    static const char suits[NSUIT + 1] = "YBWGR";
    char card[8], draw[8];
    lc_card_name(move.card, card);
    if (move.draw == 0) strcpy(draw, "deck");
    else snprintf(draw, sizeof draw, "%c", suits[move.draw - 1]);
    snprintf(text, 24, "%s %c %s", card, move.discard ? 'd' : 'p', draw);
}

static void print_policy_record(const PolicyView *view, Move move)
{
    int index = find_policy_move(view, move);
    int core = find_policy_core(view, move);
    printf("{\"complete_rank\":%d,\"probability\":%.9g,"
           "\"core_rank\":%d,\"core_mass\":%.9g}",
           index >= 0 ? move_rank(view, index) : 0,
           index >= 0 ? view->probability[index] : 0.0,
           core >= 0 ? core_rank(view, core) : 0,
           core >= 0 ? view->core[core].mass : 0.0);
}

static int admitted_baseline(const PolicyView *view,
                             const Candidate *candidate, int n)
{
    int best = 0;
    float best_probability = -1.0f;
    for (int c = 0; c < n; c++) {
        int index = find_policy_move(view, candidate[c].move);
        float probability = index >= 0 ? view->probability[index] : 0.0f;
        if (probability > best_probability) {
            best = c;
            best_probability = probability;
        }
    }
    return best;
}

typedef struct {
    int card;
    int wager;
    int unseen_copies;
    double estimate;
    double prior;
    double prior_at_least_one;
} BeliefRow;

static void print_belief(const Agent *actor, const State *state)
{
    BeliefDist belief;
    if (!belief_dist_init(actor->net, state, state->turn,
                          actor->symmetries, actor->belief_alpha, &belief)) {
        fputs("null", stdout);
        return;
    }
    BeliefRow row[NCARD];
    int nrow = 0;
    double sum = 0.0;
    for (int i = 0; i < belief.n; i++) {
        sum += belief.marginal[i];
        int wager = CARD_IS_WAGER(belief.card[i]);
        int found = -1;
        if (wager) {
            for (int r = 0; r < nrow; r++)
                if (row[r].wager &&
                    CARD_SUIT(row[r].card) == CARD_SUIT(belief.card[i])) {
                    found = r;
                    break;
                }
        }
        if (found < 0) {
            found = nrow++;
            memset(&row[found], 0, sizeof row[found]);
            row[found].card = wager
                ? CARD_MAKE(CARD_SUIT(belief.card[i]), 0)
                : belief.card[i];
            row[found].wager = wager;
        }
        row[found].unseen_copies++;
        row[found].estimate += belief.marginal[i];
    }
    const double per_card_prior =
        belief.n > 0 ? (double)belief.need / (double)belief.n : 0.0;
    for (int r = 0; r < nrow; r++) {
        row[r].prior = row[r].unseen_copies * per_card_prior;
        if (row[r].wager) {
            double none = 1.0;
            if (belief.n - row[r].unseen_copies < belief.need) {
                none = 0.0;
            } else {
                for (int i = 0; i < belief.need; i++)
                    none *= (double)(belief.n - row[r].unseen_copies - i)
                          / (double)(belief.n - i);
            }
            row[r].prior_at_least_one = 1.0 - none;
        }
    }
    for (int i = 0; i < nrow; i++) {
        int best = i;
        for (int j = i + 1; j < nrow; j++)
            if (row[j].estimate > row[best].estimate ||
                (row[j].estimate == row[best].estimate &&
                 row[j].card < row[best].card))
                best = j;
        BeliefRow swap = row[i];
        row[i] = row[best];
        row[best] = swap;
    }
    printf("{\"method\":\"fixed_cardinality_network_head\","
           "\"used_by_panel\":false,\"alpha\":%.9g,"
           "\"symmetries\":%d,\"unknown_hand\":%d,"
           "\"unknown_pool\":%d,\"marginal_sum\":%.9g,"
           "\"simple_prior\":\"uniform fixed-cardinality card count\","
           "\"wager_semantics\":\"expected indistinguishable copies\","
           "\"cards\":[",
           actor->belief_alpha, actor->symmetries, belief.need, belief.n, sum);
    for (int rank = 0; rank < nrow; rank++) {
        char card[8];
        lc_card_name(row[rank].card, card);
        if (rank) putchar(',');
        printf("{\"card\":");
        json_string(card);
        printf(",\"rank\":%d,\"metric\":\"%s\","
               "\"estimate\":%.9g,\"prior\":%.9g,"
               "\"head_minus_prior\":%.9g,\"unseen_copies\":%d",
               rank + 1, row[rank].wager
                   ? "expected_count" : "hold_probability",
               row[rank].estimate, row[rank].prior,
               row[rank].estimate - row[rank].prior,
               row[rank].unseen_copies);
        if (row[rank].wager) {
            printf(",\"expected_count\":%.9g,"
                   "\"prior_expected_count\":%.9g,"
                   "\"prior_at_least_one\":%.9g",
                   row[rank].estimate, row[rank].prior,
                   row[rank].prior_at_least_one);
        } else {
            printf(",\"probability\":%.9g", row[rank].estimate);
        }
        putchar('}');
    }
    fputs("]}", stdout);
}

static const char *audit_objective_label(const Agent *actor,
                                         const State *state)
{
    if (actor->win_q == 3 && actor->match_value)
        return state->round == MATCH_ROUNDS - 1
            ? "final_hybrid"
            : "controller_bound_full_match_value";
    if (state->round == MATCH_ROUNDS - 1 && actor->win_q == 2)
        return "final_hybrid";
    if (state->round == MATCH_ROUNDS - 1 && actor->win_q == 1)
        return "final_result";
    return "round_margin";
}

static const char *audit_objective_units(const Agent *actor,
                                         const State *state)
{
    const char *label = audit_objective_label(actor, state);
    if (!strcmp(label, "controller_bound_full_match_value"))
        return "expected_full_match_hybrid_utility_points";
    if (!strcmp(label, "final_hybrid"))
        return "hybrid_match_utility_points";
    if (!strcmp(label, "final_result"))
        return "signed_match_result_utility_points";
    return "round_points";
}

static void print_belief_actor(const char *label, const char *spec,
                               const Agent *actor, const State *state)
{
    printf("{\"label\":");
    json_string(label);
    printf(",\"spec\":");
    json_string(spec);
    printf(",\"action_panel\":false,\"belief\":");
    print_belief(actor, state);
    printf(",\"objective\":%d,\"objective_label\":",
           actor->win_q);
    json_string(audit_objective_label(actor, state));
    printf(",\"objective_units\":");
    json_string(audit_objective_units(actor, state));
    putchar('}');
}

static void print_actor(const char *label, const char *spec,
                        const Agent *actor, const State *state,
                        const PolicyView *policy,
                        const Candidate *candidate, int ncandidate,
                        const RolloutAuditPanel *panel, Move deployed,
                        const SearchStats *deployed_stats)
{
    char root_move[24], selected_move[24], baseline_move[24],
         deployed_move[24];
    move_text(policy->move[policy->move_order[0]], root_move);
    move_text(candidate[panel->selected].move, selected_move);
    move_text(candidate[panel->baseline].move, baseline_move);
    move_text(deployed, deployed_move);
    printf("{\"label\":");
    json_string(label);
    printf(",\"spec\":");
    json_string(spec);
    printf(",\"root_policy_selected\":");
    json_string(root_move);
    printf(",\"root_policy_probability\":%.9g,"
           "\"admitted_policy_baseline\":",
           policy->probability[policy->move_order[0]]);
    json_string(baseline_move);
    printf(",\"panel_selected\":");
    json_string(selected_move);
    printf(",\"deployed_selected\":");
    json_string(deployed_move);
    printf(",\"deployed_worlds\":%d,\"deployed_candidates\":%d,"
           "\"deployed_unfinished_cap_leaves\":%llu",
           deployed_stats->worlds, deployed_stats->n,
           (unsigned long long)deployed_stats->unfinished_cap_leaves);
    printf(",\"belief\":");
    print_belief(actor, state);
    printf(",\"action_panel\":true,\"objective\":%d,"
           "\"objective_label\":",
           panel->objective);
    json_string(audit_objective_label(actor, state));
    printf(",\"objective_units\":");
    json_string(audit_objective_units(actor, state));
    printf(",\"requested_worlds\":%d,"
           "\"worlds\":%d,\"hidden_support\":%d,"
           "\"exact_hidden_support\":%s,"
           "\"exact_terminal_leaves\":%llu,"
           "\"unfinished_cap_leaves\":%llu,"
           "\"cycle_breaks\":%llu,\"cap_reserve_forces\":%llu,"
           "\"rows\":[",
           panel->requested_worlds, panel->worlds,
           panel->hidden_support,
           panel->exact_hidden_support ? "true" : "false",
           (unsigned long long)panel->exact_terminal_leaves,
           (unsigned long long)panel->unfinished_cap_leaves,
           (unsigned long long)panel->cycle_breaks,
           (unsigned long long)panel->cap_reserve_forces);
    for (int c = 0; c < ncandidate; c++) {
        char move[24];
        move_text(candidate[c].move, move);
        if (c) putchar(',');
        printf("{\"move\":");
        json_string(move);
        printf(",\"q\":%.9g,\"se\":%.9g,"
               "\"delta_vs_policy_baseline\":%.9g,"
               "\"delta_se\":%.9g}",
               panel->q[c], panel->se[c], panel->delta[c],
               panel->delta_se[c]);
    }
    printf("]}");
}

int main(int argc, char **argv)
{
    const char *state_path = NULL;
    const char *reference_spec = NULL;
    const char *candidate_spec = NULL;
    uint64_t seed = UINT64_C(202608230001);
    int worlds = 8192;
    int belief_only = 0;
    float belief_alpha = 1.15f;
    const char *assert_legal[MAX_MOVES];
    int nassert_legal = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-S") && i + 1 < argc)
            state_path = argv[++i];
        else if (!strcmp(argv[i], "-a") && i + 1 < argc)
            reference_spec = argv[++i];
        else if (!strcmp(argv[i], "-b") && i + 1 < argc)
            candidate_spec = argv[++i];
        else if (!strcmp(argv[i], "-s") && i + 1 < argc)
            seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "-w") && i + 1 < argc)
            worlds = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-B") && i + 1 < argc)
            belief_alpha = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "--belief-only"))
            belief_only = 1;
        else if (!strcmp(argv[i], "--assert-legal") && i + 1 < argc &&
                 nassert_legal < MAX_MOVES)
            assert_legal[nassert_legal++] = argv[++i];
        else {
            fprintf(stderr, "usage: %s -S STATE -a REFERENCE_ROLLOUT "
                    "-b CANDIDATE_ROLLOUT [-w WORLDS] [-s SEED] "
                    "[-B BELIEF_ALPHA] [--belief-only] "
                    "[--assert-legal MOVE]\n",
                    argv[0]);
            return 2;
        }
    }
    if (!state_path || !reference_spec || !candidate_spec || worlds < 2 ||
        !isfinite(belief_alpha) || belief_alpha < 0.0f ||
        belief_alpha > 5.0f) {
        fprintf(stderr, "flagged_ply_probe: missing/invalid arguments\n");
        return 2;
    }

    State state;
    if (!load_state(state_path, &state)) {
        fprintf(stderr, "flagged_ply_probe: cannot load state %s\n",
                state_path);
        return 1;
    }
    Move legal_for_assertion[MAX_MOVES];
    int nlegal_for_assertion = lc_moves(&state, legal_for_assertion);
    for (int expected = 0; expected < nassert_legal; expected++) {
        int found = 0;
        for (int i = 0; i < nlegal_for_assertion; i++) {
            char move[24];
            move_text(legal_for_assertion[i], move);
            if (!strcmp(assert_legal[expected], move)) {
                found = 1;
                break;
            }
        }
        if (!found) {
            fprintf(stderr, "flagged_ply_probe: review move is not legal: %s\n",
                    assert_legal[expected]);
            return 1;
        }
    }
    Agent actor[2];
    spec_parse(reference_spec, &actor[0]);
    spec_parse(candidate_spec, &actor[1]);
    actor[0].belief_alpha = belief_alpha;
    actor[1].belief_alpha = belief_alpha;
    if (actor[0].kind != AG_ROLLOUT || actor[1].kind != AG_ROLLOUT ||
        !actor[0].net || !actor[1].net ||
        actor[0].exact_terminal != 1 || actor[1].exact_terminal != 1) {
        fprintf(stderr, "flagged_ply_probe: both actors must be network "
                "rollouts with exact_terminal=1\n");
        spec_release(&actor[0]);
        spec_release(&actor[1]);
        return 1;
    }

    if (belief_only) {
        printf("{\"schema\":\"lc-flagged-ply-probe-v1\","
               "\"belief_only\":true,\"state\":");
        json_string(state_path);
        printf(",\"turn\":%d,\"round\":%d,\"round_ply\":%d,"
               "\"deck_left\":%d,\"legal_moves\":%d,"
               "\"candidate_rule\":\"action panel intentionally omitted "
               "for a belief-only case\",\"evaluated_moves\":0,"
               "\"candidates\":[],\"actors\":[",
               state.turn, state.round, state.nply, state.deck_left,
               nlegal_for_assertion);
        print_belief_actor(
            "reference", reference_spec, &actor[0], &state);
        putchar(',');
        print_belief_actor(
            "candidate", candidate_spec, &actor[1], &state);
        printf("]}\n");
        spec_release(&actor[0]);
        spec_release(&actor[1]);
        return 0;
    }

    PolicyView policy[2];
    if (!build_policy_view(&actor[0], &state, &policy[0]) ||
        !build_policy_view(&actor[1], &state, &policy[1])) {
        fprintf(stderr, "flagged_ply_probe: policy evaluation failed\n");
        spec_release(&actor[0]);
        spec_release(&actor[1]);
        return 1;
    }

    Candidate candidate[2 * AUDIT_TOP_MOVES];
    int ncandidate = 0;
    memset(candidate, 0, sizeof candidate);
    for (int which = 0; which < 2; which++) {
        int keep = policy[which].nmove < AUDIT_TOP_MOVES
            ? policy[which].nmove : AUDIT_TOP_MOVES;
        for (int rank = 0; rank < keep; rank++) {
            int representative = policy[which].move_order[rank];
            add_candidate(candidate, &ncandidate,
                          policy[which].move[representative], which, rank + 1,
                          policy[which].probability[representative]);
        }
    }
    for (int i = 0; i < ncandidate; i++) {
        int best = i;
        for (int j = i + 1; j < ncandidate; j++)
            if (candidate_before(&candidate[j], &candidate[best])) best = j;
        Candidate swap = candidate[i];
        candidate[i] = candidate[best];
        candidate[best] = swap;
    }
    if (ncandidate > AUDIT_CANDIDATES) ncandidate = AUDIT_CANDIDATES;

    Move moves[AUDIT_CANDIDATES];
    for (int c = 0; c < ncandidate; c++) moves[c] = candidate[c].move;
    RolloutAuditPanel panel[2];
    Move deployed[2];
    SearchStats deployed_stats[2];
    for (int which = 0; which < 2; which++) {
        int baseline = admitted_baseline(
            &policy[which], candidate, ncandidate);
        int status = rollout_audit_panel(
            &actor[which], &state, moves, ncandidate, baseline,
            seed, worlds, &panel[which]);
        if (status != 0) {
            fprintf(stderr, "flagged_ply_probe: panel %d failed (%d)\n",
                    which, status);
            spec_release(&actor[0]);
            spec_release(&actor[1]);
            return 1;
        }
        Rng deployed_rng;
        rng_seed(
            &deployed_rng,
            seed ^ UINT64_C(0x94D049BB133111EB));
        deployed[which] = rollout_move(
            &actor[which], &state, &deployed_rng, NULL,
            &deployed_stats[which]);
    }

    printf("{\"schema\":\"lc-flagged-ply-probe-v1\","
           "\"belief_only\":%s,\"state\":",
           belief_only ? "true" : "false");
    json_string(state_path);
    printf(",\"turn\":%d,\"round\":%d,\"round_ply\":%d,"
           "\"deck_left\":%d,\"legal_moves\":%d,"
           "\"candidate_rule\":\"union of each actor's top-three "
           "complete semantic policy moves; physical wager IDs are deduped "
           "but draw sources remain distinct; at most five moves; no "
           "exhaustive legal-move rollout\","
           "\"evaluated_moves\":%d,\"candidates\":[",
           state.turn, state.round, state.nply, state.deck_left,
           policy[0].nmove, ncandidate);
    for (int c = 0; c < ncandidate; c++) {
        char move[24];
        move_text(candidate[c].move, move);
        if (c) putchar(',');
        printf("{\"move\":");
        json_string(move);
        printf(",\"admission\":{\"reference_top_move_rank\":%d,"
               "\"candidate_top_move_rank\":%d},\"reference_policy\":",
               candidate[c].source_rank[0], candidate[c].source_rank[1]);
        print_policy_record(&policy[0], candidate[c].move);
        printf(",\"candidate_policy\":");
        print_policy_record(&policy[1], candidate[c].move);
        putchar('}');
    }
    printf("],\"actors\":[");
    print_actor("reference", reference_spec, &actor[0], &state, &policy[0],
                candidate, ncandidate, &panel[0], deployed[0],
                &deployed_stats[0]);
    putchar(',');
    print_actor("candidate", candidate_spec, &actor[1], &state, &policy[1],
                candidate, ncandidate, &panel[1], deployed[1],
                &deployed_stats[1]);
    printf("]}\n");
    spec_release(&actor[0]);
    spec_release(&actor[1]);
    return 0;
}
