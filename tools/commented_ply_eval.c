/* commented_ply_eval -- deterministic, non-gating review of one saved state.
 *
 * The deployed actor is asked for its actual move once, with a declared RNG
 * seed.  Human-nominated alternatives and, when not already nominated, the
 * actor's selected action are then evaluated separately on paired, uniformly
 * sampled hidden worlds.  Every branch uses the supplied network's exact
 * suit-ensemble argmax policy to the end of the round; the rollout actor is
 * deliberately not called recursively.  This keeps the diagnostic affordable
 * and prevents a search actor from grading itself.
 */
#include "../src/agent.h"
#include "../src/search.h"
#include "../src/spec.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

#define MAXC 8

static void usage(const char *argv0)
{
    fprintf(stderr,
            "usage: %s --state PATH --actor SPEC --net PATH --seed N "
            "[--worlds N] [--symmetries 1|5|10|20|120] "
            "[--candidate \"CARD p|d DRAW\" ...] "
            "[--belief [--belief-alpha A] [--belief-card CARD]]\n",
            argv0);
}

static int parse_long(const char *text, long lo, long hi, long *out)
{
    if (!text || !*text) return 0;
    char *end = NULL;
    errno = 0;
    long value = strtol(text, &end, 10);
    if (errno || !end || end == text || *end || value < lo || value > hi)
        return 0;
    *out = value;
    return 1;
}

static int parse_seed(const char *text, uint64_t *out)
{
    if (!text || !*text) return 0;
    char *end = NULL;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno || !end || end == text || *end) return 0;
    *out = (uint64_t)value;
    return 1;
}

static int valid_symmetries(int value)
{
    return value == 1 || value == 5 || value == 10 || value == 20 ||
           value == 120;
}

static void json_string(const char *text)
{
    putchar('"');
    for (; *text; text++) {
        unsigned char c = (unsigned char)*text;
        if (c == '"' || c == '\\') printf("\\%c", c);
        else if (c == '\n') fputs("\\n", stdout);
        else if (c < 0x20) printf("\\u%04x", c);
        else putchar(c);
    }
    putchar('"');
}

/* Lowest unused physical card with this display name.  The three wagers in a
 * suit are observationally identical; assigning their ids in file order is
 * the same convention used by the other saved-state tools. */
static int name_id_free(const char *name, uint64_t used)
{
    char shown[8];
    for (int c = 0; c < NCARD; c++) {
        lc_card_name(c, shown);
        if (!strcasecmp(shown, name) && !((used >> c) & 1ULL)) return c;
    }
    return -1;
}

static int load_state(const char *path, State *st, int *deck_entries_out)
{
    FILE *file = fopen(path, "r");
    if (!file) return 0;
    memset(st, 0, sizeof *st);
    int deck_entries = 0, saw_deck = 0;
    uint64_t used = 0;
    char line[512];
    while (fgets(line, sizeof line, file)) {
        char *token = strtok(line, " \t\n");
        if (!token) continue;
        if (!strcmp(token, "turn")) {
            token = strtok(NULL, " \t\n");
            if (!token) goto bad;
            st->turn = (uint8_t)atoi(token);
        } else if (!strcmp(token, "round")) {
            token = strtok(NULL, " \t\n");
            if (!token) goto bad;
            st->round = (uint8_t)atoi(token);
        } else if (!strcmp(token, "nply")) {
            token = strtok(NULL, " \t\n");
            if (!token) goto bad;
            st->nply = (uint16_t)atoi(token);
        } else if (!strcmp(token, "deck_left")) {
            token = strtok(NULL, " \t\n");
            if (!token) goto bad;
            st->deck_left = (uint8_t)atoi(token);
        } else if (!strcmp(token, "cum")) {
            char *a = strtok(NULL, " \t\n");
            char *b = strtok(NULL, " \t\n");
            if (!a || !b) goto bad;
            st->cum[0] = (int16_t)atoi(a);
            st->cum[1] = (int16_t)atoi(b);
        } else if (!strncmp(token, "hand", 4) &&
                   token[4] >= '0' && token[4] <= '1') {
            int player = token[4] - '0';
            char *word;
            while ((word = strtok(NULL, " \t\n"))) {
                int card = name_id_free(word, used);
                if (card < 0) goto bad;
                used |= UINT64_C(1) << card;
                st->hand[player] |= UINT64_C(1) << card;
                st->hand_n[player]++;
            }
        } else if (!strncmp(token, "known", 5) &&
                   token[5] >= '0' && token[5] <= '1') {
            int player = token[5] - '0';
            char *word;
            while ((word = strtok(NULL, " \t\n"))) {
                char shown[8];
                int found = 0;
                for (int card = 0; card < NCARD; card++) {
                    lc_card_name(card, shown);
                    if (!strcasecmp(shown, word) &&
                        ((st->hand[player] >> card) & 1ULL) &&
                        !((st->known[player] >> card) & 1ULL)) {
                        st->known[player] |= UINT64_C(1) << card;
                        found = 1;
                        break;
                    }
                }
                if (!found) goto bad;
            }
        } else if (!strcmp(token, "exp")) {
            char *ptext = strtok(NULL, " \t\n");
            char *stext = strtok(NULL, " \t\n");
            if (!ptext || !stext) goto bad;
            int player = atoi(ptext), suit = atoi(stext);
            if (player < 0 || player > 1 || suit < 0 || suit >= NSUIT)
                goto bad;
            char *word;
            while ((word = strtok(NULL, " \t\n"))) {
                int card = name_id_free(word, used);
                if (card < 0) goto bad;
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
            char *stext = strtok(NULL, " \t\n");
            if (!stext) goto bad;
            int suit = atoi(stext);
            if (suit < 0 || suit >= NSUIT) goto bad;
            char *word;
            while ((word = strtok(NULL, " \t\n"))) {
                int card = name_id_free(word, used);
                if (card < 0 || st->pile_n[suit] >= NCARD) goto bad;
                used |= UINT64_C(1) << card;
                st->pile[suit][st->pile_n[suit]++] = (uint8_t)card;
                st->discarded |= UINT64_C(1) << card;
            }
        } else if (!strcmp(token, "deck")) {
            char *word;
            saw_deck = 1;
            while ((word = strtok(NULL, " \t\n"))) {
                int card = name_id_free(word, used);
                if (card < 0 || deck_entries >= NCARD) goto bad;
                used |= UINT64_C(1) << card;
                st->deck[deck_entries++] = (uint8_t)card;
            }
        }
    }
    fclose(file);
    if (saw_deck && deck_entries != st->deck_left) return 0;
    if (deck_entries_out) *deck_entries_out = saw_deck ? deck_entries : 0;
    return st->turn <= 1 && st->round < MATCH_ROUNDS &&
           st->deck_left <= NCARD && st->hand_n[0] == HAND_SIZE &&
           st->hand_n[1] == HAND_SIZE;

bad:
    fclose(file);
    return 0;
}

static int find_card(const State *st, int player, const char *name)
{
    char shown[8];
    for (int card = 0; card < NCARD; card++) {
        lc_card_name(card, shown);
        if (!strcasecmp(shown, name) &&
            ((st->hand[player] >> card) & 1ULL))
            return card;
    }
    return -1;
}

static int parse_move(const State *st, const char *text, Move *out)
{
    char card_name[16], action[16], draw_name[16];
    if (sscanf(text, "%15s %15s %15s", card_name, action, draw_name) != 3)
        return 0;
    int card = find_card(st, st->turn, card_name);
    if (card < 0) return 0;
    int draw = -1;
    static const char suits[] = "YBWGR";
    if (!strcasecmp(draw_name, "deck")) draw = 0;
    for (int suit = 0; suit < NSUIT && draw < 0; suit++)
        if (draw_name[0] == suits[suit] ||
            draw_name[0] == suits[suit] + ('a' - 'A'))
            draw = suit + 1;
    if (draw < 0) return 0;
    out->card = (uint8_t)card;
    out->discard = (uint8_t)(action[0] == 'd' || action[0] == 'D');
    out->draw = (uint8_t)draw;
    Move legal[MAX_MOVES];
    int nlegal = lc_moves(st, legal);
    for (int i = 0; i < nlegal; i++)
        if (MOVE_PACK(legal[i]) == MOVE_PACK(*out)) return 1;
    return 0;
}

static void move_text(Move move, char out[24])
{
    char card[8], draw[8];
    static const char suits[] = "YBWGR";
    lc_card_name(move.card, card);
    if (move.draw == 0) strcpy(draw, "deck");
    else snprintf(draw, sizeof draw, "%c", suits[move.draw - 1]);
    snprintf(out, 24, "%s %c %s", card, move.discard ? 'd' : 'p', draw);
}

static float move_prior(const Net *net, const State *st, Move wanted,
                        int symmetries)
{
    Move moves[MAX_MOVES];
    float probability[MAX_MOVES];
    int n = policy_probs_sym(net, st, moves, probability, NULL, symmetries);
    for (int i = 0; i < n; i++)
        if (MOVE_PACK(moves[i]) == MOVE_PACK(wanted)) return probability[i];
    return 0.0f;
}

static int policy_playout(const Net *net, State *state, int perspective,
                          int symmetries)
{
    Move moves[MAX_MOVES];
    float probability[MAX_MOVES];
    while (!state->over) {
        int n = policy_probs_sym(net, state, moves, probability, NULL,
                                 symmetries);
        if (n <= 0) return 0;
        int best = 0;
        for (int i = 1; i < n; i++)
            if (probability[i] > probability[best]) best = i;
        lc_apply(state, moves[best]);
    }
    return lc_score(state, perspective) - lc_score(state, perspective ^ 1);
}

static double log_choose(int n, int k)
{
    return lgamma((double)n + 1.0) - lgamma((double)k + 1.0) -
           lgamma((double)(n - k) + 1.0);
}

static int card_order_before(const BeliefDist *dist, int a, int b)
{
    if (dist->marginal[a] > dist->marginal[b]) return 1;
    if (dist->marginal[a] < dist->marginal[b]) return 0;
    return dist->card[a] < dist->card[b];
}

int main(int argc, char **argv)
{
    const char *state_path = NULL, *actor_spec = NULL, *net_path = NULL;
    const char *candidate_text[MAXC];
    const char *belief_card = "Y9";
    int ncandidate = 0, worlds = 1024, symmetries = 20, belief = 0;
    float belief_alpha = 1.0f;
    uint64_t seed = UINT64_C(2026082301);

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--state") && i + 1 < argc)
            state_path = argv[++i];
        else if (!strcmp(argv[i], "--actor") && i + 1 < argc)
            actor_spec = argv[++i];
        else if (!strcmp(argv[i], "--net") && i + 1 < argc)
            net_path = argv[++i];
        else if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
            if (!parse_seed(argv[++i], &seed)) {
                fprintf(stderr, "commented_ply_eval: invalid seed\n");
                return 1;
            }
        } else if (!strcmp(argv[i], "--worlds") && i + 1 < argc) {
            long value;
            if (!parse_long(argv[++i], 2, 1000000, &value)) {
                fprintf(stderr, "commented_ply_eval: invalid worlds\n");
                return 1;
            }
            worlds = (int)value;
        } else if (!strcmp(argv[i], "--symmetries") && i + 1 < argc) {
            long value;
            if (!parse_long(argv[++i], 1, 120, &value) ||
                !valid_symmetries((int)value)) {
                fprintf(stderr, "commented_ply_eval: invalid symmetries\n");
                return 1;
            }
            symmetries = (int)value;
        } else if (!strcmp(argv[i], "--candidate") && i + 1 < argc &&
                   ncandidate < MAXC) {
            candidate_text[ncandidate++] = argv[++i];
        } else if (!strcmp(argv[i], "--belief")) {
            belief = 1;
        } else if (!strcmp(argv[i], "--belief-alpha") && i + 1 < argc) {
            char *end = NULL;
            double value = strtod(argv[++i], &end);
            if (!end || *end || !isfinite(value) || value < 0.0 ||
                value > 5.0) {
                fprintf(stderr, "commented_ply_eval: invalid belief alpha\n");
                return 1;
            }
            belief_alpha = (float)value;
        } else if (!strcmp(argv[i], "--belief-card") && i + 1 < argc) {
            belief_card = argv[++i];
        } else {
            usage(argv[0]);
            return 1;
        }
    }
    if (!state_path || !actor_spec || !net_path) {
        usage(argv[0]);
        return 1;
    }

    State state;
    int input_deck_entries = 0;
    if (!load_state(state_path, &state, &input_deck_entries)) {
        fprintf(stderr, "commented_ply_eval: cannot load state %s\n",
                state_path);
        return 1;
    }
    Net *net = (Net *)malloc(sizeof *net);
    if (!net || net_load(net, net_path)) {
        fprintf(stderr, "commented_ply_eval: cannot load net %s\n", net_path);
        free(net);
        return 1;
    }
    Agent actor;
    spec_parse(actor_spec, &actor);

    /* All decision-time and counterfactual inputs start at the mover's
     * explicit information boundary.  The complete saved state is retained
     * below only as an offline truth label for the p13 belief diagnostic. */
    State information_view;
    agent_information_view(&state, state.turn, &information_view);

    Rng actor_rng;
    uint64_t actor_seed = seed ^ UINT64_C(0xD1B54A32D192ED03);
    rng_seed(&actor_rng, actor_seed);
    SearchStats actor_stats;
    memset(&actor_stats, 0, sizeof actor_stats);
    Move selected = actor.kind == AG_ROLLOUT
        ? rollout_move(&actor, &information_view, &actor_rng, NULL,
                       &actor_stats)
        : agent_move(&actor, &information_view, &actor_rng);
    char selected_text[24];
    move_text(selected, selected_text);

    Move candidate[MAXC];
    for (int i = 0; i < ncandidate; i++) {
        if (!parse_move(&state, candidate_text[i], &candidate[i])) {
            fprintf(stderr, "commented_ply_eval: illegal candidate %s\n",
                    candidate_text[i]);
            spec_release(&actor);
            free(net);
            return 1;
        }
    }
    int actor_selected_included = 0;
    for (int i = 0; i < ncandidate; i++)
        actor_selected_included |=
            MOVE_PACK(candidate[i]) == MOVE_PACK(selected);
    if (ncandidate > 0 && !actor_selected_included) {
        if (ncandidate >= MAXC) {
            fprintf(stderr, "commented_ply_eval: no room for actor selection\n");
            spec_release(&actor);
            free(net);
            return 1;
        }
        candidate[ncandidate++] = selected;
        actor_selected_included = 1;
    }

    double *value = NULL;
    unsigned long long cap_hits[MAXC] = { 0 };
    if (ncandidate > 0) {
        value = (double *)malloc(sizeof(double) * (size_t)ncandidate *
                                 (size_t)worlds);
        if (!value) {
            fprintf(stderr, "commented_ply_eval: out of memory\n");
            spec_release(&actor);
            free(net);
            return 1;
        }
        Rng world_rng;
        rng_seed(&world_rng, seed);
        for (int world_index = 0; world_index < worlds; world_index++) {
            State world;
            determinize(&information_view, state.turn, &world_rng, &world);
            for (int c = 0; c < ncandidate; c++) {
                State branch = world;
                lc_apply(&branch, candidate[c]);
                value[c * worlds + world_index] =
                    policy_playout(net, &branch, state.turn, symmetries);
                if (branch.nply >= LC_MAX_PLIES && branch.deck_left > 0)
                    cap_hits[c]++;
            }
        }
    }

    BeliefDist distribution;
    int have_belief = belief && belief_dist_init(
        net, &information_view, state.turn, symmetries, belief_alpha,
        &distribution);
    double belief_nll = 0.0;
    if (have_belief && !belief_dist_true_nll(
            &distribution, state.hand[state.turn ^ 1], &belief_nll))
        have_belief = 0;

    printf("{");
    printf("\"schema\":\"lc-commented-ply-eval-v1\",");
    printf("\"state\":{");
    printf("\"path\":"); json_string(state_path);
    printf(",\"turn\":%u,\"round\":%u,\"nply\":%u,\"deck_left\":%u,"
           "\"input_deck_entries\":%d},",
           state.turn, state.round, state.nply, state.deck_left,
           input_deck_entries);
    printf("\"actor\":{");
    printf("\"spec\":"); json_string(actor_spec);
    printf(",\"information_view\":true,\"seed\":\"%llu\",\"selected\":",
           (unsigned long long)actor_seed);
    json_string(selected_text);
    printf(",\"policy_prior\":%.9g,\"search\":",
           move_prior(net, &information_view, selected, symmetries));
    if (actor.kind != AG_ROLLOUT) {
        printf("null");
    } else {
        printf("{");
        printf("\"worlds\":%d,\"world_cap\":%d,\"candidates\":%d,",
               actor_stats.worlds, actor_stats.max_worlds, actor_stats.n);
        printf("\"confirmation_worlds\":%d,", actor_stats.confirm_worlds);
        printf("\"fresh_worlds\":%d,", actor_stats.prefix_confirm_worlds);
        printf("\"controller_veto_worlds\":%d,",
               actor_stats.prefix_veto_worlds);
        printf("\"unfinished_cap_leaves\":%llu,",
               (unsigned long long)actor_stats.unfinished_cap_leaves);
        printf("\"exact_terminal_leaves\":%llu,",
               (unsigned long long)actor_stats.exact_terminal_leaves);
        printf("\"cycle_breaks\":%llu,\"skip_reason\":%d,",
               (unsigned long long)actor_stats.cycle_breaks,
               actor_stats.skip_reason);
        printf("\"ranker_attempted\":%s,\"ranker_passed\":%s}",
               actor_stats.prefix_ranker_attempted ? "true" : "false",
               actor_stats.prefix_ranker_passed ? "true" : "false");
    }
    printf("},");

    printf("\"counterfactual\":{");
    printf("\"seed\":\"%llu\",\"worlds\":%d,",
           (unsigned long long)seed, worlds);
    printf("\"world_model\":\"uniform_exact_card_count\",");
    printf("\"root_information_view\":true,"
           "\"actor_selected_included\":%s,",
           actor_selected_included ? "true" : "false");
    printf("\"continuation\":{\"kind\":\"argmax_policy\","
           "\"symmetries\":%d,\"exact_group_average\":true,"
           "\"recursive_actor\":false},\"candidates\":[", symmetries);
    double reference_mean = 0.0;
    for (int c = 0; c < ncandidate; c++) {
        double mean = 0.0, variance = 0.0, dvariance = 0.0;
        for (int w = 0; w < worlds; w++) mean += value[c * worlds + w];
        mean /= worlds;
        if (c == 0) reference_mean = mean;
        for (int w = 0; w < worlds; w++) {
            double error = value[c * worlds + w] - mean;
            variance += error * error;
            double paired = value[c * worlds + w] - value[w];
            double paired_error = paired - (mean - reference_mean);
            dvariance += paired_error * paired_error;
        }
        variance /= worlds - 1;
        dvariance /= worlds - 1;
        char canonical[24];
        move_text(candidate[c], canonical);
        if (c) putchar(',');
        printf("{");
        printf("\"move\":"); json_string(canonical);
        printf(",\"policy_prior\":%.9g,\"q_round_margin\":%.9g,"
               "\"q_se\":%.9g,\"delta_vs_reference\":%.9g,"
               "\"delta_se\":%.9g,\"cap_hits\":%llu}",
               move_prior(net, &information_view, candidate[c], symmetries),
               mean,
               sqrt(variance / worlds), mean - reference_mean,
               sqrt(dvariance / worlds), cap_hits[c]);
    }
    printf("]},");

    printf("\"belief\":");
    if (!belief) {
        printf("null");
    } else if (!have_belief) {
        printf("{\"valid\":false}");
    } else {
        int order[NCARD];
        for (int i = 0; i < distribution.n; i++) order[i] = i;
        for (int i = 0; i < distribution.n; i++) {
            int best = i;
            for (int j = i + 1; j < distribution.n; j++)
                if (card_order_before(&distribution, order[j], order[best]))
                    best = j;
            int swap = order[i]; order[i] = order[best]; order[best] = swap;
        }
        double sum = 0.0;
        for (int i = 0; i < distribution.n; i++)
            sum += distribution.marginal[i];
        printf("{\"valid\":true,\"kind\":\"fixed_k\","
               "\"information_view\":true,"
               "\"complete_state_used_only_as_truth_label\":true,"
               "\"symmetries\":%d,\"alpha\":%.9g,\"n\":%d,"
               "\"need\":%d,\"marginal_sum\":%.12g,"
               "\"uniform_marginal\":%.12g,\"true_nll\":%.12g,"
               "\"uniform_true_nll\":%.12g,",
               symmetries, belief_alpha, distribution.n, distribution.need,
               sum, distribution.n > 0
                        ? (double)distribution.need / distribution.n : 0.0,
               belief_nll, log_choose(distribution.n, distribution.need));
        printf("\"target\":");
        int target = -1;
        char shown[8];
        for (int i = 0; i < distribution.n; i++) {
            lc_card_name(distribution.card[i], shown);
            if (!strcasecmp(shown, belief_card)) { target = i; break; }
        }
        if (target < 0) {
            printf("null");
        } else {
            lc_card_name(distribution.card[target], shown);
            printf("{\"card\":"); json_string(shown);
            printf(",\"marginal\":%.12g,\"held\":%s}",
                   distribution.marginal[target],
                   ((state.hand[state.turn ^ 1] >> distribution.card[target]) &
                    1ULL) ? "true" : "false");
        }
        printf(",\"cards\":[");
        for (int rank = 0; rank < distribution.n; rank++) {
            int i = order[rank];
            lc_card_name(distribution.card[i], shown);
            if (rank) putchar(',');
            printf("{\"card\":"); json_string(shown);
            printf(",\"marginal\":%.12g,\"held\":%s}",
                   distribution.marginal[i],
                   ((state.hand[state.turn ^ 1] >> distribution.card[i]) &
                    1ULL) ? "true" : "false");
        }
        printf("]}");
    }
    printf("}\n");

    free(value);
    spec_release(&actor);
    free(net);
    return have_belief || !belief ? 0 : 1;
}
