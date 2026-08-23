/* commented_ply_eval -- deterministic, non-gating review of one saved state.
 *
 * The deployed actor is asked for its actual move once, with a declared RNG
 * seed.  Human-nominated alternatives and, when not already nominated, the
 * actor's selected action are then evaluated separately on paired, uniformly
 * sampled hidden worlds and future deals.  Every branch uses the supplied
 * network's exact suit-ensemble argmax policy through the complete remaining
 * three-round match; the rollout actor is deliberately not called recursively.
 * This keeps the diagnostic independent of the actor's own search while
 * measuring the match-level consequences of the reviewed action.
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

typedef struct {
    double margin;
    double match_score;
    double hybrid;
} FullMatchValue;

typedef struct {
    double mean;
    double se;
    double delta;
    double delta_se;
} MetricSummary;

static uint64_t mix64(uint64_t x)
{
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

static uint64_t domain_seed(uint64_t seed, uint64_t world, uint64_t round,
                            uint64_t tag)
{
    return mix64(seed ^ mix64(world + UINT64_C(0x100000001b3)) ^
                 mix64(round + UINT64_C(0x9e3779b9)) ^ tag);
}

static uint64_t hash_init(void)
{
    return UINT64_C(14695981039346656037);
}

static uint64_t hash_extend(uint64_t hash, const void *data, size_t size)
{
    const unsigned char *bytes = (const unsigned char *)data;
    for (size_t i = 0; i < size; i++) {
        hash ^= bytes[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

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

static int same_semantic_card(int a, int b)
{
    return a == b || (CARD_IS_WAGER(a) && CARD_IS_WAGER(b) &&
                      CARD_SUIT(a) == CARD_SUIT(b));
}

static int same_semantic_move(Move a, Move b)
{
    return a.discard == b.discard && a.draw == b.draw &&
           same_semantic_card(a.card, b.card);
}

static uint32_t semantic_pack(Move move)
{
    if (CARD_IS_WAGER(move.card))
        move.card = (uint8_t)CARD_MAKE(CARD_SUIT(move.card), 0);
    return MOVE_PACK(move);
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
    Move wanted = {
        .card = (uint8_t)card,
        .discard = (uint8_t)(action[0] == 'd' || action[0] == 'D'),
        .draw = (uint8_t)draw,
    };
    Move legal[MAX_MOVES];
    int nlegal = lc_moves(st, legal);
    for (int i = 0; i < nlegal; i++)
        if (same_semantic_move(legal[i], wanted)) {
            *out = legal[i];
            return 1;
        }
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
    float semantic_probability = 0.0f;
    for (int i = 0; i < n; i++)
        if (same_semantic_move(moves[i], wanted))
            semantic_probability += probability[i];
    if (semantic_probability < 0.0f) return 0.0f;
    return semantic_probability > 1.0f ? 1.0f : semantic_probability;
}

static int add_semantic_candidate(Move candidate[MAXC], int *n, Move move)
{
    for (int i = 0; i < *n; i++)
        if (same_semantic_move(candidate[i], move)) return 0;
    if (*n >= MAXC) return -1;
    candidate[(*n)++] = move;
    return 1;
}

/* Return the top distinct complete semantic moves under the exact ensemble.
 * This is used only to provide p13, whose human comment concerned belief and
 * nominated no move, with a neutral paired action reference. */
static int add_policy_top_distinct(const Net *net, const State *state,
                                   int symmetries, Move candidate[MAXC],
                                   int *ncandidate, int wanted)
{
    Move moves[MAX_MOVES];
    float probability[MAX_MOVES];
    Move semantic[MAX_MOVES];
    float semantic_probability[MAX_MOVES];
    int order[MAX_MOVES];
    int n = policy_probs_sym(net, state, moves, probability, NULL,
                             symmetries);
    if (n <= 0) return 0;
    /* lc_moves currently emits one canonical wager per suit, but aggregate
     * here rather than depending on that implementation detail.  This keeps
     * both reported priors and the neutral top-two panel correct if another
     * policy producer ever exposes multiple physical wager ids. */
    int nsemantic = 0;
    for (int i = 0; i < n; i++) {
        int found = -1;
        for (int j = 0; j < nsemantic; j++)
            if (same_semantic_move(semantic[j], moves[i])) {
                found = j;
                break;
            }
        if (found < 0) {
            semantic[nsemantic] = moves[i];
            semantic_probability[nsemantic] = probability[i];
            nsemantic++;
        } else {
            semantic_probability[found] += probability[i];
        }
    }
    for (int i = 0; i < nsemantic; i++) order[i] = i;
    for (int i = 0; i < nsemantic; i++) {
        int best = i;
        for (int j = i + 1; j < nsemantic; j++) {
            int a = order[j], b = order[best];
            if (semantic_probability[a] > semantic_probability[b] ||
                (semantic_probability[a] == semantic_probability[b] &&
                 semantic_pack(semantic[a]) < semantic_pack(semantic[b])))
                best = j;
        }
        int swap = order[i]; order[i] = order[best]; order[best] = swap;
    }
    int added = 0;
    for (int rank = 0; rank < nsemantic && added < wanted; rank++) {
        int status = add_semantic_candidate(
            candidate, ncandidate, semantic[order[rank]]);
        if (status < 0) return 0;
        if (status > 0) added++;
    }
    return added == wanted;
}

static int exact_policy_move(const Net *net, const State *complete,
                             int symmetries, Move *selected)
{
    State view;
    agent_information_view(complete, complete->turn, &view);
    Move moves[MAX_MOVES];
    float probability[MAX_MOVES];
    int n = policy_probs_sym(net, &view, moves, probability, NULL,
                             symmetries);
    if (n <= 0) return 0;
    int best = 0;
    for (int i = 1; i < n; i++)
        if (probability[i] > probability[best]) best = i;
    *selected = moves[best];
    return 1;
}

static void shuffle_deck(uint8_t deck[NCARD], Rng *rng)
{
    for (int i = 0; i < NCARD; i++) deck[i] = (uint8_t)i;
    for (int i = NCARD - 1; i > 0; i--) {
        uint32_t j = rng_below(rng, (uint32_t)i + 1);
        uint8_t swap = deck[i]; deck[i] = deck[j]; deck[j] = swap;
    }
}

/* Complete the current deal and both future deals.  Every decision is
 * freshly sanitized to its mover and ranked by the exact policy ensemble.
 * A nonempty deck at `over` is the engine fuse rather than a rule ending and
 * invalidates the entire diagnostic instead of supplying a fake value. */
static int finish_remaining_match(
    State *root, int perspective,
    const uint8_t future[MATCH_ROUNDS][NCARD], const Net *net,
    int symmetries, uint64_t branch_domain, FullMatchValue *out)
{
    int cumulative[2] = { root->cum[0], root->cum[1] };
    for (int round = root->round; round < MATCH_ROUNDS; round++) {
        State state;
        if (round == root->round) {
            state = *root;
        } else {
            lc_deal_from_deck(&state, future[round]);
            state.round = (uint8_t)round;
            state.turn = (uint8_t)(round & 1);
            state.cum[0] = (int16_t)cumulative[0];
            state.cum[1] = (int16_t)cumulative[1];
        }
        while (!state.over) {
            /* The policy is deterministic at temperature/epsilon zero, but
             * retain an explicit branch-neutral domain in the evidence model
             * so a future stochastic teacher cannot accidentally couple to
             * candidate ordering. */
            uint64_t decision_domain = mix64(
                branch_domain ^ ((uint64_t)(unsigned)round << 48) ^
                ((uint64_t)state.nply << 8) ^ (uint64_t)state.turn);
            (void)decision_domain;
            Move move;
            if (!exact_policy_move(net, &state, symmetries, &move)) return 0;
            lc_apply(&state, move);
        }
        if (state.deck_left > 0) return 0;
        cumulative[0] += lc_score(&state, 0);
        cumulative[1] += lc_score(&state, 1);
    }
    int margin = cumulative[perspective] - cumulative[perspective ^ 1];
    int result = (margin > 0) - (margin < 0);
    out->margin = (double)margin;
    out->match_score = result > 0 ? 1.0 : (result == 0 ? 0.5 : 0.0);
    out->hybrid = 0.05 * (double)margin + 50.0 * (double)result;
    return 1;
}

static MetricSummary summarize_metric(const double *candidate,
                                      const double *reference, int worlds)
{
    MetricSummary out = { 0 };
    double reference_mean = 0.0;
    for (int w = 0; w < worlds; w++) {
        out.mean += candidate[w];
        reference_mean += reference[w];
    }
    out.mean /= worlds;
    reference_mean /= worlds;
    out.delta = out.mean - reference_mean;
    double variance = 0.0, paired_variance = 0.0;
    for (int w = 0; w < worlds; w++) {
        double error = candidate[w] - out.mean;
        double paired = candidate[w] - reference[w] - out.delta;
        variance += error * error;
        paired_variance += paired * paired;
    }
    variance /= worlds - 1;
    paired_variance /= worlds - 1;
    out.se = sqrt(variance / worlds);
    out.delta_se = candidate == reference
        ? 0.0 : sqrt(paired_variance / worlds);
    return out;
}

static void print_metric(const char *name, MetricSummary value,
                         const double *samples, int worlds)
{
    uint64_t samples_hash = hash_extend(
        hash_init(), samples, sizeof(double) * (size_t)worlds);
    json_string(name);
    printf(":{\"mean\":%.12g,\"se\":%.12g,"
           "\"delta_vs_reference\":%.12g,\"delta_se\":%.12g,"
           "\"samples_fnv1a64\":\"%016llx\"}",
           value.mean, value.se, value.delta, value.delta_se,
           (unsigned long long)samples_hash);
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
    int ncandidate_text = 0, worlds = 1024, symmetries = 20, belief = 0;
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
                   ncandidate_text < MAXC) {
            candidate_text[ncandidate_text++] = argv[++i];
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
    int ncandidate = 0;
    for (int i = 0; i < ncandidate_text; i++) {
        Move parsed;
        if (!parse_move(&state, candidate_text[i], &parsed)) {
            fprintf(stderr, "commented_ply_eval: illegal candidate %s\n",
                    candidate_text[i]);
            spec_release(&actor);
            free(net);
            return 1;
        }
        if (add_semantic_candidate(candidate, &ncandidate, parsed) < 0) {
            fprintf(stderr, "commented_ply_eval: too many candidates\n");
            spec_release(&actor);
            free(net);
            return 1;
        }
    }
    const int nominated_candidates = ncandidate;
    int policy_reference_candidates = 0;
    if (ncandidate == 0) {
        if (!add_policy_top_distinct(net, &information_view, symmetries,
                                     candidate, &ncandidate, 2)) {
            fprintf(stderr, "commented_ply_eval: cannot form neutral policy "
                            "pair\n");
            spec_release(&actor);
            free(net);
            return 1;
        }
        policy_reference_candidates = ncandidate;
    }
    int actor_selected_included = 0;
    for (int i = 0; i < ncandidate; i++)
        actor_selected_included |=
            same_semantic_move(candidate[i], selected);
    int actor_selected_appended = 0;
    if (!actor_selected_included) {
        if (add_semantic_candidate(candidate, &ncandidate, selected) < 0) {
            fprintf(stderr, "commented_ply_eval: no room for actor selection\n");
            spec_release(&actor);
            free(net);
            return 1;
        }
        actor_selected_included = 1;
        actor_selected_appended = 1;
    }

    const size_t cells = (size_t)ncandidate * (size_t)worlds;
    double *margin = (double *)malloc(sizeof(double) * cells);
    double *match_score = (double *)malloc(sizeof(double) * cells);
    double *hybrid = (double *)malloc(sizeof(double) * cells);
    if (!margin || !match_score || !hybrid) {
        fprintf(stderr, "commented_ply_eval: out of memory\n");
        free(margin); free(match_score); free(hybrid);
        spec_release(&actor);
        free(net);
        return 1;
    }
    uint64_t hidden_world_set_hash = hash_init();
    uint64_t future_deal_set_hash = hash_init();
    uint64_t branch_rng_domain_hash = hash_init();
    for (int world_index = 0; world_index < worlds; world_index++) {
        Rng world_rng;
        uint64_t world_seed = domain_seed(
            seed, (uint64_t)world_index, (uint64_t)state.round,
            UINT64_C(0x574f524c44534554));
        rng_seed(&world_rng, world_seed);
        State world;
        determinize(&information_view, state.turn, &world_rng, &world);
        hidden_world_set_hash = hash_extend(
            hidden_world_set_hash, &world, sizeof world);

        uint8_t future[MATCH_ROUNDS][NCARD];
        memset(future, 0, sizeof future);
        for (int round = state.round + 1; round < MATCH_ROUNDS; round++) {
            Rng future_rng;
            uint64_t future_seed = domain_seed(
                seed, (uint64_t)world_index, (uint64_t)round,
                UINT64_C(0x465554555245444c));
            rng_seed(&future_rng, future_seed);
            shuffle_deck(future[round], &future_rng);
            future_deal_set_hash = hash_extend(
                future_deal_set_hash, future[round], NCARD);
        }
        uint64_t branch_domain = domain_seed(
            seed, (uint64_t)world_index, (uint64_t)state.round,
            UINT64_C(0x4252414e4348524e));
        branch_rng_domain_hash = hash_extend(
            branch_rng_domain_hash, &branch_domain, sizeof branch_domain);

        for (int c = 0; c < ncandidate; c++) {
            State branch = world;
            lc_apply(&branch, candidate[c]);
            FullMatchValue value;
            if (!finish_remaining_match(
                    &branch, state.turn, future, net, symmetries,
                    branch_domain, &value)) {
                fprintf(stderr, "commented_ply_eval: counterfactual hit "
                                "LC_MAX_PLIES before a rule ending at world "
                                "%d candidate %d\n", world_index, c);
                free(margin); free(match_score); free(hybrid);
                spec_release(&actor);
                free(net);
                return 1;
            }
            size_t index = (size_t)c * (size_t)worlds +
                           (size_t)world_index;
            margin[index] = value.margin;
            match_score[index] = value.match_score;
            hybrid[index] = value.hybrid;
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
    printf("\"schema\":\"lc-commented-ply-eval-v2\",");
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
    printf("\"seed\":\"%llu\",\"requested_worlds\":%d,"
           "\"completed_worlds\":%d,\"cap_hits\":0,",
           (unsigned long long)seed, worlds, worlds);
    printf("\"world_model\":\"uniform_exact_card_count_plus_future_deals\","
           "\"hash_algorithm\":\"fnv1a64\","
           "\"shared_current_hidden_worlds\":true,"
           "\"shared_future_deals\":true,"
           "\"branch_neutral_rng_domains\":true,");
    printf("\"hidden_world_set_hash\":\"%016llx\","
           "\"future_deal_set_hash\":\"%016llx\","
           "\"branch_rng_domain_hash\":\"%016llx\",",
           (unsigned long long)hidden_world_set_hash,
           (unsigned long long)future_deal_set_hash,
           (unsigned long long)branch_rng_domain_hash);
    printf("\"root_information_view\":true,"
           "\"policy_probability_aggregation\":"
           "\"sum_by_semantic_move\","
           "\"nominated_candidates\":%d,"
           "\"policy_reference_candidates\":%d,"
           "\"actor_selected_included\":%s,"
           "\"actor_selected_appended\":%s,",
           nominated_candidates, policy_reference_candidates,
           actor_selected_included ? "true" : "false",
           actor_selected_appended ? "true" : "false");
    printf("\"continuation\":{"
           "\"kind\":\"exact_policy_argmax\","
           "\"scope\":\"full_remaining_three_round_match\","
           "\"checkpoint\":");
    json_string(net_path);
    printf(",\"temperature\":0,\"epsilon\":0,"
           "\"symmetries\":%d,\"exact_group_average\":true,"
           "\"fresh_information_view_each_node\":true,"
           "\"recursive_actor\":false},\"candidates\":[", symmetries);
    for (int c = 0; c < ncandidate; c++) {
        size_t offset = (size_t)c * (size_t)worlds;
        MetricSummary score_summary = summarize_metric(
            match_score + offset, match_score, worlds);
        MetricSummary margin_summary = summarize_metric(
            margin + offset, margin, worlds);
        MetricSummary hybrid_summary = summarize_metric(
            hybrid + offset, hybrid, worlds);
        char canonical[24];
        move_text(candidate[c], canonical);
        if (c) putchar(',');
        printf("{");
        printf("\"move\":"); json_string(canonical);
        printf(",\"semantic_key\":%u,\"policy_prior\":%.9g,"
               "\"completed_worlds\":%d,\"cap_hits\":0,\"metrics\":{",
               semantic_pack(candidate[c]),
               move_prior(net, &information_view, candidate[c], symmetries),
               worlds);
        print_metric("match_score", score_summary, match_score + offset,
                     worlds);
        putchar(',');
        print_metric("final_margin", margin_summary, margin + offset,
                     worlds);
        putchar(',');
        print_metric("hybrid", hybrid_summary, hybrid + offset, worlds);
        printf("}}");
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

    free(margin);
    free(match_score);
    free(hybrid);
    spec_release(&actor);
    free(net);
    return have_belief || !belief ? 0 : 1;
}
