/*
 * history_belief -- actor-aware hidden-hand inference from a scrubbed view.
 *
 * This is deliberately a policy-prefix tool, not a generic omniscient replay.
 * Its stdin contract contains only:
 *   - the observer's initial hand,
 *   - public actions up to the target position, and
 *   - identities of deck cards drawn by the observer.
 *
 * An opponent deck draw has no identity in the input.  Candidate deals are
 * sampled conditional on the observer's legitimate private information, then
 * retained only when the frozen, deterministic symmetry-ensemble policy
 * reproduces every observed opponent action.  The result is a Monte Carlo
 * posterior for the known AI actor, suitable for offline analysis of early
 * positions.  Search-time actions are intentionally outside this contract:
 * integrating over rollout RNG is a different inference problem.
 */
#include "../src/agent.h"
#include "../src/lc.h"
#include "../src/net.h"
#include <errno.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define VIEW_MAGIC "LCBH1"
#define MAX_VIEW_EVENTS 20
#define SEM_VALUE_SLOTS 11

typedef struct {
    int actor;
    int suit;
    int value;
    int discard;
    int draw;
    int known_suit;
    int known_value;
    int known_card;
} PublicEvent;

typedef struct {
    int observer;
    int round;
    int start_turn;
    int cum[2];
    int nevent;
    int own_n;
    int own_suit[HAND_SIZE];
    int own_value[HAND_SIZE];
    PublicEvent event[MAX_VIEW_EVENTS];
} View;

static void fail(const char *message)
{
    fprintf(stderr, "history_belief: %s\n", message);
    exit(EXIT_FAILURE);
}

static int valid_semantic(int suit, int value)
{
    return suit >= 0 && suit < NSUIT
        && (value == 0 || (value >= 2 && value <= 10));
}

static int semantic_match(int card, int suit, int value)
{
    if (CARD_SUIT(card) != suit) return 0;
    return value == 0 ? CARD_IS_WAGER(card)
                      : (!CARD_IS_WAGER(card) && CARD_VALUE(card) == value);
}

/* Assign an arbitrary physical ID to an observer-known semantic card.  The
 * maintained champion ties all three wager copies exactly; the input and
 * output remain semantic, so this arbitrary representative is unobservable. */
static int allocate_semantic(int suit, int value, uint64_t used)
{
    if (!valid_semantic(suit, value)) return -1;
    if (value != 0) {
        int card = CARD_MAKE(suit, value + 1);
        return ((used >> card) & 1ULL) ? -1 : card;
    }
    for (int rank = 0; rank < WAGERS_PER_SUIT; rank++) {
        int card = CARD_MAKE(suit, rank);
        if (!((used >> card) & 1ULL)) return card;
    }
    return -1;
}

static void read_view(View *view)
{
    char magic[16];
    memset(view, 0, sizeof *view);
    if (scanf("%15s", magic) != 1 || strcmp(magic, VIEW_MAGIC) != 0)
        fail("stdin is not an LCBH1 perspective view");
    if (scanf("%d%d%d%d%d%d",
              &view->observer, &view->round, &view->start_turn,
              &view->cum[0], &view->cum[1], &view->nevent) != 6)
        fail("short perspective-view header");
    if (view->observer < 0 || view->observer > 1)
        fail("observer must be 0 or 1");
    if (view->round < 0 || view->round >= MATCH_ROUNDS)
        fail("round is outside the supported match");
    if (view->start_turn < 0 || view->start_turn > 1)
        fail("start_turn must be 0 or 1");
    if (view->cum[0] < INT16_MIN || view->cum[0] > INT16_MAX ||
        view->cum[1] < INT16_MIN || view->cum[1] > INT16_MAX)
        fail("cumulative score does not fit State");
    if (view->nevent < 0 || view->nevent > MAX_VIEW_EVENTS)
        fail("only policy-prefix views of at most 20 actions are supported");

    if (scanf("%d", &view->own_n) != 1 || view->own_n != HAND_SIZE)
        fail("own initial hand must contain exactly eight cards");
    for (int i = 0; i < view->own_n; i++) {
        if (scanf("%d%d", &view->own_suit[i], &view->own_value[i]) != 2 ||
            !valid_semantic(view->own_suit[i], view->own_value[i]))
            fail("invalid card in own initial hand");
    }

    int expected_actor = view->start_turn;
    for (int i = 0; i < view->nevent; i++) {
        PublicEvent *e = &view->event[i];
        if (scanf("%d%d%d%d%d%d%d",
                  &e->actor, &e->suit, &e->value, &e->discard, &e->draw,
                  &e->known_suit, &e->known_value) != 7)
            fail("short public event");
        e->known_card = -1;
        if (e->actor != expected_actor)
            fail("event actors do not alternate from start_turn");
        expected_actor ^= 1;
        if (!valid_semantic(e->suit, e->value))
            fail("invalid public action card");
        if ((e->discard != 0 && e->discard != 1) ||
            e->draw < 0 || e->draw > NSUIT)
            fail("invalid public action disposition or draw source");

        if (e->draw == 0 && e->actor == view->observer) {
            if (!valid_semantic(e->known_suit, e->known_value))
                fail("observer deck draw must include its known card");
        } else if (e->known_suit != -1 || e->known_value != -1) {
            fail("opponent deck-draw identities are forbidden");
        }
    }

    char trailing[2];
    if (scanf("%1s", trailing) == 1)
        fail("trailing data after perspective view");
}

static int observed_move(const State *st, const PublicEvent *event, Move *out)
{
    Move moves[MAX_MOVES];
    int n = lc_moves(st, moves);
    for (int i = 0; i < n; i++) {
        if (semantic_match(moves[i].card, event->suit, event->value) &&
            moves[i].discard == event->discard &&
            moves[i].draw == event->draw) {
            *out = moves[i];
            return 1;
        }
    }
    return 0;
}

static Move policy_argmax(const Net *net, const State *st, int symmetries)
{
    Move moves[MAX_MOVES];
    float probability[MAX_MOVES];
    int n = policy_probs_sym(net, st, moves, probability, NULL, symmetries);
    if (n <= 0) fail("policy encountered a state with no legal move");
    int best = 0;
    for (int i = 1; i < n; i++)
        if (probability[i] > probability[best]) best = i;
    return moves[best];
}

static int same_semantic_move(Move move, const PublicEvent *event)
{
    return semantic_match(move.card, event->suit, event->value)
        && move.discard == event->discard
        && move.draw == event->draw;
}

static void shuffle(uint8_t *cards, int n, Rng *rng)
{
    for (int i = n - 1; i > 0; i--) {
        uint32_t j = rng_below(rng, (uint32_t)i + 1);
        uint8_t tmp = cards[i];
        cards[i] = cards[j];
        cards[j] = tmp;
    }
}

static int supported_symmetries(int n)
{
    return n == 1 || n == 5 || n == 10 || n == 20 || n == 120;
}

int main(int argc, char **argv)
{
    const char *net_path = "data/champion.bin";
    long worlds = 20000;
    uint64_t seed = UINT64_C(11506497556975);
    int symmetries = 20;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) {
            net_path = argv[++i];
        } else if (!strcmp(argv[i], "-w") && i + 1 < argc) {
            char *end = NULL;
            errno = 0;
            worlds = strtol(argv[++i], &end, 10);
            if (errno || !end || *end) fail("invalid world count");
        } else if (!strcmp(argv[i], "-s") && i + 1 < argc) {
            char *end = NULL;
            errno = 0;
            const char *text = argv[++i];
            if (text[0] == '-') fail("invalid annotation seed");
            seed = strtoull(text, &end, 10);
            if (errno || !end || end == text || *end)
                fail("invalid annotation seed");
        } else if (!strcmp(argv[i], "-y") && i + 1 < argc) {
            symmetries = atoi(argv[++i]);
        } else {
            fprintf(stderr,
                    "usage: %s [-n NET] [-w WORLDS] [-s SEED] "
                    "[-y SYMMETRIES] < scrubbed-view\n", argv[0]);
            return EXIT_FAILURE;
        }
    }
    if (worlds < 1 || worlds > 10000000L)
        fail("world count must be between 1 and 10,000,000");
    if (!supported_symmetries(symmetries))
        fail("symmetries must be 1, 5, 10, 20, or 120");

    View view;
    read_view(&view);

    uint64_t private_mask = 0;
    int own_card[HAND_SIZE];
    for (int i = 0; i < HAND_SIZE; i++) {
        own_card[i] = allocate_semantic(view.own_suit[i],
                                        view.own_value[i], private_mask);
        if (own_card[i] < 0)
            fail("own initial hand contains too many copies of a card");
        private_mask |= 1ULL << own_card[i];
    }

    int forced[NCARD];
    for (int i = 0; i < NCARD; i++) forced[i] = -1;
    int deck_draws = 0;
    for (int i = 0; i < view.nevent; i++) {
        PublicEvent *event = &view.event[i];
        if (event->draw != 0) continue;
        int position = 2 * HAND_SIZE + deck_draws++;
        if (position >= NCARD) fail("too many deck draws in view");
        if (event->actor != view.observer) continue;
        int card = allocate_semantic(event->known_suit,
                                     event->known_value, private_mask);
        if (card < 0)
            fail("observer draw repeats a card already in its private record");
        private_mask |= 1ULL << card;
        event->known_card = card;
        forced[position] = card;
    }

    Net *net = malloc(sizeof *net);
    if (!net) fail("out of memory allocating network");
    if (net_load(net, net_path) != 0)
        fail("could not load network");

    double sum[NSUIT][SEM_VALUE_SLOTS] = {{ 0 }};
    double sumsq[NSUIT][SEM_VALUE_SLOTS] = {{ 0 }};
    long accepted = 0;
    Rng rng;
    rng_seed(&rng, seed);

    for (long world = 0; world < worlds; world++) {
        uint8_t pool[NCARD];
        int npool = 0;
        for (int card = 0; card < NCARD; card++)
            if (!((private_mask >> card) & 1ULL))
                pool[npool++] = (uint8_t)card;
        shuffle(pool, npool, &rng);

        uint8_t deck[NCARD];
        memset(deck, 0xff, sizeof deck);
        int cursor = 0;
        int observer_start = view.observer * HAND_SIZE;
        int opponent_start = (view.observer ^ 1) * HAND_SIZE;
        for (int i = 0; i < HAND_SIZE; i++) {
            deck[observer_start + i] = (uint8_t)own_card[i];
            deck[opponent_start + i] = pool[cursor++];
        }
        for (int position = 2 * HAND_SIZE; position < NCARD; position++) {
            if (forced[position] >= 0)
                deck[position] = (uint8_t)forced[position];
            else
                deck[position] = pool[cursor++];
        }
        if (cursor != npool) fail("internal conditional-deal accounting error");

        State state;
        lc_deal_from_deck(&state, deck);
        state.turn = (uint8_t)view.start_turn;
        state.round = (uint8_t)view.round;
        state.cum[0] = (int16_t)view.cum[0];
        state.cum[1] = (int16_t)view.cum[1];

        int keep = 1;
        for (int i = 0; i < view.nevent; i++) {
            PublicEvent *event = &view.event[i];
            if (state.turn != event->actor) {
                keep = 0;
                break;
            }
            Move move;
            if (event->actor == view.observer) {
                if (!observed_move(&state, event, &move)) {
                    keep = 0;
                    break;
                }
            } else {
                move = policy_argmax(net, &state, symmetries);
                if (!same_semantic_move(move, event)) {
                    keep = 0;
                    break;
                }
            }

            if (event->known_card >= 0 &&
                (state.deck_pos >= NCARD ||
                 state.deck[state.deck_pos] != event->known_card)) {
                fail("internal observer-draw conditioning error");
            }
            lc_apply(&state, move);
        }
        if (!keep) continue;

        int opponent = view.observer ^ 1;
        if (__builtin_popcountll(state.hand[opponent]) != HAND_SIZE)
            fail("accepted world has wrong opponent hand size");
        if ((state.hand[opponent] & state.known[opponent])
            != state.known[opponent])
            fail("accepted world dropped a publicly known opponent card");

        accepted++;
        int count[NSUIT][SEM_VALUE_SLOTS] = {{ 0 }};
        uint64_t hand = state.hand[opponent];
        while (hand) {
            int card = __builtin_ctzll(hand);
            hand &= hand - 1;
            int suit = CARD_SUIT(card);
            int value = CARD_IS_WAGER(card) ? 0 : CARD_VALUE(card);
            count[suit][value]++;
        }
        for (int suit = 0; suit < NSUIT; suit++) {
            for (int value = 0; value < SEM_VALUE_SLOTS; value++) {
                if (value == 1) continue;
                double x = count[suit][value];
                sum[suit][value] += x;
                sumsq[suit][value] += x * x;
            }
        }
    }

    free(net);
    if (accepted == 0) fail("no sampled world reproduced the public prefix");

    double marginal_sum = 0.0;
    for (int suit = 0; suit < NSUIT; suit++)
        for (int value = 0; value < SEM_VALUE_SLOTS; value++)
            if (value != 1)
                marginal_sum += sum[suit][value] / (double)accepted;

    printf("{\"schema\":\"lc-history-belief-worker-v1\","
           "\"method\":\"actor_aware_rejection\","
           "\"scope\":\"early_policy_prefix\","
           "\"worlds\":%ld,\"accepted\":%ld,\"accepted_rate\":%.9f,"
           "\"seed\":%llu,\"symmetries\":%d,"
           "\"opponent_hand_size\":%d,\"marginal_sum\":%.9f,"
           "\"cards\":[",
           worlds, accepted, (double)accepted / (double)worlds,
           (unsigned long long)seed, symmetries, HAND_SIZE, marginal_sum);

    int first = 1;
    for (int suit = 0; suit < NSUIT; suit++) {
        for (int value = 0; value < SEM_VALUE_SLOTS; value++) {
            if (value == 1) continue;
            double mean = sum[suit][value] / (double)accepted;
            double variance = 0.0;
            if (accepted > 1) {
                double centered = sumsq[suit][value]
                                - sum[suit][value] * sum[suit][value]
                                  / (double)accepted;
                if (centered < 0.0 && centered > -1e-9) centered = 0.0;
                variance = centered / (double)(accepted - 1);
            }
            double se = sqrt(variance / (double)accepted);
            double maximum = value == 0 ? WAGERS_PER_SUIT : 1.0;
            double lo = mean - 1.96 * se;
            double hi = mean + 1.96 * se;
            if (lo < 0.0) lo = 0.0;
            if (hi > maximum) hi = maximum;
            printf("%s{\"suit\":%d,\"value\":%d,"
                   "\"expected_count\":%.9f,\"se\":%.9f,"
                   "\"ci95\":[%.9f,%.9f]}",
                   first ? "" : ",", suit, value, mean, se, lo, hi);
            first = 0;
        }
    }
    printf("]}\n");
    return EXIT_SUCCESS;
}
