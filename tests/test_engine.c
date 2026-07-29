/* Engine invariant and rule tests. */
#include "../src/lc.h"
#include <stdio.h>
#include <stdlib.h>

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { \
    printf("FAIL %s:%d: ", __FILE__, __LINE__); printf(__VA_ARGS__); printf("\n"); failures++; } } while (0)

static uint16_t semantic_move_key(Move m)
{
    if (CARD_IS_WAGER(m.card))
        m.card = (uint8_t)CARD_MAKE(CARD_SUIT(m.card), 0);
    return MOVE_PACK(m);
}

static void test_cards(void)
{
    int nwager = 0, nnum = 0, sum = 0;
    int seen[NCARD]; memset(seen, 0, sizeof seen);
    for (int c = 0; c < NCARD; c++) {
        CHECK(CARD_SUIT(c) >= 0 && CARD_SUIT(c) < NSUIT, "suit range");
        if (CARD_IS_WAGER(c)) { nwager++; CHECK(CARD_VALUE(c) == 0, "wager value"); }
        else { nnum++; sum += CARD_VALUE(c);
               CHECK(CARD_VALUE(c) >= 2 && CARD_VALUE(c) <= 10, "card value %d", CARD_VALUE(c)); }
        seen[c] = 1;
    }
    CHECK(nwager == 15, "15 wagers, got %d", nwager);
    CHECK(nnum == 45, "45 number cards, got %d", nnum);
    CHECK(sum == 5 * 54, "value sum %d", sum);
    /* every (suit,value) appears exactly once */
    for (int s = 0; s < NSUIT; s++)
        for (int v = 2; v <= 10; v++) {
            int cnt = 0;
            for (int c = 0; c < NCARD; c++)
                if (CARD_SUIT(c) == s && !CARD_IS_WAGER(c) && CARD_VALUE(c) == v) cnt++;
            CHECK(cnt == 1, "suit %d value %d count %d", s, v, cnt);
        }
}

static void test_scoring(void)
{
    State st; memset(&st, 0, sizeof st);
    /* empty expedition scores 0 */
    CHECK(lc_score(&st, 0) == 0, "empty score");

    /* single 2 -> 2-20 = -18 */
    st.exp_n[0][0] = 1; st.exp_sum[0][0] = 2;
    CHECK(lc_exp_score(&st, 0, 0) == -18, "single 2 => %d", lc_exp_score(&st, 0, 0));

    /* one wager and a 10 -> (10-20)*2 = -20 */
    memset(&st, 0, sizeof st);
    st.exp_n[0][1] = 2; st.exp_sum[0][1] = 10; st.exp_wager[0][1] = 1;
    CHECK(lc_exp_score(&st, 0, 1) == -20, "wager+10 => %d", lc_exp_score(&st, 0, 1));

    /* full suit: 3 wagers + 2..10 = 12 cards, sum 54 -> (54-20)*4 + 20 = 156 */
    memset(&st, 0, sizeof st);
    st.exp_n[0][2] = 12; st.exp_sum[0][2] = 54; st.exp_wager[0][2] = 3;
    CHECK(lc_exp_score(&st, 0, 2) == 156, "max expedition => %d", lc_exp_score(&st, 0, 2));

    /* 8 card bonus boundary */
    memset(&st, 0, sizeof st);
    st.exp_n[0][3] = 8; st.exp_sum[0][3] = 20;
    CHECK(lc_exp_score(&st, 0, 3) == 20, "8 cards sum 20 => %d", lc_exp_score(&st, 0, 3));
    st.exp_n[0][3] = 7;
    CHECK(lc_exp_score(&st, 0, 3) == 0, "7 cards sum 20 => %d", lc_exp_score(&st, 0, 3));
}

static void check_state(const State *st, int line)
{
    uint64_t all = 0;
    int overlap = 0;
    uint64_t parts[5] = { st->hand[0], st->hand[1], st->played[0], st->played[1], st->discarded };
    for (int i = 0; i < 5; i++) {
        if (all & parts[i]) overlap = 1;
        all |= parts[i];
    }
    for (int i = st->deck_pos; i < NCARD; i++) {
        uint64_t b = 1ULL << st->deck[i];
        if (all & b) overlap = 1;
        all |= b;
    }
    if (overlap) { printf("FAIL line %d: card in two places\n", line); failures++; }
    if (all != ((1ULL << NCARD) - 1)) { printf("FAIL line %d: cards lost (mask %llx)\n", line, (unsigned long long)all); failures++; }
    if (__builtin_popcountll(st->hand[0]) != st->hand_n[0] ||
        __builtin_popcountll(st->hand[1]) != st->hand_n[1]) { printf("FAIL line %d: hand count\n", line); failures++; }
    if (st->deck_left != NCARD - st->deck_pos) { printf("FAIL line %d: deck_left\n", line); failures++; }

    int pilecards = 0;
    for (int s = 0; s < NSUIT; s++) pilecards += st->pile_n[s];
    if (pilecards != __builtin_popcountll(st->discarded)) { printf("FAIL line %d: pile bookkeeping\n", line); failures++; }

    /* known cards must actually sit in that player's hand */
    for (int p = 0; p < 2; p++)
        if (st->known[p] & ~st->hand[p]) { printf("FAIL line %d: known card not in hand (p%d)\n", line, p); failures++; }

    /* the unseen pool must exclude everything public and everything known */
    for (int p = 0; p < 2; p++) {
        uint8_t uns[NCARD]; int un = 0;
        lc_unseen(st, p, uns, &un);
        uint64_t umask = 0;
        for (int i = 0; i < un; i++) umask |= 1ULL << uns[i];
        uint64_t banned = st->hand[p] | st->played[0] | st->played[1] | st->discarded | st->known[p ^ 1];
        if (umask & banned) { printf("FAIL line %d: unseen overlaps visible/known (p%d)\n", line, p); failures++; }
        if (__builtin_popcountll(umask) + __builtin_popcountll(banned & ((1ULL << NCARD) - 1)) != NCARD) {
            printf("FAIL line %d: unseen count wrong (p%d)\n", line, p); failures++;
        }
    }

    for (int p = 0; p < 2; p++) {
        for (int s = 0; s < NSUIT; s++) {
            int n = 0, w = 0, sum = 0, top = 0;
            for (int c = s * NRANK; c < (s + 1) * NRANK; c++) {
                if ((st->played[p] >> c) & 1ULL) {
                    n++;
                    if (CARD_IS_WAGER(c)) w++;
                    else { sum += CARD_VALUE(c); if (CARD_VALUE(c) > top) top = CARD_VALUE(c); }
                }
            }
            if (n != st->exp_n[p][s] || w != st->exp_wager[p][s] ||
                sum != st->exp_sum[p][s] || top != st->exp_top[p][s]) {
                printf("FAIL line %d: expedition cache p%d suit%d\n", line, p, s); failures++;
            }
        }
    }
}

static void test_playouts(void)
{
    Rng rng; rng_seed(&rng, 12345);
    int total_plies = 0, games = 2000;
    long long score_sum = 0;
    int min_ply = 1000, max_ply = 0;
    for (int g = 0; g < games; g++) {
        State st;
        lc_deal(&st, &rng);
        check_state(&st, __LINE__);
        CHECK(st.hand_n[0] == 8 && st.hand_n[1] == 8, "deal hand sizes");
        CHECK(st.deck_left == 44, "deck left after deal %d", st.deck_left);
        Move mv[MAX_MOVES];
        while (!st.over) {
            int n = lc_moves(&st, mv);
            CHECK(n > 0, "no legal moves at ply %d", st.nply);
            if (n == 0) break;
            uint8_t move_seen[MOVE_NPACK] = { 0 };
            /* verify legality of a sample of moves */
            for (int i = 0; i < n; i++) {
                Move m = mv[i];
                uint16_t key = semantic_move_key(m);
                CHECK(!move_seen[key], "duplicate semantic move %u",
                      (unsigned)key);
                move_seen[key] = 1;
                CHECK((st.hand[st.turn] >> m.card) & 1ULL, "move uses card not in hand");
                if (!m.discard) {
                    int suit = CARD_SUIT(m.card);
                    if (CARD_IS_WAGER(m.card)) CHECK(st.exp_top[st.turn][suit] == 0, "late wager offered");
                    else CHECK(CARD_VALUE(m.card) > st.exp_top[st.turn][suit], "descending play offered");
                }
                if (m.draw > 0) CHECK(st.pile_n[m.draw - 1] > 0, "draw from empty pile");
                else CHECK(st.deck_left > 0, "draw from empty deck");
                if (m.discard && m.draw == CARD_SUIT(m.card) + 1) CHECK(0, "take back just discarded card");
            }
            int prev_turn = st.turn;
            Move chosen_mv = mv[rng_below(&rng, (uint32_t)n)];
            uint8_t pile_top = chosen_mv.draw > 0 ? st.pile[chosen_mv.draw - 1][st.pile_n[chosen_mv.draw - 1] - 1] : 0;
            lc_apply(&st, chosen_mv);
            if (chosen_mv.draw > 0)
                CHECK((st.known[prev_turn] >> pile_top) & 1ULL, "face-up draw must be known");
            CHECK(st.turn != prev_turn, "turn must alternate");
            CHECK(st.hand_n[prev_turn] == 8, "hand size restored");
            check_state(&st, __LINE__);
            if (g == 0 && st.nply == 20) {
                const uint8_t perm[NSUIT] = { 2, 4, 1, 0, 3 };
                uint8_t inverse[NSUIT];
                for (int s = 0; s < NSUIT; s++) inverse[perm[s]] = (uint8_t)s;
                State renamed, restored;
                lc_permute_suits(&st, &renamed, perm);
                lc_permute_suits(&renamed, &restored, inverse);
                CHECK(memcmp(&st, &restored, sizeof st) == 0,
                      "suit permutation did not round-trip");
                check_state(&renamed, __LINE__);
            }
        }
        CHECK(st.deck_left == 0, "game ended with cards left");
        total_plies += st.nply;
        if (st.nply < min_ply) min_ply = st.nply;
        if (st.nply > max_ply) max_ply = st.nply;
        score_sum += lc_score(&st, 0) + lc_score(&st, 1);
    }
    printf("playouts: %d games, avg plies %.1f (min %d max %d), avg total score %.1f\n",
           games, (double)total_plies / games, min_ply, max_ply, (double)score_sum / games);
}

/* Scripted game verifying the discard/draw restriction and turn order. */
static void test_rules_detail(void)
{
    uint8_t deck[NCARD];
    for (int i = 0; i < NCARD; i++) deck[i] = (uint8_t)i;
    State st;
    lc_deal_from_deck(&st, deck);
    /* player 0 holds cards 0..7 (suit 0: 3 wagers + 2,3,4,5,6) */
    CHECK(st.hand[0] == 0xFFULL, "p0 hand");
    CHECK(st.hand[1] == 0xFF00ULL, "p1 hand");

    Move mv[MAX_MOVES];
    int n = lc_moves(&st, mv);
    /* The three same-suit wager copies collapse to one semantic action.  Along
     * with five number cards, all six card types can be played or discarded;
     * the only draw source is the deck because all piles are empty. */
    CHECK(n == 6 + 6, "expected 12 semantic opening moves, got %d", n);

    /* discard a suit-0 card, then the suit-0 pile must not be drawable */
    Move d = { 3, 1, 0 }; /* discard card 3 = suit0 value2, draw deck */
    lc_apply(&st, d);
    CHECK(st.pile_n[0] == 1, "pile after discard");
    st.turn = 0; /* hack the turn back to inspect p0's options with a live pile */
    n = lc_moves(&st, mv);
    int saw_pile0 = 0;
    for (int i = 0; i < n; i++) if (mv[i].draw == 1) saw_pile0 = 1;
    CHECK(saw_pile0, "pile 0 drawable on a later turn");

    /* the just-discarded restriction: build a fresh state */
    lc_deal_from_deck(&st, deck);
    Move d2 = { 3, 1, 0 };
    lc_apply(&st, d2);
    /* p1 discards a suit-0 card too and must not take it back */
    uint8_t c1 = 0; /* find a suit-0 card in p1's hand */
    int found = 0;
    for (int c = 0; c < NCARD; c++)
        if (((st.hand[1] >> c) & 1ULL) && CARD_SUIT(c) == 0) { c1 = (uint8_t)c; found = 1; break; }
    if (found) {
        n = lc_moves(&st, mv);
        for (int i = 0; i < n; i++)
            CHECK(!(mv[i].card == c1 && mv[i].discard && mv[i].draw == 1),
                  "p1 offered to take back its own discard");
    }
}

static void test_known_lifecycle(void)
{
    /* discard a card, have the opponent take it, then watch the known bit
     * clear when it is finally played */
    uint8_t deck[NCARD];
    for (int i = 0; i < NCARD; i++) deck[i] = (uint8_t)i;
    State st;
    lc_deal_from_deck(&st, deck);
    /* p0 discards suit0 value2 (card 3) */
    Move d = { 3, 1, 0 };
    lc_apply(&st, d);
    CHECK(st.known[1] == 0, "nothing known yet");
    /* p1 discards a suit1 card, draws the suit0 pile top (card 3) */
    Move d2 = { 8, 1, 1 };   /* card 8 = suit0? no: 8 is suit0 rank8.  p1 holds 8..15 */
    /* p1's hand is cards 8..15: suit0 ranks 8-11 and suit1 ranks 0-3.
     * discard card 12 (suit1 wager) to pile B, draw from pile Y (suit0). */
    Move d3 = { 12, 1, 1 };
    lc_apply(&st, d3);
    (void)d2;
    CHECK((st.known[1] >> 3) & 1ULL, "p1 took card 3 face up: must be known");
    CHECK((st.hand[1] >> 3) & 1ULL, "card 3 in p1 hand");
    /* unseen for p0 must not contain card 3 */
    uint8_t uns[NCARD]; int un = 0;
    lc_unseen(&st, 0, uns, &un);
    for (int i = 0; i < un; i++) CHECK(uns[i] != 3, "known card leaked into unseen");
    /* p1 plays card 3 (suit0 value2, expedition empty): known bit clears */
    st.turn = 1;
    Move pl = { 3, 0, 0 };
    lc_apply(&st, pl);
    CHECK(!((st.known[1] >> 3) & 1ULL), "known bit must clear on play");
}

static void test_indistinguishable_wager_knowledge(void)
{
    State st;
    memset(&st, 0, sizeof st);
    st.turn = 0;
    st.hand[0] = (1ULL << 0) | (1ULL << 1) | (1ULL << 2);
    st.hand_n[0] = 3;

    /* Only wager id 1 was taken face up, but playing hidden id 0 is publicly
     * indistinguishable from playing it.  No wager can remain guaranteed. */
    st.known[0] = 1ULL << 1;
    Move m = { 0, 0, 0 };
    lc_apply_play(&st, m);
    CHECK((st.known[0] & 0x7ULL) == 0,
          "playing any identical wager must discharge one known wager");

    /* With two known wagers and one hidden wager, one guarantee remains. */
    memset(&st, 0, sizeof st);
    st.turn = 0;
    st.hand[0] = (1ULL << 0) | (1ULL << 1) | (1ULL << 2);
    st.hand_n[0] = 3;
    st.known[0] = (1ULL << 0) | (1ULL << 1);
    m.card = 2;
    lc_apply_play(&st, m);
    CHECK(__builtin_popcountll(st.known[0] & 0x7ULL) == 1,
          "one of two wager guarantees must remain");
    CHECK((st.known[0] & ~st.hand[0]) == 0,
          "known wager count must still be represented by a card in hand");
}

static void test_match_context(void)
{
    Rng rng; rng_seed(&rng, 9);
    State st;
    lc_deal(&st, &rng);
    CHECK(st.round == 0 && st.cum[0] == 0 && st.cum[1] == 0, "fresh deal has empty context");
    st.round = 2; st.cum[0] = 88; st.cum[1] = -17;
    Move mv[MAX_MOVES];
    int n = lc_moves(&st, mv);
    CHECK(n > 0, "moves exist");
    lc_apply(&st, mv[0]);
    CHECK(st.round == 2 && st.cum[0] == 88 && st.cum[1] == -17, "context survives apply");
}

static void test_dead_and_dominated(void)
{
    State st; memset(&st, 0, sizeof st);
    /* nobody has played: nothing is dead */
    st.hand[0] = (1ULL << CARD_MAKE(1, 4)) | (1ULL << CARD_MAKE(4, 0));
    CHECK(lc_dead_cards(&st) == 0, "empty boards have no dead cards");

    /* Blue tops 5 (p0) and 6 (p1): B2..B5 dead (<= both tops is <=5), B6 not
     * (p0 could still play it); one Blue number each: Blue wagers dead */
    st.exp_top[0][1] = 5; st.exp_top[1][1] = 6;
    uint64_t dead = lc_dead_cards(&st);
    CHECK((dead >> CARD_MAKE(1, 3)) & 1, "B2 dead below both tops");
    CHECK((dead >> CARD_MAKE(1, 6)) & 1, "B5 dead below both tops");
    CHECK(!((dead >> CARD_MAKE(1, 7)) & 1), "B6 alive: p0 top is 5");
    CHECK((dead >> CARD_MAKE(1, 0)) & 1, "Blue wager dead once both sides run numbers");
    CHECK(!((dead >> CARD_MAKE(4, 0)) & 1), "Red wager alive");

    /* hand: dead B4 and live Rx; discarding Rx is dominated except when the
     * draw comes from the Blue pile (B4 cannot go there and be drawn over) */
    st.hand[0] = (1ULL << CARD_MAKE(1, 5)) | (1ULL << CARD_MAKE(4, 0));
    st.turn = 0;
    Move m;
    m.card = CARD_MAKE(4, 0); m.discard = 1; m.draw = 0;
    CHECK(lc_discard_dominated(&st, m, dead), "disc live Rx (deck draw) dominated by dead B4");
    m.draw = 2; /* draw from the Blue pile */
    CHECK(!lc_discard_dominated(&st, m, dead), "disc Rx drawing Blue pile not dominated: B4 cannot replace it");
    m.card = CARD_MAKE(1, 5); m.draw = 0;
    CHECK(!lc_discard_dominated(&st, m, dead), "the dead discard itself survives");
    m.discard = 0;
    CHECK(!lc_discard_dominated(&st, m, dead), "plays are never pruned");

    /* two dead cards: only the lowest id survives */
    st.exp_top[0][4] = 9; st.exp_top[1][4] = 9;
    dead = lc_dead_cards(&st);
    st.hand[0] = (1ULL << CARD_MAKE(1, 5)) | (1ULL << CARD_MAKE(4, 6));
    m.card = CARD_MAKE(4, 6); m.discard = 1; m.draw = 0;
    CHECK(lc_discard_dominated(&st, m, dead), "dead R5 deduped against lower-id dead B4");
    m.card = CARD_MAKE(1, 5);
    CHECK(!lc_discard_dominated(&st, m, dead), "canonical dead discard kept");
}

int main(void)
{
    test_cards();
    test_scoring();
    test_rules_detail();
    test_known_lifecycle();
    test_indistinguishable_wager_knowledge();
    test_match_context();
    test_dead_and_dominated();
    test_playouts();
    if (failures == 0) { printf("all engine tests passed\n"); return 0; }
    printf("%d failures\n", failures);
    return 1;
}
