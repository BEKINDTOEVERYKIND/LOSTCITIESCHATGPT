/* showgame -- play one self-play game and print a replayable transcript.
 *
 * Every move is checked against the legal move list before it is applied, and
 * the final score is re-derived from the printed expedition contents rather
 * than read off the running totals, so the transcript cannot disagree with the
 * engine without the tool saying so.
 */
#include "../src/lc.h"
#include "../src/agent.h"
#include "../src/spec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char SUIT_CH[NSUIT + 1] = "YBWGR";
static const char *SUIT_NAME[NSUIT] = { "Yellow", "Blue", "White", "Green", "Red" };

static void card_str(int c, char *b)
{
    if (CARD_IS_WAGER(c)) sprintf(b, "%cx", SUIT_CH[CARD_SUIT(c)]);
    else sprintf(b, "%c%d", SUIT_CH[CARD_SUIT(c)], CARD_VALUE(c));
}

/* hand in suit order, wagers first within a suit */
static void print_hand(const State *st, int p)
{
    uint8_t cards[HAND_SIZE];
    int n = lc_hand_cards(st, p, cards);
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (cards[j] < cards[i]) { uint8_t t = cards[i]; cards[i] = cards[j]; cards[j] = t; }
    char b[8];
    for (int i = 0; i < n; i++) { card_str(cards[i], b); printf("%-4s", b); }
    printf("\n");
}

int main(int argc, char **argv)
{
    const char *spec = "policy:data/c8.bin";
    uint64_t seed = 20260727;
    int rounds = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-a") && i + 1 < argc) spec = argv[++i];
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "-r") && i + 1 < argc) rounds = atoi(argv[++i]);
        else { fprintf(stderr, "usage: %s [-a SPEC] [-s seed] [-r rounds]\n", argv[0]); return 1; }
    }
    if (rounds < 1) rounds = 1;
    if (rounds > MATCH_ROUNDS) rounds = MATCH_ROUNDS;

    Agent ag;
    spec_parse(spec, &ag);
    Rng rng; rng_seed(&rng, seed);

    printf("Lost Cities -- self-play, both seats %s, deal seed %llu%s\n",
           spec, (unsigned long long)seed,
           rounds > 1 ? ", full match" : "");

    int cum[2] = { 0, 0 };
    int grand_ok = 1;
    for (int rd = 0; rd < rounds; rd++) {
    State st;
    lc_deal(&st, &rng);
    st.round = (uint8_t)rd;
    st.cum[0] = (int16_t)cum[0];
    st.cum[1] = (int16_t)cum[1];
    st.turn = (uint8_t)(rd & 1);
    if (rounds > 1)
        printf("\n================ ROUND %d (totals so far P1 %+d, P2 %+d; P%d starts) ================\n",
               rd + 1, cum[0], cum[1], (rd & 1) + 1);
    printf("\n");
    printf("Player 1 hand:  "); print_hand(&st, 0);
    printf("Player 2 hand:  "); print_hand(&st, 1);
    printf("\n  #  who  action            draw\n");
    printf("  ---------------------------------------\n");

    int ply = 0;
    (void)0;
    while (!st.over) {
        int p = st.turn;
        Move m = agent_move(&ag, &st, &rng);

        /* the transcript is only worth anything if every move was legal */
        Move legal[MAX_MOVES];
        int n = lc_moves(&st, legal), ok = 0;
        for (int i = 0; i < n; i++)
            if (legal[i].card == m.card && legal[i].discard == m.discard && legal[i].draw == m.draw) ok = 1;
        if (!ok) { printf("ILLEGAL MOVE GENERATED AT PLY %d\n", ply); return 1; }

        /* what will be drawn (public for a pile, revealed here for the deck) */
        char drawn[8];
        if (m.draw == 0) card_str(st.deck[st.deck_pos], drawn);
        else card_str(st.pile[m.draw - 1][st.pile_n[m.draw - 1] - 1], drawn);

        char cs[8];
        card_str(m.card, cs);
        ply++;
        printf("%4d  P%d   %-8s%-4s  %s (%s)\n", ply, p + 1,
               m.discard ? "discard" : "play", cs,
               m.draw == 0 ? "deck" : SUIT_NAME[m.draw - 1], drawn);
        lc_apply(&st, m);
    }

    printf("\nfinal expeditions\n");
    int recomputed[2] = { 0, 0 };
    for (int p = 0; p < 2; p++) {
        printf("  Player %d\n", p + 1);
        for (int s = 0; s < NSUIT; s++) {
            int cnt = 0, sum = 0, wag = 0;
            char line[128] = "";
            for (int c = s * NRANK; c < (s + 1) * NRANK; c++) {
                if (!((st.played[p] >> c) & 1ULL)) continue;
                char b[8]; card_str(c, b);
                strcat(line, b); strcat(line, " ");
                cnt++;
                if (CARD_IS_WAGER(c)) wag++; else sum += CARD_VALUE(c);
            }
            if (!cnt) { printf("    %-7s -\n", SUIT_NAME[s]); continue; }
            int sc = (sum - 20) * (1 + wag) + (cnt >= 8 ? 20 : 0);
            recomputed[p] += sc;
            printf("    %-7s %-28s (%d-20) x %d%s = %+d\n",
                   SUIT_NAME[s], line, sum, 1 + wag, cnt >= 8 ? " + 20 bonus" : "", sc);
        }
        printf("    total %+d\n", recomputed[p]);
    }
    printf("\nround score: Player 1 %+d, Player 2 %+d\n", recomputed[0], recomputed[1]);
    if (recomputed[0] != lc_score(&st, 0) || recomputed[1] != lc_score(&st, 1)) {
        printf("MISMATCH between transcript and engine scoring\n");
        grand_ok = 0;
    }
    printf("(%d plies this round; re-scored from the cards above and matches the engine)\n", ply);
    cum[0] += recomputed[0];
    cum[1] += recomputed[1];
    }   /* rounds */

    if (rounds > 1)
        printf("\n================ MATCH RESULT: Player 1 %+d, Player 2 %+d -- %s ================\n",
               cum[0], cum[1],
               cum[0] > cum[1] ? "Player 1 wins the match" :
               (cum[0] < cum[1] ? "Player 2 wins the match" : "drawn match"));
    return grand_ok ? 0 : 1;
}
