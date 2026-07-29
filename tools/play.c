/* play -- play a game of Lost Cities against a trained agent in the terminal. */
#include "../src/lc.h"
#include "../src/agent.h"
#include "../src/spec.h"
#include "../src/search.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <strings.h>
#include <time.h>

static const char *SUIT_NAME[NSUIT] = { "Yellow", "Blue", "White", "Green", "Red" };
static const char SUIT_CH[NSUIT + 1] = "YBWGR";

static void card_str(int c, char *buf)
{
    if (CARD_IS_WAGER(c)) sprintf(buf, "%cx", SUIT_CH[CARD_SUIT(c)]);
    else sprintf(buf, "%c%d", SUIT_CH[CARD_SUIT(c)], CARD_VALUE(c));
}

static void show(const State *st, int me)
{
    char b[8];
    printf("\n=========================================================\n");
    printf("deck %2d   ply %d\n", st->deck_left, st->nply);
    printf("\nopponent expeditions:\n");
    for (int s = 0; s < NSUIT; s++) {
        printf("  %-7s", SUIT_NAME[s]);
        for (int c = s * NRANK; c < (s + 1) * NRANK; c++)
            if ((st->played[me ^ 1] >> c) & 1ULL) { card_str(c, b); printf(" %s", b); }
        printf("   [%d]\n", lc_exp_score(st, me ^ 1, s));
    }
    printf("\ndiscard piles (top card last):\n");
    for (int s = 0; s < NSUIT; s++) {
        printf("  %-7s", SUIT_NAME[s]);
        for (int i = 0; i < st->pile_n[s]; i++) { card_str(st->pile[s][i], b); printf(" %s", b); }
        printf("\n");
    }
    printf("\nyour expeditions:\n");
    for (int s = 0; s < NSUIT; s++) {
        printf("  %-7s", SUIT_NAME[s]);
        for (int c = s * NRANK; c < (s + 1) * NRANK; c++)
            if ((st->played[me] >> c) & 1ULL) { card_str(c, b); printf(" %s", b); }
        printf("   [%d]\n", lc_exp_score(st, me, s));
    }
    printf("\nscore: you %d, opponent %d\n", lc_score(st, me), lc_score(st, me ^ 1));
    printf("\nyour hand:\n  ");
    uint8_t cards[HAND_SIZE];
    int n = lc_hand_cards(st, me, cards);
    for (int i = 0; i < n; i++) { card_str(cards[i], b); printf("%s ", b); }
    printf("\n");
}

int main(int argc, char **argv)
{
    const char *spec = "rollout:data/c8.bin:96:5:0.02:0.85";
    uint64_t seed = 0;
    int human_first = 1;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-a") && i + 1 < argc) spec = argv[++i];
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "-2")) human_first = 0;
        else { printf("usage: %s [-a AGENTSPEC] [-s seed] [-2]\n", argv[0]); return 1; }
    }
    Agent ai;
    spec_parse(spec, &ai);

    Rng rng;
    rng_seed(&rng, seed ? seed : (uint64_t)time(NULL));
    State st;
    lc_deal(&st, &rng);
    int me = human_first ? 0 : 1;

    printf("Lost Cities -- you are player %d, opponent is %s\n", me, spec);
    printf("Enter moves as: <card> <p|d> <deck|Y|B|W|G|R>   e.g. \"B7 p deck\"\n");

    while (!st.over) {
        if (st.turn == me) {
            show(&st, me);
            Move mv[MAX_MOVES];
            int n = lc_moves(&st, mv);
            char line[128];
            Move chosen;
            int ok = 0;
            while (!ok) {
                printf("> ");
                fflush(stdout);
                if (!fgets(line, sizeof line, stdin)) { printf("\n"); return 0; }
                char cs[16], ds[16], ws[16];
                if (sscanf(line, "%15s %15s %15s", cs, ds, ws) != 3) { printf("need three fields\n"); continue; }
                int card = -1;
                for (int c = 0; c < NCARD; c++) {
                    char b[8]; card_str(c, b);
                    if (!strcasecmp(b, cs) && ((st.hand[me] >> c) & 1ULL)) { card = c; break; }
                }
                if (card < 0) { printf("no such card in hand\n"); continue; }
                int disc = (ds[0] == 'd' || ds[0] == 'D');
                int draw = -1;
                if (!strcasecmp(ws, "deck")) draw = 0;
                else for (int s = 0; s < NSUIT; s++) if (toupper(ws[0]) == SUIT_CH[s]) draw = s + 1;
                if (draw < 0) { printf("draw must be deck or a suit letter\n"); continue; }
                chosen.card = (uint8_t)card; chosen.discard = (uint8_t)disc; chosen.draw = (uint8_t)draw;
                for (int i = 0; i < n; i++)
                    if (mv[i].card == chosen.card && mv[i].discard == chosen.discard && mv[i].draw == chosen.draw) ok = 1;
                if (!ok) printf("that is not a legal move\n");
            }
            char nm[64];
            lc_move_name(&st, chosen, nm);
            printf("you: %s\n", nm);
            lc_apply(&st, chosen);
        } else {
            Move m = agent_move(&ai, &st, &rng);
            char nm[64];
            lc_move_name(&st, m, nm);
            printf("opponent: %s\n", nm);
            lc_apply(&st, m);
        }
    }
    show(&st, me);
    int a = lc_score(&st, me), b = lc_score(&st, me ^ 1);
    printf("\nfinal: you %d, opponent %d -- %s\n", a, b,
           a > b ? "you win" : (a < b ? "you lose" : "draw"));
    return 0;
}
