/* planprobe -- inspect the exact current-hand scheduling subsolver. */
#include "../src/agent.h"
#include "../src/planner.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

static int name_id_free(const char *name, uint64_t used)
{
    char shown[8];
    for (int c = 0; c < NCARD; c++) {
        lc_card_name(c, shown);
        if (!strcasecmp(shown, name) && !((used >> c) & 1ULL)) return c;
    }
    return -1;
}

static int load_state(const char *path, State *st)
{
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    memset(st, 0, sizeof *st);
    uint64_t used = 0;
    char line[512];
    while (fgets(line, sizeof line, f)) {
        char *tok = strtok(line, " \t\n");
        if (!tok) continue;
        if (!strcmp(tok, "turn")) st->turn = (uint8_t)atoi(strtok(NULL, " \n"));
        else if (!strcmp(tok, "round")) st->round = (uint8_t)atoi(strtok(NULL, " \n"));
        else if (!strcmp(tok, "nply")) st->nply = (uint16_t)atoi(strtok(NULL, " \n"));
        else if (!strcmp(tok, "deck_left")) st->deck_left = (uint8_t)atoi(strtok(NULL, " \n"));
        else if (!strcmp(tok, "cum")) {
            st->cum[0] = (int16_t)atoi(strtok(NULL, " \n"));
            st->cum[1] = (int16_t)atoi(strtok(NULL, " \n"));
        } else if (!strncmp(tok, "hand", 4)) {
            int p = tok[4] - '0';
            char *word;
            while ((word = strtok(NULL, " \n"))) {
                int c = name_id_free(word, used);
                if (c < 0) { fclose(f); return 0; }
                used |= 1ULL << c;
                st->hand[p] |= 1ULL << c;
                st->hand_n[p]++;
            }
        } else if (!strncmp(tok, "known", 5)) {
            int p = tok[5] - '0';
            char *word;
            while ((word = strtok(NULL, " \n"))) {
                char shown[8];
                for (int c = 0; c < NCARD; c++) {
                    lc_card_name(c, shown);
                    if (!strcasecmp(shown, word) &&
                        ((st->hand[p] >> c) & 1ULL) &&
                        !((st->known[p] >> c) & 1ULL)) {
                        st->known[p] |= 1ULL << c;
                        break;
                    }
                }
            }
        } else if (!strcmp(tok, "exp")) {
            int p = atoi(strtok(NULL, " \n"));
            int s = atoi(strtok(NULL, " \n"));
            char *word;
            while ((word = strtok(NULL, " \n"))) {
                int c = name_id_free(word, used);
                if (c < 0) { fclose(f); return 0; }
                used |= 1ULL << c;
                st->played[p] |= 1ULL << c;
                st->exp_n[p][s]++;
                if (CARD_IS_WAGER(c)) st->exp_wager[p][s]++;
                else {
                    int v = CARD_VALUE(c);
                    st->exp_top[p][s] = (uint8_t)v;
                    st->exp_sum[p][s] += (uint8_t)v;
                }
            }
        } else if (!strcmp(tok, "pile")) {
            int s = atoi(strtok(NULL, " \n"));
            char *word;
            while ((word = strtok(NULL, " \n"))) {
                int c = name_id_free(word, used);
                if (c < 0) { fclose(f); return 0; }
                used |= 1ULL << c;
                st->pile[s][st->pile_n[s]++] = (uint8_t)c;
                st->discarded |= 1ULL << c;
            }
        }
    }
    fclose(f);
    return 1;
}

int main(int argc, char **argv)
{
    if (argc != 3) {
        fprintf(stderr, "usage: %s STATE NET\n", argv[0]);
        return 1;
    }
    State st;
    if (!load_state(argv[1], &st)) {
        fprintf(stderr, "planprobe: cannot load %s\n", argv[1]);
        return 1;
    }
    Net *net = malloc(sizeof *net);
    if (!net || net_load(net, argv[2])) {
        fprintf(stderr, "planprobe: cannot load %s\n", argv[2]);
        free(net);
        return 1;
    }

    Move mv[MAX_MOVES];
    float prob[MAX_MOVES];
    int n = policy_probs_sym(net, &st, mv, prob, NULL, 20);
    int order[MAX_MOVES];
    for (int i = 0; i < n; i++) order[i] = i;
    for (int i = 0; i < n; i++) {
        int best = i;
        for (int j = i + 1; j < n; j++)
            if (prob[order[j]] > prob[order[best]]) best = j;
        int tmp = order[i]; order[i] = order[best]; order[best] = tmp;
    }

    int turns = (st.deck_left + 1) / 2;
    HandPlan plan;
    hand_plan_build(&st, st.turn, turns, &plan);
    int pick = hand_plan_choose(&st, st.turn, mv, prob, order,
                                n < 8 ? n : 8, turns);
    printf("turns %d base %d guaranteed %d cards %d first", turns,
           plan.base_score, plan.score, plan.min_cards);
    for (int c = 0; c < NCARD; c++) {
        if (!((plan.first_cards >> c) & 1ULL)) continue;
        char card[8];
        lc_card_name(c, card);
        printf(" %s(block=%d)", card,
               hand_plan_block_cost(&st, st.turn, c));
    }
    printf("\n");
    if (pick >= 0) {
        char move[32];
        lc_move_name(&st, mv[pick], move);
        printf("choice %s prior %.6f\n", move, prob[pick]);
    } else {
        printf("choice none\n");
    }

    if (n > 0) {
        Move top = mv[order[0]];
        printf("top-action draw plans\n");
        for (int k = 0; k < n; k++) {
            int i = order[k];
            int same = mv[i].discard == top.discard &&
                ((CARD_IS_WAGER(mv[i].card) && CARD_IS_WAGER(top.card) &&
                  CARD_SUIT(mv[i].card) == CARD_SUIT(top.card)) ||
                 mv[i].card == top.card);
            if (!same) continue;
            char move[32];
            lc_move_name(&st, mv[i], move);
            printf("  %s prior %.6f expected-visible %.6f\n", move,
                   prob[i], hand_plan_expected_score_after_move(
                                &st, st.turn, mv[i]));
        }
    }

    int pickup = -1, pickup_score = -1000000;
    for (int i = 0; i < n; i++) {
        if (mv[i].draw == 0) continue;
        int suit = mv[i].draw - 1;
        if (st.pile_n[suit] == 0 ||
            !CARD_IS_WAGER(st.pile[suit][st.pile_n[suit] - 1]))
            continue;
        State after = st;
        lc_apply(&after, mv[i]);
        HandPlan future;
        hand_plan_build(&after, st.turn, after.deck_left / 2, &future);
        if (pickup < 0 || future.score > pickup_score ||
            (future.score == pickup_score && prob[i] > prob[pickup])) {
            pickup = i;
            pickup_score = future.score;
        }
    }
    if (pickup >= 0) {
        char move[32];
        lc_move_name(&st, mv[pickup], move);
        printf("wager-pickup %s prior %.6f guaranteed %d\n",
               move, prob[pickup], pickup_score);
    } else {
        printf("wager-pickup none\n");
    }
    free(net);
    return 0;
}
