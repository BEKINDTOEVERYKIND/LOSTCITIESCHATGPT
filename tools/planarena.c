/* planarena -- paired strength screen for the information-preserving
 * current-hand scheduler on top of the frozen 20-way champion policy.
 *
 * This is intentionally a research executable rather than a new production
 * agent spec: promotion happens only after the screen establishes that the
 * structural sequencing rule helps match play.
 */
#include "../src/agent.h"
#include "../src/planner.h"
#include "../src/net.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static Move policy_move(const Net *net, const State *st, int plan_deck_max,
                        int minimum_block_reduction)
{
    Move mv[MAX_MOVES];
    float prob[MAX_MOVES];
    int n = policy_probs_sym(net, st, mv, prob, NULL, 20);
    int order[MAX_MOVES];
    for (int i = 0; i < n; i++) order[i] = i;
    for (int i = 0; i < n; i++) {
        int best = i;
        for (int j = i + 1; j < n; j++)
            if (prob[order[j]] > prob[order[best]]) best = j;
        int tmp = order[i]; order[i] = order[best]; order[best] = tmp;
    }
    if (plan_deck_max > 0 && st->deck_left <= plan_deck_max) {
        int pick = hand_plan_conservative_choose(
            st, st->turn, mv, prob, order, n < 8 ? n : 8,
            (st->deck_left + 1) / 2, minimum_block_reduction);
        if (pick >= 0)
            return mv[pick];
    }
    return mv[order[0]];
}

static void deal_from_rng(State *st, Rng *rng, int round,
                          const int cum[2])
{
    uint8_t deck[NCARD];
    for (int i = 0; i < NCARD; i++) deck[i] = (uint8_t)i;
    for (int i = NCARD - 1; i > 0; i--) {
        int j = (int)rng_below(rng, (uint32_t)i + 1);
        uint8_t tmp = deck[i]; deck[i] = deck[j]; deck[j] = tmp;
    }
    lc_deal_from_deck(st, deck);
    st->round = (uint8_t)round;
    st->cum[0] = (int16_t)cum[0];
    st->cum[1] = (int16_t)cum[1];
    st->turn = (uint8_t)(round & 1);
}

static void make_match_decks(uint64_t seed, int pair, State out[MATCH_ROUNDS])
{
    Rng rng;
    rng_seed(&rng, seed ^ (UINT64_C(0x9E3779B97F4A7C15) *
                           (uint64_t)(pair + 1)));
    int zero[2] = { 0, 0 };
    for (int r = 0; r < MATCH_ROUNDS; r++)
        deal_from_rng(&out[r], &rng, r, zero);
}

static int play(const Net *net, const State deals[MATCH_ROUNDS],
                int planner_seat, int plan_deck_max,
                int minimum_block_reduction)
{
    int cum[2] = { 0, 0 };
    for (int r = 0; r < MATCH_ROUNDS; r++) {
        State st = deals[r];
        st.cum[0] = (int16_t)cum[0];
        st.cum[1] = (int16_t)cum[1];
        while (!st.over) {
            int dmax = st.turn == planner_seat ? plan_deck_max : 0;
            lc_apply(&st, policy_move(net, &st, dmax,
                                      minimum_block_reduction));
        }
        cum[0] += lc_score(&st, 0);
        cum[1] += lc_score(&st, 1);
    }
    return planner_seat == 0 ? cum[0] - cum[1] : cum[1] - cum[0];
}

int main(int argc, char **argv)
{
    if (argc < 2 || argc > 6) {
        fprintf(stderr,
                "usage: %s NET [pairs [seed [deck_max [block_gap]]]]\n",
                argv[0]);
        return 1;
    }
    int pairs = argc > 2 ? atoi(argv[2]) : 1000;
    uint64_t seed = argc > 3 ? strtoull(argv[3], NULL, 10) : 20260730;
    int deck_max = argc > 4 ? atoi(argv[4]) : 16;
    int block_gap = argc > 5 ? atoi(argv[5]) : 8;
    Net *net = malloc(sizeof *net);
    if (!net || net_load(net, argv[1])) {
        fprintf(stderr, "planarena: cannot load %s\n", argv[1]);
        free(net);
        return 1;
    }

    double sum = 0.0, sum2 = 0.0, wsum = 0.0, wsum2 = 0.0;
    int wins = 0, losses = 0, draws = 0;
    for (int g = 0; g < pairs; g++) {
        State deals[MATCH_ROUNDS];
        make_match_decks(seed, g, deals);
        int a = play(net, deals, 0, deck_max, block_gap);
        int b = play(net, deals, 1, deck_max, block_gap);
        double margin_pair = a + b;
        double score_pair = (a > 0 ? 1.0 : (a == 0 ? 0.5 : 0.0))
                          + (b > 0 ? 1.0 : (b == 0 ? 0.5 : 0.0));
        wins += (a > 0) + (b > 0);
        losses += (a < 0) + (b < 0);
        draws += (a == 0) + (b == 0);
        sum += margin_pair; sum2 += margin_pair * margin_pair;
        wsum += score_pair; wsum2 += score_pair * score_pair;
    }
    double mean = sum / pairs;
    double var = pairs > 1
        ? (sum2 - sum * sum / pairs) / (pairs - 1) : 0.0;
    double wmean = wsum / pairs;
    double wvar = pairs > 1
        ? (wsum2 - wsum * wsum / pairs) / (pairs - 1) : 0.0;
    if (var < 0.0) var = 0.0;
    if (wvar < 0.0) wvar = 0.0;
    printf("planner deck<=%d block-gap>=%d vs exact-20 policy: "
           "%d mirrored pairs\n", deck_max, block_gap, pairs);
    printf("margin/match %+.3f +- %.3f SE\n",
           mean / 2.0, sqrt(var / pairs) / 2.0);
    printf("match score %.3f%% +- %.3f%% SE  W-L-D %d-%d-%d\n",
           50.0 * wmean, 50.0 * sqrt(wvar / pairs),
           wins, losses, draws);
    free(net);
    return 0;
}
