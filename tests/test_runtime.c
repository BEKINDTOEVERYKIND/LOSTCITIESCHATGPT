/* Cross-module regressions for search, policy, features, and match evaluation. */
#include "../src/agent.h"
#include "../src/features.h"
#include "../src/match.h"
#include "../src/net.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { \
    printf("FAIL %s:%d: ", __FILE__, __LINE__); \
    printf(__VA_ARGS__); printf("\n"); failures++; \
} } while (0)

static void test_sampler(void)
{
    Rng rng;
    rng_seed(&rng, 1);
    const float one_positive[] = { 0.0f, 1.0f, 0.0f };
    for (int i = 0; i < 1000; i++)
        CHECK(sample_index(one_positive, 3, &rng) == 1,
              "zero-weight action selected");

    const float infinities[] = { INFINITY, 1.0f, INFINITY };
    for (int i = 0; i < 1000; i++) {
        int k = sample_index(infinities, 3, &rng);
        CHECK(k == 0 || k == 2, "finite action beat positive infinity");
    }

    const float unusable[] = { NAN, -1.0f, 0.0f };
    int k = sample_index(unusable, 3, &rng);
    CHECK(k >= 0 && k < 3, "invalid fallback index %d", k);
    CHECK(sample_index(unusable, 0, &rng) == -1, "empty sample must return -1");
}

static void test_wager_interaction_head(void)
{
    Net *net = malloc(sizeof(*net));
    Net *copy = malloc(sizeof(*copy));
    Net *grad = malloc(sizeof(*grad));
    CHECK(net && copy && grad, "network allocation");
    if (!net || !copy || !grad) goto done;
    CHECK(net_load(net, "data/best.bin") == 0, "load shipped legacy model");

    for (int i = 0; i < NET_NCOMB; i++) {
        CHECK(net->bcomb[i] == 0.0f, "legacy residual bias %d not zero", i);
        for (int h = 0; h < NET_H2; h++)
            if (net->wcomb[i][h] != 0.0f) {
                CHECK(0, "legacy residual weight %d,%d not zero", i, h);
                i = NET_NCOMB;
                break;
            }
    }

    uint8_t deck[NCARD];
    for (int i = 0; i < NCARD; i++) deck[i] = (uint8_t)i;
    State st;
    lc_deal_from_deck(&st, deck);
    Features f;
    feat_extract(&st, st.turn, &f);
    NetAct act;
    net_trunk(net, &f, &act);

    uint16_t mv[3];
    Move a = { 0, 0, 0 }, b = { 0, 0, 1 }, c = { 1, 0, 0 };
    mv[0] = MOVE_PACK(a); mv[1] = MOVE_PACK(b); mv[2] = MOVE_PACK(c);
    float before[3], after[3];
    net_policy_act(net, &act, mv, 3, before);
    int ic = (a.card * 2 + a.discard) * NET_NDRAW + a.draw;
    net->bcomb[ic] = 2.0f;
    net_policy_act(net, &act, mv, 3, after);
    CHECK(fabsf((after[0] - before[0]) - 2.0f) < 1e-5f,
          "interaction did not affect matching move");
    CHECK(after[1] == before[1] && after[2] == before[2],
          "interaction leaked to another card/draw combination");

    net_zero(grad);
    const float dlog[3] = { 1.0f, 0.0f, 0.0f };
    net_backward(net, &f, &act, 0.0f, mv, dlog, 3,
                 NULL, NULL, 0, grad);
    CHECK(grad->bcomb[ic] == 1.0f, "interaction bias gradient");

    char path[128];
    snprintf(path, sizeof path, "/tmp/lostcities-net-roundtrip-%ld.bin",
             (long)getpid());
    CHECK(net_save(net, path) == 0, "save current model");
    CHECK(net_load(copy, path) == 0, "reload current model");
    CHECK(memcmp(net, copy, sizeof(*net)) == 0, "model roundtrip differs");
    unlink(path);

done:
    free(net);
    free(copy);
    free(grad);
}

static void test_pile_order_features(void)
{
    State a, b;
    memset(&a, 0, sizeof a);
    memset(&b, 0, sizeof b);
    const int y2 = CARD_MAKE(0, 3);
    const int y3 = CARD_MAKE(0, 4);
    const int y4 = CARD_MAKE(0, 5);
    a.pile[0][0] = (uint8_t)y2;
    a.pile[0][1] = (uint8_t)y3;
    a.pile[0][2] = (uint8_t)y4;
    b.pile[0][0] = (uint8_t)y3;
    b.pile[0][1] = (uint8_t)y2;
    b.pile[0][2] = (uint8_t)y4;
    a.pile_n[0] = b.pile_n[0] = 3;
    a.discarded = b.discarded =
        (1ULL << y2) | (1ULL << y3) | (1ULL << y4);

    Features fa, fb;
    feat_extract(&a, 0, &fa);
    feat_extract(&b, 0, &fb);
    CHECK(fa.nidx != fb.nidx ||
          memcmp(fa.idx, fb.idx, (size_t)fa.nidx * sizeof(fa.idx[0])) != 0 ||
          memcmp(fa.dense, fb.dense, sizeof(fa.dense)) != 0,
          "different buried discard order has identical features");
}

static void test_match_thread_determinism(void)
{
    Agent a, b;
    agent_default(&a, AG_RANDOM, NULL);
    agent_default(&b, AG_RANDOM, NULL);
    MatchResult one, four;
    match_run_r(&a, &b, 100, 1, 424242, MATCH_ROUNDS, &one);
    match_run_r(&a, &b, 100, 4, 424242, MATCH_ROUNDS, &four);
    CHECK(one.pairs == four.pairs && one.games == four.games,
          "thread count changed match count");
    CHECK(one.margin == four.margin && one.margin_se == four.margin_se &&
          one.winrate == four.winrate && one.winrate_se == four.winrate_se &&
          one.points_a == four.points_a && one.points_b == four.points_b &&
          one.plies == four.plies && one.wins == four.wins &&
          one.losses == four.losses && one.draws == four.draws,
          "same seed differs between one and four threads");
}

int main(void)
{
    test_sampler();
    test_wager_interaction_head();
    test_pile_order_features();
    test_match_thread_determinism();
    if (failures == 0) {
        printf("all runtime regression tests passed\n");
        return 0;
    }
    printf("%d failures\n", failures);
    return 1;
}
