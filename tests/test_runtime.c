/* Cross-module regressions for search, policy, features, and match evaluation. */
#include "../src/agent.h"
#include "../src/features.h"
#include "../src/match.h"
#include "../src/net.h"
#include "../src/search.h"
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

static void test_rollout_terminal_objective(void)
{
    State st;
    memset(&st, 0, sizeof st);
    st.over = 1;
    st.cum[0] = 40;
    st.exp_n[0][0] = 1;
    st.exp_sum[0][0] = 10; /* round margin -10, match margin +30 */

    st.round = 0;
    CHECK(rollout_terminal_objective(&st, 0, 2) == -10.0,
          "hybrid objective leaked into an early round");
    st.round = MATCH_ROUNDS - 1;
    CHECK(rollout_terminal_objective(&st, 0, 0) == -10.0,
          "margin rollout objective changed");
    CHECK(rollout_terminal_objective(&st, 0, 1) == 50.0,
          "final-round match-result objective is wrong");
    CHECK(fabs(rollout_terminal_objective(&st, 0, 2) - 51.5) < 1e-9,
          "final-round hybrid objective is wrong");
    CHECK(fabs(rollout_terminal_objective(&st, 1, 2) + 51.5) < 1e-9,
          "hybrid objective is not zero-sum");

    st.cum[0] = 10; /* -10 round + 10 carried = tied match */
    CHECK(rollout_terminal_objective(&st, 0, 1) == 0.0 &&
          rollout_terminal_objective(&st, 0, 2) == 0.0,
          "final-round draw has nonzero objective");
}

static void test_rollout_value_scale(void)
{
    Net *net = malloc(sizeof(*net));
    CHECK(net != NULL, "network allocation for rollout value");
    if (!net) return;
    CHECK(net_load(net, "data/champion.bin") == 0,
          "load champion for rollout value");

    uint8_t deck[NCARD];
    for (int i = 0; i < NCARD; i++) deck[i] = (uint8_t)i;
    State st;
    lc_deal_from_deck(&st, deck);
    st.round = MATCH_ROUNDS - 1;
    st.cum[0] = 25;
    st.cum[1] = -10;

    Move mv[MAX_MOVES];
    float pr[MAX_MOVES], raw_value = 0.0f;
    (void)policy_probs(net, &st, mv, pr, &raw_value);

    Agent a;
    agent_default(&a, AG_ROLLOUT, net);
    a.dets = 2;
    a.root_width = 2;
    a.win_q = 2;
    Rng rng;
    rng_seed(&rng, 8811);
    float searched_value = 0.0f;
    (void)rollout_move(&a, &st, &rng, &searched_value, NULL);
    CHECK(searched_value == raw_value,
          "rollout out_value changed scale when search ran");
    free(net);
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

static void test_wager_parameter_projection(void)
{
    Net *net = malloc(sizeof(*net));
    Net *once = malloc(sizeof(*once));
    CHECK(net && once, "network allocation for wager projection");
    if (!net || !once) goto done;
    CHECK(net_load(net, "data/c8.bin") == 0,
          "load champion for wager projection");
    net_project_wager_symmetry(net);

    for (int plane = 0; plane < FEAT_PLANES; plane++)
        for (int s = 0; s < NSUIT; s++) {
            int c = plane * NCARD + s * NRANK;
            CHECK(memcmp(net->w1[c], net->w1[c + 1],
                         sizeof(net->w1[c])) == 0 &&
                  memcmp(net->w1[c], net->w1[c + 2],
                         sizeof(net->w1[c])) == 0,
                  "wager input rows remain distinct");
        }
    for (int s = 0; s < NSUIT; s++) {
        int card = s * NRANK;
        CHECK(memcmp(net->wbel[card], net->wbel[card + 1],
                     sizeof(net->wbel[card])) == 0 &&
              memcmp(net->wbel[card], net->wbel[card + 2],
                     sizeof(net->wbel[card])) == 0 &&
              net->bbel[card] == net->bbel[card + 1] &&
              net->bbel[card] == net->bbel[card + 2],
              "wager belief rows remain distinct");
        for (int d = 0; d < 2; d++) {
            int p0 = card * 2 + d, p1 = (card + 1) * 2 + d;
            int p2 = (card + 2) * 2 + d;
            CHECK(memcmp(net->wplay[p0], net->wplay[p1],
                         sizeof(net->wplay[p0])) == 0 &&
                  memcmp(net->wplay[p0], net->wplay[p2],
                         sizeof(net->wplay[p0])) == 0 &&
                  net->bplay[p0] == net->bplay[p1] &&
                  net->bplay[p0] == net->bplay[p2],
                  "wager policy rows remain distinct");
            for (int draw = 0; draw < NET_NDRAW; draw++) {
                int c0 = p0 * NET_NDRAW + draw;
                int c1 = p1 * NET_NDRAW + draw;
                int c2 = p2 * NET_NDRAW + draw;
                CHECK(memcmp(net->wcomb[c0], net->wcomb[c1],
                             sizeof(net->wcomb[c0])) == 0 &&
                      memcmp(net->wcomb[c0], net->wcomb[c2],
                             sizeof(net->wcomb[c0])) == 0 &&
                      net->bcomb[c0] == net->bcomb[c1] &&
                      net->bcomb[c0] == net->bcomb[c2],
                      "wager interaction rows remain distinct");
            }
        }
    }
    memcpy(once, net, sizeof(*net));
    net_project_wager_symmetry(net);
    CHECK(memcmp(net, once, sizeof(*net)) == 0,
          "wager projection is not idempotent");

done:
    free(net);
    free(once);
}

static void test_wager_tied_gradients(void)
{
    Net *g = calloc(1, sizeof(*g));
    Net *fresh = malloc(sizeof(*fresh));
    CHECK(g && fresh, "network allocation for tied wager gradients");
    if (!g || !fresh) goto done;

    int card = 2 * NRANK;
    g->w1[card][7] = 1.0f;
    g->w1[card + 1][7] = 2.0f;
    g->w1[card + 2][7] = 3.0f;
    int p0 = card * 2 + 1, p1 = (card + 1) * 2 + 1;
    int p2 = (card + 2) * 2 + 1;
    g->bplay[p0] = 4.0f; g->bplay[p1] = 5.0f; g->bplay[p2] = 6.0f;
    int c0 = p0 * NET_NDRAW + 4, c1 = p1 * NET_NDRAW + 4;
    int c2 = p2 * NET_NDRAW + 4;
    g->wcomb[c0][11] = 7.0f;
    g->wcomb[c1][11] = 8.0f;
    g->wcomb[c2][11] = 9.0f;
    net_tie_wager_gradients(g);
    CHECK(g->w1[card][7] == 6.0f &&
          g->w1[card + 1][7] == 6.0f &&
          g->w1[card + 2][7] == 6.0f,
          "wager input gradients were not summed and tied");
    CHECK(g->bplay[p0] == 15.0f && g->bplay[p1] == 15.0f &&
          g->bplay[p2] == 15.0f,
          "wager policy gradients were not summed and tied");
    CHECK(g->wcomb[c0][11] == 24.0f &&
          g->wcomb[c1][11] == 24.0f &&
          g->wcomb[c2][11] == 24.0f,
          "wager interaction gradients were not summed and tied");

    net_init(fresh, 991);
    for (int s = 0; s < NSUIT; s++) {
        int c = s * NRANK;
        CHECK(memcmp(fresh->wplay[c * 2], fresh->wplay[(c + 1) * 2],
                     sizeof(fresh->wplay[0])) == 0 &&
              memcmp(fresh->wplay[c * 2], fresh->wplay[(c + 2) * 2],
                     sizeof(fresh->wplay[0])) == 0,
              "new network starts with untied wager rows");
    }

done:
    free(g);
    free(fresh);
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

static uint8_t rotate_card(uint8_t card, int shift)
{
    return (uint8_t)CARD_MAKE((CARD_SUIT(card) + shift) % NSUIT,
                              CARD_RANK(card));
}

static Move rotate_move(Move m, int shift)
{
    m.card = rotate_card(m.card, shift);
    if (m.draw > 0) m.draw = (uint8_t)((m.draw - 1 + shift) % NSUIT + 1);
    return m;
}

static void check_suit_ensemble_equivariance(
    const Net *net, const State *a, const uint8_t perm[NSUIT],
    int symmetries, const char *label)
{
    State b;
    lc_permute_suits(a, &b, perm);
    Move am[MAX_MOVES], bm[MAX_MOVES];
    float ap[MAX_MOVES], bp[MAX_MOVES], av, bv;
    int an = policy_probs_sym(net, a, am, ap, &av, symmetries);
    int bn = policy_probs_sym(net, &b, bm, bp, &bv, symmetries);
    CHECK(an == bn, "%s ensemble changed legal move count", label);
    CHECK(fabsf(av - bv) < 5e-4f,
          "%s ensemble value is not equivariant", label);

    int by_pack[MOVE_NPACK];
    for (int i = 0; i < MOVE_NPACK; i++) by_pack[i] = -1;
    for (int i = 0; i < bn; i++) by_pack[MOVE_PACK(bm[i])] = i;
    for (int i = 0; i < an; i++) {
        Move mapped = lc_permute_move(am[i], perm);
        int j = by_pack[MOVE_PACK(mapped)];
        CHECK(j >= 0, "%s mapped legal move missing", label);
        if (j >= 0)
            CHECK(fabsf(ap[i] - bp[j]) < 5e-5f,
                  "%s ensemble policy is not equivariant", label);
    }
}

static void test_suit_symmetry_ensemble(void)
{
    Net *net = malloc(sizeof(*net));
    CHECK(net != NULL, "network allocation for suit symmetry");
    if (!net) return;
    CHECK(net_load(net, "data/c8.bin") == 0, "load champion for suit symmetry");

    uint8_t deck[NCARD], rotated[NCARD];
    for (int i = 0; i < NCARD; i++) {
        deck[i] = (uint8_t)i;
        rotated[i] = rotate_card((uint8_t)i, 1);
    }
    State a, b;
    lc_deal_from_deck(&a, deck);
    lc_deal_from_deck(&b, rotated);
    a.round = b.round = 2;
    a.cum[0] = b.cum[0] = 17;
    a.cum[1] = b.cum[1] = -9;

    const uint8_t affine[NSUIT] = { 1, 3, 0, 2, 4 };
    const uint8_t non_affine[NSUIT] = { 1, 0, 2, 3, 4 };
    check_suit_ensemble_equivariance(net, &a, affine, 20, "affine-20");
    check_suit_ensemble_equivariance(net, &a, non_affine, 120, "full-120");

    for (int ply = 0; ply < 12 && !a.over; ply++) {
        Move am[MAX_MOVES], bm[MAX_MOVES], rawm[MAX_MOVES], one[MAX_MOVES];
        float ap[MAX_MOVES], bp[MAX_MOVES], rawp[MAX_MOVES], onep[MAX_MOVES];
        float av, bv, rawv, onev;
        int an = policy_probs_sym(net, &a, am, ap, &av, 5);
        int bn = policy_probs_sym(net, &b, bm, bp, &bv, 5);
        int rn = policy_probs(net, &a, rawm, rawp, &rawv);
        int on = policy_probs_sym(net, &a, one, onep, &onev, 1);
        CHECK(an == bn && rn == on && an == rn,
              "suit ensemble changed legal move count");
        CHECK(memcmp(rawm, one, (size_t)rn * sizeof(Move)) == 0 &&
              memcmp(rawp, onep, (size_t)rn * sizeof(float)) == 0 &&
              rawv == onev, "one-symmetry mode changed legacy output");
        CHECK(fabsf(av - bv) < 2e-4f,
              "cyclic ensemble value is not rotation invariant");

        float asum = 0.0f, bsum = 0.0f;
        int by_pack[MOVE_NPACK];
        for (int i = 0; i < MOVE_NPACK; i++) by_pack[i] = -1;
        for (int i = 0; i < bn; i++) by_pack[MOVE_PACK(bm[i])] = i;
        for (int i = 0; i < an; i++) {
            int j = by_pack[MOVE_PACK(rotate_move(am[i], 1))];
            CHECK(j >= 0, "rotated legal move missing");
            if (j >= 0)
                CHECK(fabsf(ap[i] - bp[j]) < 2e-5f,
                      "cyclic ensemble policy is not rotation equivariant");
            asum += ap[i];
        }
        for (int i = 0; i < bn; i++) bsum += bp[i];
        CHECK(fabsf(asum - 1.0f) < 2e-5f &&
              fabsf(bsum - 1.0f) < 2e-5f,
              "suit ensemble probabilities do not normalize");

        int pick = ply % an;
        Move ar = am[pick], br = rotate_move(ar, 1);
        lc_apply(&a, ar);
        lc_apply(&b, br);
    }
    free(net);
}

int main(void)
{
    test_sampler();
    test_rollout_terminal_objective();
    test_rollout_value_scale();
    test_wager_interaction_head();
    test_wager_parameter_projection();
    test_wager_tied_gradients();
    test_pile_order_features();
    test_match_thread_determinism();
    test_suit_symmetry_ensemble();
    if (failures == 0) {
        printf("all runtime regression tests passed\n");
        return 0;
    }
    printf("%d failures\n", failures);
    return 1;
}
