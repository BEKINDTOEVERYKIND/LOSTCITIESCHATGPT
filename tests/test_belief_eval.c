/* Focused contracts for truth-scrubbed exact-K held-out scoring. */
#include "../src/agent.h"
#include "../src/net.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures;
#define CHECK(cond, ...) do { if (!(cond)) { \
    printf("FAIL %s:%d: ", __FILE__, __LINE__); \
    printf(__VA_ARGS__); putchar('\n'); failures++; \
} } while (0)

static double log_choose(int n, int k)
{
    return lgamma((double)n + 1.0) - lgamma((double)k + 1.0)
         - lgamma((double)(n - k) + 1.0);
}

int main(void)
{
    uint8_t deck[NCARD];
    for (int i = 0; i < NCARD; i++) deck[i] = (uint8_t)i;
    State truth;
    lc_deal_from_deck(&truth, deck);
    Move legal[MAX_MOVES];
    int nlegal = lc_moves(&truth, legal);
    CHECK(nlegal > 0, "deterministic fixture has no move");
    if (nlegal <= 0) return 1;
    lc_apply(&truth, legal[0]);

    const int p = truth.turn, o = p ^ 1;
    uint64_t hidden_hand = truth.hand[o] & ~truth.known[o];
    CHECK(hidden_hand != 0, "fixture has no unknown opponent card");
    int held_card = __builtin_ctzll(hidden_hand);
    int deck_card = truth.deck[truth.deck_pos];
    CHECK(!((truth.hand[o] >> deck_card) & 1ULL),
          "fixture deck card is already held");

    State alternative = truth;
    alternative.hand[o] &= ~(1ULL << held_card);
    alternative.hand[o] |= 1ULL << deck_card;
    alternative.deck[alternative.deck_pos] = (uint8_t)held_card;

    State view_a, view_b;
    agent_information_view(&truth, p, &view_a);
    agent_information_view(&alternative, p, &view_b);
    CHECK(memcmp(&view_a, &view_b, sizeof view_a) == 0,
          "information view retained hidden assignment");
    CHECK(view_a.hand[o] == truth.known[o],
          "information view retained unknown opponent cards");
    for (int i = 0; i < NCARD; i++)
        CHECK(view_a.deck[i] == 0,
              "information view retained future deck byte %d", i);

    Net *net = (Net *)malloc(sizeof *net);
    CHECK(net != NULL, "network allocation");
    if (!net) return 1;
    if (net_load(net, "data/champion.bin") != 0) {
        CHECK(0, "load data/champion.bin");
        free(net);
        return 1;
    }

    BeliefDist learned_a, learned_b;
    CHECK(belief_dist_init(net, &view_a, p, 20, 1.15f, &learned_a),
          "learned exact-K distribution A");
    CHECK(belief_dist_init(net, &view_b, p, 20, 1.15f, &learned_b),
          "learned exact-K distribution B");
    CHECK(learned_a.n == learned_b.n && learned_a.need == learned_b.need,
          "hidden assignment changed exact-K dimensions");
    for (int i = 0; i < learned_a.n; i++) {
        CHECK(learned_a.card[i] == learned_b.card[i],
              "hidden assignment changed candidate %d", i);
        CHECK(learned_a.marginal[i] == learned_b.marginal[i],
              "hidden assignment changed marginal %d", i);
    }
    double learned_nll_a = 0.0, learned_nll_b = 0.0;
    CHECK(belief_dist_true_nll(&learned_a, truth.hand[o], &learned_nll_a),
          "score original truth");
    CHECK(belief_dist_true_nll(&learned_b, alternative.hand[o],
                               &learned_nll_b),
          "score alternative truth");
    CHECK(lc_double_isfinite(learned_nll_a) && learned_nll_a >= 0.0,
          "invalid learned truth NLL A %.9f", learned_nll_a);
    CHECK(lc_double_isfinite(learned_nll_b) && learned_nll_b >= 0.0,
          "invalid learned truth NLL B %.9f", learned_nll_b);
    CHECK(!belief_dist_true_nll(&learned_a, truth.known[o], &learned_nll_a),
          "truth scorer accepted wrong unknown-hand cardinality");

    BeliefDist uniform;
    CHECK(belief_dist_init(NULL, &view_a, p, 20, 0.0f, &uniform),
          "uniform exact-K distribution");
    double uniform_nll_a = 0.0, uniform_nll_b = 0.0;
    CHECK(belief_dist_true_nll(&uniform, truth.hand[o], &uniform_nll_a),
          "score original truth under uniform prior");
    CHECK(belief_dist_true_nll(&uniform, alternative.hand[o], &uniform_nll_b),
          "score alternative truth under uniform prior");
    double expected = log_choose(uniform.n, uniform.need);
    CHECK(fabs(uniform_nll_a - expected) < 1e-10,
          "uniform joint NLL %.12f != log C(%d,%d) %.12f",
          uniform_nll_a, uniform.n, uniform.need, expected);
    CHECK(fabs(uniform_nll_b - expected) < 1e-10,
          "uniform NLL depends on hidden subset");

    free(net);
    if (failures) {
        printf("%d belief evaluation regression(s) failed\n", failures);
        return 1;
    }
    printf("belief evaluation regressions passed\n");
    return 0;
}
