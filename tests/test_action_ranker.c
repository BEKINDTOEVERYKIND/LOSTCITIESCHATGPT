/* Focused regressions for rollout4's direct signed pairwise veto. */
#include "../src/agent.h"
#include "../src/search.h"
#include "../src/spec.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;
#define CHECK(cond, ...) do { if (!(cond)) { \
    printf("FAIL %s:%d: ", __FILE__, __LINE__); \
    printf(__VA_ARGS__); printf("\n"); failures++; \
} } while (0)

static int combination_index(Move m)
{
    return (m.card * 2 + m.discard) * NET_NDRAW + m.draw;
}

static State dealt_state(uint64_t seed)
{
    Rng rng;
    rng_seed(&rng, seed);
    State st;
    lc_deal(&st, &rng);
    return st;
}

static void first_three_moves(const State *st, Move out[3])
{
    Move legal[MAX_MOVES];
    int n = lc_moves(st, legal);
    CHECK(n >= 3, "test state has only %d legal moves", n);
    for (int i = 0; i < 3; i++) out[i] = legal[i];
}

static void test_raw_pair_score_and_gate(void)
{
    Net *root = (Net *)malloc(sizeof *root);
    Net *ranker = (Net *)malloc(sizeof *ranker);
    CHECK(root && ranker, "network allocation failed");
    if (!root || !ranker) { free(root); free(ranker); return; }
    net_zero(root);
    net_zero(ranker);

    State st = dealt_state(UINT64_C(0x725402798));
    Move pair[3];
    first_three_moves(&st, pair);
    double score = 99.0;
    CHECK(policy_residual_log_odds_sym(
              root, root, &st, pair[0], pair[1], 1, &score) && score == 0.0,
          "root/ranker alias was not exactly neutral (%.9f)", score);

    ranker->bcomb[combination_index(pair[1])] = 2.0f;
    CHECK(policy_residual_log_odds_sym(
              root, ranker, &st, pair[0], pair[1], 1, &score) &&
              fabs(score - 2.0) < 1e-9,
          "positive residual pair score is %.9f, expected 2", score);
    double reverse = 0.0;
    CHECK(policy_residual_log_odds_sym(
              root, ranker, &st, pair[1], pair[0], 1, &reverse) &&
              fabs(score + reverse) < 1e-12,
          "pair score is not antisymmetric: %.9f vs %.9f", score, reverse);

    Agent actor;
    agent_default(&actor, AG_ROLLOUT, root);
    actor.action_ranker_net = ranker;
    actor.symmetries = 1;
    actor.action_ranker_min = 1.5f;
    int valid = 0;
    double gated_score = 0.0;
    CHECK(rollout_action_ranker_veto(
              &actor, &st, pair[0], pair[1], &gated_score, &valid) && valid &&
              fabs(gated_score - 2.0) < 1e-9,
          "positive ranker support did not retain proposal");
    actor.action_ranker_min = 2.5f;
    CHECK(!rollout_action_ranker_veto(
               &actor, &st, pair[0], pair[1], &gated_score, &valid) && valid,
          "below-threshold ranker support did not veto proposal");

    /* The helper evaluates exactly two logits; an unrelated legal action is
     * absent from the normalization and cannot change their signed score. */
    double before = gated_score;
    ranker->bcomb[combination_index(pair[2])] = 1.0e20f;
    CHECK(rollout_action_ranker_veto(
              &actor, &st, pair[0], pair[1], &gated_score, &valid) == 0 &&
              valid && gated_score == before,
          "third-move logit changed the direct pair score");

    ranker->bcomb[combination_index(pair[1])] = NAN;
    CHECK(!rollout_action_ranker_veto(
               &actor, &st, pair[0], pair[1], &gated_score, &valid) && !valid,
          "non-finite ranker score did not fail closed");
    free(root);
    free(ranker);
}

static void test_suit_equivariance(void)
{
    Net *root = (Net *)malloc(sizeof *root);
    Net *ranker = (Net *)malloc(sizeof *ranker);
    CHECK(root && ranker, "network allocation failed");
    if (!root || !ranker) { free(root); free(ranker); return; }
    net_init(root, UINT64_C(101));
    net_init(ranker, UINT64_C(202));
    State st = dealt_state(UINT64_C(303));
    Move pair[3];
    first_three_moves(&st, pair);
    double original = 0.0;
    CHECK(policy_residual_log_odds_sym(
              root, ranker, &st, pair[0], pair[1], 120, &original),
          "full-group ranker score failed");

    const uint8_t perm[NSUIT] = { 2, 4, 1, 0, 3 };
    State ps;
    lc_permute_suits(&st, &ps, perm);
    Move pbase = lc_permute_move(pair[0], perm);
    Move pproposal = lc_permute_move(pair[1], perm);
    double transformed = 0.0;
    CHECK(policy_residual_log_odds_sym(
              root, ranker, &ps, pbase, pproposal, 120, &transformed) &&
              fabs(original - transformed) < 1e-7,
          "120-way score changed under suit relabelling: %.9f vs %.9f",
          original, transformed);
    free(root);
    free(ranker);
}

static void test_hidden_information_sanitized(void)
{
    Net *root = (Net *)malloc(sizeof *root);
    Net *ranker = (Net *)malloc(sizeof *ranker);
    CHECK(root && ranker, "network allocation failed");
    if (!root || !ranker) { free(root); free(ranker); return; }
    net_init(root, UINT64_C(404));
    net_init(ranker, UINT64_C(505));
    State a = dealt_state(UINT64_C(606));
    State b = a;
    Move pair[3];
    first_three_moves(&a, pair);

    int opponent = a.turn ^ 1;
    int hidden_hand = __builtin_ctzll(a.hand[opponent]);
    int hidden_deck = a.deck[a.deck_pos];
    b.hand[opponent] &= ~(UINT64_C(1) << hidden_hand);
    b.hand[opponent] |= UINT64_C(1) << hidden_deck;
    b.deck[b.deck_pos] = (uint8_t)hidden_hand;

    Agent actor;
    agent_default(&actor, AG_ROLLOUT, root);
    actor.action_ranker_net = ranker;
    actor.symmetries = 20;
    double sa = 0.0, sb = 0.0;
    int va = 0, vb = 0;
    int pa = rollout_action_ranker_veto(
        &actor, &a, pair[0], pair[1], &sa, &va);
    int pb = rollout_action_ranker_veto(
        &actor, &b, pair[0], pair[1], &sb, &vb);
    CHECK(va && vb && pa == pb && sa == sb,
          "hidden opponent/deck assignment changed gate: %d %.9f vs %d %.9f",
          pa, sa, pb, sb);
    free(root);
    free(ranker);
}

static void test_rolloutu4_spec_roles(void)
{
    const char *tail =
        "32:3:0.02:0:1:14:0:0:0:0:3.5:2:4:20:0:0:20:1:0:"
        "32:1:0:0:0:0:0:0:3:1:0:0:0:0:0:0:1:0:0:0:1:0.25";
    char spec[512];
    snprintf(spec, sizeof spec,
             "rolloutu4:data/champion.bin:data/c8.bin:data/best.bin:%s",
             tail);
    Agent ranker;
    spec_parse(spec, &ranker);
    CHECK(ranker.kind == AG_ROLLOUT && ranker.no_belief &&
          ranker.net && ranker.continuation_net && ranker.action_ranker_net &&
          !ranker.veto_continuation_net &&
          ranker.net != ranker.continuation_net &&
          ranker.net != ranker.action_ranker_net &&
          ranker.continuation_net != ranker.action_ranker_net &&
          ranker.owns_net && ranker.owns_continuation_net &&
          ranker.owns_action_ranker_net &&
          !ranker.owns_veto_continuation_net &&
          fabsf(ranker.action_ranker_min - 0.25f) < 1e-6f,
          "rolloutu4 did not bind three distinct checkpoint roles");
    spec_release(&ranker);
    CHECK(!ranker.net && !ranker.continuation_net &&
          !ranker.action_ranker_net && !ranker.veto_continuation_net &&
          !ranker.owns_net && !ranker.owns_continuation_net &&
          !ranker.owns_action_ranker_net,
          "rolloutu4 release did not clear owned roles");

    snprintf(spec, sizeof spec,
             "rolloutu4:data/champion.bin:data/champion.bin:"
             "data/champion.bin:%s", tail);
    spec_parse(spec, &ranker);
    CHECK(ranker.net == ranker.continuation_net &&
          ranker.net == ranker.action_ranker_net &&
          !ranker.owns_continuation_net &&
          !ranker.owns_action_ranker_net &&
          !ranker.veto_continuation_net,
          "rolloutu4 same-path roles did not safely alias root");
    spec_release(&ranker);

    snprintf(spec, sizeof spec,
             "rolloutu3:data/champion.bin:data/champion.bin:"
             "data/champion.bin:%s", tail);
    /* rollout3 has no ranker threshold field, so remove field 41. */
    char *last = strrchr(spec, ':');
    CHECK(last != NULL, "could not trim rollout3 tail");
    if (last) *last = '\0';
    Agent controller;
    spec_parse(spec, &controller);
    CHECK(controller.veto_continuation_net == controller.net &&
          controller.action_ranker_net == NULL &&
          !controller.owns_action_ranker_net,
          "rolloutu3 semantics changed while adding rolloutu4");
    spec_release(&controller);
}

int main(void)
{
    test_raw_pair_score_and_gate();
    test_suit_equivariance();
    test_hidden_information_sanitized();
    test_rolloutu4_spec_roles();
    if (failures) {
        printf("%d action-ranker test(s) failed\n", failures);
        return 1;
    }
    printf("action-ranker tests passed\n");
    return 0;
}
