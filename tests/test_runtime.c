/* Cross-module regressions for search, policy, features, and match evaluation. */
#include "../src/agent.h"
#include "../src/features.h"
#include "../src/match.h"
#include "../src/net.h"
#include "../src/planner.h"
#include "../src/search.h"
#include "../src/spec.h"
#include "../tools/train_target.h"
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

static uint64_t mix64(uint64_t x)
{
    x += 0x9E3779B97F4A7C15ULL;
    x = (x ^ (x >> 30)) * 0xBF58476D1CE4E5B9ULL;
    x = (x ^ (x >> 27)) * 0x94D049BB133111EBULL;
    return x ^ (x >> 31);
}

/* Reproduce the first round of the human-reviewed UI replay.  Evaluator RNG
 * is deliberately absent: the deployed actor has its own stream. */
static State reviewed_state(const Net *net, int target_ply)
{
    const uint64_t seed = 2214615196ULL;
    Rng deal_rng, actor_rng;
    rng_seed(&deal_rng, seed);
    rng_seed(&actor_rng, mix64(seed ^ 0xA17C0AULL));
    State st;
    lc_deal(&st, &deal_rng);
    st.round = 0;
    st.turn = 0;
    Agent actor;
    agent_default(&actor, AG_POLICY, net);
    actor.symmetries = 20;
    for (int ply = 1; ply < target_ply && !st.over; ply++)
        lc_apply(&st, agent_move(&actor, &st, &actor_rng));
    return st;
}

static int named_move(Move m, const char *card, int discard, int draw)
{
    char name[8];
    lc_card_name(m.card, name);
    return !strcmp(name, card) && m.discard == discard && m.draw == draw;
}

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

static void test_belief_distribution(void)
{
    Net *net = malloc(sizeof(*net));
    CHECK(net != NULL, "network allocation for belief distribution");
    if (!net) return;
    CHECK(net_load(net, "data/champion.bin") == 0,
          "load champion for belief distribution");

    uint8_t deck[NCARD];
    for (int i = 0; i < NCARD; i++) deck[i] = (uint8_t)i;
    State initial;
    lc_deal_from_deck(&initial, deck);
    BeliefDist uniform;
    CHECK(belief_dist_init(net, &initial, initial.turn, 20, 1.15f, &uniform),
          "initialize opening belief");
    float opening = (float)uniform.need / (float)uniform.n;
    double opening_sum = 0.0;
    for (int i = 0; i < uniform.n; i++) {
        opening_sum += uniform.marginal[i];
        CHECK(fabsf(uniform.marginal[i] - opening) < 1e-6f,
              "opening belief is not exact card-count prior");
    }
    CHECK(fabs(opening_sum - uniform.need) < 1e-5,
          "opening marginals do not sum to hand size");

    State st = reviewed_state(net, 13);
    CHECK(st.nply == 12 && st.turn == 0,
          "reviewed ply 13 did not replay");
    BeliefDist dist;
    CHECK(belief_dist_init(net, &st, st.turn, 20, 1.15f, &dist),
          "initialize reviewed belief");
    double sum = 0.0;
    for (int i = 0; i < dist.n; i++) {
        CHECK(dist.marginal[i] >= 0.0f && dist.marginal[i] <= 1.0f,
              "belief marginal outside [0,1]");
        sum += dist.marginal[i];
    }
    CHECK(fabs(sum - dist.need) < 2e-5,
          "reviewed marginals sum %.8f, need %d", sum, dist.need);

    Rng rng;
    rng_seed(&rng, 130013);
    int inclusion[NCARD] = { 0 };
    const int nsample = 5000;
    for (int sample = 0; sample < nsample; sample++) {
        State world;
        belief_dist_sample(&st, st.turn, &rng, &dist, &world);
        int o = st.turn ^ 1;
        CHECK(__builtin_popcountll(world.hand[o]) == st.hand_n[o],
              "sampled opponent hand has wrong cardinality");
        CHECK((world.hand[o] & st.known[o]) == st.known[o],
              "sample dropped a publicly known card");
        CHECK(world.deck_left == st.deck_left,
              "sampled deck has wrong cardinality");
        uint64_t deck_bits = 0;
        for (int d = 0; d < world.deck_left; d++)
            deck_bits |= 1ULL << world.deck[d];
        CHECK((deck_bits & world.hand[o]) == 0,
              "sampled deck overlaps opponent hand");
        for (int i = 0; i < dist.n; i++)
            inclusion[i] += (int)((world.hand[o] >> dist.card[i]) & 1ULL);
    }
    for (int i = 0; i < dist.n; i++)
        CHECK(fabs((double)inclusion[i] / nsample - dist.marginal[i]) < 0.04,
              "sampled frequency for card %d disagrees with analytic marginal",
              dist.card[i]);

    const uint8_t perm[NSUIT] = { 1, 3, 0, 2, 4 };
    State ps;
    lc_permute_suits(&st, &ps, perm);
    BeliefDist pd;
    CHECK(belief_dist_init(net, &ps, ps.turn, 20, 1.15f, &pd),
          "initialize permuted belief");
    float mapped[NCARD];
    for (int i = 0; i < NCARD; i++) mapped[i] = -1.0f;
    for (int i = 0; i < pd.n; i++) mapped[pd.card[i]] = pd.marginal[i];
    for (int i = 0; i < dist.n; i++) {
        int c = lc_permute_card(dist.card[i], perm);
        CHECK(mapped[c] >= 0.0f &&
              fabsf(mapped[c] - dist.marginal[i]) < 2e-5f,
              "20-way belief is not affine-suit equivariant");
    }
    free(net);
}

static void test_rollout_policy_shortlist(void)
{
    Net *net = malloc(sizeof(*net));
    CHECK(net != NULL, "network allocation for rollout shortlist");
    if (!net) return;
    CHECK(net_load(net, "data/champion.bin") == 0,
          "load champion for rollout shortlist");

    Agent audit;
    agent_default(&audit, AG_ROLLOUT, net);
    audit.no_belief = 1;
    audit.dets = 2;
    audit.root_width = 8;
    audit.min_cand = 2;
    audit.gate = 0.98f;
    audit.symmetries = 20;
    audit.cand_mass = 0.995f;
    audit.batch_dets = 1;
    audit.playout_symmetries = 1;
    audit.override_k = 1.96f;
    audit.override_min = 1.0f;

    State p3 = reviewed_state(net, 3);
    SearchStats s3;
    Rng rng;
    rng_seed(&rng, 3003);
    (void)rollout_move(&audit, &p3, &rng, NULL, &s3);
    CHECK(s3.worlds == 0 && !s3.resolved && s3.n == 1,
          "near-certain ply 3 should skip comparative worlds");
    CHECK(named_move(s3.mv[0], "Bx", 0, 0),
          "ply 3 did not retain the policy leader");

    Agent singleton = audit;
    singleton.gate = 0.0f;
    singleton.min_cand = 1;
    singleton.root_width = 4;
    SearchStats s_singleton;
    rng_seed(&rng, 3003);
    (void)rollout_move(&singleton, &p3, &rng, NULL, &s_singleton);
    CHECK(s_singleton.worlds == 0 && s_singleton.n == 1 &&
          s_singleton.skip_reason == SEARCH_SKIP_POLICY_CONFIDENCE,
          "one-move policy shortlist wasted comparative rollout worlds");
    CHECK(named_move(s_singleton.mv[0], "Bx", 0, 0),
          "one-move policy shortlist changed the policy leader");

    Agent advisory = singleton;
    advisory.dets = 2;
    advisory.eval_cand = 3;
    SearchStats s_advisory;
    rng_seed(&rng, 3003);
    (void)rollout_move(&advisory, &p3, &rng, NULL, &s_advisory);
    CHECK(s_advisory.worlds == 2 && s_advisory.n == 3,
          "singleton shortcut discarded requested advisory Q targets");
    CHECK(named_move(s_advisory.mv[0], "Bx", 0, 0),
          "advisory evaluation changed the policy baseline");

    State p20 = reviewed_state(net, 20);
    SearchStats s20;
    rng_seed(&rng, 3020);
    (void)rollout_move(&audit, &p20, &rng, NULL, &s20);
    CHECK(s20.worlds >= 1 && s20.n >= 6 && s20.n < s20.nlegal,
          "ply 20 did not use a compact policy shortlist");
    Move pmv[MAX_MOVES];
    float prior[MAX_MOVES];
    int pn = policy_probs_sym(net, &p20, pmv, prior, NULL, 20);
    int order[MAX_MOVES];
    for (int i = 0; i < pn; i++) order[i] = i;
    for (int i = 0; i < pn; i++) {
        int best = i;
        for (int j = i + 1; j < pn; j++)
            if (prior[order[j]] > prior[order[best]]) best = j;
        int t = order[i]; order[i] = order[best]; order[best] = t;
    }
    int expected = 0;
    double expected_mass = 0.0;
    while (expected < 8 &&
           (expected < 2 || expected_mass < 0.995)) {
        expected_mass += prior[order[expected]];
        expected++;
    }
    CHECK(s20.n == expected,
          "shortlist has %d moves, expected policy prefix of %d",
          s20.n, expected);
    int saw_w3 = 0;
    for (int i = 0; i < s20.n; i++) {
        CHECK(MOVE_PACK(s20.mv[i]) == MOVE_PACK(pmv[order[i]]) &&
              fabs(s20.prior[i] - prior[order[i]]) < 1e-7,
              "shortlist entry %d is not the exact policy prefix", i);
        if (named_move(s20.mv[i], "W3", 1, 0)) saw_w3 = 1;
    }
    CHECK(saw_w3, "ply 20 shortlist omitted W3 discard");
    CHECK(fabs(s20.policy_mass - expected_mass) < 1e-6,
          "shortlist reports %.6f mass, expected %.6f",
          s20.policy_mass, expected_mass);
    CHECK(s20.delta[0] == 0.0 && s20.dse[0] == 0.0,
          "policy baseline paired statistics are nonzero");

    /* Phase gates must return the unmodified actor policy.  Root
     * dead-discard focusing cannot silently alter play before the configured
     * search window begins. */
    audit.prune_dom = 1;
    audit.ply_lo = 999;
    SearchStats gated;
    rng_seed(&rng, 3021);
    Move gated_move = rollout_move(&audit, &p20, &rng, NULL, &gated);
    CHECK(gated.worlds == 0 &&
          gated.skip_reason == SEARCH_SKIP_PLY_WINDOW &&
          gated.nlegal == pn,
          "ply gate did not report the raw legal policy state");
    CHECK(MOVE_PACK(gated_move) == MOVE_PACK(pmv[order[0]]) &&
          MOVE_PACK(gated.mv[0]) == MOVE_PACK(pmv[order[0]]),
          "root pruning changed the policy move outside the search window");
    audit.prune_dom = 0;
    audit.ply_lo = 0;

    /* The original audit forcibly added same-card pile-draw variants with
     * effectively zero prior.  Those exact W2 cases must stay outside the
     * top-policy shortlist at the reviewed positions. */
    for (int target = 8; target <= 10; target += 2) {
        State st = reviewed_state(net, target);
        SearchStats ss;
        rng_seed(&rng, (uint64_t)(4000 + target));
        (void)rollout_move(&audit, &st, &rng, NULL, &ss);
        for (int i = 0; i < ss.n; i++)
            CHECK(ss.mv[i].draw != 3,
                  "ply %d reintroduced a forced W2 draw variant", target);
    }

    /* Tempo expansion is bounded to draw variants of top policy actions and
     * phase-gated independently from the ordinary policy prefix. */
    Agent tempo = audit;
    tempo.root_width = 4;
    tempo.cand_mass = 0.0f;
    tempo.cand_floor = 0.02f;
    tempo.min_cand = 1;
    tempo.gate = 0.0f;
    tempo.override_k = 0.0f;
    tempo.draw_variant_cores = 2;
    tempo.draw_variant_deck_max = 0;
    SearchStats expanded;
    rng_seed(&rng, 5020);
    (void)rollout_move(&tempo, &p20, &rng, NULL, &expanded);
    CHECK(expanded.draw_variant_candidates >= 2 && expanded.n <= 8,
          "bounded top-action draw expansion added %d candidates (n=%d)",
          expanded.draw_variant_candidates, expanded.n);
    int saw_pile = 0;
    for (int i = 0; i < expanded.n; i++)
        if (expanded.mv[i].draw > 0) saw_pile = 1;
    CHECK(saw_pile, "top-action draw expansion added no pile candidate");

    tempo.draw_variant_deck_max = 1;
    SearchStats phase_blocked;
    rng_seed(&rng, 5020);
    (void)rollout_move(&tempo, &p20, &rng, NULL, &phase_blocked);
    CHECK(p20.deck_left > 1 && phase_blocked.draw_variant_candidates == 0,
          "draw-variant deck phase gate did not disable expansion");

    /* Generic pile variants are opportunistic.  They must use only slots
     * left after the focused semantic challengers, rather than filling the
     * eight-candidate budget first and silently hiding a targeted move. */
    Agent targeted = tempo;
    targeted.draw_variant_cores = 0;
    targeted.semantic_cand = 1;
    SearchStats targeted_stats;
    rng_seed(&rng, 5020);
    (void)rollout_move(&targeted, &p20, &rng, NULL, &targeted_stats);
    CHECK(targeted_stats.semantic_candidates >= 2 && targeted_stats.n < 8,
          "locked ply 20 did not expose semantic challengers (added=%d, n=%d)",
          targeted_stats.semantic_candidates, targeted_stats.n);

    Agent combined = targeted;
    combined.draw_variant_cores = 2;
    combined.draw_variant_deck_max = 0;
    SearchStats combined_stats;
    rng_seed(&rng, 5020);
    (void)rollout_move(&combined, &p20, &rng, NULL, &combined_stats);
    CHECK(combined_stats.semantic_candidates ==
              targeted_stats.semantic_candidates &&
          combined_stats.draw_variant_candidates > 0,
          "draw expansion crowded semantic candidates (semantic %d -> %d, "
          "draw=%d)",
          targeted_stats.semantic_candidates,
          combined_stats.semantic_candidates,
          combined_stats.draw_variant_candidates);
    for (int i = targeted_stats.trusted_candidates;
         i < targeted_stats.n; i++) {
        int retained = 0;
        for (int j = 0; j < combined_stats.n; j++)
            if (MOVE_PACK(targeted_stats.mv[i]) ==
                MOVE_PACK(combined_stats.mv[j])) {
                retained = 1;
                break;
            }
        CHECK(retained,
              "draw expansion omitted targeted semantic candidate %d", i);
    }

    Agent trusted = tempo;
    trusted.draw_variant_cores = 0;
    trusted.override_k = 3.5f;
    trusted.override_min = 2.0f;
    trusted.policy_prefix_mode = 1;
    SearchStats trusted_stats;
    rng_seed(&rng, 6020);
    Move trusted_move =
        rollout_move(&trusted, &p20, &rng, NULL, &trusted_stats);
    CHECK(trusted_stats.trusted_candidates == trusted_stats.n &&
          trusted_stats.selection_reference == trusted_stats.raw_best &&
          MOVE_PACK(trusted_move) ==
              MOVE_PACK(trusted_stats.mv[trusted_stats.selection_reference]),
          "trusted policy prefix did not select its numerical leader");
    CHECK(trusted_stats.trusted_prefix_override ==
              (trusted_stats.selection_reference != 0),
          "trusted-prefix override reporting is inconsistent");

    State p8 = reviewed_state(net, 8);
    Agent consensus = trusted;
    consensus.dets = 128;
    consensus.confirm_dets = 4;
    consensus.min_cand = 2;
    consensus.cand_floor = 0.01f;
    consensus.playout_sample = 2;
    consensus.playout_symmetries = 20;
    consensus.playout_prune = 1;
    consensus.policy_prefix_mode = 1;
    SearchStats discovery;
    rng_seed(&rng, 9008);
    (void)rollout_move(&consensus, &p8, &rng, NULL, &discovery);
    CHECK(discovery.selection_reference != 0,
          "locked ply 8 did not exercise a trusted-prefix proposal");
    int proposed = discovery.selection_reference;
    consensus.policy_prefix_mode = 2;
    SearchStats checked;
    rng_seed(&rng, 9008);
    Move checked_move =
        rollout_move(&consensus, &p8, &rng, NULL, &checked);
    CHECK(checked.prefix_confirm_worlds == 4 &&
          (checked.selection_reference == 0 ||
           checked.selection_reference == proposed) &&
          checked.prefix_confirmed ==
              (checked.selection_reference == proposed),
          "balanced fixed-world prefix confirmation is inconsistent");
    CHECK(MOVE_PACK(checked_move) ==
              MOVE_PACK(checked.mv[checked.selection_reference]),
          "consensus returned a move other than its selected reference");
    CHECK(checked.rdelta[checked.selection_reference] == 0.0 &&
          checked.rdse[checked.selection_reference] == 0.0,
          "selection-reference paired statistics are nonzero");

    /* Mode 3 is the same independent consensus contract, but each player gets
     * its own coherent, independently stratified suit orientation.  It must be
     * deterministic and may only keep the primary proposal or fall back. */
    consensus.policy_prefix_mode = 3;
    SearchStats role_a, role_b;
    rng_seed(&rng, 9008);
    Move role_move_a =
        rollout_move(&consensus, &p8, &rng, NULL, &role_a);
    rng_seed(&rng, 9008);
    Move role_move_b =
        rollout_move(&consensus, &p8, &rng, NULL, &role_b);
    CHECK(role_a.prefix_confirm_worlds == 4 &&
          role_a.prefix_proposed == proposed &&
          (role_a.selection_reference == 0 ||
           role_a.selection_reference == proposed) &&
          role_a.prefix_confirmed ==
              (role_a.selection_reference == proposed),
          "role-separated fixed-world prefix confirmation is inconsistent "
          "(worlds=%d proposed=%d expected=%d selected=%d confirmed=%d)",
          role_a.prefix_confirm_worlds, role_a.prefix_proposed, proposed,
          role_a.selection_reference, role_a.prefix_confirmed);
    CHECK(MOVE_PACK(role_move_a) == MOVE_PACK(role_move_b) &&
          role_a.selection_reference == role_b.selection_reference &&
          role_a.prefix_confirmed == role_b.prefix_confirmed,
          "role-separated prefix confirmation is not deterministic");

    /* A failed fresh prefix panel must remain authoritative even when the
     * separate low-prior override gate is disabled.  Previously override_k=0
     * bypassed mode 2 and returned the rejected primary-panel leader. */
    State p10 = reviewed_state(net, 10);
    Agent rejected = audit;
    rejected.dets = 8;
    rejected.root_width = 8;
    rejected.cand_mass = 0.0f;
    rejected.cand_floor = 0.001f;
    rejected.min_cand = 2;
    rejected.gate = 0.0f;
    rejected.override_k = 0.0f;
    rejected.playout_sample = 2;
    rejected.confirm_dets = 2;
    rejected.playout_prune = 1;
    rejected.policy_prefix_mode = 2;
    SearchStats rejected_stats;
    rng_seed(&rng, 1);
    Move rejected_move =
        rollout_move(&rejected, &p10, &rng, NULL, &rejected_stats);
    CHECK(rejected_stats.prefix_proposed != 0 &&
          rejected_stats.prefix_confirm_worlds == 2 &&
          !rejected_stats.prefix_confirmed &&
          rejected_stats.selection_reference == 0,
          "locked ply 10 did not reject its unstable prefix proposal");
    CHECK(MOVE_PACK(rejected_move) == MOVE_PACK(rejected_stats.mv[0]),
          "override_k=0 bypassed the rejected prefix consensus");

    /* Self-rollout must not distil the primary proposal after the independent
     * panel rejected it.  The behavior move and training target stay aligned. */
    Move target_mv[MAX_MOVES];
    float target_weight[MAX_MOVES];
    int target_mode = -1;
    int target_n = rollout_training_weights(
        &rejected_stats, rejected_move, 4.0f,
        target_mv, target_weight, &target_mode);
    CHECK(target_n == rejected_stats.n &&
          target_mode == ROLLOUT_TARGET_SELECTED,
          "rejected prefix proposal did not force a selection target");
    for (int i = 0; i < target_n; i++)
        CHECK(target_weight[i] ==
                  (MOVE_PACK(target_mv[i]) == MOVE_PACK(rejected_move)
                       ? 1.0f : 0.0f),
              "rejected prefix proposal retained training weight at row %d",
              i);

    /* When independent panels agree on the played move, keep their relative-Q
     * signal rather than needlessly collapsing every target to one-hot. */
    SearchStats agreed;
    memset(&agreed, 0, sizeof agreed);
    agreed.n = agreed.trusted_candidates = 3;
    agreed.prefix_proposed = agreed.selection_reference = 1;
    agreed.prefix_confirmed = 1;
    agreed.prefix_confirm_worlds = 32;
    agreed.mv[0] = (Move){ CARD_MAKE(0, 3), 1, 0 };
    agreed.mv[1] = (Move){ CARD_MAKE(1, 4), 0, 0 };
    agreed.mv[2] = (Move){ CARD_MAKE(2, 5), 1, 0 };
    agreed.q[0] = 1.0; agreed.q[1] = 5.0; agreed.q[2] = 2.0;
    agreed.prefix_q[0] = 2.0;
    agreed.prefix_q[1] = 6.0;
    agreed.prefix_q[2] = 3.0;
    target_mode = -1;
    target_n = rollout_training_weights(
        &agreed, agreed.mv[1], 4.0f,
        target_mv, target_weight, &target_mode);
    CHECK(target_n == agreed.n &&
          target_mode == ROLLOUT_TARGET_COHERENT &&
          target_weight[1] > target_weight[0] &&
          target_weight[1] > target_weight[2] &&
          target_weight[0] > 0.0f && target_weight[2] > 0.0f,
          "agreed panels did not preserve a coherent soft-Q target");
    free(net);
}

static void test_random_symmetry_policy_sample(void)
{
    Net *net = malloc(sizeof(*net));
    CHECK(net != NULL, "network allocation for random symmetry sample");
    if (!net) return;
    CHECK(net_load(net, "data/champion.bin") == 0,
          "load champion for random symmetry sample");

    State st = reviewed_state(net, 20);
    Move exact_mv[MAX_MOVES], sample_mv[MAX_MOVES];
    float exact[MAX_MOVES], sample[MAX_MOVES];
    double mean[MAX_MOVES] = { 0 };
    int n = policy_probs_sym(net, &st, exact_mv, exact, NULL, 20);
    Rng rng;
    rng_seed(&rng, 20260730);
    const int reps = 5000;
    for (int r = 0; r < reps; r++) {
        int sn = policy_probs_random_sym(net, &st, sample_mv, sample,
                                         &rng, 20);
        CHECK(sn == n, "random symmetry changed legal-move count");
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            CHECK(MOVE_PACK(sample_mv[i]) == MOVE_PACK(exact_mv[i]),
                  "random symmetry changed legal-move order");
            mean[i] += sample[i];
            sum += sample[i];
        }
        CHECK(fabs(sum - 1.0) < 2e-5,
              "random-symmetry policy probabilities sum to %.8f", sum);
    }
    for (int i = 0; i < n; i++) {
        mean[i] /= reps;
        CHECK(fabs(mean[i] - exact[i]) < 0.015,
              "random symmetry mean %.6f != exact ensemble %.6f",
              mean[i], exact[i]);
    }

    uint8_t perms[120][NSUIT];
    int nsym = suit_permutations(20, perms);
    double perm_mean[MAX_MOVES] = { 0 };
    for (int k = 0; k < nsym; k++) {
        float pv = 0.0f;
        int pn = policy_probs_perm(net, &st, sample_mv, sample, &pv,
                                   perms[k]);
        CHECK(pn == n, "fixed permutation changed legal-move count");
        double sum = 0.0;
        for (int i = 0; i < n; i++) {
            CHECK(MOVE_PACK(sample_mv[i]) == MOVE_PACK(exact_mv[i]),
                  "fixed permutation changed legal-move order");
            perm_mean[i] += sample[i];
            sum += sample[i];
        }
        CHECK(fabs(sum - 1.0) < 2e-5,
              "fixed-permutation probabilities sum to %.8f", sum);
    }
    for (int i = 0; i < n; i++)
        CHECK(fabs(perm_mean[i] / nsym - exact[i]) < 2e-6,
              "explicit permutation mean %.7f != exact ensemble %.7f",
              perm_mean[i] / nsym, exact[i]);
    free(net);
}

static void test_rollout_spec_tail(void)
{
    Agent a;
    agent_default(&a, AG_ROLLOUT, NULL);
    CHECK(a.playout_prune == -1 && a.draw_root_deck_max == 0 &&
          a.draw_playout_deck_max == 0,
          "rollout default no longer makes continuation pruning follow root");
    spec_parse("rolloutu:data/champion.bin:256:5:0.03:0.9:2:14:50:4:"
               "2:1:3.5:1.5:3:20:0.995:64:20:1:24:128:0:16:12:1:1:2:12:1:1.25:4:3",
               &a);
    CHECK(a.kind == AG_ROLLOUT && a.no_belief,
          "rolloutu kind/world model parsed incorrectly");
    CHECK(a.dets == 256 && a.root_width == 5 && a.min_cand == 2,
          "rollout core fields parsed incorrectly");
    CHECK(a.ply_lo == 14 && a.ply_hi == 50 && a.eval_cand == 4,
          "rollout ply fields parsed incorrectly");
    CHECK(a.win_q == 2 && a.prune_dom == 1 &&
          fabsf(a.override_k - 3.5f) < 1e-6f &&
          fabsf(a.override_min - 1.5f) < 1e-6f,
          "rollout selection fields parsed incorrectly");
    CHECK(a.playout_sample == 3 && a.symmetries == 20 &&
          fabsf(a.cand_mass - 0.995f) < 1e-6f &&
          a.batch_dets == 64 && a.playout_symmetries == 20,
          "rollout sampling fields parsed incorrectly");
    CHECK(a.discard_guard == 1 && a.deck_max == 24 &&
          a.confirm_dets == 128 && a.playout_prune == 0,
          "rollout confirmation tail parsed incorrectly");
    CHECK(a.plan_deck_max == 16 && a.plan_block_gap == 12 &&
          a.semantic_cand == 1 && a.confirm_exact5 == 1 &&
          a.draw_variant_cores == 2 && a.draw_variant_deck_max == 12 &&
          a.policy_prefix_mode == 1 &&
          fabsf(a.belief_alpha - 1.25f) < 1e-6f &&
          a.draw_root_deck_max == 4 &&
          a.draw_playout_deck_max == 3,
          "rollout planner/semantic tail parsed incorrectly");
    free((void *)a.net);

    Agent p;
    spec_parse("policy:data/champion.bin:0:20:16:12:4", &p);
    CHECK(p.kind == AG_POLICY && p.symmetries == 20 &&
          p.plan_deck_max == 16 && p.plan_block_gap == 12 &&
          p.draw_root_deck_max == 4 && p.draw_playout_deck_max == 0,
          "policy scheduling tail parsed incorrectly");
    free((void *)p.net);

    Agent champion;
    spec_parse(LC_CHAMPION_AGENT_SPEC, &champion);
    CHECK(champion.kind == AG_ROLLOUT && champion.no_belief &&
          champion.dets == 512 && champion.root_width == 5 &&
          fabsf(champion.cand_floor - 0.02f) < 1e-6f &&
          champion.gate == 0.0f && champion.min_cand == 1 &&
          champion.ply_lo == 14 && champion.ply_hi == 0 &&
          champion.eval_cand == 0 && champion.win_q == 0 &&
          champion.prune_dom == 0 &&
          fabsf(champion.override_k - 3.5f) < 1e-6f &&
          fabsf(champion.override_min - 2.0f) < 1e-6f &&
          champion.playout_sample == 2 && champion.symmetries == 20 &&
          champion.cand_mass == 0.0f && champion.batch_dets == 0 &&
          champion.playout_symmetries == 20 &&
          champion.discard_guard == 1 && champion.deck_max == 0 &&
          champion.confirm_dets == 512 && champion.playout_prune == 1 &&
          champion.plan_deck_max == 0 && champion.plan_block_gap == 0 &&
          champion.semantic_cand == 0 && champion.confirm_exact5 == 0 &&
          champion.draw_variant_cores == 0 &&
          champion.draw_variant_deck_max == 0 &&
          champion.policy_prefix_mode == 2 &&
          fabsf(champion.belief_alpha - 1.0f) < 1e-6f &&
          champion.draw_root_deck_max == 0 &&
          champion.draw_playout_deck_max == 0,
          "maintained champion spec drifted from its locked configuration");
    free((void *)champion.net);

    /* Training's live-network form must not have a private, fixed-size copy of
     * the rollout parser.  This deliberately exceeds the old 128-byte buffer
     * and checks every field that used to be silently lost after
     * playout_prune. */
    Net *live = (Net *)malloc(sizeof *live);
    CHECK(live != NULL, "live-network parser fixture allocation failed");
    const char *self_spec =
        "selfrollout:256:5:0.0123456789:0.8765432109:7:14:299:8:2:1:"
        "3.5000000001:1.5000000001:4:20:0.9950000001:128:20:1:24:128:"
        "0:16:12:1:1:2:12:3:1.25:4:3";
    CHECK(strlen(self_spec) > 128,
          "selfrollout regression no longer exceeds the old parser buffer");
    Agent live_rollout;
    spec_parse_selfrollout(self_spec, live, &live_rollout);
    CHECK(live_rollout.kind == AG_ROLLOUT && live_rollout.net == live,
          "selfrollout did not preserve its caller-owned live network");
    CHECK(live_rollout.dets == 256 &&
          live_rollout.root_width == 5 &&
          live_rollout.ply_hi == 299 &&
          live_rollout.batch_dets == 128 &&
          live_rollout.confirm_dets == 128,
          "selfrollout long core/confirmation fields were truncated");
    CHECK(live_rollout.playout_prune == 0 &&
          live_rollout.playout_sample == 4 &&
          live_rollout.plan_deck_max == 16 &&
          live_rollout.plan_block_gap == 12 &&
          live_rollout.semantic_cand == 1 &&
          live_rollout.confirm_exact5 == 1 &&
          live_rollout.draw_variant_cores == 2 &&
          live_rollout.draw_variant_deck_max == 12 &&
          live_rollout.policy_prefix_mode == 3 &&
          fabsf(live_rollout.belief_alpha - 1.25f) < 1e-6f &&
          live_rollout.draw_root_deck_max == 4 &&
          live_rollout.draw_playout_deck_max == 3,
          "selfrollout planner/semantic/consensus tail was not parsed");
    free(live);
}

static void test_information_preserving_scheduler(void)
{
    State st;
    memset(&st, 0, sizeof st);
    st.turn = 0;
    st.deck_left = 11;
    const int w7 = CARD_MAKE(2, 8);
    const int w9 = CARD_MAKE(2, 10);
    const int w10 = CARD_MAKE(2, 11);
    const int g7 = CARD_MAKE(3, 8);
    const int g10 = CARD_MAKE(3, 11);
    const int r10 = CARD_MAKE(4, 11);
    const int b9 = CARD_MAKE(1, 10);
    const int rw = CARD_MAKE(4, 0);
    st.hand[0] = (1ULL << b9) | (1ULL << w7) | (1ULL << w9) |
                 (1ULL << w10) | (1ULL << g7) | (1ULL << g10) |
                 (1ULL << rw) | (1ULL << r10);
    st.hand_n[0] = HAND_SIZE;
    st.exp_wager[0][3] = 2;
    st.exp_n[0][3] = 3;
    st.exp_top[0][3] = 3;
    st.exp_sum[0][3] = 3;
    st.exp_n[0][4] = 3;
    st.exp_top[0][4] = 8;
    st.exp_sum[0][4] = 15; /* R2, R5, R8 */

    /* Publicly locate the lower cards that should not contribute to the
     * information-blocking cost.  W4/W5 and G4 stay unseen. */
    st.played[1] = (1ULL << CARD_MAKE(2, 4)) | /* W3 */
                   (1ULL << CARD_MAKE(2, 7)) | /* W6 */
                   (1ULL << CARD_MAKE(3, 6)) | /* G5 */
                   (1ULL << CARD_MAKE(3, 7)) | /* G6 */
                   (1ULL << CARD_MAKE(3, 9)) | /* G8 */
                   (1ULL << CARD_MAKE(4, 10)); /* R9 */
    st.pile[2][0] = CARD_MAKE(2, 3); /* W2 */
    st.pile_n[2] = 1;
    st.discarded = 1ULL << CARD_MAKE(2, 3);

    HandPlan plan;
    hand_plan_build(&st, 0, 6, &plan);
    CHECK(plan.min_cards == 6 && plan.score - plan.base_score == 67,
          "scheduler did not find the six-card guaranteed finish");
    CHECK((plan.first_cards & (1ULL << w7)) &&
          (plan.first_cards & (1ULL << g7)) &&
          (plan.first_cards & (1ULL << r10)),
          "scheduler lost a commuting first play");

    Move mv[3] = {
        { (uint8_t)g7, 0, 0 },
        { (uint8_t)r10, 0, 0 },
        { (uint8_t)w7, 0, 0 },
    };
    float prob[3] = { 0.50f, 0.25f, 0.09f };
    int order[3] = { 0, 1, 2 };
    int pick = hand_plan_conservative_choose(
        &st, 0, mv, prob, order, 3, 6, 12);
    CHECK(pick == 1,
          "scheduler did not preserve lower-card information with R10");
    CHECK(hand_plan_conservative_choose(
              &st, 0, mv, prob, order, 3, 6, 13) == -1,
          "scheduler ignored its conservative block-gap threshold");

    /* A third blue wager is not part of the best guaranteed schedule: two
     * wagers plus B3/B6/B8 finish at -9, while adding the third finishes at
     * -12.  The old policy nevertheless put 97% on playing it at showcase
     * ply 32.  The scheduler must prefer a cost-free suit 10 even though a
     * wager has no ordinary lower-card blocking cost. */
    State commit;
    memset(&commit, 0, sizeof commit);
    commit.turn = 1;
    commit.deck_left = 16;
    const int bx = CARD_MAKE(1, 2);
    const int b3 = CARD_MAKE(1, 4);
    const int b6 = CARD_MAKE(1, 7);
    const int b8 = CARD_MAKE(1, 9);
    const int wx = CARD_MAKE(2, 0);
    const int w2 = CARD_MAKE(2, 3);
    const int w10b = CARD_MAKE(2, 11);
    const int r10b = CARD_MAKE(4, 11);
    commit.hand[1] = (1ULL << bx) | (1ULL << b3) | (1ULL << b6) |
                     (1ULL << b8) | (1ULL << wx) | (1ULL << w2) |
                     (1ULL << w10b) | (1ULL << r10b);
    commit.hand_n[1] = HAND_SIZE;
    commit.exp_wager[1][1] = 2;
    commit.exp_n[1][1] = 2;
    commit.exp_wager[1][2] = 1;
    commit.exp_n[1][2] = 5;
    commit.exp_top[1][2] = 9;
    commit.exp_sum[1][2] = 28;
    commit.exp_n[1][4] = 3;
    commit.exp_top[1][4] = 7;
    commit.exp_sum[1][4] = 16;
    Move commitment[3] = {
        { (uint8_t)bx, 0, 0 },
        { (uint8_t)w10b, 0, 0 },
        { (uint8_t)r10b, 0, 0 },
    };
    float commitment_prob[3] = { 0.97f, 0.02f, 0.01f };
    int commitment_order[3] = { 0, 1, 2 };
    CHECK(hand_plan_conservative_choose(
              &commit, 1, commitment, commitment_prob,
              commitment_order, 3, 8, 12) == 1,
          "scheduler did not reject a score-losing third wager commitment");
}

static void test_information_safe_draw_planner(void)
{
    State st;
    memset(&st, 0, sizeof st);
    st.turn = 0;
    st.deck_left = 2;
    const int y2 = CARD_MAKE(0, 3);
    const int y10 = CARD_MAKE(0, 11);
    const int cards[7] = {
        CARD_MAKE(1, 4), CARD_MAKE(1, 7), CARD_MAKE(2, 5),
        CARD_MAKE(2, 8), CARD_MAKE(3, 6), CARD_MAKE(4, 9),
        CARD_MAKE(4, 11),
    };
    st.hand[0] = 1ULL << y2;
    for (int i = 0; i < 7; i++) st.hand[0] |= 1ULL << cards[i];
    st.hand_n[0] = HAND_SIZE;
    st.pile[0][0] = (uint8_t)y10;
    st.pile_n[0] = 1;
    st.discarded = 1ULL << y10;

    Move deck = { (uint8_t)y2, 0, 0 };
    Move pile = { (uint8_t)y2, 0, 1 };
    double deck_score =
        hand_plan_expected_score_after_move(&st, 0, deck);
    double pile_score =
        hand_plan_expected_score_after_move(&st, 0, pile);
    CHECK(pile_score > deck_score + 9.0,
          "draw planner missed the exact extra-turn pile finish");
    Move sources[4] = {
        deck,
        pile,
        { (uint8_t)cards[0], 1, 0 },
        { (uint8_t)cards[0], 1, 1 },
    };
    float source_prior[4] = { 0.8f, 0.1f, 0.99f, 0.99f };
    CHECK(hand_plan_choose_draw_source(
              &st, 0, sources, source_prior, 4, 0) == 1,
          "draw planner did not replace the top action's bad draw source");
    CHECK(hand_plan_choose_draw_source(
              &st, 0, sources, source_prior, 4, 0) != 3,
          "draw planner escaped to an unrelated high-prior action");

    /* The two deployment controls must not alias.  With a zero network the
     * raw deterministic policy takes the first legal draw (deck).  A
     * continuation-only threshold may not alter that root move, whereas the
     * root threshold must apply the information-set draw repair. */
    Net *zero = calloc(1, sizeof *zero);
    CHECK(zero != NULL, "cannot allocate zero-network draw-planner fixture");
    Agent policy;
    agent_default(&policy, AG_POLICY, zero);
    policy.symmetries = 1;
    policy.draw_playout_deck_max = 2;
    Rng policy_rng;
    rng_seed(&policy_rng, 730032);
    Move continuation_only = agent_move(&policy, &st, &policy_rng);
    CHECK(continuation_only.draw == 0,
          "continuation draw threshold leaked into root policy play");
    policy.draw_playout_deck_max = 0;
    policy.draw_root_deck_max = 2;
    rng_seed(&policy_rng, 730032);
    Move root_repaired = agent_move(&policy, &st, &policy_rng);
    CHECK(root_repaired.card == continuation_only.card &&
          root_repaired.discard == continuation_only.discard &&
          root_repaired.draw == 1,
          "root draw threshold did not repair only the draw source");
    free(zero);

    st.deck_left = 1;
    CHECK(hand_plan_choose_draw_source(
              &st, 0, sources, source_prior, 4, 1) == 0,
          "draw planner violated last-deck weak dominance");
    st.deck_left = 2;

    /* Neither the real top deck card nor an unobserved assignment to the
     * opponent's hand may affect the information-set expectation. */
    State hidden_variant = st;
    st.deck[0] = (uint8_t)CARD_MAKE(3, 11);
    hidden_variant.deck[0] = (uint8_t)CARD_MAKE(1, 11);
    st.hand[1] = 1ULL << CARD_MAKE(3, 11);
    hidden_variant.hand[1] = 1ULL << CARD_MAKE(1, 11);
    st.hand_n[1] = hidden_variant.hand_n[1] = 1;
    CHECK(fabs(hand_plan_expected_score_after_move(&st, 0, deck) -
               hand_plan_expected_score_after_move(
                   &hidden_variant, 0, deck)) < 1e-12,
          "draw planner leaked hidden deck/hand assignment");

    /* Cross-check the optimized one-pass deck expectation against the direct
     * definition on real reachable positions. */
    Rng rng;
    rng_seed(&rng, 730031);
    for (int game = 0; game < 8; game++) {
        State live;
        lc_deal(&live, &rng);
        for (int ply = 0; ply < 34 && !live.over; ply++) {
            Move legal[MAX_MOVES];
            int nlegal = lc_moves(&live, legal);
            int chosen = (int)rng_below(&rng, (uint32_t)nlegal);
            if (legal[chosen].draw == 0 && live.deck_left <= 12) {
                State played = live;
                int p = live.turn;
                lc_apply_play(&played, legal[chosen]);
                uint8_t unseen[NCARD];
                int nunseen = 0;
                lc_unseen(&played, p, unseen, &nunseen);
                double brute = 0.0;
                for (int i = 0; i < nunseen; i++) {
                    State after = played;
                    lc_apply_draw(&after, legal[chosen], unseen[i]);
                    HandPlan plan;
                    hand_plan_build(&after, p, after.deck_left / 2, &plan);
                    brute += plan.score;
                }
                if (nunseen > 0) brute /= nunseen;
                CHECK(fabs(brute - hand_plan_expected_score_after_move(
                                       &live, p, legal[chosen])) < 1e-12,
                      "optimized draw expectation differs from brute force");
            }
            lc_apply(&live, legal[chosen]);
        }
    }
}

static void test_dead_discard_focus_equivariance(void)
{
    State st;
    memset(&st, 0, sizeof st);
    const int y2 = CARD_MAKE(0, 3);
    const int w5 = CARD_MAKE(2, 6);
    const int r2 = CARD_MAKE(4, 3);
    st.hand[0] = (1ULL << y2) | (1ULL << w5) | (1ULL << r2);
    st.hand_n[0] = 3;
    st.exp_top[0][0] = st.exp_top[1][0] = 2;
    st.exp_top[0][4] = st.exp_top[1][4] = 2;
    uint64_t dead = lc_dead_cards(&st);
    Move yd = { (uint8_t)y2, 1, 0 };
    Move wd = { (uint8_t)w5, 1, 0 };
    Move rd = { (uint8_t)r2, 1, 0 };
    CHECK(!lc_discard_dominated(&st, yd, dead) &&
          !lc_discard_dominated(&st, rd, dead),
          "one safe discard was arbitrarily preferred by card id");
    CHECK(lc_discard_dominated(&st, wd, dead),
          "live discard was not focused away when safe discards exist");

    const uint8_t swap_y_r[NSUIT] = { 4, 1, 2, 3, 0 };
    State ps;
    lc_permute_suits(&st, &ps, swap_y_r);
    uint64_t pdead = lc_dead_cards(&ps);
    CHECK(!lc_discard_dominated(&ps, lc_permute_move(yd, swap_y_r), pdead) &&
          !lc_discard_dominated(&ps, lc_permute_move(rd, swap_y_r), pdead) &&
          lc_discard_dominated(&ps, lc_permute_move(wd, swap_y_r), pdead),
          "dead-discard focus is not suit equivariant");
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

static void test_rollout_match_thread_determinism(void)
{
    Net *net = malloc(sizeof(*net));
    CHECK(net != NULL, "network allocation for rollout thread determinism");
    if (!net) return;
    CHECK(net_load(net, "data/champion.bin") == 0,
          "load champion for rollout thread determinism");

    Agent search, policy;
    agent_default(&search, AG_ROLLOUT, net);
    search.no_belief = 1;
    search.dets = 2;
    search.confirm_dets = 2;
    search.root_width = 2;
    search.min_cand = 2;
    search.ply_lo = 14;
    search.override_k = 1.96f;
    search.override_min = 1.0f;
    search.playout_symmetries = 20;
    agent_default(&policy, AG_POLICY, net);

    for (int mode = 2; mode <= 5; mode++) {
        search.playout_sample = mode;
        /* The third loop reuses measured mode-2 discovery but exercises
         * role-separated fresh consensus.  The fourth combines it with
         * independently stratified fixed roles in discovery itself. */
        if (mode == 4) {
            search.playout_sample = 2;
            search.policy_prefix_mode = 3;
        } else if (mode == 5) {
            search.playout_sample = 4;
            search.policy_prefix_mode = 3;
        } else {
            search.policy_prefix_mode = mode == 3 ? 2 : 0;
        }
        search.confirm_dets = 16;
        MatchResult one, four;
        match_run_r(&search, &policy, 2, 1,
                    830083 + (uint64_t)mode, MATCH_ROUNDS, &one);
        match_run_r(&search, &policy, 2, 4,
                    830083 + (uint64_t)mode, MATCH_ROUNDS, &four);
        CHECK(one.pairs == four.pairs && one.games == four.games &&
              one.margin == four.margin && one.margin_se == four.margin_se &&
              one.winrate == four.winrate &&
              one.winrate_se == four.winrate_se &&
              one.points_a == four.points_a &&
              one.points_b == four.points_b && one.plies == four.plies &&
              one.wins == four.wins && one.losses == four.losses &&
              one.draws == four.draws,
              "rollout mode %d differs between one and four threads", mode);
    }
    free(net);
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
    test_belief_distribution();
    test_rollout_policy_shortlist();
    test_random_symmetry_policy_sample();
    test_rollout_spec_tail();
    test_information_preserving_scheduler();
    test_information_safe_draw_planner();
    test_dead_discard_focus_equivariance();
    test_wager_interaction_head();
    test_wager_parameter_projection();
    test_wager_tied_gradients();
    test_pile_order_features();
    test_match_thread_determinism();
    test_rollout_match_thread_determinism();
    test_suit_symmetry_ensemble();
    if (failures == 0) {
        printf("all runtime regression tests passed\n");
        return 0;
    }
    printf("%d failures\n", failures);
    return 1;
}
