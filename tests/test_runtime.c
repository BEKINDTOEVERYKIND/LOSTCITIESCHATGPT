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

extern int rollout_test_late_cache_order(const Net *net, const State *st);
extern int rollout_test_unique_late_assignments(
    const State *st, int requested, int *support_out);

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

static int same_test_action(Move a, Move b)
{
    if (a.discard != b.discard) return 0;
    if (CARD_IS_WAGER(a.card) && CARD_IS_WAGER(b.card))
        return CARD_SUIT(a.card) == CARD_SUIT(b.card);
    return a.card == b.card;
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

static void test_near_greedy_confirmation_sampler(void)
{
    Move mv[4] = {
        { (uint8_t)CARD_MAKE(0, 3), 0, 0 },
        { (uint8_t)CARD_MAKE(1, 4), 1, 0 },
        { (uint8_t)CARD_MAKE(2, 5), 0, 0 },
        { (uint8_t)CARD_MAKE(3, 6), 1, 0 }
    };
    float prob[4] = { 0.7000f, 0.2990f, 0.0009f, 0.0001f };
    CHECK(rollout_near_greedy_pick(mv, prob, 4, 0.0f, 1, 0, 0) == 0,
          "zero confirmation temperature did not preserve argmax");

    Move reversed[4];
    float reversed_prob[4];
    for (int i = 0; i < 4; i++) {
        reversed[i] = mv[3 - i];
        reversed_prob[i] = prob[3 - i];
    }
    for (uint64_t seed = 1; seed <= 1000; seed++) {
        int a = rollout_near_greedy_pick(
            mv, prob, 4, 5.0f, seed, 7, 1);
        int shared_only = rollout_near_greedy_pick(
            mv, prob, 2, 5.0f, seed, 7, 1);
        int repeat = rollout_near_greedy_pick(
            mv, prob, 4, 5.0f, seed, 7, 1);
        int b = rollout_near_greedy_pick(
            reversed, reversed_prob, 4, 5.0f, seed, 7, 1);
        CHECK(a == repeat, "near-greedy confirmation is not deterministic");
        CHECK(a == shared_only,
              "unrelated low-mass moves changed shared confirmation noise");
        CHECK(a == 0 || a == 1,
              "confirmation sampled outside the top-99.5%% mass prefix");
        CHECK(MOVE_PACK(mv[a]) == MOVE_PACK(reversed[b]),
              "move-keyed confirmation noise depends on legal-move order");
    }

    /* A probability tie crossing the 99.5% boundary is one equivalence
     * class: legal-move enumeration order must not decide which tied action
     * is eligible. */
    Move tied[3] = {
        { (uint8_t)CARD_MAKE(0, 3), 0, 0 },
        { (uint8_t)CARD_MAKE(1, 4), 1, 0 },
        { (uint8_t)CARD_MAKE(2, 5), 0, 0 }
    };
    Move tied_swapped[3] = { tied[0], tied[2], tied[1] };
    float tied_prob[3] = { 0.994f, 0.003f, 0.003f };
    for (uint64_t seed = 1; seed <= 1000; seed++) {
        int a = rollout_near_greedy_pick(
            tied, tied_prob, 3, 5.0f, seed, 9, 0);
        int b = rollout_near_greedy_pick(
            tied_swapped, tied_prob, 3, 5.0f, seed, 9, 0);
        CHECK(MOVE_PACK(tied[a]) == MOVE_PACK(tied_swapped[b]),
              "99.5%% cutoff split an exact probability tie");
    }

    /* The three physical wager IDs are publicly indistinguishable.  They
     * must receive the same stateless Gumbel key against an unchanged rival. */
    Move wager0[2] = {
        { (uint8_t)CARD_MAKE(0, 0), 0, 0 },
        { (uint8_t)CARD_MAKE(1, 4), 1, 0 }
    };
    Move wager1[2] = { wager0[0], wager0[1] };
    Move wager2[2] = { wager0[0], wager0[1] };
    wager1[0].card = (uint8_t)CARD_MAKE(0, 1);
    wager2[0].card = (uint8_t)CARD_MAKE(0, 2);
    float wager_prob[2] = { 0.5f, 0.5f };
    for (uint64_t seed = 1; seed <= 1000; seed++) {
        int a = rollout_near_greedy_pick(
            wager0, wager_prob, 2, 1.0f, seed, 11, 1);
        int b = rollout_near_greedy_pick(
            wager1, wager_prob, 2, 1.0f, seed, 11, 1);
        int c = rollout_near_greedy_pick(
            wager2, wager_prob, 2, 1.0f, seed, 11, 1);
        CHECK(a == b && a == c,
              "physical wager ID changed stateless confirmation noise");
    }
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

static State exact_terminal_test_state(void)
{
    State st;
    memset(&st, 0, sizeof st);
    st.turn = 0;
    st.deck_pos = 37;
    st.deck_left = 1;
    st.deck[st.deck_pos] = (uint8_t)CARD_MAKE(4, 11);

    /* Yellow is already a -18 expedition.  Playing Y10 improves it to -8;
     * every other play starts an additional negative expedition, while every
     * discard leaves the -18 unchanged.  Y10 -> deck is therefore the unique
     * exact terminal optimum across the complete legal action-core set. */
    int y2 = CARD_MAKE(0, 3);
    st.played[0] = UINT64_C(1) << y2;
    st.exp_top[0][0] = 2;
    st.exp_n[0][0] = 1;
    st.exp_sum[0][0] = 2;
    const int hand0[HAND_SIZE] = {
        CARD_MAKE(0, 11), CARD_MAKE(1, 4), CARD_MAKE(1, 8),
        CARD_MAKE(2, 6), CARD_MAKE(2, 9), CARD_MAKE(3, 0),
        CARD_MAKE(3, 10), CARD_MAKE(4, 5)
    };
    const int hand1[HAND_SIZE] = {
        CARD_MAKE(0, 4), CARD_MAKE(0, 5), CARD_MAKE(1, 3),
        CARD_MAKE(1, 5), CARD_MAKE(2, 3), CARD_MAKE(2, 4),
        CARD_MAKE(3, 3), CARD_MAKE(4, 3)
    };
    for (int i = 0; i < HAND_SIZE; i++) {
        st.hand[0] |= UINT64_C(1) << hand0[i];
        st.hand[1] |= UINT64_C(1) << hand1[i];
    }
    st.hand_n[0] = st.hand_n[1] = HAND_SIZE;

    const int pile_card[3] = {
        CARD_MAKE(0, 6), CARD_MAKE(1, 6), CARD_MAKE(2, 7)
    };
    for (int s = 0; s < 3; s++) {
        st.pile[s][0] = (uint8_t)pile_card[s];
        st.pile_n[s] = 1;
        st.discarded |= UINT64_C(1) << pile_card[s];
    }
    return st;
}

static int exact_terminal_oracle(const State *st, const Move *mv, int n,
                                 int objective, double *out)
{
    int best = -1;
    double best_value = -INFINITY;
    for (int i = 0; i < n; i++) {
        if (mv[i].draw != 0) continue;
        State terminal = *st;
        lc_apply(&terminal, mv[i]);
        if (!terminal.over || terminal.deck_left != 0) continue;
        double value = rollout_terminal_objective(
            &terminal, st->turn, objective);
        if (best < 0 || value > best_value ||
            (value == best_value && MOVE_PACK(mv[i]) < MOVE_PACK(mv[best]))) {
            best = i;
            best_value = value;
        }
    }
    if (best >= 0 && out) *out = best_value;
    return best;
}

static void test_rollout_exact_terminal_choice(void)
{
    State st = exact_terminal_test_state();
    Move mv[MAX_MOVES];
    int n = lc_moves(&st, mv);
    float prior[MAX_MOVES];
    for (int i = 0; i < n; i++) prior[i] = 0.0f;

    int cores = 0, deck_variants = 0;
    for (int i = 0; i < n; i++) {
        if (mv[i].draw == 0) deck_variants++;
        int first = 1;
        for (int j = 0; j < i; j++)
            if (same_test_action(mv[i], mv[j])) {
                first = 0;
                break;
            }
        cores += first;
    }
    CHECK(deck_variants == cores,
          "last-deck legal set has %d cores but %d deck variants",
          cores, deck_variants);

    double expected = 0.0, actual = 0.0;
    int oracle = exact_terminal_oracle(&st, mv, n, 0, &expected);
    int selected = rollout_exact_terminal_choice(
        &st, mv, prior, n, 0, &actual);
    CHECK(oracle >= 0 && selected >= 0,
          "exact terminal solver rejected a one-card deck");
    if (oracle >= 0 && selected >= 0) {
        CHECK(MOVE_PACK(mv[selected]) == MOVE_PACK(mv[oracle]),
              "exact terminal solver did not maximize all legal cores");
        CHECK(named_move(mv[selected], "Y10", 0, 0),
              "exact terminal solver chose the wrong unique optimum");
        CHECK(fabs(actual - expected) < 1e-12,
              "exact terminal objective %.9f != oracle %.9f",
              actual, expected);
    }

    /* Neither the identity of the unseen last card nor any private opponent
     * cards can affect a turn that ends before the drawn card is playable. */
    State hidden = st;
    hidden.deck[hidden.deck_pos] = (uint8_t)CARD_MAKE(1, 11);
    hidden.hand[1] = 0;
    const int alternative_hand[HAND_SIZE] = {
        CARD_MAKE(0, 0), CARD_MAKE(0, 1), CARD_MAKE(1, 0),
        CARD_MAKE(1, 1), CARD_MAKE(2, 0), CARD_MAKE(3, 4),
        CARD_MAKE(4, 8), CARD_MAKE(4, 10)
    };
    for (int i = 0; i < HAND_SIZE; i++)
        hidden.hand[1] |= UINT64_C(1) << alternative_hand[i];
    Move hidden_mv[MAX_MOVES];
    int hidden_n = lc_moves(&hidden, hidden_mv);
    double hidden_objective = 0.0;
    int hidden_selected = rollout_exact_terminal_choice(
        &hidden, hidden_mv, NULL, hidden_n, 0, &hidden_objective);
    CHECK(hidden_selected >= 0 && selected >= 0 &&
          MOVE_PACK(hidden_mv[hidden_selected]) == MOVE_PACK(mv[selected]) &&
          fabs(hidden_objective - actual) < 1e-12,
          "hidden last card or opponent hand changed exact terminal choice");

    int suboptimal_pile = -1;
    for (int i = 0; i < n; i++)
        if (mv[i].draw != 0 && !same_test_action(mv[i], mv[oracle])) {
            suboptimal_pile = i;
            break;
        }
    int policy_terminal = rollout_policy_terminal_choice(
        mv, n, suboptimal_pile);
    CHECK(suboptimal_pile >= 0 && policy_terminal >= 0 &&
          mv[policy_terminal].draw == 0 &&
          same_test_action(mv[policy_terminal], mv[suboptimal_pile]) &&
          !same_test_action(mv[policy_terminal], mv[oracle]),
          "policy-terminal control optimized or changed the selected core");

    /* A full-move interaction can make one action best only with a pile while
     * a different action is best conditional on the required deck draw.  The
     * production forced-progress chooser must honor that conditional policy;
     * the mode-3 propagation control above intentionally keeps the old core. */
    float conditional_score[MAX_MOVES];
    for (int i = 0; i < n; i++) conditional_score[i] = 0.0f;
    conditional_score[suboptimal_pile] = 0.90f;
    conditional_score[policy_terminal] = 0.01f;
    conditional_score[oracle] = 0.20f;
    int conditional = rollout_policy_deck_choice(
        &st, mv, conditional_score, n, 0);
    CHECK(conditional == oracle && conditional != policy_terminal,
          "forced deck policy ignored the card/action x draw interaction");

    int wager_pile = -1;
    for (int i = 0; i < n; i++)
        if (mv[i].draw != 0 && CARD_IS_WAGER(mv[i].card)) {
            wager_pile = i;
            break;
        }
    int wager_terminal = rollout_policy_terminal_choice(mv, n, wager_pile);
    CHECK(wager_pile >= 0 && wager_terminal >= 0 &&
          mv[wager_terminal].draw == 0 &&
          same_test_action(mv[wager_terminal], mv[wager_pile]),
          "policy-terminal control failed to canonicalize a wager core");

    st.round = MATCH_ROUNDS - 1;
    st.cum[0] = 13;
    selected = rollout_exact_terminal_choice(
        &st, mv, prior, n, 2, &actual);
    oracle = exact_terminal_oracle(&st, mv, n, 2, &expected);
    CHECK(selected >= 0 && oracle >= 0 &&
          MOVE_PACK(mv[selected]) == MOVE_PACK(mv[oracle]),
          "exact terminal solver ignored the final-round objective");
    CHECK(fabs(actual - 50.25) < 1e-12 && fabs(actual - expected) < 1e-12,
          "final-round hybrid terminal objective is %.9f, expected 50.25",
          actual);

    Net *net = malloc(sizeof(*net));
    CHECK(net != NULL, "network allocation for exact terminal rollout");
    if (!net) return;
    net_zero(net);
    /* Make the raw policy strongly prefer the wrong semantic action and a
     * pile draw.  The root solver must still return the exact terminal move. */
    int b3 = CARD_MAKE(1, 4);
    net->bplay[b3 * 2 + 1] = 10.0f;
    net->bdraw[1] = 5.0f;
    Agent agent;
    agent_default(&agent, AG_ROLLOUT, net);
    agent.symmetries = 1;
    agent.dets = 4096;
    agent.win_q = 2;
    Rng rng, before;
    rng_seed(&rng, UINT64_C(0xE1A57));
    before = rng;
    SearchStats stats;
    Move root = rollout_move(&agent, &st, &rng, NULL, &stats);
    CHECK(MOVE_PACK(root) == MOVE_PACK(mv[oracle]),
          "root rollout did not return the exact terminal choice");
    CHECK(memcmp(&rng, &before, sizeof rng) == 0,
          "exact terminal root consumed RNG state");
    CHECK(stats.worlds == 0 && stats.skip_reason == SEARCH_SKIP_LAST_DECK &&
          stats.metric_kind == SEARCH_METRIC_LAST_DECK_RULE &&
          stats.deck_end_baseline && stats.resolved,
          "exact terminal root reported stochastic/incomplete search");
    CHECK(stats.n >= 1 && stats.visits[0] == 0.0 &&
          MOVE_PACK(stats.mv[0]) == MOVE_PACK(root) &&
          fabs(stats.q[0] - expected) < 1e-12,
          "exact terminal root stats do not attest the returned move");
    free(net);

    Agent hrollout;
    agent_default(&hrollout, AG_ROLLOUT, NULL);
    rng_seed(&rng, UINT64_C(0xE1A57));
    before = rng;
    Move hroot = rollout_move(&hrollout, &st, &rng, NULL, &stats);
    CHECK(MOVE_PACK(hroot) == MOVE_PACK(mv[oracle]),
          "heuristic rollout did not use the exact terminal choice");
    CHECK(memcmp(&rng, &before, sizeof rng) == 0,
          "heuristic exact-terminal root consumed draw-sampling RNG");
}

static void test_rollout_exact_terminal_randomized(void)
{
    Rng rng;
    rng_seed(&rng, UINT64_C(0x1A57DEC1));
    for (int trial = 0; trial < 100; trial++) {
        State st;
        lc_deal(&st, &rng);
        while (!st.over && st.deck_left > 1) {
            Move legal[MAX_MOVES];
            int n = lc_moves(&st, legal);
            int deck_index[MAX_MOVES], ndeck = 0;
            for (int i = 0; i < n; i++)
                if (legal[i].draw == 0) deck_index[ndeck++] = i;
            CHECK(ndeck > 0,
                  "reachable terminal fixture has no deck-draw move");
            if (ndeck <= 0) break;
            int chosen = deck_index[rng_below(&rng, (uint32_t)ndeck)];
            lc_apply(&st, legal[chosen]);
        }
        CHECK(!st.over && st.deck_left == 1,
              "random reachable fixture did not stop at one deck card");
        if (st.over || st.deck_left != 1) continue;

        st.round = trial % MATCH_ROUNDS;
        st.cum[0] = (int16_t)((int)rng_below(&rng, 201) - 100);
        st.cum[1] = (int16_t)((int)rng_below(&rng, 201) - 100);
        int objective = st.round == MATCH_ROUNDS - 1 ? 2 : 0;
        Move mv[MAX_MOVES];
        int n = lc_moves(&st, mv);
        double expected = 0.0, actual = 0.0;
        int oracle = exact_terminal_oracle(
            &st, mv, n, objective, &expected);
        int selected = rollout_exact_terminal_choice(
            &st, mv, NULL, n, objective, &actual);
        CHECK(oracle >= 0 && selected >= 0 &&
              MOVE_PACK(mv[oracle]) == MOVE_PACK(mv[selected]) &&
              fabs(expected - actual) < 1e-12,
              "random reachable exact terminal choice diverged at trial %d",
              trial);

        /* Brute-force the opponent's guaranteed response after every legal
         * pile stall.  For the same root card/action, ending now must be at
         * least as good as letting the opponent take one optional action and
         * end through the deck.  This checks the weak-dominance argument with
         * full engine transitions rather than duplicating solver mutation. */
        for (int i = 0; i < n; i++) {
            if (mv[i].draw == 0) continue;
            int deck = rollout_policy_terminal_choice(mv, n, i);
            CHECK(deck >= 0,
                  "pile move has no semantic deck variant at trial %d", trial);
            if (deck < 0) continue;
            State direct = st;
            lc_apply(&direct, mv[deck]);
            double direct_value = rollout_terminal_objective(
                &direct, st.turn, objective);

            State stalled = st;
            lc_apply(&stalled, mv[i]);
            CHECK(!stalled.over && stalled.deck_left == 1,
                  "pile stall unexpectedly ended the random fixture");
            Move response[MAX_MOVES];
            int nr = lc_moves(&stalled, response);
            double opponent_best = INFINITY;
            for (int j = 0; j < nr; j++) {
                if (response[j].draw != 0) continue;
                State terminal = stalled;
                lc_apply(&terminal, response[j]);
                double value = rollout_terminal_objective(
                    &terminal, st.turn, objective);
                if (value < opponent_best) opponent_best = value;
            }
            CHECK(isfinite(opponent_best) &&
                  direct_value + 1e-12 >= opponent_best,
                  "last-deck weak dominance failed at trial %d move %d: "
                  "direct %.9f, opponent response %.9f",
                  trial, i, direct_value, opponent_best);
        }
    }
}

static State deck3_replan_test_state(void)
{
    Rng rng;
    rng_seed(&rng, UINT64_C(0xDEC2A11));
    State st;
    lc_deal(&st, &rng);
    while (!st.over && st.deck_left > 3) {
        Move mv[MAX_MOVES];
        int n = lc_moves(&st, mv), chosen = -1;
        for (int i = 0; i < n; i++)
            if (mv[i].draw == 0) {
                chosen = i;
                break;
            }
        CHECK(chosen >= 0, "deck-three fixture has no deck-draw move");
        if (chosen < 0) break;
        lc_apply(&st, mv[chosen]);
    }
    CHECK(!st.over && st.deck_left == 3,
          "failed to construct a live deck-three replan fixture");
    return st;
}

/* Two known number cards in the hands and two on the piles form an exact
 * eight-ply shuttle under the policy biases installed by the test below:
 * suit-0 cards discard then take suit 1; suit-1 cards discard then take suit
 * 0.  All strategic and hidden state returns byte-for-byte except nply. */
static State late_cycle_test_state(void)
{
    State st;
    memset(&st, 0, sizeof st);
    const int a0 = CARD_MAKE(0, 3); /* Y2, player 0 */
    const int a1 = CARD_MAKE(0, 4); /* Y3, pile 0   */
    const int b0 = CARD_MAKE(1, 3); /* B2, player 1 */
    const int b1 = CARD_MAKE(1, 4); /* B3, pile 1   */
    uint64_t reserved = (UINT64_C(1) << a0) | (UINT64_C(1) << a1) |
                        (UINT64_C(1) << b0) | (UINT64_C(1) << b1);
    st.hand[0] = UINT64_C(1) << a0;
    st.hand[1] = UINT64_C(1) << b0;
    st.hand_n[0] = st.hand_n[1] = HAND_SIZE;
    st.known[0] = UINT64_C(1) << a0;
    st.known[1] = UINT64_C(1) << b0;

    int need0 = HAND_SIZE - 1, need1 = HAND_SIZE - 1, nd = 0;
    for (int c = 0; c < NCARD; c++) {
        if ((reserved >> c) & UINT64_C(1)) continue;
        if (need0 > 0) {
            st.hand[0] |= UINT64_C(1) << c;
            need0--;
        } else if (need1 > 0) {
            st.hand[1] |= UINT64_C(1) << c;
            need1--;
        } else if (nd < 3) {
            st.deck[nd++] = (uint8_t)c;
        } else {
            st.played[0] |= UINT64_C(1) << c;
        }
    }
    st.pile[0][0] = (uint8_t)a1;
    st.pile[1][0] = (uint8_t)b1;
    st.pile_n[0] = st.pile_n[1] = 1;
    st.discarded = (UINT64_C(1) << a1) | (UINT64_C(1) << b1);
    st.deck_pos = 0;
    st.deck_left = 3;
    st.turn = 0;
    return st;
}

static void test_late_rollout_cycle_break(void)
{
    Net *net = calloc(1, sizeof *net);
    CHECK(net != NULL, "network allocation for late-cycle fixture");
    if (!net) return;
    net_zero(net);
    const int shuttle[4] = {
        CARD_MAKE(0, 3), CARD_MAKE(0, 4),
        CARD_MAKE(1, 3), CARD_MAKE(1, 4)
    };
    for (int i = 0; i < 4; i++)
        net->bplay[shuttle[i] * 2 + 1] = 30.0f;
    net->bdraw[1] = 10.0f;
    net->bdraw[2] = 5.0f;

    State st = late_cycle_test_state();

    /* Popped pile slots retain stale bytes, and the three wager IDs are
     * physically distinct only inside the engine.  Neither may keep an
     * otherwise repeated strategic state out of the cycle detector. */
    State stale = st;
    stale.pile[0][stale.pile_n[0] + 5] =
        (uint8_t)CARD_MAKE(4, 11);
    CHECK(rollout_same_late_state(&st, &stale),
          "inactive pile storage changed late-state equality");
    stale.pile[0][0] = (uint8_t)CARD_MAKE(0, 5);
    CHECK(!rollout_same_late_state(&st, &stale),
          "active pile contents were ignored by late-state equality");

    State wagers_a;
    memset(&wagers_a, 0, sizeof wagers_a);
    wagers_a.deck_left = 1;
    wagers_a.deck[0] = (uint8_t)CARD_MAKE(4, 11);
    wagers_a.hand[0] = UINT64_C(1) << CARD_MAKE(0, 0);
    wagers_a.hand[1] = UINT64_C(1) << CARD_MAKE(0, 1);
    wagers_a.hand_n[0] = wagers_a.hand_n[1] = 1;
    State wagers_b = wagers_a;
    wagers_b.hand[0] = UINT64_C(1) << CARD_MAKE(0, 1);
    wagers_b.hand[1] = UINT64_C(1) << CARD_MAKE(0, 0);
    CHECK(rollout_same_late_state(&wagers_a, &wagers_b),
          "indistinguishable wager IDs changed late-state equality");
    wagers_b.hand[1] = UINT64_C(1) << CARD_MAKE(0, 4);
    CHECK(!rollout_same_late_state(&wagers_a, &wagers_b),
          "a different numbered hand card was ignored by late-state equality");

    Agent a;
    agent_default(&a, AG_ROLLOUT, net);
    a.no_belief = 1;
    a.symmetries = 1;
    a.playout_symmetries = 1;
    a.dets = 2;
    a.batch_dets = 2;
    a.root_width = 2;
    a.min_cand = 2;
    a.cand_floor = 0.0f;
    a.override_k = 0.0f;
    a.deck2_replan_worlds = 0;
    a.deck2_replan_cores = 0;

    Rng rng;
    rng_seed(&rng, UINT64_C(0xC1C1E));
    SearchStats stats;
    Move selected = rollout_move(&a, &st, &rng, NULL, &stats);
    CHECK(stats.worlds == a.dets && stats.cycle_breaks > 0,
          "forced pile shuttle was not recognized as a late cycle");
    CHECK(stats.unfinished_cap_leaves == 0,
          "late-cycle fixture still reached the unfinished ply cap");
    CHECK(stats.exact_terminal_leaves > 0,
          "cycle breaking did not propagate into exact one-card leaves");

    /* Recursive late improvement must be able to follow more than the old
     * single-stall horizon.  This pile-biased deck-three fixture repeatedly
     * revisits late information states; recursive child panels must therefore
     * reach at least a second decision level, then close the path or exhaust
     * its explicit bound and still finish through a real exact deck-one leaf. */
    Agent recursive = a;
    recursive.dets = 1;
    recursive.batch_dets = 1;
    recursive.deck2_replan_worlds = 8;
    recursive.deck2_replan_cores = 1;
    Rng recursive_rng;
    rng_seed(&recursive_rng, UINT64_C(0xA11CE57A11));
    SearchStats recursive_stats;
    (void)rollout_move(
        &recursive, &st, &recursive_rng, NULL, &recursive_stats);
    CHECK(recursive_stats.deck2_replan_root_calls > 0 &&
          recursive_stats.deck2_replans >
              recursive_stats.deck2_replan_root_calls &&
          recursive_stats.deck2_replan_max_depth >= 2 &&
          recursive_stats.deck2_replan_max_stall_chain >= 2 &&
          recursive_stats.exact_terminal_leaves > 0 &&
          recursive_stats.unfinished_cap_leaves == 0,
          "deck-three recursive stalls failed: roots=%llu replans=%llu "
          "depth=%llu stalls=%llu exact=%llu unfinished=%llu",
          (unsigned long long)recursive_stats.deck2_replan_root_calls,
          (unsigned long long)recursive_stats.deck2_replans,
          (unsigned long long)recursive_stats.deck2_replan_max_depth,
          (unsigned long long)recursive_stats.deck2_replan_max_stall_chain,
          (unsigned long long)recursive_stats.exact_terminal_leaves,
          (unsigned long long)recursive_stats.unfinished_cap_leaves);

    State reserve = st;
    Move reserve_legal[MAX_MOVES];
    int reserve_n = lc_moves(&reserve, reserve_legal);
    int reserve_deck = -1;
    for (int i = 0; i < reserve_n; i++)
        if (reserve_legal[i].draw == 0) {
            reserve_deck = i;
            break;
        }
    CHECK(reserve_deck >= 0,
          "engine-fuse reserve fixture has no deck-draw move");
    if (reserve_deck >= 0) lc_apply(&reserve, reserve_legal[reserve_deck]);
    CHECK(reserve.deck_left == 2,
          "engine-fuse reserve fixture did not reach deck two");
    reserve.nply = LC_MAX_PLIES - reserve.deck_left - 1;
    Agent reserve_agent = a;
    reserve_agent.deck2_replan_worlds = 2;
    reserve_agent.deck2_replan_cores = 2;
    Rng reserve_rng;
    rng_seed(&reserve_rng, UINT64_C(0xCA9E5E));
    SearchStats reserve_stats;
    (void)rollout_move(
        &reserve_agent, &reserve, &reserve_rng, NULL, &reserve_stats);
    CHECK(reserve_stats.cap_reserve_forces > 0 &&
          reserve_stats.deck2_replans == 0 &&
          reserve_stats.unfinished_cap_leaves == 0 &&
          reserve_stats.exact_terminal_leaves > 0,
          "engine-fuse reserve failed: forces=%llu replans=%llu "
          "unfinished=%llu exact=%llu",
          (unsigned long long)reserve_stats.cap_reserve_forces,
          (unsigned long long)reserve_stats.deck2_replans,
          (unsigned long long)reserve_stats.unfinished_cap_leaves,
          (unsigned long long)reserve_stats.exact_terminal_leaves);

    /* The detector compares complete determinizations, but that must not let
     * the root actor peek at which unseen card happens to occupy which hidden
     * location.  Swap a private opponent card with a deck card: the mover's
     * information set, seeded worlds, selected move and measured work must be
     * identical. */
    State hidden = st;
    uint64_t private_hand = hidden.hand[1] & ~hidden.known[1];
    int hc = private_hand ? __builtin_ctzll(private_hand) : -1;
    int dc = hidden.deck[hidden.deck_pos];
    CHECK(hc >= 0 && hc != dc,
          "late-cycle fixture has no hidden identities to swap");
    if (hc >= 0 && hc != dc) {
        hidden.hand[1] &= ~(UINT64_C(1) << hc);
        hidden.hand[1] |= UINT64_C(1) << dc;
        hidden.deck[hidden.deck_pos] = (uint8_t)hc;
        Rng hidden_rng;
        rng_seed(&hidden_rng, UINT64_C(0xC1C1E));
        SearchStats hidden_stats;
        Move hidden_selected = rollout_move(
            &a, &hidden, &hidden_rng, NULL, &hidden_stats);
        CHECK(MOVE_PACK(selected) == MOVE_PACK(hidden_selected) &&
              stats.n == hidden_stats.n &&
              stats.worlds == hidden_stats.worlds &&
              stats.cycle_breaks == hidden_stats.cycle_breaks &&
              stats.exact_terminal_leaves ==
                  hidden_stats.exact_terminal_leaves &&
              hidden_stats.unfinished_cap_leaves == 0,
              "hidden identity changed late-cycle rollout behavior");
        for (int i = 0; i < stats.n && i < hidden_stats.n; i++)
            CHECK(MOVE_PACK(stats.mv[i]) ==
                      MOVE_PACK(hidden_stats.mv[i]) &&
                  fabs(stats.q[i] - hidden_stats.q[i]) < 1e-12,
                  "hidden identity changed late-cycle row %d", i);
    }
    free(net);
}

static void test_deck2_continuation_replan(void)
{
    Net *net = calloc(1, sizeof *net);
    CHECK(net != NULL, "network allocation for deck-two replan");
    if (!net) return;

    State st = deck3_replan_test_state();
    CHECK(rollout_test_late_cache_order(net, &st),
          "recursive late cache depended on candidate order or ancestor path");
    int assignment_support = 0;
    CHECK(rollout_test_unique_late_assignments(
              &st, 2048, &assignment_support) == 990 &&
          assignment_support == 990,
          "deck-three 2048-world panel did not exhaust 990 unique assignments");
    CHECK(rollout_test_unique_late_assignments(
              &st, 128, &assignment_support) == 128 &&
          assignment_support == 990,
          "deck-three 128-world panel was not a unique support prefix");
    State assignment_d2 = st;
    Move assignment_legal[MAX_MOVES];
    int assignment_n = lc_moves(&assignment_d2, assignment_legal);
    int assignment_deck = -1;
    for (int i = 0; i < assignment_n; i++)
        if (assignment_legal[i].draw == 0) {
            assignment_deck = i;
            break;
        }
    CHECK(assignment_deck >= 0,
          "unique-assignment fixture has no deck draw");
    if (assignment_deck >= 0)
        lc_apply(&assignment_d2, assignment_legal[assignment_deck]);
    CHECK(assignment_d2.deck_left == 2 &&
          rollout_test_unique_late_assignments(
              &assignment_d2, 2048, &assignment_support) == 90 &&
          assignment_support == 90,
          "deck-two 2048-world panel did not exhaust 90 unique assignments");
    Agent a;
    agent_default(&a, AG_ROLLOUT, net);
    a.no_belief = 1;
    a.symmetries = 1;
    a.playout_symmetries = 1;
    a.dets = 2;
    a.root_width = 2;
    a.min_cand = 2;
    a.cand_floor = 0.0f;
    a.override_k = 0.0f;
    a.deck2_replan_worlds = 2;
    a.deck2_replan_cores = 2;

    /* Unique finite panels improve the deployed replan-off actor too.  Once
     * the mover's deck-two support is exhausted, extra requested worlds must
     * not duplicate assignments or pretend to add evidence. */
    Agent outer_census = a;
    outer_census.dets = 2048;
    outer_census.batch_dets = 2048;
    outer_census.deck2_replan_worlds = 0;
    outer_census.deck2_replan_cores = 0;
    Rng outer_census_rng;
    rng_seed(&outer_census_rng, UINT64_C(0x90C3E5));
    SearchStats outer_census_stats;
    (void)rollout_move(
        &outer_census, &assignment_d2, &outer_census_rng, NULL,
        &outer_census_stats);
    CHECK(outer_census_stats.worlds == 90 &&
          outer_census_stats.max_worlds == 90,
          "replan-off deck-two rollout did not exhaust one 90-world census");

    Agent root_only = a;
    root_only.exact_terminal = 2;
    root_only.deck2_replan_worlds = 0;
    root_only.deck2_replan_cores = 0;
    Rng root_only_rng;
    rng_seed(&root_only_rng, UINT64_C(0x1F05AFE));
    SearchStats root_only_stats;
    (void)rollout_move(
        &root_only, &st, &root_only_rng, NULL, &root_only_stats);
    CHECK(root_only_stats.worlds == 2 &&
          root_only_stats.exact_terminal_leaves == 0,
          "root-only ablation unexpectedly solved continuation leaves");

    Rng arng;
    rng_seed(&arng, UINT64_C(0x1F05AFE));
    SearchStats as;
    Move am = rollout_move(&a, &st, &arng, NULL, &as);
    CHECK(as.worlds == 2 && as.deck2_replans > 0 &&
          as.deck2_replan_root_calls > 0 &&
          as.deck2_replan_root_worlds ==
              as.deck2_replan_root_calls * a.deck2_replan_worlds &&
          as.deck2_replan_evals >= as.deck2_replan_worlds &&
          as.deck2_replan_evals <= as.deck2_replan_worlds * 2 *
                                      a.deck2_replan_cores &&
          as.exact_terminal_leaves > 0,
          "deck-two continuation replan was absent, recursive, or miscounted");
    CHECK(as.unfinished_cap_leaves == 0,
          "deck-two replan scored an unfinished ply-cap state as a leaf");

    Agent exhaustive = a;
    exhaustive.dets = 1;
    exhaustive.batch_dets = 1;
    exhaustive.deck2_replan_worlds = 128;
    /* Keep both root candidates on the deck.  After either hidden draw the
     * opponent's deck-two information set has eight unknown opposing cards
     * plus two ordered deck cards: exactly 10*9 = 90 assignments. */
    float saved_draw[NET_NDRAW];
    memcpy(saved_draw, net->bdraw, sizeof saved_draw);
    net->bdraw[0] = 30.0f;
    for (int d = 1; d < NET_NDRAW; d++) net->bdraw[d] = -30.0f;
    Rng exhaustive_rng;
    rng_seed(&exhaustive_rng, UINT64_C(0xE7A057));
    SearchStats exhaustive_stats;
    (void)rollout_move(
        &exhaustive, &st, &exhaustive_rng, NULL, &exhaustive_stats);
    memcpy(net->bdraw, saved_draw, sizeof saved_draw);
    CHECK(exhaustive_stats.deck2_replan_root_calls > 0 &&
          exhaustive_stats.deck2_replan_root_worlds ==
              exhaustive_stats.deck2_replan_root_calls * 90,
          "128-cap deck-two root panels were silently truncated: "
          "roots=%llu worlds=%llu",
          (unsigned long long)exhaustive_stats.deck2_replan_root_calls,
          (unsigned long long)exhaustive_stats.deck2_replan_root_worlds);
    CHECK(exhaustive_stats.unfinished_cap_leaves == 0,
          "exhaustive deck-two support reached an unfinished cap leaf");

    /* Swap two facts hidden from the mover: one private opponent card and one
     * deck card.  The replan must reconstruct fresh worlds from the mover's
     * information view, so its action and all measured work stay identical. */
    State hidden = st;
    int p = st.turn, o = p ^ 1;
    uint64_t private_hand = hidden.hand[o] & ~hidden.known[o];
    int hc = private_hand ? __builtin_ctzll(private_hand) : -1;
    int dc = hidden.deck_left > 0 ? hidden.deck[hidden.deck_pos] : -1;
    CHECK(hc >= 0 && dc >= 0 && hc != dc,
          "deck-two hidden-information fixture has no swappable cards");
    if (hc >= 0 && dc >= 0 && hc != dc) {
        hidden.hand[o] &= ~(UINT64_C(1) << hc);
        hidden.hand[o] |= UINT64_C(1) << dc;
        hidden.deck[hidden.deck_pos] = (uint8_t)hc;
        Rng brng;
        rng_seed(&brng, UINT64_C(0x1F05AFE));
        SearchStats bs;
        Move bm = rollout_move(&a, &hidden, &brng, NULL, &bs);
        CHECK(MOVE_PACK(am) == MOVE_PACK(bm) &&
              as.n == bs.n && as.worlds == bs.worlds &&
              as.deck2_replans == bs.deck2_replans &&
              as.deck2_replan_worlds == bs.deck2_replan_worlds &&
              as.deck2_replan_evals == bs.deck2_replan_evals &&
              as.deck2_replan_cap_hits == bs.deck2_replan_cap_hits &&
              as.deck2_replan_cache_hits == bs.deck2_replan_cache_hits &&
              as.deck2_replan_cycle_closures ==
                  bs.deck2_replan_cycle_closures &&
              as.deck2_replan_max_depth == bs.deck2_replan_max_depth &&
              as.deck2_replan_root_calls == bs.deck2_replan_root_calls &&
              as.deck2_replan_root_worlds == bs.deck2_replan_root_worlds &&
              as.deck2_replan_max_stall_chain ==
                  bs.deck2_replan_max_stall_chain &&
              as.deck2_replan_low_world_fallbacks ==
                  bs.deck2_replan_low_world_fallbacks &&
              as.exact_terminal_leaves == bs.exact_terminal_leaves &&
              as.unfinished_cap_leaves == bs.unfinished_cap_leaves,
              "hidden opponent/deck identity changed deck-two replan");
        for (int i = 0; i < as.n && i < bs.n; i++)
            CHECK(MOVE_PACK(as.mv[i]) == MOVE_PACK(bs.mv[i]) &&
                  fabs(as.q[i] - bs.q[i]) < 1e-12,
                  "hidden identity changed deck-two replan row %d", i);
    }

    /* A sharp policy used to collapse deck=2 to one root candidate, skipping
     * every exact-leaf-aware playout.  Enabling the ablation must force one
     * additional top semantic core while leaving the default-off path alone. */
    State d2 = st;
    Move legal[MAX_MOVES];
    int nlegal = lc_moves(&d2, legal), deck_move = -1;
    for (int i = 0; i < nlegal; i++)
        if (legal[i].draw == 0) {
            deck_move = i;
            break;
        }
    CHECK(deck_move >= 0, "deck-two bypass fixture has no deck move");
    if (deck_move >= 0) lc_apply(&d2, legal[deck_move]);
    CHECK(d2.deck_left == 2, "deck-two bypass fixture did not reach deck two");
    Move d2legal[MAX_MOVES];
    int nd2 = lc_moves(&d2, d2legal);
    if (nd2 > 0) {
        net_zero(net);
        int ip = d2legal[0].card * 2 + d2legal[0].discard;
        net->bplay[ip] = 20.0f;
        net->bdraw[d2legal[0].draw] = 20.0f;
        Agent off = a;
        off.dets = 2;
        off.root_width = 5;
        off.min_cand = 1;
        off.cand_floor = 0.02f;
        off.deck2_replan_worlds = 0;
        off.deck2_replan_cores = 0;
        Rng off_rng;
        rng_seed(&off_rng, UINT64_C(0x51A61E));
        SearchStats off_stats;
        (void)rollout_move(&off, &d2, &off_rng, NULL, &off_stats);
        CHECK(off_stats.worlds == 0 && off_stats.n == 1,
              "default-off sharp policy no longer takes singleton fast path");

        Agent on = off;
        on.deck2_replan_worlds = 2;
        on.deck2_replan_cores = 2;
        Rng on_rng;
        rng_seed(&on_rng, UINT64_C(0x51A61E));
        SearchStats on_stats;
        (void)rollout_move(&on, &d2, &on_rng, NULL, &on_stats);
        CHECK(on_stats.worlds == 2 && on_stats.n >= 2 &&
              on_stats.exact_terminal_leaves > 0,
              "deck-two replan did not break the one-candidate bypass");
    }
    free(net);
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

static void test_exact_k_belief_objective(void)
{
    enum { N = 5 };
    const float logits[N] = { -0.7f, 0.2f, 1.1f, -0.1f, 0.6f };
    const uint8_t held[N] = { 1, 0, 0, 1, 0 };
    float marginal[N];
    double nll = 0.0;
    CHECK(belief_exact_k_eval(logits, held, N, 2, 1.0f,
                              marginal, &nll),
          "evaluate exact-K belief likelihood");
    double msum = 0.0, gsum = 0.0;
    for (int i = 0; i < N; i++) {
        CHECK(marginal[i] >= 0.0f && marginal[i] <= 1.0f,
              "exact-K marginal outside [0,1]");
        msum += marginal[i];
        gsum += marginal[i] - held[i];
    }
    CHECK(fabs(msum - 2.0) < 1e-6,
          "exact-K marginals sum %.9f instead of two", msum);
    CHECK(fabs(gsum) < 1e-6,
          "exact-K logit gradient has nonzero common mode %.9g", gsum);

    float shifted[N], shifted_marginal[N];
    for (int i = 0; i < N; i++) shifted[i] = logits[i] + 7.25f;
    double shifted_nll = 0.0;
    CHECK(belief_exact_k_eval(shifted, held, N, 2, 1.0f,
                              shifted_marginal, &shifted_nll),
          "evaluate shifted exact-K likelihood");
    CHECK(fabs(shifted_nll - nll) < 2e-7,
          "exact-K likelihood changed under common logit shift");
    for (int i = 0; i < N; i++)
        CHECK(fabsf(shifted_marginal[i] - marginal[i]) < 2e-7f,
              "exact-K marginal changed under common logit shift");

    const float eps = 1e-3f;
    for (int i = 0; i < N; i++) {
        float plus[N], minus[N], scratch[N];
        memcpy(plus, logits, sizeof plus);
        memcpy(minus, logits, sizeof minus);
        plus[i] += eps;
        minus[i] -= eps;
        double lp = 0.0, lm = 0.0;
        CHECK(belief_exact_k_eval(plus, held, N, 2, 1.0f,
                                  scratch, &lp) &&
              belief_exact_k_eval(minus, held, N, 2, 1.0f,
                                  scratch, &lm),
              "finite-difference exact-K likelihood");
        double numerical = (lp - lm) / (2.0 * eps);
        double analytic = marginal[i] - held[i];
        CHECK(fabs(numerical - analytic) < 2e-4,
              "exact-K gradient %.7f != finite difference %.7f at %d",
              analytic, numerical, i);
    }

    const float uniform_logits[N] = { 0, 0, 0, 0, 0 };
    CHECK(belief_exact_k_eval(uniform_logits, held, N, 2, 1.0f,
                              marginal, NULL),
          "evaluate uniform exact-K posterior");
    for (int i = 0; i < N; i++)
        CHECK(fabsf(marginal[i] - 0.4f) < 1e-7f,
              "uniform exact-K marginal %.8f is not 2/5", marginal[i]);

    const uint8_t wrong_count[N] = { 1, 0, 0, 0, 0 };
    CHECK(!belief_exact_k_eval(logits, wrong_count, N, 2, 1.0f,
                               marginal, &nll),
          "exact-K likelihood accepted a wrong-cardinality label");
}

static void test_centered_mcts_value(void)
{
    Net *base = calloc(1, sizeof *base);
    Net *shifted = calloc(1, sizeof *shifted);
    CHECK(base != NULL && shifted != NULL,
          "network allocation for centered MCTS value");
    if (!base || !shifted) { free(base); free(shifted); return; }
    shifted->b3 = 7.0f;

    uint8_t deck[NCARD];
    for (int i = 0; i < NCARD; i++) deck[i] = (uint8_t)i;
    State st;
    lc_deal_from_deck(&st, deck);
    st.round = MATCH_ROUNDS - 1;
    st.cum[0] = 23;
    st.cum[1] = -11;

    float b0 = net_value_state_sym(base, &st, 0, 5);
    float b1 = net_value_state_sym(base, &st, 1, 5);
    float s0 = net_value_state_sym(shifted, &st, 0, 5);
    float s1 = net_value_state_sym(shifted, &st, 1, 5);
    CHECK(fabsf(0.5f * (b0 - b1) - 0.5f * (s0 - s1)) < 1e-6f,
          "centralized critic changed under a common b3 shift");

    Agent a, b;
    agent_default(&a, AG_MCTS, base);
    a.dets = 3;
    a.sims = 32;
    a.root_width = 4;
    a.node_width = 4;
    a.symmetries = 5;
    b = a;
    b.net = shifted;

    Rng arng, brng;
    rng_seed(&arng, 0xC3117EULL);
    rng_seed(&brng, 0xC3117EULL);
    SearchStats as, bs;
    float av = 0.0f, bv = 0.0f;
    Move am = search_move(&a, &st, &arng, &av, &as);
    Move bm = search_move(&b, &st, &brng, &bv, &bs);
    CHECK(MOVE_PACK(am) == MOVE_PACK(bm),
          "MCTS move changed under a common value-head bias");
    CHECK(as.n == bs.n && as.nlegal == bs.nlegal,
          "MCTS shortlist changed under a common value-head bias");
    CHECK(fabsf(av - bv) < 1e-5f,
          "MCTS root value changed under a common value-head bias");
    for (int i = 0; i < as.n && i < bs.n; i++) {
        CHECK(MOVE_PACK(as.mv[i]) == MOVE_PACK(bs.mv[i]) &&
              as.visits[i] == bs.visits[i] &&
              fabs(as.q[i] - bs.q[i]) < 1e-5,
              "MCTS row %d changed under a common value-head bias", i);
    }
    free(base);
    free(shifted);
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

    /* Optional hierarchical selection spends its fixed budget on distinct
     * card/action cores before considering one public-information draw repair
     * per core.  It must never displace candidate zero or exceed five. */
    Agent cores = audit;
    cores.root_width = 5;
    cores.min_cand = 3;
    cores.cand_mass = 0.0f;
    cores.cand_floor = 0.02f;
    cores.action_core_count = 3;
    SearchStats core_stats;
    rng_seed(&rng, 732020);
    (void)rollout_move(&cores, &p20, &rng, NULL, &core_stats);
    CHECK(core_stats.action_core_candidates == 3,
          "hierarchical shortlist retained %d cores, expected 3",
          core_stats.action_core_candidates);
    CHECK(core_stats.action_draw_candidates >= 0 &&
          core_stats.action_draw_candidates <= 2 &&
          core_stats.trusted_candidates <= 5 && core_stats.n <= 5,
          "hierarchical shortlist exceeded its budget (cores=%d draws=%d trusted=%d n=%d)",
          core_stats.action_core_candidates,
          core_stats.action_draw_candidates,
          core_stats.trusted_candidates, core_stats.n);
    CHECK(MOVE_PACK(core_stats.mv[0]) == MOVE_PACK(pmv[order[0]]),
          "hierarchical shortlist displaced the complete policy baseline");
    for (int i = 0; i < core_stats.action_core_candidates; i++)
        for (int j = 0; j < i; j++)
            CHECK(!same_test_action(core_stats.mv[i], core_stats.mv[j]),
                  "hierarchical core prefix contains duplicate semantic actions");
    for (int i = core_stats.action_core_candidates;
         i < core_stats.trusted_candidates; i++) {
        CHECK(core_stats.prior[i] + 1e-7 >= cores.cand_floor,
              "hierarchical draw alternative bypassed the policy floor");
        int attached = 0;
        for (int j = 0; j < core_stats.action_core_candidates; j++)
            if (same_test_action(core_stats.mv[i], core_stats.mv[j])) {
                attached = 1;
                break;
            }
        CHECK(attached,
              "hierarchical draw alternative is not attached to a selected core");
    }
    double expected_core_mass = 0.0;
    for (int i = 0; i < pn; i++) {
        int represented = 0;
        for (int j = 0; j < core_stats.action_core_candidates; j++)
            if (same_test_action(pmv[i], core_stats.mv[j])) {
                represented = 1;
                break;
            }
        if (represented) expected_core_mass += prior[i];
    }
    CHECK(fabs(core_stats.policy_mass - expected_core_mass) < 1e-6,
          "hierarchical shortlist reports %.6f representative mass, expected %.6f aggregate core mass",
          core_stats.policy_mass, expected_core_mass);

    /* Exercise production-style min_cand=1 eligibility instead of forcing
     * all requested cores.  Every nonbaseline core must clear the aggregate
     * floor, candidate zero remains the exact complete-move policy leader,
     * and the trusted budget remains bounded. */
    Agent floor_cores = cores;
    floor_cores.root_width = 3;
    floor_cores.min_cand = 1;
    floor_cores.action_core_count = 3;
    SearchStats floor_stats;
    rng_seed(&rng, 732021);
    Move floor_move = rollout_move(
        &floor_cores, &p20, &rng, NULL, &floor_stats);
    CHECK(MOVE_PACK(floor_stats.mv[0]) == MOVE_PACK(pmv[order[0]]),
          "aggregate-floor shortlist displaced the complete policy leader");
    CHECK(floor_stats.action_core_candidates >= 1 &&
          floor_stats.action_core_candidates <= 3 &&
          floor_stats.trusted_candidates <= 3,
          "aggregate-floor shortlist exceeded core budget (cores=%d trusted=%d)",
          floor_stats.action_core_candidates, floor_stats.trusted_candidates);
    double floor_mass = 0.0;
    for (int j = 0; j < floor_stats.action_core_candidates; j++) {
        double action_mass = 0.0;
        for (int i = 0; i < pn; i++)
            if (same_test_action(pmv[i], floor_stats.mv[j]))
                action_mass += prior[i];
        if (j > 0)
            CHECK(action_mass + 1e-7 >= floor_cores.cand_floor,
                  "hierarchical core %d bypassed aggregate floor (%.6f < %.6f)",
                  j, action_mass, floor_cores.cand_floor);
        floor_mass += action_mass;
    }
    CHECK(fabs(floor_stats.policy_mass - floor_mass) < 1e-6,
          "aggregate-floor shortlist reports %.6f mass, expected %.6f",
          floor_stats.policy_mass, floor_mass);

    /* Candidate construction is an information-set operation.  Exchange one
     * unknown opponent card with one hidden deck card while preserving every
     * observation; the shortlist, aggregate coverage and selected move must
     * remain byte-for-byte equivalent under the same RNG stream. */
    State hidden_variant = p20;
    int mover = p20.turn, opponent = mover ^ 1;
    uint64_t hidden = p20.hand[opponent] & ~p20.known[opponent];
    CHECK(hidden != 0 && p20.deck_left > 0,
          "locked shortlist state lacks a hidden assignment to exchange");
    if (hidden && p20.deck_left > 0) {
        int held_card = __builtin_ctzll(hidden);
        int deck_card = p20.deck[p20.deck_pos];
        hidden_variant.hand[opponent] &= ~(1ULL << held_card);
        hidden_variant.hand[opponent] |= 1ULL << deck_card;
        hidden_variant.deck[hidden_variant.deck_pos] = (uint8_t)held_card;
        SearchStats hidden_stats;
        rng_seed(&rng, 732021);
        Move hidden_move = rollout_move(
            &floor_cores, &hidden_variant, &rng, NULL, &hidden_stats);
        CHECK(MOVE_PACK(hidden_move) == MOVE_PACK(floor_move) &&
              hidden_stats.n == floor_stats.n &&
              hidden_stats.trusted_candidates ==
                  floor_stats.trusted_candidates &&
              hidden_stats.action_core_candidates ==
                  floor_stats.action_core_candidates &&
              hidden_stats.action_draw_candidates ==
                  floor_stats.action_draw_candidates &&
              hidden_stats.policy_mass == floor_stats.policy_mass,
              "hierarchical shortlist leaked hidden hand/deck assignment");
        for (int i = 0; i < hidden_stats.n && i < floor_stats.n; i++)
            CHECK(MOVE_PACK(hidden_stats.mv[i]) ==
                      MOVE_PACK(floor_stats.mv[i]) &&
                  hidden_stats.prior[i] == floor_stats.prior[i] &&
                  hidden_stats.q[i] == floor_stats.q[i] &&
                  hidden_stats.se[i] == floor_stats.se[i] &&
                  hidden_stats.delta[i] == floor_stats.delta[i] &&
                  hidden_stats.dse[i] == floor_stats.dse[i],
                  "hidden assignment changed hierarchical candidate %d", i);
    }

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
    CHECK(checked.prefix_numerical_agreement == checked.prefix_confirmed &&
          checked.prefix_gate_passed == checked.prefix_confirmed,
          "disabled paired prefix gate changed numerical consensus");
    for (int c = 0; c < checked.trusted_candidates; c++) {
        CHECK(isfinite(checked.prefix_delta[c]) &&
              isfinite(checked.prefix_dse[c]) &&
              checked.prefix_dse[c] >= 0.0,
              "fresh paired prefix statistic %d is invalid", c);
        CHECK(fabs(checked.prefix_delta[c] -
                   (checked.prefix_q[c] - checked.prefix_q[0])) < 1e-9,
              "fresh paired delta %d disagrees with candidate means", c);
    }
    CHECK(checked.prefix_delta[0] == 0.0 &&
          checked.prefix_dse[0] == 0.0,
          "fresh paired baseline statistics are nonzero");

    if (checked.prefix_numerical_agreement &&
        checked.prefix_delta[proposed] > 0.0) {
        double pd = checked.prefix_delta[proposed];
        double pse = checked.prefix_dse[proposed];
        Agent permissive_prefix = consensus;
        permissive_prefix.prefix_confirm_k = pse > 0.0
            ? (float)(pd / (4.0 * pse)) : 1.0f;
        permissive_prefix.prefix_confirm_min = (float)(pd / 4.0);
        SearchStats permissive_stats;
        rng_seed(&rng, 9008);
        (void)rollout_move(&permissive_prefix, &p8, &rng, NULL,
                           &permissive_stats);
        CHECK(permissive_stats.prefix_gate_passed &&
              permissive_stats.prefix_confirmed &&
              permissive_stats.selection_reference == proposed,
              "trusted-prefix move clearing both paired gates was rejected");

        Agent effect_blocked = permissive_prefix;
        effect_blocked.prefix_confirm_min = (float)(pd * 2.0 + 1.0);
        SearchStats effect_stats;
        rng_seed(&rng, 9008);
        (void)rollout_move(&effect_blocked, &p8, &rng, NULL, &effect_stats);
        CHECK(effect_stats.prefix_numerical_agreement &&
              !effect_stats.prefix_gate_passed &&
              effect_stats.selection_reference == 0,
              "trusted-prefix practical-effect floor was not required");

        if (pse > 0.0) {
            Agent evidence_blocked = permissive_prefix;
            evidence_blocked.prefix_confirm_k =
                (float)(pd / pse * 2.0 + 1.0);
            SearchStats evidence_stats;
            rng_seed(&rng, 9008);
            (void)rollout_move(&evidence_blocked, &p8, &rng, NULL,
                               &evidence_stats);
            CHECK(evidence_stats.prefix_numerical_agreement &&
                  !evidence_stats.prefix_gate_passed &&
                  evidence_stats.selection_reference == 0,
                  "trusted-prefix paired-SE threshold was not required");
        }
    }

    /* The optional trusted-prefix gate uses that same fresh paired panel.  An
     * intentionally unreachable evidence/effect threshold must reject the
     * proposal without changing its worlds or diagnostics. */
    Agent gated_prefix = consensus;
    gated_prefix.prefix_confirm_k = 1000000.0f;
    gated_prefix.prefix_confirm_min = 1000000.0f;
    SearchStats gated_stats;
    rng_seed(&rng, 9008);
    Move prefix_gated_move =
        rollout_move(&gated_prefix, &p8, &rng, NULL, &gated_stats);
    CHECK(gated_stats.prefix_proposed == proposed &&
          gated_stats.prefix_confirm_worlds == checked.prefix_confirm_worlds &&
          gated_stats.selection_reference == 0 &&
          !gated_stats.prefix_gate_passed &&
          !gated_stats.prefix_confirmed &&
          MOVE_PACK(prefix_gated_move) == MOVE_PACK(gated_stats.mv[0]),
          "paired trusted-prefix gate did not fall back to baseline");
    CHECK(gated_stats.prefix_numerical_agreement ==
              checked.prefix_numerical_agreement,
          "enabling the prefix gate changed fresh-panel numerical agreement");
    for (int c = 0; c < checked.trusted_candidates; c++)
        CHECK(fabs(gated_stats.prefix_delta[c] -
                   checked.prefix_delta[c]) < 1e-12 &&
              fabs(gated_stats.prefix_dse[c] -
                   checked.prefix_dse[c]) < 1e-12,
              "enabling the prefix gate changed paired diagnostic %d", c);

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
          a.draw_playout_deck_max == 0 &&
          a.prefix_confirm_k == 0.0f &&
          a.prefix_confirm_min == 0.0f &&
          a.confirm_temp == 0.0f &&
          a.action_core_count == 0 && a.deck2_replan_worlds == 0 &&
          a.deck2_replan_cores == 0 && a.bounded_late_root == 0 &&
          fabsf(a.bounded_late_min - 1.0f) < 1e-6f,
          "rollout default no longer makes continuation pruning follow root");
    spec_parse("rolloutu:data/champion.bin:256:5:0.03:0.9:2:14:50:4:"
               "2:1:3.5:1.5:3:20:0.995:64:20:1:24:128:0:16:12:1:1:2:12:1:1.25:4:3:2.75:1.125:0.2:3:1:24:3:0:2.25",
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
          a.draw_playout_deck_max == 3 &&
          fabsf(a.prefix_confirm_k - 2.75f) < 1e-6f &&
          fabsf(a.prefix_confirm_min - 1.125f) < 1e-6f &&
          fabsf(a.confirm_temp - 0.2f) < 1e-6f &&
          a.action_core_count == 3 && a.exact_terminal == 1 &&
          a.deck2_replan_worlds == 24 && a.deck2_replan_cores == 3 &&
          a.bounded_late_root == 0 &&
          fabsf(a.bounded_late_min - 2.25f) < 1e-6f,
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
          champion.draw_playout_deck_max == 0 &&
          champion.prefix_confirm_k == 0.0f &&
          champion.prefix_confirm_min == 0.0f &&
          champion.confirm_temp == 0.0f &&
          champion.action_core_count == 0 && champion.exact_terminal == 1 &&
          champion.deck2_replan_worlds == 0 &&
          champion.deck2_replan_cores == 0 &&
          champion.bounded_late_root == 0 &&
          fabsf(champion.bounded_late_min - 1.0f) < 1e-6f,
          "maintained champion spec drifted from its locked configuration");
    free((void *)champion.net);

    Agent audit;
    spec_parse(LC_AUDIT_AGENT_SPEC, &audit);
    CHECK(audit.kind == AG_ROLLOUT && audit.no_belief &&
          audit.dets == 2048 && audit.root_width == 5 &&
          fabsf(audit.cand_floor - 0.01f) < 1e-6f &&
          audit.ply_lo == 14 && audit.confirm_dets == 2048 &&
          audit.policy_prefix_mode == 2 &&
          fabsf(audit.prefix_confirm_k - 2.0f) < 1e-6f &&
          fabsf(audit.prefix_confirm_min - 1.0f) < 1e-6f &&
          audit.action_core_count == 3 && audit.exact_terminal == 1 &&
          audit.deck2_replan_worlds == 0 &&
          audit.deck2_replan_cores == 0 && audit.bounded_late_root == 1 &&
          fabsf(audit.bounded_late_min - 1.0f) < 1e-6f,
          "post-game audit spec drifted from its focused configuration");
    free((void *)audit.net);

    /* Training's live-network form must not have a private, fixed-size copy of
     * the rollout parser.  This deliberately exceeds the old 128-byte buffer
     * and checks every field that used to be silently lost after
     * playout_prune. */
    Net *live = (Net *)malloc(sizeof *live);
    CHECK(live != NULL, "live-network parser fixture allocation failed");
    const char *self_spec =
        "selfrollout:256:5:0.0123456789:0.8765432109:7:14:299:8:2:1:"
        "3.5000000001:1.5000000001:4:20:0.9950000001:128:20:1:24:128:"
        "0:16:12:1:1:2:12:3:1.25:4:3:2.75:1.125:0.2:3:1:0:0:0:3.25";
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
          live_rollout.draw_playout_deck_max == 3 &&
          fabsf(live_rollout.prefix_confirm_k - 2.75f) < 1e-6f &&
          fabsf(live_rollout.prefix_confirm_min - 1.125f) < 1e-6f &&
          fabsf(live_rollout.confirm_temp - 0.2f) < 1e-6f &&
          live_rollout.action_core_count == 3 &&
          live_rollout.exact_terminal == 1 &&
          live_rollout.deck2_replan_worlds == 0 &&
          live_rollout.deck2_replan_cores == 0 &&
          live_rollout.bounded_late_root == 0 &&
          fabsf(live_rollout.bounded_late_min - 3.25f) < 1e-6f,
          "selfrollout planner/semantic/consensus tail was not parsed");
    free(live);

    Agent legacy_tail;
    spec_parse(
        "rolloutu:data/champion.bin:8:2:0.02:0:1:0:0:0:0:0:0:4:2:1:"
        "0:0:1:1:0:8:0:0:0:0:0:0:0:0:1:0:0:0:0:0:0:0",
        &legacy_tail);
    CHECK(legacy_tail.exact_terminal == 0,
          "controlled exact-terminal ablation field was not parsed");
    free((void *)legacy_tail.net);

    Agent root_only_tail;
    spec_parse(
        "rolloutu:data/champion.bin:8:2:0.02:0:1:0:0:0:0:0:0:4:2:1:"
        "0:0:1:1:0:8:0:0:0:0:0:0:0:0:1:0:0:0:0:0:0:2",
        &root_only_tail);
    CHECK(root_only_tail.exact_terminal == 2,
          "root-only exact-terminal propagation ablation was not parsed");
    free((void *)root_only_tail.net);

    Agent policy_action_tail;
    spec_parse(
        "rolloutu:data/champion.bin:8:2:0.02:0:1:0:0:0:0:0:0:4:2:1:"
        "0:0:1:1:0:8:0:0:0:0:0:0:0:0:1:0:0:0:0:0:0:3",
        &policy_action_tail);
    CHECK(policy_action_tail.exact_terminal == 3,
          "policy-action terminal propagation control was not parsed");
    free((void *)policy_action_tail.net);
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
        search.deck2_replan_worlds = mode == 2 ? 1 : 0;
        search.deck2_replan_cores = mode == 2 ? 1 : 0;
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

static void test_trajectory_suit_augmentation(void)
{
    enum { NTRAJ = 32 };
    uint8_t sequential[NTRAJ][NSUIT];
    uint8_t partitioned[NTRAJ][NSUIT];
    for (int g = 0; g < NTRAJ; g++)
        CHECK(trajectory_suit_permutation(20, 881726ULL, (uint64_t)g,
                                          sequential[g]),
              "choose sequential trajectory permutation");
    for (int worker = 0; worker < 4; worker++)
        for (int g = worker; g < NTRAJ; g += 4)
            CHECK(trajectory_suit_permutation(20, 881726ULL, (uint64_t)g,
                                              partitioned[g]),
                  "choose partitioned trajectory permutation");
    int saw_different = 0;
    for (int g = 0; g < NTRAJ; g++) {
        CHECK(memcmp(sequential[g], partitioned[g], NSUIT) == 0,
              "trajectory permutation depends on worker partition at %d", g);
        if (g > 0 && memcmp(sequential[0], sequential[g], NSUIT) != 0)
            saw_different = 1;
    }
    CHECK(saw_different,
          "trajectory permutation selector produced no augmentation variety");
    uint8_t identity[NSUIT];
    CHECK(trajectory_suit_permutation(0, 7, 9, identity),
          "explicitly disabled trajectory permutation rejected");
    for (int s = 0; s < NSUIT; s++)
        CHECK(identity[s] == s,
              "disabled trajectory augmentation is not identity");
    CHECK(!trajectory_suit_permutation(3, 7, 9, identity),
          "invalid trajectory suit group was accepted");

    Net *net = malloc(sizeof *net);
    CHECK(net != NULL, "network allocation for trajectory augmentation");
    if (!net) return;
    CHECK(net_load(net, "data/champion.bin") == 0,
          "load champion for trajectory augmentation");
    State st = reviewed_state(net, 20), view;
    uint8_t perm[NSUIT];
    CHECK(trajectory_suit_permutation(20, 20260803ULL, 17, perm),
          "choose common-state test permutation");
    Move vm[MAX_MOVES], em[MAX_MOVES], check_mv[MAX_MOVES];
    float raw[MAX_MOVES], behavior[MAX_MOVES], check_raw[MAX_MOVES];
    const float temperature = 0.7f;
    int n = trajectory_policy_probs(net, &st, perm, temperature, &view,
                                    vm, em, raw, behavior);
    int cn = policy_probs(net, &view, check_mv, check_raw, NULL);
    CHECK(n > 1 && n == cn,
          "trajectory policy changed oriented legal-move count");
    double bsum = 0.0, expected_sum = 0.0;
    for (int i = 0; i < n; i++)
        expected_sum += pow(check_raw[i], 1.0 / temperature);

    Move legal[MAX_MOVES];
    int nlegal = lc_moves(&st, legal);
    for (int i = 0; i < n; i++) {
        CHECK(MOVE_PACK(vm[i]) == MOVE_PACK(check_mv[i]) &&
              raw[i] == check_raw[i],
              "stored state/action probability changed orientation at %d", i);
        double expected = pow(check_raw[i], 1.0 / temperature)
                        / expected_sum;
        CHECK(fabs(behavior[i] - expected) < 2e-7,
              "stored old probability is inconsistent at %d", i);
        bsum += behavior[i];
        int found = 0;
        for (int k = 0; k < nlegal; k++)
            if (MOVE_PACK(em[i]) == MOVE_PACK(legal[k])) found = 1;
        CHECK(found, "mapped-back engine move %d is not legal", i);

        State engine_next = st, view_next = view, mapped_next;
        lc_apply(&engine_next, em[i]);
        lc_apply(&view_next, vm[i]);
        lc_permute_suits(&engine_next, &mapped_next, perm);
        CHECK(memcmp(&view_next, &mapped_next, sizeof(State)) == 0,
              "mapped engine transition diverged at action %d", i);
    }
    CHECK(fabs(bsum - 1.0) < 2e-6,
          "trajectory behavior probabilities sum to %.9f", bsum);
    free(net);
}

int main(void)
{
    test_sampler();
    test_near_greedy_confirmation_sampler();
    test_rollout_terminal_objective();
    test_rollout_exact_terminal_choice();
    test_rollout_exact_terminal_randomized();
    test_late_rollout_cycle_break();
    test_deck2_continuation_replan();
    test_rollout_value_scale();
    test_belief_distribution();
    test_exact_k_belief_objective();
    test_centered_mcts_value();
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
    test_trajectory_suit_augmentation();
    if (failures == 0) {
        printf("all runtime regression tests passed\n");
        return 0;
    }
    printf("%d failures\n", failures);
    return 1;
}
