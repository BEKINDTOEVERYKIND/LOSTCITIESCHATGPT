#include "../src/late_resolver.h"
#include "../src/agent.h"
#include "../src/search.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>

extern Move rollout_move_reference_for_test(
    const Agent *a, const State *st, Rng *rng,
    float *out_value, SearchStats *stats);

static int failures;

#define CHECK(cond, ...) do {                                             \
    if (!(cond)) {                                                        \
        fprintf(stderr, "FAIL: ");                                      \
        fprintf(stderr, __VA_ARGS__);                                    \
        fprintf(stderr, "\n");                                         \
        failures++;                                                       \
    }                                                                     \
} while (0)

static State late_state(int deck_left)
{
    Rng rng;
    rng_seed(&rng, UINT64_C(0x4C4154455245534F));
    State st;
    lc_deal(&st, &rng);
    while (!st.over && st.deck_left > deck_left) {
        Move mv[MAX_MOVES];
        int n = lc_moves(&st, mv);
        int selected = -1;
        for (int i = 0; i < n; i++)
            if (mv[i].draw == 0) {
                selected = i;
                break;
            }
        if (selected < 0) break;
        lc_apply(&st, mv[selected]);
    }
    return st;
}

static int semantic_move_equal(Move a, Move b)
{
    int ar = CARD_IS_WAGER(a.card) ? 0 : CARD_VALUE(a.card);
    int br = CARD_IS_WAGER(b.card) ? 0 : CARD_VALUE(b.card);
    return CARD_SUIT(a.card) == CARD_SUIT(b.card) && ar == br &&
           a.discard == b.discard && a.draw == b.draw;
}

static int move_legal(const State *st, Move wanted)
{
    Move mv[MAX_MOVES];
    int n = lc_moves(st, mv);
    for (int i = 0; i < n; i++)
        if (semantic_move_equal(mv[i], wanted)) return 1;
    return 0;
}

static int name_id_free(const char *name, uint64_t used)
{
    char card_name[8];
    for (int c = 0; c < NCARD; c++) {
        lc_card_name(c, card_name);
        if (!strcasecmp(card_name, name) &&
            !((used >> c) & UINT64_C(1)))
            return c;
    }
    return -1;
}

/* Minimal reader for the checked-in information-state fixtures.  Hidden deck
 * contents are intentionally absent: the resolver must rebuild them solely
 * from the mover's observation. */
static int load_state(const char *path, State *st)
{
    FILE *file = fopen(path, "r");
    if (!file) return 0;
    memset(st, 0, sizeof *st);
    uint64_t used = 0;
    char line[512];
    while (fgets(line, sizeof line, file)) {
        char *token = strtok(line, " \t\n");
        if (!token) continue;
        if (!strcmp(token, "turn"))
            st->turn = (uint8_t)atoi(strtok(NULL, " \n"));
        else if (!strcmp(token, "round"))
            st->round = (uint8_t)atoi(strtok(NULL, " \n"));
        else if (!strcmp(token, "nply"))
            st->nply = (uint16_t)atoi(strtok(NULL, " \n"));
        else if (!strcmp(token, "deck_left"))
            st->deck_left = (uint8_t)atoi(strtok(NULL, " \n"));
        else if (!strcmp(token, "cum")) {
            st->cum[0] = (int16_t)atoi(strtok(NULL, " \n"));
            st->cum[1] = (int16_t)atoi(strtok(NULL, " \n"));
        } else if (!strncmp(token, "hand", 4) &&
                   token[4] >= '0' && token[4] <= '1') {
            int p = token[4] - '0';
            char *name;
            while ((name = strtok(NULL, " \n"))) {
                int c = name_id_free(name, used);
                if (c < 0) { fclose(file); return 0; }
                used |= UINT64_C(1) << c;
                st->hand[p] |= UINT64_C(1) << c;
                st->hand_n[p]++;
            }
        } else if (!strncmp(token, "known", 5) &&
                   token[5] >= '0' && token[5] <= '1') {
            int p = token[5] - '0';
            char *name;
            while ((name = strtok(NULL, " \n"))) {
                char card_name[8];
                for (int c = 0; c < NCARD; c++) {
                    lc_card_name(c, card_name);
                    if (!strcasecmp(card_name, name) &&
                        ((st->hand[p] >> c) & UINT64_C(1)) &&
                        !((st->known[p] >> c) & UINT64_C(1))) {
                        st->known[p] |= UINT64_C(1) << c;
                        break;
                    }
                }
            }
        } else if (!strcmp(token, "exp")) {
            int p = atoi(strtok(NULL, " \n"));
            int suit = atoi(strtok(NULL, " \n"));
            char *name;
            while ((name = strtok(NULL, " \n"))) {
                int c = name_id_free(name, used);
                if (c < 0) { fclose(file); return 0; }
                used |= UINT64_C(1) << c;
                st->played[p] |= UINT64_C(1) << c;
                st->exp_n[p][suit]++;
                if (CARD_IS_WAGER(c)) {
                    st->exp_wager[p][suit]++;
                } else {
                    int value = CARD_VALUE(c);
                    if (value > st->exp_top[p][suit])
                        st->exp_top[p][suit] = (uint8_t)value;
                    st->exp_sum[p][suit] =
                        (uint8_t)(st->exp_sum[p][suit] + value);
                }
            }
        } else if (!strcmp(token, "pile")) {
            int suit = atoi(strtok(NULL, " \n"));
            char *name;
            while ((name = strtok(NULL, " \n"))) {
                int c = name_id_free(name, used);
                if (c < 0) { fclose(file); return 0; }
                used |= UINT64_C(1) << c;
                st->pile[suit][st->pile_n[suit]++] = (uint8_t)c;
                st->discarded |= UINT64_C(1) << c;
            }
        }
    }
    fclose(file);
    return 1;
}

static int move_is(Move move, int suit, int value, int discard, int draw)
{
    int rank = CARD_IS_WAGER(move.card) ? 0 : CARD_VALUE(move.card);
    return CARD_SUIT(move.card) == suit && rank == value &&
           move.discard == discard && move.draw == draw;
}

static int move_named(Move move, const char *card, int discard, int draw)
{
    char name[8];
    lc_card_name(move.card, name);
    return !strcasecmp(name, card) && move.discard == discard &&
           move.draw == draw;
}

static int no_recursive_replan_work(const SearchStats *stats)
{
    return stats->deck2_replans == 0 &&
           stats->deck2_replan_worlds == 0 &&
           stats->deck2_replan_evals == 0 &&
           stats->deck2_replan_cap_hits == 0 &&
           stats->deck2_replan_cache_hits == 0 &&
           stats->deck2_replan_cycle_closures == 0 &&
           stats->deck2_replan_max_depth == 0 &&
           stats->deck2_replan_root_calls == 0 &&
           stats->deck2_replan_root_worlds == 0 &&
           stats->deck2_replan_max_stall_chain == 0 &&
           stats->deck2_replan_low_world_fallbacks == 0;
}

typedef struct {
    Move deck;
    Move pile;
    int have_deck;
    int have_pile;
} PolicyCore;

static int same_core(Move a, Move b)
{
    return CARD_SUIT(a.card) == CARD_SUIT(b.card) &&
           CARD_VALUE(a.card) == CARD_VALUE(b.card) &&
           a.discard == b.discard;
}

static int collect_policy_cores(const State *st, PolicyCore out[4])
{
    Move mv[MAX_MOVES];
    int n = lc_moves(st, mv), count = 0;
    memset(out, 0, 4 * sizeof *out);
    for (int i = 0; i < n; i++) {
        if (CARD_IS_WAGER(mv[i].card)) continue;
        int k = -1;
        for (int j = 0; j < count; j++)
            if (same_core(mv[i], out[j].deck)) {
                k = j;
                break;
            }
        if (k < 0) {
            if (count == 4) continue;
            k = count++;
            out[k].deck = mv[i];
        }
        if (mv[i].draw == 0) {
            out[k].deck = mv[i];
            out[k].have_deck = 1;
        } else if (!out[k].have_pile) {
            out[k].pile = mv[i];
            out[k].have_pile = 1;
        }
    }
    for (int i = 0; i < count; i++)
        if (!out[i].have_deck || !out[i].have_pile) return 0;
    return count;
}

static void set_move_logit(Net *net, Move move, float logit)
{
    int ip = move.card * 2 + move.discard;
    int draw = move.draw;
    net->bcomb[ip * NET_NDRAW + draw] = logit;
}

static void test_policy_shortlist_invariants(void)
{
    State st;
    CHECK(load_state("data/probes/ui_seed95647345759839_p43.state", &st),
          "cannot load shortlist fixture");
    PolicyCore core[4];
    CHECK(collect_policy_cores(&st, core) == 4,
          "shortlist fixture lacks four numbered deck/pile cores");

    Net *net = malloc(sizeof *net);
    CHECK(net != NULL, "cannot allocate shortlist net");
    if (!net) return;
    net_zero(net);

    /* Deployed greedy policy keeps the first legal move on an exact tie.  In
     * this fixture that move is not the lowest MOVE_PACK value, so the check
     * catches a resolver-only packed tie-break that would silently replace
     * the baseline candidate. */
    Move legal[MAX_MOVES];
    int nlegal = lc_moves(&st, legal);
    int lower_packed = 0;
    for (int i = 1; i < nlegal; i++)
        if (MOVE_PACK(legal[i]) < MOVE_PACK(legal[0])) lower_packed = 1;
    CHECK(nlegal > 1 && lower_packed,
          "tie-baseline fixture does not distinguish legal/packed order");
    Move tied_candidate[6];
    float tied_prior[6];
    int tied_n = late_resolver_policy_candidates(
        net, &st, 3, 1, 6, 0, tied_candidate, tied_prior);
    CHECK(tied_n > 0 && semantic_move_equal(tied_candidate[0], legal[0]),
          "resolver candidate zero drifted from deployed policy tie behavior");
    int first_deck = -1;
    for (int i = 0; i < nlegal && first_deck < 0; i++)
        if (legal[i].draw == 0) first_deck = i;
    tied_n = late_resolver_policy_candidates(
        net, &st, 3, 1, 6, 1, tied_candidate, tied_prior);
    CHECK(first_deck >= 0 && tied_n > 0 &&
          semantic_move_equal(tied_candidate[0], legal[first_deck]),
          "resolver descendant drifted from deployed policy tie behavior");

    for (int i = 0; i < NET_NCOMB; i++) net->bcomb[i] = -40.0f;

    /* Three cores have more aggregate probability through two nearly-best
     * complete moves.  The fourth nevertheless owns the literal policy
     * argmax and must remain candidate zero even with cores=3. */
    for (int i = 0; i < 3; i++) {
        set_move_logit(net, core[i].deck, 9.9f);
        set_move_logit(net, core[i].pile, 9.9f);
    }
    set_move_logit(net, core[3].deck, 10.0f);
    Move candidate[6];
    float prior[6];
    int nc = late_resolver_policy_candidates(
        net, &st, 3, 1, 6, 0, candidate, prior);
    CHECK(nc > 0 && semantic_move_equal(candidate[0], core[3].deck),
          "literal complete-move argmax was evicted by aggregate core mass");

    /* At a forced-progress node, forbidden pile probability must not promote
     * a weak deck action into the three-core shortlist. */
    net_zero(net);
    for (int i = 0; i < NET_NCOMB; i++) net->bcomb[i] = -40.0f;
    set_move_logit(net, core[0].deck, 1.0f);
    set_move_logit(net, core[0].pile, 20.0f);
    set_move_logit(net, core[1].deck, 10.0f);
    set_move_logit(net, core[2].deck, 9.0f);
    set_move_logit(net, core[3].deck, 8.0f);
    nc = late_resolver_policy_candidates(
        net, &st, 3, 1, 6, 1, candidate, prior);
    CHECK(nc == 3, "forced-progress shortlist has %d candidates", nc);
    int found_weak = 0, found_fourth = 0;
    for (int i = 0; i < nc; i++) {
        CHECK(candidate[i].draw == 0,
              "forced-progress shortlist retained a pile draw");
        if (same_core(candidate[i], core[0].deck)) found_weak = 1;
        if (same_core(candidate[i], core[3].deck)) found_fourth = 1;
    }
    CHECK(!found_weak && found_fourth,
          "forced-progress cores were ranked by forbidden pile mass");
    free(net);
}

static void test_assignment_support(void)
{
    State d2 = late_state(2), d3 = late_state(3);
    CHECK(!d2.over && d2.deck_left == 2, "failed to construct deck-two state");
    CHECK(!d3.over && d3.deck_left == 3, "failed to construct deck-three state");
    CHECK(late_resolver_assignment_count(&d2, d2.turn) == 90,
          "deck-two mover support is %d, expected 90",
          late_resolver_assignment_count(&d2, d2.turn));
    CHECK(late_resolver_assignment_count(&d3, d3.turn) == 990,
          "deck-three mover support is %d, expected 990",
          late_resolver_assignment_count(&d3, d3.turn));
}

static void test_bounded_resolver_information_invariance(void)
{
    State st = late_state(2);
    const int p = st.turn, o = p ^ 1;
    uint8_t opponent[HAND_SIZE];
    int nopp = lc_hand_cards(&st, o, opponent);
    CHECK(nopp == HAND_SIZE, "late opponent hand has %d cards", nopp);

    /* Expose seven cards so this focused regression has six ordered worlds
     * while still exercising hidden assignment construction. */
    st.known[o] = 0;
    for (int i = 0; i < HAND_SIZE - 1; i++)
        st.known[o] |= UINT64_C(1) << opponent[i];
    CHECK(late_resolver_assignment_count(&st, p) == 6,
          "known-card reduced support is %d, expected 6",
          late_resolver_assignment_count(&st, p));

    Net *net = malloc(sizeof *net);
    CHECK(net != NULL, "cannot allocate late resolver test net");
    if (!net) return;
    net_zero(net);

    Move first = { 0 }, second = { 0 };
    LateResolverStats a, b;
    int ok_a = late_resolver_choose(
        net, &st, 0, 3, 1, 6, 0.0, &first, &a);
    CHECK(!a.unavailable, "bounded resolver failed on six-particle state");
    CHECK(a.support == 6, "resolver reported support %d", a.support);
    CHECK(a.root_candidates >= 1 && a.root_candidates <= 6,
          "resolver used %d root candidates", a.root_candidates);
    CHECK(a.horizon2_nodes > 0 && a.horizon4_nodes > 0,
          "resolver did not construct both observation-keyed policies");
    CHECK(a.horizon2_transitions > 0 && a.horizon4_transitions > 0,
          "resolver did not evaluate bounded trajectories");
    CHECK(a.horizon2_exact_leaves > 0 && a.horizon4_exact_leaves > 0,
          "resolver did not propagate exact one-card leaves");
    CHECK(ok_a == a.passed,
          "resolver returned %d with pass status %d", ok_a, a.passed);
    CHECK(a.horizon2_nodes == a.horizon2_root_nodes +
                              a.horizon2_frozen_opponent_nodes &&
          a.horizon4_nodes == a.horizon4_root_nodes +
                              a.horizon4_frozen_opponent_nodes,
          "resolver node actor census is inconsistent");
    CHECK(move_legal(&st, a.horizon2_move) &&
          move_legal(&st, a.horizon4_move),
          "a horizon result is not legal at the root");
    if (ok_a) CHECK(semantic_move_equal(first, a.horizon4_move),
                    "returned move differs from stable H=4 move");

    /* Reusing a caller-owned proof must be exactly the self-contained public
     * entry point, while a foreign proof must fail closed to complete
     * inference rather than selecting another checkpoint's shortcuts. */
    NetEvalPlan eval_plan;
    net_eval_plan_init(net, &eval_plan);
    Move planned = { 0 }, foreign = { 0 };
    LateResolverStats planned_stats, foreign_stats;
    int ok_planned = late_resolver_choose_plan(
        net, &st, 0, 3, 1, 6, 0.0,
        &planned, &planned_stats, &eval_plan);
    NetEvalPlan foreign_plan = eval_plan;
    foreign_plan.owner = NULL;
    int ok_foreign = late_resolver_choose_plan(
        net, &st, 0, 3, 1, 6, 0.0,
        &foreign, &foreign_stats, &foreign_plan);
    CHECK(ok_planned == ok_a && ok_foreign == ok_a &&
          MOVE_PACK(planned) == MOVE_PACK(first) &&
          MOVE_PACK(foreign) == MOVE_PACK(first) &&
          memcmp(&planned_stats, &a, sizeof a) == 0 &&
          memcmp(&foreign_stats, &a, sizeof a) == 0,
          "planned, fallback, and fail-closed resolver outputs differ");

    /* Root candidates and priors belong to the root checkpoint; every policy
     * node after a candidate belongs to the continuation checkpoint.  Two
     * deliberately opposite continuation sentinels must therefore change
     * bounded values without changing any root row. */
    Net *play_continuation = malloc(sizeof *play_continuation);
    Net *discard_continuation = malloc(sizeof *discard_continuation);
    CHECK(play_continuation && discard_continuation,
          "cannot allocate dual-network late sentinels");
    if (play_continuation && discard_continuation) {
        net_zero(play_continuation);
        net_zero(discard_continuation);
        for (int c = 0; c < NCARD; c++) {
            play_continuation->bplay[c * 2 + 0] = 16.0f;
            play_continuation->bplay[c * 2 + 1] = -16.0f;
            discard_continuation->bplay[c * 2 + 0] = -16.0f;
            discard_continuation->bplay[c * 2 + 1] = 16.0f;
        }
        play_continuation->bdraw[0] = 8.0f;
        discard_continuation->bdraw[0] = 8.0f;
        for (int d = 1; d < NET_NDRAW; d++) {
            play_continuation->bdraw[d] = -8.0f;
            discard_continuation->bdraw[d] = -8.0f;
        }
        NetEvalPlan play_plan, discard_plan;
        net_eval_plan_init(play_continuation, &play_plan);
        net_eval_plan_init(discard_continuation, &discard_plan);
        LateResolverStats play_stats, discard_stats;
        Move play_move = { 0 }, discard_move = { 0 };
        int play_ok = late_resolver_choose_dual_plan(
            net, play_continuation, &st, 0, 3, 1, 6, 0.0,
            &play_move, &play_stats, &eval_plan, &play_plan);
        int discard_ok = late_resolver_choose_dual_plan(
            net, discard_continuation, &st, 0, 3, 1, 6, 0.0,
            &discard_move, &discard_stats, &eval_plan, &discard_plan);
        CHECK(!play_stats.unavailable && !discard_stats.unavailable,
              "dual-network late sentinel was unavailable");
        int same_root =
            play_stats.root_candidates == discard_stats.root_candidates;
        int changed_descendant = 0;
        for (int i = 0; same_root && i < play_stats.root_candidates; i++) {
            if (!semantic_move_equal(play_stats.candidate[i],
                                     discard_stats.candidate[i]) ||
                play_stats.prior[i] != discard_stats.prior[i])
                same_root = 0;
            if (play_stats.horizon2_q[i] != discard_stats.horizon2_q[i] ||
                play_stats.horizon4_q[i] != discard_stats.horizon4_q[i])
                changed_descendant = 1;
        }
        CHECK(same_root,
              "continuation sentinel changed bounded root candidates/priors");
        CHECK(changed_descendant,
              "continuation sentinel did not reach bounded descendants");

        /* Both plans are independently owner-bound.  Swapping them must fail
         * closed to complete inference for each role, byte-identically. */
        LateResolverStats swapped_stats;
        Move swapped_move = { 0 };
        int swapped_ok = late_resolver_choose_dual_plan(
            net, play_continuation, &st, 0, 3, 1, 6, 0.0,
            &swapped_move, &swapped_stats, &play_plan, &eval_plan);
        CHECK(swapped_ok == play_ok &&
              (!play_ok || semantic_move_equal(swapped_move, play_move)) &&
              memcmp(&swapped_stats, &play_stats,
                     sizeof play_stats) == 0,
              "foreign root/continuation plans did not fail closed exactly");
        (void)discard_ok;
    }
    free(play_continuation);
    free(discard_continuation);

    /* The production rollout owns one proof for its whole decision.  Compare
     * that path with the full-inference oracle at a real resolver entry,
     * including every diagnostic byte and the gameplay RNG state. */
    Agent actor;
    agent_default(&actor, AG_ROLLOUT, net);
    actor.symmetries = 1;
    actor.root_width = 6;
    actor.no_belief = 1;
    actor.exact_terminal = 1;
    actor.bounded_late_root = 1;
    actor.bounded_late_min = 0.0f;
    Rng planned_rng, reference_rng;
    rng_seed(&planned_rng, UINT64_C(0x4556414C504C414E));
    reference_rng = planned_rng;
    SearchStats rollout_stats, reference_stats;
    float rollout_value = 0.0f, reference_value = 0.0f;
    Move rollout_selected = rollout_move(
        &actor, &st, &planned_rng, &rollout_value, &rollout_stats);
    Move reference_selected = rollout_move_reference_for_test(
        &actor, &st, &reference_rng, &reference_value, &reference_stats);
    CHECK(rollout_stats.late_resolver_attempted &&
          rollout_stats.late_resolver_completed &&
          MOVE_PACK(rollout_selected) == MOVE_PACK(reference_selected) &&
          memcmp(&rollout_value, &reference_value,
                 sizeof rollout_value) == 0 &&
          memcmp(&rollout_stats, &reference_stats,
                 sizeof rollout_stats) == 0 &&
          memcmp(&planned_rng, &reference_rng,
                 sizeof planned_rng) == 0,
          "reused-plan late rollout changed move, value, diagnostics, or RNG");

    /* Swap the only hidden opponent card with a hidden deck card.  The mover
     * observation is unchanged, so results and deterministic work must match. */
    State hidden = st;
    int hidden_card = opponent[HAND_SIZE - 1];
    int deck_card = hidden.deck[hidden.deck_pos];
    hidden.hand[o] &= ~(UINT64_C(1) << hidden_card);
    hidden.hand[o] |= UINT64_C(1) << deck_card;
    hidden.deck[hidden.deck_pos] = (uint8_t)hidden_card;
    int ok_b = late_resolver_choose(
        net, &hidden, 0, 3, 1, 6, 0.0, &second, &b);
    CHECK(ok_a == ok_b && a.stable == b.stable &&
          a.unavailable == b.unavailable,
          "hidden swap changed resolver availability/stability");
    CHECK(semantic_move_equal(a.horizon2_move, b.horizon2_move) &&
          semantic_move_equal(a.horizon4_move, b.horizon4_move),
          "hidden swap changed a bounded horizon decision");
    CHECK(fabs(a.horizon2_value - b.horizon2_value) < 1e-12 &&
          fabs(a.horizon4_value - b.horizon4_value) < 1e-12,
          "hidden swap changed bounded values");
    CHECK(a.horizon2_nodes == b.horizon2_nodes &&
          a.horizon4_nodes == b.horizon4_nodes &&
          a.horizon2_transitions == b.horizon2_transitions &&
          a.horizon4_transitions == b.horizon4_transitions,
          "hidden swap changed deterministic resolver work");
    if (ok_b) CHECK(semantic_move_equal(first, second),
                    "hidden swap changed stable returned move");
    free(net);
}

static void test_locked_late_positions(void)
{
    Net *net = malloc(sizeof *net);
    CHECK(net != NULL, "cannot allocate locked-position net");
    if (!net) return;
    CHECK(net_load(net, "data/champion.bin") == 0,
          "cannot load data/champion.bin");

    Agent integrated;
    agent_default(&integrated, AG_ROLLOUT, net);
    integrated.symmetries = 20;
    integrated.root_width = 5;
    integrated.no_belief = 1;
    integrated.exact_terminal = 1;
    integrated.bounded_late_root = 1;
    integrated.bounded_late_min = 1.0f;
    CHECK(integrated.prefix_confirm_k == 0.0f &&
          integrated.prefix_confirm_min == 0.0f,
          "bounded late gain gate is coupled to ordinary prefix settings");
    Rng rng;
    rng_seed(&rng, UINT64_C(0x4C41544553544154));

    State p42;
    CHECK(load_state("data/probes/ui_seed95647345759839_p42.state", &p42),
          "cannot load locked p42 state");
    SearchStats ss42;
    Move selected42 = rollout_move(
        &integrated, &p42, &rng, NULL, &ss42);
    CHECK(ss42.late_resolver_attempted && ss42.late_resolver_completed &&
          ss42.late_resolver_used && ss42.late_resolver_retained &&
          !ss42.late_resolver_override,
          "p42 completed rejection was not an authoritative policy retain");
    CHECK(ss42.late_resolver_support == 990,
          "p42 support is %d, expected 990", ss42.late_resolver_support);
    CHECK(!ss42.late_resolver_passed && ss42.late_resolver_stable &&
          ss42.late_resolver_h2_delta <= 1.0 &&
          ss42.late_resolver_h4_delta > 1.0,
          "p42 did not retain policy on inconsistent practical gain");
    CHECK(move_named(selected42, "W8", 0, 0) &&
          semantic_move_equal(selected42, ss42.late_resolver_candidate[0]),
          "p42 did not retain its legal literal W8+deck policy baseline");
    CHECK(move_legal(&p42, selected42),
          "p42 authoritative baseline is not legal");
    CHECK(ss42.late_resolver_h2_root_nodes > 0 &&
          ss42.late_resolver_h4_root_nodes > 0 &&
          ss42.late_resolver_h2_frozen_opponent_nodes > 0 &&
          ss42.late_resolver_h4_frozen_opponent_nodes > 0,
          "p42 did not exercise root-only improvement and frozen opponent nodes");
    CHECK(ss42.late_resolver_h2_nodes ==
              ss42.late_resolver_h2_root_nodes +
              ss42.late_resolver_h2_frozen_opponent_nodes &&
          ss42.late_resolver_h4_nodes ==
              ss42.late_resolver_h4_root_nodes +
              ss42.late_resolver_h4_frozen_opponent_nodes,
          "p42 node actor census is inconsistent");
    CHECK(no_recursive_replan_work(&ss42),
          "p42 authoritative panel fell through to recursive replan work");

    State p43;
    CHECK(load_state("data/probes/ui_seed95647345759839_p43.state", &p43),
          "cannot load locked p43 state");
    Move selected43 = { 0 };
    LateResolverStats s43;
    int ok43 = late_resolver_choose(
        net, &p43, 0, 3, 20, 5, 1.0, &selected43, &s43);
    CHECK(ok43 && s43.stable && !s43.unavailable,
          "p43 bounded horizons did not stabilize");
    CHECK(s43.support == 90,
          "p43 support is %d, expected 90", s43.support);
    CHECK(!move_is(s43.horizon2_move, 1, 10, 0, 0) &&
          !move_is(s43.horizon4_move, 1, 10, 0, 0),
          "p43 selected the inferior B10+deck ending");

    int b10_y = -1, y10_g = -1;
    for (int i = 0; i < s43.root_candidates; i++) {
        if (move_is(s43.candidate[i], 1, 10, 0, 1)) b10_y = i;
        if (move_is(s43.candidate[i], 0, 10, 0, 4)) y10_g = i;
    }
    CHECK(b10_y >= 0 && y10_g >= 0,
          "p43 focused candidates omit one of the tied stalls");
    if (b10_y >= 0 && y10_g >= 0) {
        CHECK(fabs(s43.horizon2_q[b10_y] - s43.horizon2_q[y10_g]) < 1e-12,
              "p43 H=2 tied stalls differ: %.12f vs %.12f",
              s43.horizon2_q[b10_y], s43.horizon2_q[y10_g]);
        CHECK(fabs(s43.horizon4_q[b10_y] - s43.horizon4_q[y10_g]) < 1e-12,
              "p43 H=4 tied stalls differ: %.12f vs %.12f",
              s43.horizon4_q[b10_y], s43.horizon4_q[y10_g]);
    }

    rng_seed(&rng, UINT64_C(0x4C41544553544154));
    SearchStats ss;
    Move integrated_move = rollout_move(
        &integrated, &p43, &rng, NULL, &ss);
    CHECK(semantic_move_equal(integrated_move, selected43),
          "rollout integration did not deploy the passed late decision");
    CHECK(ss.late_resolver_attempted && ss.late_resolver_completed &&
          ss.late_resolver_stable && ss.late_resolver_passed &&
          ss.late_resolver_used && !ss.late_resolver_retained &&
          ss.late_resolver_override &&
          ss.late_resolver_support == s43.support &&
          ss.late_resolver_candidates == s43.root_candidates,
          "rollout integration omitted resolver lifecycle diagnostics");
    CHECK(ss.late_resolver_h2_nodes == s43.horizon2_nodes &&
          ss.late_resolver_h4_nodes == s43.horizon4_nodes &&
          ss.late_resolver_h2_frozen_opponent_nodes ==
              s43.horizon2_frozen_opponent_nodes &&
          ss.late_resolver_h4_frozen_opponent_nodes ==
              s43.horizon4_frozen_opponent_nodes &&
          ss.late_resolver_h2_exact_leaves == s43.horizon2_exact_leaves &&
          ss.late_resolver_h4_exact_leaves == s43.horizon4_exact_leaves,
          "rollout integration altered dedicated resolver work diagnostics");
    for (int i = 0; i < s43.root_candidates; i++)
        CHECK(semantic_move_equal(ss.late_resolver_candidate[i],
                                  s43.candidate[i]) &&
              fabs(ss.late_resolver_prior[i] - s43.prior[i]) < 1e-12 &&
              fabs(ss.late_resolver_h2_q[i] - s43.horizon2_q[i]) < 1e-12 &&
              fabs(ss.late_resolver_h4_q[i] - s43.horizon4_q[i]) < 1e-12,
              "rollout integration lost resolver candidate %d", i);
    CHECK(no_recursive_replan_work(&ss),
          "finite-support resolver work leaked into recursive replan counters");

    /* The resolver reasons about wager copies semantically, while lc_apply
     * requires a physical card that is actually in hand.  Swap the held and
     * already-played Yellow wager IDs, force that discard to be the literal
     * policy baseline, then make the practical gate impossible to clear.  The
     * authoritative retain must return the newly held canonical ID—not a
     * semantically equivalent but illegal ID cached by the resolver. */
    State wager_state = p43;
    uint64_t yellow_wagers = UINT64_C(7) << (0 * NRANK);
    uint64_t held_wagers = wager_state.hand[0] & yellow_wagers;
    uint64_t played_wagers = wager_state.played[0] & yellow_wagers;
    CHECK(held_wagers && played_wagers,
          "p43 lacks wagers for semantic legal-move regression");
    if (held_wagers && played_wagers) {
        int held = __builtin_ctzll(held_wagers);
        int played = __builtin_ctzll(played_wagers);
        wager_state.hand[0] &= ~(UINT64_C(1) << held);
        wager_state.hand[0] |= UINT64_C(1) << played;
        wager_state.played[0] &= ~(UINT64_C(1) << played);
        wager_state.played[0] |= UINT64_C(1) << held;

        Net *wager_net = malloc(sizeof *wager_net);
        CHECK(wager_net != NULL,
              "cannot allocate wager canonicalization net");
        if (wager_net) {
            net_zero(wager_net);
            for (int r = 0; r < WAGERS_PER_SUIT; r++)
                wager_net->bplay[(CARD_MAKE(0, r) * 2) + 1] = 20.0f;
            wager_net->bdraw[0] = 10.0f;
            Agent wager_actor;
            agent_default(&wager_actor, AG_ROLLOUT, wager_net);
            wager_actor.symmetries = 1;
            wager_actor.root_width = 5;
            wager_actor.no_belief = 1;
            wager_actor.exact_terminal = 1;
            wager_actor.bounded_late_root = 1;
            wager_actor.bounded_late_min = 1000000.0f;
            SearchStats wager_stats;
            rng_seed(&rng, UINT64_C(0x57414745524C4547));
            Move wager_move = rollout_move(
                &wager_actor, &wager_state, &rng, NULL, &wager_stats);
            CHECK(wager_stats.late_resolver_completed &&
                  wager_stats.late_resolver_used &&
                  wager_stats.late_resolver_retained &&
                  !wager_stats.late_resolver_override,
                  "wager policy baseline was not authoritatively retained");
            CHECK(move_named(wager_move, "Yx", 1, 0) &&
                  wager_move.card == played &&
                  ((wager_state.hand[0] >> wager_move.card) & UINT64_C(1)) &&
                  move_legal(&wager_state, wager_move),
                  "semantic wager retain did not return the held legal ID");
            free(wager_net);
        }
    }
    free(net);
}

int main(void)
{
    test_assignment_support();
    test_policy_shortlist_invariants();
    test_bounded_resolver_information_invariance();
    test_locked_late_positions();
    if (failures == 0) {
        printf("late resolver regression tests passed\n");
        return 0;
    }
    printf("%d late resolver regression failures\n", failures);
    return 1;
}
