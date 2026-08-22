/* White-box contracts for continuation-only PPO support reconstruction. */
#define main rl_cli_main_for_test
#include "../tools/rl.c"
#undef main

#include <math.h>
#include <stdio.h>

static int failures;
#define CHECK(cond, ...) do { if (!(cond)) {                            \
    fprintf(stderr, "FAIL %s:%d: ", __FILE__, __LINE__);              \
    fprintf(stderr, __VA_ARGS__); fputc('\n', stderr); failures++;     \
} } while (0)

static int dominated_fixture(State *st, Move *mv, int *n_out,
                             uint8_t *allowed, int *masked_out)
{
    uint8_t deck[NCARD];
    for (int i = 0; i < NCARD; i++) deck[i] = (uint8_t)i;
    lc_deal_from_deck(st, deck);
    int p = st->turn, dead_card = -1;
    uint64_t hand = st->hand[p];
    while (hand) {
        int c = __builtin_ctzll(hand);
        hand &= hand - 1;
        if (!CARD_IS_WAGER(c)) {
            dead_card = c;
            break;
        }
    }
    if (dead_card < 0) return 0;
    int suit = CARD_SUIT(dead_card), value = CARD_VALUE(dead_card);
    st->exp_top[0][suit] = (uint8_t)value;
    st->exp_top[1][suit] = (uint8_t)value;
    st->exp_n[0][suit] = st->exp_n[1][suit] = 1;
    st->exp_sum[0][suit] = st->exp_sum[1][suit] = (uint8_t)value;

    int n = lc_moves(st, mv);
    uint64_t dead = lc_dead_cards(st) & st->hand[st->turn];
    int masked = 0;
    for (int i = 0; i < n; i++) {
        allowed[i] = (uint8_t)!(dead &&
            lc_discard_dominated(st, mv[i], dead));
        masked += !allowed[i];
    }
    *n_out = n;
    *masked_out = masked;
    return n > 1 && masked > 0 && masked < n;
}

static void run_optimizer_contract(const Net *net, const Net *anchor,
                                   const RLSample *sample, float temp,
                                   float entcoef, float klcoef, Net *grad,
                                   double *ploss, double *klloss,
                                   double *clipped)
{
    int index = 0;
    OptJob job;
    memset(&job, 0, sizeof job);
    job.net = net;
    job.anchor = anchor;
    job.grad = grad;
    job.buf = sample;
    job.idx = &index;
    job.from = 0;
    job.to = 1;
    job.clip = 0.2f;
    job.vcoef = 0.0f;
    job.entcoef = entcoef;
    job.policy_scale = 1.0f;
    job.temp = temp;
    job.klcoef = klcoef;
    opt_worker(&job);
    *ploss = job.ploss;
    *klloss = job.klloss;
    *clipped = job.clipped;
}

static void test_continuation_support_and_gradients(void)
{
    State st;
    Move mv[MAX_MOVES];
    uint8_t expected[MAX_MOVES], allowed[MAX_MOVES];
    int n = 0, nmasked = 0;
    CHECK(dominated_fixture(&st, mv, &n, expected, &nmasked),
          "could not construct dominated-discard fixture");
    if (n <= 1 || nmasked <= 0) return;

    const float temp = 0.65f;
    float raw[MAX_MOVES], behavior[MAX_MOVES];
    for (int i = 0; i < n; i++) raw[i] = 1.0f / (float)n;
    uint64_t dead = 0;
    CHECK(continuation_condition_policy(
              &st, mv, n, temp, raw, behavior, allowed, &dead) == n,
          "conditional continuation policy failed");
    CHECK(dead != 0, "conditional continuation policy returned no dead mask");
    double raw_sum = 0.0, behavior_sum = 0.0;
    int chosen = -1;
    for (int i = 0; i < n; i++) {
        CHECK(allowed[i] == expected[i],
              "conditional support disagrees at move %d", i);
        if (!allowed[i]) {
            CHECK(raw[i] == 0.0f && behavior[i] == 0.0f,
                  "masked move %d retained behavior mass", i);
        } else if (chosen < 0) {
            chosen = i;
        }
        raw_sum += raw[i];
        behavior_sum += behavior[i];
    }
    CHECK(fabs(raw_sum - 1.0) < 1e-6 &&
          fabs(behavior_sum - 1.0) < 1e-6,
          "conditional policies were not normalized");

    Net *net = calloc(1, sizeof *net);
    Net *anchor = calloc(1, sizeof *anchor);
    Net *grad = malloc(sizeof *grad);
    CHECK(net && anchor && grad, "network allocation failed");
    if (!net || !anchor || !grad || chosen < 0) {
        free(net); free(anchor); free(grad);
        return;
    }
    net_zero(net);
    net_zero(anchor);
    RLSample sample;
    memset(&sample, 0, sizeof sample);
    sample.st = st;
    sample.persp = st.turn;
    sample.actor = 1;
    sample.continuation_role = 1;
    sample.chosen = MOVE_PACK(mv[chosen]);
    sample.oldp = behavior[chosen];
    sample.adv = 1.0f;

    double ploss = 0.0, klloss = 0.0, clipped = 0.0;
    float uniform_oldp = sample.oldp;
    run_optimizer_contract(net, NULL, &sample, temp, 0.0f, 0.0f, grad,
                           &ploss, &klloss, &clipped);
    CHECK(fabs(ploss + 1.0) < 2e-6 && clipped == 0.0,
          "first continuation update ratio was not one: loss %.9f clip %.1f",
          ploss, clipped);
    int allowed_gradient = 0, first_masked_comb = -1;
    for (int i = 0; i < n; i++) {
        int comb = (mv[i].card * 2 + mv[i].discard) * NET_NDRAW +
                   mv[i].draw;
        if (!allowed[i]) {
            if (first_masked_comb < 0) first_masked_comb = comb;
            CHECK(grad->bcomb[comb] == 0.0f,
                  "masked move %d received PPO/entropy gradient %.9g",
                  i, grad->bcomb[comb]);
        } else if (grad->bcomb[comb] != 0.0f) {
            allowed_gradient = 1;
        }
    }
    CHECK(allowed_gradient, "allowed continuation support received no gradient");

    /* Make the allowed policy nonuniform and isolate entropy (advantage zero)
     * so a zero masked gradient cannot be an artifact of zero entropy loss. */
    int biased_comb = (mv[chosen].card * 2 + mv[chosen].discard) *
                      NET_NDRAW + mv[chosen].draw;
    net->bcomb[biased_comb] = 1.25f;
    Move biased_mv[MAX_MOVES];
    float biased_raw[MAX_MOVES], biased_behavior[MAX_MOVES];
    int biased_n = policy_probs(net, &st, biased_mv, biased_raw, NULL);
    CHECK(biased_n == n, "biased policy changed legal support size");
    CHECK(continuation_condition_policy(
              &st, biased_mv, biased_n, temp, biased_raw, biased_behavior,
              NULL, NULL) == n,
          "biased conditional policy failed");
    int biased_chosen = -1;
    for (int i = 0; i < n; i++)
        if (MOVE_PACK(biased_mv[i]) == sample.chosen) biased_chosen = i;
    CHECK(biased_chosen >= 0, "biased policy lost chosen move");
    if (biased_chosen >= 0) sample.oldp = biased_behavior[biased_chosen];
    sample.adv = 0.0f;
    run_optimizer_contract(net, NULL, &sample, temp, 0.25f, 0.0f, grad,
                           &ploss, &klloss, &clipped);
    int entropy_allowed_gradient = 0;
    for (int i = 0; i < n; i++) {
        int comb = (mv[i].card * 2 + mv[i].discard) * NET_NDRAW + mv[i].draw;
        if (!allowed[i])
            CHECK(grad->bcomb[comb] == 0.0f,
                  "masked move %d received entropy gradient %.9g",
                  i, grad->bcomb[comb]);
        else if (grad->bcomb[comb] != 0.0f)
            entropy_allowed_gradient = 1;
    }
    CHECK(entropy_allowed_gradient,
          "nonuniform allowed policy received no entropy gradient");
    net->bcomb[biased_comb] = 0.0f;
    sample.oldp = uniform_oldp;
    sample.adv = 1.0f;

    /* The behavior mask does not erase the conservative full-legal anchor:
     * only KL is allowed to move a masked logit. */
    CHECK(first_masked_comb >= 0, "fixture had no masked combination");
    if (first_masked_comb >= 0) {
        anchor->bcomb[first_masked_comb] = 6.0f;
        run_optimizer_contract(net, anchor, &sample, temp, 0.0f, 0.5f, grad,
                               &ploss, &klloss, &clipped);
        CHECK(klloss > 0.0 && grad->bcomb[first_masked_comb] != 0.0f,
              "full-legal anchor did not protect a masked logit");
    }

    /* Suit augmentation must reconstruct the mask in the stored coordinate
     * system.  The same frozen model then gives ratio exactly one. */
    const uint8_t perm[NSUIT] = { 2, 0, 4, 1, 3 };
    State view;
    Move view_mv[MAX_MOVES], engine_mv[MAX_MOVES];
    float view_raw[MAX_MOVES], view_behavior[MAX_MOVES];
    NetEvalPlan plan;
    net_eval_plan_init(net, &plan);
    int vn = continuation_trajectory_policy_probs_plan(
        net, &plan, &st, perm, temp, &view, view_mv, engine_mv,
        view_raw, view_behavior);
    CHECK(vn > 0, "augmented conditional policy failed");
    int view_chosen = -1, view_masked = 0;
    uint64_t view_dead = lc_dead_cards(&view) & view.hand[view.turn];
    for (int i = 0; i < vn; i++) {
        int is_masked = view_dead &&
            lc_discard_dominated(&view, view_mv[i], view_dead);
        view_masked += is_masked;
        CHECK((view_raw[i] == 0.0f) == !!is_masked,
              "augmented stored-state mask mismatch at move %d", i);
        if (!is_masked && view_chosen < 0) view_chosen = i;
    }
    CHECK(view_masked > 0 && view_chosen >= 0,
          "augmented fixture lost its mixed support");
    if (view_chosen >= 0) {
        sample.st = view;
        sample.persp = view.turn;
        sample.chosen = MOVE_PACK(view_mv[view_chosen]);
        sample.oldp = view_behavior[view_chosen];
        run_optimizer_contract(net, NULL, &sample, temp, 0.0f, 0.0f, grad,
                               &ploss, &klloss, &clipped);
        CHECK(fabs(ploss + 1.0) < 2e-6 && clipped == 0.0,
              "augmented first-update ratio was not one: %.9f", ploss);
    }

    free(net);
    free(anchor);
    free(grad);
}

int main(void)
{
    test_continuation_support_and_gradients();
    if (failures == 0) {
        puts("continuation PPO support tests passed");
        return 0;
    }
    fprintf(stderr, "%d continuation PPO support failures\n", failures);
    return 1;
}
