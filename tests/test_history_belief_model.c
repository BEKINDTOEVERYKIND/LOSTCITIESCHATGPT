#include "../src/agent.h"
#include "../src/history_belief_model.h"
#include "../src/lc.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void require(int condition, const char *message)
{
    if (!condition) {
        fprintf(stderr, "test_history_belief_model: %s\n", message);
        exit(EXIT_FAILURE);
    }
}

static int feature_count(const uint32_t *feature, int n, uint32_t target)
{
    int count = 0;
    for (int i = 0; i < n; i++) count += feature[i] == target;
    return count;
}

static void permute_trace(const HistoryBeliefTrace *src,
                          HistoryBeliefTrace *dst,
                          const uint8_t permutation[NSUIT])
{
    *dst = *src;
    for (int i = 0; i < dst->n; i++) {
        dst->event[i].suit = permutation[src->event[i].suit];
        if (src->event[i].draw > 0)
            dst->event[i].draw = (uint8_t)(permutation[
                src->event[i].draw - 1] + 1);
    }
}

int main(void)
{
    Rng rng;
    rng_seed(&rng, UINT64_C(202608290111));
    State complete;
    lc_deal(&complete, &rng);
    complete.turn = 0;
    State opening_complete = complete;

    HistoryBeliefTrace trace;
    history_belief_trace_init(&trace, 1);
    Move first = { .card = CARD_MAKE(2, 5), .discard = 0, .draw = 0 };
    require(history_belief_trace_push(&trace, &complete, first),
            "could not record public opponent action");
    require(trace.n == 1, "opponent action was not retained");
    complete.nply = 1;
    complete.turn = 1;
    Move intervening = {
        .card = CARD_MAKE(1, 7), .discard = 0, .draw = 0
    };
    require(history_belief_trace_push(&trace, &complete, intervening),
            "could not record intervening own public action");
    complete.nply = 2;
    complete.turn = 0;
    Move second = { .card = CARD_MAKE(4, 8), .discard = 1, .draw = 3 };
    complete.pile_n[2] = 1;
    complete.pile[2][0] = CARD_MAKE(2, 9);
    complete.discarded |= UINT64_C(1) << CARD_MAKE(2, 9);
    require(history_belief_trace_push(&trace, &complete, second),
            "could not record second public opponent action");
    require(trace.n == 3, "complete public transcript was not retained");
    require(trace.event[1].opponent == 0 &&
            trace.event[2].draw_rank ==
                CARD_VALUE(CARD_MAKE(2, 9)) - 1,
            "public context or exact face-up draw was not retained");

    State view;
    complete.nply = 3;
    complete.turn = 1;
    agent_information_view(&complete, 1, &view);
    uint8_t candidates[] = {
        CARD_MAKE(2, 0), CARD_MAKE(2, 1), CARD_MAKE(2, 2),
        CARD_MAKE(2, 6), CARD_MAKE(3, 6)
    };
    HistoryBeliefModel model;
    history_belief_model_init(&model);
    Net *actor = malloc(sizeof *actor);
    Net *other_actor = malloc(sizeof *other_actor);
    require(actor && other_actor, "network allocation failed");
    net_init(actor, UINT64_C(202608290109));
    net_project_belief_wager_symmetry(actor);
    *other_actor = *actor;
    other_actor->bbel[0] += 0.25f;
    unsigned char exclusion_sha256[32];
    for (int i = 0; i < 32; i++) exclusion_sha256[i] = (unsigned char)(i + 1);
    require(history_belief_model_bind(&model, actor, actor, 120, 0.5f,
                                      HISTORY_BELIEF_BASE_ALPHA,
                                      exclusion_sha256),
            "could not bind model provenance");
    HistoryBeliefRuntime runtime;
    require(history_belief_runtime_init(
                &runtime, &model, actor, actor, 120, 0.5f,
                HISTORY_BELIEF_BASE_ALPHA, exclusion_sha256),
            "could not initialize validated inference runtime");

    float base[5] = { -0.75f, -0.75f, -0.75f, 0.5f, 1.25f };

    float zero[5];
    require(history_belief_logits(&model, &trace, &view, 1,
                                  candidates, base, 5, zero),
            "zero-model inference failed");
    for (int i = 0; i < 5; i++)
        require(zero[i] == base[i],
                "zero residual did not preserve current-state baseline");
    float saturated_base[5] = { 20.0f, 20.0f, -20.0f, -20.0f, 3.0f };
    float saturated_combined[5], saturated_expected[5], saturated_actual[5];
    require(history_belief_logits(&model, &trace, &view, 1,
                                  candidates, saturated_base, 5,
                                  saturated_combined) &&
            history_belief_exact_k_eval(saturated_base, NULL, 5, 2,
                                        saturated_expected, NULL) &&
            history_belief_exact_k_eval(saturated_combined, NULL, 5, 2,
                                        saturated_actual, NULL) &&
            !memcmp(saturated_expected, saturated_actual,
                    sizeof saturated_expected),
            "zero residual changed a saturated base distribution");

    HistoryBeliefDist combined;
    require(history_belief_dist_init(&runtime, &trace,
                                     &view, 1, &combined),
            "zero residual exact-K reconstruction failed");
    for (int i = 0; i < combined.base.n; i++)
        require(fabs(combined.marginal[i] - combined.base.marginal[i]) <
                    2e-6,
                "zero residual changed current-state exact-K posterior");

    HistoryBeliefTrace empty;
    history_belief_trace_init(&empty, 0);
    State opening_view;
    agent_information_view(&opening_complete, 0, &opening_view);
    float structural_uniform[5];
    require(history_belief_logits(&model, &empty, &opening_view, 0,
                                  candidates, base, 5, structural_uniform),
            "empty-history inference failed");
    for (int i = 0; i < 5; i++)
        require(structural_uniform[i] == 0.0f,
                "no-action state was not structurally uniform");
    State later_view0;
    agent_information_view(&complete, 0, &later_view0);
    require(!history_belief_logits(&model, &empty, &later_view0, 0,
                                   candidates, base, 5,
                                   structural_uniform),
            "empty later-ply trace bypassed chronology validation");

    HistoryBeliefTrace own_only;
    history_belief_trace_init(&own_only, 0);
    require(history_belief_trace_push(&own_only, &opening_complete, first),
            "could not construct own-only transcript");
    State after_own = opening_complete;
    after_own.nply = 1;
    after_own.turn = 1;
    State after_own_view;
    agent_information_view(&after_own, 0, &after_own_view);
    float own_only_score[5];
    require(history_belief_logits(&model, &own_only, &after_own_view, 0,
                                  candidates, base, 5, own_only_score),
            "own-only transcript inference failed");
    for (int i = 0; i < 5; i++)
        require(own_only_score[i] == 0.0f,
                "own-only transcript bypassed structural uniform prior");

    State alternate_hidden = opening_complete;
    alternate_hidden.deck[0] ^= 1;
    alternate_hidden.hand[1] ^= UINT64_C(3);
    HistoryBeliefTrace hidden_a, hidden_b;
    history_belief_trace_init(&hidden_a, 0);
    history_belief_trace_init(&hidden_b, 0);
    require(history_belief_trace_push(&hidden_a, &opening_complete, first) &&
            history_belief_trace_push(&hidden_b, &alternate_hidden, first) &&
            !memcmp(&hidden_a, &hidden_b, sizeof hidden_a),
            "hidden hand or deck bytes contaminated the public transcript");

    uint32_t original[HISTORY_BELIEF_MAX_ACTIVE];
    int no = history_belief_active_features(
        &trace, &view, 1, candidates[3], original);
    require(no == 7, "unexpected active feature count");

    HistoryBeliefTrace reversed = trace;
    HistoryBeliefEvent temporary = reversed.event[0];
    reversed.event[0] = reversed.event[2];
    reversed.event[2] = temporary;
    /* Keep the transcript causally ordered by public ply while swapping the
     * two opponent-action semantics. */
    reversed.event[0].ply = 0;
    reversed.event[0].opponent = 1;
    reversed.event[2].ply = 2;
    reversed.event[2].opponent = 1;
    uint32_t swapped[HISTORY_BELIEF_MAX_ACTIVE];
    int ns = history_belief_active_features(
        &reversed, &view, 1, candidates[3], swapped);
    require(ns == no, "swapped history feature count changed");
    uint32_t distinguishing = UINT32_MAX;
    for (int i = 0; i < no; i++) {
        if (feature_count(original, no, original[i]) !=
            feature_count(swapped, ns, original[i])) {
            distinguishing = original[i];
            break;
        }
    }
    require(distinguishing != UINT32_MAX,
            "action-order features did not distinguish transcript order");
    HistoryBeliefTrace malformed = trace;
    malformed.event[1].ply = 9;
    require(history_belief_active_features(
                &malformed, &view, 1, candidates[3], swapped) < 0,
            "malformed trace chronology was accepted");
    model.weight[distinguishing] = 1.0f;
    float ordered_score, reversed_score;
    require(history_belief_logits(&model, &trace, &view, 1,
                                  &candidates[3], &base[3], 1,
                                  &ordered_score) &&
            history_belief_logits(&model, &reversed, &view, 1,
                                  &candidates[3], &base[3], 1,
                                  &reversed_score) &&
            ordered_score != reversed_score,
            "model ignored action order");

    /* Physical wager IDs are unobservable and must have identical logits. */
    float wager_score[3];
    require(history_belief_logits(&model, &trace, &view, 1,
                                  candidates, base, 3, wager_score),
            "wager inference failed");
    require(wager_score[0] == wager_score[1] &&
            wager_score[1] == wager_score[2],
            "physical wagers received different scores");
    int wager_a = -1, wager_b = -1;
    for (int i = 0; i < combined.base.n && wager_b < 0; i++) {
        if (!CARD_IS_WAGER(combined.base.card[i])) continue;
        for (int j = i + 1; j < combined.base.n; j++)
            if (CARD_IS_WAGER(combined.base.card[j]) &&
                CARD_SUIT(combined.base.card[i]) ==
                    CARD_SUIT(combined.base.card[j])) {
                wager_a = i;
                wager_b = j;
                break;
            }
    }
    require(wager_a >= 0 && wager_b >= 0 &&
            combined.base.marginal[wager_a] ==
                combined.base.marginal[wager_b],
            "frozen baseline broke physical-wager symmetry");

    /* Exact suit relabelling must map scores without changing their bits. */
    uint8_t permutation[NSUIT] = { 2, 4, 1, 0, 3 };
    State permuted_view;
    lc_permute_suits(&view, &permuted_view, permutation);
    HistoryBeliefTrace permuted_trace;
    permute_trace(&trace, &permuted_trace, permutation);
    uint8_t permuted_candidates[5];
    for (int i = 0; i < 5; i++)
        permuted_candidates[i] = lc_permute_card(candidates[i], permutation);
    float before[5], after[5];
    require(history_belief_logits(&model, &trace, &view, 1,
                                  candidates, base, 5, before) &&
            history_belief_logits(&model, &permuted_trace, &permuted_view, 1,
                                  permuted_candidates, base, 5, after),
            "suit-equivariant inference failed");
    require(!memcmp(before, after, sizeof before),
            "suit relabelling changed model scores");
    HistoryBeliefDist real_before, real_after;
    require(history_belief_dist_init(&runtime, &trace, &view, 1,
                                     &real_before) &&
            history_belief_dist_init(&runtime, &permuted_trace,
                                     &permuted_view, 1, &real_after),
            "real baseline suit-equivariance inference failed");
    for (int i = 0; i < real_before.base.n; i++) {
        uint8_t mapped = lc_permute_card(real_before.base.card[i],
                                         permutation);
        int found = -1;
        for (int j = 0; j < real_after.base.n; j++)
            if (real_after.base.card[j] == mapped) {
                found = j;
                break;
            }
        require(found >= 0 &&
                fabs(real_before.marginal[i] - real_after.marginal[found]) <
                    2e-6,
                "real baseline plus residual broke suit equivariance");
    }

    /* A complete referee state must fail the inference boundary. */
    float rejected;
    require(!history_belief_logits(&model, &trace, &complete, 1,
                                   &candidates[3], &base[3], 1, &rejected),
            "complete hidden state crossed the information boundary");

    float logits[5], marginal[5];
    uint8_t held[5] = { 1, 1, 0, 0, 0 };
    double nll;
    require(history_belief_logits(&model, &trace, &view, 1,
                                  candidates, base, 5, logits) &&
            history_belief_exact_k_eval(logits, held, 5, 2,
                                        marginal, &nll),
            "exact-K evaluation failed");
    double sum = 0.0;
    for (int i = 0; i < 5; i++) sum += marginal[i];
    require(fabs(sum - 2.0) < 1e-6 && nll >= 0.0,
            "exact-K marginals violated cardinality");
    float extreme[5] = { 1000.0f, 900.0f, -1000.0f, -900.0f, 0.0f };
    require(history_belief_exact_k_eval(extreme, held, 5, 2,
                                        marginal, &nll) &&
            lc_double_isfinite(nll),
            "unclipped log-space exact-K evaluation was not stable");

    float clipped_raw[5] = { 50.0f, -45.0f, 1.5f, -0.5f, 0.25f };
    float clipped_marginal[5], clipped_gradient[5];
    require(belief_exact_k_eval(clipped_raw, held, 5, 2, 1.0f,
                                clipped_marginal, &nll) &&
            history_belief_clipped_raw_gradient(
                clipped_raw, clipped_marginal, held, 5, 1.0f,
                clipped_gradient),
            "clipped exact-K gradient failed");
    double gradient_sum = 0.0;
    for (int i = 0; i < 5; i++) {
        float plus[5], minus[5], scratch[5];
        memcpy(plus, clipped_raw, sizeof plus);
        memcpy(minus, clipped_raw, sizeof minus);
        plus[i] += 1e-3f;
        minus[i] -= 1e-3f;
        double plus_nll = NAN, minus_nll = NAN;
        int plus_ok = belief_exact_k_eval(plus, held, 5, 2, 1.0f,
                                          scratch, &plus_nll);
        int minus_ok = belief_exact_k_eval(minus, held, 5, 2, 1.0f,
                                           scratch, &minus_nll);
        require(plus_ok && minus_ok && lc_double_isfinite(plus_nll) &&
                lc_double_isfinite(minus_nll),
                "finite-difference exact-K evaluation failed");
        double finite_difference = (plus_nll - minus_nll) / 0.002;
        require(fabs(finite_difference - clipped_gradient[i]) < 2e-3,
                "clipped exact-K gradient disagreed with finite difference");
        gradient_sum += clipped_gradient[i];
    }
    require(fabs(gradient_sum) < 1e-6,
            "centered clipped gradient did not sum to zero");

    char path[128];
    snprintf(path, sizeof path, "/tmp/lc-history-belief-%ld.bin",
             (long)getpid());
    model.train_seed = UINT64_C(202608290112);
    model.train_states = 1234;
    require(history_belief_model_save(&model, path) == 0,
            "model save failed");
    HistoryBeliefModel loaded;
    require(history_belief_model_load(&loaded, path) == 0,
            "model load failed");
    require(history_belief_model_fingerprint(&loaded) ==
            history_belief_model_fingerprint(&model) &&
            loaded.train_seed == model.train_seed &&
            loaded.train_states == model.train_states &&
            history_belief_model_compatible(&loaded, actor, actor,
                                            120, 0.5f,
                                            HISTORY_BELIEF_BASE_ALPHA,
                                            exclusion_sha256),
            "model artifact did not round trip");
    uint64_t frozen_fingerprint = history_belief_model_fingerprint(&loaded);
    loaded.train_states++;
    require(history_belief_model_fingerprint(&loaded) != frozen_fingerprint,
            "training provenance was omitted from model fingerprint");
    loaded.train_states--;
    unsigned char wrong_exclusion_sha256[32];
    memcpy(wrong_exclusion_sha256, exclusion_sha256, 32);
    wrong_exclusion_sha256[0] ^= 1;
    require(!history_belief_model_compatible(&loaded, other_actor, actor,
                                             120, 0.5f,
                                             HISTORY_BELIEF_BASE_ALPHA,
                                             exclusion_sha256) &&
            !history_belief_model_compatible(&loaded, actor, other_actor,
                                             120, 0.5f,
                                             HISTORY_BELIEF_BASE_ALPHA,
                                             exclusion_sha256) &&
            !history_belief_model_compatible(&loaded, actor, actor,
                                             10, 0.5f,
                                             HISTORY_BELIEF_BASE_ALPHA,
                                             exclusion_sha256) &&
            !history_belief_model_compatible(&loaded, actor, actor,
                                             120, 0.6f,
                                             HISTORY_BELIEF_BASE_ALPHA,
                                             exclusion_sha256) &&
            !history_belief_model_compatible(&loaded, actor, actor,
                                             120, 0.5f, 1.0f,
                                             exclusion_sha256) &&
            !history_belief_model_compatible(&loaded, actor, actor,
                                             120, 0.5f,
                                             HISTORY_BELIEF_BASE_ALPHA,
                                             wrong_exclusion_sha256),
            "model accepted mismatched provenance");

    HistoryBeliefModel deterministic;
    history_belief_model_init(&deterministic);
    require(history_belief_model_bind(&deterministic, actor, actor,
                                      120, 0.0f, 1.0f,
                                      exclusion_sha256) &&
            history_belief_model_compatible(&deterministic, actor, actor,
                                            120, 0.0f, 1.0f,
                                            exclusion_sha256) &&
            !history_belief_model_compatible(&deterministic, actor, actor,
                                             120, 0.0001f, 1.0f,
                                             exclusion_sha256),
            "deterministic argmax temperature was not bound exactly");

    char truncated_path[144];
    snprintf(truncated_path, sizeof truncated_path,
             "/tmp/lc-history-belief-%ld-truncated.bin", (long)getpid());
    FILE *source = fopen(path, "rb");
    FILE *truncated = fopen(truncated_path, "wb");
    require(source && truncated, "could not create truncated artifact");
    require(fseek(source, 0, SEEK_END) == 0,
            "could not size saved artifact");
    long artifact_size = ftell(source);
    require(artifact_size > 1 && fseek(source, 0, SEEK_SET) == 0,
            "invalid saved artifact size");
    for (long i = 0; i < artifact_size - 1; i++) {
        int byte = fgetc(source);
        require(byte != EOF && fputc(byte, truncated) != EOF,
                "could not write truncated artifact");
    }
    require(fclose(source) == 0 && fclose(truncated) == 0,
            "could not seal truncated artifact");
    require(history_belief_model_load(&loaded, truncated_path) != 0,
            "truncated artifact was accepted");
    unlink(truncated_path);

    char corrupt_path[144];
    snprintf(corrupt_path, sizeof corrupt_path,
             "/tmp/lc-history-belief-%ld-corrupt.bin", (long)getpid());
    source = fopen(path, "rb");
    FILE *corrupt = fopen(corrupt_path, "wb");
    require(source && corrupt, "could not create corrupted artifact");
    for (long i = 0; i < artifact_size; i++) {
        int byte = fgetc(source);
        require(byte != EOF, "could not read artifact for corruption test");
        if (i == 80) byte ^= 1;
        require(fputc(byte, corrupt) != EOF,
                "could not write corrupted artifact");
    }
    require(fclose(source) == 0 && fclose(corrupt) == 0,
            "could not seal corrupted artifact");
    require(history_belief_model_load(&loaded, corrupt_path) != 0,
            "corrupted exclusion provenance was accepted");
    unlink(corrupt_path);
    unlink(path);
    free(actor);
    free(other_actor);

    puts("history belief model tests passed");
    return 0;
}
