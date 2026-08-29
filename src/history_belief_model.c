#include "history_belief_model.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const unsigned char HISTORY_BELIEF_MAGIC[8] = {
    'L', 'C', 'B', 'H', 'M', '1', '\0', '\0'
};

enum {
    HISTORY_BELIEF_HEADER_BYTES = 112
};

static const uint64_t HISTORY_BELIEF_SCHEMA =
    UINT64_C(0x7d5f6e1aeb5d2101);

static uint64_t fnv_update(uint64_t hash, const unsigned char *p, size_t n)
{
    for (size_t i = 0; i < n; i++) {
        hash ^= p[i];
        hash *= UINT64_C(1099511628211);
    }
    return hash;
}

static void encode_u32(unsigned char out[4], uint32_t x)
{
    out[0] = (unsigned char)x;
    out[1] = (unsigned char)(x >> 8);
    out[2] = (unsigned char)(x >> 16);
    out[3] = (unsigned char)(x >> 24);
}

static void encode_u64(unsigned char out[8], uint64_t x)
{
    for (int i = 0; i < 8; i++) out[i] = (unsigned char)(x >> (8 * i));
}

static uint32_t decode_u32(const unsigned char in[4])
{
    return (uint32_t)in[0] | (uint32_t)in[1] << 8 |
           (uint32_t)in[2] << 16 | (uint32_t)in[3] << 24;
}

static uint64_t decode_u64(const unsigned char in[8])
{
    uint64_t x = 0;
    for (int i = 0; i < 8; i++) x |= (uint64_t)in[i] << (8 * i);
    return x;
}

static uint32_t float_bits(float x)
{
    uint32_t bits;
    memcpy(&bits, &x, sizeof bits);
    return bits;
}

static float bits_float(uint32_t bits)
{
    float x;
    memcpy(&x, &bits, sizeof x);
    return x;
}

static int semantic_rank(int card)
{
    return CARD_IS_WAGER(card) ? 0 : CARD_VALUE(card) - 1;
}

static int phase_bin(int ply)
{
    if (ply < 4) return 0;
    if (ply < 8) return 1;
    if (ply < 14) return 2;
    if (ply < 22) return 3;
    if (ply < 34) return 4;
    return 5;
}

static int deck_phase_bin(int cards)
{
    if (cards > 36) return 0;
    if (cards > 28) return 1;
    if (cards > 20) return 2;
    if (cards > 12) return 3;
    if (cards > 4) return 4;
    return 5;
}

void history_belief_model_init(HistoryBeliefModel *model)
{
    if (!model) return;
    memset(model, 0, sizeof *model);
    model->version = HISTORY_BELIEF_VERSION;
    model->base_alpha = HISTORY_BELIEF_BASE_ALPHA;
}

uint64_t history_belief_actor_fingerprint(const Net *actor)
{
    if (!actor) return 0;
    return fnv_update(UINT64_C(1469598103934665603),
                      (const unsigned char *)actor, sizeof *actor);
}

int history_belief_model_bind(HistoryBeliefModel *model, const Net *actor,
                              const Net *base_net,
                              int training_symmetries,
                              float training_temperature,
                              float base_alpha,
                              const unsigned char exclusion_sha256[32])
{
    if (!model || model->version != HISTORY_BELIEF_VERSION || !actor ||
        !base_net || !exclusion_sha256 ||
        (training_symmetries != 1 && training_symmetries != 5 &&
         training_symmetries != 10 && training_symmetries != 20 &&
         training_symmetries != 120) ||
        !lc_float_isfinite(training_temperature) ||
        training_temperature < 0.0f || training_temperature > 5.0f ||
        !lc_float_isfinite(base_alpha) || base_alpha < 0.0f ||
        base_alpha > 5.0f)
        return 0;
    int exclusion_nonzero = 0;
    for (int i = 0; i < 32; i++) exclusion_nonzero |= exclusion_sha256[i];
    if (!exclusion_nonzero) return 0;
    model->actor_fingerprint = history_belief_actor_fingerprint(actor);
    model->base_net_fingerprint =
        history_belief_actor_fingerprint(base_net);
    model->training_symmetries = (uint32_t)training_symmetries;
    model->training_temperature = training_temperature;
    model->base_alpha = base_alpha;
    memcpy(model->exclusion_sha256, exclusion_sha256, 32);
    return model->actor_fingerprint != 0 &&
           model->base_net_fingerprint != 0;
}

int history_belief_model_validate(const HistoryBeliefModel *model)
{
    if (!model || model->version != HISTORY_BELIEF_VERSION ||
        model->actor_fingerprint == 0 || model->base_net_fingerprint == 0 ||
        (model->training_symmetries != 1 &&
         model->training_symmetries != 5 &&
         model->training_symmetries != 10 &&
         model->training_symmetries != 20 &&
         model->training_symmetries != 120) ||
        !lc_float_isfinite(model->training_temperature) ||
        model->training_temperature < 0.0f ||
        model->training_temperature > 5.0f ||
        !lc_float_isfinite(model->base_alpha) || model->base_alpha < 0.0f ||
        model->base_alpha > 5.0f)
        return 0;
    int exclusion_nonzero = 0;
    for (int i = 0; i < 32; i++)
        exclusion_nonzero |= model->exclusion_sha256[i];
    if (!exclusion_nonzero) return 0;
    for (int i = 0; i < HISTORY_BELIEF_FEATURES; i++)
        if (!lc_float_isfinite(model->weight[i]) ||
            fabsf(model->weight[i]) > 100.0f)
            return 0;
    return 1;
}

int history_belief_model_compatible(const HistoryBeliefModel *model,
                                    const Net *actor,
                                    const Net *base_net,
                                    int training_symmetries,
                                    float training_temperature,
                                    float base_alpha,
                                    const unsigned char exclusion_sha256[32])
{
    return history_belief_model_validate(model) && actor && base_net &&
           exclusion_sha256 &&
           model->actor_fingerprint ==
               history_belief_actor_fingerprint(actor) &&
           model->base_net_fingerprint ==
               history_belief_actor_fingerprint(base_net) &&
           model->training_symmetries == (uint32_t)training_symmetries &&
           float_bits(model->training_temperature) ==
               float_bits(training_temperature) &&
           float_bits(model->base_alpha) == float_bits(base_alpha) &&
           memcmp(model->exclusion_sha256, exclusion_sha256, 32) == 0;
}

void history_belief_trace_init(HistoryBeliefTrace *trace, int observer)
{
    if (!trace) return;
    memset(trace, 0, sizeof *trace);
    trace->observer = (uint8_t)observer;
}

int history_belief_trace_push(HistoryBeliefTrace *trace,
                              const State *state_before, Move move)
{
    if (!trace || !state_before || trace->observer > 1 ||
        state_before->turn > 1 || move.card >= NCARD || move.discard > 1 ||
        move.draw > NSUIT || state_before->nply >= LC_MAX_PLIES ||
        trace->n != state_before->nply)
        return 0;
    if (trace->n >= HISTORY_BELIEF_MAX_EVENTS) return 0;
    uint8_t draw_rank = HISTORY_BELIEF_NO_DRAW_RANK;
    if (move.draw > 0) {
        int suit = move.draw - 1;
        if (state_before->pile_n[suit] == 0) return 0;
        int drawn = state_before->pile[suit][state_before->pile_n[suit] - 1];
        if (drawn >= NCARD || CARD_SUIT(drawn) != suit) return 0;
        draw_rank = (uint8_t)semantic_rank(drawn);
    }
    HistoryBeliefEvent *event = &trace->event[trace->n];
    event->ply = state_before->nply;
    event->suit = (uint8_t)CARD_SUIT(move.card);
    event->rank = (uint8_t)semantic_rank(move.card);
    event->discard = move.discard;
    event->draw = move.draw;
    event->draw_rank = draw_rank;
    event->opponent = (uint8_t)(state_before->turn != trace->observer);
    event->deck_left = state_before->deck_left;
    trace->n++;
    return 1;
}

int history_belief_trace_opponent_actions(const HistoryBeliefTrace *trace)
{
    if (!trace || trace->n > HISTORY_BELIEF_MAX_EVENTS) return -1;
    int count = 0;
    for (int i = 0; i < trace->n; i++) {
        if (trace->event[i].opponent > 1) return -1;
        count += trace->event[i].opponent;
    }
    return count;
}

static int information_view_valid(const State *view, int observer)
{
    if (!view || observer < 0 || observer > 1 || view->turn > 1 ||
        view->hand[observer ^ 1] != view->known[observer ^ 1] ||
        view->deck_pos != 0)
        return 0;
    /* agent_information_view() zeros the deck.  Exact suit relabelling maps
     * those placeholder bytes as if they were physical wager IDs, so their
     * post-permutation values are deliberately unspecified.  deck_pos==0 is
     * the durable boundary proof; this model never reads deck[]. */
    return 1;
}

int history_belief_active_features(const HistoryBeliefTrace *trace,
                                   const State *view, int observer,
                                   uint8_t candidate,
                                   uint32_t out[HISTORY_BELIEF_MAX_ACTIVE])
{
    if (!trace || !out || trace->observer != observer ||
        trace->n > HISTORY_BELIEF_MAX_EVENTS || candidate >= NCARD ||
        !information_view_valid(view, observer) || trace->n != view->nply)
        return -1;
    int initial_turn = view->turn ^ (trace->n & 1);
    for (int i = 0; i < trace->n; i++) {
        int expected_opponent = ((initial_turn ^ (i & 1)) != observer);
        if (trace->event[i].ply != i ||
            trace->event[i].suit >= NSUIT ||
            trace->event[i].rank >= HISTORY_BELIEF_RANKS ||
            trace->event[i].discard > 1 || trace->event[i].draw > NSUIT ||
            trace->event[i].draw_rank >= HISTORY_BELIEF_DRAW_RANKS ||
            trace->event[i].opponent != expected_opponent ||
            trace->event[i].deck_left > NCARD ||
            (trace->event[i].draw == 0 &&
             trace->event[i].draw_rank != HISTORY_BELIEF_NO_DRAW_RANK) ||
            (trace->event[i].draw > 0 &&
             trace->event[i].draw_rank == HISTORY_BELIEF_NO_DRAW_RANK))
            return -1;
    }
    if (trace->n == 0) return 0;

    const int candidate_suit = CARD_SUIT(candidate);
    const int candidate_rank = semantic_rank(candidate);
    int nout = 0;

    int age = 0;
    for (int i = (int)trace->n - 1; i >= 0; i--, age++) {
        const HistoryBeliefEvent *event = &trace->event[i];
        if (!event->opponent) continue;
        int age_bin = age < HISTORY_BELIEF_AGES - 1
                    ? age : HISTORY_BELIEF_AGES - 1;
        int same = event->suit == candidate_suit;
        int draw_class = event->draw == 0 ? 0
                       : event->draw - 1 == candidate_suit ? 1 : 2;

        size_t pair = (size_t)age_bin;
        pair = pair * 2 + event->discard;
        pair = pair * 3 + draw_class;
        pair = pair * 2 + same;
        pair = pair * HISTORY_BELIEF_RANKS + candidate_rank;
        pair = pair * HISTORY_BELIEF_RANKS + event->rank;
        out[nout++] = (uint32_t)pair;

        size_t draw = (size_t)age_bin;
        draw = draw * 3 + draw_class;
        draw = draw * 2 + same;
        draw = draw * HISTORY_BELIEF_RANKS + candidate_rank;
        draw = draw * HISTORY_BELIEF_DRAW_RANKS + event->draw_rank;
        out[nout++] = (uint32_t)(HISTORY_BELIEF_PAIR_FEATURES + draw);

        size_t event_phase = (size_t)phase_bin(event->ply);
        event_phase = event_phase * HISTORY_BELIEF_PHASES +
                      deck_phase_bin(event->deck_left);
        event_phase = event_phase * 2 + event->discard;
        event_phase = event_phase * 2 + same;
        event_phase = event_phase * HISTORY_BELIEF_RANKS + candidate_rank;
        event_phase = event_phase * HISTORY_BELIEF_RANKS + event->rank;
        out[nout++] = (uint32_t)(HISTORY_BELIEF_PAIR_FEATURES +
                                 HISTORY_BELIEF_DRAW_FEATURES + event_phase);

        if (i > 0) {
            const HistoryBeliefEvent *previous = &trace->event[i - 1];
            int previous_kind = previous->discard * 2 +
                                (previous->draw > 0);
            int previous_same_event = previous->suit == event->suit;
            int previous_same_candidate =
                previous->suit == candidate_suit;
            size_t context = (size_t)age_bin;
            context = context * 4 + previous_kind;
            context = context * 2 + previous_same_event;
            context = context * 2 + previous_same_candidate;
            context = context * HISTORY_BELIEF_RANKS + candidate_rank;
            context = context * HISTORY_BELIEF_RANKS + event->rank;
            context = context * HISTORY_BELIEF_RANKS + previous->rank;
            out[nout++] = (uint32_t)(HISTORY_BELIEF_PAIR_FEATURES +
                HISTORY_BELIEF_DRAW_FEATURES +
                HISTORY_BELIEF_EVENT_PHASE_FEATURES + context);
        }
    }
    if (nout > HISTORY_BELIEF_MAX_ACTIVE) return -1;
    for (int i = 0; i < nout; i++)
        if (out[i] >= HISTORY_BELIEF_FEATURES) return -1;
    return nout;
}

int history_belief_logits(const HistoryBeliefModel *model,
                          const HistoryBeliefTrace *trace,
                          const State *view, int observer,
                          const uint8_t *card, const float *base_log_weight,
                          int n, float *logit)
{
    /* Do not rescan all 22k model parameters for every state in a
     * many-million-state campaign.  Load/save and campaign preflight perform
     * the exhaustive validation; each actually consumed score is checked
     * below before it can enter exact-K arithmetic. */
    if (!model || model->version != HISTORY_BELIEF_VERSION || !trace ||
        !card || !logit || n < 0 || n > NCARD ||
        (trace->n > 0 && !base_log_weight))
        return 0;
    uint32_t active[HISTORY_BELIEF_MAX_ACTIVE];
    for (int i = 0; i < n; i++) {
        int nf = history_belief_active_features(trace, view, observer,
                                                card[i], active);
        if (nf < 0) return 0;
        /* No opponent action means no behavioral evidence.  Preserve the
         * exact uniform card-count prior structurally.  Once there is action
         * evidence, learn only a residual over the frozen current-state
         * head's alpha-scaled exact-K log weight. */
        double score = nf > 0 ? base_log_weight[i] : 0.0;
        if (!lc_double_isfinite(score)) return 0;
        for (int j = 0; j < nf; j++) score += model->weight[active[j]];
        if (!lc_double_isfinite(score)) return 0;
        logit[i] = (float)score;
        if (!lc_float_isfinite(logit[i])) return 0;
    }
    return 1;
}

static double log_add(double a, double b)
{
    if (a == -INFINITY) return b;
    if (b == -INFINITY) return a;
    double high = a > b ? a : b;
    double low = a > b ? b : a;
    return high + log1p(exp(low - high));
}

int history_belief_exact_k_eval(const float *log_weight,
                                const uint8_t *held,
                                int n, int need,
                                float *marginal, double *nll)
{
    if (!log_weight || !marginal || (nll && !held) || n < 0 || n > NCARD ||
        need < 0 || need > HAND_SIZE || need > n)
        return 0;
    for (int i = 0; i < n; i++)
        if (!lc_float_isfinite(log_weight[i])) return 0;

    double prefix[NCARD + 1][HAND_SIZE + 1];
    double suffix[NCARD + 1][HAND_SIZE + 1];
    for (int i = 0; i <= n; i++)
        for (int k = 0; k <= HAND_SIZE; k++)
            prefix[i][k] = suffix[i][k] = -INFINITY;
    prefix[0][0] = 0.0;
    for (int i = 0; i < n; i++) {
        prefix[i + 1][0] = 0.0;
        for (int k = 1; k <= need; k++)
            prefix[i + 1][k] = log_add(
                prefix[i][k],
                prefix[i][k - 1] + (double)log_weight[i]);
    }
    suffix[n][0] = 0.0;
    for (int i = n - 1; i >= 0; i--) {
        suffix[i][0] = 0.0;
        for (int k = 1; k <= need; k++)
            suffix[i][k] = log_add(
                suffix[i + 1][k],
                suffix[i + 1][k - 1] + (double)log_weight[i]);
    }
    double log_z = suffix[0][need];
    if (!lc_double_isfinite(log_z)) return 0;
    for (int i = 0; i < n; i++) {
        double excluded = -INFINITY;
        for (int left = 0; left < need; left++)
            excluded = log_add(excluded,
                prefix[i][left] + suffix[i + 1][need - 1 - left]);
        double value = exp((double)log_weight[i] + excluded - log_z);
        if (!lc_double_isfinite(value) || value < 0.0 || value > 1.000001)
            return 0;
        marginal[i] = (float)value;
    }
    if (nll) {
        int count = 0;
        double selected = 0.0;
        for (int i = 0; i < n; i++) {
            if (held[i] > 1) return 0;
            if (held[i]) {
                count++;
                selected += log_weight[i];
            }
        }
        if (count != need) return 0;
        *nll = log_z - selected;
        if (!lc_double_isfinite(*nll)) return 0;
    }
    return 1;
}

int history_belief_clipped_raw_gradient(const float *raw_logit,
                                        const float *marginal,
                                        const uint8_t *held,
                                        int n, float alpha,
                                        float *gradient)
{
    if (!raw_logit || !marginal || !held || !gradient || n <= 0 ||
        n > NCARD || !lc_float_isfinite(alpha) || alpha < 0.0f)
        return 0;
    double mean = 0.0;
    for (int i = 0; i < n; i++) {
        if (!lc_float_isfinite(raw_logit[i]) ||
            !lc_float_isfinite(marginal[i]) || marginal[i] < 0.0f ||
            marginal[i] > 1.0f || held[i] > 1)
            return 0;
        mean += raw_logit[i];
    }
    mean /= (double)n;
    double active_sum = 0.0;
    for (int i = 0; i < n; i++) {
        double centered = (double)alpha *
                          ((double)raw_logit[i] - mean);
        if (centered > -20.0 && centered < 20.0)
            active_sum += (double)marginal[i] - held[i];
    }
    for (int i = 0; i < n; i++) {
        double centered = (double)alpha *
                          ((double)raw_logit[i] - mean);
        double own = (centered > -20.0 && centered < 20.0)
                   ? (double)marginal[i] - held[i] : 0.0;
        gradient[i] = (float)((double)alpha *
                              (own - active_sum / (double)n));
        if (!lc_float_isfinite(gradient[i])) return 0;
    }
    return 1;
}

int history_belief_runtime_init(
    HistoryBeliefRuntime *runtime,
    const HistoryBeliefModel *model,
    const Net *actor, const Net *base_net,
    int symmetries, float temperature, float base_alpha,
    const unsigned char exclusion_sha256[32])
{
    if (!runtime || !history_belief_model_compatible(
            model, actor, base_net, symmetries, temperature, base_alpha,
            exclusion_sha256))
        return 0;
    runtime->model = model;
    runtime->actor = actor;
    runtime->base_net = base_net;
    runtime->actor_fingerprint = model->actor_fingerprint;
    runtime->base_net_fingerprint = model->base_net_fingerprint;
    runtime->symmetries = (uint32_t)symmetries;
    runtime->temperature = temperature;
    runtime->base_alpha = base_alpha;
    return 1;
}

int history_belief_dist_init(const HistoryBeliefRuntime *runtime,
                             const HistoryBeliefTrace *trace,
                             const State *view, int observer,
                             HistoryBeliefDist *dist)
{
    /* Full parameter validation is deliberately a load/preflight operation.
     * Re-scanning hundreds of thousands of weights at every state would make
     * a large campaign intractable.  history_belief_logits() checks every
     * active weight and result before consumption. */
    if (!runtime || !runtime->model || !runtime->actor ||
        !runtime->base_net || !trace || !view || !dist)
        return 0;
    const HistoryBeliefModel *model = runtime->model;
    if (
        model->version != HISTORY_BELIEF_VERSION ||
        model->actor_fingerprint != runtime->actor_fingerprint ||
        model->base_net_fingerprint != runtime->base_net_fingerprint ||
        model->training_symmetries != runtime->symmetries ||
        float_bits(model->training_temperature) !=
            float_bits(runtime->temperature) ||
        float_bits(model->base_alpha) != float_bits(runtime->base_alpha))
        return 0;
    memset(dist, 0, sizeof *dist);
    if (!belief_dist_init(runtime->base_net, view, observer,
                          (int)runtime->symmetries,
                          model->base_alpha, &dist->base))
        return 0;
    float base_log_weight[NCARD];
    for (int i = 0; i < dist->base.n; i++) {
        if (!(dist->base.weight[i] > 0.0) ||
            !lc_double_isfinite(dist->base.weight[i]))
            return 0;
        base_log_weight[i] = (float)log(dist->base.weight[i]);
    }
    return history_belief_logits(model, trace, view, observer,
                                 dist->base.card, base_log_weight,
                                 dist->base.n, dist->logit) &&
           history_belief_exact_k_eval(dist->logit, NULL,
                                       dist->base.n, dist->base.need,
                                       dist->marginal, NULL);
}

uint64_t history_belief_model_fingerprint(const HistoryBeliefModel *model)
{
    if (!history_belief_model_validate(model)) return 0;
    uint64_t hash = UINT64_C(1469598103934665603);
    unsigned char bytes[8];
    encode_u64(bytes, model->train_seed);
    hash = fnv_update(hash, bytes, 8);
    encode_u64(bytes, model->train_states);
    hash = fnv_update(hash, bytes, 8);
    encode_u64(bytes, model->actor_fingerprint);
    hash = fnv_update(hash, bytes, 8);
    encode_u64(bytes, model->base_net_fingerprint);
    hash = fnv_update(hash, bytes, 8);
    encode_u32(bytes, model->training_symmetries);
    hash = fnv_update(hash, bytes, 4);
    encode_u32(bytes, float_bits(model->training_temperature));
    hash = fnv_update(hash, bytes, 4);
    encode_u32(bytes, float_bits(model->base_alpha));
    hash = fnv_update(hash, bytes, 4);
    hash = fnv_update(hash, model->exclusion_sha256, 32);
    for (int i = 0; i < HISTORY_BELIEF_FEATURES; i++) {
        encode_u32(bytes, float_bits(model->weight[i]));
        hash = fnv_update(hash, bytes, 4);
    }
    return hash;
}

int history_belief_model_save(const HistoryBeliefModel *model,
                              const char *path)
{
    if (!path || !*path || !history_belief_model_validate(model)) return -1;
    FILE *fp = fopen(path, "wb");
    if (!fp) return -2;
    unsigned char header[HISTORY_BELIEF_HEADER_BYTES] = { 0 };
    memcpy(header, HISTORY_BELIEF_MAGIC, sizeof HISTORY_BELIEF_MAGIC);
    encode_u32(header + 8, HISTORY_BELIEF_VERSION);
    encode_u32(header + 12, HISTORY_BELIEF_HEADER_BYTES);
    encode_u32(header + 16, HISTORY_BELIEF_FEATURES);
    encode_u32(header + 20, model->training_symmetries);
    encode_u64(header + 24, model->train_seed);
    encode_u64(header + 32, model->train_states);
    encode_u64(header + 40, history_belief_model_fingerprint(model));
    encode_u64(header + 48, HISTORY_BELIEF_SCHEMA);
    encode_u64(header + 56, model->actor_fingerprint);
    encode_u32(header + 64, float_bits(model->training_temperature));
    encode_u32(header + 68, float_bits(model->base_alpha));
    encode_u64(header + 72, model->base_net_fingerprint);
    memcpy(header + 80, model->exclusion_sha256, 32);
    int ok = fwrite(header, 1, sizeof header, fp) == sizeof header;
    unsigned char bytes[4];
    for (int i = 0; ok && i < HISTORY_BELIEF_FEATURES; i++) {
        encode_u32(bytes, float_bits(model->weight[i]));
        ok = fwrite(bytes, 1, sizeof bytes, fp) == sizeof bytes;
    }
    if (fclose(fp) != 0) ok = 0;
    return ok ? 0 : -3;
}

int history_belief_model_load(HistoryBeliefModel *model,
                              const char *path)
{
    if (!model || !path || !*path) return -1;
    FILE *fp = fopen(path, "rb");
    if (!fp) return -2;
    unsigned char header[HISTORY_BELIEF_HEADER_BYTES];
    int ok = fread(header, 1, sizeof header, fp) == sizeof header &&
             memcmp(header, HISTORY_BELIEF_MAGIC,
                    sizeof HISTORY_BELIEF_MAGIC) == 0 &&
             decode_u32(header + 8) == HISTORY_BELIEF_VERSION &&
             decode_u32(header + 12) == HISTORY_BELIEF_HEADER_BYTES &&
             decode_u32(header + 16) == HISTORY_BELIEF_FEATURES &&
             decode_u64(header + 48) == HISTORY_BELIEF_SCHEMA;
    HistoryBeliefModel tmp;
    history_belief_model_init(&tmp);
    if (ok) {
        tmp.training_symmetries = decode_u32(header + 20);
        tmp.train_seed = decode_u64(header + 24);
        tmp.train_states = decode_u64(header + 32);
        tmp.actor_fingerprint = decode_u64(header + 56);
        tmp.training_temperature = bits_float(decode_u32(header + 64));
        tmp.base_alpha = bits_float(decode_u32(header + 68));
        tmp.base_net_fingerprint = decode_u64(header + 72);
        memcpy(tmp.exclusion_sha256, header + 80, 32);
    }
    unsigned char bytes[4];
    for (int i = 0; ok && i < HISTORY_BELIEF_FEATURES; i++) {
        ok = fread(bytes, 1, sizeof bytes, fp) == sizeof bytes;
        if (ok) tmp.weight[i] = bits_float(decode_u32(bytes));
    }
    if (ok) ok = fgetc(fp) == EOF;
    if (fclose(fp) != 0) ok = 0;
    if (!ok || !history_belief_model_validate(&tmp) ||
        decode_u64(header + 40) != history_belief_model_fingerprint(&tmp))
        return -3;
    *model = tmp;
    return 0;
}
