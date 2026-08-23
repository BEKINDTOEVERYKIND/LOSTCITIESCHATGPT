/* build_match_value -- deterministic Monte Carlo Bellman table builder.
 *
 * This builds a raw policy-evaluation table and, separately, an explicitly
 * flagged isotonic regularization of that table; neither is an efficacy test.
 * Deals and player-role mappings are counter-keyed by (seed, round, sample),
 * never worker number, so changing --threads cannot change a byte of the
 * resulting artifact.  The same deal/roles are reused across all 301
 * policy-visible score leads, which is a common-random-numbers variance
 * reduction for adjacent table entries.
 */
#include "../src/agent.h"
#include "../src/match_value.h"
#include <errno.h>
#include <inttypes.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define POLICY_LEADS (2 * MATCH_VALUE_POLICY_LEAD_LIMIT + 1)
#define ROUND_MARGINS (2 * MATCH_VALUE_ROUND_MARGIN_LIMIT + 1)

typedef struct {
    const Net *net;
    const MatchValueController *controller;
    const uint8_t *decks;
    const uint8_t *roles;
    uint32_t *histogram;
    uint32_t samples;
    int thread;
    int nthread;
    atomic_int *failed;
} BuildJob;

static uint64_t mix64(uint64_t x)
{
    x ^= x >> 30;
    x *= UINT64_C(0xBF58476D1CE4E5B9);
    x ^= x >> 27;
    x *= UINT64_C(0x94D049BB133111EB);
    return x ^ (x >> 31);
}

static size_t histogram_index(int round_slot, int lead_slot, int margin)
{
    return ((size_t)round_slot * POLICY_LEADS + (size_t)lead_slot) *
               ROUND_MARGINS +
           (size_t)(margin + MATCH_VALUE_ROUND_MARGIN_LIMIT);
}

static void deck_for_sample(uint64_t seed, int round, uint32_t sample,
                            uint8_t deck[NCARD])
{
    Rng rng;
    uint64_t key = seed ^ (UINT64_C(0x9E3779B97F4A7C15) *
                           (uint64_t)(round + 1)) ^
                   (UINT64_C(0xD1B54A32D192ED03) *
                    (uint64_t)(sample + 1));
    rng_seed(&rng, mix64(key));
    for (int i = 0; i < NCARD; i++) deck[i] = (uint8_t)i;
    for (int i = NCARD - 1; i > 0; i--) {
        uint32_t j = rng_below(&rng, (uint32_t)i + 1);
        uint8_t temporary = deck[i];
        deck[i] = deck[j];
        deck[j] = temporary;
    }
}

static int prepare_corpus(uint64_t seed, uint32_t samples, int symmetries,
                          uint8_t *decks, uint8_t *roles)
{
    uint8_t group[120][NSUIT];
    int n = suit_permutations(symmetries, group);
    if (n != symmetries) return 0;
    for (int round_slot = 0; round_slot < 2; round_slot++) {
        int round = round_slot + 1;
        uint64_t offset_key = mix64(
            seed ^ UINT64_C(0xA0761D6478BD642F) ^
            ((uint64_t)(round + 1) * UINT64_C(0xE7037ED1A0B428DB)));
        int first_offset = (int)(offset_key % (uint64_t)n);
        int other_offset = (int)(rotl64(offset_key, 31) % (uint64_t)n);
        for (uint32_t sample = 0; sample < samples; sample++) {
            size_t corpus_index =
                (size_t)round_slot * samples + sample;
            deck_for_sample(seed, round, sample,
                            decks + corpus_index * NCARD);
            int row = (int)((uint64_t)sample % (uint64_t)n);
            int block = (int)(((uint64_t)sample / (uint64_t)n) %
                              (uint64_t)n);
            int index[2] = {
                (first_offset + row) % n,
                (other_offset + block + row) % n
            };
            for (int p = 0; p < 2; p++)
                memcpy(roles + (corpus_index * 2U + (size_t)p) * NSUIT,
                       group[index[p]], NSUIT);
        }
    }
    return 1;
}

static void *build_worker(void *opaque)
{
    BuildJob *job = (BuildJob *)opaque;
    NetEvalPlan eval_plan;
    net_eval_plan_init(job->net, &eval_plan);
    for (int lead_slot = job->thread; lead_slot < POLICY_LEADS;
         lead_slot += job->nthread) {
        if (atomic_load_explicit(job->failed, memory_order_relaxed)) break;
        int lead = lead_slot - MATCH_VALUE_POLICY_LEAD_LIMIT;
        for (int round_slot = 0; round_slot < 2; round_slot++) {
            int round = round_slot + 1;
            for (uint32_t sample = 0; sample < job->samples; sample++) {
                State state;
                size_t corpus_index =
                    (size_t)round_slot * job->samples + sample;
                lc_deal_from_deck(
                    &state, job->decks + corpus_index * NCARD);
                state.round = (uint8_t)round;
                state.cum[0] = (int16_t)lead;
                state.cum[1] = 0;
                /* Build the "perspective does not start" kernel.  Exact
                 * player-swap antisymmetry supplies the opposite orientation. */
                state.turn = 1;
                const uint8_t (*role)[NSUIT] =
                    (const uint8_t (*)[NSUIT])
                    (job->roles + corpus_index * 2U * NSUIT);
                int margin = 0;
                if (rollout_match_value_round(
                        job->net, &eval_plan, job->controller, &state,
                        role, &margin) != 0 ||
                    margin < -MATCH_VALUE_ROUND_MARGIN_LIMIT ||
                    margin > MATCH_VALUE_ROUND_MARGIN_LIMIT) {
                    atomic_store_explicit(
                        job->failed, 1, memory_order_relaxed);
                    return NULL;
                }
                size_t index = histogram_index(
                    round_slot, lead_slot, margin);
                job->histogram[index]++;
            }
        }
    }
    return NULL;
}

static int clamp_policy_lead(int lead)
{
    if (lead < -MATCH_VALUE_POLICY_LEAD_LIMIT)
        return -MATCH_VALUE_POLICY_LEAD_LIMIT;
    if (lead > MATCH_VALUE_POLICY_LEAD_LIMIT)
        return MATCH_VALUE_POLICY_LEAD_LIMIT;
    return lead;
}

static double final_utility(int lead)
{
    return 0.05 * (double)lead +
           50.0 * (double)((lead > 0) - (lead < 0));
}

/* Equal-weight pool-adjacent-violators regularization.  Optimal game value is
 * nondecreasing in carried lead, but the policy value of a fixed neural
 * controller need not be: changing its lead feature can change its behavior.
 * We therefore emit the unmodified policy-evaluation table alongside this
 * explicitly flagged structural prior when --raw-out is requested. */
static double isotonic_fit(double *value, int count)
{
    typedef struct { int first, last, weight; double sum; } Block;
    Block *block = (Block *)malloc(sizeof *block * (size_t)count);
    double *original = (double *)malloc(sizeof *original * (size_t)count);
    if (!block || !original) {
        free(block); free(original);
        return -1.0;
    }
    memcpy(original, value, sizeof *value * (size_t)count);
    int nblock = 0;
    for (int i = 0; i < count; i++) {
        block[nblock++] = (Block){ i, i, 1, value[i] };
        while (nblock >= 2) {
            Block *a = &block[nblock - 2];
            Block *b = &block[nblock - 1];
            if (a->sum / a->weight <= b->sum / b->weight) break;
            a->last = b->last;
            a->sum += b->sum;
            a->weight += b->weight;
            nblock--;
        }
    }
    for (int b = 0; b < nblock; b++) {
        double mean = block[b].sum / block[b].weight;
        for (int i = block[b].first; i <= block[b].last; i++)
            value[i] = mean;
    }
    double max_adjustment = 0.0;
    for (int i = 0; i < count; i++) {
        double adjustment = fabs(value[i] - original[i]);
        if (adjustment > max_adjustment) max_adjustment = adjustment;
    }
    free(block); free(original);
    return max_adjustment;
}

static void mirror_starting(const double *not_starting, double *starting,
                            int count)
{
    for (int i = 0; i < count; i++)
        starting[i] = -not_starting[count - 1 - i];
}

static int build_values(const uint32_t *histogram, uint32_t samples,
                        int isotonic_projected, MatchValueTable *table)
{
    double inverse = 1.0 / (double)samples;
    for (int lead = -MATCH_VALUE_R2_LEAD_LIMIT;
         lead <= MATCH_VALUE_R2_LEAD_LIMIT; lead++) {
        int policy_lead = clamp_policy_lead(lead);
        int lead_slot = policy_lead + MATCH_VALUE_POLICY_LEAD_LIMIT;
        double total = 0.0;
        for (int margin = -MATCH_VALUE_ROUND_MARGIN_LIMIT;
             margin <= MATCH_VALUE_ROUND_MARGIN_LIMIT; margin++) {
            uint32_t count = histogram[histogram_index(1, lead_slot, margin)];
            total += (double)count * final_utility(lead + margin);
        }
        table->before_round2[0]
            [lead + MATCH_VALUE_R2_LEAD_LIMIT] = total * inverse;
    }
    double adjustment2 = isotonic_projected
        ? isotonic_fit(table->before_round2[0], MATCH_VALUE_R2_COUNT)
        : 0.0;
    if (adjustment2 < 0.0) return 0;
    table->max_isotonic_adjustment[1] = adjustment2;
    mirror_starting(table->before_round2[0], table->before_round2[1],
                    MATCH_VALUE_R2_COUNT);

    for (int lead = -MATCH_VALUE_R1_LEAD_LIMIT;
         lead <= MATCH_VALUE_R1_LEAD_LIMIT; lead++) {
        int policy_lead = clamp_policy_lead(lead);
        int lead_slot = policy_lead + MATCH_VALUE_POLICY_LEAD_LIMIT;
        double total = 0.0;
        for (int margin = -MATCH_VALUE_ROUND_MARGIN_LIMIT;
             margin <= MATCH_VALUE_ROUND_MARGIN_LIMIT; margin++) {
            uint32_t count = histogram[histogram_index(0, lead_slot, margin)];
            int next_lead = lead + margin;
            total += (double)count * table->before_round2[1]
                [next_lead + MATCH_VALUE_R2_LEAD_LIMIT];
        }
        table->before_round1[0]
            [lead + MATCH_VALUE_R1_LEAD_LIMIT] = total * inverse;
    }
    double adjustment1 = isotonic_projected
        ? isotonic_fit(table->before_round1[0], MATCH_VALUE_R1_COUNT)
        : 0.0;
    if (adjustment1 < 0.0) return 0;
    table->max_isotonic_adjustment[0] = adjustment1;
    mirror_starting(table->before_round1[0], table->before_round1[1],
                    MATCH_VALUE_R1_COUNT);
    return 1;
}

static int parse_u64(const char *text, uint64_t *value)
{
    if (!text || !*text || text[0] == '-') return 0;
    char *end = NULL;
    errno = 0;
    unsigned long long parsed = strtoull(text, &end, 10);
    if (errno || !end || *end) return 0;
    *value = (uint64_t)parsed;
    return 1;
}

static void usage(const char *program)
{
    fprintf(stderr,
            "usage: %s --model PATH --out PATH [--samples N] "
            "[--raw-out PATH] [--threads N] [--seed N] "
            "[--playout-symmetries N]\n",
            program);
}

int main(int argc, char **argv)
{
    const char *model_path = NULL, *out_path = NULL, *raw_out_path = NULL;
    uint64_t seed = UINT64_C(7331001);
    uint32_t samples = 16000;
    int nthread = 4;
    int playout_symmetries = 20;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--model") && i + 1 < argc)
            model_path = argv[++i];
        else if (!strcmp(argv[i], "--out") && i + 1 < argc)
            out_path = argv[++i];
        else if (!strcmp(argv[i], "--raw-out") && i + 1 < argc)
            raw_out_path = argv[++i];
        else if (!strcmp(argv[i], "--samples") && i + 1 < argc) {
            uint64_t x;
            if (!parse_u64(argv[++i], &x) || x == 0 || x > 1000000U) {
                usage(argv[0]); return 2;
            }
            samples = (uint32_t)x;
        } else if (!strcmp(argv[i], "--threads") && i + 1 < argc) {
            uint64_t x;
            if (!parse_u64(argv[++i], &x) || x == 0 || x > POLICY_LEADS) {
                usage(argv[0]); return 2;
            }
            nthread = (int)x;
        } else if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
            if (!parse_u64(argv[++i], &seed)) { usage(argv[0]); return 2; }
        } else if (!strcmp(argv[i], "--playout-symmetries") &&
                   i + 1 < argc) {
            uint64_t x;
            if (!parse_u64(argv[++i], &x) ||
                (x != 5 && x != 10 && x != 20 && x != 120)) {
                usage(argv[0]); return 2;
            }
            playout_symmetries = (int)x;
        } else if (!strcmp(argv[i], "--help")) {
            usage(argv[0]); return 0;
        } else {
            usage(argv[0]); return 2;
        }
    }
    if (!model_path || !out_path ||
        (raw_out_path && !strcmp(raw_out_path, out_path))) {
        usage(argv[0]); return 2;
    }

    Net *net = (Net *)malloc(sizeof *net);
    MatchValueTable *table = (MatchValueTable *)calloc(1, sizeof *table);
    size_t histogram_cells = 2U * POLICY_LEADS * ROUND_MARGINS;
    uint32_t *histogram = (uint32_t *)calloc(
        histogram_cells, sizeof *histogram);
    size_t corpus_count = 2U * (size_t)samples;
    uint8_t *decks = (uint8_t *)malloc(corpus_count * NCARD);
    uint8_t *roles = (uint8_t *)malloc(corpus_count * 2U * NSUIT);
    if (!net || !table || !histogram || !decks || !roles) {
        fprintf(stderr, "out of memory\n");
        free(net); free(table); free(histogram); free(decks); free(roles);
        return 1;
    }
    if (net_load(net, model_path) != 0) {
        fprintf(stderr, "cannot load model '%s'\n", model_path);
        free(net); free(table); free(histogram); free(decks); free(roles);
        return 1;
    }
    MatchValueController controller = {
        .net_fingerprint = match_value_net_fingerprint(net),
        .controller_abi = MATCH_VALUE_CONTROLLER_ABI,
        .build_profile = match_value_build_profile(),
        .objective = 0,
        .playout_symmetries = (uint32_t)playout_symmetries,
        .playout_sample = 4,
        .playout_prune = 1,
        .exact_terminal = 1,
        .plan_deck_max = 0,
        .plan_block_gap = 0,
        .draw_playout_deck_max = 0,
        .deck2_replan_worlds = 0,
        .deck2_replan_cores = 0,
        .max_plies = LC_MAX_PLIES
    };
    if (!prepare_corpus(seed, samples, (int)controller.playout_symmetries,
                        decks, roles)) {
        fprintf(stderr, "cannot prepare deterministic build corpus\n");
        free(net); free(table); free(histogram); free(decks); free(roles);
        return 1;
    }

    pthread_t *thread = (pthread_t *)calloc((size_t)nthread, sizeof *thread);
    BuildJob *job = (BuildJob *)calloc((size_t)nthread, sizeof *job);
    atomic_int failed;
    atomic_init(&failed, 0);
    if (!thread || !job) {
        fprintf(stderr, "out of memory\n");
        free(thread); free(job); free(net); free(table); free(histogram);
        free(decks); free(roles);
        return 1;
    }
    int created = 0;
    for (int t = 0; t < nthread; t++) {
        job[t] = (BuildJob){
            .net = net, .controller = &controller, .histogram = histogram,
            .decks = decks, .roles = roles, .samples = samples, .thread = t,
            .nthread = nthread, .failed = &failed
        };
        if (pthread_create(&thread[t], NULL, build_worker, &job[t]) != 0) {
            atomic_store_explicit(&failed, 1, memory_order_relaxed);
            break;
        }
        created++;
    }
    for (int t = 0; t < created; t++)
        if (pthread_join(thread[t], NULL) != 0)
            atomic_store_explicit(&failed, 1, memory_order_relaxed);
    free(thread); free(job); free(decks); free(roles);
    if (created != nthread || atomic_load_explicit(&failed,
                                                   memory_order_relaxed)) {
        fprintf(stderr, "match-value transition generation failed\n");
        free(net); free(table); free(histogram);
        return 1;
    }

    table->version = MATCH_VALUE_VERSION;
    table->samples_per_policy_lead = samples;
    table->role_cycle_size =
        controller.playout_symmetries * controller.playout_symmetries;
    table->role_balance_complete =
        samples % table->role_cycle_size == 0;
    table->isotonic_projected = 1;
    table->source_seed = seed;
    table->controller = controller;
    if (!build_values(histogram, samples, 1, table) ||
        !match_value_validate(table)) {
        fprintf(stderr, "generated match-value table failed validation\n");
        free(net); free(table); free(histogram);
        return 1;
    }
    MatchValueTable *raw_table = NULL;
    if (raw_out_path) {
        raw_table = (MatchValueTable *)calloc(1, sizeof *raw_table);
        if (!raw_table) {
            fprintf(stderr, "out of memory\n");
            free(net); free(table); free(histogram);
            return 1;
        }
        raw_table->version = MATCH_VALUE_VERSION;
        raw_table->samples_per_policy_lead = samples;
        raw_table->role_cycle_size = table->role_cycle_size;
        raw_table->role_balance_complete = table->role_balance_complete;
        raw_table->isotonic_projected = 0;
        raw_table->source_seed = seed;
        raw_table->controller = controller;
        if (!build_values(histogram, samples, 0, raw_table)) {
            fprintf(stderr, "generated raw match-value table failed validation\n");
            free(net); free(table); free(raw_table); free(histogram);
            return 1;
        }
        raw_table->max_isotonic_adjustment[0] =
            table->max_isotonic_adjustment[0];
        raw_table->max_isotonic_adjustment[1] =
            table->max_isotonic_adjustment[1];
        if (!match_value_validate(raw_table)) {
            fprintf(stderr, "generated raw match-value table failed validation\n");
            free(net); free(table); free(raw_table); free(histogram);
            return 1;
        }
    }
    int saved = match_value_save(table, out_path);
    if (saved != 0) {
        fprintf(stderr, "cannot create match-value table '%s' (error %d)\n",
                out_path, saved);
        free(net); free(table); free(raw_table); free(histogram);
        return 1;
    }
    if (raw_table) {
        int raw_saved = match_value_save(raw_table, raw_out_path);
        if (raw_saved != 0) {
            fprintf(stderr,
                    "cannot create raw match-value table '%s' (error %d)\n",
                    raw_out_path, raw_saved);
            if (remove(out_path) != 0)
                fprintf(stderr,
                        "warning: could not roll back paired output '%s'\n",
                        out_path);
            free(net); free(table); free(raw_table); free(histogram);
            return 1;
        }
    }
    printf("match-value table: variant=isotonic samples=%u seed=%" PRIu64
           " model=%016" PRIx64 " abi=%u build=%08" PRIx32
           " role_cycle=%u role_balance=%s "
           "iso_r1=%.9g iso_r2=%.9g output=%s\n",
           samples, seed, controller.net_fingerprint,
           controller.controller_abi, controller.build_profile,
           table->role_cycle_size,
           table->role_balance_complete ? "complete" : "incomplete",
           table->max_isotonic_adjustment[0],
           table->max_isotonic_adjustment[1], out_path);
    if (raw_table)
        printf("match-value table: variant=raw samples=%u seed=%" PRIu64
               " model=%016" PRIx64 " abi=%u build=%08" PRIx32
               " role_cycle=%u role_balance=%s "
               "iso_r1=%.9g iso_r2=%.9g output=%s\n",
               samples, seed, controller.net_fingerprint,
               controller.controller_abi, controller.build_profile,
               raw_table->role_cycle_size,
               raw_table->role_balance_complete
                   ? "complete" : "incomplete",
               raw_table->max_isotonic_adjustment[0],
               raw_table->max_isotonic_adjustment[1],
               raw_out_path);
    if (!table->role_balance_complete)
        fprintf(stderr,
                "warning: incomplete role cycle; the production rollout "
                "parser will reject this development fixture\n");
    free(net); free(table); free(raw_table); free(histogram);
    return 0;
}
