#define _POSIX_C_SOURCE 200809L

/* history_belief_train -- standalone causal action-history belief fitting.
 *
 * The frozen actor generates trajectories but is never changed.  At each
 * scored state, model inputs are built from agent_information_view() and the
 * public opponent-action transcript; the complete referee hand is consulted
 * only afterward to form the exact-K label.  Train and evaluation seeds are
 * explicit, so large disjoint panels can be sharded without reading any
 * reviewed-ply artifact.
 */
#include "../src/agent.h"
#include "../src/history_belief_exclusion.h"
#include "../src/history_belief_model.h"
#include "../src/lc.h"
#include "../src/net.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

typedef struct {
    uint32_t h[8];
    uint64_t bits;
    unsigned char block[64];
    size_t used;
} Sha256;

typedef struct {
    uint64_t states;
    uint64_t cards;
    uint64_t positives;
    double nll;
    double brier;
    double top_hits;
} Metrics;

typedef struct {
    float *gradient;
    double *sum_square;
    uint32_t *mark;
    uint32_t *touched;
    uint32_t generation;
} Optimizer;

typedef struct {
    Adam adam;
    Net gradient;
    uint64_t batch_states;
    uint64_t trained_states;
    uint64_t optimizer_steps;
} ControlOptimizer;

typedef struct {
    const Net *actor;
    const Net *base_net;
    Net *control_net;
    const Net *matched_net;
    const HistoryBeliefRuntime *runtime;
    const HistoryBeliefExclusions *exclusions;
    HistoryBeliefModel *model;
    Optimizer *optimizer;
    ControlOptimizer *control_optimizer;
    int training;
    int training_control;
    int symmetries;
    float temperature;
    float learning_rate;
    float l2;
    float matched_alpha;
    float incumbent_alpha;
    long control_batch_size;
    int max_scored_ply;
    FILE *match_jsonl;
    uint64_t seed;
    uint64_t actor_fingerprint;
    uint64_t base_net_fingerprint;
    uint64_t model_fingerprint;
    uint64_t matched_net_fingerprint;
    uint64_t source_match_id;
    uint64_t source_state_count;
    Sha256 source_manifest;
    uint64_t rounds_completed;
    uint64_t capped_rounds;
    uint64_t excluded_states;
    uint64_t match_excluded_states;
    Metrics history;
    Metrics current;
    Metrics uniform;
    Metrics matched;
    Metrics incumbent;
    Metrics post_history;
    Metrics post_current;
    Metrics post_uniform;
    Metrics post_matched;
    Metrics post_incumbent;
    Metrics match_history;
    Metrics match_current;
    Metrics match_uniform;
    Metrics match_matched;
    Metrics match_incumbent;
    Metrics match_post_history;
    Metrics match_post_current;
    Metrics match_post_uniform;
    Metrics match_post_matched;
    Metrics match_post_incumbent;
} Run;

static size_t belief_head_offset(void);
static size_t belief_head_bytes(void);

static uint32_t rotr32(uint32_t x, int n)
{
    return (x >> n) | (x << (32 - n));
}

static void sha256_transform(Sha256 *s, const unsigned char block[64])
{
    static const uint32_t k[64] = {
        0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,
        0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
        0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,
        0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
        0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,
        0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
        0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,
        0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
        0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,
        0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
        0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,
        0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
        0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,
        0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
        0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,
        0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2,
    };
    uint32_t w[64];
    for (int i = 0; i < 16; i++)
        w[i] = (uint32_t)block[4*i] << 24 |
               (uint32_t)block[4*i+1] << 16 |
               (uint32_t)block[4*i+2] << 8 | block[4*i+3];
    for (int i = 16; i < 64; i++) {
        uint32_t a = rotr32(w[i-15], 7) ^ rotr32(w[i-15], 18) ^
                     (w[i-15] >> 3);
        uint32_t b = rotr32(w[i-2], 17) ^ rotr32(w[i-2], 19) ^
                     (w[i-2] >> 10);
        w[i] = w[i-16] + a + w[i-7] + b;
    }
    uint32_t a=s->h[0], b=s->h[1], c=s->h[2], d=s->h[3];
    uint32_t e=s->h[4], f=s->h[5], g=s->h[6], h=s->h[7];
    for (int i = 0; i < 64; i++) {
        uint32_t s1 = rotr32(e,6)^rotr32(e,11)^rotr32(e,25);
        uint32_t ch = (e & f) ^ (~e & g);
        uint32_t t1 = h + s1 + ch + k[i] + w[i];
        uint32_t s0 = rotr32(a,2)^rotr32(a,13)^rotr32(a,22);
        uint32_t maj = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = s0 + maj;
        h=g; g=f; f=e; e=d+t1; d=c; c=b; b=a; a=t1+t2;
    }
    s->h[0]+=a; s->h[1]+=b; s->h[2]+=c; s->h[3]+=d;
    s->h[4]+=e; s->h[5]+=f; s->h[6]+=g; s->h[7]+=h;
}

static void sha256_init(Sha256 *s)
{
    static const uint32_t initial[8] = {
        0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,
        0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19,
    };
    memset(s, 0, sizeof *s);
    memcpy(s->h, initial, sizeof initial);
}

static void sha256_update(Sha256 *s, const void *data, size_t n)
{
    const unsigned char *p = data;
    s->bits += (uint64_t)n * 8;
    while (n) {
        size_t take = 64 - s->used;
        if (take > n) take = n;
        memcpy(s->block + s->used, p, take);
        s->used += take;
        p += take;
        n -= take;
        if (s->used == 64) {
            sha256_transform(s, s->block);
            s->used = 0;
        }
    }
}

static void sha256_final(Sha256 *s, unsigned char out[32])
{
    uint64_t bits = s->bits;
    s->block[s->used++] = 0x80;
    if (s->used > 56) {
        memset(s->block + s->used, 0, 64 - s->used);
        sha256_transform(s, s->block);
        s->used = 0;
    }
    memset(s->block + s->used, 0, 56 - s->used);
    for (int i = 0; i < 8; i++)
        s->block[63-i] = (unsigned char)(bits >> (8*i));
    sha256_transform(s, s->block);
    for (int i = 0; i < 8; i++) {
        out[4*i] = (unsigned char)(s->h[i] >> 24);
        out[4*i+1] = (unsigned char)(s->h[i] >> 16);
        out[4*i+2] = (unsigned char)(s->h[i] >> 8);
        out[4*i+3] = (unsigned char)s->h[i];
    }
}

static void sha256_u64(Sha256 *hash, uint64_t value)
{
    unsigned char encoded[8];
    for (int i = 0; i < 8; i++)
        encoded[i] = (unsigned char)(value >> (8 * i));
    sha256_update(hash, encoded, sizeof encoded);
}

static void digest_hex(const unsigned char digest[32], char hex[65])
{
    static const char digit[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) {
        hex[2 * i] = digit[digest[i] >> 4];
        hex[2 * i + 1] = digit[digest[i] & 15];
    }
    hex[64] = '\0';
}

/* Hash only public/source inputs and do so before any truth hand is read.
 * This lets independently executed residual and head-control jobs prove that
 * they consumed exactly the same causal examples. */
static int source_manifest_add(
    Run *run, const State *view, const HistoryBeliefTrace *trace,
    int observer, const unsigned char orbit[32],
    const uint8_t *candidate, int n, int need)
{
    static const unsigned char domain[] =
        "lc-history-belief-public-source-v1";
    if (!run || !view || !trace || !orbit || !candidate || n < 0 ||
        n > NCARD || need < 0 || need > n || observer != view->turn)
        return 0;
    sha256_update(&run->source_manifest, domain, sizeof domain);
    sha256_u64(&run->source_manifest, run->source_match_id);
    sha256_u64(&run->source_manifest, (uint64_t)view->round);
    sha256_u64(&run->source_manifest, (uint64_t)view->nply);
    sha256_u64(&run->source_manifest, (uint64_t)observer);
    sha256_u64(&run->source_manifest, (uint64_t)n);
    sha256_u64(&run->source_manifest, (uint64_t)need);
    sha256_update(&run->source_manifest, orbit, 32);
    sha256_update(&run->source_manifest, candidate, (size_t)n);
    sha256_u64(&run->source_manifest, (uint64_t)trace->n);
    for (uint16_t i = 0; i < trace->n; i++) {
        const HistoryBeliefEvent *event = &trace->event[i];
        sha256_u64(&run->source_manifest, event->ply);
        sha256_u64(&run->source_manifest, event->suit);
        sha256_u64(&run->source_manifest, event->rank);
        sha256_u64(&run->source_manifest, event->discard);
        sha256_u64(&run->source_manifest, event->draw);
        sha256_u64(&run->source_manifest, event->draw_rank);
        sha256_u64(&run->source_manifest, event->opponent);
        sha256_u64(&run->source_manifest, event->deck_left);
    }
    run->source_state_count++;
    return 1;
}

static int file_sha256(const char *path, unsigned char digest[32])
{
    if (!path || !digest) return 0;
    FILE *fp = fopen(path, "rb");
    if (!fp) return 0;
    Sha256 hash;
    sha256_init(&hash);
    unsigned char buffer[65536];
    size_t n;
    while ((n = fread(buffer, 1, sizeof buffer, fp)) != 0)
        sha256_update(&hash, buffer, n);
    int ok = !ferror(fp) && fclose(fp) == 0;
    if (!ok) return 0;
    sha256_final(&hash, digest);
    return 1;
}

static int file_sha256_hex(const char *path, char hex[65])
{
    unsigned char digest[32];
    if (!hex || !file_sha256(path, digest)) return 0;
    digest_hex(digest, hex);
    return 1;
}

#define CONTROL_STATE_VERSION 3U
#define CONTROL_STATE_HEADER_BYTES 224U

static void encode_u32le(unsigned char *out, uint32_t value)
{
    for (int i = 0; i < 4; i++) out[i] = (unsigned char)(value >> (8 * i));
}

static void encode_u64le(unsigned char *out, uint64_t value)
{
    for (int i = 0; i < 8; i++) out[i] = (unsigned char)(value >> (8 * i));
}

static uint32_t decode_u32le(const unsigned char *in)
{
    uint32_t value = 0;
    for (int i = 0; i < 4; i++) value |= (uint32_t)in[i] << (8 * i);
    return value;
}

static uint64_t decode_u64le(const unsigned char *in)
{
    uint64_t value = 0;
    for (int i = 0; i < 8; i++) value |= (uint64_t)in[i] << (8 * i);
    return value;
}

static void encode_f32le(unsigned char *out, float value)
{
    uint32_t bits;
    memcpy(&bits, &value, sizeof bits);
    encode_u32le(out, bits);
}

static float decode_f32le(const unsigned char *in)
{
    uint32_t bits = decode_u32le(in);
    float value;
    memcpy(&value, &bits, sizeof value);
    return value;
}

static char *temporary_path(const char *path)
{
    size_t n = strlen(path);
    char *tmp = malloc(n + sizeof ".tmp.XXXXXX");
    if (!tmp) return NULL;
    memcpy(tmp, path, n);
    memcpy(tmp + n, ".tmp.XXXXXX", sizeof ".tmp.XXXXXX");
    return tmp;
}

static int publish_net_exclusive(const Net *net, const char *path)
{
    char *tmp = temporary_path(path);
    if (!tmp) return 0;
    int fd = mkstemp(tmp);
    if (fd < 0) { free(tmp); return 0; }
    int ok = close(fd) == 0 && net_save(net, tmp) == 0 &&
             link(tmp, path) == 0;
    unlink(tmp);
    free(tmp);
    return ok;
}

static int publish_model_exclusive(const HistoryBeliefModel *model,
                                   const char *path)
{
    char *tmp = temporary_path(path);
    if (!tmp) return 0;
    int fd = mkstemp(tmp);
    if (fd < 0) { free(tmp); return 0; }
    int ok = close(fd) == 0 && history_belief_model_save(model, tmp) == 0 &&
             link(tmp, path) == 0;
    unlink(tmp);
    free(tmp);
    return ok;
}

static int write_f32_region(FILE *fp, const void *region, size_t bytes)
{
    const float *value = region;
    unsigned char encoded[4];
    if (bytes % sizeof(float)) return 0;
    for (size_t i = 0; i < bytes / sizeof(float); i++) {
        if (!lc_float_isfinite(value[i])) return 0;
        encode_f32le(encoded, value[i]);
        if (fwrite(encoded, 1, sizeof encoded, fp) != sizeof encoded)
            return 0;
    }
    return 1;
}

static int read_f32_region(FILE *fp, void *region, size_t bytes)
{
    float *value = region;
    unsigned char encoded[4];
    if (bytes % sizeof(float)) return 0;
    for (size_t i = 0; i < bytes / sizeof(float); i++) {
        if (fread(encoded, 1, sizeof encoded, fp) != sizeof encoded)
            return 0;
        value[i] = decode_f32le(encoded);
        if (!lc_float_isfinite(value[i])) return 0;
    }
    return 1;
}

static int control_state_save(
    const ControlOptimizer *optimizer, const Net *net, const char *path,
    uint64_t actor_fingerprint, uint64_t seed, uint64_t next_match_start,
    int rounds, int max_scored_ply, long control_batch_size,
    int symmetries, float temperature, float alpha, float lr, float l2,
    const unsigned char exclusion_sha256[32],
    const unsigned char source_manifest_sha256[32],
    const unsigned char checkpoint_sha256[32])
{
    if (!optimizer || !net || !path || !*path || !exclusion_sha256 ||
        !source_manifest_sha256 || !checkpoint_sha256 || rounds < 1 ||
        max_scored_ply < 0 || control_batch_size < 1)
        return 0;
    unsigned char header[CONTROL_STATE_HEADER_BYTES];
    memset(header, 0, sizeof header);
    memcpy(header, "LCBHCS1", 7);
    encode_u32le(header + 8, CONTROL_STATE_VERSION);
    encode_u32le(header + 12, CONTROL_STATE_HEADER_BYTES);
    encode_u64le(header + 16, actor_fingerprint);
    encode_u64le(header + 24, history_belief_actor_fingerprint(net));
    encode_u64le(header + 32, seed);
    encode_u64le(header + 40, next_match_start);
    encode_u64le(header + 48, optimizer->trained_states);
    encode_u64le(header + 56, optimizer->optimizer_steps);
    encode_u64le(header + 64, optimizer->batch_states);
    encode_u32le(header + 72, (uint32_t)symmetries);
    encode_f32le(header + 80, temperature);
    encode_f32le(header + 84, alpha);
    encode_f32le(header + 88, lr);
    encode_f32le(header + 92, l2);
    memcpy(header + 96, exclusion_sha256, 32);
    encode_u64le(header + 128, 3 * (uint64_t)belief_head_bytes());
    encode_u64le(header + 136, (uint64_t)optimizer->adam.t);
    encode_u32le(header + 144, (uint32_t)rounds);
    encode_u32le(header + 148, (uint32_t)max_scored_ply);
    encode_u64le(header + 152, (uint64_t)control_batch_size);
    memcpy(header + 160, source_manifest_sha256, 32);
    memcpy(header + 192, checkpoint_sha256, 32);

    char *tmp = temporary_path(path);
    if (!tmp) return 0;
    int fd = mkstemp(tmp);
    if (fd < 0) { free(tmp); return 0; }
    FILE *fp = fdopen(fd, "wb");
    int ok = fp && fwrite(header, 1, sizeof header, fp) == sizeof header &&
        write_f32_region(fp, (const unsigned char *)&optimizer->adam.m +
                        belief_head_offset(), belief_head_bytes()) &&
        write_f32_region(fp, (const unsigned char *)&optimizer->adam.v +
                        belief_head_offset(), belief_head_bytes()) &&
        write_f32_region(fp, (const unsigned char *)&optimizer->gradient +
                        belief_head_offset(), belief_head_bytes());
    if (fp) {
        if (fflush(fp) != 0 || fsync(fileno(fp)) != 0 || fclose(fp) != 0)
            ok = 0;
    } else close(fd);
    if (ok && link(tmp, path) != 0) ok = 0;
    unlink(tmp);
    free(tmp);
    return ok;
}

static int control_state_load(
    ControlOptimizer *optimizer, const Net *net, const char *path,
    uint64_t actor_fingerprint, uint64_t seed, uint64_t match_start,
    int rounds, int max_scored_ply, long control_batch_size,
    int symmetries, float temperature, float alpha, float lr, float l2,
    const unsigned char exclusion_sha256[32],
    const unsigned char checkpoint_sha256[32])
{
    if (!optimizer || !net || !path || !*path || !exclusion_sha256 ||
        !checkpoint_sha256)
        return 0;
    FILE *fp = fopen(path, "rb");
    unsigned char header[CONTROL_STATE_HEADER_BYTES];
    if (!fp || fread(header, 1, sizeof header, fp) != sizeof header) {
        if (fp) fclose(fp);
        return 0;
    }
    float saved_temperature = decode_f32le(header + 80);
    float saved_alpha = decode_f32le(header + 84);
    float saved_lr = decode_f32le(header + 88);
    float saved_l2 = decode_f32le(header + 92);
    uint64_t saved_t = decode_u64le(header + 136);
    int ok = !memcmp(header, "LCBHCS1", 7) && header[7] == 0 &&
        decode_u32le(header + 8) == CONTROL_STATE_VERSION &&
        decode_u32le(header + 12) == CONTROL_STATE_HEADER_BYTES &&
        decode_u64le(header + 16) == actor_fingerprint &&
        decode_u64le(header + 24) == history_belief_actor_fingerprint(net) &&
        decode_u64le(header + 32) == seed &&
        decode_u64le(header + 40) == match_start &&
        decode_u32le(header + 72) == (uint32_t)symmetries &&
        saved_temperature == temperature && saved_alpha == alpha &&
        saved_lr == lr && saved_l2 == l2 &&
        !memcmp(header + 96, exclusion_sha256, 32) &&
        decode_u64le(header + 128) == 3 * (uint64_t)belief_head_bytes() &&
        saved_t <= (uint64_t)LONG_MAX &&
        decode_u32le(header + 144) == (uint32_t)rounds &&
        decode_u32le(header + 148) == (uint32_t)max_scored_ply &&
        decode_u64le(header + 152) == (uint64_t)control_batch_size &&
        !memcmp(header + 192, checkpoint_sha256, 32);
    memset(optimizer, 0, sizeof *optimizer);
    if (ok) {
        optimizer->trained_states = decode_u64le(header + 48);
        optimizer->optimizer_steps = decode_u64le(header + 56);
        optimizer->batch_states = decode_u64le(header + 64);
        optimizer->adam.t = (long)saved_t;
        ok = optimizer->optimizer_steps == saved_t &&
             optimizer->batch_states < (uint64_t)control_batch_size &&
             read_f32_region(fp, (unsigned char *)&optimizer->adam.m +
                             belief_head_offset(), belief_head_bytes()) &&
             read_f32_region(fp, (unsigned char *)&optimizer->adam.v +
                             belief_head_offset(), belief_head_bytes()) &&
             read_f32_region(fp, (unsigned char *)&optimizer->gradient +
                             belief_head_offset(), belief_head_bytes());
    }
    if (ok) {
        int trailing = fgetc(fp);
        if (trailing != EOF || ferror(fp)) ok = 0;
    }
    if (fclose(fp) != 0) ok = 0;
    return ok;
}

static void write_match_metric(FILE *fp, const char *name,
                               const Metrics *metric)
{
    fprintf(fp,
            "\"%s\":{\"brier_sum\":%.17g,\"nll_sum\":%.17g,"
            "\"positive_count\":%llu,\"state_count\":%llu,"
            "\"top_hits_sum\":%.17g,\"uncertain_card_count\":%llu}",
            name, metric->brier, metric->nll,
            (unsigned long long)metric->positives,
            (unsigned long long)metric->states, metric->top_hits,
            (unsigned long long)metric->cards);
}

static void write_metric_group(FILE *fp, const Metrics *history,
                               const Metrics *current,
                               const Metrics *matched,
                               const Metrics *incumbent,
                               const Metrics *uniform)
{
    fputc('{', fp);
    write_match_metric(fp, "base_262k_head", current);
    if (matched) {
        fputc(',', fp);
        write_match_metric(fp, "matched_head_control", matched);
    }
    if (incumbent) {
        fputc(',', fp);
        write_match_metric(fp, "incumbent_head", incumbent);
    }
    fputc(',', fp);
    write_match_metric(fp, "history", history);
    fputc(',', fp);
    write_match_metric(fp, "uniform_exact_k", uniform);
    fputc('}', fp);
}

static int write_match_row(const Run *run, uint64_t source_match_id,
                           uint64_t capped_rounds,
                           uint64_t rounds_completed,
                           const Metrics *all_history,
                           const Metrics *all_current,
                           const Metrics *all_matched,
                           const Metrics *all_incumbent,
                           const Metrics *all_uniform,
                           const Metrics *post_history,
                           const Metrics *post_current,
                           const Metrics *post_matched,
                           const Metrics *post_incumbent,
                           const Metrics *post_uniform)
{
    if (!run->match_jsonl) return 1;
    FILE *fp = run->match_jsonl;
    fprintf(fp, "{\"actor_fingerprint\":\"%016llx\","
            "\"base_alpha\":%.9g,"
            "\"base_net_fingerprint\":\"%016llx\","
            "\"matched_base_alpha\":",
            (unsigned long long)run->actor_fingerprint,
            run->model->base_alpha,
            (unsigned long long)run->base_net_fingerprint);
    if (run->matched_net)
        fprintf(fp, "%.9g", run->matched_alpha);
    else
        fputs("null", fp);
    fprintf(fp, ",\"incumbent_alpha\":%.9g,"
            "\"incumbent_net_fingerprint\":\"%016llx\"",
            run->incumbent_alpha,
            (unsigned long long)run->actor_fingerprint);
    fputs(",\"matched_base_net_fingerprint\":", fp);
    if (run->matched_net)
        fprintf(fp, "\"%016llx\"",
                (unsigned long long)run->matched_net_fingerprint);
    else
        fputs("null", fp);
    fprintf(fp,
            ",\"capped_rounds\":%llu,\"excluded_state_count\":%llu,"
            "\"exclusion_manifest_count\":%d,"
            "\"exclusion_manifest_sha256\":",
            (unsigned long long)capped_rounds,
            (unsigned long long)run->match_excluded_states,
            run->exclusions ? run->exclusions->count : 0);
    if (run->exclusions)
        fprintf(fp, "\"%s\"", run->exclusions->manifest_sha256_hex);
    else
        fputs("null", fp);
    fprintf(fp, ","
            "\"history_model_fingerprint\":\"%016llx\","
            "\"max_scored_ply\":%d,\"metrics\":{\"all_states\":",
            (unsigned long long)run->model_fingerprint,
            run->max_scored_ply);
    write_metric_group(fp, all_history, all_current, all_matched,
                       all_incumbent, all_uniform);
    fputs(",\"post_opponent_action\":", fp);
    write_metric_group(fp, post_history, post_current, post_matched,
                       post_incumbent,
                       post_uniform);
    fprintf(fp, "},\"rounds_completed\":%llu,"
            "\"reviewed_ply_inputs_used\":false,"
            "\"schema\":\"lc-history-belief-match-v1\","
            "\"seed_root\":%llu,\"source_match_id\":%llu,"
            "\"structural_contract\":{"
            "\"action_history_public_only\":true,"
            "\"current_view_truth_scrubbed\":true,"
            "\"opening_history_uniform\":true,"
            "\"playing_actor_changed\":false,"
            "\"public_transcript_complete\":true,"
            "\"residual_features_opponent_action_anchored\":true,"
            "\"reviewed_ply_orbit_exclusion_enabled\":%s,"
            "\"suit_equivariant_features\":true,"
            "\"truth_read_after_prediction\":true,"
            "\"wager_identity_collapsed\":true},"
            "\"symmetries\":%d,\"temperature\":%.9g}\n",
            (unsigned long long)rounds_completed,
            (unsigned long long)run->seed,
            (unsigned long long)source_match_id,
            run->exclusions ? "true" : "false",
            run->symmetries, run->temperature);
    return !ferror(fp);
}

static uint64_t mix64(uint64_t x)
{
    x += UINT64_C(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94D049BB133111EB);
    return x ^ (x >> 31);
}

static int parse_u64(const char *text, uint64_t *out)
{
    if (!text || !*text || text[0] == '-') return 0;
    char *end = NULL;
    errno = 0;
    unsigned long long value = strtoull(text, &end, 10);
    if (errno || !end || end == text || *end) return 0;
    *out = (uint64_t)value;
    return 1;
}

static int parse_long(const char *text, long lo, long hi, long *out)
{
    uint64_t value;
    if (!parse_u64(text, &value) || value < (uint64_t)lo ||
        value > (uint64_t)hi)
        return 0;
    *out = (long)value;
    return 1;
}

static int parse_float(const char *text, float lo, float hi, float *out)
{
    if (!text || !*text) return 0;
    char *end = NULL;
    errno = 0;
    double value = strtod(text, &end);
    if (errno || !end || end == text || *end ||
        !lc_double_isfinite(value) || value < lo || value > hi)
        return 0;
    *out = (float)value;
    return 1;
}

static double log_choose(int n, int k)
{
    return lgamma((double)n + 1.0) - lgamma((double)k + 1.0)
         - lgamma((double)(n - k) + 1.0);
}

/* Tie-correct expected top-K hits.  Truth must not break equal-score ties. */
static double top_hits(const float *marginal, const uint8_t *held,
                       int n, int need)
{
    int order[NCARD];
    for (int i = 0; i < n; i++) order[i] = i;
    for (int i = 1; i < n; i++) {
        int x = order[i], j = i;
        while (j > 0 && marginal[x] > marginal[order[j - 1]]) {
            order[j] = order[j - 1];
            j--;
        }
        order[j] = x;
    }
    int slots = need;
    double hit = 0.0;
    for (int at = 0; at < n && slots > 0;) {
        int end = at + 1;
        int positives = held[order[at]] ? 1 : 0;
        while (end < n &&
               marginal[order[end]] == marginal[order[at]]) {
            positives += held[order[end]] ? 1 : 0;
            end++;
        }
        int group = end - at;
        int take = slots < group ? slots : group;
        hit += (double)take * (double)positives / (double)group;
        slots -= take;
        at = end;
    }
    return hit;
}

static void metrics_add(Metrics *metric, const float *marginal,
                        const uint8_t *held, int n, int need, double nll)
{
    metric->states++;
    metric->cards += (uint64_t)n;
    metric->positives += (uint64_t)need;
    metric->nll += nll;
    for (int i = 0; i < n; i++) {
        double error = (double)marginal[i] - held[i];
        metric->brier += error * error;
    }
    metric->top_hits += top_hits(marginal, held, n, need);
}

static int valid_marginals(const float *marginal, int n, int need)
{
    double total = 0.0;
    if (!marginal || n < 0 || n > NCARD || need < 0 || need > n)
        return 0;
    for (int i = 0; i < n; i++) {
        if (!lc_float_isfinite(marginal[i]) || marginal[i] < -1e-6f ||
            marginal[i] > 1.000001f)
            return 0;
        total += marginal[i];
    }
    return fabs(total - (double)need) <= 2e-4;
}

static int valid_nll(double *nll)
{
    if (!nll || !lc_double_isfinite(*nll) || *nll < -1e-9) return 0;
    if (*nll < 0.0) *nll = 0.0;
    return 1;
}

static int optimizer_init(Optimizer *optimizer)
{
    memset(optimizer, 0, sizeof *optimizer);
    optimizer->gradient = calloc(HISTORY_BELIEF_FEATURES, sizeof(float));
    optimizer->sum_square = calloc(HISTORY_BELIEF_FEATURES, sizeof(double));
    optimizer->mark = calloc(HISTORY_BELIEF_FEATURES, sizeof(uint32_t));
    optimizer->touched = malloc(HISTORY_BELIEF_FEATURES * sizeof(uint32_t));
    return optimizer->gradient && optimizer->sum_square && optimizer->mark &&
           optimizer->touched;
}

static void optimizer_free(Optimizer *optimizer)
{
    if (!optimizer) return;
    free(optimizer->gradient);
    free(optimizer->sum_square);
    free(optimizer->mark);
    free(optimizer->touched);
    memset(optimizer, 0, sizeof *optimizer);
}

static size_t belief_head_offset(void)
{
    return offsetof(Net, wbel);
}

static size_t belief_head_bytes(void)
{
    return offsetof(Net, wcomb) - offsetof(Net, wbel);
}

static int control_only_belief_head_changed(const Net *before,
                                            const Net *after)
{
    size_t from = belief_head_offset();
    size_t to = from + belief_head_bytes();
    return before && after &&
           memcmp(before, after, from) == 0 &&
           memcmp((const unsigned char *)before + to,
                  (const unsigned char *)after + to,
                  sizeof(Net) - to) == 0;
}

static int finite_region(const void *data, size_t bytes)
{
    const float *value = data;
    if (bytes % sizeof(float)) return 0;
    for (size_t i = 0; i < bytes / sizeof(float); i++)
        if (!lc_float_isfinite(value[i])) return 0;
    return 1;
}

static int control_apply_batch(Run *run)
{
    ControlOptimizer *optimizer = run->control_optimizer;
    if (!optimizer) return 0;
    if (optimizer->batch_states == 0) return 1;
    net_tie_wager_gradients(&optimizer->gradient);
    net_adam_step_belief(run->control_net, &optimizer->gradient,
                         &optimizer->adam, run->learning_rate,
                         1.0f / (float)optimizer->batch_states, run->l2);
    net_project_belief_wager_symmetry(run->control_net);
    memset((unsigned char *)&optimizer->gradient + belief_head_offset(), 0,
           belief_head_bytes());
    optimizer->batch_states = 0;
    optimizer->optimizer_steps++;
    return finite_region((unsigned char *)run->control_net +
                         belief_head_offset(), belief_head_bytes());
}

/* The head-only control sees one deterministic member of the same suit group
 * per source state.  Across a very large corpus this gives exact scheduled
 * augmentation without paying for 20 trunks per training example. */
static int control_train_state(Run *run, const State *view, int observer,
                               const uint8_t *card, const uint8_t *held,
                               int n, int need, float *marginal,
                               double *nll)
{
    uint8_t permutation[120][NSUIT];
    int nperm = suit_permutations(run->symmetries, permutation);
    if (nperm <= 0 || !run->control_net || !run->control_optimizer)
        return 0;
    uint64_t key = run->seed ^
        (run->source_match_id + 1) * UINT64_C(0xd6e8feb86659fd93) ^
        ((uint64_t)view->round << 48) ^ ((uint64_t)view->nply << 16) ^
        (uint64_t)observer;
    int selected = (int)(mix64(key) % (uint64_t)nperm);
    State permuted;
    lc_permute_suits(view, &permuted, permutation[selected]);
    Features feature;
    feat_extract(&permuted, observer, &feature);
    NetAct activation;
    net_trunk(run->control_net, &feature, &activation);
    uint8_t mapped[NCARD];
    float raw[NCARD], derivative[NCARD];
    for (int i = 0; i < n; i++)
        mapped[i] = lc_permute_card(card[i], permutation[selected]);
    net_belief_act(run->control_net, &activation, mapped, n, raw);
    float alpha = run->model ? run->model->base_alpha : run->matched_alpha;
    if (!belief_exact_k_eval(raw, held, n, need, alpha, marginal, nll))
        return 0;
    if (!history_belief_clipped_raw_gradient(
            raw, marginal, held, n, alpha, derivative))
        return 0;
    net_backward_belief_head(&activation, mapped, derivative, n,
                             &run->control_optimizer->gradient);
    run->control_optimizer->batch_states++;
    run->control_optimizer->trained_states++;
    return run->control_optimizer->batch_states <
               (uint64_t)run->control_batch_size || control_apply_batch(run);
}

static int train_state(Run *run, const HistoryBeliefTrace *trace,
                       const State *view, int observer,
                       const uint8_t *card, const uint8_t *held,
                       int n, const float *marginal)
{
    Optimizer *optimizer = run->optimizer;
    if (!optimizer || ++optimizer->generation == 0) {
        if (!optimizer) return 0;
        memset(optimizer->mark, 0,
               HISTORY_BELIEF_FEATURES * sizeof(uint32_t));
        optimizer->generation = 1;
    }
    uint32_t ntouched = 0;
    uint32_t active[HISTORY_BELIEF_MAX_ACTIVE];
    for (int i = 0; i < n; i++) {
        float derivative = marginal[i] - held[i];
        int nf = history_belief_active_features(
            trace, view, observer, card[i], active);
        if (nf < 0) return 0;
        for (int j = 0; j < nf; j++) {
            uint32_t feature = active[j];
            if (optimizer->mark[feature] != optimizer->generation) {
                optimizer->mark[feature] = optimizer->generation;
                optimizer->gradient[feature] = 0.0f;
                optimizer->touched[ntouched++] = feature;
            }
            optimizer->gradient[feature] += derivative;
        }
    }
    for (uint32_t i = 0; i < ntouched; i++) {
        uint32_t feature = optimizer->touched[i];
        double gradient = optimizer->gradient[feature]
                        + (double)run->l2 * run->model->weight[feature];
        optimizer->sum_square[feature] += gradient * gradient;
        double step = (double)run->learning_rate * gradient /
                      (sqrt(optimizer->sum_square[feature]) + 1e-8);
        double updated = (double)run->model->weight[feature] - step;
        if (!lc_double_isfinite(updated)) return 0;
        if (updated > 20.0) updated = 20.0;
        if (updated < -20.0) updated = -20.0;
        run->model->weight[feature] = (float)updated;
    }
    run->model->train_states++;
    return 1;
}

static int score_state(Run *run, const State *complete,
                       const HistoryBeliefTrace *trace)
{
    const int observer = complete->turn;
    unsigned char orbit[HISTORY_BELIEF_EXCLUSION_DIGEST_BYTES];
    if (run->exclusions) {
        int excluded = history_belief_exclusions_check(
            run->exclusions, complete, observer, orbit);
        if (excluded < 0) return 0;
        if (excluded > 0) {
            run->excluded_states++;
            run->match_excluded_states++;
            return 1;
        }
    }
    State view;
    agent_information_view(complete, observer, &view);
    int opponent_actions = history_belief_trace_opponent_actions(trace);
    if (opponent_actions < 0) return 0;

    if (run->training_control) {
        uint8_t card[NCARD], held[NCARD];
        int n = 0;
        lc_unseen(&view, observer, card, &n);
        int opponent = observer ^ 1;
        int need = (int)view.hand_n[opponent] -
                   __builtin_popcountll(view.known[opponent]);
        if (need < 0 || need > HAND_SIZE || need > n) return 0;
        if (n <= 0 || need <= 0 || need >= n || opponent_actions == 0)
            return 1;
        if (!source_manifest_add(run, &view, trace, observer, orbit,
                                 card, n, need))
            return 0;
        /* Exclusion and source identity above this line read no hidden deck
         * byte or opponent private-hand truth. */
        for (int i = 0; i < n; i++)
            held[i] = (uint8_t)((complete->hand[opponent] >> card[i]) & 1ULL);
        float marginal[NCARD];
        double nll;
        if (!control_train_state(run, &view, observer, card, held, n, need,
                                 marginal, &nll) ||
            !valid_marginals(marginal, n, need) || !valid_nll(&nll))
            return 0;
        metrics_add(&run->history, marginal, held, n, need, nll);
        metrics_add(&run->post_history, marginal, held, n, need, nll);
        return 1;
    }

    HistoryBeliefDist prediction;
    if (!history_belief_dist_init(run->runtime, trace, &view, observer,
                                  &prediction))
        return 0;
    int n = prediction.base.n;
    int need = prediction.base.need;
    if (n <= 0 || need <= 0 || need >= n) return 1;
    if (opponent_actions > 0 &&
        !source_manifest_add(run, &view, trace, observer, orbit,
                             prediction.base.card, n, need))
        return 0;

    BeliefDist matched;
    memset(&matched, 0, sizeof matched);
    if (run->matched_net) {
        if (!belief_dist_init(run->matched_net, &view, observer,
                              run->symmetries, run->matched_alpha,
                              &matched) || matched.n != n ||
            matched.need != need ||
            memcmp(matched.card, prediction.base.card, (size_t)n))
            return 0;
    }
    BeliefDist incumbent;
    memset(&incumbent, 0, sizeof incumbent);
    if (run->incumbent_alpha >= 0.0f) {
        if (!belief_dist_init(run->actor, &view, observer,
                              run->symmetries, run->incumbent_alpha,
                              &incumbent) || incumbent.n != n ||
            incumbent.need != need ||
            memcmp(incumbent.card, prediction.base.card, (size_t)n))
            return 0;
    }

    /* Prediction is complete before the referee-only truth hand is read.
     * The label may now score and train the already frozen logits. */
    int opponent = observer ^ 1;
    uint8_t held[NCARD];
    for (int i = 0; i < n; i++)
        held[i] = (uint8_t)((complete->hand[opponent] >>
                             prediction.base.card[i]) & 1ULL);
    double nll, current_nll, matched_nll = 0.0, incumbent_nll = 0.0;
    if (!history_belief_exact_k_eval(prediction.logit, held, n, need,
                                     prediction.marginal, &nll) ||
        !belief_dist_true_nll(&prediction.base, complete->hand[opponent],
                              &current_nll) ||
        (run->matched_net &&
         !belief_dist_true_nll(&matched, complete->hand[opponent],
                               &matched_nll)) ||
        (run->incumbent_alpha >= 0.0f &&
         !belief_dist_true_nll(&incumbent, complete->hand[opponent],
                               &incumbent_nll)) ||
        !valid_marginals(prediction.marginal, n, need) ||
        !valid_marginals(prediction.base.marginal, n, need) ||
        (run->matched_net &&
         !valid_marginals(matched.marginal, n, need)) ||
        (run->incumbent_alpha >= 0.0f &&
         !valid_marginals(incumbent.marginal, n, need)) ||
        !valid_nll(&nll) || !valid_nll(&current_nll) ||
        (run->matched_net && !valid_nll(&matched_nll)) ||
        (run->incumbent_alpha >= 0.0f && !valid_nll(&incumbent_nll)))
        return 0;
    metrics_add(&run->history, prediction.marginal, held, n, need, nll);
    if (run->match_jsonl)
        metrics_add(&run->match_history, prediction.marginal,
                    held, n, need, nll);
    if (opponent_actions > 0) {
        metrics_add(&run->post_history, prediction.marginal,
                    held, n, need, nll);
        if (run->match_jsonl)
            metrics_add(&run->match_post_history, prediction.marginal, held,
                        n, need, nll);
    }

    if (run->matched_net) {
        metrics_add(&run->matched, matched.marginal, held, n, need,
                    matched_nll);
        if (run->match_jsonl)
            metrics_add(&run->match_matched, matched.marginal, held, n,
                        need, matched_nll);
        if (opponent_actions > 0) {
            metrics_add(&run->post_matched, matched.marginal, held, n,
                        need, matched_nll);
            if (run->match_jsonl)
                metrics_add(&run->match_post_matched, matched.marginal,
                            held, n, need, matched_nll);
        }
    }
    if (run->incumbent_alpha >= 0.0f) {
        metrics_add(&run->incumbent, incumbent.marginal, held, n, need,
                    incumbent_nll);
        if (run->match_jsonl)
            metrics_add(&run->match_incumbent, incumbent.marginal, held,
                        n, need, incumbent_nll);
        if (opponent_actions > 0) {
            metrics_add(&run->post_incumbent, incumbent.marginal, held, n,
                        need, incumbent_nll);
            if (run->match_jsonl)
                metrics_add(&run->match_post_incumbent,
                            incumbent.marginal, held, n, need,
                            incumbent_nll);
        }
    }
    if (run->training && opponent_actions > 0 &&
        !train_state(run, trace, &view, observer,
                     prediction.base.card, held, n,
                     prediction.marginal))
        return 0;

    metrics_add(&run->current, prediction.base.marginal,
                held, n, need, current_nll);
    if (run->match_jsonl)
        metrics_add(&run->match_current, prediction.base.marginal, held,
                    n, need, current_nll);
    if (opponent_actions > 0) {
        metrics_add(&run->post_current, prediction.base.marginal,
                    held, n, need, current_nll);
        if (run->match_jsonl)
            metrics_add(&run->match_post_current, prediction.base.marginal,
                        held,
                        n, need, current_nll);
    }

    float prior[NCARD];
    float p = (float)need / (float)n;
    for (int i = 0; i < n; i++) prior[i] = p;
    double prior_nll = log_choose(n, need);
    if (!valid_marginals(prior, n, need) || !valid_nll(&prior_nll))
        return 0;
    metrics_add(&run->uniform, prior, held, n, need, prior_nll);
    if (run->match_jsonl)
        metrics_add(&run->match_uniform, prior, held, n, need, prior_nll);
    if (opponent_actions > 0) {
        metrics_add(&run->post_uniform, prior, held, n, need, prior_nll);
        if (run->match_jsonl)
            metrics_add(&run->match_post_uniform, prior, held,
                        n, need, prior_nll);
    }
    return 1;
}

static int actor_move(const Run *run, const State *complete, Rng *rng,
                      Move *chosen)
{
    State view;
    agent_information_view(complete, complete->turn, &view);
    Move move[MAX_MOVES];
    float probability[MAX_MOVES], weight[MAX_MOVES];
    int n = policy_probs_sym(run->actor, &view, move, probability,
                             NULL, run->symmetries);
    if (n <= 0) return 0;
    if (run->temperature == 0.0f) {
        int best = 0;
        for (int i = 0; i < n; i++)
            if (!(probability[i] >= 0.0f) ||
                !lc_float_isfinite(probability[i]))
                return 0;
        for (int i = 1; i < n; i++)
            if (probability[i] > probability[best]) best = i;
        *chosen = move[best];
        return 1;
    }
    double max_log_weight = -INFINITY;
    for (int i = 0; i < n; i++) {
        if (!(probability[i] >= 0.0f) ||
            !lc_float_isfinite(probability[i]))
            return 0;
        if (probability[i] > 0.0f) {
            double value = log((double)probability[i]) /
                           (double)run->temperature;
            if (value > max_log_weight) max_log_weight = value;
        }
    }
    if (!lc_double_isfinite(max_log_weight)) return 0;
    double total = 0.0;
    for (int i = 0; i < n; i++) {
        double value = probability[i] > 0.0f
                     ? exp(log((double)probability[i]) /
                           (double)run->temperature - max_log_weight)
                     : 0.0;
        if (!lc_double_isfinite(value)) return 0;
        weight[i] = (float)value;
        total += value;
    }
    if (!(total > 0.0) || !lc_double_isfinite(total)) return 0;
    int picked = sample_index(weight, n, rng);
    *chosen = move[picked];
    return 1;
}

static int run_matches(Run *run, long matches, int rounds, uint64_t seed,
                       uint64_t match_start)
{
    for (long match = 0; match < matches; match++) {
        uint64_t source_match_id = match_start + (uint64_t)match;
        run->source_match_id = source_match_id;
        memset(&run->match_history, 0, sizeof run->match_history);
        memset(&run->match_current, 0, sizeof run->match_current);
        memset(&run->match_uniform, 0, sizeof run->match_uniform);
        memset(&run->match_matched, 0, sizeof run->match_matched);
        memset(&run->match_incumbent, 0, sizeof run->match_incumbent);
        memset(&run->match_post_history, 0,
               sizeof run->match_post_history);
        memset(&run->match_post_current, 0,
               sizeof run->match_post_current);
        memset(&run->match_post_uniform, 0,
               sizeof run->match_post_uniform);
        memset(&run->match_post_matched, 0,
               sizeof run->match_post_matched);
        memset(&run->match_post_incumbent, 0,
               sizeof run->match_post_incumbent);
        run->match_excluded_states = 0;
        uint64_t match_rounds_completed = 0;
        uint64_t match_capped_rounds = 0;
        int cumulative[2] = { 0, 0 };
        for (int round = 0; round < rounds; round++) {
            uint64_t episode = source_match_id * (uint64_t)rounds
                             + (uint64_t)round;
            Rng deal_rng, behavior_rng;
            rng_seed(&deal_rng, mix64(seed ^ UINT64_C(0x243f6a8885a308d3)
                                      ^ episode));
            rng_seed(&behavior_rng,
                     mix64(seed ^ UINT64_C(0x13198a2e03707344)
                           ^ episode));
            State state;
            lc_deal(&state, &deal_rng);
            state.round = (uint8_t)round;
            state.cum[0] = (int16_t)cumulative[0];
            state.cum[1] = (int16_t)cumulative[1];
            state.turn = (uint8_t)(round & 1);
            HistoryBeliefTrace trace[2];
            history_belief_trace_init(&trace[0], 0);
            history_belief_trace_init(&trace[1], 1);

            while (!state.over) {
                if (state.nply <= run->max_scored_ply &&
                    !score_state(run, &state, &trace[state.turn]))
                    return 0;
                Move chosen;
                if (!actor_move(run, &state, &behavior_rng, &chosen) ||
                    !history_belief_trace_push(&trace[0], &state, chosen) ||
                    !history_belief_trace_push(&trace[1], &state, chosen))
                    return 0;
                lc_apply(&state, chosen);
            }
            /* Prefixes before an engine cap remain valid causal evidence.
             * Mark them explicitly and do not fabricate later rounds from
             * an incomplete cumulative score. */
            if (state.deck_left != 0) {
                run->capped_rounds++;
                match_capped_rounds++;
                break;
            }
            run->rounds_completed++;
            match_rounds_completed++;
            cumulative[0] += lc_score(&state, 0);
            cumulative[1] += lc_score(&state, 1);
        }
        if (run->match_jsonl) {
            if (!write_match_row(run, source_match_id,
                                 match_capped_rounds,
                                 match_rounds_completed,
                                 &run->match_history,
                                 &run->match_current,
                                 run->matched_net ? &run->match_matched : NULL,
                                 run->incumbent_alpha >= 0.0f
                                     ? &run->match_incumbent : NULL,
                                 &run->match_uniform,
                                 &run->match_post_history,
                                 &run->match_post_current,
                                 run->matched_net
                                     ? &run->match_post_matched : NULL,
                                 run->incumbent_alpha >= 0.0f
                                     ? &run->match_post_incumbent : NULL,
                                 &run->match_post_uniform))
                return 0;
        }
    }
    return 1;
}

static void print_metrics(const char *name, const Metrics *metric)
{
    double states = metric->states ? (double)metric->states : 1.0;
    double cards = metric->cards ? (double)metric->cards : 1.0;
    double positives = metric->positives ? (double)metric->positives : 1.0;
    printf("\"%s\":{\"nll_per_state\":%.12g,"
           "\"nll_per_card\":%.12g,\"brier\":%.12g,"
           "\"top_k_recall\":%.12g}",
           name, metric->nll / states, metric->nll / cards,
           metric->brier / cards, metric->top_hits / positives);
}

static void usage(const char *argv0)
{
    fprintf(stderr,
            "usage: %s train --out MODEL [options]\n"
            "       %s train-control --out NET --control-state-out STATE "
            "[options]\n"
            "       %s eval --model MODEL [options]\n"
            "options: --actor-net PATH [--base-net PATH] "
            "--matches N --rounds 1..3 "
            "--seed N --match-start N --max-ply 0..300\n"
            "         --symmetries 1|5|10|20|120 --temperature T "
            "--base-alpha A\n"
            "         --exclusions PATH --exclusions-sha256 HEX\n"
            "eval:    --match-jsonl PATH [--matched-base-net PATH "
            "--matched-base-alpha A] --incumbent-alpha A\n"
            "train:   --epochs N --lr X --l2 X\n"
            "control: [--control-state-in PATH] --control-state-out PATH "
            "[--control-batch-states N] [--control-finalize]\n",
            argv0, argv0, argv0);
}

int main(int argc, char **argv)
{
    if (argc < 2 || (strcmp(argv[1], "train") &&
                     strcmp(argv[1], "train-control") &&
                     strcmp(argv[1], "eval"))) {
        usage(argv[0]);
        return 2;
    }
    int training = !strcmp(argv[1], "train");
    int training_control = !strcmp(argv[1], "train-control");
    const char *actor_path = "data/champion.bin";
    const char *base_net_path = NULL;
    const char *model_path = NULL;
    const char *match_jsonl_path = NULL;
    const char *matched_net_path = NULL;
    const char *control_state_in_path = NULL;
    const char *control_state_out_path = NULL;
    const char *exclusions_path = NULL;
    const char *exclusions_sha256 = NULL;
    long matches = 100, rounds = MATCH_ROUNDS, epochs = 1;
    long symmetries = 1, max_scored_ply = LC_MAX_PLIES;
    long control_batch_size = 256;
    uint64_t seed = UINT64_C(202608290101);
    uint64_t match_start = 0;
    float temperature = 0.0f, base_alpha = HISTORY_BELIEF_BASE_ALPHA;
    float matched_alpha = -1.0f;
    float incumbent_alpha = -1.0f;
    float learning_rate = 0.03f, l2 = 1e-5f;
    int control_finalize = 0;

    for (int i = 2; i < argc; i++) {
        if (!strcmp(argv[i], "--actor-net") && i + 1 < argc)
            actor_path = argv[++i];
        else if (!strcmp(argv[i], "--base-net") && i + 1 < argc)
            base_net_path = argv[++i];
        else if (!strcmp(argv[i], "--matched-base-net") && i + 1 < argc)
            matched_net_path = argv[++i];
        else if (!strcmp(argv[i], "--control-state-in") && i + 1 < argc)
            control_state_in_path = argv[++i];
        else if (!strcmp(argv[i], "--control-state-out") && i + 1 < argc)
            control_state_out_path = argv[++i];
        else if ((!strcmp(argv[i], "--out") ||
                  !strcmp(argv[i], "--model")) && i + 1 < argc)
            model_path = argv[++i];
        else if (!strcmp(argv[i], "--matches") && i + 1 < argc) {
            if (!parse_long(argv[++i], 1, 1000000, &matches)) goto bad;
        } else if (!strcmp(argv[i], "--rounds") && i + 1 < argc) {
            if (!parse_long(argv[++i], 1, MATCH_ROUNDS, &rounds)) goto bad;
        } else if (!strcmp(argv[i], "--epochs") && i + 1 < argc) {
            if (!parse_long(argv[++i], 1, 100, &epochs)) goto bad;
        } else if (!strcmp(argv[i], "--symmetries") && i + 1 < argc) {
            if (!parse_long(argv[++i], 1, 120, &symmetries) ||
                (symmetries != 1 && symmetries != 5 && symmetries != 10 &&
                 symmetries != 20 && symmetries != 120))
                goto bad;
        } else if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
            if (!parse_u64(argv[++i], &seed)) goto bad;
        } else if (!strcmp(argv[i], "--match-start") && i + 1 < argc) {
            if (!parse_u64(argv[++i], &match_start)) goto bad;
        } else if (!strcmp(argv[i], "--max-ply") && i + 1 < argc) {
            if (!parse_long(argv[++i], 0, LC_MAX_PLIES,
                            &max_scored_ply))
                goto bad;
        } else if (!strcmp(argv[i], "--match-jsonl") && i + 1 < argc) {
            match_jsonl_path = argv[++i];
            if (!*match_jsonl_path) goto bad;
        } else if (!strcmp(argv[i], "--exclusions") && i + 1 < argc) {
            exclusions_path = argv[++i];
            if (!*exclusions_path) goto bad;
        } else if (!strcmp(argv[i], "--exclusions-sha256") &&
                   i + 1 < argc) {
            exclusions_sha256 = argv[++i];
            if (!*exclusions_sha256) goto bad;
        } else if (!strcmp(argv[i], "--temperature") && i + 1 < argc) {
            if (!parse_float(argv[++i], 0.0f, 5.0f, &temperature)) goto bad;
        } else if (!strcmp(argv[i], "--base-alpha") && i + 1 < argc) {
            if (!parse_float(argv[++i], 0.0f, 5.0f, &base_alpha)) goto bad;
        } else if (!strcmp(argv[i], "--matched-base-alpha") &&
                   i + 1 < argc) {
            if (!parse_float(argv[++i], 0.0f, 5.0f, &matched_alpha)) goto bad;
        } else if (!strcmp(argv[i], "--incumbent-alpha") &&
                   i + 1 < argc) {
            if (!parse_float(argv[++i], 0.0f, 5.0f, &incumbent_alpha))
                goto bad;
        } else if (!strcmp(argv[i], "--control-batch-states") &&
                   i + 1 < argc) {
            if (!parse_long(argv[++i], 1, 1000000,
                            &control_batch_size)) goto bad;
        } else if (!strcmp(argv[i], "--control-finalize")) {
            control_finalize = 1;
        } else if (!strcmp(argv[i], "--lr") && i + 1 < argc) {
            if (!parse_float(argv[++i], 1e-6f, 1.0f, &learning_rate)) goto bad;
        } else if (!strcmp(argv[i], "--l2") && i + 1 < argc) {
            if (!parse_float(argv[++i], 0.0f, 1.0f, &l2)) goto bad;
        } else goto bad;
    }
    if (!model_path || !*model_path ||
        ((!training && !training_control) && epochs != 1) ||
        ((training || training_control) && match_jsonl_path) ||
        !exclusions_path ||
        !exclusions_sha256)
        goto bad;
    if (strcmp(exclusions_sha256,
               HISTORY_BELIEF_EXACT17_CANONICAL_SHA256) != 0) {
        fprintf(stderr,
                "history_belief_train: noncanonical exact-17 digest\n");
        return 1;
    }
    if ((training || training_control) && matched_net_path) goto bad;
    if ((training || training_control) && incumbent_alpha >= 0.0f) goto bad;
    if (!training_control && (control_state_in_path ||
                              control_state_out_path || control_finalize ||
                              control_batch_size != 256))
        goto bad;
    if (training_control && (!control_state_out_path ||
                             !*control_state_out_path || epochs != 1))
        goto bad;
    if ((matched_net_path == NULL) != (matched_alpha < 0.0f)) goto bad;
    if (!training && !training_control && incumbent_alpha < 0.0f) goto bad;
    if (training_control && matched_alpha >= 0.0f) goto bad;
    if (match_start > UINT64_MAX - (uint64_t)(matches - 1) ||
        match_start + (uint64_t)(matches - 1) >
            (UINT64_MAX - (uint64_t)(rounds - 1)) / (uint64_t)rounds)
        goto bad;
    if (!base_net_path) base_net_path = actor_path;

    HistoryBeliefExclusions exclusions;
    if (!history_belief_exclusions_load(
            exclusions_path, exclusions_sha256, &exclusions)) {
        fprintf(stderr,
                "history_belief_train: exclusion binding failed\n");
        return 1;
    }

    Net *actor = malloc(sizeof *actor);
    Net *base_net = malloc(sizeof *base_net);
    Net *matched_net = matched_net_path ? malloc(sizeof *matched_net) : NULL;
    Net *control_net = training_control ? malloc(sizeof *control_net) : NULL;
    HistoryBeliefModel *model = training_control ? NULL : malloc(sizeof *model);
    ControlOptimizer *control_optimizer = training_control
        ? calloc(1, sizeof *control_optimizer) : NULL;
    Optimizer optimizer;
    memset(&optimizer, 0, sizeof optimizer);
    HistoryBeliefRuntime runtime;
    memset(&runtime, 0, sizeof runtime);
    Run run;
    memset(&run, 0, sizeof run);
    int result = 1;
    if (!actor || !base_net || (!training_control && !model) ||
        (training_control && (!control_net || !control_optimizer)) ||
        (matched_net_path && !matched_net) ||
        net_load(actor, actor_path) != 0 ||
        net_load(base_net, base_net_path) != 0 ||
        (matched_net && net_load(matched_net, matched_net_path) != 0)) {
        fprintf(stderr,
                "history_belief_train: cannot load frozen network\n");
        goto cleanup;
    }
    if (training_control) memcpy(control_net, base_net, sizeof *control_net);

    unsigned char input_checkpoint_digest[32] = {0};
    char input_checkpoint_sha256[65] = {0};
    if (training_control) {
        if (!file_sha256(base_net_path, input_checkpoint_digest)) {
            fprintf(stderr,
                    "history_belief_train: cannot hash control input checkpoint\n");
            goto cleanup;
        }
        digest_hex(input_checkpoint_digest, input_checkpoint_sha256);
    }

    if ((training || training_control) && access(model_path, F_OK) == 0) {
        fprintf(stderr, "history_belief_train: output already exists\n");
        goto cleanup;
    }
    if (training_control && access(control_state_out_path, F_OK) == 0) {
        fprintf(stderr,
                "history_belief_train: control state output already exists\n");
        goto cleanup;
    }

    if (training) {
        history_belief_model_init(model);
        if (!history_belief_model_bind(model, actor, base_net,
                                       (int)symmetries, temperature,
                                       base_alpha,
                                       exclusions.manifest_sha256)) {
            fprintf(stderr, "history_belief_train: cannot bind model\n");
            goto cleanup;
        }
    } else if (!training_control &&
               history_belief_model_load(model, model_path) != 0) {
        fprintf(stderr, "history_belief_train: cannot load model\n");
        goto cleanup;
    }
    if (!training_control && !history_belief_runtime_init(
            &runtime, model, actor, base_net, (int)symmetries,
            temperature, base_alpha, exclusions.manifest_sha256)) {
        fprintf(stderr,
                "history_belief_train: model/actor provenance mismatch\n");
        goto cleanup;
    }
    if (training && !optimizer_init(&optimizer)) {
        fprintf(stderr, "history_belief_train: optimizer allocation failed\n");
        goto cleanup;
    }
    uint64_t actor_fingerprint = history_belief_actor_fingerprint(actor);
    if (training_control && control_state_in_path &&
        !control_state_load(control_optimizer, control_net,
                            control_state_in_path, actor_fingerprint,
                            seed, match_start, (int)rounds,
                            (int)max_scored_ply, control_batch_size,
                            (int)symmetries,
                            temperature, base_alpha, learning_rate, l2,
                            exclusions.manifest_sha256,
                            input_checkpoint_digest)) {
        fprintf(stderr,
                "history_belief_train: control resume provenance mismatch\n");
        goto cleanup;
    }

    run = (Run){
        .actor = actor,
        .base_net = base_net,
        .control_net = control_net,
        .matched_net = matched_net,
        .runtime = training_control ? NULL : &runtime,
        .exclusions = &exclusions,
        .model = model,
        .optimizer = training ? &optimizer : NULL,
        .control_optimizer = control_optimizer,
        .training = training,
        .training_control = training_control,
        .symmetries = (int)symmetries,
        .temperature = temperature,
        .learning_rate = learning_rate,
        .l2 = l2,
        .matched_alpha = training_control ? base_alpha : matched_alpha,
        .incumbent_alpha = incumbent_alpha,
        .control_batch_size = control_batch_size,
        .max_scored_ply = (int)max_scored_ply,
        .seed = seed,
        .actor_fingerprint = actor_fingerprint,
        .base_net_fingerprint = history_belief_actor_fingerprint(base_net),
        .model_fingerprint = model
            ? history_belief_model_fingerprint(model) : 0,
        .matched_net_fingerprint = matched_net
            ? history_belief_actor_fingerprint(matched_net) : 0
    };
    sha256_init(&run.source_manifest);

    if (match_jsonl_path) {
        run.match_jsonl = fopen(match_jsonl_path, "wx");
        if (!run.match_jsonl) {
            fprintf(stderr,
                    "history_belief_train: cannot open match JSONL\n");
            goto cleanup;
        }
    }
    for (long epoch = 0; epoch < epochs; epoch++) {
        uint64_t epoch_seed = mix64(seed ^ (uint64_t)epoch *
                                   UINT64_C(0xd6e8feb86659fd93));
        if (!run_matches(&run, matches, (int)rounds, epoch_seed,
                         match_start)) {
            fprintf(stderr,
                    "history_belief_train: trajectory or evidence failure\n");
            goto cleanup;
        }
    }
    if (training_control && control_finalize && !control_apply_batch(&run)) {
        fprintf(stderr,
                "history_belief_train: cannot finalize control batch\n");
        goto cleanup;
    }
    if (run.history.states == 0 || run.history.cards == 0 ||
        (!training_control && !history_belief_model_validate(model)) ||
        (training_control &&
         !control_only_belief_head_changed(base_net, control_net))) {
        fprintf(stderr, "history_belief_train: no valid scored states\n");
        goto cleanup;
    }
    if (training && model->train_states != run.source_state_count) {
        fprintf(stderr,
                "history_belief_train: trained/source state mismatch\n");
        goto cleanup;
    }

    Sha256 source_copy = run.source_manifest;
    unsigned char source_digest[32];
    char source_manifest_sha256[65];
    sha256_final(&source_copy, source_digest);
    digest_hex(source_digest, source_manifest_sha256);

    unsigned char output_checkpoint_digest[32] = {0};
    char output_sha256[65] = {0}, control_state_sha256[65] = {0};
    if (training) {
        model->train_seed = seed;
        if (!publish_model_exclusive(model, model_path) ||
            !file_sha256_hex(model_path, output_sha256)) {
            fprintf(stderr, "history_belief_train: cannot save model\n");
            goto cleanup;
        }
    } else if (training_control) {
        if (!publish_net_exclusive(control_net, model_path) ||
            !file_sha256(model_path, output_checkpoint_digest)) {
            fprintf(stderr,
                    "history_belief_train: cannot save control checkpoint\n");
            goto cleanup;
        }
        digest_hex(output_checkpoint_digest, output_sha256);
        if (!control_state_save(
                control_optimizer, control_net, control_state_out_path,
                actor_fingerprint, seed,
                match_start + (uint64_t)matches, (int)rounds,
                (int)max_scored_ply, control_batch_size,
                (int)symmetries,
                temperature, base_alpha, learning_rate, l2,
                exclusions.manifest_sha256, source_digest,
                output_checkpoint_digest) ||
            !file_sha256_hex(control_state_out_path,
                             control_state_sha256)) {
            fprintf(stderr,
                    "history_belief_train: cannot save control artifacts\n");
            goto cleanup;
        }
    }

    char match_jsonl_sha256[65] = { 0 };
    if (run.match_jsonl) {
        if (fclose(run.match_jsonl) != 0 ||
            !file_sha256_hex(match_jsonl_path, match_jsonl_sha256)) {
            run.match_jsonl = NULL;
            fprintf(stderr,
                    "history_belief_train: cannot seal match JSONL\n");
            goto cleanup;
        }
        run.match_jsonl = NULL;
    }
    if (training_control) {
        printf("{\"schema\":\"lc-history-belief-control-run-v1\","
               "\"mode\":\"train-control\",\"seed\":%llu,"
               "\"matches\":%ld,\"match_start\":%llu,"
               "\"next_match_start\":%llu,\"rounds\":%ld,"
               "\"max_scored_ply\":%ld,\"symmetries\":%ld,"
               "\"temperature\":%.9g,\"base_alpha\":%.9g,"
               "\"lr\":%.9g,\"l2\":%.9g,"
               "\"control_batch_states\":%ld,"
               "\"control_finalized\":%s,"
               "\"pending_batch_states\":%llu,"
               "\"trained_state_count\":%llu,"
               "\"optimizer_steps\":%llu,"
               "\"source_state_count\":%llu,"
               "\"source_manifest_sha256\":\"%s\","
               "\"control_state_source_manifest_scope\":"
               "\"current_invocation\","
               "\"actor_fingerprint\":\"%016llx\","
               "\"input_net_fingerprint\":\"%016llx\","
               "\"input_checkpoint_sha256\":\"%s\","
               "\"output_net_fingerprint\":\"%016llx\","
               "\"output_sha256\":\"%s\","
               "\"control_state_checkpoint_sha256\":\"%s\","
               "\"control_state_sha256\":\"%s\","
               "\"exclusion_manifest_count\":%d,"
               "\"exclusion_manifest_sha256\":\"%s\","
               "\"excluded_state_count\":%llu,"
               "\"playing_actor_changed\":false,"
               "\"control_changed_only_belief_head\":true,"
               "\"training_augmentation\":"
               "\"one deterministic scheduled member per state from the "
               "declared suit group\"}\n",
               (unsigned long long)seed, matches,
               (unsigned long long)match_start,
               (unsigned long long)(match_start + (uint64_t)matches),
               rounds, max_scored_ply, symmetries, temperature, base_alpha,
               learning_rate, l2, control_batch_size,
               control_finalize ? "true" : "false",
               (unsigned long long)control_optimizer->batch_states,
               (unsigned long long)control_optimizer->trained_states,
               (unsigned long long)control_optimizer->optimizer_steps,
               (unsigned long long)run.source_state_count,
               source_manifest_sha256,
               (unsigned long long)actor_fingerprint,
               (unsigned long long)run.base_net_fingerprint,
               input_checkpoint_sha256,
               (unsigned long long)history_belief_actor_fingerprint(
                   control_net), output_sha256, output_sha256,
               control_state_sha256,
               exclusions.count, exclusions.manifest_sha256_hex,
               (unsigned long long)run.excluded_states);
    } else {
        printf("{\"schema\":\"lc-history-belief-run-v1\","
               "\"mode\":\"%s\",\"seed\":%llu,\"matches\":%ld,"
               "\"match_start\":%llu,\"max_scored_ply\":%ld,"
               "\"rounds\":%ld,\"epochs\":%ld,\"symmetries\":%ld,"
               "\"temperature\":%.9g,\"states\":%llu,"
               "\"source_state_count\":%llu,"
               "\"source_manifest_sha256\":\"%s\","
               "\"rounds_completed\":%llu,\"capped_rounds\":%llu,"
               "\"excluded_state_count\":%llu,"
               "\"exclusion_manifest_count\":%d,"
               "\"exclusion_manifest_sha256\":\"%s\","
               "\"uncertain_cards\":%llu,\"model_fingerprint\":"
               "\"%016llx\",\"actor_fingerprint\":\"%016llx\","
               "\"base_net_fingerprint\":\"%016llx\","
               "\"base_alpha\":%.9g,"
               "\"matched_base_net_fingerprint\":",
               training ? "train" : "eval", (unsigned long long)seed,
               matches, (unsigned long long)match_start, max_scored_ply,
               rounds, epochs, symmetries, temperature,
               (unsigned long long)run.history.states,
               (unsigned long long)run.source_state_count,
               source_manifest_sha256,
               (unsigned long long)run.rounds_completed,
               (unsigned long long)run.capped_rounds,
               (unsigned long long)run.excluded_states,
               exclusions.count, exclusions.manifest_sha256_hex,
               (unsigned long long)run.history.cards,
               (unsigned long long)history_belief_model_fingerprint(model),
               (unsigned long long)model->actor_fingerprint,
               (unsigned long long)model->base_net_fingerprint,
               model->base_alpha);
        if (matched_net)
            printf("\"%016llx\"",
                   (unsigned long long)run.matched_net_fingerprint);
        else printf("null");
        printf(",\"matched_base_alpha\":");
        if (matched_net) printf("%.9g", matched_alpha);
        else printf("null");
        printf(",\"incumbent_alpha\":");
        if (!training) printf("%.9g", incumbent_alpha);
        else printf("null");
        printf(",\"incumbent_net_fingerprint\":");
        if (!training)
            printf("\"%016llx\"", (unsigned long long)actor_fingerprint);
        else printf("null");
        printf(",\"training_learning_rate\":");
        if (training) printf("%.9g", learning_rate);
        else printf("null");
        printf(",\"training_l2\":");
        if (training) printf("%.9g", l2);
        else printf("null");
        printf(",\"model_train_states\":%llu",
               (unsigned long long)model->train_states);
        printf(",\"output_sha256\":");
        if (training) printf("\"%s\"", output_sha256);
        else printf("null");
        printf(",\"metrics\":{\"all_states\":{");
        print_metrics("history", &run.history);
        putchar(',');
        print_metrics("base_262k_head", &run.current);
        if (matched_net) {
            putchar(',');
            print_metrics("matched_head_control", &run.matched);
        }
        if (!training) {
            putchar(',');
            print_metrics("incumbent_head", &run.incumbent);
        }
        putchar(',');
        print_metrics("uniform_exact_k", &run.uniform);
        printf("},\"post_opponent_action\":{");
        print_metrics("history", &run.post_history);
        putchar(',');
        print_metrics("base_262k_head", &run.post_current);
        if (matched_net) {
            putchar(',');
            print_metrics("matched_head_control", &run.post_matched);
        }
        if (!training) {
            putchar(',');
            print_metrics("incumbent_head", &run.post_incumbent);
        }
        putchar(',');
        print_metrics("uniform_exact_k", &run.post_uniform);
        printf("}},\"match_jsonl_sha256\":");
        if (match_jsonl_path) printf("\"%s\"", match_jsonl_sha256);
        else printf("null");
        printf(",\"structural_contract\":{"
               "\"action_history_public_only\":true,"
               "\"current_view_truth_scrubbed\":true,"
               "\"opening_history_uniform\":true,"
               "\"playing_actor_changed\":false,"
               "\"public_transcript_complete\":true,"
               "\"residual_features_opponent_action_anchored\":true,"
               "\"reviewed_ply_orbit_exclusion_enabled\":true,"
               "\"reviewed_ply_inputs_used\":false,"
               "\"suit_equivariant_features\":true,"
               "\"truth_read_after_prediction\":true,"
               "\"wager_identity_collapsed\":true},"
               "\"playing_actor_changed\":false}\n");
    }
    result = 0;

cleanup:
    if (run.match_jsonl) fclose(run.match_jsonl);
    optimizer_free(&optimizer);
    free(control_optimizer);
    free(actor);
    free(base_net);
    free(matched_net);
    free(control_net);
    free(model);
    return result;

bad:
    usage(argv[0]);
    return 2;
}
