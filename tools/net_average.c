/* net_average -- deterministic equal-weight averaging of Net checkpoints. */
#define _XOPEN_SOURCE 700

#include "../src/net.h"
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

typedef struct {
    char canonical[PATH_MAX];
    dev_t device;
    ino_t inode;
    off_t size;
    uint64_t file_fingerprint;
    uint64_t model_fingerprint;
    unsigned char file_sha256[32];
    Net *model;
} Input;

typedef struct {
    char canonical[PATH_MAX];
    dev_t device;
    ino_t inode;
    int exists;
} OutputIdentity;

_Static_assert(sizeof(Net) % sizeof(float) == 0,
               "Net must be a contiguous array of floats");

typedef struct {
    uint32_t state[8];
    uint64_t bit_count;
    unsigned char block[64];
    size_t used;
} Sha256;

static uint32_t rotate_right(uint32_t x, unsigned int n)
{
    return (x >> n) | (x << (32U - n));
}

static void sha256_transform(Sha256 *ctx, const unsigned char block[64])
{
    static const uint32_t constant[64] = {
        UINT32_C(0x428a2f98), UINT32_C(0x71374491), UINT32_C(0xb5c0fbcf),
        UINT32_C(0xe9b5dba5), UINT32_C(0x3956c25b), UINT32_C(0x59f111f1),
        UINT32_C(0x923f82a4), UINT32_C(0xab1c5ed5), UINT32_C(0xd807aa98),
        UINT32_C(0x12835b01), UINT32_C(0x243185be), UINT32_C(0x550c7dc3),
        UINT32_C(0x72be5d74), UINT32_C(0x80deb1fe), UINT32_C(0x9bdc06a7),
        UINT32_C(0xc19bf174), UINT32_C(0xe49b69c1), UINT32_C(0xefbe4786),
        UINT32_C(0x0fc19dc6), UINT32_C(0x240ca1cc), UINT32_C(0x2de92c6f),
        UINT32_C(0x4a7484aa), UINT32_C(0x5cb0a9dc), UINT32_C(0x76f988da),
        UINT32_C(0x983e5152), UINT32_C(0xa831c66d), UINT32_C(0xb00327c8),
        UINT32_C(0xbf597fc7), UINT32_C(0xc6e00bf3), UINT32_C(0xd5a79147),
        UINT32_C(0x06ca6351), UINT32_C(0x14292967), UINT32_C(0x27b70a85),
        UINT32_C(0x2e1b2138), UINT32_C(0x4d2c6dfc), UINT32_C(0x53380d13),
        UINT32_C(0x650a7354), UINT32_C(0x766a0abb), UINT32_C(0x81c2c92e),
        UINT32_C(0x92722c85), UINT32_C(0xa2bfe8a1), UINT32_C(0xa81a664b),
        UINT32_C(0xc24b8b70), UINT32_C(0xc76c51a3), UINT32_C(0xd192e819),
        UINT32_C(0xd6990624), UINT32_C(0xf40e3585), UINT32_C(0x106aa070),
        UINT32_C(0x19a4c116), UINT32_C(0x1e376c08), UINT32_C(0x2748774c),
        UINT32_C(0x34b0bcb5), UINT32_C(0x391c0cb3), UINT32_C(0x4ed8aa4a),
        UINT32_C(0x5b9cca4f), UINT32_C(0x682e6ff3), UINT32_C(0x748f82ee),
        UINT32_C(0x78a5636f), UINT32_C(0x84c87814), UINT32_C(0x8cc70208),
        UINT32_C(0x90befffa), UINT32_C(0xa4506ceb), UINT32_C(0xbef9a3f7),
        UINT32_C(0xc67178f2)
    };
    uint32_t w[64];
    for (int i = 0; i < 16; i++) {
        const unsigned char *p = block + 4 * i;
        w[i] = ((uint32_t)p[0] << 24) | ((uint32_t)p[1] << 16) |
               ((uint32_t)p[2] << 8) | (uint32_t)p[3];
    }
    for (int i = 16; i < 64; i++) {
        uint32_t s0 = rotate_right(w[i - 15], 7) ^
                      rotate_right(w[i - 15], 18) ^ (w[i - 15] >> 3);
        uint32_t s1 = rotate_right(w[i - 2], 17) ^
                      rotate_right(w[i - 2], 19) ^ (w[i - 2] >> 10);
        w[i] = w[i - 16] + s0 + w[i - 7] + s1;
    }
    uint32_t a = ctx->state[0], b = ctx->state[1];
    uint32_t c = ctx->state[2], d = ctx->state[3];
    uint32_t e = ctx->state[4], f = ctx->state[5];
    uint32_t g = ctx->state[6], h = ctx->state[7];
    for (int i = 0; i < 64; i++) {
        uint32_t s1 = rotate_right(e, 6) ^ rotate_right(e, 11) ^
                      rotate_right(e, 25);
        uint32_t choose = (e & f) ^ (~e & g);
        uint32_t t1 = h + s1 + choose + constant[i] + w[i];
        uint32_t s0 = rotate_right(a, 2) ^ rotate_right(a, 13) ^
                      rotate_right(a, 22);
        uint32_t majority = (a & b) ^ (a & c) ^ (b & c);
        uint32_t t2 = s0 + majority;
        h = g; g = f; f = e; e = d + t1;
        d = c; c = b; b = a; a = t1 + t2;
    }
    ctx->state[0] += a; ctx->state[1] += b;
    ctx->state[2] += c; ctx->state[3] += d;
    ctx->state[4] += e; ctx->state[5] += f;
    ctx->state[6] += g; ctx->state[7] += h;
}

static void sha256_init(Sha256 *ctx)
{
    static const uint32_t initial[8] = {
        UINT32_C(0x6a09e667), UINT32_C(0xbb67ae85),
        UINT32_C(0x3c6ef372), UINT32_C(0xa54ff53a),
        UINT32_C(0x510e527f), UINT32_C(0x9b05688c),
        UINT32_C(0x1f83d9ab), UINT32_C(0x5be0cd19)
    };
    memcpy(ctx->state, initial, sizeof initial);
    ctx->bit_count = 0;
    ctx->used = 0;
}

static void sha256_update(Sha256 *ctx, const unsigned char *data, size_t size)
{
    ctx->bit_count += (uint64_t)size * UINT64_C(8);
    while (size > 0) {
        size_t space = sizeof ctx->block - ctx->used;
        size_t take = size < space ? size : space;
        memcpy(ctx->block + ctx->used, data, take);
        ctx->used += take;
        data += take;
        size -= take;
        if (ctx->used == sizeof ctx->block) {
            sha256_transform(ctx, ctx->block);
            ctx->used = 0;
        }
    }
}

static void sha256_final(Sha256 *ctx, unsigned char digest[32])
{
    uint64_t length = ctx->bit_count;
    ctx->block[ctx->used++] = 0x80;
    if (ctx->used > 56) {
        memset(ctx->block + ctx->used, 0, sizeof ctx->block - ctx->used);
        sha256_transform(ctx, ctx->block);
        ctx->used = 0;
    }
    memset(ctx->block + ctx->used, 0, 56 - ctx->used);
    for (int i = 0; i < 8; i++)
        ctx->block[56 + i] = (unsigned char)(length >> (56 - 8 * i));
    sha256_transform(ctx, ctx->block);
    for (int i = 0; i < 8; i++) {
        digest[4 * i] = (unsigned char)(ctx->state[i] >> 24);
        digest[4 * i + 1] = (unsigned char)(ctx->state[i] >> 16);
        digest[4 * i + 2] = (unsigned char)(ctx->state[i] >> 8);
        digest[4 * i + 3] = (unsigned char)ctx->state[i];
    }
}

static uint64_t fingerprint_bytes(const void *data, size_t size)
{
    const unsigned char *p = (const unsigned char *)data;
    uint64_t h = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < size; i++) {
        h ^= p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static int fingerprint_file(const char *path, uint64_t *fingerprint,
                            uint64_t *size_out,
                            unsigned char sha256[32])
{
    FILE *fp = fopen(path, "rb");
    if (!fp) return 0;
    uint64_t h = UINT64_C(1469598103934665603), size = 0;
    Sha256 sha;
    sha256_init(&sha);
    unsigned char block[16384];
    for (;;) {
        size_t n = fread(block, 1, sizeof block, fp);
        for (size_t i = 0; i < n; i++) {
            h ^= block[i];
            h *= UINT64_C(1099511628211);
        }
        sha256_update(&sha, block, n);
        size += (uint64_t)n;
        if (n < sizeof block) {
            if (ferror(fp)) {
                fclose(fp);
                return 0;
            }
            break;
        }
    }
    if (fclose(fp) != 0) return 0;
    *fingerprint = h;
    *size_out = size;
    sha256_final(&sha, sha256);
    return 1;
}

static int input_identity(const char *path, Input *input)
{
    struct stat sb;
    memset(input, 0, sizeof *input);
    if (!realpath(path, input->canonical) ||
        stat(input->canonical, &sb) != 0 || !S_ISREG(sb.st_mode))
        return 0;
    input->device = sb.st_dev;
    input->inode = sb.st_ino;
    input->size = sb.st_size;
    return 1;
}

static int output_identity(const char *path, OutputIdentity *output)
{
    struct stat sb;
    memset(output, 0, sizeof *output);
    if (realpath(path, output->canonical)) {
        if (stat(output->canonical, &sb) != 0 || !S_ISREG(sb.st_mode))
            return 0;
        output->device = sb.st_dev;
        output->inode = sb.st_ino;
        output->exists = 1;
        return 1;
    }
    if (errno != ENOENT && errno != ENOTDIR) return 0;

    size_t len = strlen(path);
    if (len == 0 || len >= PATH_MAX || path[len - 1] == '/') return 0;
    const char *base = path;
    const char *slash = strrchr(path, '/');
    char parent[PATH_MAX], resolved_parent[PATH_MAX];
    if (!slash) {
        memcpy(parent, ".", 2);
    } else {
        base = slash + 1;
        size_t parent_len = slash == path ? 1 : (size_t)(slash - path);
        if (parent_len >= sizeof parent) return 0;
        memcpy(parent, path, parent_len);
        parent[parent_len] = '\0';
    }
    if (!*base || !strcmp(base, ".") || !strcmp(base, "..") ||
        !realpath(parent, resolved_parent))
        return 0;
    int written = snprintf(output->canonical, sizeof output->canonical,
                           "%s/%s", resolved_parent, base);
    return written >= 0 && (size_t)written < sizeof output->canonical;
}

static int input_compare(const void *av, const void *bv)
{
    const Input *a = (const Input *)av;
    const Input *b = (const Input *)bv;
    if (a->model_fingerprint < b->model_fingerprint) return -1;
    if (a->model_fingerprint > b->model_fingerprint) return 1;
    int model_order = memcmp(a->model, b->model, sizeof *a->model);
    if (model_order != 0) return model_order;
    return strcmp(a->canonical, b->canonical);
}

static void free_inputs(Input *input, int count)
{
    if (!input) return;
    for (int i = 0; i < count; i++) free(input[i].model);
    free(input);
}

static void print_sha256(const unsigned char digest[32])
{
    for (int i = 0; i < 32; i++) printf("%02x", digest[i]);
}

static int output_aliases_input(const OutputIdentity *output,
                                const Input *input)
{
    if (!strcmp(output->canonical, input->canonical)) return 1;
    return output->exists && output->device == input->device &&
           output->inode == input->inode;
}

static int finite_model(const Net *net, size_t *bad_parameter)
{
    const float *value = (const float *)net;
    const size_t count = sizeof *net / sizeof *value;
    for (size_t i = 0; i < count; i++) {
        if (!lc_float_isfinite(value[i])) {
            if (bad_parameter) *bad_parameter = i;
            return 0;
        }
    }
    return 1;
}

static int atomic_net_save(const Net *net, const char *output_path)
{
    char temporary[PATH_MAX];
    int written = snprintf(temporary, sizeof temporary, "%s.tmp.XXXXXX",
                           output_path);
    if (written < 0 || (size_t)written >= sizeof temporary) return 0;
    int temporary_fd = mkstemp(temporary);
    if (temporary_fd < 0) return 0;
    if (close(temporary_fd) != 0) {
        unlink(temporary);
        return 0;
    }
    if (net_save(net, temporary) != 0) {
        unlink(temporary);
        return 0;
    }
    temporary_fd = open(temporary, O_RDONLY);
    int durable = temporary_fd >= 0;
    if (durable && fsync(temporary_fd) != 0) durable = 0;
    if (temporary_fd >= 0 && close(temporary_fd) != 0) durable = 0;
    if (!durable) {
        unlink(temporary);
        return 0;
    }
    if (rename(temporary, output_path) != 0) {
        unlink(temporary);
        return 0;
    }
    return 1;
}

int main(int argc, char **argv)
{
    if (argc < 4) {
        fprintf(stderr,
                "usage: %s OUTPUT.bin INPUT1.bin INPUT2.bin [INPUT.bin ...]\n",
                argv[0]);
        return 1;
    }
    const char *output_path = argv[1];
    const int input_count = argc - 2;
    Input *input = (Input *)calloc((size_t)input_count, sizeof *input);
    if (!input) {
        fprintf(stderr, "out of memory\n");
        return 1;
    }
    for (int i = 0; i < input_count; i++) {
        if (!input_identity(argv[i + 2], &input[i])) {
            fprintf(stderr, "cannot resolve regular input checkpoint %s\n",
                    argv[i + 2]);
            free_inputs(input, input_count);
            return 1;
        }
    }

    OutputIdentity output_identity_value;
    if (!output_identity(output_path, &output_identity_value)) {
        fprintf(stderr, "cannot resolve output checkpoint %s\n", output_path);
        free_inputs(input, input_count);
        return 1;
    }
    for (int i = 0; i < input_count; i++) {
        if (output_aliases_input(&output_identity_value, &input[i])) {
            fprintf(stderr,
                    "output checkpoint %s aliases input checkpoint %s\n",
                    output_path, input[i].canonical);
            free_inputs(input, input_count);
            return 1;
        }
    }

    Net *average = (Net *)malloc(sizeof *average);
    const size_t parameter_count = sizeof(Net) / sizeof(float);
    if (!average) {
        fprintf(stderr, "out of memory\n");
        free_inputs(input, input_count);
        return 1;
    }

    for (int i = 0; i < input_count; i++) {
        input[i].model = (Net *)malloc(sizeof *input[i].model);
        if (!input[i].model) {
            fprintf(stderr, "out of memory\n");
            free(average);
            free_inputs(input, input_count);
            return 1;
        }
        if (net_load(input[i].model, input[i].canonical) != 0) {
            fprintf(stderr, "malformed checkpoint %s\n", input[i].canonical);
            free(average);
            free_inputs(input, input_count);
            return 1;
        }
        size_t bad = 0;
        if (!finite_model(input[i].model, &bad)) {
            fprintf(stderr,
                    "non-finite parameter %zu in checkpoint %s\n",
                    bad, input[i].canonical);
            free(average);
            free_inputs(input, input_count);
            return 1;
        }
        uint64_t raw_size = 0;
        if (!fingerprint_file(input[i].canonical,
                              &input[i].file_fingerprint, &raw_size,
                              input[i].file_sha256) ||
            raw_size != (uint64_t)input[i].size) {
            fprintf(stderr, "cannot fingerprint checkpoint %s\n",
                    input[i].canonical);
            free(average);
            free_inputs(input, input_count);
            return 1;
        }
        input[i].model_fingerprint =
            fingerprint_bytes(input[i].model, sizeof *input[i].model);
    }
    /* FNV is only a fast primary ordering key.  Full model bytes break every
     * collision, and canonical paths break ties between identical models.
     * Therefore argument order and path spelling cannot perturb arithmetic. */
    qsort(input, (size_t)input_count, sizeof *input, input_compare);

    float *averaged_value = (float *)average;
    const double divisor = (double)input_count;
    size_t exact_parameters = 0;
    for (size_t i = 0; i < parameter_count; i++) {
        const float first_value = ((const float *)input[0].model)[i];
        uint32_t first_bits;
        memcpy(&first_bits, &first_value, sizeof first_bits);
        double sum = (double)first_value;
        int exact = 1;
        for (int model = 1; model < input_count; model++) {
            float next = ((const float *)input[model].model)[i];
            uint32_t next_bits;
            memcpy(&next_bits, &next, sizeof next_bits);
            if (next_bits != first_bits) exact = 0;
            sum += (double)next;
        }
        if (exact) {
            memcpy(&averaged_value[i], &first_bits, sizeof first_bits);
            exact_parameters++;
        } else {
            double mean = sum / divisor;
            averaged_value[i] = (float)mean;
            if (!lc_double_isfinite(mean) ||
                !lc_float_isfinite(averaged_value[i])) {
                fprintf(stderr,
                        "non-finite averaged parameter %zu; output not written\n",
                        i);
                free(average);
                free_inputs(input, input_count);
                return 1;
            }
        }
    }
    int identical = exact_parameters == parameter_count;

    /* Re-resolve immediately before installation to close the ordinary window
     * in which an existing output could be replaced by an input alias. */
    OutputIdentity final_output_identity;
    if (!output_identity(output_identity_value.canonical,
                         &final_output_identity)) {
        fprintf(stderr, "cannot re-resolve output checkpoint %s\n",
                output_identity_value.canonical);
        free(average);
        free_inputs(input, input_count);
        return 1;
    }
    for (int i = 0; i < input_count; i++) {
        if (output_aliases_input(&final_output_identity, &input[i])) {
            fprintf(stderr,
                    "output checkpoint %s aliases input checkpoint %s\n",
                    output_identity_value.canonical, input[i].canonical);
            free(average);
            free_inputs(input, input_count);
            return 1;
        }
    }
    if (!atomic_net_save(average, output_identity_value.canonical)) {
        fprintf(stderr, "cannot save output checkpoint %s\n", output_path);
        free(average);
        free_inputs(input, input_count);
        return 1;
    }
    uint64_t output_file_fingerprint = 0, output_size = 0;
    unsigned char output_sha256[32];
    if (!fingerprint_file(output_identity_value.canonical,
                          &output_file_fingerprint, &output_size,
                          output_sha256)) {
        fprintf(stderr, "cannot fingerprint output checkpoint %s\n",
                output_path);
        free(average);
        free_inputs(input, input_count);
        return 1;
    }
    uint64_t output_model_fingerprint =
        fingerprint_bytes(average, sizeof *average);

    printf("net_average provenance: equal_weight=%d parameters=%zu "
           "ordering=model_fnv1a_then_bytes_then_path "
           "accumulation=binary64_sequential rounding=binary32 "
           "exact_shared_parameters=%zu identical_fast_path=%d\n",
           input_count, parameter_count, exact_parameters, identical);
    for (int i = 0; i < input_count; i++) {
        printf("input[%d]=%s size=%jd file_fnv1a=%016" PRIx64
               " sha256=",
               i, input[i].canonical, (intmax_t)input[i].size,
               input[i].file_fingerprint);
        print_sha256(input[i].file_sha256);
        printf(" model_fnv1a=%016" PRIx64 "\n",
               input[i].model_fingerprint);
    }
    printf("output=%s size=%" PRIu64 " file_fnv1a=%016" PRIx64
           " sha256=",
           output_identity_value.canonical, output_size,
           output_file_fingerprint);
    print_sha256(output_sha256);
    printf(" model_fnv1a=%016" PRIx64 "\n", output_model_fingerprint);

    free(average);
    free_inputs(input, input_count);
    return 0;
}
