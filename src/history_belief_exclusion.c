#define _POSIX_C_SOURCE 200809L
/* history_belief_exclusion.c -- exact-17 reviewed-ply suit-orbit firewall.
 *
 * The semantic serialization and SHA-256 implementation below are a direct
 * extraction of the locked policy_cost_dataset hash-probe contract.  Keeping
 * this implementation independent of training labels makes the required
 * ordering mechanically possible: scrub -> hash -> reject -> label.
 */
#include "history_belief_exclusion.h"

#include "agent.h" /* suit_permutations declaration only */

#include <ctype.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/types.h>

#define SEMANTIC_VIEW_MAX 256
#define EXCLUSION_SCHEMA "lc-policy-cost-exclusions-v1\n"

typedef struct {
    uint32_t h[8];
    uint64_t bits;
    unsigned char block[64];
    size_t used;
} Sha256;

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
        uint32_t a = rotr32(w[i-15],7) ^ rotr32(w[i-15],18) ^
                     (w[i-15] >> 3);
        uint32_t b = rotr32(w[i-2],17) ^ rotr32(w[i-2],19) ^
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

static void sha_bytes(const void *data, size_t n, unsigned char out[32])
{
    Sha256 s;
    sha256_init(&s);
    sha256_update(&s, data, n);
    sha256_final(&s, out);
}

static int file_sha256(const char *path, unsigned char out[32])
{
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    Sha256 s;
    sha256_init(&s);
    unsigned char buffer[65536];
    size_t n;
    while ((n = fread(buffer, 1, sizeof buffer, file)) != 0)
        sha256_update(&s, buffer, n);
    int ok = !ferror(file) && fclose(file) == 0;
    if (!ok) return 0;
    sha256_final(&s, out);
    return 1;
}

static void digest_hex(const unsigned char digest[32], char out[65])
{
    static const char digit[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) {
        out[2*i] = digit[digest[i] >> 4];
        out[2*i+1] = digit[digest[i] & 15];
    }
    out[64] = '\0';
}

static int hex_digest(const char *text, unsigned char out[32])
{
    if (!text || strlen(text) != 64) return 0;
    for (int i = 0; i < 32; i++) {
        int hi = isdigit((unsigned char)text[2*i]) ? text[2*i]-'0' :
                 text[2*i]>='a'&&text[2*i]<='f' ? text[2*i]-'a'+10 : -1;
        int lo = isdigit((unsigned char)text[2*i+1]) ? text[2*i+1]-'0' :
                 text[2*i+1]>='a'&&text[2*i+1]<='f' ?
                     text[2*i+1]-'a'+10 : -1;
        if (hi < 0 || lo < 0) return 0;
        out[i] = (unsigned char)(hi*16+lo);
    }
    return 1;
}

typedef struct {
    unsigned char data[SEMANTIC_VIEW_MAX];
    size_t n;
    int ok;
} ByteWriter;

static void bw_u8(ByteWriter *w, unsigned value)
{
    if (!w->ok || w->n >= sizeof w->data) {
        w->ok = 0;
        return;
    }
    w->data[w->n++] = (unsigned char)value;
}

static void bw_u16(ByteWriter *w, unsigned value)
{
    bw_u8(w, value);
    bw_u8(w, value >> 8);
}

/* Byte-for-byte locked semantic view serialization from policy_cost_dataset.
 * It omits hidden deck order and collapses the three physical wager cards in
 * each suit to a count. */
static int semantic_view_bytes(const State *view,
                               unsigned char out[SEMANTIC_VIEW_MAX],
                               size_t *nout)
{
    ByteWriter w = { .n = 0, .ok = 1 };
    bw_u8(&w, 1);
    bw_u8(&w, view->turn);
    bw_u8(&w, view->round);
    bw_u16(&w, view->nply);
    bw_u8(&w, view->deck_left);
    bw_u8(&w, view->hand_n[0]);
    bw_u8(&w, view->hand_n[1]);
    bw_u16(&w, (uint16_t)view->cum[0]);
    bw_u16(&w, (uint16_t)view->cum[1]);
    const uint64_t *kind[3] = { view->hand, view->known, view->played };
    for (int k = 0; k < 3; k++)
        for (int p = 0; p < 2; p++)
            for (int suit = 0; suit < NSUIT; suit++) {
                int wagers = 0;
                unsigned ranks = 0;
                for (int rank = 0; rank < NRANK; rank++)
                    if (kind[k][p] &
                        (UINT64_C(1) << CARD_MAKE(suit,rank))) {
                        if (rank < WAGERS_PER_SUIT) wagers++;
                        else ranks |= 1u << (rank - WAGERS_PER_SUIT);
                    }
                bw_u8(&w, (unsigned)wagers);
                bw_u16(&w, ranks);
            }
    for (int p = 0; p < 2; p++)
        for (int suit = 0; suit < NSUIT; suit++) {
            bw_u8(&w, view->exp_wager[p][suit]);
            bw_u8(&w, view->exp_top[p][suit]);
            bw_u8(&w, view->exp_n[p][suit]);
            bw_u8(&w, view->exp_sum[p][suit]);
        }
    for (int suit = 0; suit < NSUIT; suit++) {
        bw_u8(&w, view->pile_n[suit]);
        for (int i = 0; i < view->pile_n[suit]; i++) {
            int card = view->pile[suit][i];
            bw_u8(&w, CARD_IS_WAGER(card) ? 0 : CARD_RANK(card));
        }
    }
    if (!w.ok) return 0;
    memcpy(out, w.data, w.n);
    *nout = w.n;
    return 1;
}

/* Construct the hashable information view without ever copying or reading the
 * hidden deck order, deck position, or opponent private hand.  Keep this
 * projector local to the firewall so a later broadening of the playing
 * actor's information-view helper cannot weaken pre-truth exclusion. */
static int project_information_view(const State *complete, int observer,
                                    State *view)
{
    if (!complete || !view || observer < 0 || observer > 1 ||
        observer != complete->turn)
        return 0;

    /* Build in a separate zeroed object so the helper stays alias-safe without
     * ever taking a whole-State copy of complete.  Any future State member is
     * excluded until it is explicitly classified and assigned here. */
    State projected = {0};
    int opponent = observer ^ 1;
    projected.deck_left = complete->deck_left;
    projected.hand[observer] = complete->hand[observer];
    projected.hand[opponent] = complete->known[opponent];
    for (int player = 0; player < 2; player++) {
        projected.hand_n[player] = complete->hand_n[player];
        projected.played[player] = complete->played[player];
        projected.known[player] = complete->known[player];
        for (int suit = 0; suit < NSUIT; suit++) {
            projected.exp_wager[player][suit] =
                complete->exp_wager[player][suit];
            projected.exp_top[player][suit] =
                complete->exp_top[player][suit];
            projected.exp_n[player][suit] = complete->exp_n[player][suit];
            projected.exp_sum[player][suit] =
                complete->exp_sum[player][suit];
        }
    }
    projected.discarded = complete->discarded;
    for (int suit = 0; suit < NSUIT; suit++) {
        if (complete->pile_n[suit] > NRANK) return 0;
        projected.pile_n[suit] = complete->pile_n[suit];
        /* The complete pile backing array is public.  Preserve inactive tail
         * bytes as well as active cards so the projected State remains
         * byte-compatible with the existing actor information view. */
        for (int index = 0; index < NRANK; index++)
            projected.pile[suit][index] = complete->pile[suit][index];
    }
    projected.turn = complete->turn;
    projected.over = complete->over;
    projected.nply = complete->nply;
    projected.round = complete->round;
    projected.cum[0] = complete->cum[0];
    projected.cum[1] = complete->cum[1];
    *view = projected;
    return 1;
}

/* This validation duplicates the locked hash-probe's native information
 * boundary.  In particular, opponent private cards and deck bytes must have
 * been removed before semantic serialization can run. */
static int information_view_valid(const State *view)
{
    if (!view || view->turn > 1 || view->round >= MATCH_ROUNDS || view->over ||
        view->deck_left > NCARD || view->deck_pos != 0 ||
        view->nply >= LC_MAX_PLIES)
        return 0;
    for (int i = 0; i < NCARD; i++)
        if (view->deck[i] != 0) return 0;
    int player = view->turn;
    int opponent = player ^ 1;
    if (__builtin_popcountll(view->hand[player]) != view->hand_n[player] ||
        __builtin_popcountll(view->hand[opponent]) >
            view->hand_n[opponent] ||
        view->hand[opponent] != view->known[opponent] ||
        (view->known[player] & ~view->hand[player]))
        return 0;
    uint64_t piles = 0;
    for (int suit = 0; suit < NSUIT; suit++) {
        if (view->pile_n[suit] > NRANK) return 0;
        for (int i = 0; i < view->pile_n[suit]; i++) {
            int card = view->pile[suit][i];
            if (card >= NCARD || CARD_SUIT(card) != suit ||
                (piles & (UINT64_C(1) << card)))
                return 0;
            piles |= UINT64_C(1) << card;
        }
    }
    if (piles != view->discarded) return 0;
    State again;
    unsigned char a[SEMANTIC_VIEW_MAX], b[SEMANTIC_VIEW_MAX];
    size_t na = 0, nb = 0;
    return project_information_view(view, view->turn, &again) &&
           semantic_view_bytes(view, a, &na) &&
           semantic_view_bytes(&again, b, &nb) && na == nb &&
           memcmp(a, b, na) == 0 &&
           lc_moves(view, (Move[MAX_MOVES]){{0}}) > 0;
}

static int orbit_from_view(const State *view, unsigned char digest[32])
{
    uint8_t permutation[120][NSUIT];
    if (suit_permutations(120, permutation) != 120) return 0;
    unsigned char best[SEMANTIC_VIEW_MAX], current[SEMANTIC_VIEW_MAX];
    size_t nbest = 0, ncurrent = 0;
    for (int i = 0; i < 120; i++) {
        State permuted;
        lc_permute_suits(view, &permuted, permutation[i]);
        if (!semantic_view_bytes(&permuted, current, &ncurrent)) return 0;
        if (i == 0 || ncurrent < nbest ||
            (ncurrent == nbest && memcmp(current, best, ncurrent) < 0)) {
            memcpy(best, current, ncurrent);
            nbest = ncurrent;
        }
    }
    sha_bytes(best, nbest, digest);
    return 1;
}

int history_belief_exclusions_load(const char *path,
                                   const char expected_sha256_hex[65],
                                   HistoryBeliefExclusions *out)
{
    if (!out) return 0;
    memset(out, 0, sizeof *out);
    unsigned char expected[32], actual[32];
    if (!path || !expected_sha256_hex ||
        !hex_digest(expected_sha256_hex, expected) ||
        !file_sha256(path, actual) || memcmp(expected, actual, 32) != 0)
        return 0;
    FILE *file = fopen(path, "rb");
    if (!file) return 0;
    char *line = NULL;
    size_t capacity = 0;
    ssize_t length = getline(&line, &capacity, file);
    if (length < 0 || strcmp(line, EXCLUSION_SCHEMA) != 0) goto invalid;
    while ((length = getline(&line, &capacity, file)) >= 0) {
        if (length != 65 || line[64] != '\n' ||
            out->count >= HISTORY_BELIEF_EXCLUSION_COUNT)
            goto invalid;
        line[64] = '\0';
        if (!hex_digest(line, out->orbit[out->count])) goto invalid;
        for (int i = 0; i < out->count; i++)
            if (memcmp(out->orbit[i], out->orbit[out->count], 32) == 0)
                goto invalid;
        out->count++;
    }
    int read_error = ferror(file);
    int close_error = fclose(file) != 0;
    file = NULL;
    if (read_error || close_error ||
        out->count != HISTORY_BELIEF_EXCLUSION_COUNT) {
        goto invalid;
    }
    memcpy(out->manifest_sha256, actual, 32);
    digest_hex(actual, out->manifest_sha256_hex);
    free(line);
    return 1;

invalid:
    free(line);
    if (file) fclose(file);
    memset(out, 0, sizeof *out);
    return 0;
}

int history_belief_exclusion_orbit(const State *complete, int observer,
                                   unsigned char out[32])
{
    if (!complete || !out || observer < 0 || observer > 1 ||
        observer != complete->turn)
        return 0;
    State view;
    if (!project_information_view(complete, observer, &view)) return 0;
    if (!information_view_valid(&view)) return 0;
    return orbit_from_view(&view, out);
}

int history_belief_exclusions_contains(const HistoryBeliefExclusions *set,
                                       const unsigned char orbit[32])
{
    if (!set || !orbit || set->count != HISTORY_BELIEF_EXCLUSION_COUNT)
        return 0;
    for (int i = 0; i < set->count; i++)
        if (memcmp(set->orbit[i], orbit, 32) == 0) return 1;
    return 0;
}

int history_belief_exclusions_check(const HistoryBeliefExclusions *set,
                                    const State *complete, int observer,
                                    unsigned char orbit_out[32])
{
    if (!set || set->count != HISTORY_BELIEF_EXCLUSION_COUNT) return -1;
    unsigned char orbit[32];
    if (!history_belief_exclusion_orbit(complete, observer, orbit)) return -1;
    if (orbit_out) memcpy(orbit_out, orbit, 32);
    return history_belief_exclusions_contains(set, orbit) ? 1 : 0;
}
