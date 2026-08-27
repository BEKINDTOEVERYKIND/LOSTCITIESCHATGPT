#define _POSIX_C_SOURCE 200809L
/* policy_cost_dataset -- immutable offline evidence for policy-frequency cost.
 *
 * There are deliberately only two campaign data-bearing modes:
 *
 *   discover  Generate exact policy-20 self-play matches, sanitize every
 *             root, reject a bound set of commented information states, and
 *             write the complete legal policy plus exact 1%/2% runtime
 *             shortlists.  There is no state/probe import path.
 *
 *   evaluate  Read an already selected, content-bound allocation manifest.
 *             Evaluate each exact runtime mask separately on common P800 and
 *             independent F800 worlds, and evaluate their (untruncated)
 *             union with an independent exact-policy-20 truth continuation.
 *
 *   hash-probe  A deliberately non-campaign, no-output helper used only while
 *             sealing the exact-17 exclusion list.  It reads one historical
 *             text state, immediately projects it to the mover's information
 *             view, and prints its orbit digest.  It cannot discover, label,
 *             allocate, or evaluate a state.
 *             TRAIN uses T512; SELECT and TEST use T1024.
 *
 * The tool intentionally does not implement quota filling, threshold fitting,
 * or candidate selection.  Those are campaign-layer decisions.  It emits
 * deterministic JSONL and a strict state reservoir that are sufficient for an
 * independent verifier to reconstruct first and second moments.  Outputs are
 * installed atomically with link(2), so an existing evidence path is never
 * replaced.
 */
#include "../src/agent.h"
#include "../src/match_value.h"
#include "../src/policy_cost.h"
#include "../src/search.h"
#include "../src/spec.h"

#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <strings.h>
#include <sys/stat.h>
#include <unistd.h>

#define PC_SYMMETRIES 20
#define PC_PRIMARY_WORLDS 800
#define PC_FRESH_WORLDS 800
#define PC_TRAIN_TRUTH_WORLDS 512
#define PC_HOLDOUT_TRUTH_WORLDS 1024
#define PC_MASKS 2
#define PC_MASK_MAX 5
#define PC_UNION_MAX 5
#define PC_TRUTH_MAX 6
#define PC_PLY_BINS 24
#define PC_RATIO_BINS 6
#define PC_PAIR_TYPES 2
#define PC_TRAIN_QUOTA 16
#define PC_VECTOR_QUOTA 64
#define PC_EXCLUSION_COUNT 17
#define PC_STATE_MAX 256
#define PC_TOKEN_MAX 160

#define TAG_DEAL UINT64_C(0x4445414c5f504332)
#define TAG_PRIMARY UINT64_C(0x5052494d41525932)
#define TAG_FRESH UINT64_C(0x46524553485f5032)
#define TAG_TRUTH_WORLD UINT64_C(0x5452555448574c44)
#define TAG_TRUTH_FUTURE UINT64_C(0x5452555448465554)
#define TAG_MAINTAINED UINT64_C(0x4d41494e5441494e)

typedef enum { SPLIT_TRAIN = 0, SPLIT_SELECT = 1, SPLIT_TEST = 2 } Split;

typedef struct {
    const char *name;
    uint64_t discovery_seed;
    uint64_t primary_seed;
    uint64_t fresh_seed;
    uint64_t truth_seed;
    uint64_t maintained_seed;
    int truth_worlds;
} SplitDomain;

static const SplitDomain SPLIT_DOMAIN[3] = {
    { "TRAIN",  UINT64_C(202612100101), UINT64_C(202612110101),
      UINT64_C(202612120101), UINT64_C(202612130101),
      UINT64_C(202612160101),
      PC_TRAIN_TRUTH_WORLDS },
    { "SELECT", UINT64_C(202612100201), UINT64_C(202612110201),
      UINT64_C(202612120201), UINT64_C(202612130201),
      UINT64_C(202612160201),
      PC_HOLDOUT_TRUTH_WORLDS },
    { "TEST",   UINT64_C(202612100301), UINT64_C(202612110301),
      UINT64_C(202612120301), UINT64_C(202612130301),
      UINT64_C(202612160301),
      PC_HOLDOUT_TRUTH_WORLDS },
};

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
    const unsigned char *p = (const unsigned char *)data;
    s->bits += (uint64_t)n * 8;
    while (n) {
        size_t take = 64 - s->used;
        if (take > n) take = n;
        memcpy(s->block + s->used, p, take);
        s->used += take; p += take; n -= take;
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
    Sha256 s; sha256_init(&s); sha256_update(&s, data, n); sha256_final(&s, out);
}

static void digest_hex(const unsigned char digest[32], char out[65])
{
    static const char x[] = "0123456789abcdef";
    for (int i = 0; i < 32; i++) {
        out[2*i] = x[digest[i] >> 4];
        out[2*i+1] = x[digest[i] & 15];
    }
    out[64] = 0;
}

static int hex_digest(const char *text, unsigned char out[32])
{
    if (!text || strlen(text) != 64) return 0;
    for (int i = 0; i < 32; i++) {
        int hi = isdigit((unsigned char)text[2*i]) ? text[2*i]-'0' :
                 text[2*i]>='a'&&text[2*i]<='f' ? text[2*i]-'a'+10 : -1;
        int lo = isdigit((unsigned char)text[2*i+1]) ? text[2*i+1]-'0' :
                 text[2*i+1]>='a'&&text[2*i+1]<='f' ? text[2*i+1]-'a'+10 : -1;
        if (hi < 0 || lo < 0) return 0;
        out[i] = (unsigned char)(hi*16+lo);
    }
    return 1;
}

static int file_sha256(const char *path, unsigned char out[32])
{
    FILE *f = fopen(path, "rb");
    if (!f) return 0;
    Sha256 s; sha256_init(&s);
    unsigned char buf[65536];
    size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) != 0) sha256_update(&s, buf, n);
    int ok = !ferror(f) && fclose(f) == 0;
    if (!ok) return 0;
    sha256_final(&s, out);
    return 1;
}

static uint64_t mix64(uint64_t x)
{
    x += UINT64_C(0x9e3779b97f4a7c15);
    x = (x ^ (x >> 30)) * UINT64_C(0xbf58476d1ce4e5b9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94d049bb133111eb);
    return x ^ (x >> 31);
}

static uint64_t domain_seed(uint64_t root, uint64_t match,
                            uint64_t state, uint64_t world, uint64_t tag)
{
    return mix64(root ^ mix64(match) ^ mix64(state) ^
                 mix64(world + UINT64_C(0x100000001b3)) ^ tag);
}

static int parse_u64(const char *text, uint64_t *out)
{
    if (!text || !*text || text[0] == '-') return 0;
    char *end = NULL; errno = 0;
    unsigned long long v = strtoull(text, &end, 10);
    if (errno || !end || *end) return 0;
    *out = (uint64_t)v;
    return 1;
}

static int parse_int(const char *text, int lo, int hi, int *out)
{
    if (!text || !*text) return 0;
    char *end = NULL; errno = 0;
    long v = strtol(text, &end, 10);
    if (errno || !end || *end || v < lo || v > hi) return 0;
    *out = (int)v;
    return 1;
}

static int parse_split(const char *text, Split *out)
{
    for (int i = 0; i < 3; i++) if (!strcmp(text, SPLIT_DOMAIN[i].name)) {
        *out = (Split)i; return 1;
    }
    return 0;
}

typedef struct {
    unsigned char data[PC_STATE_MAX];
    size_t n;
    int ok;
} ByteWriter;

static void bw_u8(ByteWriter *w, unsigned x)
{
    if (!w->ok || w->n >= sizeof w->data) { w->ok = 0; return; }
    w->data[w->n++] = (unsigned char)x;
}

static void bw_u16(ByteWriter *w, unsigned x)
{
    bw_u8(w,x); bw_u8(w,x>>8);
}

static void bw_u64(ByteWriter *w, uint64_t x)
{
    for (int i = 0; i < 8; i++) bw_u8(w, (unsigned)(x >> (8*i)));
}

typedef struct {
    const unsigned char *data;
    size_t n, at;
    int ok;
} ByteReader;

static unsigned br_u8(ByteReader *r)
{
    if (!r->ok || r->at >= r->n) { r->ok = 0; return 0; }
    return r->data[r->at++];
}

static unsigned br_u16(ByteReader *r)
{
    unsigned a=br_u8(r), b=br_u8(r); return a | b<<8;
}

static uint64_t br_u64(ByteReader *r)
{
    uint64_t x=0; for (int i=0;i<8;i++) x |= (uint64_t)br_u8(r)<<(8*i);
    return x;
}

/* Explicit field encoding avoids structure padding, native endianness, and
 * inactive pile bytes.  Deck order is absent by construction. */
static int encode_view(const State *s, unsigned char out[PC_STATE_MAX],
                       size_t *nout)
{
    ByteWriter w = { .n=0, .ok=1 };
    bw_u8(&w, 1); /* encoding version */
    bw_u8(&w, s->deck_left);
    bw_u8(&w, s->hand_n[0]); bw_u8(&w, s->hand_n[1]);
    for (int p=0;p<2;p++) bw_u64(&w,s->hand[p]);
    for (int p=0;p<2;p++) bw_u64(&w,s->played[p]);
    bw_u64(&w,s->discarded);
    for (int p=0;p<2;p++) bw_u64(&w,s->known[p]);
    for (int p=0;p<2;p++) for (int q=0;q<NSUIT;q++) bw_u8(&w,s->exp_wager[p][q]);
    for (int p=0;p<2;p++) for (int q=0;q<NSUIT;q++) bw_u8(&w,s->exp_top[p][q]);
    for (int p=0;p<2;p++) for (int q=0;q<NSUIT;q++) bw_u8(&w,s->exp_n[p][q]);
    for (int p=0;p<2;p++) for (int q=0;q<NSUIT;q++) bw_u8(&w,s->exp_sum[p][q]);
    for (int q=0;q<NSUIT;q++) {
        bw_u8(&w,s->pile_n[q]);
        for (int i=0;i<NRANK;i++)
            bw_u8(&w, i<s->pile_n[q] ? s->pile[q][i] : 0);
    }
    bw_u8(&w,s->turn); bw_u8(&w,s->over); bw_u16(&w,s->nply);
    bw_u8(&w,s->round); bw_u16(&w,(uint16_t)s->cum[0]);
    bw_u16(&w,(uint16_t)s->cum[1]);
    if (!w.ok) return 0;
    memcpy(out,w.data,w.n); *nout=w.n; return 1;
}

static int decode_view(const unsigned char *data, size_t n, State *s)
{
    ByteReader r={.data=data,.n=n,.at=0,.ok=1};
    memset(s,0,sizeof *s);
    if (br_u8(&r)!=1) return 0;
    s->deck_left=(uint8_t)br_u8(&r);
    s->hand_n[0]=(uint8_t)br_u8(&r); s->hand_n[1]=(uint8_t)br_u8(&r);
    for(int p=0;p<2;p++)s->hand[p]=br_u64(&r);
    for(int p=0;p<2;p++)s->played[p]=br_u64(&r);
    s->discarded=br_u64(&r);
    for(int p=0;p<2;p++)s->known[p]=br_u64(&r);
    for(int p=0;p<2;p++)for(int q=0;q<NSUIT;q++)s->exp_wager[p][q]=(uint8_t)br_u8(&r);
    for(int p=0;p<2;p++)for(int q=0;q<NSUIT;q++)s->exp_top[p][q]=(uint8_t)br_u8(&r);
    for(int p=0;p<2;p++)for(int q=0;q<NSUIT;q++)s->exp_n[p][q]=(uint8_t)br_u8(&r);
    for(int p=0;p<2;p++)for(int q=0;q<NSUIT;q++)s->exp_sum[p][q]=(uint8_t)br_u8(&r);
    for(int q=0;q<NSUIT;q++){
        s->pile_n[q]=(uint8_t)br_u8(&r);
        for(int i=0;i<NRANK;i++)s->pile[q][i]=(uint8_t)br_u8(&r);
    }
    s->turn=(uint8_t)br_u8(&r); s->over=(uint8_t)br_u8(&r);
    s->nply=(uint16_t)br_u16(&r); s->round=(uint8_t)br_u8(&r);
    s->cum[0]=(int16_t)br_u16(&r); s->cum[1]=(int16_t)br_u16(&r);
    return r.ok && r.at==r.n;
}

static int view_valid(const State *s)
{
    if (!s || s->turn>1 || s->round>=MATCH_ROUNDS || s->over ||
        s->deck_left>NCARD || s->deck_pos!=0 || s->nply>=LC_MAX_PLIES)
        return 0;
    for(int i=0;i<NCARD;i++) if(s->deck[i]!=0) return 0;
    int p=s->turn,o=p^1;
    if (__builtin_popcountll(s->hand[p])!=s->hand_n[p] ||
        __builtin_popcountll(s->hand[o])>s->hand_n[o] ||
        s->hand[o]!=s->known[o] || (s->known[p]&~s->hand[p])) return 0;
    uint64_t piles=0;
    for(int q=0;q<NSUIT;q++){
        if(s->pile_n[q]>NRANK)return 0;
        for(int i=0;i<s->pile_n[q];i++){
            int c=s->pile[q][i];
            if(c>=NCARD||CARD_SUIT(c)!=q||(piles&(UINT64_C(1)<<c)))return 0;
            piles|=UINT64_C(1)<<c;
        }
        /* lc_apply_draw intentionally leaves popped cards in inactive tail
         * slots.  They are neither visible nor state-relevant, and
         * encode_view canonicalizes them to zero. */
    }
    if(piles!=s->discarded)return 0;
    State again; agent_information_view(s,s->turn,&again);
    unsigned char a[PC_STATE_MAX],b[PC_STATE_MAX]; size_t na=0,nb=0;
    return encode_view(s,a,&na)&&encode_view(&again,b,&nb)&&na==nb&&
           !memcmp(a,b,na)&&lc_moves(s,(Move[MAX_MOVES]){{0}})>0;
}

static void bytes_hex(const unsigned char *data,size_t n,char *out)
{
    static const char x[]="0123456789abcdef";
    for(size_t i=0;i<n;i++){out[2*i]=x[data[i]>>4];out[2*i+1]=x[data[i]&15];}
    out[2*n]=0;
}

static int parse_hex_bytes(const char *text,unsigned char *out,size_t n)
{
    if(strlen(text)!=2*n)return 0;
    for(size_t i=0;i<n;i++){
        char pair[3]={text[2*i],text[2*i+1],0}; char *end=NULL;
        if(!isxdigit((unsigned char)pair[0])||!isxdigit((unsigned char)pair[1]))return 0;
        unsigned long v=strtoul(pair,&end,16); if(!end||*end)return 0;
        out[i]=(unsigned char)v;
    }
    return 1;
}

static int same_action(Move a,Move b)
{
    if(a.discard!=b.discard)return 0;
    return a.card==b.card || (CARD_IS_WAGER(a.card)&&CARD_IS_WAGER(b.card)&&
           CARD_SUIT(a.card)==CARD_SUIT(b.card));
}

static uint16_t semantic_move_pack(Move m)
{
    if(CARD_IS_WAGER(m.card))m.card=(uint8_t)CARD_MAKE(CARD_SUIT(m.card),0);
    return MOVE_PACK(m);
}

/* Semantic serialization used only for leak exclusion.  It intentionally
 * drops physical wager ids and chooses the lexicographically smallest of all
 * 120 suit renamings. */
static int semantic_view_bytes(const State *s,unsigned char out[PC_STATE_MAX],
                               size_t *nout)
{
    ByteWriter w={.n=0,.ok=1};
    bw_u8(&w,1); bw_u8(&w,s->turn); bw_u8(&w,s->round);
    bw_u16(&w,s->nply); bw_u8(&w,s->deck_left);
    bw_u8(&w,s->hand_n[0]);bw_u8(&w,s->hand_n[1]);
    bw_u16(&w,(uint16_t)s->cum[0]);bw_u16(&w,(uint16_t)s->cum[1]);
    const uint64_t *kind[3]={s->hand,s->known,s->played};
    for(int k=0;k<3;k++)for(int p=0;p<2;p++)for(int q=0;q<NSUIT;q++){
        int wc=0;unsigned bits=0;
        for(int r=0;r<NRANK;r++)if(kind[k][p]&(UINT64_C(1)<<CARD_MAKE(q,r))){
            if(r<WAGERS_PER_SUIT)wc++; else bits|=1u<<(r-WAGERS_PER_SUIT);
        }
        bw_u8(&w,(unsigned)wc);bw_u16(&w,bits);
    }
    for(int p=0;p<2;p++)for(int q=0;q<NSUIT;q++){
        bw_u8(&w,s->exp_wager[p][q]);bw_u8(&w,s->exp_top[p][q]);
        bw_u8(&w,s->exp_n[p][q]);bw_u8(&w,s->exp_sum[p][q]);
    }
    for(int q=0;q<NSUIT;q++){
        bw_u8(&w,s->pile_n[q]);
        for(int i=0;i<s->pile_n[q];i++){
            int c=s->pile[q][i]; bw_u8(&w,CARD_IS_WAGER(c)?0:CARD_RANK(c));
        }
    }
    if(!w.ok)return 0;
    memcpy(out,w.data,w.n);*nout=w.n;return 1;
}

static int orbit_digest(const State *s,unsigned char digest[32])
{
    uint8_t perm[120][NSUIT];
    if(suit_permutations(120,perm)!=120)return 0;
    unsigned char best[PC_STATE_MAX],cur[PC_STATE_MAX]; size_t nb=0,nc=0;
    for(int i=0;i<120;i++){
        State ps;lc_permute_suits(s,&ps,perm[i]);
        if(!semantic_view_bytes(&ps,cur,&nc))return 0;
        if(i==0||nc<nb||(nc==nb&&memcmp(cur,best,nc)<0)){
            memcpy(best,cur,nc);nb=nc;
        }
    }
    sha_bytes(best,nb,digest);return 1;
}

/* The text fixture parser is intentionally confined to hash-probe.  Neither
 * discover nor evaluate accepts a saved state or probe path. */
static int probe_name_id_free(const char *name, uint64_t used)
{
    char shown[8];
    for (int c=0;c<NCARD;c++) {
        lc_card_name(c,shown);
        if (!strcasecmp(shown,name) && !(used&(UINT64_C(1)<<c))) return c;
    }
    return -1;
}

static int load_probe_state(const char *path, State *st)
{
    FILE *file=fopen(path,"r");
    if (!file) return 0;
    memset(st,0,sizeof *st);
    int deck_entries=0,saw_deck=0;
    uint64_t used=0;
    char line[512];
    while (fgets(line,sizeof line,file)) {
        char *token=strtok(line," \t\n");
        if (!token) continue;
        if (!strcmp(token,"turn")) {
            token=strtok(NULL," \t\n"); if (!token) goto bad;
            st->turn=(uint8_t)atoi(token);
        } else if (!strcmp(token,"round")) {
            token=strtok(NULL," \t\n"); if (!token) goto bad;
            st->round=(uint8_t)atoi(token);
        } else if (!strcmp(token,"nply")) {
            token=strtok(NULL," \t\n"); if (!token) goto bad;
            st->nply=(uint16_t)atoi(token);
        } else if (!strcmp(token,"deck_left")) {
            token=strtok(NULL," \t\n"); if (!token) goto bad;
            st->deck_left=(uint8_t)atoi(token);
        } else if (!strcmp(token,"cum")) {
            char *a=strtok(NULL," \t\n"),*b=strtok(NULL," \t\n");
            if (!a||!b) goto bad;
            st->cum[0]=(int16_t)atoi(a); st->cum[1]=(int16_t)atoi(b);
        } else if (!strncmp(token,"hand",4) && token[4]>='0'&&token[4]<='1') {
            int p=token[4]-'0',c; char *word;
            while ((word=strtok(NULL," \t\n"))) {
                c=probe_name_id_free(word,used); if (c<0) goto bad;
                used|=UINT64_C(1)<<c; st->hand[p]|=UINT64_C(1)<<c; st->hand_n[p]++;
            }
        } else if (!strncmp(token,"known",5) && token[5]>='0'&&token[5]<='1') {
            int p=token[5]-'0'; char *word;
            while ((word=strtok(NULL," \t\n"))) {
                char shown[8]; int found=0;
                for (int c=0;c<NCARD;c++) { lc_card_name(c,shown);
                    if (!strcasecmp(shown,word)&&(st->hand[p]&(UINT64_C(1)<<c))&&
                        !(st->known[p]&(UINT64_C(1)<<c))) {
                        st->known[p]|=UINT64_C(1)<<c; found=1; break;
                    }
                }
                if (!found) goto bad;
            }
        } else if (!strcmp(token,"exp")) {
            char *pt=strtok(NULL," \t\n"),*qt=strtok(NULL," \t\n"),*word;
            if (!pt||!qt) goto bad;
            int p=atoi(pt),q=atoi(qt),c;
            if (p<0||p>1||q<0||q>=NSUIT) goto bad;
            while ((word=strtok(NULL," \t\n"))) {
                c=probe_name_id_free(word,used); if(c<0)goto bad;
                used|=UINT64_C(1)<<c; st->played[p]|=UINT64_C(1)<<c; st->exp_n[p][q]++;
                if (CARD_IS_WAGER(c)) st->exp_wager[p][q]++;
                else { int v=CARD_VALUE(c); if(v>st->exp_top[p][q])st->exp_top[p][q]=(uint8_t)v;
                       st->exp_sum[p][q]=(uint8_t)(st->exp_sum[p][q]+v); }
            }
        } else if (!strcmp(token,"pile")) {
            char *qt=strtok(NULL," \t\n"),*word; if(!qt)goto bad;
            int q=atoi(qt),c; if(q<0||q>=NSUIT)goto bad;
            while ((word=strtok(NULL," \t\n"))) {
                c=probe_name_id_free(word,used); if(c<0||st->pile_n[q]>=NCARD)goto bad;
                used|=UINT64_C(1)<<c; st->pile[q][st->pile_n[q]++]=(uint8_t)c;
                st->discarded|=UINT64_C(1)<<c;
            }
        } else if (!strcmp(token,"deck")) {
            char *word; int c; saw_deck=1;
            while ((word=strtok(NULL," \t\n"))) {
                c=probe_name_id_free(word,used); if(c<0||deck_entries>=NCARD)goto bad;
                used|=UINT64_C(1)<<c; st->deck[deck_entries++]=(uint8_t)c;
            }
        } else goto bad;
    }
    if (fclose(file)!=0) return 0;
    return (!saw_deck||deck_entries==st->deck_left) && st->turn<=1 &&
        st->round<MATCH_ROUNDS && st->deck_left<=NCARD &&
        st->hand_n[0]==HAND_SIZE && st->hand_n[1]==HAND_SIZE;
bad:
    fclose(file); return 0;
}

static int hash_probe_args(int argc,char **argv,const char **path)
{
    *path=NULL;
    for (int i=2;i<argc;i++) {
        if (!strcmp(argv[i],"--state") && ++i<argc && !*path) *path=argv[i];
        else return 0;
    }
    return *path!=NULL;
}

static void json_string(FILE *f,const char *s);

static int run_hash_probe(const char *path)
{
    /* Deliberately audit-only: no Net, Agent, policy, allocation, search, or
     * evidence output path is reachable from this mode. */
    State complete,view; unsigned char orbit[32],state[PC_STATE_MAX],info[PC_STATE_MAX];
    unsigned char file_digest[32],state_digest[32],view_digest[32];
    size_t n=0,ni=0; char orbit_hex[65],state_hex[65],view_hex[65],file_hex[65];
    if (!load_probe_state(path,&complete)) {
        fprintf(stderr,"hash-probe: cannot load state %s\n",path); return 1;
    }
    agent_information_view(&complete,complete.turn,&view);
    if (!view_valid(&view)||!orbit_digest(&view,orbit)||
        !encode_view(&complete,state,&n)||!encode_view(&view,info,&ni)||
        !file_sha256(path,file_digest)) {
        fprintf(stderr,"hash-probe: invalid information view %s\n",path); return 1;
    }
    sha_bytes(state,n,state_digest);sha_bytes(info,ni,view_digest);
    digest_hex(file_digest,file_hex);digest_hex(orbit,orbit_hex);
    digest_hex(state_digest,state_hex);digest_hex(view_digest,view_hex);
    printf("{\"schema\":\"lc-policy-cost-probe-orbit-v1\","
           "\"state_path\":");json_string(stdout,path);
    printf(",\"state_file_sha256\":\"%s\",\"state_sha256\":\"%s\","
           "\"information_view_sha256\":\"%s\","
           "\"suit_orbit_information_view_sha256\":\"%s\"}\n",
           file_hex,state_hex,view_hex,orbit_hex);
    return 0;
}

typedef struct { unsigned char hash[PC_EXCLUSION_COUNT][32]; int n; } Exclusions;

static int load_exclusions(const char *path,const char *expected,
                           Exclusions *out,unsigned char file_hash[32])
{
    unsigned char want[32];
    if(!path||!expected||!hex_digest(expected,want)||
       !file_sha256(path,file_hash)||memcmp(want,file_hash,32))return 0;
    FILE *f=fopen(path,"rb");if(!f)return 0;
    char *line=NULL;size_t cap=0;ssize_t n;
    n=getline(&line,&cap,f);
    if(n<0||strcmp(line,"lc-policy-cost-exclusions-v1\n")){free(line);fclose(f);return 0;}
    memset(out,0,sizeof *out);
    while((n=getline(&line,&cap,f))>=0){
        if(n!=65||line[64]!='\n'||out->n>=PC_EXCLUSION_COUNT){
            free(line);fclose(f);return 0;
        }
        /* hex_digest deliberately accepts only an exact 64-byte lowercase
         * token.  Validate the canonical line terminator above, then exclude
         * it from the token passed to the strict parser. */
        line[64]='\0';
        if(!hex_digest(line,out->hash[out->n])){free(line);fclose(f);return 0;}
        for(int j=0;j<out->n;j++)if(!memcmp(out->hash[j],out->hash[out->n],32)){
            free(line);fclose(f);return 0;
        }
        out->n++;
    }
    int ok=!ferror(f)&&fclose(f)==0&&out->n==PC_EXCLUSION_COUNT;
    free(line);return ok;
}

static int excluded(const Exclusions *x,const unsigned char hash[32])
{
    for(int i=0;i<x->n;i++)if(!memcmp(x->hash[i],hash,32))return 1;
    return 0;
}

typedef struct {
    float floor;
    int n, index[PC_MASK_MAX], core_candidates, draw_candidates;
    double complete_mass, core_mass;
    unsigned char hash[32];
} RuntimeMask;

static int mask_from_policy_cost_support(const Move *mv,const float *prob,int n,
                                         const RolloutPolicyCostSupport *support,
                                         int which,float floor,RuntimeMask *out)
{
    memset(out,0,sizeof *out);out->floor=floor;
    if(!support||which<0||which>=ROLLOUT_POLICY_COST_FLOORS)return 0;
    out->n=support->n[which];out->core_candidates=support->core_candidates[which];
    out->draw_candidates=support->draw_candidates[which];
    if(out->n<1||out->n>PC_MASK_MAX||out->core_candidates<1||
       out->core_candidates+out->draw_candidates!=out->n)return 0;
    memcpy(out->index,support->order[which],(size_t)out->n*sizeof out->index[0]);
    for(int c=0;c<out->n;c++){
        int ix=out->index[c];if(ix<0||ix>=n)return 0;
        for(int j=0;j<c;j++)if(out->index[j]==ix)return 0;
        out->complete_mass+=prob[ix];
    }
    for(int i=0;i<n;i++)for(int c=0;c<out->core_candidates;c++)
        if(same_action(mv[i],mv[out->index[c]])){out->core_mass+=prob[i];break;}
    unsigned char b[2+2*PC_MASK_MAX];size_t at=0;
    uint32_t bits;memcpy(&bits,&floor,sizeof bits);
    b[at++]=(unsigned char)out->n;b[at++]=(unsigned char)(bits^bits>>8^bits>>16^bits>>24);
    for(int c=0;c<out->n;c++){
        uint16_t p=semantic_move_pack(mv[out->index[c]]);
        b[at++]=(unsigned char)p;b[at++]=(unsigned char)(p>>8);
    }
    sha_bytes(b,at,out->hash);return 1;
}

static int mask_union(const RuntimeMask mask[PC_MASKS],int out[PC_UNION_MAX])
{
    int n=0;
    for(int m=0;m<PC_MASKS;m++)for(int c=0;c<mask[m].n;c++){
        int ix=mask[m].index[c],seen=0;
        for(int j=0;j<n;j++)seen|=out[j]==ix;
        if(!seen){if(n>=PC_UNION_MAX)return -1;out[n++]=ix;}
    }
    return n;
}

static int policy_snapshot(const Net *net,const State *s,Move *mv,float *prob,
                           int *n,int *baseline,RuntimeMask mask[PC_MASKS],
                           int union_index[PC_UNION_MAX],int *nunion)
{
    *n=policy_probs_sym(net,s,mv,prob,NULL,PC_SYMMETRIES);
    if(*n<1||*n>MAX_MOVES)return 0;
    double total=0.0;*baseline=0;
    for(int i=0;i<*n;i++){
        if(!lc_float_isfinite(prob[i])||prob[i]<0.0f)return 0;
        total+=prob[i];if(prob[i]>prob[*baseline])*baseline=i;
    }
    if(fabs(total-1.0)>1e-5)return 0;
    RolloutPolicyCostSupport support;
    if(!rollout_policy_cost_support(s,mv,prob,*n,*baseline,&support)||
       !mask_from_policy_cost_support(mv,prob,*n,&support,0,0.01f,&mask[0])||
       !mask_from_policy_cost_support(mv,prob,*n,&support,1,0.02f,&mask[1]))return 0;
    /* Keep an independent local proof at the collector boundary: the 2% mask
     * is a no-refill subsequence of the one-time 1% master. */
    int previous=-1;
    for(int c=0;c<mask[1].n;c++){
        int found=-1;
        for(int m=previous+1;m<mask[0].n;m++)if(mask[1].index[c]==mask[0].index[m]){found=m;break;}
        if(found<0)return 0;
        previous=found;
    }
    *nunion=mask_union(mask,union_index);
    return *nunion>=1&&*nunion<=PC_UNION_MAX;
}

static int frontier_present(const Move *mv,const float *prob,int n,
                            const RuntimeMask mask[PC_MASKS])
{
    for(int c=0;c<mask[0].core_candidates;c++){
        int ix=mask[0].index[c],present_at_two=0;double mass=0.0;
        for(int i=0;i<n;i++)if(same_action(mv[i],mv[ix]))mass+=prob[i];
        for(int j=0;j<mask[1].core_candidates;j++)
            if(same_action(mv[ix],mv[mask[1].index[j]]))present_at_two=1;
        if(!present_at_two&&mass>=(double)0.01f&&mass<(double)0.02f)return 1;
    }
    return 0;
}

typedef struct {
    char *tmp;
    const char *final;
    FILE *file;
} AtomicFile;

static int atomic_open(AtomicFile *a,const char *path)
{
    memset(a,0,sizeof *a);a->final=path;
    struct stat st;
    if(!path||!*path||lstat(path,&st)==0||errno!=ENOENT)return 0;
    size_t n=strlen(path)+48;a->tmp=(char *)malloc(n);if(!a->tmp)return 0;
    snprintf(a->tmp,n,"%s.partial.%ld.XXXXXX",path,(long)getpid());
    int fd=mkstemp(a->tmp);if(fd<0){free(a->tmp);a->tmp=NULL;return 0;}
    a->file=fdopen(fd,"wb");
    if(!a->file){close(fd);unlink(a->tmp);free(a->tmp);a->tmp=NULL;return 0;}
    return 1;
}

static void atomic_abort(AtomicFile *a)
{
    if(!a)return;
    if(a->file)fclose(a->file);
    if(a->tmp)unlink(a->tmp);
    free(a->tmp);memset(a,0,sizeof *a);
}

static int atomic_finish(AtomicFile *a)
{
    if(!a||!a->file||!a->tmp)return 0;
    int fd=fileno(a->file);
    int ok=fflush(a->file)==0&&fsync(fd)==0&&fclose(a->file)==0;
    a->file=NULL;
    if(ok)ok=link(a->tmp,a->final)==0;
    if(unlink(a->tmp)!=0)ok=0;
    free(a->tmp);a->tmp=NULL;
    return ok;
}

static void json_string(FILE *f,const char *s)
{
    fputc('"',f);
    for(;s&&*s;s++){
        unsigned char c=(unsigned char)*s;
        if(c=='"'||c=='\\')fprintf(f,"\\%c",c);
        else if(c=='\n')fputs("\\n",f);
        else if(c=='\r')fputs("\\r",f);
        else if(c=='\t')fputs("\\t",f);
        else if(c<0x20)fprintf(f,"\\u%04x",c);
        else fputc(c,f);
    }
    fputc('"',f);
}

static uint32_t float_bits(float x)
{
    uint32_t b;memcpy(&b,&x,sizeof b);return b;
}

static void source_id(Split split,uint64_t match,char out[32])
{
    snprintf(out,32,"%s-%012llu",SPLIT_DOMAIN[split].name,
             (unsigned long long)match);
}

static void mask_digest_hex(const RuntimeMask *m,char out[65])
{
    digest_hex(m->hash,out);
}

static void union_digest(const Move *mv,const int *index,int n,
                         unsigned char out[32])
{
    unsigned char b[2*PC_UNION_MAX+1];size_t at=0;b[at++]=(unsigned char)n;
    for(int i=0;i<n;i++){
        uint16_t p=semantic_move_pack(mv[index[i]]);
        b[at++]=(unsigned char)p;b[at++]=(unsigned char)(p>>8);
    }
    sha_bytes(b,at,out);
}

static void pair_state_digest(const unsigned char state_hash[32],uint16_t a,
                              uint16_t b,unsigned char out[32])
{
    unsigned char bytes[36];memcpy(bytes,state_hash,32);
    bytes[32]=(unsigned char)a;bytes[33]=(unsigned char)(a>>8);
    bytes[34]=(unsigned char)b;bytes[35]=(unsigned char)(b>>8);
    sha_bytes(bytes,sizeof bytes,out);
}

static void print_mask(FILE *f,const RuntimeMask *m)
{
    char h[65];mask_digest_hex(m,h);
    fprintf(f,"{\"floor\":%.2f,\"floor_bits\":\"%08x\","
              "\"count\":%d,\"core_candidates\":%d,"
              "\"draw_candidates\":%d,\"complete_move_mass\":%.17g,"
              "\"semantic_core_mass\":%.17g,\"sha256\":\"%s\","
              "\"legal_indices\":[",
            m->floor,float_bits(m->floor),m->n,m->core_candidates,
            m->draw_candidates,m->complete_mass,m->core_mass,h);
    for(int i=0;i<m->n;i++)fprintf(f,"%s%d",i?",":"",m->index[i]);
    fputs("]}",f);
}

/* Discovery retains only a content-hash-priority reservoir.  A campaign with
 * 65,536 matches therefore remains bounded, while the census and chained
 * commitments still account for every generated root before any Q/truth
 * outcome exists. */
typedef struct {
    unsigned char priority[32],orbit[32],state_hash[32];
    unsigned char mask_hash[2][32],union_hash[32];
    unsigned char state[PC_STATE_MAX];size_t state_n;
    uint64_t source_match_index;int source_state_index,round,pbin,rbin,type;
    uint16_t pair_move[2];
} ReservoirEntry;

typedef struct {
    ReservoirEntry *entry;int n,cap;uint64_t attempted;
} CellReservoir;

typedef struct {
    unsigned char priority[32],orbit[32],state_hash[32];
    unsigned char mask_hash[PC_MASKS][32],master_hash[32];
    unsigned char state[PC_STATE_MAX];size_t state_n;
    uint64_t source_match_index;int source_state_index,round,pbin,frontier;
    int master_width,allocation_slot;
} VectorEntry;

typedef struct {
    VectorEntry *entry;int n,cap;uint64_t eligible,width_count[PC_MASK_MAX+1];
} VectorReservoir;

typedef struct {
    CellReservoir cell[3][PC_PLY_BINS][PC_RATIO_BINS][PC_PAIR_TYPES];
    VectorReservoir vector[3][PC_PLY_BINS][2][PC_MASK_MAX];
    uint64_t accepted_states[3],pooled_ge64[3],exact_terminal_preempted[3];
    uint64_t mask_width[2][PC_MASK_MAX+1],union_width[PC_UNION_MAX+1];
    unsigned char state_chain[32];
} DiscoveryCensus;

static int ply_bin(int nply);
static int ratio_bin(double ratio);
static void policy_terms(const Move *mv,const float *prob,int n,int index,
                         double *action,double *draw);

static int priority_less(const unsigned char a[32],const unsigned char b[32])
{
    return memcmp(a,b,32)<0;
}

static int reservoir_insert(CellReservoir *c,const ReservoirEntry *entry)
{
    if(c->n<c->cap){c->entry[c->n++]=*entry;return 1;}
    int worst=0;
    for(int i=1;i<c->n;i++)if(priority_less(c->entry[worst].priority,
                                            c->entry[i].priority))worst=i;
    if(!priority_less(entry->priority,c->entry[worst].priority))return 0;
    c->entry[worst]=*entry;return 1;
}

static int vector_insert(VectorReservoir *c,const VectorEntry *entry)
{
    if(c->n<c->cap){c->entry[c->n++]=*entry;return 1;}
    int worst=0;
    for(int i=1;i<c->n;i++)if(priority_less(c->entry[worst].priority,
                                            c->entry[i].priority))worst=i;
    if(!priority_less(entry->priority,c->entry[worst].priority))return 0;
    c->entry[worst]=*entry;return 1;
}

static void chain_state(DiscoveryCensus *c,int accepted,uint64_t match,int state,
                        const unsigned char orbit[32],const unsigned char sh[32],
                        const unsigned char policy_hash[32])
{
    Sha256 h;sha256_init(&h);sha256_update(&h,c->state_chain,32);
    unsigned char status=(unsigned char)accepted;sha256_update(&h,&status,1);
    unsigned char id[12];for(int i=0;i<8;i++)id[i]=(unsigned char)(match>>(8*i));
    for(int i=0;i<4;i++)id[8+i]=(unsigned char)((unsigned)state>>(8*i));
    sha256_update(&h,id,sizeof id);sha256_update(&h,orbit,32);sha256_update(&h,sh,32);
    if(accepted)sha256_update(&h,policy_hash,32);
    sha256_final(&h,c->state_chain);
}

static void policy_commitment(const Move *mv,const float *prob,int n,
                              const RuntimeMask mask[2],const int *uindex,
                              int nunion,unsigned char out[32])
{
    Sha256 h;sha256_init(&h);unsigned char count=(unsigned char)n;
    sha256_update(&h,&count,1);
    for(int i=0;i<n;i++){
        uint16_t p=semantic_move_pack(mv[i]);uint32_t b=float_bits(prob[i]);
        unsigned char row[6]={(unsigned char)p,(unsigned char)(p>>8),
            (unsigned char)b,(unsigned char)(b>>8),(unsigned char)(b>>16),(unsigned char)(b>>24)};
        sha256_update(&h,row,sizeof row);
    }
    sha256_update(&h,mask[0].hash,32);sha256_update(&h,mask[1].hash,32);
    unsigned char uh[32];union_digest(mv,uindex,nunion,uh);sha256_update(&h,uh,32);
    sha256_final(&h,out);
}

static void entry_priority(uint64_t seed,const ReservoirEntry *e,unsigned char out[32])
{
    Sha256 h;sha256_init(&h);const char tag[]="lc-policy-cost-reservoir-priority-v1";
    sha256_update(&h,tag,sizeof tag-1);unsigned char b[32];size_t at=0;
    for(int i=0;i<8;i++)b[at++]=(unsigned char)(seed>>(8*i));
    for(int i=0;i<8;i++)b[at++]=(unsigned char)(e->source_match_index>>(8*i));
    for(int i=0;i<4;i++)b[at++]=(unsigned char)((unsigned)e->source_state_index>>(8*i));
    b[at++]=(unsigned char)e->round;b[at++]=(unsigned char)e->pbin;
    b[at++]=(unsigned char)e->rbin;b[at++]=(unsigned char)e->type;
    b[at++]=(unsigned char)e->pair_move[0];b[at++]=(unsigned char)(e->pair_move[0]>>8);
    b[at++]=(unsigned char)e->pair_move[1];b[at++]=(unsigned char)(e->pair_move[1]>>8);
    sha256_update(&h,b,at);sha256_update(&h,e->state_hash,32);sha256_final(&h,out);
}

static int vector_allocation_slot(uint64_t seed,const unsigned char state_hash[32],
                                  int width)
{
    Sha256 h;sha256_init(&h);
    const char tag[]="lc-policy-cost-vector-slot-v1";
    sha256_update(&h,tag,sizeof tag-1);
    unsigned char b[8];for(int i=0;i<8;i++)b[i]=(unsigned char)(seed>>(8*i));
    sha256_update(&h,b,sizeof b);sha256_update(&h,state_hash,32);
    unsigned char digest[32];sha256_final(&h,digest);
    uint64_t value=0;for(int i=0;i<8;i++)value|=(uint64_t)digest[i]<<(8*i);
    return width>0?(int)(value%(uint64_t)width):-1;
}

static void vector_priority(uint64_t seed,const VectorEntry *e,unsigned char out[32])
{
    Sha256 h;sha256_init(&h);
    const char tag[]="lc-policy-cost-vector-priority-v1";
    sha256_update(&h,tag,sizeof tag-1);unsigned char b[28];size_t at=0;
    for(int i=0;i<8;i++)b[at++]=(unsigned char)(seed>>(8*i));
    for(int i=0;i<8;i++)b[at++]=(unsigned char)(e->source_match_index>>(8*i));
    for(int i=0;i<4;i++)b[at++]=(unsigned char)((unsigned)e->source_state_index>>(8*i));
    b[at++]=(unsigned char)e->round;b[at++]=(unsigned char)e->pbin;
    b[at++]=(unsigned char)e->frontier;b[at++]=(unsigned char)e->allocation_slot;
    sha256_update(&h,b,at);sha256_update(&h,e->state_hash,32);sha256_final(&h,out);
}

static void discovery_pairs(DiscoveryCensus *c,uint64_t seed,uint64_t match,
                            int state_index,const State *view,const Move *mv,
                            const float *prob,int n,const RuntimeMask mask[2],
                            const int *uindex,int nunion,
                            const unsigned char orbit[32],
                            const unsigned char *state,size_t state_n,
                            const unsigned char state_hash[32])
{
    int pb=ply_bin(view->nply);
    if(view->deck_left<=1||pb<0||pb>=PC_PLY_BINS)return;
    unsigned char uh[32];union_digest(mv,uindex,nunion,uh);
    for(int x=0;x<nunion;x++)for(int y=x+1;y<nunion;y++){
        int ix=uindex[x],iy=uindex[y],type=same_action(mv[ix],mv[iy])?1:0;
        if(type&&mv[ix].draw==mv[iy].draw)continue;
        double ax,adx,ay,ady;policy_terms(mv,prob,n,ix,&ax,&adx);
        policy_terms(mv,prob,n,iy,&ay,&ady);
        double px=type?adx:ax,py=type?ady:ay;
        if(!(px>0.0)||!(py>0.0))continue;
        int high=ix,low=iy;
        if(py>px||(py==px&&semantic_move_pack(mv[iy])<semantic_move_pack(mv[ix]))){
            high=iy;low=ix;double t=px;px=py;py=t;
        }
        int rb=ratio_bin(px/py);if(rb<0)continue;
        CellReservoir *cell=&c->cell[view->round][pb][rb][type];cell->attempted++;
        ReservoirEntry e;memset(&e,0,sizeof e);
        e.source_match_index=match;e.source_state_index=state_index;e.round=view->round;
        e.pbin=pb;e.rbin=rb;e.type=type;e.pair_move[0]=semantic_move_pack(mv[high]);
        e.pair_move[1]=semantic_move_pack(mv[low]);e.state_n=state_n;
        memcpy(e.state,state,state_n);memcpy(e.orbit,orbit,32);memcpy(e.state_hash,state_hash,32);
        memcpy(e.mask_hash[0],mask[0].hash,32);memcpy(e.mask_hash[1],mask[1].hash,32);
        memcpy(e.union_hash,uh,32);entry_priority(seed,&e,e.priority);
        reservoir_insert(cell,&e);
    }
}

static void discovery_account(DiscoveryCensus *c,const State *view,
                              const RuntimeMask mask[PC_MASKS],int nunion)
{
    int rd=view->round,pb=ply_bin(view->nply);
    c->accepted_states[rd]++;
    c->mask_width[0][mask[0].n]++;c->mask_width[1][mask[1].n]++;
    c->union_width[nunion]++;
    if(view->deck_left<=1)c->exact_terminal_preempted[rd]++;
    else if(pb==PC_PLY_BINS)c->pooled_ge64[rd]++;
}

static void discovery_vector(DiscoveryCensus *c,uint64_t seed,uint64_t match,
                             int state_index,const State *view,const Move *mv,
                             const float *prob,int n,const RuntimeMask mask[PC_MASKS],
                             const unsigned char orbit[32],const unsigned char *state,
                             size_t state_n,const unsigned char state_hash[32])
{
    int pb=ply_bin(view->nply);
    if(view->deck_left<=1||pb<0||pb>=PC_PLY_BINS)return;
    int frontier=frontier_present(mv,prob,n,mask),width=mask[0].n;
    int slot=vector_allocation_slot(seed,state_hash,width);
    if(slot<0||slot>=width)return;
    VectorReservoir *cell=&c->vector[view->round][pb][frontier][slot];
    cell->eligible++;cell->width_count[width]++;
    VectorEntry e;memset(&e,0,sizeof e);e.source_match_index=match;
    e.source_state_index=state_index;e.round=view->round;e.pbin=pb;
    e.frontier=frontier;e.master_width=width;e.allocation_slot=slot;
    e.state_n=state_n;memcpy(e.state,state,state_n);memcpy(e.orbit,orbit,32);
    memcpy(e.state_hash,state_hash,32);memcpy(e.mask_hash,mask[0].hash,32);
    memcpy(e.mask_hash[1],mask[1].hash,32);memcpy(e.master_hash,mask[0].hash,32);
    vector_priority(seed,&e,e.priority);vector_insert(cell,&e);
}

static int entry_compare(const void *aa,const void *bb)
{
    const ReservoirEntry *a=(const ReservoirEntry *)aa,*b=(const ReservoirEntry *)bb;
    return memcmp(a->priority,b->priority,32);
}

static int vector_compare(const void *aa,const void *bb)
{
    const VectorEntry *a=(const VectorEntry *)aa,*b=(const VectorEntry *)bb;
    return memcmp(a->priority,b->priority,32);
}

static int census_init(DiscoveryCensus *c,int capacity,uint64_t seed)
{
    memset(c,0,sizeof *c);Sha256 h;sha256_init(&h);
    const char schema[]="lc-policy-cost-state-chain-v1";sha256_update(&h,schema,sizeof schema-1);
    unsigned char sb[8];for(int i=0;i<8;i++)sb[i]=(unsigned char)(seed>>(8*i));
    sha256_update(&h,sb,sizeof sb);sha256_final(&h,c->state_chain);
    for(int r=0;r<3;r++)for(int p=0;p<PC_PLY_BINS;p++)
      for(int g=0;g<PC_RATIO_BINS;g++)for(int t=0;t<PC_PAIR_TYPES;t++){
        CellReservoir *cell=&c->cell[r][p][g][t];cell->cap=capacity;
        cell->entry=(ReservoirEntry *)calloc((size_t)capacity,sizeof *cell->entry);
        if(!cell->entry)return 0;
    }
    for(int r=0;r<3;r++)for(int p=0;p<PC_PLY_BINS;p++)for(int f=0;f<2;f++)
      for(int j=0;j<PC_MASK_MAX;j++){
        VectorReservoir *cell=&c->vector[r][p][f][j];cell->cap=capacity;
        cell->entry=(VectorEntry *)calloc((size_t)capacity,sizeof *cell->entry);
        if(!cell->entry)return 0;
    }
    return 1;
}

static void census_free(DiscoveryCensus *c)
{
    for(int r=0;r<3;r++)for(int p=0;p<PC_PLY_BINS;p++)
      for(int g=0;g<PC_RATIO_BINS;g++)for(int t=0;t<PC_PAIR_TYPES;t++)
        free(c->cell[r][p][g][t].entry);
    for(int r=0;r<3;r++)for(int p=0;p<PC_PLY_BINS;p++)for(int f=0;f<2;f++)
      for(int j=0;j<PC_MASK_MAX;j++)free(c->vector[r][p][f][j].entry);
    memset(c,0,sizeof *c);
}

typedef struct {
    const char *out_path,*reservoir_path,*net_path,*exclusion_path,*exclusion_sha;
    Split split;int have_split,matches,reservoir_per_cell;uint64_t match_start;
    int smoke,have_reservoir_per_cell;uint64_t smoke_seed;
} DiscoverConfig;

static int smoke_seed_valid(uint64_t seed)
{
    char b[32];snprintf(b,sizeof b,"%llu",(unsigned long long)seed);
    return !strncmp(b,"20261229",8);
}

static int discover_args(int argc,char **argv,DiscoverConfig *c)
{
    memset(c,0,sizeof *c);c->matches=1;
    for(int i=2;i<argc;i++){
        if(!strcmp(argv[i],"--out")&&++i<argc)c->out_path=argv[i];
        else if(!strcmp(argv[i],"--reservoir-out")&&++i<argc)c->reservoir_path=argv[i];
        else if(!strcmp(argv[i],"--net")&&++i<argc)c->net_path=argv[i];
        else if(!strcmp(argv[i],"--split")&&++i<argc&&parse_split(argv[i],&c->split))c->have_split=1;
        else if(!strcmp(argv[i],"--matches")&&++i<argc&&parse_int(argv[i],1,1000000,&c->matches)){}
        else if(!strcmp(argv[i],"--reservoir-per-cell")&&++i<argc&&
                parse_int(argv[i],1,4096,&c->reservoir_per_cell))c->have_reservoir_per_cell=1;
        else if(!strcmp(argv[i],"--match-start")&&++i<argc&&parse_u64(argv[i],&c->match_start)){}
        else if(!strcmp(argv[i],"--exclusions")&&++i<argc)c->exclusion_path=argv[i];
        else if(!strcmp(argv[i],"--exclusions-sha256")&&++i<argc)c->exclusion_sha=argv[i];
        else if(!strcmp(argv[i],"--smoke-seed")&&++i<argc&&parse_u64(argv[i],&c->smoke_seed))c->smoke=1;
        else return 0;
    }
    if(c->have_split&&!c->have_reservoir_per_cell)
        c->reservoir_per_cell=c->split==SPLIT_TRAIN?1024:PC_VECTOR_QUOTA;
    int frozen=c->split==SPLIT_TRAIN?1024:PC_VECTOR_QUOTA;
    return c->out_path&&c->reservoir_path&&c->net_path&&c->have_split&&
           c->exclusion_path&&c->exclusion_sha&&
           (!c->smoke||smoke_seed_valid(c->smoke_seed))&&
           (c->smoke||c->reservoir_per_cell==frozen);
}

static int run_discover(const DiscoverConfig *c)
{
    Exclusions exclusions;unsigned char exclusion_hash[32],net_hash[32];
    if(!load_exclusions(c->exclusion_path,c->exclusion_sha,&exclusions,exclusion_hash)){
        fprintf(stderr,"invalid or unbound 17-hash exclusion manifest\n");return 1;
    }
    if(!file_sha256(c->net_path,net_hash)){fprintf(stderr,"cannot hash net\n");return 1;}
    Net *net=(Net *)malloc(sizeof *net);
    if(!net||net_load(net,c->net_path)){fprintf(stderr,"cannot load net\n");free(net);return 1;}
    AtomicFile evidence={0},reservoir={0};
    if(!atomic_open(&evidence,c->out_path)||!atomic_open(&reservoir,c->reservoir_path)){
        fprintf(stderr,"output exists or cannot be created\n");atomic_abort(&evidence);atomic_abort(&reservoir);free(net);return 1;
    }
    char nh[65],eh[65];digest_hex(net_hash,nh);digest_hex(exclusion_hash,eh);
    uint64_t root_seed=c->smoke?c->smoke_seed:SPLIT_DOMAIN[c->split].discovery_seed;
    DiscoveryCensus census={0};
    if(!census_init(&census,c->reservoir_per_cell,root_seed)){
        fprintf(stderr,"cannot allocate bounded discovery reservoir\n");goto fail;
    }
    fprintf(evidence.file,"{\"schema\":\"lc-policy-cost-discovery-v2\","
        "\"record_type\":\"header\",\"split\":\"%s\","
        "\"purpose\":\"%s\",\"seed\":\"%llu\","
        "\"seed_domain\":\"%s\",\"match_start\":%llu,"
        "\"requested_matches\":%d,\"generator\":\"policy20_self_play\","
        "\"state_import_supported\":false,\"symmetries\":20,"
        "\"net_path\":",SPLIT_DOMAIN[c->split].name,
        c->smoke?"feasibility_smoke_excluded_from_campaign":"locked_campaign",
        (unsigned long long)root_seed,c->smoke?"20261229-smoke":"locked-discovery",
        (unsigned long long)c->match_start,c->matches);
    json_string(evidence.file,c->net_path);
    fprintf(evidence.file,",\"net_sha256\":\"%s\",\"exclusion_manifest_sha256\":\"%s\","
        "\"exclusion_orbits\":17,\"floor_bits\":[\"%08x\",\"%08x\"],"
        "\"shortlist\":{\"root_width\":5,\"action_core_count\":3,"
        "\"min_candidates\":1,\"candidate_mass\":0},"
        "\"master_max\":5,\"truth_support_max\":6,"
        "\"reservoir_method\":\"bounded_sha256_priority\","
        "\"reservoir_per_subcell\":%d,\"ply_bins\":[",
        nh,eh,float_bits(0.01f),float_bits(0.02f),c->reservoir_per_cell);
    for(int p=0;p<PC_PLY_BINS;p++){
        int lo=p<22?2*p:p==22?44:48;
        int hi=p<22?2*p+2:p==22?48:64;
        fprintf(evidence.file,"%s[%d,%d]",p?",":"",lo,hi);
    }
    fprintf(evidence.file,"],\"pooled_ge64\":\"census_only\","
        "\"vector_slot\":\"sha256(split_seed,state_sha256) mod master_width\","
        "\"frontier\":\"1pct admits aggregate semantic core in [0.01,0.02) removed by 2pct\","
        "\"burned_source_deal_seeds\":\"1..200, maintained-800 seed 1, "
        "202611010101, all policy-cost-v1 fixed seeds in "
        "20261110/11/12/13/14/15/16/21/22, every 20261129 "
        "feasibility-smoke seed, 202612010101, and every 20261229 "
        "feasibility-smoke seed\"}\n");
    fprintf(reservoir.file,"%s\nsplit\t%s\npurpose\t%s\nseed\t%llu\n"
        "net_sha256\t%s\nexclusion_sha256\t%s\nreservoir_per_subcell\t%d\n",
        c->split==SPLIT_TRAIN?"LCPOLICYCOST-TRAIN-RESERVOIR-V3":
                              "LCPOLICYCOST-VECTOR-RESERVOIR-V1",
        SPLIT_DOMAIN[c->split].name,c->smoke?"smoke":"campaign",
        (unsigned long long)root_seed,nh,eh,c->reservoir_per_cell);

    uint64_t attempted=0,accepted=0,rejected=0,caps=0;
    for(int mi=0;mi<c->matches;mi++){
        uint64_t match_index=c->match_start+(uint64_t)mi;
        int cumulative[2]={0,0},state_index=0;
        for(int round=0;round<MATCH_ROUNDS;round++){
            Rng deal_rng;rng_seed(&deal_rng,domain_seed(root_seed,match_index,round,0,TAG_DEAL));
            State complete;lc_deal(&complete,&deal_rng);complete.round=(uint8_t)round;
            complete.turn=(uint8_t)(round&1);complete.cum[0]=(int16_t)cumulative[0];
            complete.cum[1]=(int16_t)cumulative[1];
            while(!complete.over){
                State view;agent_information_view(&complete,complete.turn,&view);
                attempted++;
                unsigned char orbit[32];
                if(!view_valid(&view)||!orbit_digest(&view,orbit)){
                    fprintf(stderr,"invalid generated information view\n");goto fail;
                }
                unsigned char sb[PC_STATE_MAX],sh[32],policy_hash[32]={0};size_t sn=0;
                if(!encode_view(&view,sb,&sn)){fprintf(stderr,"state encoding failed\n");goto fail;}
                sha_bytes(sb,sn,sh);
                if(excluded(&exclusions,orbit)){
                    rejected++;chain_state(&census,0,match_index,state_index,orbit,sh,policy_hash);
                    /* Rejection precedes policy classification/allocation. */
                    Move pmv[MAX_MOVES];float pprob[MAX_MOVES];
                    int pn=policy_probs_sym(net,&view,pmv,pprob,NULL,PC_SYMMETRIES),pbest=0;
                    if(pn<1){fprintf(stderr,"policy failure after exclusion\n");goto fail;}
                    for(int pi=1;pi<pn;pi++)if(pprob[pi]>pprob[pbest])pbest=pi;
                    lc_apply(&complete,pmv[pbest]);state_index++;continue;
                }
                Move mv[MAX_MOVES];float prob[MAX_MOVES];RuntimeMask mask[PC_MASKS];
                int n=0,baseline=0,uindex[PC_UNION_MAX],nunion=0;
                if(!policy_snapshot(net,&view,mv,prob,&n,&baseline,mask,uindex,&nunion)){
                    fprintf(stderr,"invalid exact policy or shortlist\n");goto fail;
                }
                accepted++;policy_commitment(mv,prob,n,mask,uindex,nunion,policy_hash);
                chain_state(&census,1,match_index,state_index,orbit,sh,policy_hash);
                discovery_account(&census,&view,mask,nunion);
                if(c->split==SPLIT_TRAIN)
                    discovery_pairs(&census,root_seed,match_index,state_index,&view,mv,prob,n,
                                    mask,uindex,nunion,orbit,sb,sn,sh);
                else
                    discovery_vector(&census,root_seed,match_index,state_index,&view,mv,prob,n,
                                     mask,orbit,sb,sn,sh);
                lc_apply(&complete,mv[baseline]);state_index++;
            }
            if(complete.deck_left>0){caps++;fprintf(stderr,"policy self-play hit ply cap\n");goto fail;}
            cumulative[0]+=lc_score(&complete,0);cumulative[1]+=lc_score(&complete,1);
        }
    }
    char chain_hex[65];digest_hex(census.state_chain,chain_hex);
    fprintf(evidence.file,"{\"schema\":\"lc-policy-cost-discovery-v2\","
        "\"record_type\":\"census\",\"state_commitment_chain_sha256\":\"%s\","
        "\"accepted_by_round\":[%llu,%llu,%llu],"
        "\"pooled_ge64_by_round\":[%llu,%llu,%llu],"
        "\"exact_terminal_preempted_by_round\":[%llu,%llu,%llu],"
        "\"mask_width_counts\":[[",
        chain_hex,(unsigned long long)census.accepted_states[0],
        (unsigned long long)census.accepted_states[1],
        (unsigned long long)census.accepted_states[2],
        (unsigned long long)census.pooled_ge64[0],
        (unsigned long long)census.pooled_ge64[1],
        (unsigned long long)census.pooled_ge64[2],
        (unsigned long long)census.exact_terminal_preempted[0],
        (unsigned long long)census.exact_terminal_preempted[1],
        (unsigned long long)census.exact_terminal_preempted[2]);
    for(int w=1;w<=PC_MASK_MAX;w++)fprintf(evidence.file,"%s%llu",w>1?",":"",
        (unsigned long long)census.mask_width[0][w]);
    fputs("],[",evidence.file);
    for(int w=1;w<=PC_MASK_MAX;w++)fprintf(evidence.file,"%s%llu",w>1?",":"",
        (unsigned long long)census.mask_width[1][w]);
    fputs("]],\"union_width_counts\":[",evidence.file);
    for(int w=1;w<=PC_UNION_MAX;w++)fprintf(evidence.file,"%s%llu",w>1?",":"",
        (unsigned long long)census.union_width[w]);
    fputs("],\"eligible_master_width_counts\":[",evidence.file);
    for(int w=1;w<=PC_MASK_MAX;w++){
      uint64_t total_width=0;
      for(int rd=0;rd<3;rd++)for(int pb=0;pb<PC_PLY_BINS;pb++)
       for(int fr=0;fr<2;fr++)for(int sl=0;sl<PC_MASK_MAX;sl++)
        total_width+=census.vector[rd][pb][fr][sl].width_count[w];
      fprintf(evidence.file,"%s%llu",w>1?",":"",(unsigned long long)total_width);
    }
    fputs("],\"allocation_cells\":[",evidence.file);int first_cell=1;
    if(c->split==SPLIT_TRAIN){
      for(int rd=0;rd<3;rd++)for(int pb=0;pb<PC_PLY_BINS;pb++)
       for(int rb=0;rb<PC_RATIO_BINS;rb++)for(int tp=0;tp<PC_PAIR_TYPES;tp++){
        CellReservoir *cell=&census.cell[rd][pb][rb][tp];
        fprintf(evidence.file,"%s{\"cell\":\"r%d.p%d.g%d.t%d\","
            "\"eligible_units\":%llu,\"retained_units\":%d}",
            first_cell?"":",",rd,pb,rb,tp,(unsigned long long)cell->attempted,cell->n);
        first_cell=0;
      }
    }else{
      for(int rd=0;rd<3;rd++)for(int pb=0;pb<PC_PLY_BINS;pb++)
       for(int fr=0;fr<2;fr++)for(int sl=0;sl<PC_MASK_MAX;sl++){
        VectorReservoir *cell=&census.vector[rd][pb][fr][sl];
        fprintf(evidence.file,"%s{\"cell\":\"r%d:p%02d:f%d:j%d\","
            "\"eligible_vectors\":%llu,\"retained_vectors\":%d,"
            "\"master_width_histogram\":[%llu,%llu,%llu,%llu,%llu]}",
            first_cell?"":",",rd,pb,fr,sl,(unsigned long long)cell->eligible,cell->n,
            (unsigned long long)cell->width_count[1],
            (unsigned long long)cell->width_count[2],
            (unsigned long long)cell->width_count[3],
            (unsigned long long)cell->width_count[4],
            (unsigned long long)cell->width_count[5]);
        first_cell=0;
      }
    }
    fputs("]}\n",evidence.file);

    uint64_t retained=0,eligible_units=0;
    if(c->split==SPLIT_TRAIN){
      fputs("columns\tcell\tpriority_sha256\tsource_match_index\t"
          "source_state_index\tsource_match_id\tround\tply_bin\tratio_bin\t"
          "pair_type\tpair_move_a\tpair_move_b\torbit_sha256\tstate_sha256\t"
          "mask_001_sha256\tmask_002_sha256\tmaster_sha256\tstate_hex\n",reservoir.file);
      for(int rd=0;rd<3;rd++)for(int pb=0;pb<PC_PLY_BINS;pb++)
       for(int rb=0;rb<PC_RATIO_BINS;rb++)for(int tp=0;tp<PC_PAIR_TYPES;tp++){
        CellReservoir *cell=&census.cell[rd][pb][rb][tp];
        qsort(cell->entry,(size_t)cell->n,sizeof *cell->entry,entry_compare);
        eligible_units+=cell->attempted;
        for(int i=0;i<cell->n;i++){
            ReservoirEntry *e=&cell->entry[i];char pri[65],oh[65],shx[65],m0[65],m1[65],uh[65];
            char state_hex[2*PC_STATE_MAX+1],sid[32];digest_hex(e->priority,pri);
            digest_hex(e->orbit,oh);digest_hex(e->state_hash,shx);digest_hex(e->mask_hash[0],m0);
            digest_hex(e->mask_hash[1],m1);digest_hex(e->union_hash,uh);bytes_hex(e->state,e->state_n,state_hex);
            source_id(c->split,e->source_match_index,sid);
            fprintf(reservoir.file,"r%d.p%d.g%d.t%d\t%s\t%llu\t%d\t%s\t%d\t%d\t%d\t%d\t"
                "%u\t%u\t%s\t%s\t%s\t%s\t%s\t%s\n",rd,pb,rb,tp,pri,
                (unsigned long long)e->source_match_index,e->source_state_index,sid,
                rd,pb,rb,tp,e->pair_move[0],e->pair_move[1],oh,shx,m0,m1,uh,state_hex);
            retained++;
        }
      }
    }else{
      fputs("columns\tcell\tpriority_sha256\tsource_match_index\t"
          "source_state_index\tsource_match_id\tround\tply_bin\tfrontier_present\t"
          "allocation_slot\tmaster_width\torbit_sha256\tstate_sha256\t"
          "mask_001_sha256\tmask_002_sha256\tmaster_sha256\tstate_hex\n",reservoir.file);
      for(int rd=0;rd<3;rd++)for(int pb=0;pb<PC_PLY_BINS;pb++)
       for(int fr=0;fr<2;fr++)for(int sl=0;sl<PC_MASK_MAX;sl++){
        VectorReservoir *cell=&census.vector[rd][pb][fr][sl];
        qsort(cell->entry,(size_t)cell->n,sizeof *cell->entry,vector_compare);
        eligible_units+=cell->eligible;
        for(int i=0;i<cell->n;i++){
            VectorEntry *e=&cell->entry[i];char pri[65],oh[65],shx[65],m0[65],m1[65],mh[65];
            char state_hex[2*PC_STATE_MAX+1],sid[32];digest_hex(e->priority,pri);
            digest_hex(e->orbit,oh);digest_hex(e->state_hash,shx);
            digest_hex(e->mask_hash[0],m0);digest_hex(e->mask_hash[1],m1);
            digest_hex(e->master_hash,mh);bytes_hex(e->state,e->state_n,state_hex);
            source_id(c->split,e->source_match_index,sid);
            fprintf(reservoir.file,"r%d:p%02d:f%d:j%d\t%s\t%llu\t%d\t%s\t%d\t%d\t%d\t%d\t%d\t"
                "%s\t%s\t%s\t%s\t%s\t%s\n",rd,pb,fr,sl,pri,
                (unsigned long long)e->source_match_index,e->source_state_index,sid,
                rd,pb,fr,sl,e->master_width,oh,shx,m0,m1,mh,state_hex);
            retained++;
        }
      }
    }
    fprintf(reservoir.file,"footer\teligible_units\t%llu\tretained_units\t%llu\t"
        "rejected_by_bound\t%llu\tstate_commitment_chain_sha256\t%s\t"
        "pooled_ge64_observed\t%llu\n",(unsigned long long)eligible_units,
        (unsigned long long)retained,(unsigned long long)(eligible_units-retained),chain_hex,
        (unsigned long long)(census.pooled_ge64[0]+census.pooled_ge64[1]+census.pooled_ge64[2]));
    fprintf(evidence.file,"{\"schema\":\"lc-policy-cost-discovery-v2\","
        "\"record_type\":\"footer\",\"requested_matches\":%d,"
        "\"completed_matches\":%d,\"attempted_states\":%llu,"
        "\"accepted_states\":%llu,\"probe_orbit_rejections\":%llu,"
        "\"cap_hits\":%llu,\"eligible_units\":%llu,"
        "\"retained_units\":%llu,\"units_rejected_by_bound\":%llu}\n",
        c->matches,c->matches,
        (unsigned long long)attempted,(unsigned long long)accepted,
        (unsigned long long)rejected,(unsigned long long)caps,
        (unsigned long long)eligible_units,(unsigned long long)retained,
        (unsigned long long)(eligible_units-retained));
    free(net);census_free(&census);
    if(!atomic_finish(&reservoir)||!atomic_finish(&evidence)){
        fprintf(stderr,"cannot atomically install discovery outputs\n");return 1;
    }
    unsigned char h[32];char hex[65];
    if(file_sha256(c->out_path,h)){digest_hex(h,hex);printf("evidence_sha256=%s\n",hex);}
    if(file_sha256(c->reservoir_path,h)){digest_hex(h,hex);printf("reservoir_sha256=%s\n",hex);}
    return 0;
fail:
    census_free(&census);free(net);atomic_abort(&reservoir);atomic_abort(&evidence);return 1;
}

static int ply_bin(int nply)
{
    if(nply<0||nply>=LC_MAX_PLIES)return -1;
    if(nply<44)return nply/2;       /* 22 strata: [0,2) through [42,44). */
    if(nply<48)return 22;           /* Runtime-anchor-aligned sparse tail. */
    if(nply<64)return 23;
    return PC_PLY_BINS;             /* pooled diagnostic only */
}

static int ratio_bin(double ratio)
{
    if(!(ratio>=1.0)||!lc_double_isfinite(ratio))return -1;
    if(ratio<1.25)return 0;
    if(ratio<2.0)return 1;
    if(ratio<4.0)return 2;
    if(ratio<8.0)return 3;
    if(ratio<32.0)return 4;
    return 5;
}

typedef struct {
    uint64_t allocation_id,source_match_index;
    int source_state_index,round,ply_bin,ratio_bin,pair_type;
    int frontier,allocation_slot,master_width,slot_eligible,slot_quota;
    uint64_t weight_numerator,weight_denominator;
    char source_match_id[32],cell[64];
    uint16_t pair_move[2];
    unsigned char priority[32],orbit[32],state_hash[32],mask_hash[2][32],union_hash[32];
    unsigned char state_bytes[PC_STATE_MAX];size_t state_n;
} Allocation;

typedef struct {
    Split split;
    unsigned char discovery_hash[32],reservoir_hash[32];
    unsigned char source_net_hash[32],source_exclusion_hash[32];
    unsigned char commitment_hash[32],allocation_rule_hash[32];
    int is_vector,quota_per_cell,records,source_minimum;
    uint64_t pooled_ge64_observed;
    uint64_t eligible_units,retained_units,total_census;
    uint64_t probe_orbit_rejections;
    uint64_t post_n[3][PC_PLY_BINS][2][PC_MASK_MAX];
    int post_q[3][PC_PLY_BINS][2][PC_MASK_MAX];
    Allocation *row;
} AllocationManifest;

static int clean_token(const char *s)
{
    if(!s||!*s||strlen(s)>=PC_TOKEN_MAX)return 0;
    for(;*s;s++)if(!(isalnum((unsigned char)*s)||strchr("_.:-",*s)))return 0;
    return 1;
}

static int exact_fields(char *line,char **field,int wanted)
{
    int n=0;char *p=line;
    while(1){
        if(n>=wanted)return 0;
        field[n++]=p;
        char *tab=strchr(p,'\t');if(!tab)break;*tab=0;p=tab+1;
    }
    return n==wanted;
}

static int header_value(FILE *f,char **line,size_t *cap,const char *key,
                        char **value)
{
    ssize_t n=getline(line,cap,f);if(n<0||n<2||(*line)[n-1]!='\n')return 0;
    (*line)[n-1]=0;char *tab=strchr(*line,'\t');if(!tab)return 0;
    *tab=0;if(strcmp(*line,key))return 0;*value=tab+1;return **value!=0;
}

static int expected_source_id(Split split,uint64_t match,const char *actual)
{
    char want[32];source_id(split,match,want);return !strcmp(want,actual);
}

static int source_match_in_range(Split split,uint64_t match)
{
    return match<(uint64_t)(split==SPLIT_TRAIN?65536:32768);
}

static void allocation_free(AllocationManifest *m)
{
    free(m->row);memset(m,0,sizeof *m);
}

static int u64_compare(const void *aa,const void *bb)
{
    uint64_t a=*(const uint64_t *)aa,b=*(const uint64_t *)bb;
    return (a>b)-(a<b);
}

typedef struct { uint64_t source;int state; } StateId;
static int state_id_compare(const void *aa,const void *bb)
{
    const StateId *a=(const StateId *)aa,*b=(const StateId *)bb;
    if(a->source!=b->source)return (a->source>b->source)-(a->source<b->source);
    return (a->state>b->state)-(a->state<b->state);
}

static void expected_allocation_rule(int is_vector,unsigned char out[32])
{
    static const char train[]=
        "lc-policy-cost-train-allocation-v2|canonical-greedy-selection|"
        "global-source-unique|quota16|rank-major-diagonal-cell-interleave-v1";
    static const char vector[]=
        "lc-policy-cost-vector-allocation-v1|64-even-feasible-j|priority-v1|"
        "rank-major-three-band-base-interleave-v1";
    sha_bytes(is_vector?vector:train,is_vector?sizeof vector-1:sizeof train-1,out);
}

static void train_scheduled_cell(int row,int *rd,int *pb,int *ratio,int *type)
{
    int position=row%(3*PC_PLY_BINS*PC_RATIO_BINS*PC_PAIR_TYPES);
    *pb=position%PC_PLY_BINS;position/=PC_PLY_BINS;
    *rd=position%3;position/=3;
    int base_ratio=position%PC_RATIO_BINS;
    int base_type=position/PC_RATIO_BINS;
    *ratio=(base_ratio+*pb)%PC_RATIO_BINS;
    *type=(base_type+*rd+*pb)%PC_PAIR_TYPES;
}

static void vector_scheduled_base(int row,int *rd,int *pb,int *frontier)
{
    int position=row%(3*PC_PLY_BINS*2);
    *rd=position%3;position/=3;
    *frontier=position%2;position/=2;
    int band=position%3,low_ply=position/3;
    *pb=low_ply+8*band;
}

static int allocation_tuple_before(const Allocation *left,
                                   const Allocation *right)
{
    int order=memcmp(left->priority,right->priority,32);
    if(order)return order<0;
    order=memcmp(left->state_hash,right->state_hash,32);
    if(order)return order<0;
    if(left->source_match_index!=right->source_match_index)
        return left->source_match_index<right->source_match_index;
    return left->source_state_index<right->source_state_index;
}

static int parse_u64_csv5(const char *text,uint64_t out[5])
{
    const char *p=text;
    for(int i=0;i<5;i++){
        const char *end=i==4?NULL:strchr(p,',');char part[32];size_t n=end?(size_t)(end-p):strlen(p);
        if(n==0||n>=sizeof part||(i==4&&strchr(p,',')))return 0;
        memcpy(part,p,n);part[n]=0;
        if(!parse_u64(part,&out[i]))return 0;
        if(i<4)p=end+1;
    }
    return 1;
}

static int load_allocation(const char *path,const char *expected_sha,
                           AllocationManifest *m,unsigned char file_hash[32])
{
    unsigned char want[32];
    if(!path||!expected_sha||!hex_digest(expected_sha,want)||
       !file_sha256(path,file_hash)||memcmp(want,file_hash,32))return 0;
    FILE *f=fopen(path,"rb");if(!f)return 0;
    memset(m,0,sizeof *m);char *line=NULL,*v=NULL;size_t cap=0;ssize_t n;
    n=getline(&line,&cap,f);if(n<0)goto bad;
    if(!strcmp(line,"LCPOLICYCOST-TRAIN-ALLOCATION-V2\n"))m->is_vector=0;
    else if(!strcmp(line,"LCPOLICYCOST-VECTOR-ALLOCATION-V1\n"))m->is_vector=1;
    else goto bad;
    if(!header_value(f,&line,&cap,"split",&v)||!parse_split(v,&m->split))goto bad;
    if((m->split==SPLIT_TRAIN)==m->is_vector)goto bad;
    if(!header_value(f,&line,&cap,"purpose",&v)||strcmp(v,"campaign"))goto bad;
    if(!header_value(f,&line,&cap,"discovery_sha256",&v)||!hex_digest(v,m->discovery_hash))goto bad;
    if(!header_value(f,&line,&cap,"reservoir_sha256",&v)||!hex_digest(v,m->reservoir_hash))goto bad;
    if(!header_value(f,&line,&cap,"source_net_sha256",&v)||!hex_digest(v,m->source_net_hash))goto bad;
    if(!header_value(f,&line,&cap,"source_exclusion_sha256",&v)||!hex_digest(v,m->source_exclusion_hash))goto bad;
    if(!m->is_vector){
      if(!header_value(f,&line,&cap,"eligible_pair_commitment_sha256",&v)||
         !hex_digest(v,m->commitment_hash)||
         !header_value(f,&line,&cap,"allocation_rule_sha256",&v)||
         !hex_digest(v,m->allocation_rule_hash)||
         !header_value(f,&line,&cap,"quota_per_cell",&v)||
         !parse_int(v,PC_TRAIN_QUOTA,PC_TRAIN_QUOTA,&m->quota_per_cell)||
         !header_value(f,&line,&cap,"eligible_units",&v)||!parse_u64(v,&m->eligible_units)||
         !header_value(f,&line,&cap,"retained_reservoir_units",&v)||!parse_u64(v,&m->retained_units)||
         !header_value(f,&line,&cap,"probe_orbit_rejections",&v)||!parse_u64(v,&m->probe_orbit_rejections)||
         !header_value(f,&line,&cap,"pooled_ge64_observed",&v)||!parse_u64(v,&m->pooled_ge64_observed)||
         !header_value(f,&line,&cap,"records",&v)||!parse_int(v,1,1000000,&m->records)||
         m->records!=3*PC_PLY_BINS*PC_RATIO_BINS*PC_PAIR_TYPES*PC_TRAIN_QUOTA)goto bad;
      n=getline(&line,&cap,f);
      if(n<0||strcmp(line,"columns\tallocation_id\tsource_match_index\t"
          "source_state_index\tsource_match_id\tstate_id\tpair_id\tcell\tround\tply_bin\t"
          "ratio_bin\tpair_type\tpair_move_a\tpair_move_b\torbit_sha256\t"
          "state_sha256\tpair_sha256\tallocation_priority_sha256\tmask_001_sha256\t"
          "mask_002_sha256\tmaster_sha256\tstate_hex\n"))goto bad;
    }else{
      int post_cells=0;
      uint64_t aggregate_width[PC_MASK_MAX]={0},seen_width[PC_MASK_MAX]={0};
      if(!header_value(f,&line,&cap,"eligible_state_commitment_sha256",&v)||
         !hex_digest(v,m->commitment_hash)||
         !header_value(f,&line,&cap,"allocation_rule_sha256",&v)||
         !hex_digest(v,m->allocation_rule_hash)||
         !header_value(f,&line,&cap,"quota_per_base_cell",&v)||
         !parse_int(v,PC_VECTOR_QUOTA,PC_VECTOR_QUOTA,&m->quota_per_cell)||
         !header_value(f,&line,&cap,"source_minimum_per_positive_slot",&v)||
         !parse_int(v,8,8,&m->source_minimum)||
         !header_value(f,&line,&cap,"total_census",&v)||!parse_u64(v,&m->total_census)||
         !header_value(f,&line,&cap,"retained_reservoir_vectors",&v)||!parse_u64(v,&m->retained_units)||
         !header_value(f,&line,&cap,"poststratum_cells",&v)||
         !parse_int(v,720,720,&post_cells)||
         !header_value(f,&line,&cap,"aggregate_master_width_histogram",&v)||
         !parse_u64_csv5(v,aggregate_width)||
         !header_value(f,&line,&cap,"probe_orbit_rejections",&v)||!parse_u64(v,&m->probe_orbit_rejections)||
         !header_value(f,&line,&cap,"pooled_ge64_observed",&v)||!parse_u64(v,&m->pooled_ge64_observed)||
         !header_value(f,&line,&cap,"records",&v)||!parse_int(v,9216,9216,&m->records))goto bad;
      uint64_t sum_n=0;int sum_q=0;
      for(int rd=0;rd<3;rd++)for(int pb=0;pb<PC_PLY_BINS;pb++)
       for(int fr=0;fr<2;fr++)for(int sl=0;sl<PC_MASK_MAX;sl++){
        n=getline(&line,&cap,f);if(n<0||n<2||line[n-1]!='\n')goto bad;
        line[n-1]=0;char *x[7];uint64_t nn,den,width_sum=0,wc[PC_MASK_MAX];int qq;
        if(!exact_fields(line,x,7)||strcmp(x[0],"poststratum")||
           !parse_u64(x[2],&nn)||!parse_int(x[3],0,PC_VECTOR_QUOTA,&qq)||
           !parse_u64(x[4],&den)||den!=nn||!parse_u64(x[5],&den)||den!=m->total_census)goto bad;
        if(!parse_u64_csv5(x[6],wc))goto bad;
        for(int w=0;w<PC_MASK_MAX;w++){width_sum+=wc[w];seen_width[w]+=wc[w];
          /* J is SHA mod K, so a J slot cannot contain K <= J. */
          if(w<sl&&wc[w])goto bad;}
        char id[64];snprintf(id,sizeof id,"r%d:p%02d:f%d:j%d",rd,pb,fr,sl);
        if(strcmp(id,x[1])||(nn==0)!=(qq==0)||nn<(uint64_t)qq||width_sum!=nn)goto bad;
        m->post_n[rd][pb][fr][sl]=nn;m->post_q[rd][pb][fr][sl]=qq;
        sum_n+=nn;sum_q+=qq;
      }
      {uint64_t aggregate_sum=0;for(int w=0;w<PC_MASK_MAX;w++){
         aggregate_sum+=aggregate_width[w];if(aggregate_width[w]!=seen_width[w])goto bad;}
       if(aggregate_sum!=m->total_census)goto bad;}
      if(sum_n!=m->total_census||sum_q!=m->records)goto bad;
      n=getline(&line,&cap,f);
      if(n<0||strcmp(line,"columns\tallocation_id\tsource_match_index\t"
          "source_state_index\tsource_match_id\tunit\tround\tply_stratum\tfrontier_present\t"
          "allocation_slot\tpost_stratum\tmaster_width\tcensus_count\tallocation_quota\t"
          "weight_numerator\tweight_denominator\torbit_sha256\tstate_sha256\tallocation_priority_sha256\t"
          "mask_001_sha256\tmask_002_sha256\tmaster_sha256\tstate_hex\tdiscovery_sha256\n"))goto bad;
    }
    {unsigned char expected_rule[32];expected_allocation_rule(m->is_vector,expected_rule);
     if(memcmp(expected_rule,m->allocation_rule_hash,32))goto bad;}
    m->row=(Allocation *)calloc((size_t)m->records,sizeof *m->row);if(!m->row)goto bad;
    int train_counts[3][PC_PLY_BINS][PC_RATIO_BINS][PC_PAIR_TYPES]={0};
    int vector_counts[3][PC_PLY_BINS][2][PC_MASK_MAX]={0};
    int train_previous[3][PC_PLY_BINS][PC_RATIO_BINS][PC_PAIR_TYPES];
    int vector_previous[3][PC_PLY_BINS][2][PC_MASK_MAX];
    for(int rd=0;rd<3;rd++)for(int pb=0;pb<PC_PLY_BINS;pb++){
      for(int ratio=0;ratio<PC_RATIO_BINS;ratio++)
       for(int type=0;type<PC_PAIR_TYPES;type++)
        train_previous[rd][pb][ratio][type]=-1;
      for(int frontier=0;frontier<2;frontier++)
       for(int slot=0;slot<PC_MASK_MAX;slot++)
        vector_previous[rd][pb][frontier][slot]=-1;
    }
    size_t expected_state_n=0;
    { State z;memset(&z,0,sizeof z);unsigned char b[PC_STATE_MAX];
      if(!encode_view(&z,b,&expected_state_n))goto bad; }
    for(int r=0;r<m->records;r++){
        n=getline(&line,&cap,f);if(n<0||n<2||line[n-1]!='\n')goto bad;
        line[n-1]=0;Allocation *a=&m->row[r];uint64_t temp;
        if(!m->is_vector){
          char *x[21];if(!exact_fields(line,x,21)||
             !parse_u64(x[0],&a->allocation_id)||a->allocation_id!=(uint64_t)r||
             !parse_u64(x[1],&a->source_match_index)||
             !source_match_in_range(m->split,a->source_match_index)||
             !parse_int(x[2],0,3*LC_MAX_PLIES-1,&a->source_state_index)||
             !clean_token(x[3])||strlen(x[3])>=sizeof a->source_match_id||
             !expected_source_id(m->split,a->source_match_index,x[3])||
             !clean_token(x[4])||!clean_token(x[5])||
             !clean_token(x[6])||strlen(x[6])>=sizeof a->cell||
             !parse_int(x[7],0,2,&a->round)||!parse_int(x[8],0,PC_PLY_BINS-1,&a->ply_bin)||
             !parse_int(x[9],0,PC_RATIO_BINS-1,&a->ratio_bin)||
             !parse_int(x[10],0,PC_PAIR_TYPES-1,&a->pair_type)||
             !parse_u64(x[11],&temp)||temp>UINT16_MAX)goto bad;
          a->pair_move[0]=(uint16_t)temp;
          if(!parse_u64(x[12],&temp)||temp>UINT16_MAX)goto bad;
          a->pair_move[1]=(uint16_t)temp;
          unsigned char pair_hash[32],claimed_pair_hash[32];
          if(a->pair_move[0]==a->pair_move[1]||!hex_digest(x[13],a->orbit)||
             !hex_digest(x[14],a->state_hash)||!hex_digest(x[15],claimed_pair_hash)||
             !hex_digest(x[16],a->priority)||!hex_digest(x[17],a->mask_hash[0])||
             !hex_digest(x[18],a->mask_hash[1])||!hex_digest(x[19],a->union_hash)||
             !parse_hex_bytes(x[20],a->state_bytes,expected_state_n))goto bad;
          char state_id[80],pair_id[32];snprintf(state_id,sizeof state_id,"%s:s%03d",x[3],a->source_state_index);
          snprintf(pair_id,sizeof pair_id,"%05u-%05u",a->pair_move[0],a->pair_move[1]);
          pair_state_digest(a->state_hash,a->pair_move[0],a->pair_move[1],pair_hash);
          if(strcmp(state_id,x[4])||strcmp(pair_id,x[5])||memcmp(pair_hash,claimed_pair_hash,32))goto bad;
          strcpy(a->source_match_id,x[3]);strcpy(a->cell,x[6]);
          char id[64];snprintf(id,sizeof id,"r%d.p%d.g%d.t%d",a->round,a->ply_bin,a->ratio_bin,a->pair_type);
          int er,ep,eg,et;train_scheduled_cell(r,&er,&ep,&eg,&et);
          if(strcmp(id,a->cell)||a->round!=er||a->ply_bin!=ep||
             a->ratio_bin!=eg||a->pair_type!=et)goto bad;
          int prior=train_previous[a->round][a->ply_bin][a->ratio_bin][a->pair_type];
          if(prior>=0&&!allocation_tuple_before(&m->row[prior],a))goto bad;
          train_previous[a->round][a->ply_bin][a->ratio_bin][a->pair_type]=r;
          train_counts[a->round][a->ply_bin][a->ratio_bin][a->pair_type]++;
        }else{
          char *x[23];uint64_t census,quota;
          if(!exact_fields(line,x,23)||!parse_u64(x[0],&a->allocation_id)||
             a->allocation_id!=(uint64_t)r||!parse_u64(x[1],&a->source_match_index)||
             !source_match_in_range(m->split,a->source_match_index)||
             !parse_int(x[2],0,3*LC_MAX_PLIES-1,&a->source_state_index)||
             !clean_token(x[3])||strlen(x[3])>=sizeof a->source_match_id||
             !expected_source_id(m->split,a->source_match_index,x[3])||
             !clean_token(x[4])||!parse_int(x[5],0,2,&a->round)||
             !parse_int(x[6],0,PC_PLY_BINS-1,&a->ply_bin)||!parse_int(x[7],0,1,&a->frontier)||
             !parse_int(x[8],0,PC_MASK_MAX-1,&a->allocation_slot)||
             !clean_token(x[9])||strlen(x[9])>=sizeof a->cell||
             !parse_int(x[10],1,PC_MASK_MAX,&a->master_width)||a->allocation_slot>=a->master_width||
             !parse_u64(x[11],&census)||!parse_u64(x[12],&quota)||
             !parse_u64(x[13],&a->weight_numerator)||!parse_u64(x[14],&a->weight_denominator)||
             !hex_digest(x[15],a->orbit)||!hex_digest(x[16],a->state_hash)||
             !hex_digest(x[17],a->priority)||!hex_digest(x[18],a->mask_hash[0])||
             !hex_digest(x[19],a->mask_hash[1])||!hex_digest(x[20],a->union_hash)||
             !parse_hex_bytes(x[21],a->state_bytes,expected_state_n)||
             !hex_digest(x[22],want)||memcmp(want,m->discovery_hash,32))goto bad;
          char unit[80];snprintf(unit,sizeof unit,"%s:s%03d",x[3],a->source_state_index);
          if(strcmp(unit,x[4]))goto bad;
          char id[64];snprintf(id,sizeof id,"r%d:p%02d:f%d:j%d",a->round,a->ply_bin,
                               a->frontier,a->allocation_slot);
          if(strcmp(id,x[9])||census!=m->post_n[a->round][a->ply_bin][a->frontier][a->allocation_slot]||
             quota!=(uint64_t)m->post_q[a->round][a->ply_bin][a->frontier][a->allocation_slot]||
             a->weight_numerator!=census||a->weight_denominator!=quota*m->total_census)goto bad;
          a->slot_eligible=(int)census;a->slot_quota=(int)quota;
          strcpy(a->source_match_id,x[3]);strcpy(a->cell,x[9]);
          int er,ep,ef;vector_scheduled_base(r,&er,&ep,&ef);
          if(a->round!=er||a->ply_bin!=ep||a->frontier!=ef)goto bad;
          int prior=vector_previous[a->round][a->ply_bin][a->frontier][a->allocation_slot];
          if(prior>=0&&!allocation_tuple_before(&m->row[prior],a))goto bad;
          vector_previous[a->round][a->ply_bin][a->frontier][a->allocation_slot]=r;
          vector_counts[a->round][a->ply_bin][a->frontier][a->allocation_slot]++;
        }
        a->state_n=expected_state_n;
    }
    if(!m->is_vector){
      for(int rd=0;rd<3;rd++)for(int p=0;p<PC_PLY_BINS;p++)
       for(int g=0;g<PC_RATIO_BINS;g++)for(int t=0;t<PC_PAIR_TYPES;t++)
        if(train_counts[rd][p][g][t]!=PC_TRAIN_QUOTA)goto bad;
      uint64_t *source=(uint64_t *)malloc((size_t)m->records*sizeof *source);if(!source)goto bad;
      for(int i=0;i<m->records;i++)source[i]=m->row[i].source_match_index;
      qsort(source,(size_t)m->records,sizeof *source,u64_compare);
      for(int i=1;i<m->records;i++)if(source[i]==source[i-1]){free(source);goto bad;}
      free(source);
    }else{
      for(int rd=0;rd<3;rd++)for(int p=0;p<PC_PLY_BINS;p++)for(int fr=0;fr<2;fr++){
        int active=0;for(int sl=0;sl<PC_MASK_MAX;sl++)if(m->post_n[rd][p][fr][sl]>0)active++;
        if(active<1)goto bad;
        int even=PC_VECTOR_QUOTA/active,remainder=PC_VECTOR_QUOTA%active;
        int base=0,active_index=0;for(int sl=0;sl<PC_MASK_MAX;sl++){
          int expected_q=0;
          if(m->post_n[rd][p][fr][sl]>0)expected_q=even+(active_index++<remainder);
          if(m->post_q[rd][p][fr][sl]!=expected_q)goto bad;
          if(vector_counts[rd][p][fr][sl]!=m->post_q[rd][p][fr][sl])goto bad;
          if(m->post_q[rd][p][fr][sl]>0){
            int unique=0;
            for(int i=0;i<m->records;i++)if(m->row[i].round==rd&&m->row[i].ply_bin==p&&
                 m->row[i].frontier==fr&&m->row[i].allocation_slot==sl){
              int seen=0;for(int j=0;j<i;j++)if(m->row[j].round==rd&&m->row[j].ply_bin==p&&
                 m->row[j].frontier==fr&&m->row[j].allocation_slot==sl&&
                 m->row[j].source_match_index==m->row[i].source_match_index){seen=1;break;}
              if(!seen)unique++;
            }
            if(unique<m->source_minimum)goto bad;
          }
          base+=vector_counts[rd][p][fr][sl];
        }if(base!=PC_VECTOR_QUOTA)goto bad;
      }
      StateId *id=(StateId *)malloc((size_t)m->records*sizeof *id);if(!id)goto bad;
      for(int i=0;i<m->records;i++){id[i].source=m->row[i].source_match_index;id[i].state=m->row[i].source_state_index;}
      qsort(id,(size_t)m->records,sizeof *id,state_id_compare);
      for(int i=1;i<m->records;i++)if(!state_id_compare(&id[i-1],&id[i])){free(id);goto bad;}
      free(id);
    }
    if(getline(&line,&cap,f)>=0||ferror(f))goto bad;
    free(line);fclose(f);return 1;
bad:
    free(line);fclose(f);allocation_free(m);return 0;
}

typedef struct {
    Allocation value;
    unsigned char priority[32];
} BoundReservoirRow;

typedef struct {
    Split split;uint64_t seed,eligible,retained,rejected,pooled;
    unsigned char net_hash[32],exclusion_hash[32],commitment_hash[32];
    int is_vector,cap,n,allocated,indexed;BoundReservoirRow *row;
} BoundReservoir;

static void bound_reservoir_free(BoundReservoir *r)
{
    free(r->row);memset(r,0,sizeof *r);
}

/* The allocation manifest contains only a small outcome-blind sample of a
 * potentially 884,736-row TRAIN reservoir.  A linear membership scan for
 * every allocation would therefore perform more than twelve billion rich
 * comparisons before opening a single P/F/T panel.  Sort once by the exact
 * allocation identity and use a logarithmic lookup.  Duplicate identities
 * are invalid rather than order-dependent. */
static int allocation_key_compare(const Allocation *a,const Allocation *b)
{
#define CMP_FIELD(field) do { \
    if(a->field<b->field)return -1; \
    if(a->field>b->field)return 1; \
} while(0)
    CMP_FIELD(source_match_index);CMP_FIELD(source_state_index);
    CMP_FIELD(round);CMP_FIELD(ply_bin);CMP_FIELD(ratio_bin);CMP_FIELD(pair_type);
    CMP_FIELD(pair_move[0]);CMP_FIELD(pair_move[1]);CMP_FIELD(frontier);
    CMP_FIELD(allocation_slot);CMP_FIELD(master_width);
#undef CMP_FIELD
    return strcmp(a->cell,b->cell);
}

static int bound_row_compare(const void *aa,const void *bb)
{
    const BoundReservoirRow *a=(const BoundReservoirRow *)aa;
    const BoundReservoirRow *b=(const BoundReservoirRow *)bb;
    return allocation_key_compare(&a->value,&b->value);
}

static int bound_reservoir_index(BoundReservoir *r)
{
    if(!r||r->n<0||(r->n>0&&!r->row))return 0;
    if(r->n>1)qsort(r->row,(size_t)r->n,sizeof *r->row,bound_row_compare);
    for(int i=1;i<r->n;i++)
        if(!allocation_key_compare(&r->row[i-1].value,&r->row[i].value))
            return 0;
    r->indexed=1;return 1;
}

static int reservoir_counts_match_manifest(const BoundReservoir *r,
                                            const AllocationManifest *m)
{
    if(!r||!m||r->is_vector!=m->is_vector||
       r->retained!=m->retained_units)return 0;
    return r->is_vector?r->eligible==m->total_census:
                        r->eligible==m->eligible_units;
}

static int load_bound_reservoir(const char *path,const char *expected_sha,
                                const AllocationManifest *manifest,
                                BoundReservoir *out,unsigned char file_hash[32])
{
    unsigned char want[32];
    if(!path||!expected_sha||!hex_digest(expected_sha,want)||
       !file_sha256(path,file_hash)||memcmp(want,file_hash,32)||
       memcmp(want,manifest->reservoir_hash,32))return 0;
    FILE *f=fopen(path,"rb");if(!f)return 0;
    memset(out,0,sizeof *out);char *line=NULL,*v=NULL;size_t cap=0;ssize_t n;
    n=getline(&line,&cap,f);if(n<0)goto bad;
    if(!strcmp(line,"LCPOLICYCOST-TRAIN-RESERVOIR-V3\n"))out->is_vector=0;
    else if(!strcmp(line,"LCPOLICYCOST-VECTOR-RESERVOIR-V1\n"))out->is_vector=1;
    else goto bad;
    if(!header_value(f,&line,&cap,"split",&v)||!parse_split(v,&out->split)||
       out->split!=manifest->split||out->is_vector!=manifest->is_vector)goto bad;
    if(!header_value(f,&line,&cap,"purpose",&v)||strcmp(v,"campaign"))goto bad;
    if(!header_value(f,&line,&cap,"seed",&v)||!parse_u64(v,&out->seed)||
       out->seed!=SPLIT_DOMAIN[out->split].discovery_seed)goto bad;
    if(!header_value(f,&line,&cap,"net_sha256",&v)||!hex_digest(v,out->net_hash) ||
       memcmp(out->net_hash,manifest->source_net_hash,32))goto bad;
    if(!header_value(f,&line,&cap,"exclusion_sha256",&v)||!hex_digest(v,out->exclusion_hash) ||
       memcmp(out->exclusion_hash,manifest->source_exclusion_hash,32))goto bad;
    if(!header_value(f,&line,&cap,"reservoir_per_subcell",&v)||
       !parse_int(v,out->is_vector?PC_VECTOR_QUOTA:1024,
                    out->is_vector?PC_VECTOR_QUOTA:1024,&out->cap))goto bad;
    n=getline(&line,&cap,f);
    if(n<0||(!out->is_vector&&strcmp(line,"columns\tcell\tpriority_sha256\tsource_match_index\t"
        "source_state_index\tsource_match_id\tround\tply_bin\tratio_bin\t"
        "pair_type\tpair_move_a\tpair_move_b\torbit_sha256\tstate_sha256\t"
        "mask_001_sha256\tmask_002_sha256\tmaster_sha256\tstate_hex\n"))||
       (out->is_vector&&strcmp(line,"columns\tcell\tpriority_sha256\tsource_match_index\t"
        "source_state_index\tsource_match_id\tround\tply_bin\tfrontier_present\t"
        "allocation_slot\tmaster_width\torbit_sha256\tstate_sha256\t"
        "mask_001_sha256\tmask_002_sha256\tmaster_sha256\tstate_hex\n")))goto bad;
    size_t expected_state_n=0;{State z;memset(&z,0,sizeof z);unsigned char b[PC_STATE_MAX];
        if(!encode_view(&z,b,&expected_state_n))goto bad;}
    while((n=getline(&line,&cap,f))>=0){
        if(!strncmp(line,"footer\t",7))break;
        if(n<2||line[n-1]!='\n')goto bad;
        line[n-1]=0;char *x[17];int wanted_fields=out->is_vector?16:17;
        if(!exact_fields(line,x,wanted_fields))goto bad;
        if(out->n==out->allocated){int next=out->allocated?out->allocated*2:4096;
            BoundReservoirRow *nr=(BoundReservoirRow *)realloc(out->row,(size_t)next*sizeof *nr);
            if(!nr)goto bad;
            out->row=nr;out->allocated=next;}
        BoundReservoirRow *rr=&out->row[out->n];memset(rr,0,sizeof *rr);Allocation *a=&rr->value;
        uint64_t temp;
        if(!clean_token(x[0])||strlen(x[0])>=sizeof a->cell||!hex_digest(x[1],rr->priority)||
           !parse_u64(x[2],&a->source_match_index)||
           !source_match_in_range(out->split,a->source_match_index)||
           !parse_int(x[3],0,3*LC_MAX_PLIES-1,&a->source_state_index)||
           !clean_token(x[4])||strlen(x[4])>=sizeof a->source_match_id||
           !expected_source_id(out->split,a->source_match_index,x[4])||
           !parse_int(x[5],0,2,&a->round)||!parse_int(x[6],0,PC_PLY_BINS-1,&a->ply_bin))goto bad;
        unsigned char priority[32];char cell[64];
        if(!out->is_vector){
          if(!parse_int(x[7],0,PC_RATIO_BINS-1,&a->ratio_bin)||
             !parse_int(x[8],0,PC_PAIR_TYPES-1,&a->pair_type)||
             !parse_u64(x[9],&temp)||temp>UINT16_MAX)goto bad;
          a->pair_move[0]=(uint16_t)temp;
          if(!parse_u64(x[10],&temp)||temp>UINT16_MAX)goto bad;
          a->pair_move[1]=(uint16_t)temp;
          if(!hex_digest(x[11],a->orbit)||!hex_digest(x[12],a->state_hash)||
             !hex_digest(x[13],a->mask_hash[0])||!hex_digest(x[14],a->mask_hash[1])||
             !hex_digest(x[15],a->union_hash)||!parse_hex_bytes(x[16],a->state_bytes,expected_state_n))goto bad;
          snprintf(cell,sizeof cell,"r%d.p%d.g%d.t%d",a->round,a->ply_bin,a->ratio_bin,a->pair_type);
          ReservoirEntry e;memset(&e,0,sizeof e);e.source_match_index=a->source_match_index;
          e.source_state_index=a->source_state_index;e.round=a->round;e.pbin=a->ply_bin;
          e.rbin=a->ratio_bin;e.type=a->pair_type;e.pair_move[0]=a->pair_move[0];
          e.pair_move[1]=a->pair_move[1];memcpy(e.state_hash,a->state_hash,32);
          entry_priority(out->seed,&e,priority);
        }else{
          if(!parse_int(x[7],0,1,&a->frontier)||
             !parse_int(x[8],0,PC_MASK_MAX-1,&a->allocation_slot)||
             !parse_int(x[9],1,PC_MASK_MAX,&a->master_width)||
             a->allocation_slot>=a->master_width||!hex_digest(x[10],a->orbit)||
             !hex_digest(x[11],a->state_hash)||!hex_digest(x[12],a->mask_hash[0])||
             !hex_digest(x[13],a->mask_hash[1])||!hex_digest(x[14],a->union_hash)||
             memcmp(a->mask_hash[0],a->union_hash,32)||
             !parse_hex_bytes(x[15],a->state_bytes,expected_state_n))goto bad;
          snprintf(cell,sizeof cell,"r%d:p%02d:f%d:j%d",a->round,a->ply_bin,
                   a->frontier,a->allocation_slot);
          VectorEntry e;memset(&e,0,sizeof e);e.source_match_index=a->source_match_index;
          e.source_state_index=a->source_state_index;e.round=a->round;e.pbin=a->ply_bin;
          e.frontier=a->frontier;e.allocation_slot=a->allocation_slot;e.master_width=a->master_width;
          memcpy(e.state_hash,a->state_hash,32);
          if(vector_allocation_slot(out->seed,a->state_hash,a->master_width)!=a->allocation_slot)goto bad;
          vector_priority(out->seed,&e,priority);
        }
        a->state_n=expected_state_n;strcpy(a->cell,x[0]);strcpy(a->source_match_id,x[4]);
        if(strcmp(cell,a->cell)||memcmp(priority,rr->priority,32))goto bad;
        memcpy(a->priority,rr->priority,32);
        out->n++;
    }
    if(n<0||strncmp(line,"footer\t",7))goto bad;
    line[n-1]=0;char *foot[11];if(!exact_fields(line,foot,11)||strcmp(foot[0],"footer")||
       strcmp(foot[1],"eligible_units")||!parse_u64(foot[2],&out->eligible)||
       strcmp(foot[3],"retained_units")||!parse_u64(foot[4],&out->retained)||
       strcmp(foot[5],"rejected_by_bound")||!parse_u64(foot[6],&out->rejected)||
       strcmp(foot[7],"state_commitment_chain_sha256")||!hex_digest(foot[8],out->commitment_hash)||
       strcmp(foot[9],"pooled_ge64_observed")||!parse_u64(foot[10],&out->pooled)||
       out->retained!=(uint64_t)out->n||out->eligible!=out->retained+out->rejected||
       out->pooled!=manifest->pooled_ge64_observed||
       memcmp(out->commitment_hash,manifest->commitment_hash,32)||
       !reservoir_counts_match_manifest(out,manifest)||
       !bound_reservoir_index(out))goto bad;
    if(getline(&line,&cap,f)>=0||ferror(f))goto bad;
    free(line);fclose(f);return 1;
bad:
    free(line);fclose(f);bound_reservoir_free(out);return 0;
}

static int allocation_index_position(const Allocation *a,const BoundReservoir *r,
                                     int *comparison_count)
{
    if(comparison_count)*comparison_count=0;
    if(!a||!r||!r->indexed||r->n<1)return -1;
    int lo=0,hi=r->n;
    while(lo<hi){
        int mid=lo+(hi-lo)/2;
        int cmp=allocation_key_compare(a,&r->row[mid].value);
        if(comparison_count)(*comparison_count)++;
        if(cmp>0)lo=mid+1;else hi=mid;
    }
    if(lo>=r->n)return -1;
    if(comparison_count)(*comparison_count)++;
    return allocation_key_compare(a,&r->row[lo].value)?-1:lo;
}

static int allocation_in_reservoir(const Allocation *a,const BoundReservoir *r)
{
    int position=allocation_index_position(a,r,NULL);
    if(position<0)return 0;
    const Allocation *b=&r->row[position].value;
    return !strcmp(a->source_match_id,b->source_match_id)&&
           !memcmp(a->priority,b->priority,32)&&
           !memcmp(a->orbit,b->orbit,32)&&!memcmp(a->state_hash,b->state_hash,32)&&
           !memcmp(a->mask_hash,b->mask_hash,sizeof a->mask_hash)&&
           !memcmp(a->union_hash,b->union_hash,32)&&a->state_n==b->state_n&&
           !memcmp(a->state_bytes,b->state_bytes,a->state_n);
}

static void shuffle_deck(uint8_t deck[NCARD],Rng *rng)
{
    for(int i=0;i<NCARD;i++)deck[i]=(uint8_t)i;
    for(int i=NCARD-1;i>0;i--){uint32_t j=rng_below(rng,(uint32_t)i+1);
        uint8_t x=deck[i];deck[i]=deck[j];deck[j]=x;}
}

static int exact_policy_move(const Net *net,const State *complete,Move *out)
{
    State view;agent_information_view(complete,complete->turn,&view);
    Move mv[MAX_MOVES];float prob[MAX_MOVES];
    int n=policy_probs_sym(net,&view,mv,prob,NULL,PC_SYMMETRIES);if(n<1)return 0;
    int best=0;for(int i=1;i<n;i++)if(prob[i]>prob[best])best=i;
    *out=mv[best];return 1;
}

typedef struct { double round_margin,final_margin,match_score,hybrid; } TruthValue;

static int truth_finish(State *root,int perspective,
                        const uint8_t future[MATCH_ROUNDS][NCARD],
                        const Net *net,TruthValue *out)
{
    int cumulative[2]={root->cum[0],root->cum[1]};
    for(int round=root->round;round<MATCH_ROUNDS;round++){
        State s;
        if(round==root->round)s=*root;
        else{lc_deal_from_deck(&s,future[round]);s.round=(uint8_t)round;
             s.turn=(uint8_t)(round&1);s.cum[0]=(int16_t)cumulative[0];
             s.cum[1]=(int16_t)cumulative[1];}
        while(!s.over){Move m;if(!exact_policy_move(net,&s,&m))return 0;lc_apply(&s,m);}
        if(s.deck_left>0)return 0;
        int rm=lc_score(&s,perspective)-lc_score(&s,perspective^1);
        if(round==root->round)out->round_margin=(double)rm;
        cumulative[0]+=lc_score(&s,0);cumulative[1]+=lc_score(&s,1);
    }
    int fm=cumulative[perspective]-cumulative[perspective^1];
    int win=(fm>0)-(fm<0);out->final_margin=(double)fm;
    out->match_score=win>0?1.0:win==0?0.5:0.0;
    out->hybrid=0.05*(double)fm+50.0*(double)win;return 1;
}

typedef struct {
    int n,worlds;
    double *round_margin,*final_margin,*match_score,*hybrid;
    uint64_t cap_hits;
    unsigned char hidden_hash[32],future_hash[32];
} TruthPanel;

static void truth_free(TruthPanel *t)
{
    free(t->round_margin);free(t->final_margin);free(t->match_score);free(t->hybrid);
    memset(t,0,sizeof *t);
}

static int truth_panel(const State *view,const Move *mv,const int *uindex,int n,
                       const Net *net,uint64_t seed,uint64_t source_match,
                       uint64_t state_index,int worlds,TruthPanel *out)
{
    memset(out,0,sizeof *out);out->n=n;out->worlds=worlds;
    size_t cells=(size_t)n*(size_t)worlds;
    out->round_margin=(double *)malloc(sizeof(double)*cells);
    out->final_margin=(double *)malloc(sizeof(double)*cells);
    out->match_score=(double *)malloc(sizeof(double)*cells);
    out->hybrid=(double *)malloc(sizeof(double)*cells);
    if(!out->round_margin||!out->final_margin||!out->match_score||!out->hybrid){truth_free(out);return 0;}
    Sha256 hidden,future_hash;sha256_init(&hidden);sha256_init(&future_hash);
    for(int w=0;w<worlds;w++){
        Rng wrng;rng_seed(&wrng,domain_seed(seed,source_match,state_index,w,TAG_TRUTH_WORLD));
        State world;determinize(view,view->turn,&wrng,&world);
        unsigned char wb[PC_STATE_MAX];size_t nw=0;
        /* A complete world is deliberately hashed as raw bytes only for this
         * compiler-bound evidence.  Selection never consumes this hash. */
        sha256_update(&hidden,&world,sizeof world);
        uint8_t future[MATCH_ROUNDS][NCARD];memset(future,0,sizeof future);
        for(int rd=view->round+1;rd<MATCH_ROUNDS;rd++){
            Rng frng;rng_seed(&frng,domain_seed(seed,source_match,state_index,
                ((uint64_t)w<<8)|(uint64_t)rd,TAG_TRUTH_FUTURE));
            shuffle_deck(future[rd],&frng);sha256_update(&future_hash,future[rd],NCARD);
        }
        (void)wb;(void)nw;
        for(int c=0;c<n;c++){
            State branch=world;lc_apply(&branch,mv[uindex[c]]);TruthValue v={0};
            if(!truth_finish(&branch,view->turn,future,net,&v)){out->cap_hits++;continue;}
            size_t at=(size_t)c*(size_t)worlds+(size_t)w;
            out->round_margin[at]=v.round_margin;out->final_margin[at]=v.final_margin;
            out->match_score[at]=v.match_score;out->hybrid[at]=v.hybrid;
        }
    }
    sha256_final(&hidden,out->hidden_hash);sha256_final(&future_hash,out->future_hash);
    return out->cap_hits==0;
}

static void sample_digest(const double *x,int n,unsigned char out[32])
{
    Sha256 s;sha256_init(&s);
    for(int i=0;i<n;i++){
        uint64_t bits;memcpy(&bits,&x[i],sizeof bits);unsigned char b[8];
        for(int j=0;j<8;j++)b[j]=(unsigned char)(bits>>(8*j));
        sha256_update(&s,b,sizeof b);
    }
    sha256_final(&s,out);
}

static void sample_moments(const double *x,int n,double *sum,double *sumsq,
                           double *mean,double *se)
{
    *sum=0.0;*sumsq=0.0;
    for(int i=0;i<n;i++){*sum+=x[i];*sumsq+=x[i]*x[i];}
    *mean=*sum/n;double centered=*sumsq-(double)n*(*mean)*(*mean);
    if(centered<0.0&&centered>-1e-9)centered=0.0;
    *se=sqrt(centered/(n-1)/n);
}

static void print_sample_metric(FILE *f,const char *name,const double *x,
                                int ncand,int worlds)
{
    fprintf(f,"\"%s\":{\"actions\":[",name);
    for(int c=0;c<ncand;c++){
        const double *v=x+(size_t)c*(size_t)worlds;
        double sum,sumsq,mean,se;unsigned char d[32];char h[65];
        sample_moments(v,worlds,&sum,&sumsq,&mean,&se);sample_digest(v,worlds,d);digest_hex(d,h);
        fprintf(f,"%s{\"position\":%d,\"mean\":%.17g,\"se\":%.17g,"
                  "\"sum\":%.17g,\"sum_squares\":%.17g,"
                  "\"samples_sha256\":\"%s\"}",c?",":"",c,mean,se,sum,sumsq,h);
    }
    fputs("],\"pairs\":[",f);int first=1;
    for(int a=0;a<ncand;a++)for(int b=a+1;b<ncand;b++){
        const double *xv=x+(size_t)a*worlds,*yv=x+(size_t)b*worlds;
        double sx,sx2,mx,sex,sy,sy2,my,sey,cross=0.0,ssd=0.0;
        sample_moments(xv,worlds,&sx,&sx2,&mx,&sex);
        sample_moments(yv,worlds,&sy,&sy2,&my,&sey);
        for(int w=0;w<worlds;w++){cross+=xv[w]*yv[w];double z=(xv[w]-yv[w])-(mx-my);ssd+=z*z;}
        fprintf(f,"%s{\"a\":%d,\"b\":%d,\"delta_a_minus_b\":%.17g,"
                  "\"paired_se\":%.17g,\"sum_products\":%.17g}",
                first?"":",",a,b,mx-my,sqrt(ssd/(worlds-1)/worlds),cross);first=0;
    }
    fputs("]}",f);
}

typedef struct {
    RolloutAuditPanel audit[PC_MASKS];
    PolicyCostDecision decision[PC_MASKS];
    int have_decision[PC_MASKS];
} SearchPanels;

/* SELECT's two-percent support is a frozen ordered subset of the complete
 * one-percent master.  Search the master exactly once, then project every
 * first- and second-moment entry into that no-refill subset.  This is both
 * cheaper and stronger than hoping two nominally identical reruns remain
 * bit-exact: every reported subset statistic is literally one entry of the
 * same max-five fixed-world vector. */
static int derive_audit_subset(const RolloutAuditPanel *master,
                               const RuntimeMask *master_mask,
                               const RuntimeMask *subset_mask,
                               RolloutAuditPanel *subset)
{
    if(!master||!master_mask||!subset_mask||!subset||
       master->n!=master_mask->n||master->n<1||
       master->n>PC_MASK_MAX||subset_mask->n<1||
       subset_mask->n>master_mask->n||master->baseline!=0||
       subset_mask->index[0]!=master_mask->index[0])return 0;
    int map[PC_MASK_MAX],cursor=0;
    for(int c=0;c<subset_mask->n;c++){
        while(cursor<master_mask->n&&
              master_mask->index[cursor]!=subset_mask->index[c])cursor++;
        if(cursor>=master_mask->n)return 0;
        map[c]=cursor++;
    }
    *subset=*master;
    subset->n=subset_mask->n;subset->baseline=0;subset->selected=0;
    for(int c=0;c<subset_mask->n;c++){
        int source=map[c];
        subset->q[c]=master->q[source];subset->se[c]=master->se[source];
        subset->delta[c]=master->delta[source];
        subset->delta_se[c]=master->delta_se[source];
        if(c>0&&subset->q[c]>subset->q[subset->selected])subset->selected=c;
        for(int rival=0;rival<subset_mask->n;rival++){
            subset->pair_delta[c][rival]=
                master->pair_delta[source][map[rival]];
            subset->pair_delta_se[c][rival]=
                master->pair_delta_se[source][map[rival]];
        }
    }
    return 1;
}

static int independent_hidden_panel_or_complete_census(
    const RolloutAuditPanel *primary,const RolloutAuditPanel *fresh)
{
    if(!primary||!fresh)return 0;
    if(primary->hidden_world_fingerprint!=fresh->hidden_world_fingerprint)
        return 1;
    /* When both roles enumerate the entire intrinsic finite hidden support,
     * equality of the ordered-world fingerprint is not reuse of a random
     * sample.  Never extend this exception to an 800-world or partial panel. */
    return primary->requested_worlds==PC_PRIMARY_WORLDS&&
           fresh->requested_worlds==PC_PRIMARY_WORLDS&&
           primary->exact_hidden_support&&fresh->exact_hidden_support&&
           primary->worlds>=2&&primary->worlds<PC_PRIMARY_WORLDS&&
           fresh->worlds==primary->worlds&&
           primary->hidden_support==primary->worlds&&
           fresh->hidden_support==fresh->worlds;
}

static int evaluate_search_panels(const Agent *actor,const PolicyCostTable *table,
                                  const State *view,const Move *legal,
                                  const float *prob,int nlegal,
                                  const RuntimeMask mask[PC_MASKS],
                                  int enabled_bits,int panel_role,uint64_t seed,double z,
                                  SearchPanels *out)
{
    memset(out,0,sizeof *out);
    if(enabled_bits<1||enabled_bits>3)return 0;
    int evaluated=enabled_bits==2?1:0;
    Move cand[PC_MASK_MAX];
    for(int c=0;c<mask[evaluated].n;c++)
        cand[c]=legal[mask[evaluated].index[c]];
    int rc=rollout_audit_panel_role(actor,view,cand,mask[evaluated].n,0,
             panel_role,seed,PC_PRIMARY_WORLDS,&out->audit[evaluated]);
    if(rc||out->audit[evaluated].requested_worlds!=PC_PRIMARY_WORLDS||
       out->audit[evaluated].worlds<2||
       out->audit[evaluated].unfinished_cap_leaves!=0||
       out->audit[evaluated].panel_role!=panel_role)return 0;
    if(enabled_bits==3&&!derive_audit_subset(
            &out->audit[0],&mask[0],&mask[1],&out->audit[1]))return 0;
    for(int m=0;m<PC_MASKS;m++){
        if(!(enabled_bits&(1<<m)))continue;
        if(table){
            out->have_decision[m]=policy_cost_decide_summary(table,view->round,
                view->nply,legal,prob,nlegal,mask[m].index,mask[m].n,
                out->audit[m].q,&out->audit[m].pair_delta_se[0][0],
                ROLLOUT_MAX_CANDIDATES,z,
                &out->decision[m]);
            if(!out->have_decision[m])return 0;
        }
    }
    return 1;
}

static void print_search_panels(FILE *f,const char *label,const RuntimeMask mask[PC_MASKS],
                                const SearchPanels *p,int enabled_bits,uint64_t seed,double z)
{
    fprintf(f,"\"%s\":{\"seed\":\"%llu\",\"requested_worlds\":800,"
              "\"common_worlds_across_actions\":true,"
              "\"same_seed_across_exact_masks\":%s,\"z\":%.17g,"
              "\"overlap_bit_exact\":true,\"masks\":[",label,
            (unsigned long long)seed,enabled_bits==3?"true":"false",z);
    int printed=0;for(int m=0;m<PC_MASKS;m++){
        if(!(enabled_bits&(1<<m)))continue;
        const RolloutAuditPanel *a=&p->audit[m];
        fprintf(f,"%s{\"mask_index\":%d,\"worlds\":%d,"
                  "\"panel_role\":%d,\"hidden_world_fingerprint\":\"%016llx\","
                  "\"exact_hidden_support\":%s,\"hidden_support\":%d,"
                  "\"exact_terminal_leaves\":%llu,"
                  "\"unfinished_cap_leaves\":%llu,\"cycle_breaks\":%llu,"
                  "\"cap_reserve_forces\":%llu,\"actions\":[",
                printed?",":"",m,a->worlds,a->panel_role,
                (unsigned long long)a->hidden_world_fingerprint,
                a->exact_hidden_support?"true":"false",
                a->hidden_support,(unsigned long long)a->exact_terminal_leaves,
                (unsigned long long)a->unfinished_cap_leaves,
                (unsigned long long)a->cycle_breaks,
                (unsigned long long)a->cap_reserve_forces);
        for(int c=0;c<a->n;c++){
            double sum=(double)a->worlds*a->q[c];
            double sumsq=a->se[c]*a->se[c]*a->worlds*(a->worlds-1)+
                         (double)a->worlds*a->q[c]*a->q[c];
            fprintf(f,"%s{\"position\":%d,\"legal_index\":%d,"
                      "\"mean\":%.17g,\"se\":%.17g,\"sum\":%.17g,"
                      "\"sum_squares\":%.17g}",c?",":"",c,
                    mask[m].index[c],a->q[c],a->se[c],sum,sumsq);
        }
        fputs("],\"pairs\":[",f);int first=1;
        for(int x=0;x<a->n;x++)for(int y=x+1;y<a->n;y++){
            double sx=(double)a->worlds*a->q[x],sy=(double)a->worlds*a->q[y];
            double sx2=a->se[x]*a->se[x]*a->worlds*(a->worlds-1)+
                       (double)a->worlds*a->q[x]*a->q[x];
            double sy2=a->se[y]*a->se[y]*a->worlds*(a->worlds-1)+
                       (double)a->worlds*a->q[y]*a->q[y];
            double dm=a->pair_delta[x][y],ds=a->pair_delta_se[x][y];
            double diff2=ds*ds*a->worlds*(a->worlds-1)+(double)a->worlds*dm*dm;
            double cross=0.5*(sx2+sy2-diff2);
            (void)sx;(void)sy;
            fprintf(f,"%s{\"a\":%d,\"b\":%d,"
                      "\"delta_a_minus_b\":%.17g,\"paired_se\":%.17g,"
                      "\"sum_products\":%.17g}",first?"":",",x,y,dm,ds,cross);first=0;
        }
        fputs("],\"policy_cost_decision\":",f);
        if(!p->have_decision[m])fputs("null",f);
        else{
            const PolicyCostDecision *d=&p->decision[m];
            fprintf(f,"{\"leader_position\":%d,\"selected_position\":%d,"
                "\"anchor_interval\":%d,\"beta_search\":%.17g,"
                "\"alpha_action\":%.17g,\"alpha_draw\":%.17g,"
                "\"lambda_action\":%.17g,\"lambda_draw\":%.17g,"
                "\"all_pair_passed\":%s,"
                "\"prior_protected_rivals\":%d,\"actions\":[",
                d->leader,d->selected,d->anchor_interval,d->beta,
                d->alpha_action,d->alpha_draw,d->lambda_action,d->lambda_draw,
                d->all_pair_passed?"true":"false",
                d->prior_protected_rivals);
            for(int c=0;c<a->n;c++)fprintf(f,"%s{\"position\":%d,"
                "\"semantic_prior\":%.17g,\"conditional_draw_prior\":%.17g,"
                "\"cost\":%.17g,\"adjusted_q\":%.17g}",c?",":"",c,
                d->semantic_prior[c],d->conditional_draw_prior[c],d->cost[c],d->adjusted_q[c]);
            fputs("]}",f);
        }
        fputc('}',f);
        printed++;
    }
    fputs("]}",f);
}

typedef struct {
    int exact_valid,selected_position,selected_legal_index;
    const char *gate_reason;
} ComposedDecision;

static ComposedDecision compose_policy_cost(const State *view,const Move *legal,
                                            const RuntimeMask *mask,
                                            const PolicyCostDecision *primary,
                                            const PolicyCostDecision *fresh,
                                            int active)
{
    ComposedDecision out={1,0,mask->index[0],"before_onset_policy_baseline"};
    if(!active)return out;
    if(!primary||!fresh){out.exact_valid=0;out.gate_reason="invalid_panel";return out;}
    int proposed=primary->leader;
    if(proposed==0){out.gate_reason="adjusted_baseline";return out;}
    if(!primary->all_pair_passed||primary->selected!=proposed){
        out.gate_reason="primary_evidence";return out;
    }
    if(fresh->leader!=proposed){out.gate_reason="fresh_leader_mismatch";return out;}
    if(!fresh->all_pair_passed||fresh->selected!=proposed){
        out.gate_reason="fresh_evidence";return out;
    }
    if(proposed<0||proposed>=mask->n){out.exact_valid=0;out.gate_reason="invalid_position";return out;}
    uint64_t dead=lc_dead_cards(view);
    if(!lc_discard_dominated(view,legal[mask->index[0]],dead)&&
       lc_discard_dominated(view,legal[mask->index[proposed]],dead)){
        out.gate_reason="discard_guard";return out;
    }
    out.selected_position=proposed;out.selected_legal_index=mask->index[proposed];
    out.gate_reason="selected";return out;
}

static void print_one_composed(FILE *f,const ComposedDecision *d)
{
    fprintf(f,"{\"selected_legal_index\":%d,\"selected_position\":%d,"
        "\"gate_reason\":\"%s\",\"exact_valid\":%s,\"capped\":0}",
        d->selected_legal_index,d->selected_position,d->gate_reason,
        d->exact_valid?"true":"false");
}

static void print_production_decisions(FILE *f,const State *view,const Move *mv,
                                       const RuntimeMask mask[PC_MASKS],
                                       const SearchPanels *primary,
                                       const SearchPanels *fresh,int enabled_bits,
                                       int split,const PolicyCostTable *table)
{
    fputs("\"production_decisions\":{",f);int first=1;
    if(split==SPLIT_TEST&&table){
        int m=table->controller.cand_floor==0.01f?0:1;
        int active=view->nply>=table->controller.ply_lo;
        ComposedDecision d=compose_policy_cost(
            view,mv,&mask[m],&primary->decision[m],&fresh->decision[m],active);
        fprintf(f,"\"floor-%.2f\":",mask[m].floor);print_one_composed(f,&d);
    }else for(int m=0;m<PC_MASKS;m++)if(enabled_bits&(1<<m)){
        ComposedDecision d=compose_policy_cost(view,mv,&mask[m],&primary->decision[m],
                                                &fresh->decision[m],1);
        fprintf(f,"%s\"floor-%.2f\":",first?"":",",mask[m].floor);
        print_one_composed(f,&d);first=0;
    }
    fputs("},\"config_decisions\":",f);
    if(split!=SPLIT_SELECT){fputs("null",f);return;}
    static const int onset[1]={0};fputc('{',f);first=1;
    for(int o=0;o<1;o++)for(int m=1;m>=0;m--){
        int active=view->nply>=onset[o];
        ComposedDecision d=compose_policy_cost(view,mv,&mask[m],&primary->decision[m],
                                                &fresh->decision[m],active);
        fprintf(f,"%s\"floor-%.2f_ply-%02d\":",first?"":",",mask[m].floor,onset[o]);
        print_one_composed(f,&d);first=0;
    }
    fputc('}',f);
}

static void print_truth(FILE *f,const TruthPanel *t,uint64_t seed)
{
    char hh[65],fh[65];digest_hex(t->hidden_hash,hh);digest_hex(t->future_hash,fh);
    fprintf(f,"\"truth\":{\"controller\":\"exact_policy20_full_remaining_match\","
        "\"information_view_each_node\":true,\"temperature\":0,\"epsilon\":0,"
        "\"seed\":\"%llu\",\"requested_worlds\":%d,\"worlds\":%d,"
        "\"union_untruncated\":true,\"union_count\":%d,"
        "\"hidden_worlds_sha256\":\"%s\",\"future_deals_sha256\":\"%s\","
        "\"cap_hits\":%llu,\"metrics\":{",
        (unsigned long long)seed,t->worlds,t->worlds,t->n,hh,fh,
        (unsigned long long)t->cap_hits);
    print_sample_metric(f,"current_round_margin",t->round_margin,t->n,t->worlds);fputc(',',f);
    print_sample_metric(f,"full_match_margin",t->final_margin,t->n,t->worlds);fputc(',',f);
    print_sample_metric(f,"full_match_score",t->match_score,t->n,t->worlds);fputc(',',f);
    print_sample_metric(f,"full_match_hybrid",t->hybrid,t->n,t->worlds);
    fputs("}}",f);
}

static int evaluate_train_pair(const Agent *actor,const State *view,
                               const Move *legal,const int pair_index[2],
                               int panel_role,uint64_t seed,
                               RolloutAuditPanel *out)
{
    Move pair[2]={legal[pair_index[0]],legal[pair_index[1]]};
    int rc=rollout_audit_panel_role(actor,view,pair,2,0,panel_role,seed,
                                    PC_PRIMARY_WORLDS,out);
    return rc==0&&out->n==2&&out->requested_worlds==PC_PRIMARY_WORLDS&&
           out->worlds>=2&&out->unfinished_cap_leaves==0&&
           out->panel_role==panel_role;
}

static void print_train_pair_panel(FILE *f,const char *label,
                                   const RolloutAuditPanel *a,
                                   const int pair_index[2],uint64_t seed)
{
    fprintf(f,"\"%s\":{\"seed\":\"%llu\",\"requested_worlds\":800,"
        "\"panel_role\":%d,\"hidden_world_fingerprint\":\"%016llx\","
        "\"worlds\":%d,\"common_worlds_across_pair\":true,"
        "\"exact_hidden_support\":%s,\"hidden_support\":%d,"
        "\"exact_terminal_leaves\":%llu,\"unfinished_cap_leaves\":%llu,"
        "\"cycle_breaks\":%llu,\"cap_reserve_forces\":%llu,\"actions\":[",
        label,(unsigned long long)seed,a->panel_role,
        (unsigned long long)a->hidden_world_fingerprint,a->worlds,
        a->exact_hidden_support?"true":"false",a->hidden_support,
        (unsigned long long)a->exact_terminal_leaves,
        (unsigned long long)a->unfinished_cap_leaves,
        (unsigned long long)a->cycle_breaks,
        (unsigned long long)a->cap_reserve_forces);
    double sumsq[2];
    for(int c=0;c<2;c++){
        double sum=(double)a->worlds*a->q[c];
        sumsq[c]=a->se[c]*a->se[c]*a->worlds*(a->worlds-1)+
                  (double)a->worlds*a->q[c]*a->q[c];
        fprintf(f,"%s{\"position\":%d,\"legal_index\":%d,"
            "\"mean\":%.17g,\"se\":%.17g,\"sum\":%.17g,"
            "\"sum_squares\":%.17g}",c?",":"",c,pair_index[c],
            a->q[c],a->se[c],sum,sumsq[c]);
    }
    double dm=a->pair_delta[0][1],ds=a->pair_delta_se[0][1];
    double diff2=ds*ds*a->worlds*(a->worlds-1)+(double)a->worlds*dm*dm;
    double cross=0.5*(sumsq[0]+sumsq[1]-diff2);
    fprintf(f,"],\"pair\":{\"delta_a_minus_b\":%.17g,"
        "\"paired_se\":%.17g,\"sum_products\":%.17g}}",dm,ds,cross);
}

typedef struct {
    const char *out_path,*manifest_path,*manifest_sha,*actor_spec;
    const char *maintained_actor_spec,*reservoir_path,*reservoir_sha;
    const char *truth_net_path,*policy_cost_path;
    int allocation_start,allocation_count,have_allocation_start,have_allocation_count;
} EvaluateConfig;

typedef struct {
    const char *out_path,*manifest_path,*manifest_sha,*reservoir_path;
    const char *reservoir_sha,*net_path,*exclusion_path,*exclusion_sha;
} VerifyReservoirConfig;

static int verify_reservoir_args(int argc,char **argv,VerifyReservoirConfig *c)
{
    memset(c,0,sizeof *c);
    for(int i=2;i<argc;i++){
        if(!strcmp(argv[i],"--out")&&++i<argc)c->out_path=argv[i];
        else if(!strcmp(argv[i],"--manifest")&&++i<argc)c->manifest_path=argv[i];
        else if(!strcmp(argv[i],"--manifest-sha256")&&++i<argc)c->manifest_sha=argv[i];
        else if(!strcmp(argv[i],"--reservoir")&&++i<argc)c->reservoir_path=argv[i];
        else if(!strcmp(argv[i],"--reservoir-sha256")&&++i<argc)c->reservoir_sha=argv[i];
        else if(!strcmp(argv[i],"--net")&&++i<argc)c->net_path=argv[i];
        else if(!strcmp(argv[i],"--exclusions")&&++i<argc)c->exclusion_path=argv[i];
        else if(!strcmp(argv[i],"--exclusions-sha256")&&++i<argc)c->exclusion_sha=argv[i];
        else return 0;
    }
    return c->out_path&&c->manifest_path&&c->manifest_sha&&c->reservoir_path&&
           c->reservoir_sha&&c->net_path&&c->exclusion_path&&c->exclusion_sha;
}

static int evaluate_args(int argc,char **argv,EvaluateConfig *c)
{
    memset(c,0,sizeof *c);
    for(int i=2;i<argc;i++){
        if(!strcmp(argv[i],"--out")&&++i<argc)c->out_path=argv[i];
        else if(!strcmp(argv[i],"--manifest")&&++i<argc)c->manifest_path=argv[i];
        else if(!strcmp(argv[i],"--manifest-sha256")&&++i<argc)c->manifest_sha=argv[i];
        else if(!strcmp(argv[i],"--reservoir")&&++i<argc)c->reservoir_path=argv[i];
        else if(!strcmp(argv[i],"--reservoir-sha256")&&++i<argc)c->reservoir_sha=argv[i];
        else if(!strcmp(argv[i],"--actor")&&++i<argc)c->actor_spec=argv[i];
        else if(!strcmp(argv[i],"--maintained-actor")&&++i<argc)c->maintained_actor_spec=argv[i];
        else if(!strcmp(argv[i],"--truth-net")&&++i<argc)c->truth_net_path=argv[i];
        else if(!strcmp(argv[i],"--policy-cost")&&++i<argc)c->policy_cost_path=argv[i];
        else if(!strcmp(argv[i],"--allocation-start")&&++i<argc&&
                parse_int(argv[i],0,1000000,&c->allocation_start))c->have_allocation_start=1;
        else if(!strcmp(argv[i],"--allocation-count")&&++i<argc&&
                parse_int(argv[i],1,1000000,&c->allocation_count))c->have_allocation_count=1;
        else return 0;
    }
    return c->out_path&&c->manifest_path&&c->manifest_sha&&c->reservoir_path&&
           c->reservoir_sha&&c->actor_spec&&c->truth_net_path&&
           c->have_allocation_start==c->have_allocation_count;
}

static int allowed_onset(int n)
{
    static const int value[]={0,4,8,10,12,14};
    for(size_t i=0;i<sizeof value/sizeof value[0];i++)if(n==value[i])return 1;
    return 0;
}

static int frozen_base_actor(const Agent *a,const Net *truth)
{
    if(!a||a->kind!=AG_ROLLOUT||!a->net||!a->continuation_net||
       match_value_net_fingerprint(a->net)!=match_value_net_fingerprint(truth)||
       match_value_net_fingerprint(a->continuation_net)!=match_value_net_fingerprint(truth)||
       match_value_net_fingerprint(a->net)!=match_value_net_fingerprint(a->continuation_net)||
       a->veto_continuation_net||a->action_ranker_net||a->dets!=800||
       a->confirm_dets!=800||a->root_width!=5||a->action_core_count!=3||
       a->min_cand!=1||!allowed_onset(a->ply_lo)||a->ply_hi!=0||
       (a->cand_floor!=0.01f&&a->cand_floor!=0.02f)||a->cand_mass!=0.0f||
       a->gate!=0.0f||a->eval_cand!=0||a->batch_dets!=0||
       (a->win_q!=0&&a->win_q!=3)||a->prune_dom!=0||
       a->override_k!=(float)POLICY_COST_PRIMARY_Z||a->override_min!=0.0f||
       a->playout_sample!=4||a->symmetries!=20||a->playout_symmetries!=20||
       a->discard_guard!=1||a->deck_max!=0||a->playout_prune!=1||
       a->plan_deck_max!=0||a->plan_block_gap!=0||a->semantic_cand!=0||
       a->confirm_exact5!=0||a->draw_variant_cores!=0||
       a->draw_variant_deck_max!=0||a->policy_prefix_mode!=0||
       a->no_belief!=1||a->belief_alpha!=1.0f||a->draw_root_deck_max!=0||
       a->draw_playout_deck_max!=0||a->prefix_confirm_k!=0.0f||
       a->prefix_confirm_min!=0.0f||a->confirm_temp!=0.0f||
       a->exact_terminal!=1||a->deck2_replan_worlds!=0||
       a->deck2_replan_cores!=0||a->bounded_late_root!=0||
       a->bounded_late_min!=1.0f||a->action_ranker_min!=0.0f)
        return 0;
    if(a->win_q==3)
        return a->match_value&&match_value_validate(a->match_value)&&
               match_value_balanced_roles(a->match_value);
    return !a->match_value&&!a->owns_match_value;
}

static int frozen_maintained_actor(const Agent *a,const Net *truth)
{
    if(!a||a->kind!=AG_ROLLOUT||!a->net||!a->continuation_net||
       match_value_net_fingerprint(a->net)!=match_value_net_fingerprint(truth)||
       match_value_net_fingerprint(a->continuation_net)!=match_value_net_fingerprint(truth)||
       match_value_net_fingerprint(a->net)!=match_value_net_fingerprint(a->continuation_net)||
       a->veto_continuation_net||a->action_ranker_net||a->policy_cost||
       a->owns_policy_cost||a->dets!=800||a->confirm_dets!=800||
       a->root_width!=5||a->action_core_count!=0||a->min_cand!=1||
       a->ply_hi!=0||a->cand_floor!=0.02f||
       a->cand_mass!=0.0f||a->gate!=0.0f||a->eval_cand!=0||
       a->batch_dets!=0||a->prune_dom!=0||
       a->override_k!=(float)POLICY_COST_PRIMARY_Z||a->override_min!=2.0f||
       a->playout_sample!=4||a->symmetries!=20||a->playout_symmetries!=20||
       a->discard_guard!=1||a->deck_max!=0||a->playout_prune!=1||
       a->plan_deck_max!=0||a->plan_block_gap!=0||a->semantic_cand!=0||
       a->confirm_exact5!=0||a->draw_variant_cores!=0||
       a->draw_variant_deck_max!=0||a->policy_prefix_mode!=3||
       a->no_belief!=1||a->belief_alpha!=1.0f||a->draw_root_deck_max!=0||
       a->draw_playout_deck_max!=0||a->prefix_confirm_k!=0.0f||
       a->prefix_confirm_min!=0.0f||a->confirm_temp!=0.0f||
       a->exact_terminal!=1||a->deck2_replan_worlds!=0||
       a->deck2_replan_cores!=0||a->bounded_late_root!=0||
       a->bounded_late_min!=1.0f||a->action_ranker_min!=0.0f)
        return 0;
    /* The prerequisite disposition has exactly two possible maintained
     * profiles.  A failed Objective-3 campaign retains the verified legacy
     * round-margin actor from ply 14; a promoted Objective-3 actor uses its
     * bound Bellman table at every ply.  Do not admit cross-products such as
     * objective 3 at ply 14 or objective 0 at all plies.  The Python evidence
     * converter additionally binds the literal actor/table/net identities to
     * the authoritative prerequisite result and execution transport. */
    if(a->win_q==3)
        return a->ply_lo==0&&a->match_value&&
               match_value_validate(a->match_value)&&
               match_value_balanced_roles(a->match_value);
    if(a->win_q==0)
        return a->ply_lo==14&&!a->match_value&&!a->owns_match_value;
    return 0;
}

static int table_matches_counterfactual_actor(const PolicyCostTable *table,
                                              const Agent *actor)
{
    if(!table)return 1;
    if(!policy_cost_validate(table)||
       (table->controller.cand_floor!=0.01f&&
        table->controller.cand_floor!=0.02f)||
       !allowed_onset((int)table->controller.ply_lo))return 0;
    Agent bound=*actor;
    bound.policy_cost=table;bound.owns_policy_cost=0;
    /* SELECT/TEST may deliberately replay another preregistered floor/onset;
     * every other controller bit must match the authoritative artifact. */
    bound.cand_floor=table->controller.cand_floor;
    bound.ply_lo=(int)table->controller.ply_lo;
    return policy_cost_matches_agent(&bound);
}

static int legal_by_semantic_pack(const Move *mv,const int *uindex,int n,
                                  uint16_t pack)
{
    int found=-1;
    for(int i=0;i<n;i++)if(semantic_move_pack(mv[uindex[i]])==pack){
        if(found>=0)return -2;
        found=i;
    }
    return found;
}

static void policy_terms(const Move *mv,const float *prob,int n,int index,
                         double *action,double *draw)
{
    double a=0.0,j=0.0;
    for(int i=0;i<n;i++)if(same_action(mv[i],mv[index])){
        a+=prob[i];if(mv[i].draw==mv[index].draw)j+=prob[i];
    }
    *action=a;*draw=j/a;
}

static int verify_allocation_state(const Allocation *a,const State *view,
                                   const Move *mv,const float *prob,int n,
                                   const RuntimeMask mask[PC_MASKS],
                                   const int *uindex,int nunion)
{
    unsigned char bytes[PC_STATE_MAX],hash[32],orbit[32],uh[32];size_t nb=0;
    if(!view_valid(view)||!encode_view(view,bytes,&nb)||nb!=a->state_n||
       memcmp(bytes,a->state_bytes,nb))return 0;
    sha_bytes(bytes,nb,hash);if(memcmp(hash,a->state_hash,32))return 0;
    if(!orbit_digest(view,orbit)||memcmp(orbit,a->orbit,32))return 0;
    if(memcmp(mask[0].hash,a->mask_hash[0],32)||memcmp(mask[1].hash,a->mask_hash[1],32))return 0;
    if(view->round!=a->round||ply_bin(view->nply)!=a->ply_bin)return 0;
    if(a->master_width>0){
        if(a->master_width!=mask[0].n||memcmp(mask[0].hash,a->union_hash,32)||
           a->frontier!=frontier_present(mv,prob,n,mask))return 0;
        return 1;
    }
    union_digest(mv,uindex,nunion,uh);if(memcmp(uh,a->union_hash,32))return 0;
    int pa=legal_by_semantic_pack(mv,uindex,nunion,a->pair_move[0]);
    int pb=legal_by_semantic_pack(mv,uindex,nunion,a->pair_move[1]);
    if(pa<0||pb<0||pa==pb)return 0;
    int ia=uindex[pa],ib=uindex[pb];
    int type=same_action(mv[ia],mv[ib])?1:0;
    if(type==1&&mv[ia].draw==mv[ib].draw)return 0;
    if(type!=a->pair_type)return 0;
    double aa,ad,ba,bd;policy_terms(mv,prob,n,ia,&aa,&ad);policy_terms(mv,prob,n,ib,&ba,&bd);
    double high=type?ad:aa,low=type?bd:ba;
    if(high<low){double t=high;high=low;low=t;}
    if(!(low>0.0)||ratio_bin(high/low)!=a->ratio_bin)return 0;
    return 1;
}

static int verify_retained_origin(const Allocation *a,const Net *net,
                                  const Exclusions *exclusions)
{
    State view;unsigned char bytes[PC_STATE_MAX],state_hash[32],orbit[32];
    size_t nbytes=0;Move mv[MAX_MOVES];float prob[MAX_MOVES];
    RuntimeMask mask[PC_MASKS];int n=0,baseline=0,uindex[PC_UNION_MAX],nunion=0;
    if(!a||!net||!exclusions||a->state_n!=174||
       !source_match_in_range(a->master_width>0?SPLIT_SELECT:SPLIT_TRAIN,
                              a->source_match_index)||
       a->source_state_index<0||a->source_state_index>=3*LC_MAX_PLIES||
       !decode_view(a->state_bytes,a->state_n,&view)||!view_valid(&view)||
       !encode_view(&view,bytes,&nbytes)||nbytes!=a->state_n||
       memcmp(bytes,a->state_bytes,nbytes))return 0;
    sha_bytes(bytes,nbytes,state_hash);
    if(memcmp(state_hash,a->state_hash,32)||!orbit_digest(&view,orbit)||
       memcmp(orbit,a->orbit,32)||excluded(exclusions,orbit))return 0;
    return policy_snapshot(net,&view,mv,prob,&n,&baseline,mask,uindex,&nunion)&&
           verify_allocation_state(a,&view,mv,prob,n,mask,uindex,nunion);
}

static int run_verify_reservoir(const VerifyReservoirConfig *c)
{
    AllocationManifest manifest;BoundReservoir reservoir;
    unsigned char manifest_hash[32],reservoir_hash[32],net_hash[32],exclusion_hash[32];
    Exclusions exclusions;
    if(!load_allocation(c->manifest_path,c->manifest_sha,&manifest,manifest_hash)){
        fprintf(stderr,"invalid allocation manifest for reservoir verification\n");return 1;
    }
    if(!load_bound_reservoir(c->reservoir_path,c->reservoir_sha,&manifest,
                             &reservoir,reservoir_hash)){
        fprintf(stderr,"invalid discovery reservoir for verification\n");
        allocation_free(&manifest);return 1;
    }
    for(int i=0;i<manifest.records;i++)if(!allocation_in_reservoir(&manifest.row[i],&reservoir)){
        fprintf(stderr,"allocation is absent from verified reservoir\n");
        bound_reservoir_free(&reservoir);allocation_free(&manifest);return 1;
    }
    if(!file_sha256(c->net_path,net_hash)||memcmp(net_hash,manifest.source_net_hash,32)||
       !load_exclusions(c->exclusion_path,c->exclusion_sha,&exclusions,exclusion_hash)||
       memcmp(exclusion_hash,manifest.source_exclusion_hash,32)){
        fprintf(stderr,"reservoir verifier model/exclusion binding drift\n");
        bound_reservoir_free(&reservoir);allocation_free(&manifest);return 1;
    }
    Net *net=(Net *)malloc(sizeof *net);
    if(!net||net_load(net,c->net_path)){
        fprintf(stderr,"reservoir verifier cannot load policy net\n");free(net);
        bound_reservoir_free(&reservoir);allocation_free(&manifest);return 1;
    }
    Sha256 chain;sha256_init(&chain);
    static const char tag[]="lc-policy-cost-verified-reservoir-v1";
    sha256_update(&chain,tag,sizeof tag-1);sha256_update(&chain,manifest_hash,32);
    sha256_update(&chain,reservoir_hash,32);sha256_update(&chain,net_hash,32);
    sha256_update(&chain,exclusion_hash,32);
    for(int i=0;i<reservoir.n;i++){
        Allocation *a=&reservoir.row[i].value;
        if(!verify_retained_origin(a,net,&exclusions)){
            fprintf(stderr,"reservoir row %d has state/orbit/policy/firewall drift\n",i);goto bad;
        }
        unsigned char id[12];
        for(int j=0;j<8;j++)id[j]=(unsigned char)(a->source_match_index>>(8*j));
        for(int j=0;j<4;j++)id[8+j]=(unsigned char)((unsigned)a->source_state_index>>(8*j));
        sha256_update(&chain,id,sizeof id);sha256_update(&chain,a->state_hash,32);
        sha256_update(&chain,a->orbit,32);sha256_update(&chain,a->mask_hash,
                                                       sizeof a->mask_hash);
        sha256_update(&chain,a->union_hash,32);
    }
    {unsigned char chain_hash[32];char mh[65],rh[65],nh[65],eh[65],ch[65];
     sha256_final(&chain,chain_hash);digest_hex(manifest_hash,mh);
     digest_hex(reservoir_hash,rh);digest_hex(net_hash,nh);
     digest_hex(exclusion_hash,eh);digest_hex(chain_hash,ch);
     AtomicFile output={0};if(!atomic_open(&output,c->out_path)){
        fprintf(stderr,"cannot create reservoir verification proof\n");goto bad;
     }
     fprintf(output.file,"{\"all_orbits_recomputed\":true,"
        "\"all_policy_masks_exact\":true,\"all_state_hashes_exact\":true,"
        "\"all_views_native_valid\":true,\"allocation_sha256\":\"%s\","
        "\"eligible_units\":%llu,\"excluded_orbits_found\":0,"
        "\"exclusion_sha256\":\"%s\",\"rejected_by_bound\":%llu,"
        "\"reservoir_sha256\":\"%s\",\"retained_rows\":%d,"
        "\"schema\":\"lc-policy-cost-verified-reservoir-v1\","
        "\"source_net_sha256\":\"%s\",\"split\":\"%s\","
        "\"state_bytes\":174,\"verified_chain_sha256\":\"%s\"}\n",
        mh,(unsigned long long)reservoir.eligible,eh,
        (unsigned long long)reservoir.rejected,rh,reservoir.n,nh,
        SPLIT_DOMAIN[manifest.split].name,ch);
     if(!atomic_finish(&output)){fprintf(stderr,"cannot install reservoir proof\n");goto bad;}
     unsigned char proof_hash[32];char ph[65];if(file_sha256(c->out_path,proof_hash)){
        digest_hex(proof_hash,ph);printf("proof_sha256=%s\n",ph);
     }}
    free(net);bound_reservoir_free(&reservoir);allocation_free(&manifest);return 0;
bad:
    free(net);bound_reservoir_free(&reservoir);allocation_free(&manifest);return 1;
}

static void print_eval_policy(FILE *f,const Move *mv,const float *prob,int n,
                              int baseline,const RuntimeMask mask[PC_MASKS],
                              const int *uindex,int nunion)
{
    fprintf(f,"\"policy\":{\"symmetries\":20,\"exact_group_average\":true,"
              "\"legal_count\":%d,\"literal_argmax_index\":%d,\"legal\":[",n,baseline);
    for(int i=0;i<n;i++){
        double a,d;policy_terms(mv,prob,n,i,&a,&d);
        fprintf(f,"%s{\"index\":%d,\"move_pack\":%u,\"semantic_move_pack\":%u,"
                  "\"card\":%u,\"discard\":%u,\"draw\":%u,"
                  "\"probability\":%.9g,\"probability_bits\":\"%08x\","
                  "\"semantic_action_probability\":%.17g,"
                  "\"conditional_draw_probability\":%.17g}",i?",":"",i,
                MOVE_PACK(mv[i]),semantic_move_pack(mv[i]),mv[i].card,mv[i].discard,
                mv[i].draw,prob[i],float_bits(prob[i]),a,d);
    }
    fputs("],\"runtime_masks\":[",f);print_mask(f,&mask[0]);fputc(',',f);print_mask(f,&mask[1]);
    fputs("],\"union\":{\"untruncated\":true,\"legal_indices\":[",f);
    for(int i=0;i<nunion;i++)fprintf(f,"%s%d",i?",":"",uindex[i]);
    fputs("]}}",f);
}

static int run_evaluate(const EvaluateConfig *c)
{
    AllocationManifest manifest;unsigned char manifest_hash[32],truth_hash[32];
    if(!load_allocation(c->manifest_path,c->manifest_sha,&manifest,manifest_hash)){
        fprintf(stderr,"invalid, incomplete, sparse, or unbound allocation manifest\n");return 1;
    }
    int allocation_start=c->have_allocation_start?c->allocation_start:0;
    int allocation_count=c->have_allocation_count?c->allocation_count:manifest.records;
    if(allocation_start<0||allocation_start>=manifest.records||allocation_count<1||
       allocation_count>manifest.records-allocation_start){
        fprintf(stderr,"allocation slice is outside full immutable manifest\n");
        allocation_free(&manifest);return 1;
    }
    BoundReservoir reservoir;unsigned char reservoir_hash[32];
    if(!load_bound_reservoir(c->reservoir_path,c->reservoir_sha,&manifest,
                             &reservoir,reservoir_hash)){
        fprintf(stderr,"invalid or unbound discovery reservoir\n");allocation_free(&manifest);return 1;
    }
    for(int i=0;i<manifest.records;i++)if(!allocation_in_reservoir(&manifest.row[i],&reservoir)){
        fprintf(stderr,"allocation row %d is absent from bound reservoir\n",i);
        bound_reservoir_free(&reservoir);allocation_free(&manifest);return 1;
    }
    if(!file_sha256(c->truth_net_path,truth_hash)||
       memcmp(truth_hash,manifest.source_net_hash,32)){
        fprintf(stderr,"truth net does not match the discovery policy checkpoint\n");
        bound_reservoir_free(&reservoir);allocation_free(&manifest);return 1;
    }
    Net *truth=(Net *)malloc(sizeof *truth);
    if(!truth||net_load(truth,c->truth_net_path)){fprintf(stderr,"cannot load truth net\n");free(truth);bound_reservoir_free(&reservoir);allocation_free(&manifest);return 1;}
    Agent actor;spec_parse(c->actor_spec,&actor);
    if(!frozen_base_actor(&actor,truth)||
       (actor.policy_cost&&!policy_cost_matches_agent(&actor))){
        fprintf(stderr,"actor is not the frozen P800/F800 policy-cost controller\n");
        spec_release(&actor);free(truth);bound_reservoir_free(&reservoir);allocation_free(&manifest);return 1;
    }
    Agent maintained;memset(&maintained,0,sizeof maintained);int have_maintained=0;
    if(manifest.split!=SPLIT_TRAIN){
        if(!c->maintained_actor_spec){
            fprintf(stderr,"SELECT/TEST require --maintained-actor\n");
            spec_release(&actor);free(truth);bound_reservoir_free(&reservoir);
            allocation_free(&manifest);return 1;
        }
        spec_parse(c->maintained_actor_spec,&maintained);have_maintained=1;
        if(!frozen_maintained_actor(&maintained,truth)||
           match_value_net_fingerprint(maintained.net)!=
               match_value_net_fingerprint(actor.net)||
           match_value_net_fingerprint(maintained.continuation_net)!=
               match_value_net_fingerprint(actor.continuation_net)||
           maintained.win_q!=actor.win_q||
           (actor.win_q==3&&(!actor.match_value||!maintained.match_value||
             actor.match_value->payload_fingerprint!=
                 maintained.match_value->payload_fingerprint))){
            fprintf(stderr,"maintained actor does not match frozen legacy baseline\n");
            spec_release(&maintained);spec_release(&actor);free(truth);
            bound_reservoir_free(&reservoir);allocation_free(&manifest);return 1;
        }
    }
    PolicyCostTable *owned_table=NULL;const PolicyCostTable *table=actor.policy_cost;
    unsigned char table_hash[32]={0};
    if(c->policy_cost_path){
        if(table){fprintf(stderr,"policy-cost table supplied twice\n");goto fail;}
        int error=0;owned_table=policy_cost_load(c->policy_cost_path,&error);
        if(!owned_table||!file_sha256(c->policy_cost_path,table_hash)){
            fprintf(stderr,"invalid policy-cost table\n");goto fail;
        }
        table=owned_table;
    } else if(table) {
        /* The actor parser already content-validates it.  Its file path lives
         * in the immutable actor spec; payload_fingerprint is emitted below. */
    }
    if(!table_matches_counterfactual_actor(table,&actor)){
        fprintf(stderr,"policy-cost artifact/base-controller binding mismatch\n");goto fail;
    }
    if(manifest.is_vector&&!table){
        fprintf(stderr,"SELECT/TEST require a bound policy-cost table\n");goto fail;
    }
    AtomicFile output={0};if(!atomic_open(&output,c->out_path)){
        fprintf(stderr,"output exists or cannot be created\n");goto fail;
    }
    char mh[65],rh[65],th[65],dh[65],pch[65];digest_hex(manifest_hash,mh);
    digest_hex(reservoir_hash,rh);
    digest_hex(truth_hash,th);digest_hex(manifest.discovery_hash,dh);
    if(c->policy_cost_path)digest_hex(table_hash,pch);else strcpy(pch,table?"actor-bound":"none");
    const SplitDomain *domain=&SPLIT_DOMAIN[manifest.split];
    fprintf(output.file,"{\"schema\":\"lc-policy-cost-evaluation-v1\","
        "\"record_type\":\"header\",\"split\":\"%s\","
        "\"manifest_sha256\":\"%s\",\"reservoir_sha256\":\"%s\","
        "\"discovery_sha256\":\"%s\","
        "\"allocation_kind\":\"%s\",\"quota\":%d,"
        "\"pooled_ge64_observed\":%llu,"
        "\"full_manifest_records\":%d,\"allocation_start\":%d,"
        "\"allocation_count\":%d,\"eligible_units\":%llu,"
        "\"retained_reservoir_units\":%llu,"
        "\"probe_orbit_rejections\":%llu,\"actor_spec\":",
        domain->name,mh,rh,dh,manifest.is_vector?"complete_master_vector":"pair_state",
        manifest.quota_per_cell,(unsigned long long)manifest.pooled_ge64_observed,
        manifest.records,allocation_start,allocation_count,
        (unsigned long long)(manifest.is_vector?manifest.total_census:manifest.eligible_units),
        (unsigned long long)manifest.retained_units,
        (unsigned long long)manifest.probe_orbit_rejections);
    json_string(output.file,c->actor_spec);
    fprintf(output.file,",\"maintained_actor_spec\":");
    if(have_maintained)json_string(output.file,c->maintained_actor_spec);else fputs("null",output.file);
    fprintf(output.file,",\"root_net_fingerprint\":\"%016llx\","
        "\"continuation_net_fingerprint\":\"%016llx\","
        "\"candidate_match_value_fingerprint\":\"%016llx\","
        "\"maintained_match_value_fingerprint\":\"%016llx\","
        "\"truth_net_sha256\":\"%s\",\"policy_cost_sha256\":\"%s\","
        "\"policy_cost_payload_fingerprint\":\"%016llx\","
        "\"evaluation_support\":\"%s\","
        "\"primary\":{\"worlds\":800,\"seed\":\"%llu\"},"
        "\"fresh\":{\"worlds\":800,\"seed\":\"%llu\"},"
        "\"maintained_root_seed\":\"%llu\","
        "\"truth\":{\"worlds\":%d,\"seed\":\"%llu\","
        "\"controller\":\"exact_policy20_full_remaining_match\"},"
        "\"seed_domains_pairwise_disjoint\":true,"
        "\"burned_source_deal_seeds\":\"1..200, maintained-800 seed 1, "
        "202611010101, all policy-cost-v1 fixed seeds in "
        "20261110/11/12/13/14/15/16/21/22, every 20261129 "
        "feasibility-smoke seed, 202612010101, and every 20261229 "
        "feasibility-smoke seed\","
        "\"burned_seed_intersection\":0,\"arbitrary_top_five_truncation\":false}\n",
        (unsigned long long)match_value_net_fingerprint(actor.net),
        (unsigned long long)match_value_net_fingerprint(actor.continuation_net),
        (unsigned long long)(actor.match_value?actor.match_value->payload_fingerprint:0),
        (unsigned long long)(have_maintained&&maintained.match_value?
                             maintained.match_value->payload_fingerprint:0),
        th,pch,(unsigned long long)(table?table->payload_fingerprint:0),
        manifest.split==SPLIT_TRAIN?"one_pair_per_state":"exact_masks_plus_untruncated_union",
        (unsigned long long)domain->primary_seed,
        (unsigned long long)domain->fresh_seed,
        (unsigned long long)domain->maintained_seed,domain->truth_worlds,
        (unsigned long long)domain->truth_seed);

    uint64_t primary_caps=0,fresh_caps=0,truth_caps=0,maintained_caps=0;
    uint64_t exact_leaves=0,maintained_exact_leaves=0;
    for(int r=allocation_start;r<allocation_start+allocation_count;r++){
        Allocation *a=&manifest.row[r];State view;
        if(!decode_view(a->state_bytes,a->state_n,&view)){fprintf(stderr,"state decode failed\n");goto output_fail;}
        Move mv[MAX_MOVES];float prob[MAX_MOVES];RuntimeMask mask[PC_MASKS];
        int n=0,baseline=0,uindex[PC_UNION_MAX],nunion=0;
        if(!policy_snapshot(actor.net,&view,mv,prob,&n,&baseline,mask,uindex,&nunion)||
           !verify_allocation_state(a,&view,mv,prob,n,mask,uindex,nunion)){
            fprintf(stderr,"allocation row %d drifted from discovery evidence\n",r);goto output_fail;
        }
        uint64_t ps=domain_seed(domain->primary_seed,a->source_match_index,
            (uint64_t)a->source_state_index,0,TAG_PRIMARY);
        uint64_t fs=domain_seed(domain->fresh_seed,a->source_match_index,
            (uint64_t)a->source_state_index,0,TAG_FRESH);
        if(ps==fs){fprintf(stderr,"seed-domain collision\n");goto output_fail;}
        int pair_index[2]={-1,-1};
        if(!manifest.is_vector){
            int pair_pos[2]={
                legal_by_semantic_pack(mv,uindex,nunion,a->pair_move[0]),
                legal_by_semantic_pack(mv,uindex,nunion,a->pair_move[1])};
            if(pair_pos[0]<0||pair_pos[1]<0){fprintf(stderr,"pair left exact master\n");goto output_fail;}
            pair_index[0]=uindex[pair_pos[0]];pair_index[1]=uindex[pair_pos[1]];
        }
        int truth_index_storage[PC_TRUTH_MAX],truth_n=0,maintained_position=-1;
        uint64_t maintained_root_seed=0;Move maintained_move={0};SearchStats maintained_stats;
        memset(&maintained_stats,0,sizeof maintained_stats);
        if(!manifest.is_vector){
            truth_index_storage[0]=pair_index[0];truth_index_storage[1]=pair_index[1];truth_n=2;
        }else{
            for(int i=0;i<mask[0].n;i++)truth_index_storage[truth_n++]=mask[0].index[i];
            maintained_root_seed=domain_seed(domain->maintained_seed,
                a->source_match_index,(uint64_t)a->source_state_index,0,TAG_MAINTAINED);
            Rng maintained_rng;rng_seed(&maintained_rng,maintained_root_seed);
            maintained_move=rollout_move(&maintained,&view,&maintained_rng,NULL,&maintained_stats);
            int maintained_legal=-1;
            for(int i=0;i<n;i++)if(semantic_move_pack(mv[i])==
                                      semantic_move_pack(maintained_move)){
                if(maintained_legal>=0){fprintf(stderr,"ambiguous maintained action\n");goto output_fail;}
                maintained_legal=i;
            }
            if(maintained_legal<0||maintained_stats.unfinished_cap_leaves!=0){
                fprintf(stderr,"invalid/capped maintained actor decision\n");goto output_fail;
            }
            maintained_caps+=maintained_stats.unfinished_cap_leaves;
            maintained_exact_leaves+=maintained_stats.exact_terminal_leaves;
            for(int i=0;i<truth_n;i++)if(truth_index_storage[i]==maintained_legal)
                maintained_position=i;
            if(maintained_position<0){
                if(truth_n>=PC_TRUTH_MAX){fprintf(stderr,"truth support exceeds six\n");goto output_fail;}
                maintained_position=truth_n;truth_index_storage[truth_n++]=maintained_legal;
            }
        }
        SearchPanels pp={0},fp={0};RolloutAuditPanel train_p={0},train_f={0};
        /* SELECT and untouched TEST both evaluate the complete 1% master
         * vector once and derive the exact 2% no-refill submatrix.  TEST's
         * selected onset affects only composition: its pre-onset actor action
         * remains literal policy, but the frozen P800/F800 vector evidence is
         * still present for every allocated record. */
        int enabled_bits=manifest.is_vector?3:0;
        if(!manifest.is_vector){
            if(!evaluate_train_pair(&actor,&view,mv,pair_index,
                                    ROLLOUT_AUDIT_PANEL_PRIMARY,ps,&train_p)||
               !evaluate_train_pair(&actor,&view,mv,pair_index,
                                    ROLLOUT_AUDIT_PANEL_FRESH,fs,&train_f)||
               !independent_hidden_panel_or_complete_census(&train_p,&train_f)){
                fprintf(stderr,"invalid/capped/non-independent TRAIN pair panel at row %d\n",r);
                goto output_fail;
            }
        }else if(enabled_bits&&(!evaluate_search_panels(&actor,table,&view,mv,prob,n,mask,
                                         enabled_bits,ROLLOUT_AUDIT_PANEL_PRIMARY,ps,
                                         POLICY_COST_PRIMARY_Z,&pp)||
                 !evaluate_search_panels(&actor,table,&view,mv,prob,n,mask,
                                         enabled_bits,ROLLOUT_AUDIT_PANEL_FRESH,fs,
                                         POLICY_COST_FRESH_Z,&fp)||
                 !independent_hidden_panel_or_complete_census(
                     &pp.audit[enabled_bits==2?1:0],
                     &fp.audit[enabled_bits==2?1:0]))){
            fprintf(stderr,"invalid/capped/unequal runtime mask panel at row %d\n",r);
            goto output_fail;
        }
        uint64_t ts=domain_seed(domain->truth_seed,a->source_match_index,
            (uint64_t)a->source_state_index,0,TAG_TRUTH_WORLD);
        TruthPanel truth_panel_value;
        if(!truth_panel(&view,mv,truth_index_storage,truth_n,truth,ts,a->source_match_index,
                        (uint64_t)a->source_state_index,domain->truth_worlds,
                        &truth_panel_value)){
            truth_caps+=truth_panel_value.cap_hits;truth_free(&truth_panel_value);
            fprintf(stderr,"truth continuation cap/invalidity at row %d\n",r);goto output_fail;
        }
        char oh[65],sh[65],ph[65];digest_hex(a->orbit,oh);digest_hex(a->state_hash,sh);
        digest_hex(a->priority,ph);
        fprintf(output.file,"{\"schema\":\"lc-policy-cost-evaluation-v1\","
            "\"record_type\":\"allocation\",\"allocation_id\":%llu,"
            "\"source_match_index\":%llu,\"source_match_id\":\"%s\","
            "\"source_state_index\":%d,\"round\":%d,\"nply\":%u,"
            "\"ply_stratum\":%d,",
            (unsigned long long)a->allocation_id,
            (unsigned long long)a->source_match_index,a->source_match_id,
            a->source_state_index,a->round,view.nply,a->ply_bin);
        if(!manifest.is_vector)fprintf(output.file,
            "\"cell\":\"%s\",\"ratio_bin\":%d,\"pair_type\":%d,"
            "\"pair_semantic_moves\":[%u,%u],",a->cell,a->ratio_bin,
            a->pair_type,a->pair_move[0],a->pair_move[1]);
        else fprintf(output.file,
            "\"frontier_present\":%s,\"allocation_slot\":%d,"
            "\"post_stratum\":\"%s\",\"unit\":\"%s:s%03d\",\"master_width\":%d,"
            "\"census_count\":%d,\"allocation_quota\":%d,"
            "\"weight_numerator\":%llu,\"weight_denominator\":%llu,"
            "\"allocation_priority_sha256\":\"%s\","
            "\"discovery_sha256\":\"%s\",",a->frontier?"true":"false",
            a->allocation_slot,a->cell,a->source_match_id,a->source_state_index,
            a->master_width,a->slot_eligible,a->slot_quota,
            (unsigned long long)a->weight_numerator,
            (unsigned long long)a->weight_denominator,ph,dh);
        fprintf(output.file,"\"orbit_sha256\":\"%s\",\"state_sha256\":\"%s\",",oh,sh);
        print_eval_policy(output.file,mv,prob,n,baseline,mask,uindex,nunion);fputc(',',output.file);
        fputs("\"maintained_baseline\":",output.file);
        if(manifest.split==SPLIT_TRAIN)fputs("null",output.file);
        else fprintf(output.file,"{\"actor_selected\":true,\"information_view\":true,"
            "\"root_seed\":\"%llu\",\"semantic_move_pack\":%u,"
            "\"truth_position\":%d,\"appended_outside_new_union\":%s,"
            "\"search_worlds\":%d,\"search_candidates\":%d,"
            "\"unfinished_cap_leaves\":%llu}",
            (unsigned long long)maintained_root_seed,
            semantic_move_pack(maintained_move),maintained_position,
            maintained_position>=mask[0].n?"true":"false",maintained_stats.worlds,
            maintained_stats.n,(unsigned long long)maintained_stats.unfinished_cap_leaves);
        fputc(',',output.file);
        fputs("\"truth_support_legal_indices\":[",output.file);
        for(int i=0;i<truth_n;i++)fprintf(output.file,"%s%d",i?",":"",truth_index_storage[i]);
        fputs("],",output.file);
        fputs("\"search\":{",output.file);
        if(!manifest.is_vector){
            print_train_pair_panel(output.file,"primary",&train_p,pair_index,ps);fputc(',',output.file);
            print_train_pair_panel(output.file,"fresh",&train_f,pair_index,fs);
        }else if(enabled_bits){
            print_search_panels(output.file,"primary",mask,&pp,enabled_bits,ps,POLICY_COST_PRIMARY_Z);fputc(',',output.file);
            print_search_panels(output.file,"fresh",mask,&fp,enabled_bits,fs,POLICY_COST_FRESH_Z);
        }else{
            fputs("\"not_opened_before_selected_onset\":true",output.file);
        }
        fputs("},",output.file);
        if(manifest.is_vector){
            print_production_decisions(output.file,&view,mv,mask,&pp,&fp,
                                       enabled_bits,manifest.split,table);fputc(',',output.file);
        }
        print_truth(output.file,&truth_panel_value,ts);fputs("}\n",output.file);
        if(!manifest.is_vector){
            primary_caps+=train_p.unfinished_cap_leaves;fresh_caps+=train_f.unfinished_cap_leaves;
            exact_leaves+=train_p.exact_terminal_leaves+train_f.exact_terminal_leaves;
        }else for(int m=0;m<PC_MASKS;m++)if((enabled_bits&(1<<m))&&
                                             !(enabled_bits==3&&m==1)){
            /* In SELECT, mask 1 is an exact submatrix of the one master
             * execution.  Count continuation work once, not once per view. */
            primary_caps+=pp.audit[m].unfinished_cap_leaves;
            fresh_caps+=fp.audit[m].unfinished_cap_leaves;
            exact_leaves+=pp.audit[m].exact_terminal_leaves+fp.audit[m].exact_terminal_leaves;
        }
        truth_free(&truth_panel_value);
    }
    fprintf(output.file,"{\"schema\":\"lc-policy-cost-evaluation-v1\","
        "\"record_type\":\"footer\",\"full_manifest_records\":%d,"
        "\"allocation_start\":%d,\"records\":%d,"
        "\"primary_unfinished_cap_leaves\":%llu,"
        "\"fresh_unfinished_cap_leaves\":%llu,\"truth_cap_hits\":%llu,"
        "\"maintained_unfinished_cap_leaves\":%llu,"
        "\"exact_terminal_leaves\":%llu,"
        "\"maintained_exact_terminal_leaves\":%llu,\"all_exact\":true,"
        "\"all_mask_overlaps_bit_exact\":%s}\n",manifest.records,allocation_start,
        allocation_count,
        (unsigned long long)primary_caps,(unsigned long long)fresh_caps,
        (unsigned long long)truth_caps,(unsigned long long)maintained_caps,
        (unsigned long long)exact_leaves,
        (unsigned long long)maintained_exact_leaves,
        manifest.split==SPLIT_SELECT?"true":"null");
    if(!atomic_finish(&output)){fprintf(stderr,"cannot atomically install evaluation output\n");goto fail;}
    {unsigned char h[32];char hex[65];if(file_sha256(c->out_path,h)){
        digest_hex(h,hex);printf("evidence_sha256=%s\n",hex);}}
    policy_cost_free(owned_table);if(have_maintained)spec_release(&maintained);
    spec_release(&actor);free(truth);bound_reservoir_free(&reservoir);allocation_free(&manifest);return 0;
output_fail:
    atomic_abort(&output);
fail:
    policy_cost_free(owned_table);if(have_maintained)spec_release(&maintained);
    spec_release(&actor);free(truth);bound_reservoir_free(&reservoir);allocation_free(&manifest);return 1;
}

static int self_test(void)
{
    unsigned char h[32];char hex[65];sha_bytes("abc",3,h);digest_hex(h,hex);
    if(strcmp(hex,"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"))return 1;
    uint64_t used[15];int nu=0;
    for(int s=0;s<3;s++){
        const uint64_t v[5]={SPLIT_DOMAIN[s].discovery_seed,SPLIT_DOMAIN[s].primary_seed,
            SPLIT_DOMAIN[s].fresh_seed,SPLIT_DOMAIN[s].truth_seed,
            SPLIT_DOMAIN[s].maintained_seed};
        for(int k=0;k<5;k++){
            if(v[k]<=200)return 2;
            for(int j=0;j<nu;j++)if(used[j]==v[k])return 3;
            used[nu++]=v[k];
        }
    }
    if(ply_bin(0)!=0||ply_bin(23)!=11||ply_bin(24)!=12||ply_bin(43)!=21||
       ply_bin(44)!=22||ply_bin(47)!=22||ply_bin(48)!=23||ply_bin(63)!=23||
       ply_bin(64)!=24||ply_bin(299)!=24||ply_bin(300)!=-1)return 4;
    Rng rng;rng_seed(&rng,UINT64_C(987654321));State complete;lc_deal(&complete,&rng);
    State view;agent_information_view(&complete,complete.turn,&view);
    unsigned char state[PC_STATE_MAX],again[PC_STATE_MAX],a[32],b[32];size_t ns=0,na=0;
    State decoded;
    if(!encode_view(&view,state,&ns)||!decode_view(state,ns,&decoded)||
       !encode_view(&decoded,again,&na)||ns!=na||memcmp(state,again,ns)||!view_valid(&decoded))return 5;
    if(!orbit_digest(&view,a))return 6;
    uint8_t perm[120][NSUIT];if(suit_permutations(120,perm)!=120)return 7;
    State relabel;lc_permute_suits(&view,&relabel,perm[73]);
    if(!orbit_digest(&relabel,b)||memcmp(a,b,32))return 8;
    /* Physical wager identity is excluded from the orbit serialization. */
    int p=view.turn;
    for(int suit=0;suit<NSUIT;suit++){
        uint64_t mask=((UINT64_C(1)<<WAGERS_PER_SUIT)-1)<<CARD_MAKE(suit,0);
        uint64_t held=view.hand[p]&mask;
        if(__builtin_popcountll(held)==1){
            int old=__builtin_ctzll(held),fresh=-1;
            for(int r=0;r<WAGERS_PER_SUIT;r++)if(!(view.hand[p]&(UINT64_C(1)<<CARD_MAKE(suit,r)))){fresh=CARD_MAKE(suit,r);break;}
            if(fresh>=0){State swapped=view;swapped.hand[p]&=~(UINT64_C(1)<<old);
                swapped.hand[p]|=UINT64_C(1)<<fresh;
                if(!orbit_digest(&swapped,b)||memcmp(a,b,32))return 9;
            }
            break;
        }
    }
    puts("policy_cost_dataset self-test: ok");return 0;
}

static void usage(FILE *f,const char *argv0)
{
    fprintf(f,
      "usage:\n"
      "  %s discover --out EVIDENCE.jsonl --reservoir-out STATES.tsv "
      "--net NET --split TRAIN|SELECT|TEST --matches N --match-start N "
      "--exclusions HASHES --exclusions-sha256 HEX "
      "[--reservoir-per-cell N] [--smoke-seed 20261229...]\n"
      "  %s evaluate --out EVIDENCE.jsonl --manifest ALLOCATION.tsv "
      "--manifest-sha256 HEX --reservoir RESERVOIR.tsv "
      "--reservoir-sha256 HEX --actor SPEC --truth-net NET "
      "[--policy-cost TABLE] [--maintained-actor LEGACY_SPEC]\n"
      "  %s verify-reservoir --out PROOF.json --manifest ALLOCATION.tsv "
      "--manifest-sha256 HEX --reservoir RESERVOIR.tsv "
      "--reservoir-sha256 HEX --net NET --exclusions HASHES "
      "--exclusions-sha256 HEX\n"
      "  %s hash-probe --state PATH\n"
      "  %s self-test\n\n"
      "discover accepts generated exact-policy-20 self-play only; it has no "
      "saved-state or probe option. Campaign seeds are fixed by split. A smoke "
      "seed must be in the burned 20261229 namespace and is labeled unusable "
      "for allocation. hash-probe is audit-only: it reads a text state solely "
      "to print its canonical state, information-view, and suit-orbit hashes; "
      "it cannot discover, allocate, or evaluate evidence. evaluate requires an exact, full 3-round x 24 pre-64 "
      "ply-bin x 6 ratio-bin x 2 pair-type allocation; nply>=64 is pooled "
      "diagnostic occupancy with a predeclared minimum. Sparse cells fail "
      "closed: there is no replacement, top-up, optional stopping, or union "
      "truncation. Source deal seeds 1..200, the interrupted seed-1 maintained "
      "smoke, 202611010101, every policy-cost-v1 fixed seed and 20261129 "
      "smoke seed, 202612010101, and every 20261229 feasibility-smoke seed "
      "are permanently outside all fixed "
       "campaign domains.\n",argv0,argv0,argv0,argv0,argv0);
}

int main(int argc,char **argv)
{
    if(argc==2&&!strcmp(argv[1],"self-test"))return self_test();
    if(argc>=2&&!strcmp(argv[1],"hash-probe")){
        const char *path=NULL;
        if(!hash_probe_args(argc,argv,&path)){usage(stderr,argv[0]);return 1;}
        return run_hash_probe(path);
    }
    if(argc>=2&&!strcmp(argv[1],"discover")){
        DiscoverConfig c;if(!discover_args(argc,argv,&c)){usage(stderr,argv[0]);return 1;}
        return run_discover(&c);
    }
    if(argc>=2&&!strcmp(argv[1],"evaluate")){
        EvaluateConfig c;if(!evaluate_args(argc,argv,&c)){usage(stderr,argv[0]);return 1;}
        return run_evaluate(&c);
    }
    if(argc>=2&&!strcmp(argv[1],"verify-reservoir")){
        VerifyReservoirConfig c;
        if(!verify_reservoir_args(argc,argv,&c)){usage(stderr,argv[0]);return 1;}
        return run_verify_reservoir(&c);
    }
    usage(argc>1&&(!strcmp(argv[1],"-h")||!strcmp(argv[1],"--help"))?stdout:stderr,argv[0]);
    return argc>1&&(!strcmp(argv[1],"-h")||!strcmp(argv[1],"--help"))?0:1;
}
