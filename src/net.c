#include "net.h"
#include <math.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

static void net_copy_wager_symmetry(Net *n);

static float gauss(Rng *r)
{
    float u1 = rng_float(r) + 1e-7f, u2 = rng_float(r);
    return sqrtf(-2.0f * logf(u1)) * cosf(6.2831853f * u2);
}

void net_init(Net *n, uint64_t seed)
{
    Rng r; rng_seed(&r, seed);
    float s1 = sqrtf(2.0f / (float)FEAT_DIM);
    for (int i = 0; i < FEAT_DIM; i++)
        for (int h = 0; h < NET_H1; h++) n->w1[i][h] = gauss(&r) * s1;
    for (int h = 0; h < NET_H1; h++) n->b1[h] = 0.0f;
    float s2 = sqrtf(2.0f / (float)NET_H1);
    for (int i = 0; i < NET_H1; i++)
        for (int h = 0; h < NET_H2; h++) n->w2[i][h] = gauss(&r) * s2;
    for (int h = 0; h < NET_H2; h++) n->b2[h] = 0.0f;
    float s3 = sqrtf(1.0f / (float)NET_H2);
    for (int h = 0; h < NET_H2; h++) n->w3[h] = gauss(&r) * s3;
    n->b3 = 0.0f;
    float s4 = 0.1f * sqrtf(1.0f / (float)NET_H2);
    for (int i = 0; i < NET_NPLAY; i++) {
        for (int h = 0; h < NET_H2; h++) n->wplay[i][h] = gauss(&r) * s4;
        n->bplay[i] = 0.0f;
    }
    for (int i = 0; i < NET_NDRAW; i++) {
        for (int h = 0; h < NET_H2; h++) n->wdraw[i][h] = gauss(&r) * s4;
        n->bdraw[i] = 0.0f;
    }
    for (int i = 0; i < NCARD; i++) {
        for (int h = 0; h < NET_H2; h++) n->wbel[i][h] = gauss(&r) * s4;
        n->bbel[i] = 0.0f;
    }
    /* Start the interaction head as an exact additive-policy residual.  Its
     * rows receive independent gradients from the first policy update. */
    for (int i = 0; i < NET_NCOMB; i++) {
        for (int h = 0; h < NET_H2; h++) n->wcomb[i][h] = 0.0f;
        n->bcomb[i] = 0.0f;
    }
    /* Physical wager IDs are not observable card types.  Randomly
     * initializing three separate rows needlessly breaks an exact symmetry
     * before the first sample; copy one correctly-scaled draw instead. */
    net_copy_wager_symmetry(n);
}

/* random-init only the belief head (for upgrading older files) */
static void net_init_belief(Net *n, uint64_t seed)
{
    Rng r; rng_seed(&r, seed);
    float s4 = 0.1f * sqrtf(1.0f / (float)NET_H2);
    for (int i = 0; i < NCARD; i++) {
        for (int h = 0; h < NET_H2; h++) n->wbel[i][h] = gauss(&r) * s4;
        n->bbel[i] = 0.0f;
    }
}

void net_zero(Net *n)
{
    memset(n, 0, sizeof(*n));
}

static void tie_three_rows(float *a, size_t stride, int i0, int i1, int i2)
{
    float *r0 = a + (size_t)i0 * stride;
    float *r1 = a + (size_t)i1 * stride;
    float *r2 = a + (size_t)i2 * stride;
    for (size_t j = 0; j < stride; j++) {
        if (r0[j] == r1[j] && r0[j] == r2[j]) continue;
        float mean = (r0[j] + r1[j] + r2[j]) * (1.0f / 3.0f);
        r0[j] = mean;
        r1[j] = mean;
        r2[j] = mean;
    }
}

static void copy_three_rows(float *a, size_t stride, int i0, int i1, int i2)
{
    float *r0 = a + (size_t)i0 * stride;
    float *r1 = a + (size_t)i1 * stride;
    float *r2 = a + (size_t)i2 * stride;
    memcpy(r1, r0, stride * sizeof(float));
    memcpy(r2, r0, stride * sizeof(float));
}

static void sum_three_rows(float *a, size_t stride, int i0, int i1, int i2)
{
    float *r0 = a + (size_t)i0 * stride;
    float *r1 = a + (size_t)i1 * stride;
    float *r2 = a + (size_t)i2 * stride;
    for (size_t j = 0; j < stride; j++) {
        float sum = r0[j] + r1[j] + r2[j];
        r0[j] = sum;
        r1[j] = sum;
        r2[j] = sum;
    }
}

typedef void (*row_op)(float *, size_t, int, int, int);

static void wager_belief_groups(Net *n, row_op op)
{
    for (int s = 0; s < NSUIT; s++) {
        int card = s * NRANK;
        op(&n->wbel[0][0], NET_H2, card, card + 1, card + 2);
        op(n->bbel, 1, card, card + 1, card + 2);
    }
}

static void wager_row_groups(Net *n, row_op op)
{
    for (int plane = 0; plane < FEAT_PLANES; plane++)
        for (int s = 0; s < NSUIT; s++) {
            int base = plane * NCARD + s * NRANK;
            op(&n->w1[0][0], NET_H1, base, base + 1, base + 2);
        }

    for (int s = 0; s < NSUIT; s++) {
        int card = s * NRANK;
        for (int discard = 0; discard < 2; discard++) {
            int p0 = card * 2 + discard;
            int p1 = (card + 1) * 2 + discard;
            int p2 = (card + 2) * 2 + discard;
            op(&n->wplay[0][0], NET_H2, p0, p1, p2);
            op(n->bplay, 1, p0, p1, p2);

            for (int draw = 0; draw < NET_NDRAW; draw++) {
                int c0 = p0 * NET_NDRAW + draw;
                int c1 = p1 * NET_NDRAW + draw;
                int c2 = p2 * NET_NDRAW + draw;
                op(&n->wcomb[0][0], NET_H2, c0, c1, c2);
                op(n->bcomb, 1, c0, c1, c2);
            }
        }
    }
    wager_belief_groups(n, op);
}

static void net_copy_wager_symmetry(Net *n)
{
    wager_row_groups(n, copy_three_rows);
}

void net_project_wager_symmetry(Net *n)
{
    wager_row_groups(n, tie_three_rows);
}

void net_project_belief_wager_symmetry(Net *n)
{
    wager_belief_groups(n, tie_three_rows);
}

void net_tie_wager_gradients(Net *g)
{
    wager_row_groups(g, sum_three_rows);
}

void net_trunk(const Net *n, const Features *f, NetAct *act)
{
    float h1[NET_H1];
    for (int h = 0; h < NET_H1; h++) h1[h] = n->b1[h];
    for (int k = 0; k < f->nidx; k++) {
        const float *w = n->w1[f->idx[k]];
        for (int h = 0; h < NET_H1; h++) h1[h] += w[h];
    }
    for (int j = 0; j < FEAT_DENSE; j++) {
        float x = f->dense[j];
        if (x == 0.0f) continue;
        const float *w = n->w1[FEAT_BIN + j];
        for (int h = 0; h < NET_H1; h++) h1[h] += x * w[h];
    }
    for (int h = 0; h < NET_H1; h++) act->a1[h] = h1[h] > 0.0f ? h1[h] : 0.0f;

    float h2[NET_H2];
    for (int h = 0; h < NET_H2; h++) h2[h] = n->b2[h];
    for (int i = 0; i < NET_H1; i++) {
        float a = act->a1[i];
        if (a == 0.0f) continue;
        const float *w = n->w2[i];
        for (int h = 0; h < NET_H2; h++) h2[h] += a * w[h];
    }
    for (int h = 0; h < NET_H2; h++) act->a2[h] = h2[h] > 0.0f ? h2[h] : 0.0f;
}

float net_value_act(const Net *n, const NetAct *act)
{
    float o = n->b3;
    for (int h = 0; h < NET_H2; h++) o += act->a2[h] * n->w3[h];
    return o;
}

static inline float dot_h2(const float *w, const float *a)
{
    float o = 0.0f;
    for (int h = 0; h < NET_H2; h++) o += a[h] * w[h];
    return o;
}

void net_policy_act(const Net *n, const NetAct *act, const uint16_t *mv, int nmv, float *logits)
{
    float pl[NET_NPLAY], dr[NET_NDRAW];
    uint8_t hp[NET_NPLAY] = { 0 }, hd[NET_NDRAW] = { 0 };
    for (int i = 0; i < nmv; i++) {
        int ip = MOVE_CARD(mv[i]) * 2 + MOVE_DISC(mv[i]);
        int id = MOVE_DRAW(mv[i]);
        int ic = ip * NET_NDRAW + id;
        if (!hp[ip]) { pl[ip] = n->bplay[ip] + dot_h2(n->wplay[ip], act->a2); hp[ip] = 1; }
        if (!hd[id]) { dr[id] = n->bdraw[id] + dot_h2(n->wdraw[id], act->a2); hd[id] = 1; }
        logits[i] = pl[ip] + dr[id]
                  + n->bcomb[ic] + dot_h2(n->wcomb[ic], act->a2);
    }
}

void net_belief_act(const Net *n, const NetAct *act, const uint8_t *cards, int nc, float *logits)
{
    for (int i = 0; i < nc; i++)
        logits[i] = n->bbel[cards[i]] + dot_h2(n->wbel[cards[i]], act->a2);
}

float net_value(const Net *n, const Features *f)
{
    NetAct act;
    net_trunk(n, f, &act);
    return net_value_act(n, &act);
}

void net_backward(const Net *n, const Features *f, const NetAct *act,
                  float dvalue, const uint16_t *mv, const float *dlogit, int nmv,
                  const uint8_t *bc, const float *dbel, int nb,
                  Net *g)
{
    float d2[NET_H2];
    for (int h = 0; h < NET_H2; h++) d2[h] = 0.0f;

    if (dvalue != 0.0f) {
        g->b3 += dvalue;
        for (int h = 0; h < NET_H2; h++) {
            g->w3[h] += dvalue * act->a2[h];
            d2[h] += dvalue * n->w3[h];
        }
    }
    if (dlogit) {
        /* Sum the per-move gradient into the components it is built from, then
         * touch each component's weight row once. */
        float sp[NET_NPLAY], sd[NET_NDRAW];
        int plist[MAX_MOVES], dlist[NET_NDRAW];
        int np = 0, nd = 0;
        uint8_t hp[NET_NPLAY] = { 0 }, hd[NET_NDRAW] = { 0 };
        for (int i = 0; i < nmv; i++) {
            int ip = MOVE_CARD(mv[i]) * 2 + MOVE_DISC(mv[i]);
            int id = MOVE_DRAW(mv[i]);
            int ic = ip * NET_NDRAW + id;
            if (!hp[ip]) { hp[ip] = 1; sp[ip] = 0.0f; plist[np++] = ip; }
            if (!hd[id]) { hd[id] = 1; sd[id] = 0.0f; dlist[nd++] = id; }
            sp[ip] += dlogit[i];
            sd[id] += dlogit[i];
            float d = dlogit[i];
            if (d != 0.0f) {
                float *gw = g->wcomb[ic];
                const float *w = n->wcomb[ic];
                g->bcomb[ic] += d;
                for (int h = 0; h < NET_H2; h++) {
                    gw[h] += d * act->a2[h];
                    d2[h] += d * w[h];
                }
            }
        }
        for (int k = 0; k < np; k++) {
            int ip = plist[k];
            float d = sp[ip];
            if (d == 0.0f) continue;
            float *gw = g->wplay[ip];
            const float *w = n->wplay[ip];
            g->bplay[ip] += d;
            for (int h = 0; h < NET_H2; h++) { gw[h] += d * act->a2[h]; d2[h] += d * w[h]; }
        }
        for (int k = 0; k < nd; k++) {
            int id = dlist[k];
            float d = sd[id];
            if (d == 0.0f) continue;
            float *gw = g->wdraw[id];
            const float *w = n->wdraw[id];
            g->bdraw[id] += d;
            for (int h = 0; h < NET_H2; h++) { gw[h] += d * act->a2[h]; d2[h] += d * w[h]; }
        }
    }
    if (dbel) {
        for (int i = 0; i < nb; i++) {
            float dv = dbel[i];
            if (dv == 0.0f) continue;
            int card = bc[i];
            float *gw = g->wbel[card];
            const float *w = n->wbel[card];
            g->bbel[card] += dv;
            for (int h = 0; h < NET_H2; h++) { gw[h] += dv * act->a2[h]; d2[h] += dv * w[h]; }
        }
    }
    for (int h = 0; h < NET_H2; h++) if (act->a2[h] <= 0.0f) d2[h] = 0.0f;

    float d1[NET_H1];
    for (int h = 0; h < NET_H2; h++) g->b2[h] += d2[h];
    for (int i = 0; i < NET_H1; i++) {
        float a = act->a1[i];
        if (a != 0.0f) {
            float *gw = g->w2[i];
            const float *w = n->w2[i];
            float acc = 0.0f;
            for (int h = 0; h < NET_H2; h++) { gw[h] += a * d2[h]; acc += w[h] * d2[h]; }
            d1[i] = acc;
        } else {
            d1[i] = 0.0f;
        }
    }
    for (int h = 0; h < NET_H1; h++) g->b1[h] += d1[h];
    for (int k = 0; k < f->nidx; k++) {
        float *gw = g->w1[f->idx[k]];
        for (int h = 0; h < NET_H1; h++) gw[h] += d1[h];
    }
    for (int j = 0; j < FEAT_DENSE; j++) {
        float x = f->dense[j];
        if (x == 0.0f) continue;
        float *gw = g->w1[FEAT_BIN + j];
        for (int h = 0; h < NET_H1; h++) gw[h] += x * d1[h];
    }
}

void net_backward_belief_head(const NetAct *act, const uint8_t *bc,
                              const float *dbel, int nb, Net *g)
{
    if (!act || !bc || !dbel || !g) return;
    for (int i = 0; i < nb; i++) {
        float d = dbel[i];
        if (d == 0.0f) continue;
        int card = bc[i];
        g->bbel[card] += d;
        for (int h = 0; h < NET_H2; h++)
            g->wbel[card][h] += d * act->a2[h];
    }
}

static void adam_step_range(Net *n, const Net *g, Adam *a, float lr,
                            float scale, float wd, size_t from, size_t to)
{
    a->t++;
    const float b1 = 0.9f, b2 = 0.999f, eps = 1e-8f;
    float bc1 = 1.0f - powf(b1, (float)a->t);
    float bc2 = 1.0f - powf(b2, (float)a->t);
    float step = lr * sqrtf(bc2) / bc1;

    float *w = (float *)n, *gm = (float *)&a->m, *gv = (float *)&a->v;
    const float *gr = (const float *)g;
    for (size_t i = from; i < to; i++) {
        float grad = gr[i] * scale + wd * w[i];
        gm[i] = b1 * gm[i] + (1.0f - b1) * grad;
        gv[i] = b2 * gv[i] + (1.0f - b2) * grad * grad;
        w[i] -= step * gm[i] / (sqrtf(gv[i]) + eps);
    }
}

void net_adam_step(Net *n, const Net *g, Adam *a, float lr, float scale, float wd)
{
    size_t nw = sizeof(Net) / sizeof(float);
    adam_step_range(n, g, a, lr, scale, wd, 0, nw);
}

void net_adam_step_belief(Net *n, const Net *g, Adam *a,
                          float lr, float scale, float wd)
{
    _Static_assert(offsetof(Net, wbel) % sizeof(float) == 0,
                   "belief head must be float aligned");
    _Static_assert(offsetof(Net, wcomb) ==
                   offsetof(Net, bbel) + sizeof(((Net *)0)->bbel),
                   "belief weights and biases must be one contiguous range");
    size_t from = offsetof(Net, wbel) / sizeof(float);
    size_t to = offsetof(Net, wcomb) / sizeof(float);
    adam_step_range(n, g, a, lr, scale, wd, from, to);
}

#define NET_MAGIC 0x4C435651U /* "LCVQ" */
#define NET_VERSION 6U

_Static_assert(NET_NCOMB == MOVE_NPACK,
               "combination policy head must cover every packed move");
_Static_assert(FEAT_LEGACY_DIM == 556,
               "legacy loader requires the historical 556-feature layout");
_Static_assert(FEAT_DIM > FEAT_LEGACY_DIM,
               "new pile-order features must be appended to the old layout");
_Static_assert(offsetof(Net, b1) ==
               (size_t)FEAT_DIM * NET_H1 * sizeof(float),
               "first-layer rows must be contiguous");
_Static_assert(sizeof(Net) ==
               offsetof(Net, bcomb) + sizeof(((Net *)0)->bcomb),
               "v6 payload must end immediately after combination head");

static int write_exact(FILE *fp, const void *p, size_t n)
{
    return fwrite(p, 1, n, fp) == n;
}

static int read_exact(FILE *fp, void *p, size_t n)
{
    return fread(p, 1, n, fp) == n;
}

/* Versions 3--5 all used the 556-feature first layer.  The larger v6 w1
 * matrix inserts new rows before b1, so an old raw struct cannot be read as a
 * prefix of the new one.  Read its fields explicitly and leave the appended
 * input rows (and any head not present in that version) at zero. */
static int read_legacy(FILE *fp, Net *n, uint32_t version)
{
    size_t old_w1 = (size_t)FEAT_LEGACY_DIM * sizeof(n->w1[0]);
    if (!read_exact(fp, n->w1, old_w1)
        || !read_exact(fp, n->b1, sizeof(n->b1))
        || !read_exact(fp, n->w2, sizeof(n->w2))
        || !read_exact(fp, n->b2, sizeof(n->b2))
        || !read_exact(fp, n->w3, sizeof(n->w3))
        || !read_exact(fp, &n->b3, sizeof(n->b3))
        || !read_exact(fp, n->wplay, sizeof(n->wplay))
        || !read_exact(fp, n->bplay, sizeof(n->bplay))
        || !read_exact(fp, n->wdraw, sizeof(n->wdraw))
        || !read_exact(fp, n->bdraw, sizeof(n->bdraw)))
        return 0;

    if (version >= 4U
        && (!read_exact(fp, n->wbel, sizeof(n->wbel))
            || !read_exact(fp, n->bbel, sizeof(n->bbel))))
        return 0;
    if (version >= 5U
        && (!read_exact(fp, n->wcomb, sizeof(n->wcomb))
            || !read_exact(fp, n->bcomb, sizeof(n->bcomb))))
        return 0;
    return 1;
}

int net_save(const Net *n, const char *path)
{
    FILE *fp = fopen(path, "wb");
    if (!fp) return -1;
    const uint32_t hdr[6] = {
        NET_MAGIC, FEAT_DIM, NET_H1, NET_H2, NET_NPLAY, NET_VERSION
    };
    int ok = write_exact(fp, hdr, sizeof(hdr))
          && write_exact(fp, n, sizeof(*n));
    if (fclose(fp) != 0) ok = 0;
    return ok ? 0 : -1;
}

int net_load(Net *n, const char *path)
{
    FILE *fp = fopen(path, "rb");
    if (!fp) return -1;
    uint32_t hdr[6];
    if (!read_exact(fp, hdr, sizeof(hdr))) { fclose(fp); return -1; }
    if (hdr[0] != NET_MAGIC || hdr[2] != NET_H1 ||
        hdr[3] != NET_H2 || hdr[4] != NET_NPLAY) { fclose(fp); return -2; }

    int current = hdr[5] == NET_VERSION && hdr[1] == FEAT_DIM;
    int legacy = hdr[5] >= 3U && hdr[5] <= 5U
              && hdr[1] == FEAT_LEGACY_DIM;
    if (!current && !legacy) { fclose(fp); return -2; }

    /* Load into a zeroed temporary object so a short/corrupt file neither
     * partially mutates the caller's net nor leaves newly added rows live. */
    Net *tmp = (Net *)calloc(1, sizeof(*tmp));
    if (!tmp) { fclose(fp); return -1; }
    int ok = current ? read_exact(fp, tmp, sizeof(*tmp))
                     : read_legacy(fp, tmp, hdr[5]);
    if (ok && hdr[5] == 3U) net_init_belief(tmp, 0xBE11EFULL);

    /* A version denotes one exact raw payload layout.  Reject both truncation
     * and trailing data rather than accepting a mismatched struct silently. */
    if (ok) {
        int c = fgetc(fp);
        if (c != EOF || ferror(fp)) ok = 0;
    }
    if (fclose(fp) != 0) ok = 0;
    if (!ok) { free(tmp); return -1; }

    memcpy(n, tmp, sizeof(*n));
    free(tmp);
    return 0;
}
