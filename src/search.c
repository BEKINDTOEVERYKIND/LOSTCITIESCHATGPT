#include "search.h"
#include "agent.h"
#include <math.h>
#include <stdlib.h>
#include <string.h>

#define MAXC 16

typedef struct {
    uint16_t mv[MAXC];
    int32_t  child[MAXC];
    float    prior[MAXC];
    int32_t  nvisit[MAXC];
    float    wsum[MAXC];
    int32_t  visits;
    int16_t  nchild;
    uint8_t  expanded;
    float    value;      /* leaf value, node player's view, VAL_SCALE units */
} Node;

typedef struct {
    Node *nodes;
    int nnode, cap;
    const Net *net;
    float cpuct;
    int width;
} Tree;

static _Thread_local Node *g_pool = NULL;
static _Thread_local int g_pool_cap = 0;

static void node_reset(Node *n)
{
    memset(n, 0, sizeof(*n));
    for (int i = 0; i < MAXC; i++) n->child[i] = -1;
}

static int new_node(Tree *t)
{
    if (t->nnode >= t->cap) return -1;
    Node *n = &t->nodes[t->nnode];
    node_reset(n);
    return t->nnode++;
}

static float terminal_value(const State *s)
{
    int p = s->turn;
    return (float)(lc_score(s, p) - lc_score(s, p ^ 1)) / VAL_SCALE;
}

/* Fill a node from a set of moves and their priors, keeping the best `width`. */
static void node_fill(Node *nd, const Move *mv, const float *prob, int n, int width)
{
    node_reset(nd);
    if (width > MAXC) width = MAXC;
    Move lm[MAX_MOVES];
    float lp[MAX_MOVES];
    for (int i = 0; i < n; i++) { lm[i] = mv[i]; lp[i] = prob[i]; }
    if (n > width) {
        for (int i = 0; i < width; i++) {
            int best = i;
            for (int j = i + 1; j < n; j++) if (lp[j] > lp[best]) best = j;
            if (best != i) {
                float t = lp[i]; lp[i] = lp[best]; lp[best] = t;
                Move m = lm[i]; lm[i] = lm[best]; lm[best] = m;
            }
        }
        n = width;
    }
    float sum = 0.0f;
    for (int i = 0; i < n; i++) sum += lp[i];
    if (sum <= 0.0f) { for (int i = 0; i < n; i++) lp[i] = 1.0f; sum = (float)n; }
    for (int i = 0; i < n; i++) {
        nd->mv[i] = MOVE_PACK(lm[i]);
        nd->child[i] = -1;
        nd->nvisit[i] = 0;
        nd->wsum[i] = 0.0f;
        nd->prior[i] = lp[i] / sum;
    }
    nd->nchild = (int16_t)n;
    nd->expanded = 1;
}

/* One trunk pass gives both the leaf value and the priors. */
static void expand(Tree *t, Node *nd, const State *s, int width)
{
    Move mv[MAX_MOVES];
    float prob[MAX_MOVES];
    float value = 0.0f;
    int n = policy_probs(t->net, s, mv, prob, &value);
    node_fill(nd, mv, prob, n, width);
    nd->value = value / VAL_SCALE;
}

static float simulate(Tree *t, State *s, int node)
{
    if (s->over) return terminal_value(s);
    Node *nd = &t->nodes[node];
    if (!nd->expanded) {
        expand(t, nd, s, t->width);
        return nd->value;
    }
    float sq = sqrtf((float)(nd->visits + 1));
    int best = 0;
    float bestu = -1e30f;
    for (int i = 0; i < nd->nchild; i++) {
        float q = nd->nvisit[i] > 0 ? nd->wsum[i] / (float)nd->nvisit[i]
                                    : nd->value - 0.05f; /* first play urgency */
        float u = q + t->cpuct * nd->prior[i] * sq / (float)(1 + nd->nvisit[i]);
        if (u > bestu) { bestu = u; best = i; }
    }
    uint16_t pk = nd->mv[best];
    Move m = { MOVE_CARD(pk), MOVE_DISC(pk), MOVE_DRAW(pk) };
    lc_apply(s, m);
    int c = nd->child[best];
    if (c < 0) {
        c = new_node(t);
        if (c < 0) {
            float lv;
            if (s->over) lv = terminal_value(s);
            else { Features f; feat_extract(s, s->turn, &f); lv = net_value(t->net, &f); }
            nd->nvisit[best]++;
            nd->wsum[best] += -lv;
            nd->visits++;
            return -lv;
        }
        nd->child[best] = c;
    }
    float v = -simulate(t, s, c);
    nd->nvisit[best]++;
    nd->wsum[best] += v;
    nd->visits++;
    return v;
}

Move search_move(const struct Agent *a, const State *st, Rng *rng,
                 float *out_value, SearchStats *stats)
{
    Move rmv[MAX_MOVES];
    float rprob[MAX_MOVES];
    float rvalue = 0.0f;
    int rn = policy_probs(a->net, st, rmv, rprob, &rvalue);
    if (rn <= 1) {
        if (out_value) *out_value = rvalue;
        if (stats) {
            stats->n = rn;
            if (rn == 1) { stats->mv[0] = rmv[0]; stats->visits[0] = 1; stats->q[0] = rvalue; }
            stats->value = rvalue;
        }
        return rmv[0];
    }
    /* The root policy and value depend only on the mover's information set, so
     * they are identical in every determinization and are computed once. */
    int rw = a->root_width > MAXC ? MAXC : a->root_width;

    int cap = a->sims + 4;
    if (g_pool_cap < cap) {
        free(g_pool);
        g_pool = (Node *)malloc(sizeof(Node) * (size_t)cap);
        g_pool_cap = cap;
    }

    Node root_tmpl;
    node_fill(&root_tmpl, rmv, rprob, rn, rw);
    root_tmpl.value = rvalue / VAL_SCALE;
    int nroot = root_tmpl.nchild;

    double agg_visits[MAXC], agg_w[MAXC];
    for (int i = 0; i < nroot; i++) { agg_visits[i] = 0.0; agg_w[i] = 0.0; }

    Tree t;
    t.nodes = g_pool; t.cap = cap; t.net = a->net;
    t.cpuct = a->cpuct; t.width = a->node_width;

    for (int d = 0; d < a->dets; d++) {
        State root;
        determinize_b(st, st->turn, rng, a->net, &root);
        t.nnode = 0;
        int rootn = new_node(&t);
        t.nodes[rootn] = root_tmpl;
        for (int i = 0; i < a->sims; i++) {
            State s = root;
            simulate(&t, &s, rootn);
        }
        Node *nd = &t.nodes[rootn];
        for (int i = 0; i < nd->nchild; i++) {
            agg_visits[i] += nd->nvisit[i];
            agg_w[i] += nd->wsum[i];
        }
    }

    int best = 0;
    double bestv = -1e30;
    double totv = 0.0, totw = 0.0;
    for (int i = 0; i < nroot; i++) {
        totv += agg_visits[i];
        totw += agg_w[i];
        double q = agg_visits[i] > 0 ? agg_w[i] / agg_visits[i] : -1e9;
        double s = agg_visits[i] + 0.001 * q;
        if (s > bestv) { bestv = s; best = i; }
    }
    float rootq = totv > 0 ? (float)(totw / totv) * VAL_SCALE : rvalue;
    if (stats) {
        stats->n = nroot;
        for (int i = 0; i < nroot; i++) {
            uint16_t pk = root_tmpl.mv[i];
            Move m = { MOVE_CARD(pk), MOVE_DISC(pk), MOVE_DRAW(pk) };
            stats->mv[i] = m;
            stats->visits[i] = agg_visits[i];
            stats->q[i] = agg_visits[i] > 0 ? agg_w[i] / agg_visits[i] * VAL_SCALE : 0.0;
        }
        stats->value = rootq;
    }
    if (out_value) *out_value = rootq;
    uint16_t pk = root_tmpl.mv[best];
    Move m = { MOVE_CARD(pk), MOVE_DISC(pk), MOVE_DRAW(pk) };
    return m;
}
