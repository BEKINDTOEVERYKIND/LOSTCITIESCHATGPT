#include "agent.h"
#include "heuristic.h"
#include "search.h"
#include <math.h>

void agent_default(Agent *a, AgentKind k, const Net *net)
{
    memset(a, 0, sizeof(*a));
    a->kind = k;
    a->net = net;
    a->draw_samples = 6;
    a->temp = 0.0f;
    a->eps = 0.0f;
    a->dets = 16;
    a->sims = 160;
    a->root_width = 14;
    a->node_width = 8;
    a->cpuct = 1.4f;
    a->cand_floor = 0.02f;
    a->min_cand = 1;
    a->ply_lo = 0;
    a->ply_hi = 0;
    a->eval_cand = 0;
    a->win_q = 0;
    /* This is an optional search-focus heuristic, not true dominance:
     * changing which discard pile is covered can change later play.  Keep it
     * opt-in until exhaustive endgame tests can establish safe conditions. */
    a->prune_dom = 0;
    a->override_k = 0.0f;
    a->override_min = 4.0f;
    a->playout_sample = 0;
    switch (k) {
    case AG_RANDOM: a->name = "random"; break;
    case AG_HEUR:   a->name = "heuristic"; break;
    case AG_NET:    a->name = "net1"; break;
    case AG_POLICY: a->name = "policy"; break;
    case AG_MCTS:   a->name = "mcts"; break;
    case AG_ROLLOUT: a->name = "rollout"; a->dets = 128; a->root_width = 4; break;
    }
}

void determinize(const State *st, int p, Rng *rng, State *out)
{
    *out = *st;
    uint8_t unseen[NCARD];
    int n = 0;
    lc_unseen(st, p, unseen, &n);   /* already excludes cards known to be held */
    for (int i = n - 1; i > 0; i--) {
        uint32_t j = rng_below(rng, (uint32_t)i + 1);
        uint8_t t = unseen[i]; unseen[i] = unseen[j]; unseen[j] = t;
    }
    const int o = p ^ 1;
    /* the opponent certainly holds every card they took face up */
    out->hand[o] = st->known[o];
    int need = (int)st->hand_n[o] - __builtin_popcountll(st->known[o]);
    int k = 0;
    while (need-- > 0) out->hand[o] |= 1ULL << unseen[k++];
    out->deck_pos = 0;
    memset(out->deck, 0, sizeof(out->deck));
    int d = 0;
    while (k < n) out->deck[d++] = unseen[k++];
    out->deck_left = (uint8_t)d;
}

/* Sample the opponent's unknown cards from the belief posterior using
 * Gumbel-top-k on the logits: an exact draw from the Plackett-Luce
 * distribution the logits induce, so likelier hands appear in more worlds. */
void determinize_b(const State *st, int p, Rng *rng, const Net *net, State *out)
{
    if (!net) { determinize(st, p, rng, out); return; }
    *out = *st;
    uint8_t unseen[NCARD];
    int n = 0;
    lc_unseen(st, p, unseen, &n);
    const int o = p ^ 1;
    int need = (int)st->hand_n[o] - __builtin_popcountll(st->known[o]);
    if (need <= 0 || n == 0) { determinize(st, p, rng, out); return; }

    Features f;
    feat_extract(st, p, &f);
    NetAct act;
    net_trunk(net, &f, &act);
    float logit[NCARD];
    net_belief_act(net, &act, unseen, n, logit);

    /* keys = logit + Gumbel noise; the top `need` keys form the hand */
    float key[NCARD];
    int order[NCARD];
    for (int i = 0; i < n; i++) {
        float u = rng_float(rng) + 1e-7f;
        float l = logit[i];
        if (l > 15.0f) l = 15.0f;
        if (l < -15.0f) l = -15.0f;
        key[i] = l - logf(-logf(u));
        order[i] = i;
    }
    for (int i = 0; i < need; i++) {
        int best = i;
        for (int j = i + 1; j < n; j++) if (key[order[j]] > key[order[best]]) best = j;
        int t = order[i]; order[i] = order[best]; order[best] = t;
    }
    out->hand[o] = st->known[o];
    for (int i = 0; i < need; i++) out->hand[o] |= 1ULL << unseen[order[i]];
    /* remaining cards form the deck in random order */
    out->deck_pos = 0;
    memset(out->deck, 0, sizeof(out->deck));
    int d = 0;
    for (int i = need; i < n; i++) out->deck[d++] = unseen[order[i]];
    for (int i = d - 1; i > 0; i--) {
        uint32_t j = rng_below(rng, (uint32_t)i + 1);
        uint8_t t = out->deck[i]; out->deck[i] = out->deck[j]; out->deck[j] = t;
    }
    out->deck_left = (uint8_t)d;
}

void draw_samples_init(const State *st, int p, Rng *rng, int k, DrawSamples *ds)
{
    uint8_t unseen[NCARD];
    int n = 0;
    lc_unseen(st, p, unseen, &n);
    if (k > MAX_DRAW_SAMPLES) k = MAX_DRAW_SAMPLES;
    if (k < 1) k = 1;
    if (k >= n) {
        for (int i = 0; i < n; i++) ds->card[i] = unseen[i];
        ds->n = n;
        return;
    }
    /* sample k distinct cards: partial Fisher-Yates */
    for (int i = 0; i < k; i++) {
        uint32_t j = i + rng_below(rng, (uint32_t)(n - i));
        uint8_t t = unseen[i]; unseen[i] = unseen[j]; unseen[j] = t;
        ds->card[i] = unseen[i];
    }
    ds->n = k;
}

float move_value_net(const Net *net, const State *st, Move m, const DrawSamples *ds)
{
    const int p = st->turn;
    State base = *st;
    lc_apply_play(&base, m);
    Features f;

    if (m.draw > 0) {
        State s2 = base;
        lc_apply_draw(&s2, m, -1);   /* the pile top is public */
        if (s2.over) return (float)(lc_score(&s2, p) - lc_score(&s2, p ^ 1));
        feat_extract(&s2, p, &f);
        return net_value(net, &f) * VAL_SCALE;
    }
    if (ds->n == 0) {
        State s2 = base;
        lc_apply_draw(&s2, m, -1);
        if (s2.over) return (float)(lc_score(&s2, p) - lc_score(&s2, p ^ 1));
        feat_extract(&s2, p, &f);
        return net_value(net, &f) * VAL_SCALE;
    }
    float sum = 0.0f;
    for (int i = 0; i < ds->n; i++) {
        State s2 = base;
        lc_apply_draw(&s2, m, ds->card[i]);
        if (s2.over) {
            sum += (float)(lc_score(&s2, p) - lc_score(&s2, p ^ 1));
        } else {
            feat_extract(&s2, p, &f);
            sum += net_value(net, &f) * VAL_SCALE;
        }
    }
    return sum / (float)ds->n;
}

float move_value_heur(const State *st, Move m, const DrawSamples *ds)
{
    const int p = st->turn;
    State base = *st;
    lc_apply_play(&base, m);

    if (m.draw > 0 || ds->n == 0) {
        State s2 = base;
        lc_apply_draw(&s2, m, -1);
        if (s2.over) return (float)(lc_score(&s2, p) - lc_score(&s2, p ^ 1));
        return heur_eval(&s2, p);
    }
    float sum = 0.0f;
    for (int i = 0; i < ds->n; i++) {
        State s2 = base;
        lc_apply_draw(&s2, m, ds->card[i]);
        sum += s2.over ? (float)(lc_score(&s2, p) - lc_score(&s2, p ^ 1)) : heur_eval(&s2, p);
    }
    return sum / (float)ds->n;
}

int policy_probs(const Net *net, const State *st, Move *mv, float *prob, float *value)
{
    int n = lc_moves(st, mv);
    if (n == 0) return 0;
    uint16_t pk[MAX_MOVES];
    for (int i = 0; i < n; i++) pk[i] = MOVE_PACK(mv[i]);
    Features f;
    feat_extract(st, st->turn, &f);
    NetAct act;
    net_trunk(net, &f, &act);
    if (value) *value = net_value_act(net, &act) * VAL_SCALE;
    float lg[MAX_MOVES];
    net_policy_act(net, &act, pk, n, lg);
    float mx = lg[0];
    for (int i = 1; i < n; i++) if (lg[i] > mx) mx = lg[i];
    float sum = 0.0f;
    for (int i = 0; i < n; i++) { prob[i] = expf(lg[i] - mx); sum += prob[i]; }
    float inv = 1.0f / sum;
    for (int i = 0; i < n; i++) prob[i] *= inv;
    return n;
}

int sample_index(const float *w, int n, Rng *rng)
{
    if (n <= 0) return -1;

    /* Infinite weights dominate finite ones.  Choose uniformly among them
     * rather than allowing an inf/inf normalization to produce NaNs. */
    int ninf = 0;
    for (int i = 0; i < n; i++)
        if (isinf(w[i]) && w[i] > 0.0f) ninf++;
    if (ninf > 0) {
        int pick = (int)rng_below(rng, (uint32_t)ninf);
        for (int i = 0; i < n; i++) {
            if (isinf(w[i]) && w[i] > 0.0f && pick-- == 0) return i;
        }
    }

    /* Scaling by the largest usable weight prevents both overflow in the sum
     * and a positive distribution degenerating through underflow. */
    float scale = 0.0f;
    for (int i = 0; i < n; i++)
        if (isfinite(w[i]) && w[i] > scale) scale = w[i];
    if (scale > 0.0f) {
        double sum = 0.0;
        for (int i = 0; i < n; i++)
            if (isfinite(w[i]) && w[i] > 0.0f) sum += (double)w[i] / scale;

        double r = (double)rng_float(rng) * sum;
        int last = -1;
        for (int i = 0; i < n; i++) {
            if (!isfinite(w[i]) || w[i] <= 0.0f) continue;
            last = i;
            double wi = (double)w[i] / scale;
            if (r < wi) return i;  /* strict comparison: r == 0 skips zeroes */
            r -= wi;
        }
        if (last >= 0) return last; /* rounding fallback remains positive */
    }

    /* There is no valid weighted choice (all zero, negative, or NaN). */
    return (int)rng_below(rng, (uint32_t)n);
}

int agent_move_values(const Agent *a, const State *st, Rng *rng, Move *mv, float *val)
{
    int n = lc_moves(st, mv);
    if (a->kind == AG_RANDOM) {
        for (int i = 0; i < n; i++) val[i] = 0.0f;
        return n;
    }
    DrawSamples ds;
    draw_samples_init(st, st->turn, rng, a->draw_samples, &ds);
    for (int i = 0; i < n; i++) {
        if (a->kind == AG_HEUR) val[i] = move_value_heur(st, mv[i], &ds);
        else                    val[i] = move_value_net(a->net, st, mv[i], &ds);
    }
    return n;
}

static Move pick_from_values(const Agent *a, Move *mv, float *val, int n, Rng *rng)
{
    if (a->eps > 0.0f && rng_float(rng) < a->eps) return mv[rng_below(rng, (uint32_t)n)];
    if (a->temp > 0.0f) {
        float best = -1e30f;
        for (int i = 0; i < n; i++) if (val[i] > best) best = val[i];
        float sum = 0.0f, w[MAX_MOVES];
        for (int i = 0; i < n; i++) { w[i] = expf((val[i] - best) / a->temp); sum += w[i]; }
        float r = rng_float(rng) * sum;
        for (int i = 0; i < n; i++) { r -= w[i]; if (r <= 0.0f) return mv[i]; }
        return mv[n - 1];
    }
    float best = -1e30f;
    int nbest = 0;
    Move bm = mv[0];
    for (int i = 0; i < n; i++) {
        if (val[i] > best + 1e-6f) { best = val[i]; bm = mv[i]; nbest = 1; }
        else if (val[i] > best - 1e-6f) { nbest++; if (rng_below(rng, (uint32_t)nbest) == 0) bm = mv[i]; }
    }
    return bm;
}

Move agent_move(const Agent *a, const State *st, Rng *rng)
{
    Move mv[MAX_MOVES];
    float val[MAX_MOVES];
    if (a->kind == AG_MCTS) return search_move(a, st, rng, NULL, NULL);
    if (a->kind == AG_ROLLOUT) return rollout_move(a, st, rng, NULL, NULL);
    if (a->kind == AG_POLICY) {
        float prob[MAX_MOVES];
        int n = policy_probs(a->net, st, mv, prob, NULL);
        if (a->eps > 0.0f && rng_float(rng) < a->eps) return mv[rng_below(rng, (uint32_t)n)];
        if (a->temp > 0.0f) {
            if (a->temp != 1.0f)
                for (int i = 0; i < n; i++) prob[i] = powf(prob[i], 1.0f / a->temp);
            return mv[sample_index(prob, n, rng)];
        }
        int best = 0;
        for (int i = 1; i < n; i++) if (prob[i] > prob[best]) best = i;
        return mv[best];
    }
    int n = agent_move_values(a, st, rng, mv, val);
    if (a->kind == AG_RANDOM) return mv[rng_below(rng, (uint32_t)n)];
    return pick_from_values(a, mv, val, n, rng);
}
