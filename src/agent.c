#include "agent.h"
#include "heuristic.h"
#include "planner.h"
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
    a->symmetries = 1;
    a->dets = 16;
    a->sims = 160;
    a->root_width = 14;
    a->node_width = 8;
    a->cpuct = 1.4f;
    a->cand_floor = 0.02f;
    a->cand_mass = 0.0f;
    a->min_cand = 1;
    a->ply_lo = 0;
    a->ply_hi = 0;
    a->eval_cand = 0;
    a->batch_dets = 0;
    a->playout_symmetries = 1;
    a->confirm_dets = 256;
    a->playout_prune = -1;
    a->win_q = 0;
    /* This is an optional search-focus heuristic, not true dominance:
     * changing which discard pile is covered can change later play.  Keep it
     * opt-in until exhaustive endgame tests can establish safe conditions. */
    a->prune_dom = 0;
    a->override_k = 0.0f;
    a->override_min = 4.0f;
    a->playout_sample = 0;
    a->confirm_exact5 = 0;
    a->draw_variant_cores = 0;
    a->draw_variant_deck_max = 0;
    a->policy_prefix_mode = 0;
    a->belief_alpha = 1.0f;
    a->draw_root_deck_max = 0;
    a->draw_playout_deck_max = 0;
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

/* Average belief logits under the same exact suit groups used by the policy
 * ensemble, mapping every card score back to the original state. */
static void belief_logits_sym(const Net *net, const State *st, int p,
                              const uint8_t *card, int n, int symmetries,
                              float *logit)
{
    for (int i = 0; i < n; i++) logit[i] = 0.0f;
    uint8_t perms[120][NSUIT];
    int nsym = suit_permutations(symmetries, perms);
    for (int k = 0; k < nsym; k++) {
        State ps;
        lc_permute_suits(st, &ps, perms[k]);
        Features f;
        feat_extract(&ps, p, &f);
        NetAct act;
        net_trunk(net, &f, &act);
        uint8_t mapped[NCARD];
        float plogit[NCARD];
        for (int i = 0; i < n; i++)
            mapped[i] = lc_permute_card(card[i], perms[k]);
        net_belief_act(net, &act, mapped, n, plogit);
        for (int i = 0; i < n; i++) logit[i] += plogit[i];
    }
    float inv = 1.0f / (float)nsym;
    for (int i = 0; i < n; i++) logit[i] *= inv;
}

int belief_dist_init(const Net *net, const State *st, int p, int symmetries,
                     float alpha, BeliefDist *dist)
{
    memset(dist, 0, sizeof *dist);
    lc_unseen(st, p, dist->card, &dist->n);
    const int o = p ^ 1;
    dist->need = (int)st->hand_n[o] - __builtin_popcountll(st->known[o]);
    if (dist->need < 0 || dist->need > HAND_SIZE || dist->need > dist->n)
        return 0;

    /* Before the opponent has acted there is no behavioural evidence.  Make
     * the exact card-count prior a hard invariant instead of allowing a suit
     * slot artefact to masquerade as information. */
    if (!net || alpha <= 0.0f || st->nply == 0) {
        for (int i = 0; i < dist->n; i++) dist->weight[i] = 1.0;
    } else {
        float logit[NCARD];
        belief_logits_sym(net, st, p, dist->card, dist->n, symmetries, logit);
        double mean = 0.0;
        for (int i = 0; i < dist->n; i++) mean += logit[i];
        if (dist->n > 0) mean /= dist->n;
        for (int i = 0; i < dist->n; i++) {
            double u = (double)alpha * ((double)logit[i] - mean);
            if (u > 20.0) u = 20.0;
            if (u < -20.0) u = -20.0;
            dist->weight[i] = exp(u);
        }
    }

    /* Elementary-symmetric-polynomial DP.  Conditioning independent odds on
     * exactly K held cards yields a coherent joint hand distribution:
     * P(S | |S|=K) is proportional to the product of its card weights. */
    dist->suffix[dist->n][0] = 1.0;
    for (int i = dist->n - 1; i >= 0; i--) {
        dist->suffix[i][0] = 1.0;
        for (int r = 1; r <= dist->need; r++)
            dist->suffix[i][r] = dist->suffix[i + 1][r]
                               + dist->weight[i] * dist->suffix[i + 1][r - 1];
    }
    double z = dist->suffix[0][dist->need];
    if (!(z > 0.0) || !lc_double_isfinite(z)) return 0;

    double prefix[NCARD + 1][HAND_SIZE + 1];
    memset(prefix, 0, sizeof prefix);
    prefix[0][0] = 1.0;
    for (int i = 0; i < dist->n; i++) {
        double include = 0.0;
        for (int r = 0; r < dist->need; r++)
            include += prefix[i][r]
                     * dist->suffix[i + 1][dist->need - 1 - r];
        dist->marginal[i] = (float)(dist->weight[i] * include / z);
        prefix[i + 1][0] = 1.0;
        for (int r = 1; r <= dist->need; r++)
            prefix[i + 1][r] = prefix[i][r]
                             + dist->weight[i] * prefix[i][r - 1];
    }
    return 1;
}

void belief_dist_sample(const State *st, int p, Rng *rng,
                        const BeliefDist *dist, State *out)
{
    *out = *st;
    const int o = p ^ 1;
    out->hand[o] = st->known[o];
    uint8_t selected[NCARD] = { 0 };
    int need = dist->need;
    for (int i = 0; i < dist->n; i++) {
        int remaining = dist->n - i;
        int take = 0;
        if (need == remaining) {
            take = 1;
        } else if (need > 0) {
            double den = dist->suffix[i][need];
            double num = dist->weight[i] * dist->suffix[i + 1][need - 1];
            double probability = den > 0.0 ? num / den : 0.0;
            take = (double)rng_float(rng) < probability;
        }
        if (take) {
            selected[i] = 1;
            out->hand[o] |= 1ULL << dist->card[i];
            need--;
        }
    }

    /* Every unselected unseen card is in the deck; its order remains uniform. */
    out->deck_pos = 0;
    memset(out->deck, 0, sizeof(out->deck));
    int d = 0;
    for (int i = 0; i < dist->n; i++)
        if (!selected[i]) out->deck[d++] = dist->card[i];
    for (int i = d - 1; i > 0; i--) {
        uint32_t j = rng_below(rng, (uint32_t)i + 1);
        uint8_t t = out->deck[i]; out->deck[i] = out->deck[j]; out->deck[j] = t;
    }
    out->deck_left = (uint8_t)d;
}

void determinize_b(const State *st, int p, Rng *rng, const Net *net, State *out)
{
    if (!net) { determinize(st, p, rng, out); return; }
    BeliefDist dist;
    if (!belief_dist_init(net, st, p, 1, 1.0f, &dist)) {
        determinize(st, p, rng, out);
        return;
    }
    belief_dist_sample(st, p, rng, &dist, out);
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

static int policy_probs_raw(const Net *net, const State *st, Move *mv,
                            float *prob, float *value)
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

/* Return an exact subgroup of the 120 suit permutations.  The 20-element
 * affine group s -> a*s+b (mod 5) is 2-transitive, so every ordered pair of
 * suits visits every pair of network slots equally often at one sixth of the
 * cost of the full group. */
int suit_permutations(int requested, uint8_t out[120][NSUIT])
{
    if (requested != 5 && requested != 10 &&
        requested != 20 && requested != 120) {
        for (int s = 0; s < NSUIT; s++) out[0][s] = (uint8_t)s;
        return 1;
    }
    if (requested != 120) {
        static const int mult5[4] = { 1, 4, 2, 3 };
        int na = requested == 5 ? 1 : (requested == 10 ? 2 : 4);
        int n = 0;
        for (int ai = 0; ai < na; ai++)
            for (int b = 0; b < NSUIT; b++) {
                for (int s = 0; s < NSUIT; s++)
                    out[n][s] = (uint8_t)((mult5[ai] * s + b) % NSUIT);
                n++;
            }
        return n;
    }

    int n = 0;
    for (int a = 0; a < NSUIT; a++)
    for (int b = 0; b < NSUIT; b++) if (b != a)
    for (int c = 0; c < NSUIT; c++) if (c != a && c != b)
    for (int d = 0; d < NSUIT; d++) if (d != a && d != b && d != c)
    for (int e = 0; e < NSUIT; e++) if (e != a && e != b && e != c && e != d) {
        out[n][0] = (uint8_t)a; out[n][1] = (uint8_t)b;
        out[n][2] = (uint8_t)c; out[n][3] = (uint8_t)d;
        out[n][4] = (uint8_t)e;
        n++;
    }
    return n;
}

int policy_probs_sym(const Net *net, const State *st, Move *mv, float *prob,
                     float *value, int symmetries)
{
    if (symmetries <= 1)
        return policy_probs_raw(net, st, mv, prob, value);

    int n = lc_moves(st, mv);
    if (n == 0) return 0;
    for (int i = 0; i < n; i++) prob[i] = 0.0f;
    float value_sum = 0.0f;

    uint8_t perms[120][NSUIT];
    int nsym = suit_permutations(symmetries, perms);
    for (int k = 0; k < nsym; k++) {
        State ps;
        lc_permute_suits(st, &ps, perms[k]);
        Move pmv[MAX_MOVES];
        float pp[MAX_MOVES], pv = 0.0f;
        int pn = policy_probs_raw(net, &ps, pmv, pp, value ? &pv : NULL);

        int by_pack[MOVE_NPACK];
        for (int i = 0; i < MOVE_NPACK; i++) by_pack[i] = -1;
        for (int i = 0; i < pn; i++) by_pack[MOVE_PACK(pmv[i])] = i;
        for (int i = 0; i < n; i++) {
            Move mapped = lc_permute_move(mv[i], perms[k]);
            int j = by_pack[MOVE_PACK(mapped)];
            if (j >= 0) prob[i] += pp[j];
        }
        value_sum += pv;
    }

    float inv = 1.0f / (float)nsym;
    for (int i = 0; i < n; i++) prob[i] *= inv;
    if (value) *value = value_sum * inv;
    return n;
}

int policy_probs_perm(const Net *net, const State *st, Move *mv, float *prob,
                      float *value, const uint8_t perm[NSUIT])
{
    State ps;
    lc_permute_suits(st, &ps, perm);
    Move pmv[MAX_MOVES];
    float pp[MAX_MOVES];
    float pv = 0.0f;
    int pn = policy_probs_raw(net, &ps, pmv, pp, value ? &pv : NULL);

    int n = lc_moves(st, mv);
    int by_pack[MOVE_NPACK];
    for (int i = 0; i < MOVE_NPACK; i++) by_pack[i] = -1;
    for (int i = 0; i < pn; i++) by_pack[MOVE_PACK(pmv[i])] = i;
    for (int i = 0; i < n; i++) {
        Move mapped = lc_permute_move(mv[i], perm);
        int j = by_pack[MOVE_PACK(mapped)];
        prob[i] = j >= 0 ? pp[j] : 0.0f;
    }
    if (value) *value = pv;
    return n;
}

int policy_probs_random_sym(const Net *net, const State *st, Move *mv,
                            float *prob, Rng *rng, int symmetries)
{
    uint8_t perms[120][NSUIT];
    int nsym = suit_permutations(symmetries, perms);
    if (nsym <= 1)
        return policy_probs_raw(net, st, mv, prob, NULL);
    int k = (int)rng_below(rng, (uint32_t)nsym);
    return policy_probs_perm(net, st, mv, prob, NULL, perms[k]);
}

int policy_probs(const Net *net, const State *st, Move *mv, float *prob,
                 float *value)
{
    return policy_probs_raw(net, st, mv, prob, value);
}

int sample_index(const float *w, int n, Rng *rng)
{
    if (n <= 0) return -1;

    /* Infinite weights dominate finite ones.  Choose uniformly among them
     * rather than allowing an inf/inf normalization to produce NaNs. */
    int ninf = 0;
    for (int i = 0; i < n; i++)
        if (lc_float_is_pos_inf(w[i])) ninf++;
    if (ninf > 0) {
        int pick = (int)rng_below(rng, (uint32_t)ninf);
        for (int i = 0; i < n; i++) {
            if (lc_float_is_pos_inf(w[i]) && pick-- == 0) return i;
        }
    }

    /* Scaling by the largest usable weight prevents both overflow in the sum
     * and a positive distribution degenerating through underflow. */
    float scale = 0.0f;
    for (int i = 0; i < n; i++)
        if (lc_float_isfinite(w[i]) && w[i] > scale) scale = w[i];
    if (scale > 0.0f) {
        double sum = 0.0;
        for (int i = 0; i < n; i++)
            if (lc_float_isfinite(w[i]) && w[i] > 0.0f)
                sum += (double)w[i] / scale;

        double r = (double)rng_float(rng) * sum;
        int last = -1;
        for (int i = 0; i < n; i++) {
            if (!lc_float_isfinite(w[i]) || w[i] <= 0.0f) continue;
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
        int n = policy_probs_sym(a->net, st, mv, prob, NULL,
                                 a->symmetries);
        if (a->eps > 0.0f && rng_float(rng) < a->eps) return mv[rng_below(rng, (uint32_t)n)];
        if (a->temp > 0.0f) {
            if (a->temp != 1.0f)
                for (int i = 0; i < n; i++) prob[i] = powf(prob[i], 1.0f / a->temp);
            return mv[sample_index(prob, n, rng)];
        }
        int best = 0;
        for (int i = 1; i < n; i++) if (prob[i] > prob[best]) best = i;
        if (a->plan_deck_max > 0 && a->plan_block_gap > 0 &&
            st->deck_left <= a->plan_deck_max) {
            int order[MAX_MOVES];
            for (int i = 0; i < n; i++) order[i] = i;
            int keep = n < 8 ? n : 8;
            for (int i = 0; i < keep; i++) {
                int top = i;
                for (int j = i + 1; j < n; j++)
                    if (prob[order[j]] > prob[order[top]]) top = j;
                int tmp = order[i]; order[i] = order[top]; order[top] = tmp;
            }
            int planned = hand_plan_conservative_choose(
                st, st->turn, mv, prob, order, keep,
                (st->deck_left + 1) / 2, a->plan_block_gap);
            if (planned >= 0) best = planned;
        }
        if (a->draw_root_deck_max > 0 &&
            st->deck_left <= a->draw_root_deck_max)
            best = hand_plan_choose_draw_source(
                st, st->turn, mv, prob, n, best);
        return mv[best];
    }
    int n = agent_move_values(a, st, rng, mv, val);
    if (a->kind == AG_RANDOM) return mv[rng_below(rng, (uint32_t)n)];
    return pick_from_values(a, mv, val, n, rng);
}
