/* belief_eval -- deterministic held-out evaluation of the exact-K belief head.
 *
 * A complete referee State is retained solely to apply real draws and label
 * the opponent's current hand.  The frozen exact-policy actor is loaded from
 * a checkpoint separate from the belief candidate, and both receive
 * agent_information_view(), which erases that hand and the future deck order
 * before any network call.  Beliefs therefore cannot influence the trajectory,
 * candidate policy changes cannot change the scored states, and truth cannot
 * influence either network input.
 */
#include "../src/agent.h"
#include "../src/lc.h"
#include "../src/net.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define DEFAULT_SEED UINT64_C(202608030917)

typedef struct {
    long states;
    long cards;
    long positives;
    long ranking_states;
    long ranking_positives;
    double nll;
    double prior_nll;
    double brier;
    double prior_brier;
    double auc_state_sum;
    double auc_points;
    double auc_pairs;
    double top_hits;
    double prior_top_hits;
} Metrics;

enum { M_NLL_STATE, M_NLL_CARD, M_BRIER, M_AUC_STATE,
       M_AUC_WEIGHTED, M_TOP_RECALL, M_COUNT };

typedef struct {
    long n;
    double sum_w[M_COUNT], sum_w2[M_COUNT];
    double exact_x2[M_COUNT], exact_xw[M_COUNT];
    double delta_x2[M_COUNT], delta_xw[M_COUNT];
} ClusterStats;

static uint64_t mix64(uint64_t x)
{
    x += UINT64_C(0x9E3779B97F4A7C15);
    x = (x ^ (x >> 30)) * UINT64_C(0xBF58476D1CE4E5B9);
    x = (x ^ (x >> 27)) * UINT64_C(0x94D049BB133111EB);
    return x ^ (x >> 31);
}

static uint64_t model_fingerprint(const Net *net)
{
    const unsigned char *p = (const unsigned char *)net;
    uint64_t h = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < sizeof *net; i++) {
        h ^= p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

static int valid_symmetries(int n)
{
    return n == 1 || n == 5 || n == 10 || n == 20 || n == 120;
}

static int parse_long(const char *text, long lo, long hi, long *out)
{
    if (!text || !*text) return 0;
    for (const char *p = text; *p; p++)
        if (*p < '0' || *p > '9') return 0;
    char *end = NULL;
    errno = 0;
    long v = strtol(text, &end, 10);
    if (errno || !end || *end || v < lo || v > hi) return 0;
    *out = v;
    return 1;
}

static int parse_seed(const char *text, uint64_t *out)
{
    if (!text || !*text) return 0;
    for (const char *p = text; *p; p++)
        if (*p < '0' || *p > '9') return 0;
    char *end = NULL;
    errno = 0;
    unsigned long long v = strtoull(text, &end, 10);
    if (errno || !end || *end) return 0;
    *out = (uint64_t)v;
    return 1;
}

static int parse_alpha(const char *text, float *out)
{
    if (!text || !*text) return 0;
    if (!((*text >= '0' && *text <= '9') || *text == '.' ||
          *text == '+' || *text == '-'))
        return 0;
    char *end = NULL;
    errno = 0;
    double v = strtod(text, &end);
    if (errno || !end || end == text || *end || !lc_double_isfinite(v) ||
        v < 0.0 ||
        v > 5.0)
        return 0;
    *out = (float)v;
    return 1;
}

static void usage(const char *argv0)
{
    fprintf(stderr,
            "usage: %s [--net PATH] [--games N] [--rounds 1..3] "
            "[--seed N]\n"
            "          [--actor-net PATH]\n"
            "          [--symmetries 1|5|10|20|120] [--alpha 0..5]\n"
            "          [--min-ply N] [--max-ply N] [--json]\n"
            "\n"
            "Defaults: evaluate data/champion.bin under a frozen "
            "data/champion.bin actor,\n"
            "20 three-round matches, seed "
            "%llu,\n"
            "20-way exact symmetry, alpha 1.15, and post-opening states.\n",
            argv0, (unsigned long long)DEFAULT_SEED);
}

static int frozen_policy_move(const Net *net, const State *complete,
                              int symmetries, Move *chosen)
{
    State view;
    agent_information_view(complete, complete->turn, &view);
    Move moves[MAX_MOVES];
    float prob[MAX_MOVES];
    int n = policy_probs_sym(net, &view, moves, prob, NULL, symmetries);
    if (n <= 0) return 0;
    int best = 0;
    for (int i = 1; i < n; i++)
        if (prob[i] > prob[best]) best = i;
    *chosen = moves[best];
    return 1;
}

static double log_choose(int n, int k)
{
    return lgamma((double)n + 1.0) - lgamma((double)k + 1.0)
         - lgamma((double)(n - k) + 1.0);
}

static void metrics_add(Metrics *dst, const Metrics *src)
{
    dst->states += src->states;
    dst->cards += src->cards;
    dst->positives += src->positives;
    dst->ranking_states += src->ranking_states;
    dst->ranking_positives += src->ranking_positives;
    dst->nll += src->nll;
    dst->prior_nll += src->prior_nll;
    dst->brier += src->brier;
    dst->prior_brier += src->prior_brier;
    dst->auc_state_sum += src->auc_state_sum;
    dst->auc_points += src->auc_points;
    dst->auc_pairs += src->auc_pairs;
    dst->top_hits += src->top_hits;
    dst->prior_top_hits += src->prior_top_hits;
}

static int metric_components(const Metrics *m, double exact[M_COUNT],
                             double prior[M_COUNT], double denom[M_COUNT])
{
    if (m->states <= 0 || m->cards <= 0 || m->ranking_states <= 0 ||
        m->auc_pairs <= 0.0 || m->ranking_positives <= 0)
        return 0;
    exact[M_NLL_STATE] = m->nll;
    exact[M_NLL_CARD] = m->nll;
    exact[M_BRIER] = m->brier;
    exact[M_AUC_STATE] = m->auc_state_sum;
    exact[M_AUC_WEIGHTED] = m->auc_points;
    exact[M_TOP_RECALL] = m->top_hits;
    prior[M_NLL_STATE] = m->prior_nll;
    prior[M_NLL_CARD] = m->prior_nll;
    prior[M_BRIER] = m->prior_brier;
    prior[M_AUC_STATE] = 0.5 * (double)m->ranking_states;
    prior[M_AUC_WEIGHTED] = 0.5 * m->auc_pairs;
    prior[M_TOP_RECALL] = m->prior_top_hits;
    denom[M_NLL_STATE] = (double)m->states;
    denom[M_NLL_CARD] = (double)m->cards;
    denom[M_BRIER] = (double)m->cards;
    denom[M_AUC_STATE] = (double)m->ranking_states;
    denom[M_AUC_WEIGHTED] = m->auc_pairs;
    denom[M_TOP_RECALL] = (double)m->ranking_positives;
    return 1;
}

static int metric_values(const Metrics *m, double exact[M_COUNT],
                         double prior[M_COUNT])
{
    double denom[M_COUNT];
    if (!metric_components(m, exact, prior, denom)) return 0;
    for (int i = 0; i < M_COUNT; i++) {
        exact[i] /= denom[i];
        prior[i] /= denom[i];
    }
    return 1;
}

static void cluster_add(ClusterStats *s, const Metrics *match)
{
    double exact[M_COUNT], prior[M_COUNT], denom[M_COUNT];
    if (!metric_components(match, exact, prior, denom)) return;
    s->n++;
    for (int i = 0; i < M_COUNT; i++) {
        double x = exact[i], w = denom[i];
        double delta = x - prior[i];
        s->sum_w[i] += w;
        s->sum_w2[i] += w * w;
        s->exact_x2[i] += x * x;
        s->exact_xw[i] += x * w;
        s->delta_x2[i] += delta * delta;
        s->delta_xw[i] += delta * w;
    }
}

/* Finite-cluster sandwich SE for the pooled ratio X/W.  This targets the
 * exact state/card/pair-weighted estimator printed above, unlike an unweighted
 * standard deviation of per-match ratios when trajectories have unequal
 * lengths. */
static double cluster_se(const ClusterStats *s, int metric, double estimate,
                         int delta)
{
    if (s->n <= 1 || !(s->sum_w[metric] > 0.0)) return 0.0;
    double x2 = delta ? s->delta_x2[metric] : s->exact_x2[metric];
    double xw = delta ? s->delta_xw[metric] : s->exact_xw[metric];
    double residual = x2 - 2.0 * estimate * xw
                    + estimate * estimate * s->sum_w2[metric];
    if (residual < 0.0 && residual > -1e-10) residual = 0.0;
    if (residual < 0.0) return 0.0;
    double correction = (double)s->n / (double)(s->n - 1);
    return sqrt(correction * residual) / s->sum_w[metric];
}

/* Expected hits when exactly K highest scores are selected, splitting a
 * boundary tie uniformly.  This makes the all-equal card-count prior's
 * top-K recall exactly K/N instead of depending on card-id ordering. */
static double tie_correct_top_hits(const BeliefDist *dist,
                                   const uint8_t held[NCARD])
{
    int order[NCARD];
    for (int i = 0; i < dist->n; i++) order[i] = i;
    for (int i = 1; i < dist->n; i++) {
        int x = order[i], j = i;
        while (j > 0 && dist->marginal[x] > dist->marginal[order[j - 1]]) {
            order[j] = order[j - 1];
            j--;
        }
        order[j] = x;
    }

    int slots = dist->need;
    double hits = 0.0;
    for (int at = 0; at < dist->n && slots > 0;) {
        int end = at + 1, positives = held[order[at]] ? 1 : 0;
        while (end < dist->n &&
               dist->marginal[order[end]] == dist->marginal[order[at]]) {
            positives += held[order[end]] ? 1 : 0;
            end++;
        }
        int group = end - at;
        int take = slots < group ? slots : group;
        hits += (double)take * (double)positives / (double)group;
        slots -= take;
        at = end;
    }
    return hits;
}

static int score_state(const Net *net, const State *complete,
                       int symmetries, float alpha, Metrics *m)
{
    int p = complete->turn, o = p ^ 1;
    State view;
    agent_information_view(complete, p, &view);
    BeliefDist dist;
    if (!belief_dist_init(net, &view, p, symmetries, alpha, &dist)) return 0;

    uint8_t held[NCARD] = { 0 };
    int held_count = 0;
    double marginal_sum = 0.0;
    for (int i = 0; i < dist.n; i++) {
        held[i] = (uint8_t)((complete->hand[o] >> dist.card[i]) & 1ULL);
        held_count += held[i];
        marginal_sum += dist.marginal[i];
    }
    if (held_count != dist.need ||
        fabs(marginal_sum - (double)dist.need) > 2e-4)
        return 0;

    double nll;
    if (!belief_dist_true_nll(&dist, complete->hand[o], &nll)) return 0;
    double prior = dist.n > 0 ? (double)dist.need / (double)dist.n : 0.0;
    m->states++;
    m->cards += dist.n;
    m->positives += dist.need;
    m->nll += nll;
    m->prior_nll += log_choose(dist.n, dist.need);
    for (int i = 0; i < dist.n; i++) {
        double e = (double)dist.marginal[i] - held[i];
        double pe = prior - held[i];
        m->brier += e * e;
        m->prior_brier += pe * pe;
    }

    if (dist.need > 0 && dist.need < dist.n) {
        double points = 0.0;
        double pairs = (double)dist.need * (double)(dist.n - dist.need);
        for (int i = 0; i < dist.n; i++) if (held[i])
            for (int j = 0; j < dist.n; j++) if (!held[j]) {
                if (dist.marginal[i] > dist.marginal[j]) points += 1.0;
                else if (dist.marginal[i] == dist.marginal[j]) points += 0.5;
            }
        m->ranking_states++;
        m->ranking_positives += dist.need;
        m->auc_state_sum += points / pairs;
        m->auc_points += points;
        m->auc_pairs += pairs;
        m->top_hits += tie_correct_top_hits(&dist, held);
        m->prior_top_hits += (double)dist.need * prior;
    }
    return 1;
}

static void json_string(const char *s)
{
    putchar('"');
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        if (c == '"' || c == '\\') printf("\\%c", c);
        else if (c == '\n') fputs("\\n", stdout);
        else if (c < 0x20) printf("\\u%04x", c);
        else putchar(c);
    }
    putchar('"');
}

int main(int argc, char **argv)
{
    const char *net_path = "data/champion.bin";
    const char *actor_path = "data/champion.bin";
    long games = 20, rounds = MATCH_ROUNDS;
    long symmetries = 20, min_ply = 1, max_ply = 0;
    uint64_t seed = DEFAULT_SEED;
    float alpha = 1.15f;
    int json = 0;

    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "--net") && i + 1 < argc) net_path = argv[++i];
        else if (!strcmp(argv[i], "--actor-net") && i + 1 < argc)
            actor_path = argv[++i];
        else if (!strcmp(argv[i], "--games") && i + 1 < argc) {
            if (!parse_long(argv[++i], 1, 1000000, &games)) goto bad_option;
        } else if (!strcmp(argv[i], "--rounds") && i + 1 < argc) {
            if (!parse_long(argv[++i], 1, MATCH_ROUNDS, &rounds)) goto bad_option;
        } else if (!strcmp(argv[i], "--seed") && i + 1 < argc) {
            if (!parse_seed(argv[++i], &seed)) goto bad_option;
        } else if (!strcmp(argv[i], "--symmetries") && i + 1 < argc) {
            if (!parse_long(argv[++i], 1, 120, &symmetries) ||
                !valid_symmetries((int)symmetries))
                goto bad_option;
        } else if (!strcmp(argv[i], "--alpha") && i + 1 < argc) {
            if (!parse_alpha(argv[++i], &alpha)) goto bad_option;
        } else if (!strcmp(argv[i], "--min-ply") && i + 1 < argc) {
            if (!parse_long(argv[++i], 0, LC_MAX_PLIES, &min_ply))
                goto bad_option;
        } else if (!strcmp(argv[i], "--max-ply") && i + 1 < argc) {
            if (!parse_long(argv[++i], 0, LC_MAX_PLIES, &max_ply))
                goto bad_option;
        } else if (!strcmp(argv[i], "--json")) json = 1;
        else if (!strcmp(argv[i], "--help")) { usage(argv[0]); return 0; }
        else goto bad_option;
    }
    if (max_ply > 0 && max_ply < min_ply) goto bad_option;

    Net *net = (Net *)malloc(sizeof *net);
    Net *actor_net = (Net *)malloc(sizeof *actor_net);
    if (!net || net_load(net, net_path) != 0) {
        fprintf(stderr, "belief_eval: cannot load %s\n", net_path);
        free(net);
        free(actor_net);
        return 1;
    }
    if (!actor_net || net_load(actor_net, actor_path) != 0) {
        fprintf(stderr, "belief_eval: cannot load frozen actor %s\n",
                actor_path);
        free(net);
        free(actor_net);
        return 1;
    }
    uint64_t fingerprint = model_fingerprint(net);
    uint64_t actor_fingerprint = model_fingerprint(actor_net);
    Metrics metric = { 0 };
    ClusterStats clusters = { 0 };
    long completed_rounds = 0;

    for (long g = 0; g < games; g++) {
        Metrics match_metric = { 0 };
        int cum[2] = { 0, 0 };
        for (long r = 0; r < rounds; r++) {
            uint64_t deal_seed = mix64(seed
                ^ (uint64_t)g * UINT64_C(0xD6E8FEB86659FD93)
                ^ (uint64_t)r * UINT64_C(0xA0761D6478BD642F));
            Rng deal_rng;
            rng_seed(&deal_rng, deal_seed);
            State st;
            lc_deal(&st, &deal_rng);
            st.round = (uint8_t)r;
            st.cum[0] = (int16_t)cum[0];
            st.cum[1] = (int16_t)cum[1];
            st.turn = (uint8_t)(r & 1);

            while (!st.over) {
                if (st.nply >= min_ply &&
                    (max_ply == 0 || st.nply <= max_ply) &&
                    !score_state(net, &st, (int)symmetries, alpha,
                                 &match_metric)) {
                    fprintf(stderr,
                            "belief_eval: invalid belief state in match %ld "
                            "round %ld ply %u\n", g, r, st.nply);
                    free(net);
                    free(actor_net);
                    return 1;
                }
                Move chosen;
                if (!frozen_policy_move(actor_net, &st, (int)symmetries,
                                        &chosen)) {
                    fprintf(stderr,
                            "belief_eval: frozen policy failed in match %ld "
                            "round %ld ply %u\n", g, r, st.nply);
                    free(net);
                    free(actor_net);
                    return 1;
                }
                lc_apply(&st, chosen);
            }
            cum[0] += lc_score(&st, 0);
            cum[1] += lc_score(&st, 1);
            completed_rounds++;
        }
        metrics_add(&metric, &match_metric);
        cluster_add(&clusters, &match_metric);
    }

    if (metric.states == 0 || metric.cards == 0 ||
        metric.ranking_states == 0 || metric.positives == 0) {
        fprintf(stderr, "belief_eval: the requested ply window selected no "
                        "rankable states\n");
        free(net);
        free(actor_net);
        return 1;
    }

    double exact[M_COUNT], prior[M_COUNT];
    if (!metric_values(&metric, exact, prior)) {
        fprintf(stderr, "belief_eval: internal metric aggregation failed\n");
        free(net);
        free(actor_net);
        return 1;
    }

    if (json) {
        printf("{\"schema\":\"lc-belief-eval-v1\",\"model\":");
        json_string(net_path);
        printf(",\"model_fingerprint\":\"%016llx\","
               "\"seed\":%llu,\"matches\":%ld,\"rounds_per_match\":%ld,"
               "\"rounds_completed\":%ld,\"symmetries\":%ld,"
               "\"alpha\":%.9g,\"min_ply\":%ld,\"max_ply\":%ld,"
               "\"actor\":{\"type\":\"exact_policy_argmax\","
               "\"model\":",
               (unsigned long long)fingerprint,
               (unsigned long long)seed, games, rounds, completed_rounds,
               symmetries, alpha, min_ply, max_ply);
        json_string(actor_path);
        printf(",\"model_fingerprint\":\"%016llx\","
               "\"uses_belief\":false,\"truth_scrubbed\":true},"
               "\"sample\":{\"states\":%ld,\"ranking_states\":%ld,"
               "\"uncertain_cards\":%ld,\"unknown_held_cards\":%ld,"
               "\"match_clusters\":%ld},"
               "\"exact_k\":{\"nll_per_state\":%.12g,"
               "\"nll_per_uncertain_card\":%.12g,\"brier\":%.12g,"
               "\"auc_within_state\":%.12g,"
               "\"auc_pair_weighted\":%.12g,\"top_k_recall\":%.12g},"
               "\"uniform_card_count_prior\":{\"nll_per_state\":%.12g,"
               "\"nll_per_uncertain_card\":%.12g,\"brier\":%.12g,"
               "\"auc_within_state\":0.5,\"auc_pair_weighted\":0.5,"
               "\"top_k_recall\":%.12g},"
               "\"match_clustered_se\":{\"exact_k\":{"
               "\"nll_per_state\":%.12g,\"nll_per_uncertain_card\":%.12g,"
               "\"brier\":%.12g,\"auc_within_state\":%.12g,"
               "\"auc_pair_weighted\":%.12g,\"top_k_recall\":%.12g},"
               "\"exact_minus_uniform\":{\"nll_per_state\":%.12g,"
               "\"nll_per_uncertain_card\":%.12g,\"brier\":%.12g,"
               "\"auc_within_state\":%.12g,"
               "\"auc_pair_weighted\":%.12g,\"top_k_recall\":%.12g}}}\n",
               (unsigned long long)actor_fingerprint,
               metric.states, metric.ranking_states, metric.cards,
               metric.positives, clusters.n,
               exact[M_NLL_STATE], exact[M_NLL_CARD], exact[M_BRIER],
               exact[M_AUC_STATE], exact[M_AUC_WEIGHTED],
               exact[M_TOP_RECALL], prior[M_NLL_STATE], prior[M_NLL_CARD],
               prior[M_BRIER], prior[M_TOP_RECALL],
               cluster_se(&clusters, M_NLL_STATE,
                          exact[M_NLL_STATE], 0),
               cluster_se(&clusters, M_NLL_CARD,
                          exact[M_NLL_CARD], 0),
               cluster_se(&clusters, M_BRIER, exact[M_BRIER], 0),
               cluster_se(&clusters, M_AUC_STATE,
                          exact[M_AUC_STATE], 0),
               cluster_se(&clusters, M_AUC_WEIGHTED,
                          exact[M_AUC_WEIGHTED], 0),
               cluster_se(&clusters, M_TOP_RECALL,
                          exact[M_TOP_RECALL], 0),
               cluster_se(&clusters, M_NLL_STATE,
                          exact[M_NLL_STATE] - prior[M_NLL_STATE], 1),
               cluster_se(&clusters, M_NLL_CARD,
                          exact[M_NLL_CARD] - prior[M_NLL_CARD], 1),
               cluster_se(&clusters, M_BRIER,
                          exact[M_BRIER] - prior[M_BRIER], 1),
               cluster_se(&clusters, M_AUC_STATE,
                          exact[M_AUC_STATE] - prior[M_AUC_STATE], 1),
               cluster_se(&clusters, M_AUC_WEIGHTED,
                          exact[M_AUC_WEIGHTED] - prior[M_AUC_WEIGHTED], 1),
               cluster_se(&clusters, M_TOP_RECALL,
                          exact[M_TOP_RECALL] - prior[M_TOP_RECALL], 1));
    } else {
        printf("Exact-K held-out belief evaluation\n");
        printf("  model %s  fingerprint %016llx\n", net_path,
               (unsigned long long)fingerprint);
        printf("  frozen exact-policy argmax %s  fingerprint %016llx\n",
               actor_path, (unsigned long long)actor_fingerprint);
        printf("  %ld x %ld rounds; seed %llu\n",
               games, rounds, (unsigned long long)seed);
        printf("  network inputs truth-scrubbed; belief never drives play\n");
        printf("  symmetry %ld  alpha %.3f  ply window %ld..",
               symmetries, alpha, min_ply);
        if (max_ply > 0) printf("%ld\n", max_ply); else printf("end\n");
        printf("  %ld states, %ld uncertain card labels\n",
               metric.states, metric.cards);
        printf("\n"
               "                         exact-K       uniform card-count\n"
               "  joint NLL / state     %10.6f    %10.6f\n"
               "  joint NLL / card      %10.6f    %10.6f\n"
               "  Brier / card          %10.6f    %10.6f\n"
               "  within-state AUC      %10.6f    %10.6f\n"
               "  pair-weighted AUC     %10.6f    %10.6f\n"
               "  tie-correct top-K     %10.6f    %10.6f\n"
               "\n  match-clustered SE (exact-K): NLL/state %.6f, "
               "Brier %.6f, AUC %.6f, top-K %.6f\n",
               exact[M_NLL_STATE], prior[M_NLL_STATE], exact[M_NLL_CARD],
               prior[M_NLL_CARD], exact[M_BRIER], prior[M_BRIER],
               exact[M_AUC_STATE], 0.5, exact[M_AUC_WEIGHTED], 0.5,
               exact[M_TOP_RECALL], prior[M_TOP_RECALL],
               cluster_se(&clusters, M_NLL_STATE,
                          exact[M_NLL_STATE], 0),
               cluster_se(&clusters, M_BRIER, exact[M_BRIER], 0),
               cluster_se(&clusters, M_AUC_STATE,
                          exact[M_AUC_STATE], 0),
               cluster_se(&clusters, M_TOP_RECALL,
                          exact[M_TOP_RECALL], 0));
    }
    free(net);
    free(actor_net);
    return 0;

bad_option:
    usage(argv[0]);
    return 2;
}
