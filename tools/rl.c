/* rl -- self-play policy optimisation (PPO) for the Lost Cities network.
 *
 * Why policy gradient rather than expert iteration: candidate moves in this
 * game differ by one or two points while a finished game's margin swings by
 * sixty, so no value function accurate enough to rank moves by one-ply
 * lookahead is learnable, and a search built on such a value function is no
 * stronger than the policy that seeded it (measured, not assumed).  What does
 * work is improving the policy directly from played outcomes: the value head
 * only has to serve as a baseline, where its errors cancel instead of
 * corrupting the ranking.
 *
 * Ordinary generation uses the same live network in both seats.  An optional
 * frozen-opponent population alternates the learner's seat and records policy
 * gradients only on learner decisions; mixing those games with live self-play
 * breaks self-play blind spots without pretending opponent actions are
 * on-policy.  A full-legal-action KL anchor and v6-only warm-up keep this
 * deliberately conservative around an established champion.  Optional suit
 * augmentation fixes one exact relabelling for a complete match, and an
 * independent belief-only mode can calibrate the exact-K posterior without
 * changing a single trunk, policy, or value parameter.
 */
#include "../src/lc.h"
#include "../src/net.h"
#include "../src/agent.h"
#include "../src/heuristic.h"
#include "../src/match.h"
#include "../src/spec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <stddef.h>
#include <time.h>
#include <pthread.h>

typedef struct {
    State st;
    float vtarget;    /* lambda-return, points, perspective player's view */
    float adv;        /* advantage, points (actor samples only)           */
    float oldp;       /* policy probability of the move actually played   */
    uint16_t chosen;  /* packed move                                      */
    uint8_t persp;
    uint8_t actor;    /* 1 when persp is the player who moved             */
} RLSample;

typedef struct {
    const Net *net;
    const Agent *opponent; /* optional frozen population member             */
    float opponent_mix;    /* fraction of matches played against opponent  */
    int games;          /* matches per iteration */
    uint64_t seed;
    int thread, nthread;
    RLSample *out;
    size_t nout, nseen, cap;
    double plies, absmargin, score_sum, entropy;
    double p0_match_wins;
    double learner_match_wins;
    int opponent_games;
    int learner_seat_games[2];
    uint64_t augmentation_fingerprint;
    long entropy_n;
    int done;
    float lambda;
    float temp;
    float winbonus;     /* terminal reward for winning the match, in points */
    float mw;           /* weight of the margin term in the return          */
    int rounds;
    int trajectory_symmetries; /* 0 off; otherwise one fixed group member */
} GenJob;

static int16_t checked_cumulative_score(int score)
{
    if (score < INT16_MIN || score > INT16_MAX) {
        fprintf(stderr, "cumulative score %d does not fit in State.cum\n", score);
        exit(EXIT_FAILURE);
    }
    return (int16_t)score;
}

#define CHAIN_MAX (MATCH_ROUNDS * LC_MAX_PLIES + 4)
static _Thread_local State chain[CHAIN_MAX];
static _Thread_local float chain_v[2][CHAIN_MAX];
static _Thread_local uint16_t chain_mv[CHAIN_MAX];
static _Thread_local float chain_p[CHAIN_MAX];
static _Thread_local uint8_t chain_actor[CHAIN_MAX];

static uint64_t mix64(uint64_t x)
{
    x ^= x >> 30;
    x *= UINT64_C(0xBF58476D1CE4E5B9);
    x ^= x >> 27;
    x *= UINT64_C(0x94D049BB133111EB);
    return x ^ (x >> 31);
}

/* Keep a uniform sample of arbitrarily long games instead of silently
 * retaining only their earliest plies when a fixed worker buffer fills. */
static size_t reservoir_slot(GenJob *j, Rng *rng)
{
    size_t seen = ++j->nseen;
    if (j->nout < j->cap) return j->nout++;
    if (j->cap == 0) return SIZE_MAX;
    uint64_t pick = (uint64_t)(((__uint128_t)rng_next(rng) * seen) >> 64);
    return pick < j->cap ? (size_t)pick : SIZE_MAX;
}

static void *gen_worker(void *arg)
{
    GenJob *j = (GenJob *)arg;
    Rng rng; rng_seed(&rng, j->seed + 0x9E3779B9ULL * (uint64_t)(j->thread + 1));
    Rng reservoir_rng;
    rng_seed(&reservoir_rng, j->seed ^ (0xD1B54A32D192ED03ULL
                                     * (uint64_t)(j->thread + 1)));
    Rng opponent_rng;
    rng_seed(&opponent_rng, j->seed ^ (0x8CB92BA72F3D8DD7ULL
                                    * (uint64_t)(j->thread + 1)));
    Move mv[MAX_MOVES], engine_mv[MAX_MOVES];
    float pr[MAX_MOVES], behavior[MAX_MOVES];

    for (int g = j->thread; g < j->games; g += j->nthread) {
        /* Select frozen-opponent games in adjacent pairs.  Games 2k and
         * 2k+1 use opposite learner seats, so every complete selected pair
         * balances seat and starter exposure exactly. */
        uint64_t selector = mix64(j->seed ^ ((uint64_t)(g / 2)
                                * UINT64_C(0x9E3779B97F4A7C15)));
        double unit = (double)(selector >> 11) * (1.0 / 9007199254740992.0);
        int versus_opponent = j->opponent && unit < j->opponent_mix;
        int learner_seat = g & 1;

        /* A relabelling is fixed for the entire generated trajectory.  The
         * two legs of a frozen-opponent pair deliberately share both their
         * deal and relabelling so augmentation does not add noise to the
         * mirrored comparison.  Live self-play matches remain independent. */
        uint64_t augmentation_trajectory = versus_opponent
                                         ? (uint64_t)(g / 2)
                                         : (uint64_t)g;
        uint8_t trajectory_perm[NSUIT];
        if (!trajectory_suit_permutation(j->trajectory_symmetries, j->seed,
                                         augmentation_trajectory,
                                         trajectory_perm)) {
            fprintf(stderr, "invalid trajectory suit group\n");
            exit(EXIT_FAILURE);
        }
        int augment = j->trajectory_symmetries > 0;
        if (augment) {
            uint64_t code = 0;
            for (int s = 0; s < NSUIT; s++)
                code = code * NSUIT + trajectory_perm[s];
            j->augmentation_fingerprint ^=
                mix64((uint64_t)g * UINT64_C(0xD6E8FEB86659FD93)
                      ^ code ^ UINT64_C(0x8EBC6AF09C88C6E3));
        }
        if (versus_opponent) {
            j->opponent_games++;
            j->learner_seat_games[learner_seat]++;
        }
        /* Frozen-opponent games are true mirrored pairs: both learner seats
         * see the same three deals.  Gameplay randomness remains separate,
         * so only deal luck is cancelled rather than coupling behavior. */
        Rng pair_deal_rng;
        if (versus_opponent)
            rng_seed(&pair_deal_rng,
                     mix64(j->seed ^ ((uint64_t)(g / 2) *
                           UINT64_C(0xD1B54A32D192ED03))));
        /* one episode = one full match of j->rounds rounds */
        int T = 0;
        int cum[2] = { 0, 0 };
        for (int rd = 0; rd < j->rounds; rd++) {
        State st;
        lc_deal(&st, versus_opponent ? &pair_deal_rng : &rng);
        st.round = (uint8_t)rd;
        /* Preserve the exact match context used by src/match.c.  Clamping
         * the two totals independently changes the lead. */
        st.cum[0] = checked_cumulative_score(cum[0]);
        st.cum[1] = checked_cumulative_score(cum[1]);
        st.turn = (uint8_t)(rd & 1);
        int Tstop = T + LC_MAX_PLIES;
        while (!st.over && T < Tstop) {
            if (augment) lc_permute_suits(&st, &chain[T], trajectory_perm);
            else chain[T] = st;
            Move played = { 0, 0, 0 }, stored_played = { 0, 0, 0 };
            if (versus_opponent && st.turn != learner_seat) {
                played = agent_move(j->opponent, &st, &opponent_rng);
                stored_played = augment
                              ? lc_permute_move(played, trajectory_perm)
                              : played;
                chain_actor[T] = 0;
                chain_p[T] = 1.0f;
            } else {
                int n;
                if (augment) {
                    n = trajectory_policy_probs(
                        j->net, &st, trajectory_perm, j->temp,
                        &chain[T], mv, engine_mv, pr, behavior);
                    if (n <= 0) {
                        fprintf(stderr,
                                "trajectory policy produced no legal move\n");
                        exit(EXIT_FAILURE);
                    }
                } else {
                    n = policy_probs(j->net, &st, mv, pr, NULL);
                }
                double h = 0.0;
                for (int i = 0; i < n; i++)
                    if (pr[i] > 1e-9f) h -= pr[i] * log(pr[i]);
                j->entropy += h;
                j->entropy_n++;

                int c;
                if (augment) {
                    c = sample_index(behavior, n, &rng);
                    chain_p[T] = behavior[c];
                    stored_played = mv[c];
                    played = engine_mv[c];
                } else {
                    if (j->temp != 1.0f) {
                        /* Sampling off-policy is fine as long as the recorded
                         * probability is the behaviour policy's, since that
                         * is what the PPO ratio divides by. */
                        float w[MAX_MOVES], sum = 0.0f;
                        for (int i = 0; i < n; i++) {
                            w[i] = powf(pr[i], 1.0f / j->temp);
                            sum += w[i];
                        }
                        c = sample_index(w, n, &rng);
                        chain_p[T] = w[c] / sum;
                    } else {
                        c = sample_index(pr, n, &rng);
                        chain_p[T] = pr[c];
                    }
                    stored_played = mv[c];
                    played = mv[c];
                }
                chain_actor[T] = 1;
            }
            chain_mv[T] = MOVE_PACK(stored_played);
            T++;
            lc_apply(&st, played);
        }
        cum[0] += lc_score(&st, 0);
        cum[1] += lc_score(&st, 1);
        j->plies += st.nply;
        }   /* rounds */

        int score[2] = { cum[0], cum[1] };
        if (score[0] > score[1]) j->p0_match_wins += 1.0;
        else if (score[0] == score[1]) j->p0_match_wins += 0.5;
        if (versus_opponent) {
            if (score[learner_seat] > score[learner_seat ^ 1])
                j->learner_match_wins += 1.0;
            else if (score[learner_seat] == score[learner_seat ^ 1])
                j->learner_match_wins += 0.5;
        }

        /* Self-play stores the complete state, so use a centralized zero-sum
         * critic for bootstraps.  The antisymmetric projection removes the
         * large common bias of two independently evaluated perspectives and
         * guarantees that both lambda-return chains stay exact negatives. */
        for (int t = 0; t < T; t++) {
            float v0 = net_value_state_sym(j->net, &chain[t], 0, 1);
            float v1 = net_value_state_sym(j->net, &chain[t], 1, 1);
            chain_v[0][t] = 0.5f * (v0 - v1);
            chain_v[1][t] = -chain_v[0][t];
        }

        for (int p = 0; p < 2; p++) {
            /* terminal return: mw * match margin + winbonus * result.
             * Early training runs with mw = 1 so the dense margin signal
             * teaches point play; the finishing phase drops mw to ~0.05 so
             * winning is nearly all that matters -- a 5% chance to steal the
             * match beats a certain narrow loss, exactly as it should. */
            float G = (float)(score[p] - score[p ^ 1]) * j->mw;
            if (score[p] > score[p ^ 1]) G += j->winbonus;
            else if (score[p] < score[p ^ 1]) G -= j->winbonus;
            for (int t = T - 1; t >= 0; t--) {
                if (t < T - 1) G = (1.0f - j->lambda) * chain_v[p][t + 1] + j->lambda * G;
                size_t slot = reservoir_slot(j, &reservoir_rng);
                if (slot == SIZE_MAX) continue;
                RLSample *s = &j->out[slot];
                s->st = chain[t];
                s->persp = (uint8_t)p;
                s->vtarget = G;
                if (chain[t].turn == p && chain_actor[t]) {
                    s->actor = 1;
                    s->chosen = chain_mv[t];
                    s->oldp = chain_p[t];
                    s->adv = G - chain_v[p][t];
                } else {
                    s->actor = 0;
                    s->chosen = 0;
                    s->oldp = 1.0f;
                    s->adv = 0.0f;
                }
            }
        }
        j->absmargin += fabs((double)(score[0] - score[1]));
        j->score_sum += score[0] + score[1];
        j->done++;
    }
    return NULL;
}

/* ---------------- optimisation ---------------------------------------- */

typedef struct {
    const Net *net;
    const Net *anchor;
    Net *grad;
    const RLSample *buf;
    const int *idx;
    int from, to;
    float clip, vcoef, entcoef, policy_scale, bw, temp, klcoef;
    int belief_only;
    double ploss, vloss, bloss, klloss, clipped;
    int pn;
    long bn;
} OptJob;

static void *opt_worker(void *arg)
{
    OptJob *t = (OptJob *)arg;
    net_zero(t->grad);
    double ploss = 0, vloss = 0, bloss = 0, klloss = 0, clipped = 0;
    int pn = 0;
    long bn = 0;
    Features f, of;
    NetAct act, oact;
    Move mv[MAX_MOVES];
    uint16_t pk[MAX_MOVES];
    float logit[MAX_MOVES], prob[MAX_MOVES], rawprob[MAX_MOVES];
    float dlog[MAX_MOVES], alogit[MAX_MOVES], aprob[MAX_MOVES];
    uint8_t bcard[NCARD], held[NCARD];
    float blogit[NCARD], bmarg[NCARD], dbel[NCARD];

    for (int i = t->from; i < t->to; i++) {
        const RLSample *s = &t->buf[t->idx[i]];
        feat_extract(&s->st, s->persp, &f);
        net_trunk(t->net, &f, &act);
        float dcenter = 0.0f;
        if (!t->belief_only) {
            feat_extract(&s->st, s->persp ^ 1, &of);
            net_trunk(t->net, &of, &oact);

            float vp = net_value_act(t->net, &act);
            float vo = net_value_act(t->net, &oact);
            float v = 0.5f * (vp - vo);
            float y = s->vtarget / VAL_SCALE;
            float e = v - y;
            vloss += (double)e * e;
            /* Each complete state occurs once from each perspective.  Half
             * weight per duplicate gives one centralized squared loss. */
            dcenter = 0.5f * e * t->vcoef;
        }

        /* Match training to deployment's fixed-cardinality posterior.  The
         * sampler is queried only for the player to move, so restrict labels
         * to that same information-state distribution.  Ply zero is the
         * exact uniform prior by construction. */
        int nb = 0;
        if (t->bw > 0.0f && s->persp == s->st.turn && s->st.nply > 0) {
            const State *st = &s->st;
            int p = s->persp, o = p ^ 1;
            lc_unseen(st, p, bcard, &nb);
            int need = st->hand_n[o] - __builtin_popcountll(st->known[o]);
            for (int k = 0; k < nb; k++)
                held[k] = (uint8_t)((st->hand[o] >> bcard[k]) & 1ULL);
            net_belief_act(t->net, &act, bcard, nb, blogit);
            double nll = 0.0;
            if (belief_exact_k_eval(blogit, held, nb, need, 1.0f,
                                    bmarg, &nll)) {
                float scale = t->bw / (float)(nb > 0 ? nb : 1);
                for (int k = 0; k < nb; k++)
                    dbel[k] = scale * (bmarg[k] - held[k]);
                bloss += nll / (double)(nb > 0 ? nb : 1);
                bn++;
            } else {
                nb = 0;
            }
        }

        if (t->belief_only) {
            if (nb > 0)
                net_backward_belief_head(&act, bcard, dbel, nb, t->grad);
            continue;
        }

        int n = 0;
        if (s->actor) {
            n = lc_moves(&s->st, mv);
            int ci = -1;
            for (int k = 0; k < n; k++) {
                pk[k] = MOVE_PACK(mv[k]);
                if (pk[k] == s->chosen) ci = k;
            }
            if (ci < 0) {
                net_backward(t->net, &f, &act, dcenter, pk, NULL, 0,
                             nb > 0 ? bcard : NULL,
                             nb > 0 ? dbel : NULL, nb, t->grad);
                if (dcenter != 0.0f)
                    net_backward(t->net, &of, &oact, -dcenter, pk, NULL,
                                 0, NULL, NULL, 0, t->grad);
                continue;
            }
            net_policy_act(t->net, &act, pk, n, logit);
            float rawmx = logit[0];
            for (int k = 1; k < n; k++)
                if (logit[k] > rawmx) rawmx = logit[k];
            float rawsum = 0.0f;
            for (int k = 0; k < n; k++) {
                rawprob[k] = expf(logit[k] - rawmx);
                rawsum += rawprob[k];
            }
            for (int k = 0; k < n; k++) rawprob[k] /= rawsum;
            /* Data collection samples softmax(logit / temp), so PPO must
             * compare oldp with that same behaviour-policy family. */
            if (t->temp != 1.0f)
                for (int k = 0; k < n; k++) logit[k] /= t->temp;
            float mx = logit[0];
            for (int k = 1; k < n; k++) if (logit[k] > mx) mx = logit[k];
            float sum = 0.0f;
            for (int k = 0; k < n; k++) { prob[k] = expf(logit[k] - mx); sum += prob[k]; }
            float inv = 1.0f / sum;
            float ent = 0.0f;
            for (int k = 0; k < n; k++) {
                prob[k] *= inv;
                if (prob[k] > 1e-9f) ent -= prob[k] * logf(prob[k]);
            }
            float A = s->adv;
            float ratio = prob[ci] / (s->oldp > 1e-9f ? s->oldp : 1e-9f);
            float lo = 1.0f - t->clip, hi = 1.0f + t->clip;
            /* PPO: gradient flows only when the unclipped branch is the
             * binding one, which is what stops a single batch from moving the
             * policy too far off the data it was collected under. */
            int use = 1;
            if (ratio > hi && A > 0.0f) use = 0;
            if (ratio < lo && A < 0.0f) use = 0;
            if (!use) clipped += 1.0;
            ploss += -(double)(ratio < lo ? lo : (ratio > hi ? hi : ratio)) * A;
            float gsurr = use ? -A * ratio : 0.0f;
            for (int k = 0; k < n; k++) {
                float dsurr = gsurr * ((k == ci ? 1.0f : 0.0f) - prob[k]);
                float dent = t->entcoef * prob[k] * (logf(prob[k] + 1e-9f) + ent);
                /* net_backward differentiates the untempered network logit. */
                dlog[k] = t->policy_scale * (dsurr + dent) / t->temp;
            }
            if (t->anchor && t->klcoef > 0.0f) {
                NetAct aact;
                net_trunk(t->anchor, &f, &aact);
                net_policy_act(t->anchor, &aact, pk, n, alogit);
                float amx = alogit[0];
                for (int k = 1; k < n; k++)
                    if (alogit[k] > amx) amx = alogit[k];
                float asum = 0.0f;
                for (int k = 0; k < n; k++) {
                    aprob[k] = expf(alogit[k] - amx);
                    asum += aprob[k];
                }
                for (int k = 0; k < n; k++) {
                    aprob[k] /= asum;
                    klloss += aprob[k] *
                        (logf(aprob[k] + 1e-9f) -
                         logf(rawprob[k] + 1e-9f));
                    /* Full-action KL: even legal moves not sampled by PPO
                     * remain anchored to the proven checkpoint. */
                    dlog[k] += t->policy_scale * t->klcoef *
                               (rawprob[k] - aprob[k]);
                }
            }
            pn++;
            net_backward(t->net, &f, &act, dcenter, pk, dlog, n,
                         nb > 0 ? bcard : NULL,
                         nb > 0 ? dbel : NULL, nb, t->grad);
        } else {
            net_backward(t->net, &f, &act, dcenter, pk, NULL, 0,
                         nb > 0 ? bcard : NULL,
                         nb > 0 ? dbel : NULL, nb, t->grad);
        }
        if (dcenter != 0.0f)
            net_backward(t->net, &of, &oact, -dcenter, pk, NULL, 0,
                         NULL, NULL, 0, t->grad);
    }
    t->ploss = ploss; t->vloss = vloss; t->bloss = bloss;
    t->klloss = klloss; t->pn = pn; t->bn = bn;
    t->clipped = clipped;
    return NULL;
}

static void grad_accumulate(Net *dst, Net *const *src, int n)
{
    float *d = (float *)dst;
    size_t nw = sizeof(Net) / sizeof(float);
    for (int k = 1; k < n; k++) {
        const float *s = (const float *)src[k];
        for (size_t i = 0; i < nw; i++) d[i] += s[i];
    }
}

/* Warm up only the capacity appended after the inherited v4 checkpoint.
 * Ordered-pile rows and complete-move residuals may learn; every proven
 * legacy parameter is restored byte-for-byte after each optimiser step. */
static void restore_legacy_parameters(Net *net, const Net *base)
{
    memcpy(net->w1, base->w1,
           (size_t)FEAT_LEGACY_DIM * NET_H1 * sizeof(float));
    size_t from = offsetof(Net, b1);
    size_t to = offsetof(Net, wcomb);
    memcpy((unsigned char *)net + from,
           (const unsigned char *)base + from, to - from);
}

int main(int argc, char **argv)
{
    const char *out_path = "data/rl.bin";
    const char *init_path = "data/champion.bin";
    const char *ref_spec = "heur";
    const char *gen_opponent_spec = NULL;
    const char *anchor_path = NULL;
    int iters = 30, games = 4000, nthread = 4, batch = 512, epochs = 2;
    int eval_pairs = 400, eval_every = 1;
    float lr = 3e-4f, wd = 1e-7f, lambda = 0.85f, clip = 0.2f;
    float vcoef = 1.0f, entcoef = 0.004f, temp = 1.0f;
    float winbonus = 15.0f, bw = 1.0f, mw = 1.0f;
    float opponent_mix = 0.0f, klcoef = 0.0f;
    int v6_only = 0, belief_only = 0;
    int rounds = MATCH_ROUNDS, trajectory_symmetries = 0;
    uint64_t seed = 7, eval_seed = 20260727ULL;

    for (int i = 1; i < argc; i++) {
        const char *k = argv[i];
        #define ARG(name) (!strcmp(k, name) && i + 1 < argc)
        if (ARG("--out")) out_path = argv[++i];
        else if (ARG("--init")) init_path = argv[++i];
        else if (ARG("--ref")) ref_spec = argv[++i];
        else if (ARG("--gen-opponent")) gen_opponent_spec = argv[++i];
        else if (ARG("--opponent-mix")) opponent_mix = (float)atof(argv[++i]);
        else if (ARG("--anchor")) anchor_path = argv[++i];
        else if (ARG("--kl")) klcoef = (float)atof(argv[++i]);
        else if (ARG("--iters")) iters = atoi(argv[++i]);
        else if (ARG("--games")) games = atoi(argv[++i]);
        else if (ARG("--threads")) nthread = atoi(argv[++i]);
        else if (ARG("--batch")) batch = atoi(argv[++i]);
        else if (ARG("--epochs")) epochs = atoi(argv[++i]);
        else if (ARG("--lr")) lr = (float)atof(argv[++i]);
        else if (ARG("--lambda")) lambda = (float)atof(argv[++i]);
        else if (ARG("--clip")) clip = (float)atof(argv[++i]);
        else if (ARG("--vcoef")) vcoef = (float)atof(argv[++i]);
        else if (ARG("--ent")) entcoef = (float)atof(argv[++i]);
        else if (ARG("--temp")) temp = (float)atof(argv[++i]);
        else if (ARG("--winbonus")) winbonus = (float)atof(argv[++i]);
        else if (ARG("--bw")) bw = (float)atof(argv[++i]);
        else if (ARG("--mw")) mw = (float)atof(argv[++i]);
        else if (ARG("--rounds")) rounds = atoi(argv[++i]);
        else if (ARG("--wd")) wd = (float)atof(argv[++i]);
        else if (ARG("--eval")) eval_pairs = atoi(argv[++i]);
        else if (ARG("--eval-every")) eval_every = atoi(argv[++i]);
        else if (ARG("--eval-seed")) eval_seed = strtoull(argv[++i], NULL, 10);
        else if (ARG("--seed")) seed = strtoull(argv[++i], NULL, 10);
        else if (ARG("--trajectory-symmetries")) {
            const char *v = argv[++i];
            char *end = NULL;
            long parsed = strtol(v, &end, 10);
            trajectory_symmetries = (end == v || *end != '\0' ||
                                     parsed < INT_MIN || parsed > INT_MAX)
                                  ? -1 : (int)parsed;
        }
        else if (!strcmp(k, "--v6-only")) v6_only = 1;
        else if (!strcmp(k, "--belief-only")) belief_only = 1;
        else { fprintf(stderr, "unknown option %s\n", k); return 1; }
        #undef ARG
    }
    if (!(temp > 0.0f) || !lc_float_isfinite(temp)) {
        fprintf(stderr, "--temp must be finite and greater than zero\n");
        return 1;
    }
    if (trajectory_symmetries != 0 && trajectory_symmetries != 1 &&
        trajectory_symmetries != 5 && trajectory_symmetries != 10 &&
        trajectory_symmetries != 20 && trajectory_symmetries != 120) {
        fprintf(stderr,
                "--trajectory-symmetries must be 0, 1, 5, 10, 20, or 120\n");
        return 1;
    }
    if (bw < 0.0f || !lc_float_isfinite(bw)) {
        fprintf(stderr, "--bw must be finite and non-negative\n");
        return 1;
    }
    if (belief_only && !(bw > 0.0f)) {
        fprintf(stderr, "--belief-only requires --bw greater than zero\n");
        return 1;
    }
    if (belief_only && v6_only) {
        fprintf(stderr, "--belief-only and --v6-only are mutually exclusive\n");
        return 1;
    }
    if (opponent_mix < 0.0f || opponent_mix > 1.0f ||
        !lc_float_isfinite(opponent_mix)) {
        fprintf(stderr, "--opponent-mix must be between zero and one\n");
        return 1;
    }
    if (opponent_mix > 0.0f && !gen_opponent_spec) {
        fprintf(stderr, "--opponent-mix requires --gen-opponent SPEC\n");
        return 1;
    }
    if (opponent_mix > 0.0f && (games & 1)) {
        fprintf(stderr, "population training requires an even --games count\n");
        return 1;
    }
    if (klcoef < 0.0f || !lc_float_isfinite(klcoef)) {
        fprintf(stderr, "--kl must be finite and non-negative\n");
        return 1;
    }
    if (klcoef > 0.0f && !anchor_path) {
        fprintf(stderr, "--kl requires --anchor PATH\n");
        return 1;
    }
    if (belief_only && klcoef > 0.0f) {
        fprintf(stderr, "--belief-only cannot optimize an anchor KL\n");
        return 1;
    }
    if (rounds < 1 || rounds > MATCH_ROUNDS) {
        fprintf(stderr, "--rounds must be between 1 and %d\n", MATCH_ROUNDS);
        return 1;
    }

    Net *net = (Net *)malloc(sizeof(Net));
    Net *frozen = (Net *)malloc(sizeof(Net));
    Net *anchor = anchor_path ? (Net *)malloc(sizeof(Net)) : NULL;
    Net *legacy_base = v6_only ? (Net *)malloc(sizeof(Net)) : NULL;
    Adam *adam = (Adam *)calloc(1, sizeof(Adam));
    if (!net || !frozen || (anchor_path && !anchor) ||
        (v6_only && !legacy_base) || !adam) {
        fprintf(stderr, "network allocation failed\n");
        return 1;
    }
    if (net_load(net, init_path) != 0) { fprintf(stderr, "cannot load %s\n", init_path); return 1; }
    if (belief_only) net_project_belief_wager_symmetry(net);
    else net_project_wager_symmetry(net);
    if (anchor && net_load(anchor, anchor_path) != 0) {
        fprintf(stderr, "cannot load anchor %s\n", anchor_path);
        return 1;
    }
    if (anchor) net_project_wager_symmetry(anchor);
    if (legacy_base) memcpy(legacy_base, net, sizeof(Net));
    printf("initialised from %s\n", init_path);

    Agent ref;
    spec_parse(ref_spec, &ref);
    Agent gen_opponent;
    const Agent *gen_opponent_ptr = NULL;
    if (gen_opponent_spec) {
        spec_parse(gen_opponent_spec, &gen_opponent);
        gen_opponent_ptr = &gen_opponent;
    }

    Net **grads = (Net **)calloc((size_t)nthread, sizeof(Net *));
    for (int i = 0; i < nthread; i++) grads[i] = (Net *)malloc(sizeof(Net));

    size_t cap = (size_t)games * 210 * (size_t)rounds;
    RLSample *buf = (RLSample *)malloc(sizeof(RLSample) * cap);
    if (!buf) { fprintf(stderr, "sample buffer allocation failed\n"); return 1; }
    int *order = (int *)malloc(sizeof(int) * cap);

    printf("ppo: %d iters x %d matches of %d round(s), batch %d, %d epochs, lr %.1e, "
           "lambda %.2f, ent %.4f, winbonus %.0f, margin weight %.2f, KL %.4f%s\n",
           iters, games, rounds, batch, epochs, lr, lambda, entcoef, winbonus,
           mw, klcoef, v6_only ? ", v6-only" : "");
    if (gen_opponent_ptr)
        printf("     opponent population: %.1f%% %s, %.1f%% live self-play\n",
               100.0f * opponent_mix, gen_opponent_spec,
               100.0f * (1.0f - opponent_mix));
    if (trajectory_symmetries > 0)
        printf("     trajectory suit augmentation: exact group %d, one fixed "
               "mapping per match\n", trajectory_symmetries);
    if (belief_only)
        printf("     optimizer: belief head only (trunk, policy and value frozen)\n");
    fflush(stdout);

    for (int it = 1; it <= iters; it++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        memcpy(frozen, net, sizeof(Net));

        GenJob *jobs = (GenJob *)calloc((size_t)nthread, sizeof(GenJob));
        pthread_t *th = (pthread_t *)calloc((size_t)nthread, sizeof(pthread_t));
        size_t per = cap / (size_t)nthread;
        for (int i = 0; i < nthread; i++) {
            jobs[i].net = frozen;
            jobs[i].opponent = gen_opponent_ptr;
            jobs[i].opponent_mix = opponent_mix;
            jobs[i].games = games;
            jobs[i].seed = seed * 7919ULL + (uint64_t)it * 104729ULL;
            jobs[i].thread = i; jobs[i].nthread = nthread;
            jobs[i].out = buf + per * (size_t)i;
            jobs[i].cap = per;
            jobs[i].lambda = lambda;
            jobs[i].temp = temp;
            jobs[i].winbonus = winbonus;
            jobs[i].mw = mw;
            jobs[i].rounds = rounds;
            jobs[i].trajectory_symmetries = trajectory_symmetries;
        }
        for (int i = 0; i < nthread; i++) pthread_create(&th[i], NULL, gen_worker, &jobs[i]);
        for (int i = 0; i < nthread; i++) pthread_join(th[i], NULL);

        /* compact the per-thread blocks into one contiguous array */
        size_t n = 0;
        for (int i = 0; i < nthread; i++) {
            if (jobs[i].out != buf + n) memmove(buf + n, jobs[i].out, jobs[i].nout * sizeof(RLSample));
            n += jobs[i].nout;
        }
        double plies = 0, absm = 0, pts = 0, ent = 0, p0w = 0, learnerw = 0;
        size_t seen = 0;
        long entn = 0;
        int gdone = 0, opponent_games = 0;
        uint64_t augmentation_fingerprint = 0;
        int learner_seat_games[2] = {0, 0};
        for (int i = 0; i < nthread; i++) {
            plies += jobs[i].plies; absm += jobs[i].absmargin; pts += jobs[i].score_sum;
            ent += jobs[i].entropy; entn += jobs[i].entropy_n;
            p0w += jobs[i].p0_match_wins;
            learnerw += jobs[i].learner_match_wins;
            opponent_games += jobs[i].opponent_games;
            augmentation_fingerprint ^= jobs[i].augmentation_fingerprint;
            learner_seat_games[0] += jobs[i].learner_seat_games[0];
            learner_seat_games[1] += jobs[i].learner_seat_games[1];
            seen += jobs[i].nseen;
            gdone += jobs[i].done;
        }
        free(jobs); free(th);

        /* standardise advantages */
        double am = 0, av = 0;
        long an = 0;
        for (size_t i = 0; i < n; i++) if (buf[i].actor) { am += buf[i].adv; an++; }
        am /= (an ? an : 1);
        for (size_t i = 0; i < n; i++) if (buf[i].actor) { double d = buf[i].adv - am; av += d * d; }
        av = sqrt(av / (an ? an : 1)) + 1e-6;
        for (size_t i = 0; i < n; i++) if (buf[i].actor) buf[i].adv = (float)((buf[i].adv - am) / av);
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double gs = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
        printf("iter %2d: %d matches %.1fs (%.0f m/s), %zu samples, plies %.1f, "
               "points/side %.1f, |margin| %.1f, p0 wins %.1f%%, entropy %.2f, adv sd %.1f\n",
               it, gdone, gs, gdone / gs, n, plies / gdone, pts / (2 * gdone),
               absm / gdone, 100.0 * p0w / gdone, ent / entn, av);
        if (trajectory_symmetries > 0)
            printf("         policy-gradient rows %ld/%zu; augmentation "
                   "fingerprint %016llx\n", an, n,
                   (unsigned long long)augmentation_fingerprint);
        if (opponent_games > 0)
            printf("         frozen-opponent games %d, learner seats %d/%d, "
                   "learner score %.1f%%\n",
                   opponent_games, learner_seat_games[0], learner_seat_games[1],
                   100.0 * learnerw / opponent_games);
        if (seen > n)
            printf("         reservoir retained %zu/%zu generated samples (%.1f%%)\n",
                   n, seen, 100.0 * (double)n / (double)seen);
        fflush(stdout);

        clock_gettime(CLOCK_MONOTONIC, &t0);
        Rng r; rng_seed(&r, seed + 31ULL * (uint64_t)it);
        double pl = 0, vl = 0, bl = 0, kl = 0, cl = 0;
        long pn = 0, bcnt = 0, steps = 0;
        for (int ep = 0; ep < epochs; ep++) {
            for (size_t i = 0; i < n; i++) order[i] = (int)i;
            for (size_t i = n - 1; i > 0; i--) {
                uint32_t jx = rng_below(&r, (uint32_t)i + 1);
                int t = order[i]; order[i] = order[jx]; order[jx] = t;
            }
            for (size_t off = 0; off + (size_t)batch <= n; off += (size_t)batch) {
                OptJob tj[64];
                pthread_t tt[64];
                int nt = nthread > 64 ? 64 : nthread;
                int chunk = (batch + nt - 1) / nt;
                int actor_count = 0;
                for (int i = 0; i < batch; i++)
                    actor_count += buf[order[off + (size_t)i]].actor != 0;
                /* Frozen-opponent samples have one policy actor rather than
                 * two, but both perspectives still provide value/belief
                 * labels.  Compensate for the smaller actor fraction while
                 * preserving the historical self-play policy:value scale. */
                float policy_scale = 1.0f;
                if (opponent_mix > 0.0f && actor_count > 0)
                    policy_scale =
                        0.5f * (float)batch / (float)actor_count;
                for (int i = 0; i < nt; i++) {
                    tj[i].net = net; tj[i].anchor = anchor;
                    tj[i].grad = grads[i]; tj[i].buf = buf;
                    tj[i].idx = order + off;
                    tj[i].from = i * chunk > batch ? batch : i * chunk;
                    tj[i].to = (i + 1) * chunk > batch ? batch : (i + 1) * chunk;
                    tj[i].clip = clip; tj[i].vcoef = vcoef; tj[i].entcoef = entcoef;
                    tj[i].policy_scale = policy_scale;
                    tj[i].bw = bw; tj[i].temp = temp;
                    tj[i].klcoef = klcoef;
                    tj[i].belief_only = belief_only;
                }
                for (int i = 0; i < nt; i++) pthread_create(&tt[i], NULL, opt_worker, &tj[i]);
                for (int i = 0; i < nt; i++) pthread_join(tt[i], NULL);
                for (int i = 0; i < nt; i++) {
                    pl += tj[i].ploss; vl += tj[i].vloss;
                    bl += tj[i].bloss; kl += tj[i].klloss;
                    pn += tj[i].pn; bcnt += tj[i].bn;
                    cl += tj[i].clipped;
                }
                grad_accumulate(grads[0], grads, nt);
                net_tie_wager_gradients(grads[0]);
                if (belief_only)
                    net_adam_step_belief(net, grads[0], adam, lr,
                                         1.0f / (float)batch, wd);
                else
                    net_adam_step(net, grads[0], adam, lr,
                                  1.0f / (float)batch, wd);
                if (legacy_base) restore_legacy_parameters(net, legacy_base);
                steps++;
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ts = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
        if (belief_only)
            printf("         %ld belief-head-only updates in %.1fs: "
                   "exact-K nll/card %.3f\n",
                   steps, ts, bcnt ? bl / bcnt : 0.0);
        else
            printf("         %ld updates in %.1fs: value rmse %.1f pts, surrogate %.4f, "
                   "belief exact-K nll/card %.3f, anchor KL %.5f, clipped %.1f%%\n",
                   steps, ts,
                   sqrt(vl / ((double)steps * batch)) * VAL_SCALE,
                   pn ? pl / pn : 0.0, bcnt ? bl / bcnt : 0.0,
                   pn ? kl / pn : 0.0,
                   pn ? 100.0 * cl / pn : 0.0);
        fflush(stdout);

        char path[512];
        net_save(net, out_path);
        snprintf(path, sizeof path, "%s.it%d", out_path, it);
        net_save(net, path);

        if (eval_pairs > 0 && (it % eval_every == 0 || it == iters)) {
            Agent cur;
            agent_default(&cur, AG_POLICY, net);
            MatchResult mr;
            match_run_r(&cur, &ref, eval_pairs, nthread, eval_seed, rounds, &mr);
            printf("         vs %s: margin %+.2f +- %.2f, match wins %.1f%%, plies %.0f\n",
                   ref_spec, mr.margin, mr.margin_se, 100 * mr.winrate, mr.plies);
            fflush(stdout);
        }
    }
    return 0;
}
