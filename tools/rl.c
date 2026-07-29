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
 * Both seats are the same network, so every game yields training data from two
 * points of view.  Generation is one forward pass per ply, which is fast enough
 * that the data is always fresh and on-policy.
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
    int games;          /* matches per iteration */
    uint64_t seed;
    int thread, nthread;
    RLSample *out;
    size_t nout, nseen, cap;
    double plies, absmargin, score_sum, entropy;
    double p0_match_wins;
    long entropy_n;
    int done;
    float lambda;
    float temp;
    float winbonus;     /* terminal reward for winning the match, in points */
    float mw;           /* weight of the margin term in the return          */
    int rounds;
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
    Features f;
    Move mv[MAX_MOVES];
    float pr[MAX_MOVES];

    for (int g = j->thread; g < j->games; g += j->nthread) {
        /* one episode = one full match of j->rounds rounds */
        int T = 0;
        int cum[2] = { 0, 0 };
        for (int rd = 0; rd < j->rounds; rd++) {
        State st;
        lc_deal(&st, &rng);
        st.round = (uint8_t)rd;
        /* Preserve the exact match context used by src/match.c.  Clamping
         * the two totals independently changes the lead. */
        st.cum[0] = checked_cumulative_score(cum[0]);
        st.cum[1] = checked_cumulative_score(cum[1]);
        st.turn = (uint8_t)(rd & 1);
        int Tstop = T + LC_MAX_PLIES;
        while (!st.over && T < Tstop) {
            chain[T] = st;
            int n = policy_probs(j->net, &st, mv, pr, NULL);
            double h = 0.0;
            for (int i = 0; i < n; i++) if (pr[i] > 1e-9f) h -= pr[i] * log(pr[i]);
            j->entropy += h; j->entropy_n++;

            int c;
            if (j->temp != 1.0f) {
                /* Sampling off-policy is fine as long as the recorded
                 * probability is the behaviour policy's, since that is what the
                 * PPO ratio divides by. */
                float w[MAX_MOVES], sum = 0.0f;
                for (int i = 0; i < n; i++) { w[i] = powf(pr[i], 1.0f / j->temp); sum += w[i]; }
                c = sample_index(w, n, &rng);
                chain_p[T] = w[c] / sum;
            } else {
                c = sample_index(pr, n, &rng);
                chain_p[T] = pr[c];
            }
            chain_mv[T] = MOVE_PACK(mv[c]);
            T++;
            lc_apply(&st, mv[c]);
        }
        cum[0] += lc_score(&st, 0);
        cum[1] += lc_score(&st, 1);
        j->plies += st.nply;
        }   /* rounds */

        int score[2] = { cum[0], cum[1] };
        if (score[0] > score[1]) j->p0_match_wins += 1.0;
        else if (score[0] == score[1]) j->p0_match_wins += 0.5;

        /* values of every state from both points of view */
        for (int p = 0; p < 2; p++)
            for (int t = 0; t < T; t++) {
                feat_extract(&chain[t], p, &f);
                chain_v[p][t] = net_value(j->net, &f) * VAL_SCALE;
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
                if (chain[t].turn == p) {
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
    Net *grad;
    const RLSample *buf;
    const int *idx;
    int from, to;
    float clip, vcoef, entcoef, advscale, bw, temp;
    double ploss, vloss, bloss, clipped;
    int pn;
    long bn;
} OptJob;

static void *opt_worker(void *arg)
{
    OptJob *t = (OptJob *)arg;
    net_zero(t->grad);
    double ploss = 0, vloss = 0, bloss = 0, clipped = 0;
    int pn = 0;
    long bn = 0;
    Features f;
    NetAct act;
    Move mv[MAX_MOVES];
    uint16_t pk[MAX_MOVES];
    float logit[MAX_MOVES], prob[MAX_MOVES], dlog[MAX_MOVES];
    uint8_t bcard[NCARD];
    float blogit[NCARD], dbel[NCARD];

    for (int i = t->from; i < t->to; i++) {
        const RLSample *s = &t->buf[t->idx[i]];
        feat_extract(&s->st, s->persp, &f);
        net_trunk(t->net, &f, &act);

        float v = net_value_act(t->net, &act);
        float y = s->vtarget / VAL_SCALE;
        float e = v - y;
        vloss += (double)e * e;
        float dvalue = 2.0f * e * t->vcoef;

        /* belief head: the sample stores the true opponent hand, so every
         * state is a free supervised example of "what do their actions imply
         * about their cards" */
        int nb = 0;
        {
            const State *st = &s->st;
            int p = s->persp, o = p ^ 1;
            uint64_t vis = st->hand[p] | st->played[0] | st->played[1] |
                           st->discarded | st->known[o];
            uint64_t cands = ~vis & ((1ULL << NCARD) - 1);
            uint64_t m2 = cands;
            while (m2) { int cc = __builtin_ctzll(m2); m2 &= m2 - 1; bcard[nb++] = (uint8_t)cc; }
            net_belief_act(t->net, &act, bcard, nb, blogit);
            float scale = t->bw / (float)(nb > 0 ? nb : 1);
            for (int k = 0; k < nb; k++) {
                float lab = ((st->hand[o] >> bcard[k]) & 1ULL) ? 1.0f : 0.0f;
                /* clamp before the sigmoid: with -ffast-math an overflowing
                 * expf poisons the gradient with NaN, and a NaN here would
                 * flow into the shared trunk and destroy all three heads */
                float l = blogit[k];
                if (l > 15.0f) l = 15.0f;
                if (l < -15.0f) l = -15.0f;
                float pr2 = 1.0f / (1.0f + expf(-l));
                bloss += -(double)(lab * logf(pr2 + 1e-6f) + (1.0f - lab) * logf(1.0f - pr2 + 1e-6f));
                dbel[k] = scale * (pr2 - lab);
            }
            bn += nb;
        }

        int n = 0;
        if (s->actor) {
            n = lc_moves(&s->st, mv);
            int ci = -1;
            for (int k = 0; k < n; k++) {
                pk[k] = MOVE_PACK(mv[k]);
                if (pk[k] == s->chosen) ci = k;
            }
            if (ci < 0) { net_backward(t->net, &f, &act, dvalue, pk, NULL, 0, bcard, dbel, nb, t->grad); continue; }
            net_policy_act(t->net, &act, pk, n, logit);
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
            float A = s->adv * t->advscale;
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
                dlog[k] = (dsurr + dent) / t->temp;
            }
            pn++;
            net_backward(t->net, &f, &act, dvalue, pk, dlog, n, bcard, dbel, nb, t->grad);
        } else {
            net_backward(t->net, &f, &act, dvalue, pk, NULL, 0, bcard, dbel, nb, t->grad);
        }
    }
    t->ploss = ploss; t->vloss = vloss; t->bloss = bloss; t->pn = pn; t->bn = bn;
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

int main(int argc, char **argv)
{
    const char *out_path = "data/rl.bin";
    const char *init_path = "data/champion.bin";
    const char *ref_spec = "heur";
    int iters = 30, games = 4000, nthread = 4, batch = 512, epochs = 2;
    int eval_pairs = 400, eval_every = 1;
    float lr = 3e-4f, wd = 1e-7f, lambda = 0.85f, clip = 0.2f;
    float vcoef = 1.0f, entcoef = 0.004f, temp = 1.0f;
    float winbonus = 15.0f, bw = 1.0f, mw = 1.0f;
    int rounds = MATCH_ROUNDS;
    uint64_t seed = 7, eval_seed = 20260727ULL;

    for (int i = 1; i < argc; i++) {
        const char *k = argv[i];
        #define ARG(name) (!strcmp(k, name) && i + 1 < argc)
        if (ARG("--out")) out_path = argv[++i];
        else if (ARG("--init")) init_path = argv[++i];
        else if (ARG("--ref")) ref_spec = argv[++i];
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
        else { fprintf(stderr, "unknown option %s\n", k); return 1; }
        #undef ARG
    }
    if (!(temp > 0.0f) || !isfinite(temp)) {
        fprintf(stderr, "--temp must be finite and greater than zero\n");
        return 1;
    }
    if (rounds < 1 || rounds > MATCH_ROUNDS) {
        fprintf(stderr, "--rounds must be between 1 and %d\n", MATCH_ROUNDS);
        return 1;
    }

    Net *net = (Net *)malloc(sizeof(Net));
    Net *frozen = (Net *)malloc(sizeof(Net));
    Adam *adam = (Adam *)calloc(1, sizeof(Adam));
    if (net_load(net, init_path) != 0) { fprintf(stderr, "cannot load %s\n", init_path); return 1; }
    net_project_wager_symmetry(net);
    printf("initialised from %s\n", init_path);

    Agent ref;
    spec_parse(ref_spec, &ref);

    Net **grads = (Net **)calloc((size_t)nthread, sizeof(Net *));
    for (int i = 0; i < nthread; i++) grads[i] = (Net *)malloc(sizeof(Net));

    size_t cap = (size_t)games * 210 * (size_t)rounds;
    RLSample *buf = (RLSample *)malloc(sizeof(RLSample) * cap);
    if (!buf) { fprintf(stderr, "sample buffer allocation failed\n"); return 1; }
    int *order = (int *)malloc(sizeof(int) * cap);

    printf("ppo: %d iters x %d matches of %d round(s), batch %d, %d epochs, lr %.1e, "
           "lambda %.2f, ent %.4f, winbonus %.0f, margin weight %.2f\n",
           iters, games, rounds, batch, epochs, lr, lambda, entcoef, winbonus, mw);
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
        }
        for (int i = 0; i < nthread; i++) pthread_create(&th[i], NULL, gen_worker, &jobs[i]);
        for (int i = 0; i < nthread; i++) pthread_join(th[i], NULL);

        /* compact the per-thread blocks into one contiguous array */
        size_t n = 0;
        for (int i = 0; i < nthread; i++) {
            if (jobs[i].out != buf + n) memmove(buf + n, jobs[i].out, jobs[i].nout * sizeof(RLSample));
            n += jobs[i].nout;
        }
        double plies = 0, absm = 0, pts = 0, ent = 0, p0w = 0;
        size_t seen = 0;
        long entn = 0;
        int gdone = 0;
        for (int i = 0; i < nthread; i++) {
            plies += jobs[i].plies; absm += jobs[i].absmargin; pts += jobs[i].score_sum;
            ent += jobs[i].entropy; entn += jobs[i].entropy_n;
            p0w += jobs[i].p0_match_wins;
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
        if (seen > n)
            printf("         reservoir retained %zu/%zu generated samples (%.1f%%)\n",
                   n, seen, 100.0 * (double)n / (double)seen);
        fflush(stdout);

        clock_gettime(CLOCK_MONOTONIC, &t0);
        Rng r; rng_seed(&r, seed + 31ULL * (uint64_t)it);
        double pl = 0, vl = 0, bl = 0, cl = 0;
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
                for (int i = 0; i < nt; i++) {
                    tj[i].net = net; tj[i].grad = grads[i]; tj[i].buf = buf;
                    tj[i].idx = order + off;
                    tj[i].from = i * chunk > batch ? batch : i * chunk;
                    tj[i].to = (i + 1) * chunk > batch ? batch : (i + 1) * chunk;
                    tj[i].clip = clip; tj[i].vcoef = vcoef; tj[i].entcoef = entcoef;
                    tj[i].advscale = 1.0f; tj[i].bw = bw; tj[i].temp = temp;
                }
                for (int i = 0; i < nt; i++) pthread_create(&tt[i], NULL, opt_worker, &tj[i]);
                for (int i = 0; i < nt; i++) pthread_join(tt[i], NULL);
                for (int i = 0; i < nt; i++) { pl += tj[i].ploss; vl += tj[i].vloss; bl += tj[i].bloss; pn += tj[i].pn; bcnt += tj[i].bn; cl += tj[i].clipped; }
                grad_accumulate(grads[0], grads, nt);
                net_tie_wager_gradients(grads[0]);
                net_adam_step(net, grads[0], adam, lr, 1.0f / (float)batch, wd);
                steps++;
            }
        }
        clock_gettime(CLOCK_MONOTONIC, &t1);
        double ts = (t1.tv_sec - t0.tv_sec) + 1e-9 * (t1.tv_nsec - t0.tv_nsec);
        printf("         %ld updates in %.1fs: value rmse %.1f pts, surrogate %.4f, "
               "belief bce %.3f, clipped %.1f%%\n",
               steps, ts, sqrt(vl / ((double)steps * batch)) * VAL_SCALE,
               pn ? pl / pn : 0.0, bcnt ? bl / bcnt : 0.0, pn ? 100.0 * cl / pn : 0.0);
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
