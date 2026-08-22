#define _XOPEN_SOURCE 700

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
#include "../src/search.h"
#include "../src/spec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <limits.h>
#include <stddef.h>
#include <time.h>
#include <pthread.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>
#include <errno.h>

typedef struct {
    State st;
    float vtarget;    /* lambda-return, points, perspective player's view */
    float adv;        /* advantage, points (actor samples only)           */
    float oldp;       /* policy probability of the move actually played   */
    uint16_t chosen;  /* packed move                                      */
    uint8_t persp;
    uint8_t actor;    /* 1 when persp is the player who moved             */
    uint8_t continuation_role; /* conditional dead-discard support applies */
    /* Suit relabelling from st's owner-role coordinates to the other
     * player's independently fixed role.  Used only in continuation mode to
     * reconstruct the exact centralized critic pair during optimization. */
    uint8_t other_role_perm[NSUIT];
} RLSample;

typedef struct {
    const Net *net;
    const Net *champion; /* immutable prefix/root policy in continuation mode */
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
    int continuation_start; /* 0 ordinary PPO; currently 14 when enabled */
    int continuation_objective; /* deployed rollout objective: legacy 0 or 2 */
    int continuation_independent_roles; /* one fixed suit map per player */
    int continuation_role_group;
    uint32_t *continuation_role_pair_count;
    int continuation_rounds;
    int continuation_baseline_roots;
    int continuation_challenger_roots;
    int continuation_exact_moves;
    int continuation_cycle_forces;
    int continuation_cap_forces;
    int continuation_min_actor_ply;
    uint64_t continuation_fingerprint;
} GenJob;

enum {
    CONTINUATION_START_PLY = 14,
    CONTINUATION_ROOT_WIDTH = 5,
    CONTINUATION_ROOT_SYMMETRIES = 20,
    CONTINUATION_PLAYOUT_PRUNE = 1
};

static const float CONTINUATION_ROOT_FLOOR = 0.02f;

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
/* Continuation mode may give the two policy roles independent, temporally
 * coherent suit mappings.  Keep one stored view per perspective so actor,
 * critic, and PPO reconstruction all use that player's deployed role. */
static _Thread_local State continuation_chain[2][CHAIN_MAX];

static uint64_t mix64(uint64_t x)
{
    x ^= x >> 30;
    x *= UINT64_C(0xBF58476D1CE4E5B9);
    x ^= x >> 27;
    x *= UINT64_C(0x94D049BB133111EB);
    return x ^ (x >> 31);
}

/* Select the fixed suit role(s) used for one continuation trajectory/tail.
 * The disabled path is deliberately the historical trajectory helper so an
 * explicit `shared` configuration remains byte-for-byte compatible.  The
 * independent path mirrors rollout mode 4's balanced ordered-product walk:
 * over every n*n consecutive trajectory ids, each ordered player-role pair
 * occurs exactly once.  Offsets are seed-derived but worker-independent. */
static int continuation_role_permutations(
    int symmetries, uint64_t seed, uint64_t trajectory, int independent,
    uint8_t perm[2][NSUIT], int selected[2])
{
    if (!perm) return 0;
    if (!independent) {
        if (!trajectory_suit_permutation(
                symmetries, seed, trajectory, perm[0]))
            return 0;
        memcpy(perm[1], perm[0], NSUIT);
        if (selected) selected[0] = selected[1] = -1;
        return 1;
    }
    if (symmetries != 5 && symmetries != 10 &&
        symmetries != 20 && symmetries != 120)
        return 0;

    uint8_t group[120][NSUIT];
    int n = suit_permutations(symmetries, group);
    if (n != symmetries) return 0;
    uint64_t schedule_seed = mix64(
        seed ^ UINT64_C(0xA0761D6478BD642F));
    int fixed_offset = (int)(schedule_seed % (uint64_t)n);
    int other_offset = (int)(rotl64(schedule_seed, 31) % (uint64_t)n);
    uint64_t row = trajectory % (uint64_t)n;
    uint64_t block = (trajectory / (uint64_t)n) % (uint64_t)n;
    int first = (fixed_offset + (int)row) % n;
    int other = (other_offset + (int)block + (int)row) % n;
    memcpy(perm[0], group[first], NSUIT);
    memcpy(perm[1], group[other], NSUIT);
    if (selected) {
        selected[0] = first;
        selected[1] = other;
    }
    return 1;
}

/* Compose original->owner and original->other suit maps into owner->other. */
static int continuation_role_permutation_valid(const uint8_t perm[NSUIT])
{
    if (!perm) return 0;
    uint8_t seen = 0;
    for (int s = 0; s < NSUIT; s++) {
        if (perm[s] >= NSUIT || (seen & (uint8_t)(1u << perm[s])))
            return 0;
        seen |= (uint8_t)(1u << perm[s]);
    }
    return 1;
}

static int continuation_relative_role_permutation(
    const uint8_t owner[NSUIT], const uint8_t other[NSUIT],
    uint8_t relative[NSUIT])
{
    if (!relative || !continuation_role_permutation_valid(owner) ||
        !continuation_role_permutation_valid(other))
        return 0;
    for (int s = 0; s < NSUIT; s++) {
        relative[owner[s]] = other[s];
    }
    return 1;
}

static uint64_t net_byte_fingerprint(const Net *net)
{
    const unsigned char *p = (const unsigned char *)net;
    uint64_t h = UINT64_C(1469598103934665603);
    for (size_t i = 0; i < sizeof *net; i++) {
        h ^= p[i];
        h *= UINT64_C(1099511628211);
    }
    return h;
}

typedef struct {
    char canonical[PATH_MAX];
    dev_t device;
    ino_t inode;
    int exists;
} CheckpointIdentity;

/* Resolve both existing files and a not-yet-created output.  Existing
 * symlinks collapse through realpath, while stat's device/inode pair also
 * catches hard links.  For a new output, resolving its parent protects the
 * same file spelled through relative components or a symlinked directory. */
static int checkpoint_identity(const char *path, CheckpointIdentity *out)
{
    if (!path || !*path || !out) return -1;
    memset(out, 0, sizeof *out);
    struct stat sb;
    if (realpath(path, out->canonical)) {
        if (stat(path, &sb) != 0) return -1;
        out->device = sb.st_dev;
        out->inode = sb.st_ino;
        out->exists = 1;
        return 0;
    }
    if (errno != ENOENT && errno != ENOTDIR) return -1;

    size_t len = strlen(path);
    if (len == 0 || len >= PATH_MAX || path[len - 1] == '/') return -1;
    char parent[PATH_MAX], resolved_parent[PATH_MAX];
    const char *base = path;
    const char *slash = strrchr(path, '/');
    if (!slash) {
        memcpy(parent, ".", 2);
    } else {
        base = slash + 1;
        size_t plen = slash == path ? 1 : (size_t)(slash - path);
        if (plen >= sizeof parent) return -1;
        memcpy(parent, path, plen);
        parent[plen] = '\0';
    }
    if (!*base || !strcmp(base, ".") || !strcmp(base, "..") ||
        !realpath(parent, resolved_parent))
        return -1;
    int written = snprintf(out->canonical, sizeof out->canonical,
                           "%s/%s", resolved_parent, base);
    return written < 0 || (size_t)written >= sizeof out->canonical ? -1 : 0;
}

static int checkpoint_same_file(const CheckpointIdentity *a,
                                const CheckpointIdentity *b)
{
    if (!strcmp(a->canonical, b->canonical)) return 1;
    return a->exists && b->exists && a->device == b->device &&
           a->inode == b->inode;
}

static int continuation_checkpoint_preflight(
    const char *out_path, int iters, const char *init_path,
    const char *root_path, const char *anchor_path)
{
    const char *protected_path[3] = { init_path, root_path, anchor_path };
    CheckpointIdentity protected_id[3];
    int nprotected = anchor_path ? 3 : 2;
    for (int i = 0; i < nprotected; i++) {
        if (checkpoint_identity(protected_path[i], &protected_id[i]) != 0 ||
            !protected_id[i].exists) {
            fprintf(stderr, "cannot resolve protected checkpoint %s\n",
                    protected_path[i]);
            return 0;
        }
    }

    for (int i = 0;; i++) {
        char generated[PATH_MAX];
        const char *candidate = out_path;
        if (i > 0) {
            int written = snprintf(generated, sizeof generated, "%s.it%d",
                                   out_path, i);
            if (written < 0 || (size_t)written >= sizeof generated) {
                fprintf(stderr, "generated checkpoint path is too long\n");
                return 0;
            }
            candidate = generated;
        }
        CheckpointIdentity output_id;
        if (checkpoint_identity(candidate, &output_id) != 0) {
            fprintf(stderr, "cannot resolve output checkpoint %s\n",
                    candidate);
            return 0;
        }
        for (int p = 0; p < nprotected; p++) {
            if (checkpoint_same_file(&output_id, &protected_id[p])) {
                fprintf(stderr,
                        "output checkpoint %s aliases protected checkpoint %s\n",
                        candidate, protected_path[p]);
                return 0;
            }
        }
        if (i >= iters) break;
    }
    return 1;
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
                s->continuation_role = 0;
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

/* Condition both the raw and tempered policies on the maintained rollout
 * continuation's exact allowed-action support.  Dominated discards remain
 * legal engine moves, but playout_prune=1 assigns them zero behavior mass.
 * Apply the predicate in the same coordinate system stored in RLSample so
 * suit augmentation and PPO reconstruct precisely the same support. */
static int continuation_condition_policy(
    const State *st, const Move *mv, int n, float temperature,
    float *raw_prob, float *behavior_prob, uint8_t *allowed_out,
    uint64_t *dead_out)
{
    if (!st || !mv || !raw_prob || !behavior_prob || n <= 0 ||
        n > MAX_MOVES || !(temperature > 0.0f) ||
        !lc_float_isfinite(temperature))
        return 0;
    uint64_t dead = CONTINUATION_PLAYOUT_PRUNE
        ? (lc_dead_cards(st) & st->hand[st->turn]) : 0;
    double raw_sum = 0.0;
    int allowed_n = 0;
    for (int i = 0; i < n; i++) {
        int allowed = !(dead && lc_discard_dominated(st, mv[i], dead));
        if (allowed_out) allowed_out[i] = (uint8_t)allowed;
        if (allowed && raw_prob[i] > 0.0f &&
            lc_float_isfinite(raw_prob[i])) {
            raw_sum += raw_prob[i];
            allowed_n++;
        } else {
            raw_prob[i] = 0.0f;
        }
    }
    if (dead_out) *dead_out = dead;
    if (allowed_n <= 0 || !(raw_sum > 0.0) ||
        !lc_double_isfinite(raw_sum))
        return 0;
    float raw_inv = (float)(1.0 / raw_sum);
    double behavior_sum = 0.0;
    for (int i = 0; i < n; i++) {
        if (raw_prob[i] == 0.0f) {
            behavior_prob[i] = 0.0f;
            continue;
        }
        raw_prob[i] *= raw_inv;
        behavior_prob[i] = temperature == 1.0f
            ? raw_prob[i] : powf(raw_prob[i], 1.0f / temperature);
        behavior_sum += behavior_prob[i];
    }
    if (!(behavior_sum > 0.0) || !lc_double_isfinite(behavior_sum))
        return 0;
    float behavior_inv = (float)(1.0 / behavior_sum);
    for (int i = 0; i < n; i++) behavior_prob[i] *= behavior_inv;
    return n;
}

static int continuation_trajectory_policy_probs_plan(
    const Net *net, const NetEvalPlan *plan, const State *engine_state,
    const uint8_t perm[NSUIT], float temperature,
    State *view, Move *view_mv, Move *engine_mv,
    float *raw_prob, float *behavior_prob)
{
    uint8_t inverse[NSUIT], seen = 0;
    for (int s = 0; s < NSUIT; s++) {
        if (perm[s] >= NSUIT || (seen & (uint8_t)(1u << perm[s])))
            return 0;
        seen |= (uint8_t)(1u << perm[s]);
        inverse[perm[s]] = (uint8_t)s;
    }
    lc_permute_suits(engine_state, view, perm);
    int n = policy_probs_sym_plan(
        net, view, view_mv, raw_prob, NULL, 1, plan);
    if (n <= 0) return n;
    for (int i = 0; i < n; i++) {
        engine_mv[i] = lc_permute_move(view_mv[i], inverse);
    }
    return continuation_condition_policy(
        view, view_mv, n, temperature, raw_prob, behavior_prob, NULL, NULL);
}

/* State-start PPO for the role actually served by a rollout continuation
 * network.  This is deliberately a separate generator: when the opt-in mode
 * is disabled, not one branch, random draw, or sample in historical PPO is
 * changed.
 *
 * The immutable champion owns the real prefix (plies 0..13) and the exact
 * production root shortlist at ply 14.  The mover's information state is
 * uniformly determinized before the gradient-free root move is applied.
 * Only decisions strictly below that root are sampled from the current
 * iteration's frozen learner.  The one-card-deck rules solver is also outside
 * the learned policy and therefore contributes no PPO row. */
static void *gen_continuation_worker(void *arg)
{
    GenJob *j = (GenJob *)arg;
    NetEvalPlan champion_plan, learner_plan;
    net_eval_plan_init(j->champion, &champion_plan);
    net_eval_plan_init(j->net, &learner_plan);
    Rng reservoir_rng;
    rng_seed(&reservoir_rng, j->seed ^ (UINT64_C(0xD1B54A32D192ED03)
                                     * (uint64_t)(j->thread + 1)));
    Move mv[MAX_MOVES], engine_mv[MAX_MOVES];
    float pr[MAX_MOVES], behavior[MAX_MOVES];

    for (int g = j->thread; g < j->games; g += j->nthread) {
        int augment = j->trajectory_symmetries > 0;
        uint8_t shared_role_perm[2][NSUIT];
        if (!j->continuation_independent_roles) {
            if (!continuation_role_permutations(
                    j->trajectory_symmetries, j->seed, (uint64_t)g, 0,
                    shared_role_perm, NULL)) {
                fprintf(stderr,
                        "invalid continuation shared suit configuration\n");
                exit(EXIT_FAILURE);
            }
            /* Preserve continuation-v1's exact per-match fingerprint. */
            if (augment) {
                uint64_t code = 0;
                for (int s = 0; s < NSUIT; s++)
                    code = code * NSUIT + shared_role_perm[0][s];
                j->augmentation_fingerprint ^=
                    mix64((uint64_t)g * UINT64_C(0xD6E8FEB86659FD93)
                          ^ code ^ UINT64_C(0x8EBC6AF09C88C6E3));
            }
        }

        int cum[2] = { 0, 0 };
        for (int rd = 0; rd < j->rounds; rd++) {
            uint64_t episode = (uint64_t)g * MATCH_ROUNDS + (uint64_t)rd;
            Rng deal_rng;
            rng_seed(&deal_rng,
                     mix64(j->seed ^ UINT64_C(0x243F6A8885A308D3)
                           ^ (episode + 1) *
                             UINT64_C(0x9E3779B97F4A7C15)));
            State real;
            lc_deal(&real, &deal_rng);
            real.round = (uint8_t)rd;
            real.cum[0] = checked_cumulative_score(cum[0]);
            real.cum[1] = checked_cumulative_score(cum[1]);
            real.turn = (uint8_t)(rd & 1);

            /* The deployed actor is policy-only before its ply-14 handoff.
             * Exact 20-way averaging and strict-first argmax ties match
             * agent_move(AG_POLICY) without exposing any private state. */
            while (!real.over && real.nply < j->continuation_start) {
                int n = policy_probs_sym_plan(
                    j->champion, &real, mv, pr, NULL,
                    CONTINUATION_ROOT_SYMMETRIES, &champion_plan);
                if (n <= 0) {
                    fprintf(stderr,
                            "continuation champion prefix has no legal move\n");
                    exit(EXIT_FAILURE);
                }
                int best = 0;
                for (int i = 1; i < n; i++)
                    if (pr[i] > pr[best]) best = i;
                lc_apply(&real, mv[best]);
            }
            if (real.over || real.nply != j->continuation_start) {
                fprintf(stderr,
                        "continuation prefix did not reach round ply %d\n",
                        j->continuation_start);
                exit(EXIT_FAILURE);
            }

            const int root_player = real.turn;
            uint8_t role_perm[2][NSUIT];
            if (j->continuation_independent_roles) {
                uint8_t scheduled[2][NSUIT];
                int selected_role[2];
                uint64_t trajectory = (uint64_t)g * (uint64_t)j->rounds
                                    + (uint64_t)rd;
                if (!continuation_role_permutations(
                        j->trajectory_symmetries, j->seed, trajectory, 1,
                        scheduled, selected_role)) {
                    fprintf(stderr,
                            "invalid continuation independent role schedule\n");
                    exit(EXIT_FAILURE);
                }
                if (!j->continuation_role_pair_count ||
                    j->continuation_role_group !=
                        j->trajectory_symmetries ||
                    selected_role[0] < 0 || selected_role[1] < 0) {
                    fprintf(stderr,
                            "continuation role diagnostics are not "
                            "initialized\n");
                    exit(EXIT_FAILURE);
                }
                size_t role_cell =
                    (size_t)selected_role[0] *
                        (size_t)j->continuation_role_group
                    + (size_t)selected_role[1];
                if (j->continuation_role_pair_count[role_cell] ==
                    UINT32_MAX) {
                    fprintf(stderr,
                            "continuation role diagnostic counter overflow\n");
                    exit(EXIT_FAILURE);
                }
                j->continuation_role_pair_count[role_cell]++;
                /* Match deployed mode 4: the first product coordinate is the
                 * root player's role; the second belongs to its opponent. */
                memcpy(role_perm[root_player], scheduled[0], NSUIT);
                memcpy(role_perm[root_player ^ 1], scheduled[1], NSUIT);
                uint64_t code[2] = { 0, 0 };
                for (int p = 0; p < 2; p++)
                    for (int s = 0; s < NSUIT; s++)
                        code[p] = code[p] * NSUIT + role_perm[p][s];
                j->augmentation_fingerprint ^=
                    mix64((trajectory + 1) *
                              UINT64_C(0xD6E8FEB86659FD93)
                          ^ code[root_player]
                          ^ rotl64(code[root_player ^ 1], 17)
                          ^ ((uint64_t)root_player << 47)
                          ^ UINT64_C(0x8EBC6AF09C88C6E3));
            } else {
                memcpy(role_perm, shared_role_perm, sizeof role_perm);
            }
            State root_view;
            agent_information_view(&real, root_player, &root_view);
            int nroot = policy_probs_sym_plan(
                j->champion, &root_view, mv, pr, NULL,
                CONTINUATION_ROOT_SYMMETRIES, &champion_plan);
            if (nroot <= 0) {
                fprintf(stderr, "continuation root has no legal move\n");
                exit(EXIT_FAILURE);
            }
            int baseline = 0;
            for (int i = 1; i < nroot; i++)
                if (pr[i] > pr[baseline]) baseline = i;
            int order[8];
            int admitted = rollout_policy_prefix_indices(
                mv, pr, nroot, baseline, CONTINUATION_ROOT_WIDTH,
                CONTINUATION_ROOT_FLOOR, 0.0f, 1, order);
            if (admitted <= 0 || order[0] != baseline) {
                fprintf(stderr,
                        "continuation root policy-prefix admission failed\n");
                exit(EXIT_FAILURE);
            }

            Rng root_rng;
            rng_seed(&root_rng,
                     mix64(j->seed ^ UINT64_C(0xA4093822299F31D0)
                           ^ (episode + 1) *
                             UINT64_C(0xBF58476D1CE4E5B9)));
            int picked = 0;
            if (admitted > 1 && rng_below(&root_rng, 2) != 0)
                picked = 1 + (int)rng_below(
                    &root_rng, (uint32_t)(admitted - 1));
            if (picked == 0) j->continuation_baseline_roots++;
            else j->continuation_challenger_roots++;
            Move root_move = mv[order[picked]];
            j->continuation_fingerprint ^=
                mix64((episode + 1) * UINT64_C(0xD6E8FEB86659FD93)
                      ^ (uint64_t)MOVE_PACK(root_move)
                      ^ ((uint64_t)admitted << 16));
            Rng determinization_rng;
            rng_seed(&determinization_rng,
                     mix64(j->seed ^ UINT64_C(0x13198A2E03707344)
                           ^ (episode + 1) *
                             UINT64_C(0x94D049BB133111EB)));
            State world;
            /* Root admission is a property of the original information
             * state.  Only after selecting that same legal candidate do we
             * sample the uniform hidden world used for its continuation. */
            determinize(&root_view, root_player, &determinization_rng, &world);
            /* This is the action whose continuations rollout compares.  It is
             * intentionally absent from chain[] and can receive no gradient. */
            lc_apply(&world, root_move);

            int T = 0;
            RolloutLateCycleHistory cycle_history;
            rollout_late_cycle_init(&cycle_history);
            Rng behavior_rng;
            rng_seed(&behavior_rng,
                     mix64(j->seed ^ UINT64_C(0x082EFA98EC4E6C89)
                           ^ (episode + 1) *
                             UINT64_C(0x8CB92BA72F3D8DD7)));
            while (!world.over) {
                if (world.deck_left == 1) {
                    int n = lc_moves(&world, mv);
                    int exact = rollout_exact_terminal_choice(
                        &world, mv, NULL, n, j->continuation_objective,
                        NULL);
                    if (exact < 0) {
                        fprintf(stderr,
                                "continuation exact deck-one solver failed\n");
                        exit(EXIT_FAILURE);
                    }
                    lc_apply(&world, mv[exact]);
                    j->continuation_exact_moves++;
                    continue;
                }
                int force_cycle_deck =
                    rollout_late_cycle_repeated(&cycle_history, &world);
                int force_cap_reserve =
                    (int)world.nply + (int)world.deck_left >= LC_MAX_PLIES;
                if (force_cycle_deck || force_cap_reserve) {
                    /* Match rollout.c's engine-fuse reserve.  An unfinished
                     * 300-ply score is not a game objective and must never
                     * become a PPO target.  This conditional policy action is
                     * forced progress, so it receives no actor row. */
                    int n;
                    const Move *forced_mv = mv;
                    if (augment) {
                        State forced_view;
                        const uint8_t *turn_perm = role_perm[world.turn];
                        n = continuation_trajectory_policy_probs_plan(
                            j->net, &learner_plan, &world,
                            turn_perm, j->temp,
                            &forced_view, mv, engine_mv, pr, behavior);
                        forced_mv = engine_mv;
                    } else {
                        n = policy_probs_sym_plan(
                            j->net, &world, mv, pr, NULL, 1,
                            &learner_plan);
                        if (n > 0)
                            n = continuation_condition_policy(
                                &world, mv, n, j->temp, pr, behavior,
                                NULL, NULL);
                    }
                    uint64_t dead =
                        lc_dead_cards(&world) & world.hand[world.turn];
                    int deck = rollout_policy_deck_choice(
                        &world, forced_mv, pr, n, dead);
                    if (n <= 0 || deck < 0) {
                        fprintf(stderr,
                                "continuation forced-progress policy failed\n");
                        exit(EXIT_FAILURE);
                    }
                    lc_apply(&world, forced_mv[deck]);
                    if (force_cycle_deck)
                        j->continuation_cycle_forces++;
                    else
                        j->continuation_cap_forces++;
                    continue;
                }
                if (T >= CHAIN_MAX) {
                    fprintf(stderr, "continuation trajectory is too long\n");
                    exit(EXIT_FAILURE);
                }

                for (int p = 0; p < 2; p++) {
                    if (augment)
                        lc_permute_suits(
                            &world, &continuation_chain[p][T], role_perm[p]);
                    else
                        continuation_chain[p][T] = world;
                }
                int n;
                Move played, stored_played;
                if (augment) {
                    int actor = world.turn;
                    n = continuation_trajectory_policy_probs_plan(
                        j->net, &learner_plan, &world,
                        role_perm[actor], j->temp,
                        &continuation_chain[actor][T], mv, engine_mv,
                        pr, behavior);
                    if (n <= 0) {
                        fprintf(stderr,
                                "continuation trajectory policy has no legal move\n");
                        exit(EXIT_FAILURE);
                    }
                } else {
                    n = policy_probs_sym_plan(
                        j->net, &world, mv, pr, NULL, 1, &learner_plan);
                    if (n <= 0) {
                        fprintf(stderr,
                                "continuation learner has no legal move\n");
                        exit(EXIT_FAILURE);
                    }
                    n = continuation_condition_policy(
                        &world, mv, n, j->temp, pr, behavior, NULL, NULL);
                    if (n <= 0) {
                        fprintf(stderr,
                                "continuation learner conditional policy failed\n");
                        exit(EXIT_FAILURE);
                    }
                }

                double h = 0.0;
                for (int i = 0; i < n; i++)
                    if (pr[i] > 1e-9f) h -= pr[i] * log(pr[i]);
                j->entropy += h;
                j->entropy_n++;

                int c;
                c = sample_index(behavior, n, &behavior_rng);
                chain_p[T] = behavior[c];
                if (augment) {
                    stored_played = mv[c];
                    played = engine_mv[c];
                } else {
                    stored_played = mv[c];
                    played = mv[c];
                }
                chain_actor[T] = 1;
                chain_mv[T] = MOVE_PACK(stored_played);
                int actor_ply = world.nply;
                if (j->continuation_min_actor_ply == 0 ||
                    actor_ply < j->continuation_min_actor_ply)
                    j->continuation_min_actor_ply = actor_ply;
                T++;
                lc_apply(&world, played);
            }
            if (world.deck_left > 0) {
                fprintf(stderr,
                        "continuation generated an unfinished capped round\n");
                exit(EXIT_FAILURE);
            }

            /* Reconstruct each player's critic in the same fixed suit role
             * used by that player's continuation policy.  Mode 2 changes
             * only real round index 2 to the deployed final-match hybrid;
             * rollout_terminal_objective itself preserves round-margin
             * semantics in rounds 0 and 1.  Continuation mode requires
             * lambda=1, so the critic remains only an action-independent
             * advantage baseline and cannot alter the terminal target. */
            for (int t = 0; t < T; t++) {
                float v0 = net_value_state_sym(
                    j->net, &continuation_chain[0][t], 0, 1);
                float v1 = net_value_state_sym(
                    j->net, &continuation_chain[1][t], 1, 1);
                chain_v[0][t] = 0.5f * (v0 - v1);
                chain_v[1][t] = -chain_v[0][t];
            }
            for (int p = 0; p < 2; p++) {
                float G = (float)rollout_terminal_objective(
                    &world, p, j->continuation_objective);
                uint8_t other_role_perm[NSUIT];
                if (!continuation_relative_role_permutation(
                        role_perm[p], role_perm[p ^ 1], other_role_perm)) {
                    fprintf(stderr,
                            "invalid continuation relative role mapping\n");
                    exit(EXIT_FAILURE);
                }
                for (int t = T - 1; t >= 0; t--) {
                    size_t slot = reservoir_slot(j, &reservoir_rng);
                    if (slot == SIZE_MAX) continue;
                    RLSample *s = &j->out[slot];
                    s->st = continuation_chain[p][t];
                    s->persp = (uint8_t)p;
                    s->vtarget = G;
                    s->continuation_role = 1;
                    memcpy(s->other_role_perm, other_role_perm, NSUIT);
                    if (continuation_chain[p][t].turn == p) {
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

            int rs0 = lc_score(&world, 0);
            int rs1 = lc_score(&world, 1);
            cum[0] += rs0;
            cum[1] += rs1;
            j->plies += world.nply;
            j->continuation_rounds++;
        }

        if (cum[0] > cum[1]) j->p0_match_wins += 1.0;
        else if (cum[0] == cum[1]) j->p0_match_wins += 0.5;
        j->absmargin += fabs((double)(cum[0] - cum[1]));
        j->score_sum += cum[0] + cum[1];
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
    uint8_t allowed[MAX_MOVES];
    uint8_t bcard[NCARD], held[NCARD];
    float blogit[NCARD], bmarg[NCARD], dbel[NCARD];

    for (int i = t->from; i < t->to; i++) {
        const RLSample *s = &t->buf[t->idx[i]];
        feat_extract(&s->st, s->persp, &f);
        net_trunk(t->net, &f, &act);
        float dcenter = 0.0f;
        if (!t->belief_only) {
            const State *other_role_state = &s->st;
            State reconstructed_other_role;
            if (s->continuation_role) {
                if (!continuation_role_permutation_valid(
                        s->other_role_perm)) {
                    fprintf(stderr,
                            "continuation PPO sample has invalid relative "
                            "role mapping\n");
                    exit(EXIT_FAILURE);
                }
                lc_permute_suits(
                    &s->st, &reconstructed_other_role,
                    s->other_role_perm);
                other_role_state = &reconstructed_other_role;
            }
            feat_extract(other_role_state, s->persp ^ 1, &of);
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
                allowed[k] = 1;
            }
            if (s->continuation_role) {
                uint64_t dead =
                    lc_dead_cards(&s->st) & s->st.hand[s->st.turn];
                for (int k = 0; k < n; k++)
                    allowed[k] = (uint8_t)!(
                        dead && lc_discard_dominated(&s->st, mv[k], dead));
                if (ci >= 0 && !allowed[ci]) {
                    fprintf(stderr,
                            "continuation PPO sample selected a masked move\n");
                    exit(EXIT_FAILURE);
                }
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
            int first = 0;
            if (s->continuation_role) {
                while (first < n && !allowed[first]) first++;
                if (first >= n) {
                    fprintf(stderr,
                            "continuation PPO sample has empty support\n");
                    exit(EXIT_FAILURE);
                }
            }
            float mx = logit[first];
            for (int k = first + 1; k < n; k++)
                if ((!s->continuation_role || allowed[k]) && logit[k] > mx)
                    mx = logit[k];
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                if (s->continuation_role && !allowed[k]) {
                    prob[k] = 0.0f;
                    continue;
                }
                prob[k] = expf(logit[k] - mx);
                sum += prob[k];
            }
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
                if (s->continuation_role && !allowed[k]) {
                    /* Deployment assigns this move zero behavior mass.  PPO
                     * and entropy therefore have no gradient here; the
                     * full-legal anchor KL below may still protect it. */
                    dlog[k] = 0.0f;
                    continue;
                }
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
    const char *continuation_root_path = NULL;
    int iters = 30, games = 4000, nthread = 4, batch = 512, epochs = 2;
    int eval_pairs = 400, eval_every = 1;
    float lr = 3e-4f, wd = 1e-7f, lambda = 0.85f, clip = 0.2f;
    float vcoef = 1.0f, entcoef = 0.004f, temp = 1.0f;
    float winbonus = 15.0f, bw = 1.0f, mw = 1.0f;
    float opponent_mix = 0.0f, klcoef = 0.0f;
    int v6_only = 0, belief_only = 0;
    int rounds = MATCH_ROUNDS, trajectory_symmetries = 0;
    int continuation_start = 0;
    int continuation_objective = 0;
    int continuation_independent_roles = 0;
    int continuation_objective_explicit = 0;
    int continuation_roles_explicit = 0;
    int bw_explicit = 0, eval_explicit = 0, lambda_explicit = 0;
    int winbonus_explicit = 0, mw_explicit = 0;
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
        else if (ARG("--continuation-root"))
            continuation_root_path = argv[++i];
        else if (ARG("--kl")) klcoef = (float)atof(argv[++i]);
        else if (ARG("--iters")) iters = atoi(argv[++i]);
        else if (ARG("--games")) games = atoi(argv[++i]);
        else if (ARG("--threads")) nthread = atoi(argv[++i]);
        else if (ARG("--batch")) batch = atoi(argv[++i]);
        else if (ARG("--epochs")) epochs = atoi(argv[++i]);
        else if (ARG("--lr")) lr = (float)atof(argv[++i]);
        else if (ARG("--lambda")) {
            lambda = (float)atof(argv[++i]);
            lambda_explicit = 1;
        }
        else if (ARG("--clip")) clip = (float)atof(argv[++i]);
        else if (ARG("--vcoef")) vcoef = (float)atof(argv[++i]);
        else if (ARG("--ent")) entcoef = (float)atof(argv[++i]);
        else if (ARG("--temp")) temp = (float)atof(argv[++i]);
        else if (ARG("--winbonus")) {
            winbonus = (float)atof(argv[++i]);
            winbonus_explicit = 1;
        }
        else if (ARG("--bw")) {
            bw = (float)atof(argv[++i]);
            bw_explicit = 1;
        }
        else if (ARG("--mw")) {
            mw = (float)atof(argv[++i]);
            mw_explicit = 1;
        }
        else if (ARG("--rounds")) rounds = atoi(argv[++i]);
        else if (ARG("--wd")) wd = (float)atof(argv[++i]);
        else if (ARG("--eval")) {
            eval_pairs = atoi(argv[++i]);
            eval_explicit = 1;
        }
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
        else if (ARG("--continuation-start"))
            continuation_start = atoi(argv[++i]);
        else if (!strcmp(k, "--continuation-objective")) {
            if (++i >= argc ||
                (strcmp(argv[i], "0") != 0 && strcmp(argv[i], "2") != 0)) {
                fprintf(stderr,
                        "--continuation-objective must be exactly 0 or 2\n");
                return 1;
            }
            continuation_objective = argv[i][0] - '0';
            continuation_objective_explicit = 1;
        }
        else if (!strcmp(k, "--continuation-role-mappings")) {
            if (++i >= argc ||
                (strcmp(argv[i], "shared") != 0 &&
                 strcmp(argv[i], "independent") != 0)) {
                fprintf(stderr,
                        "--continuation-role-mappings must be exactly "
                        "shared or independent\n");
                return 1;
            }
            continuation_independent_roles =
                strcmp(argv[i], "independent") == 0;
            continuation_roles_explicit = 1;
        }
        else if (!strcmp(k, "--v6-only")) v6_only = 1;
        else if (!strcmp(k, "--belief-only")) belief_only = 1;
        else { fprintf(stderr, "unknown option %s\n", k); return 1; }
        #undef ARG
    }
    if (continuation_start != 0 &&
        continuation_start != CONTINUATION_START_PLY) {
        fprintf(stderr, "--continuation-start must be 0 or %d\n",
                CONTINUATION_START_PLY);
        return 1;
    }
    if (!lc_float_isfinite(lambda) || lambda < 0.0f || lambda > 1.0f) {
        fprintf(stderr, "--lambda must be finite and between zero and one\n");
        return 1;
    }
    if (continuation_start > 0) {
        /* The first campaign learns only the rolloutu continuation role.
         * Uniform worlds intentionally contain no behavioural hand signal,
         * and standalone-policy checkpoint evaluation is not the deployed
         * dual-network actor. */
        if (!bw_explicit) bw = 0.0f;
        if (!eval_explicit) eval_pairs = 0;
        if (!lambda_explicit) lambda = 1.0f;
        if (lambda != 1.0f) {
            fprintf(stderr,
                    "continuation-only PPO requires --lambda 1 exactly\n");
            return 1;
        }
        if (bw > 0.0f || belief_only) {
            fprintf(stderr,
                    "continuation-only PPO requires --bw 0 and cannot use "
                    "--belief-only\n");
            return 1;
        }
        if (gen_opponent_spec || opponent_mix > 0.0f) {
            fprintf(stderr,
                    "continuation-only PPO cannot use opponent-population "
                    "generation\n");
            return 1;
        }
        if (!continuation_root_path) {
            fprintf(stderr,
                    "--continuation-start 14 requires --continuation-root "
                    "PATH\n");
            return 1;
        }
        if (eval_pairs > 0) {
            fprintf(stderr,
                    "continuation-only PPO requires --eval 0; qualify the "
                    "checkpoint as a rollout2 actor\n");
            return 1;
        }
        if (winbonus_explicit || mw_explicit) {
            fprintf(stderr,
                    "continuation-only PPO does not use --winbonus or --mw; "
                    "configure its terminal return with "
                    "--continuation-objective 0 or 2\n");
            return 1;
        }
        if (continuation_objective == 2 && rounds != MATCH_ROUNDS) {
            fprintf(stderr,
                    "--continuation-objective 2 requires --rounds %d so "
                    "the final-round target is present\n",
                    MATCH_ROUNDS);
            return 1;
        }
        if (continuation_independent_roles &&
            trajectory_symmetries <= 1) {
            fprintf(stderr,
                    "independent continuation role mappings require "
                    "--trajectory-symmetries 5, 10, 20, or 120\n");
            return 1;
        }
    } else {
        if (continuation_root_path) {
            fprintf(stderr,
                    "--continuation-root requires --continuation-start 14\n");
            return 1;
        }
        if (continuation_objective_explicit) {
            fprintf(stderr,
                    "--continuation-objective requires "
                    "--continuation-start 14\n");
            return 1;
        }
        if (continuation_roles_explicit) {
            fprintf(stderr,
                    "--continuation-role-mappings requires "
                    "--continuation-start 14\n");
            return 1;
        }
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
    if (continuation_start > 0 &&
        !continuation_checkpoint_preflight(
            out_path, iters, init_path, continuation_root_path, anchor_path))
        return 1;

    Net *net = (Net *)malloc(sizeof(Net));
    Net *frozen = (Net *)malloc(sizeof(Net));
    Net *champion = continuation_start > 0
                  ? (Net *)malloc(sizeof(Net)) : NULL;
    Net *anchor = anchor_path ? (Net *)malloc(sizeof(Net)) : NULL;
    Net *legacy_base = v6_only ? (Net *)malloc(sizeof(Net)) : NULL;
    Adam *adam = (Adam *)calloc(1, sizeof(Adam));
    if (!net || !frozen || (continuation_start > 0 && !champion) ||
        (anchor_path && !anchor) ||
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
    if (champion) {
        if (net_load(champion, continuation_root_path) != 0) {
            fprintf(stderr, "cannot load continuation root %s\n",
                    continuation_root_path);
            return 1;
        }
    }
    uint64_t champion_fingerprint = champion
        ? net_byte_fingerprint(champion) : 0;
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

    if (continuation_start > 0)
        printf("ppo: %d iters x %d matches of %d round(s), batch %d, %d "
               "epochs, lr %.1e, lambda %.2f, ent %.4f, continuation "
               "objective %d, KL %.4f%s\n",
               iters, games, rounds, batch, epochs, lr, lambda, entcoef,
               continuation_objective, klcoef,
               v6_only ? ", v6-only" : "");
    else
        printf("ppo: %d iters x %d matches of %d round(s), batch %d, %d "
               "epochs, lr %.1e, lambda %.2f, ent %.4f, winbonus %.0f, "
               "margin weight %.2f, KL %.4f%s\n",
               iters, games, rounds, batch, epochs, lr, lambda, entcoef,
               winbonus, mw, klcoef, v6_only ? ", v6-only" : "");
    if (gen_opponent_ptr)
        printf("     opponent population: %.1f%% %s, %.1f%% live self-play\n",
               100.0f * opponent_mix, gen_opponent_spec,
               100.0f * (1.0f - opponent_mix));
    if (trajectory_symmetries > 0 &&
        !(continuation_start > 0 && continuation_independent_roles))
        printf("     trajectory suit augmentation: exact group %d, one fixed "
               "mapping per match\n", trajectory_symmetries);
    if (continuation_start > 0 && continuation_independent_roles)
        printf("     continuation suit roles: independent ordered-product "
               "group %d, one fixed mapping per player per round tail\n",
               trajectory_symmetries);
    if (belief_only)
        printf("     optimizer: belief head only (trunk, policy and value frozen)\n");
    if (continuation_start > 0)
        printf("     continuation-only state start: immutable champion %s; "
               "plies "
               "0..13, production root at ply 14 (20-way, width 5, floor "
               "%.2f), 50%% baseline / 50%% admitted challenger, uniform "
               "worlds, objective mode %d (%s); suit roles %s\n"
               "     immutable champion fingerprint: %016llx\n",
               continuation_root_path, CONTINUATION_ROOT_FLOOR,
               continuation_objective,
               continuation_objective == 2
                   ? "rounds 0/1 margin; round 2 "
                     "0.05*final-margin+50*result"
                   : "round margin in every round",
               continuation_independent_roles
                   ? "independent per-player ordered product"
                   : "shared legacy mapping",
               (unsigned long long)champion_fingerprint);
    fflush(stdout);

    for (int it = 1; it <= iters; it++) {
        struct timespec t0, t1;
        clock_gettime(CLOCK_MONOTONIC, &t0);
        memcpy(frozen, net, sizeof(Net));

        GenJob *jobs = (GenJob *)calloc((size_t)nthread, sizeof(GenJob));
        pthread_t *th = (pthread_t *)calloc((size_t)nthread, sizeof(pthread_t));
        size_t role_cells = continuation_independent_roles
            ? (size_t)trajectory_symmetries *
              (size_t)trajectory_symmetries : 0;
        uint32_t *role_counts = role_cells > 0
            ? (uint32_t *)calloc(
                (size_t)nthread * role_cells, sizeof(uint32_t)) : NULL;
        if (!jobs || !th || (role_cells > 0 && !role_counts)) {
            fprintf(stderr, "generation diagnostic allocation failed\n");
            free(jobs); free(th); free(role_counts);
            return 1;
        }
        size_t per = cap / (size_t)nthread;
        for (int i = 0; i < nthread; i++) {
            jobs[i].net = frozen;
            jobs[i].champion = champion;
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
            jobs[i].continuation_start = continuation_start;
            jobs[i].continuation_objective = continuation_objective;
            jobs[i].continuation_independent_roles =
                continuation_independent_roles;
            jobs[i].continuation_role_group = trajectory_symmetries;
            jobs[i].continuation_role_pair_count = role_cells > 0
                ? role_counts + (size_t)i * role_cells : NULL;
        }
        void *(*generator)(void *) = continuation_start > 0
                                   ? gen_continuation_worker : gen_worker;
        for (int i = 0; i < nthread; i++)
            pthread_create(&th[i], NULL, generator, &jobs[i]);
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
        int continuation_rounds = 0;
        int continuation_baseline_roots = 0;
        int continuation_challenger_roots = 0;
        int continuation_exact_moves = 0;
        int continuation_cycle_forces = 0;
        int continuation_cap_forces = 0;
        int continuation_min_actor_ply = 0;
        uint64_t augmentation_fingerprint = 0;
        uint64_t continuation_fingerprint = 0;
        uint64_t role_pair_min = 0, role_pair_max = 0;
        uint64_t role_first_min = 0, role_first_max = 0;
        uint64_t role_other_min = 0, role_other_max = 0;
        uint64_t role_tail_count = 0;
        int learner_seat_games[2] = {0, 0};
        for (int i = 0; i < nthread; i++) {
            plies += jobs[i].plies; absm += jobs[i].absmargin; pts += jobs[i].score_sum;
            ent += jobs[i].entropy; entn += jobs[i].entropy_n;
            p0w += jobs[i].p0_match_wins;
            learnerw += jobs[i].learner_match_wins;
            opponent_games += jobs[i].opponent_games;
            augmentation_fingerprint ^= jobs[i].augmentation_fingerprint;
            continuation_fingerprint ^= jobs[i].continuation_fingerprint;
            continuation_rounds += jobs[i].continuation_rounds;
            continuation_baseline_roots +=
                jobs[i].continuation_baseline_roots;
            continuation_challenger_roots +=
                jobs[i].continuation_challenger_roots;
            continuation_exact_moves += jobs[i].continuation_exact_moves;
            continuation_cycle_forces += jobs[i].continuation_cycle_forces;
            continuation_cap_forces += jobs[i].continuation_cap_forces;
            if (jobs[i].continuation_min_actor_ply > 0 &&
                (continuation_min_actor_ply == 0 ||
                 jobs[i].continuation_min_actor_ply <
                    continuation_min_actor_ply))
                continuation_min_actor_ply =
                    jobs[i].continuation_min_actor_ply;
            learner_seat_games[0] += jobs[i].learner_seat_games[0];
            learner_seat_games[1] += jobs[i].learner_seat_games[1];
            seen += jobs[i].nseen;
            gdone += jobs[i].done;
        }
        if (continuation_independent_roles) {
            uint64_t first_marginal[120] = { 0 };
            uint64_t other_marginal[120] = { 0 };
            int group = trajectory_symmetries;
            role_pair_min = UINT64_MAX;
            for (int first = 0; first < group; first++) {
                for (int other = 0; other < group; other++) {
                    size_t cell = (size_t)first * (size_t)group
                                + (size_t)other;
                    uint64_t count = 0;
                    for (int i = 0; i < nthread; i++)
                        count += role_counts[(size_t)i * role_cells + cell];
                    if (count < role_pair_min) role_pair_min = count;
                    if (count > role_pair_max) role_pair_max = count;
                    first_marginal[first] += count;
                    other_marginal[other] += count;
                    role_tail_count += count;
                }
            }
            role_first_min = role_other_min = UINT64_MAX;
            for (int k = 0; k < group; k++) {
                if (first_marginal[k] < role_first_min)
                    role_first_min = first_marginal[k];
                if (first_marginal[k] > role_first_max)
                    role_first_max = first_marginal[k];
                if (other_marginal[k] < role_other_min)
                    role_other_min = other_marginal[k];
                if (other_marginal[k] > role_other_max)
                    role_other_max = other_marginal[k];
            }
            if (role_tail_count != (uint64_t)continuation_rounds ||
                role_pair_max > role_pair_min + UINT64_C(1) ||
                role_first_max > role_first_min + UINT64_C(1) ||
                role_other_max > role_other_min + UINT64_C(1)) {
                fprintf(stderr,
                        "continuation role product diagnostics failed: "
                        "tails %llu/%d pair %llu..%llu marginals "
                        "%llu..%llu/%llu..%llu\n",
                        (unsigned long long)role_tail_count,
                        continuation_rounds,
                        (unsigned long long)role_pair_min,
                        (unsigned long long)role_pair_max,
                        (unsigned long long)role_first_min,
                        (unsigned long long)role_first_max,
                        (unsigned long long)role_other_min,
                        (unsigned long long)role_other_max);
                free(jobs); free(th); free(role_counts);
                return 1;
            }
        }
        free(jobs); free(th); free(role_counts);

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
        if (continuation_start > 0)
            printf("         continuation roots %d: baseline %d, challenger "
                   "%d; first actor round ply %d; root actor rows 0; exact "
                   "deck-one moves %d; cycle-forced moves %d; "
                   "cap-reserve moves %d; fingerprint "
                   "%016llx\n",
                   continuation_rounds, continuation_baseline_roots,
                   continuation_challenger_roots,
                   continuation_min_actor_ply, continuation_exact_moves,
                   continuation_cycle_forces,
                   continuation_cap_forces,
                   (unsigned long long)continuation_fingerprint);
        if (continuation_independent_roles)
            printf("         continuation role product group %d: tails "
                   "%llu; pair-count %llu..%llu; root/opponent marginals "
                   "%llu..%llu/%llu..%llu\n",
                   trajectory_symmetries,
                   (unsigned long long)role_tail_count,
                   (unsigned long long)role_pair_min,
                   (unsigned long long)role_pair_max,
                   (unsigned long long)role_first_min,
                   (unsigned long long)role_first_max,
                   (unsigned long long)role_other_min,
                   (unsigned long long)role_other_max);
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
                   steps ? sqrt(vl / ((double)steps * batch)) * VAL_SCALE
                         : 0.0,
                   pn ? pl / pn : 0.0, bcnt ? bl / bcnt : 0.0,
                   pn ? kl / pn : 0.0,
                   pn ? 100.0 * cl / pn : 0.0);
        if (champion) {
            uint64_t after = net_byte_fingerprint(champion);
            if (after != champion_fingerprint) {
                fprintf(stderr,
                        "immutable continuation champion was modified\n");
                return 1;
            }
            printf("         immutable champion verified %016llx\n",
                   (unsigned long long)after);
        }
        fflush(stdout);

        char path[PATH_MAX];
        net_save(net, out_path);
        int path_n = snprintf(path, sizeof path, "%s.it%d", out_path, it);
        if (path_n < 0 || (size_t)path_n >= sizeof path) {
            fprintf(stderr, "generated checkpoint path is too long\n");
            return 1;
        }
        net_save(net, path);

        if (eval_pairs > 0 && (it % eval_every == 0 || it == iters)) {
            Agent cur;
            agent_default(&cur, AG_POLICY, net);
            MatchResult mr;
            if (match_run_r(&cur, &ref, eval_pairs, nthread, eval_seed,
                            rounds, &mr) != 0) {
                fprintf(stderr, "checkpoint evaluation match failed\n");
                return 1;
            }
            printf("         vs %s: margin %+.2f +- %.2f, match wins %.1f%%, plies %.0f\n",
                   ref_spec, mr.margin, mr.margin_se, 100 * mr.winrate, mr.plies);
            fflush(stdout);
        }
    }
    spec_release(&ref);
    if (gen_opponent_ptr) spec_release(&gen_opponent);
    for (int i = 0; i < nthread; i++) free(grads[i]);
    free(grads);
    free(order);
    free(buf);
    free(adam);
    free(legacy_base);
    free(anchor);
    free(champion);
    free(frozen);
    free(net);
    return 0;
}
