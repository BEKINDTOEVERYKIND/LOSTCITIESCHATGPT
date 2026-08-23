#include "spec.h"
#include "match_value.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_NAMES 16
static char g_names[MAX_NAMES][512];
static int g_nname = 0;

static Net *load_net(const char *path)
{
    Net *n = (Net *)malloc(sizeof(Net));
    if (!n) { fprintf(stderr, "out of memory\n"); exit(1); }
    int r = net_load(n, path);
    if (r != 0) { fprintf(stderr, "cannot load net '%s' (error %d)\n", path, r); exit(1); }
    return n;
}

enum { ROLLOUT_TAIL_FIELDS = 42 };

static int valid_suit_group(int n)
{
    return n == 1 || n == 5 || n == 10 || n == 20 || n == 120;
}

static int match_value_matches_rollout(const Agent *a)
{
    if (!a) return 0;
    if (a->win_q != 3)
        return a->match_value == NULL && !a->owns_match_value;
    if (!a->match_value || !a->continuation_net ||
        !match_value_validate(a->match_value) ||
        !match_value_balanced_roles(a->match_value) ||
        a->veto_continuation_net ||
        a->bounded_late_root ||
        a->deck2_replan_worlds != 0 || a->deck2_replan_cores != 0)
        return 0;
    const MatchValueController *c = &a->match_value->controller;
    int playout_prune = a->playout_prune < 0
        ? a->prune_dom : a->playout_prune != 0;
    return c->net_fingerprint ==
               match_value_net_fingerprint(a->continuation_net) &&
           c->playout_symmetries == (uint32_t)a->playout_symmetries &&
           c->playout_sample == (uint32_t)a->playout_sample &&
           c->playout_prune == (uint32_t)playout_prune &&
           c->exact_terminal == (uint32_t)a->exact_terminal &&
           c->plan_deck_max == (uint32_t)a->plan_deck_max &&
           c->plan_block_gap == (uint32_t)a->plan_block_gap &&
           c->draw_playout_deck_max ==
               (uint32_t)a->draw_playout_deck_max &&
           c->deck2_replan_worlds ==
               (uint32_t)a->deck2_replan_worlds &&
           c->deck2_replan_cores == (uint32_t)a->deck2_replan_cores &&
           c->max_plies == LC_MAX_PLIES;
}

static void validate_rollout(const Agent *a, const char *label)
{
    int valid =
        a->dets >= 1 && a->dets <= 1000000 &&
        a->root_width >= 1 && a->root_width <= MAX_MOVES &&
        isfinite(a->cand_floor) && a->cand_floor >= 0.0f &&
        a->cand_floor <= 1.0f &&
        isfinite(a->gate) && a->gate >= 0.0f && a->gate <= 1.0f &&
        a->min_cand >= 0 && a->min_cand <= MAX_MOVES &&
        a->ply_lo >= 0 && a->ply_lo <= 300 &&
        a->ply_hi >= 0 && a->ply_hi <= 300 &&
        (a->ply_hi == 0 || a->ply_hi >= a->ply_lo) &&
        a->eval_cand >= 0 && a->eval_cand <= MAX_MOVES &&
        a->win_q >= 0 && a->win_q <= 3 &&
        (a->prune_dom == 0 || a->prune_dom == 1) &&
        isfinite(a->override_k) && a->override_k >= 0.0f &&
        isfinite(a->override_min) && a->override_min >= 0.0f &&
        a->playout_sample >= 0 && a->playout_sample <= 4 &&
        valid_suit_group(a->symmetries) &&
        isfinite(a->cand_mass) && a->cand_mass >= 0.0f &&
        a->cand_mass <= 1.0f &&
        a->batch_dets >= 0 && a->batch_dets <= a->dets &&
        valid_suit_group(a->playout_symmetries) &&
        (a->discard_guard == 0 || a->discard_guard == 1) &&
        a->deck_max >= 0 && a->deck_max <= NCARD &&
        a->confirm_dets >= 0 && a->confirm_dets <= 1000000 &&
        a->playout_prune >= -1 && a->playout_prune <= 1 &&
        a->plan_deck_max >= 0 && a->plan_deck_max <= NCARD &&
        a->plan_block_gap >= 0 && a->plan_block_gap <= 1000000 &&
        (a->semantic_cand == 0 || a->semantic_cand == 1) &&
        (a->confirm_exact5 == 0 || a->confirm_exact5 == 1) &&
        a->draw_variant_cores >= 0 && a->draw_variant_cores <= 2 &&
        a->draw_variant_deck_max >= 0 &&
        a->draw_variant_deck_max <= NCARD &&
        a->policy_prefix_mode >= 0 && a->policy_prefix_mode <= 3 &&
        isfinite(a->prefix_confirm_k) && a->prefix_confirm_k >= 0.0f &&
        isfinite(a->prefix_confirm_min) && a->prefix_confirm_min >= 0.0f &&
        ((a->prefix_confirm_k == 0.0f &&
          a->prefix_confirm_min == 0.0f) ||
         (a->prefix_confirm_k > 0.0f &&
          a->prefix_confirm_min > 0.0f)) &&
        isfinite(a->belief_alpha) && a->belief_alpha >= 0.0f &&
        a->belief_alpha <= 5.0f &&
        a->draw_root_deck_max >= 0 &&
        a->draw_root_deck_max <= NCARD &&
        a->draw_playout_deck_max >= 0 &&
        a->draw_playout_deck_max <= NCARD &&
        isfinite(a->confirm_temp) && a->confirm_temp >= 0.0f &&
        a->confirm_temp <= 5.0f &&
        a->action_core_count >= 0 && a->action_core_count <= 5 &&
        a->exact_terminal >= 0 && a->exact_terminal <= 3 &&
        a->deck2_replan_worlds >= 0 && a->deck2_replan_worlds <= 4096 &&
        a->deck2_replan_cores >= 0 && a->deck2_replan_cores <= 3 &&
        ((a->deck2_replan_worlds == 0 && a->deck2_replan_cores == 0) ||
         (a->deck2_replan_worlds > 0 && a->deck2_replan_cores > 0 &&
          a->exact_terminal == 1 && a->no_belief)) &&
        (a->bounded_late_root == 0 || a->bounded_late_root == 1) &&
        isfinite(a->bounded_late_min) && a->bounded_late_min >= 0.0f &&
        a->bounded_late_min <= 1000.0f &&
        lc_float_isfinite(a->action_ranker_min) &&
        a->action_ranker_min >= 0.0f &&
        a->action_ranker_min <= 1000.0f &&
        (!a->veto_continuation_net ||
         (a->net && a->continuation_net && a->policy_prefix_mode >= 2 &&
          !a->action_ranker_net)) &&
        (!a->action_ranker_net ||
         (a->net && a->continuation_net && a->policy_prefix_mode >= 2 &&
          !a->veto_continuation_net)) &&
        (a->action_ranker_net || a->action_ranker_min == 0.0f) &&
        match_value_matches_rollout(a) &&
        (!a->bounded_late_root ||
         (a->exact_terminal == 1 && a->no_belief &&
          a->deck2_replan_worlds == 0 && a->deck2_replan_cores == 0 &&
          a->plan_deck_max == 0 && a->plan_block_gap == 0 &&
          a->draw_root_deck_max == 0));
    if (!valid) {
        fprintf(stderr, "agent '%s' has an invalid rollout configuration\n",
                label);
        exit(1);
    }
}

/* Keep rollout:PATH and the live-network selfrollout generator on one field
 * order.  Use an exact-sized buffer: training specs are experiment inputs and
 * silently dropping their late fields can change the generated policy while
 * leaving the command looking valid. */
static void parse_rollout_tail(const char *tail, Agent *a, const char *label)
{
    size_t len = strlen(tail);
    char *buf = (char *)malloc(len + 1);
    if (!buf) { fprintf(stderr, "out of memory\n"); exit(1); }
    memcpy(buf, tail, len + 1);

    char *save = NULL;
    char *v = strtok_r(buf, ":", &save);
    int field = 0;
    while (v) {
        if (field >= ROLLOUT_TAIL_FIELDS) {
            fprintf(stderr, "agent '%s' has unsupported rollout field %d ('%s')\n",
                    label, field + 1, v);
            free(buf);
            exit(1);
        }
        switch (field++) {
        case 0:  a->dets = atoi(v); break;
        case 1:  a->root_width = atoi(v); break;
        case 2:  a->cand_floor = (float)atof(v); break;
        case 3:  a->gate = (float)atof(v); break;
        case 4:  a->min_cand = atoi(v); break;
        case 5:  a->ply_lo = atoi(v); break;
        case 6:  a->ply_hi = atoi(v); break;
        case 7:  a->eval_cand = atoi(v); break;
        case 8:  a->win_q = atoi(v); break;
        case 9:  a->prune_dom = atoi(v); break;
        case 10: a->override_k = (float)atof(v); break;
        case 11: a->override_min = (float)atof(v); break;
        case 12: a->playout_sample = atoi(v); break;
        case 13: a->symmetries = atoi(v); break;
        case 14: a->cand_mass = (float)atof(v); break;
        case 15: a->batch_dets = atoi(v); break;
        case 16: a->playout_symmetries = atoi(v); break;
        case 17: a->discard_guard = atoi(v); break;
        case 18: a->deck_max = atoi(v); break;
        case 19: a->confirm_dets = atoi(v); break;
        case 20: a->playout_prune = atoi(v); break;
        case 21: a->plan_deck_max = atoi(v); break;
        case 22: a->plan_block_gap = atoi(v); break;
        case 23: a->semantic_cand = atoi(v); break;
        case 24: a->confirm_exact5 = atoi(v); break;
        case 25: a->draw_variant_cores = atoi(v); break;
        case 26: a->draw_variant_deck_max = atoi(v); break;
        case 27: a->policy_prefix_mode = atoi(v); break;
        case 28: a->belief_alpha = (float)atof(v); break;
        case 29: a->draw_root_deck_max = atoi(v); break;
        case 30: a->draw_playout_deck_max = atoi(v); break;
        case 31: a->prefix_confirm_k = (float)atof(v); break;
        case 32: a->prefix_confirm_min = (float)atof(v); break;
        case 33: a->confirm_temp = (float)atof(v); break;
        case 34: a->action_core_count = atoi(v); break;
        case 35: a->exact_terminal = atoi(v); break;
        case 36: a->deck2_replan_worlds = atoi(v); break;
        case 37: a->deck2_replan_cores = atoi(v); break;
        case 38: a->bounded_late_root = atoi(v); break;
        case 39: a->bounded_late_min = (float)atof(v); break;
        case 40: a->action_ranker_min = (float)atof(v); break;
        case 41: {
            int error = 0;
            MatchValueTable *table = match_value_load(v, &error);
            if (!table) {
                fprintf(stderr,
                        "agent '%s' cannot load match-value table '%s' "
                        "(error %d)\n", label, v, error);
                free(buf);
                exit(1);
            }
            a->match_value = table;
            a->owns_match_value = 1;
            break;
        }
        }
        v = strtok_r(NULL, ":", &save);
    }
    free(buf);
    if (field > 40 && !a->action_ranker_net && !a->match_value) {
        fprintf(stderr,
                "agent '%s' has action-ranker field 41 without a ranker role\n",
                label);
        exit(1);
    }
    validate_rollout(a, label);
}

void spec_parse_selfrollout(const char *spec, const Net *net, Agent *a)
{
    static const char prefix[] = "selfrollout";
    const size_t npre = sizeof prefix - 1;
    if (strncmp(spec, prefix, npre) != 0 ||
        (spec[npre] != '\0' && spec[npre] != ':')) {
        fprintf(stderr, "invalid live rollout spec '%s'\n", spec);
        exit(1);
    }
    agent_default(a, AG_ROLLOUT, net);
    parse_rollout_tail(spec[npre] == ':' ? spec + npre + 1 : "", a, spec);
    if (a->match_value) {
        fprintf(stderr,
                "live selfrollout cannot use a match-value table: the "
                "artifact is bound to immutable network bytes\n");
        spec_release(a);
        exit(1);
    }
}

void spec_release(Agent *a)
{
    if (!a) return;
    const Net *root = a->net;
    const Net *continuation = a->continuation_net;
    const Net *veto = a->veto_continuation_net;
    const Net *ranker = a->action_ranker_net;
    const MatchValueTable *match_value = a->match_value;
    if (a->owns_action_ranker_net && ranker &&
        ranker != veto && ranker != continuation && ranker != root)
        free((void *)ranker);
    if (a->owns_veto_continuation_net && veto &&
        veto != ranker && veto != continuation && veto != root)
        free((void *)veto);
    if (a->owns_continuation_net && continuation && continuation != root)
        free((void *)continuation);
    if (a->owns_net && root) free((void *)root);
    if (a->owns_match_value && match_value)
        match_value_free((MatchValueTable *)match_value);
    a->net = NULL;
    a->continuation_net = NULL;
    a->veto_continuation_net = NULL;
    a->action_ranker_net = NULL;
    a->match_value = NULL;
    a->owns_net = 0;
    a->owns_continuation_net = 0;
    a->owns_veto_continuation_net = 0;
    a->owns_action_ranker_net = 0;
    a->owns_match_value = 0;
}

void spec_parse(const char *spec, Agent *a)
{
    char buf[512];
    int copied = snprintf(buf, sizeof buf, "%s", spec);
    if (copied < 0 || (size_t)copied >= sizeof buf) {
        fprintf(stderr, "agent spec is too long (%zu bytes; maximum %zu)\n",
                strlen(spec), sizeof buf - 1);
        exit(1);
    }
    char *save = NULL;
    char *tok = strtok_r(buf, ":", &save);
    if (!tok) { fprintf(stderr, "empty agent spec\n"); exit(1); }
    if (!strcmp(tok, "random")) { agent_default(a, AG_RANDOM, NULL); return; }
    if (!strcmp(tok, "heur"))   { agent_default(a, AG_HEUR, NULL); return; }
    if (!strcmp(tok, "hrollout")) {
        /* classical baseline: hand-crafted evaluation with perfect-information
         * Monte Carlo over sampled worlds, no network anywhere */
        agent_default(a, AG_ROLLOUT, NULL);
        a->name = "hrollout";
        char *v;
        if ((v = strtok_r(NULL, ":", &save))) a->dets = atoi(v);
        if ((v = strtok_r(NULL, ":", &save))) a->root_width = atoi(v);
        return;
    }
    if (!strcmp(tok, "net") || !strcmp(tok, "mcts") || !strcmp(tok, "policy") ||
        !strcmp(tok, "rollout") || !strcmp(tok, "rolloutu") ||
        !strcmp(tok, "rollout2") || !strcmp(tok, "rolloutu2") ||
        !strcmp(tok, "rollout3") || !strcmp(tok, "rolloutu3") ||
        !strcmp(tok, "rollout4") || !strcmp(tok, "rolloutu4")) {
        int is_mcts = !strcmp(tok, "mcts");
        int is_policy = !strcmp(tok, "policy");
        int is_rollout = !strcmp(tok, "rollout") || !strcmp(tok, "rolloutu") ||
                         !strcmp(tok, "rollout2") || !strcmp(tok, "rolloutu2") ||
                         !strcmp(tok, "rollout3") || !strcmp(tok, "rolloutu3") ||
                         !strcmp(tok, "rollout4") || !strcmp(tok, "rolloutu4");
        int has_continuation_path =
            !strcmp(tok, "rollout2") || !strcmp(tok, "rolloutu2") ||
            !strcmp(tok, "rollout3") || !strcmp(tok, "rolloutu3") ||
            !strcmp(tok, "rollout4") || !strcmp(tok, "rolloutu4");
        int has_veto_path =
            !strcmp(tok, "rollout3") || !strcmp(tok, "rolloutu3");
        int has_ranker_path =
            !strcmp(tok, "rollout4") || !strcmp(tok, "rolloutu4");
        int is_uniform = !strcmp(tok, "rolloutu") ||
                         !strcmp(tok, "rolloutu2") ||
                         !strcmp(tok, "rolloutu3") ||
                         !strcmp(tok, "rolloutu4");
        char *path = strtok_r(NULL, ":", &save);
        if (!path) { fprintf(stderr, "agent '%s' needs a network path\n", tok); exit(1); }
        Net *n = load_net(path);
        agent_default(a, is_rollout ? AG_ROLLOUT :
                         (is_mcts ? AG_MCTS : (is_policy ? AG_POLICY : AG_NET)), n);
        a->owns_net = 1;
        char *v;
        if (is_rollout) {
            const char *continuation_path = path;
            if (has_continuation_path) {
                continuation_path = strtok_r(NULL, ":", &save);
                if (!continuation_path) {
                    fprintf(stderr,
                            "agent '%s' needs a continuation network path\n",
                            tok);
                    exit(1);
                }
                a->continuation_net = strcmp(path, continuation_path) == 0
                    ? n : load_net(continuation_path);
                a->owns_continuation_net = a->continuation_net != n;
            }
            if (has_veto_path) {
                char *veto_path = strtok_r(NULL, ":", &save);
                if (!veto_path) {
                    fprintf(stderr,
                            "agent '%s' needs a veto continuation network path\n",
                            tok);
                    exit(1);
                }
                if (strcmp(path, veto_path) == 0) {
                    a->veto_continuation_net = n;
                } else if (strcmp(continuation_path, veto_path) == 0) {
                    a->veto_continuation_net = a->continuation_net;
                } else {
                    a->veto_continuation_net = load_net(veto_path);
                    a->owns_veto_continuation_net = 1;
                }
            }
            if (has_ranker_path) {
                char *ranker_path = strtok_r(NULL, ":", &save);
                if (!ranker_path) {
                    fprintf(stderr,
                            "agent '%s' needs an action-ranker network path\n",
                            tok);
                    exit(1);
                }
                if (strcmp(path, ranker_path) == 0) {
                    a->action_ranker_net = n;
                } else if (strcmp(continuation_path, ranker_path) == 0) {
                    a->action_ranker_net = a->continuation_net;
                } else {
                    a->action_ranker_net = load_net(ranker_path);
                    a->owns_action_ranker_net = 1;
                }
            }
            a->no_belief = is_uniform;
            parse_rollout_tail(save ? save : "", a, spec);
        } else if (is_policy) {
            if ((v = strtok_r(NULL, ":", &save))) a->temp = (float)atof(v);
            if ((v = strtok_r(NULL, ":", &save))) a->symmetries = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) a->plan_deck_max = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) a->plan_block_gap = atoi(v);
            if ((v = strtok_r(NULL, ":", &save)))
                a->draw_root_deck_max = atoi(v);
            if (a->draw_root_deck_max < 0 ||
                a->draw_root_deck_max > NCARD) {
                fprintf(stderr, "agent '%s' has an invalid root draw-planner "
                                "threshold\n", spec);
                exit(1);
            }
        } else if (is_mcts) {
            if ((v = strtok_r(NULL, ":", &save))) a->dets = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) a->sims = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) a->root_width = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) a->node_width = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) a->symmetries = atoi(v);
        } else {
            if ((v = strtok_r(NULL, ":", &save))) a->draw_samples = atoi(v);
        }
        if (g_nname < MAX_NAMES) {
            snprintf(g_names[g_nname], sizeof g_names[0], "%s", spec);
            a->name = g_names[g_nname++];
        }
        return;
    }
    fprintf(stderr, "unknown agent kind '%s'\n", tok);
    exit(1);
}
