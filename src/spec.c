#include "spec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define MAX_NAMES 16
static char g_names[MAX_NAMES][96];
static int g_nname = 0;

static Net *load_net(const char *path)
{
    Net *n = (Net *)malloc(sizeof(Net));
    if (!n) { fprintf(stderr, "out of memory\n"); exit(1); }
    int r = net_load(n, path);
    if (r != 0) { fprintf(stderr, "cannot load net '%s' (error %d)\n", path, r); exit(1); }
    return n;
}

void spec_parse(const char *spec, Agent *a)
{
    char buf[512];
    snprintf(buf, sizeof buf, "%s", spec);
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
        !strcmp(tok, "rollout") || !strcmp(tok, "rolloutu")) {
        int is_mcts = !strcmp(tok, "mcts");
        int is_policy = !strcmp(tok, "policy");
        int is_rollout = !strcmp(tok, "rollout") || !strcmp(tok, "rolloutu");
        int is_uniform = !strcmp(tok, "rolloutu");
        char *path = strtok_r(NULL, ":", &save);
        if (!path) { fprintf(stderr, "agent '%s' needs a network path\n", tok); exit(1); }
        Net *n = load_net(path);
        agent_default(a, is_rollout ? AG_ROLLOUT :
                         (is_mcts ? AG_MCTS : (is_policy ? AG_POLICY : AG_NET)), n);
        char *v;
        if (is_rollout) {
            a->no_belief = is_uniform;
            if ((v = strtok_r(NULL, ":", &save))) a->dets = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) a->root_width = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) a->cand_floor = (float)atof(v);
            if ((v = strtok_r(NULL, ":", &save))) a->gate = (float)atof(v);
            if ((v = strtok_r(NULL, ":", &save))) a->min_cand = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) a->ply_lo = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) a->ply_hi = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) a->eval_cand = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) a->win_q = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) a->prune_dom = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) a->override_k = (float)atof(v);
            if ((v = strtok_r(NULL, ":", &save))) a->override_min = (float)atof(v);
            if ((v = strtok_r(NULL, ":", &save))) a->playout_sample = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) a->symmetries = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) a->cand_mass = (float)atof(v);
            if ((v = strtok_r(NULL, ":", &save))) a->batch_dets = atoi(v);
            if ((v = strtok_r(NULL, ":", &save))) a->playout_symmetries = atoi(v);
        } else if (is_policy) {
            if ((v = strtok_r(NULL, ":", &save))) a->temp = (float)atof(v);
            if ((v = strtok_r(NULL, ":", &save))) a->symmetries = atoi(v);
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
