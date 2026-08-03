/* qpair -- paired rollout Q comparison for chosen candidate moves at a
 * replayed position.
 *
 * The analysis dump only searches the moves the policy proposes, so a move
 * the policy assigns ~0% never gets a Q value even when a human wants to see
 * one.  This tool answers "what was move X actually worth there?": it replays
 * a recorded game (same seed, same moves, so the hidden deck is identical) to
 * a given ply, then evaluates any moves you name with the same machinery the
 * rollout agent uses -- shared belief-sampled worlds, argmax-policy playouts
 * to the end of the round -- and reports each candidate's Q with a standard
 * error, plus the *paired* difference against the first candidate, which is
 * the number that actually decides between moves.
 *
 * The moves file is one move per line, "CARD ACT DRAW" (e.g. "Y2 d deck",
 * "Gx p Y"); plies 1..P-1 are replayed, matching the "n" field of the
 * analysis JSON.  Multi-round replay follows the match loop of analyze.c.
 *
 *   ./bin/qpair -n NET.bin -s SEED -f moves.txt -p 5 -w 4000 \
 *               -y 20 -c "Y2 d deck" -c "W4 p deck"
 */
#include "../src/lc.h"
#include "../src/agent.h"
#include "../src/net.h"
#include "../src/search.h"
#include "../src/spec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <math.h>

#define MAXC 8

/* find CARD (by display name) in p's hand; any wager of the suit matches */
static int find_card(const State *st, int p, const char *name)
{
    char b[8];
    for (int c = 0; c < NCARD; c++) {
        lc_card_name(c, b);
        if (!strcasecmp(b, name) && ((st->hand[p] >> c) & 1ULL)) return c;
    }
    return -1;
}

static int parse_move(const State *st, int p, const char *cs, const char *as,
                      const char *ws, Move *out)
{
    static const char SUIT_CH[NSUIT + 1] = "YBWGR";
    int card = find_card(st, p, cs);
    if (card < 0) { fprintf(stderr, "qpair: %s not in hand\n", cs); return 0; }
    int disc = (as[0] == 'd' || as[0] == 'D');
    int draw = -1;
    if (!strcasecmp(ws, "deck")) draw = 0;
    else for (int s = 0; s < NSUIT; s++)
        if (ws[0] == SUIT_CH[s] || ws[0] == SUIT_CH[s] + 32) draw = s + 1;
    if (draw < 0) { fprintf(stderr, "qpair: bad draw '%s'\n", ws); return 0; }
    out->card = (uint8_t)card;
    out->discard = (uint8_t)disc;
    out->draw = (uint8_t)draw;
    Move mv[MAX_MOVES];
    int n = lc_moves(st, mv);
    for (int i = 0; i < n; i++)
        if (mv[i].card == out->card && mv[i].discard == out->discard &&
            mv[i].draw == out->draw) return 1;
    fprintf(stderr, "qpair: %s %s %s is not legal here\n", cs, as, ws);
    return 0;
}

/* ---- state-file loading (-S): direct position reconstruction ----------- */

/* lowest unused card id with this display name (wagers have three copies) */
static int name_id_free(const char *nm, uint64_t used)
{
    char b[8];
    for (int c = 0; c < NCARD; c++) {
        lc_card_name(c, b);
        if (!strcasecmp(b, nm) && !((used >> c) & 1ULL)) return c;
    }
    return -1;
}

/* Rebuild a State from tools/statedump.py output.  Only the mover's
 * information set has to be faithful: the belief determinizer resamples the
 * opponent hand and the whole deck from it anyway. */
static int load_state(const char *path, State *st)
{
    FILE *f = fopen(path, "r");
    if (!f) return 0;
    memset(st, 0, sizeof *st);
    uint64_t used = 0;
    char line[512];
    while (fgets(line, sizeof line, f)) {
        char *tok = strtok(line, " \t\n");
        if (!tok) continue;
        if (!strcmp(tok, "turn")) st->turn = (uint8_t)atoi(strtok(NULL, " \n"));
        else if (!strcmp(tok, "round")) st->round = (uint8_t)atoi(strtok(NULL, " \n"));
        else if (!strcmp(tok, "nply")) st->nply = (uint16_t)atoi(strtok(NULL, " \n"));
        else if (!strcmp(tok, "deck_left")) st->deck_left = (uint8_t)atoi(strtok(NULL, " \n"));
        else if (!strcmp(tok, "cum")) {
            int a = atoi(strtok(NULL, " \n")), b = atoi(strtok(NULL, " \n"));
            st->cum[0] = (int16_t)a;
            st->cum[1] = (int16_t)b;
        } else if (!strncmp(tok, "hand", 4) && tok[4] >= '0' && tok[4] <= '1') {
            int pl = tok[4] - '0';
            char *w;
            while ((w = strtok(NULL, " \n"))) {
                int c = name_id_free(w, used);
                if (c < 0) { fclose(f); return 0; }
                used |= 1ULL << c;
                st->hand[pl] |= 1ULL << c;
                st->hand_n[pl]++;
            }
        } else if (!strncmp(tok, "known", 5) && tok[5] >= '0' && tok[5] <= '1') {
            int pl = tok[5] - '0';
            char *w;
            while ((w = strtok(NULL, " \n"))) {
                char b[8];
                for (int c = 0; c < NCARD; c++) {
                    lc_card_name(c, b);
                    if (!strcasecmp(b, w) && ((st->hand[pl] >> c) & 1ULL) &&
                        !((st->known[pl] >> c) & 1ULL)) {
                        st->known[pl] |= 1ULL << c;
                        break;
                    }
                }
            }
        } else if (!strcmp(tok, "exp")) {
            int pl = atoi(strtok(NULL, " \n"));
            int s = atoi(strtok(NULL, " \n"));
            char *w;
            while ((w = strtok(NULL, " \n"))) {
                int c = name_id_free(w, used);
                if (c < 0) { fclose(f); return 0; }
                used |= 1ULL << c;
                st->played[pl] |= 1ULL << c;
                st->exp_n[pl][s]++;
                if (CARD_IS_WAGER(c)) st->exp_wager[pl][s]++;
                else {
                    int v = CARD_VALUE(c);
                    if (v > st->exp_top[pl][s]) st->exp_top[pl][s] = (uint8_t)v;
                    st->exp_sum[pl][s] = (uint8_t)(st->exp_sum[pl][s] + v);
                }
            }
        } else if (!strcmp(tok, "pile")) {
            int s = atoi(strtok(NULL, " \n"));
            char *w;
            while ((w = strtok(NULL, " \n"))) {
                int c = name_id_free(w, used);
                if (c < 0) { fclose(f); return 0; }
                used |= 1ULL << c;
                st->pile[s][st->pile_n[s]++] = (uint8_t)c;
                st->discarded |= 1ULL << c;
            }
        }
    }
    fclose(f);
    return 1;
}

/* Continuation to the end of the round, three flavours.  Default is the
 * argmax-policy playout of rollout.c.  temp > 0 samples the policy instead
 * (p^(1/T)), which probes whether a Q difference is an artifact of the
 * deterministic playout lines rather than a property of the position.  A
 * continuation agent (-A) replaces the policy entirely -- e.g. the gated
 * rollout agent, so both sides keep *searching* inside the playout; slow,
 * but the least biased estimate this codebase can produce.  The rng is
 * seeded per world, identically for every candidate, so all three stay
 * paired comparisons. */
static int playout(const Net *net, const Agent *cont, float temp,
                   State *s, int p, uint64_t wseed)
{
    Rng prng;
    rng_seed(&prng, wseed);
    Move mv[MAX_MOVES];
    float score[MAX_MOVES];
    while (!s->over) {
        if (cont) {
            lc_apply(s, agent_move(cont, s, &prng));
            continue;
        }
        int n = policy_probs(net, s, mv, score, NULL);
        if (n <= 0) break;
        int pick = 0;
        if (temp > 0.0f) {
            float w[MAX_MOVES];
            for (int i = 0; i < n; i++) w[i] = powf(score[i], 1.0f / temp);
            pick = sample_index(w, n, &prng);
        } else {
            for (int i = 1; i < n; i++) if (score[i] > score[pick]) pick = i;
        }
        lc_apply(s, mv[pick]);
    }
    return lc_score(s, p) - lc_score(s, p ^ 1);
}

int main(int argc, char **argv)
{
    const char *netpath = NULL, *movespath = NULL, *contspec = NULL;
    const char *evalspec = NULL, *holdcard = NULL;
    const char *statepath = NULL;
    uint64_t seed = 1;
    int target = 1, worlds = 2000;
    int uniform_worlds = 0, trajectory_symmetries = 1;
    float temp = 0.0f;
    const char *cand_str[MAXC];
    int ncand = 0;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-n") && i + 1 < argc) netpath = argv[++i];
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "-f") && i + 1 < argc) movespath = argv[++i];
        else if (!strcmp(argv[i], "-S") && i + 1 < argc) statepath = argv[++i];
        else if (!strcmp(argv[i], "-p") && i + 1 < argc) target = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-w") && i + 1 < argc) worlds = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-T") && i + 1 < argc) temp = (float)atof(argv[++i]);
        else if (!strcmp(argv[i], "-A") && i + 1 < argc) contspec = argv[++i];
        else if (!strcmp(argv[i], "-E") && i + 1 < argc) evalspec = argv[++i];
        else if (!strcmp(argv[i], "-U")) uniform_worlds = 1;
        else if (!strcmp(argv[i], "-y") && i + 1 < argc)
            trajectory_symmetries = atoi(argv[++i]);
        else if (!strcmp(argv[i], "-H") && i + 1 < argc) holdcard = argv[++i];
        else if (!strcmp(argv[i], "-c") && i + 1 < argc && ncand < MAXC) cand_str[ncand++] = argv[++i];
        else {
            fprintf(stderr, "usage: %s -n NET -s SEED "
                    "(-f MOVES -p PLY | -S STATE) "
                    "[-w WORLDS] [-T temp] [-A contspec] [-U] "
                    "[-y trajectory_symmetries] "
                    "[-E rollout_spec | -c \"CARD p|d DRAW\" [-c ...]]\n",
                    argv[0]);
            return 1;
        }
    }
    if (!netpath || (!movespath && !statepath) ||
        (!evalspec && ncand == 0)) {
        fprintf(stderr, "qpair: need -n, -f or -S, and -E or at least one -c\n");
        return 1;
    }

    Net *net = (Net *)malloc(sizeof(Net));
    if (!net || net_load(net, netpath)) { fprintf(stderr, "qpair: cannot load %s\n", netpath); return 1; }

    Rng rng;
    rng_seed(&rng, seed);
    State st;
    char cs[16], as[16], ws[16];
    if (statepath) {
        /* direct reconstruction: the only way to reach rounds 1-2, whose
         * deals depend on RNG the generating search consumed */
        if (!load_state(statepath, &st)) { fprintf(stderr, "qpair: bad state file %s\n", statepath); return 1; }
    } else {
    FILE *mf = fopen(movespath, "r");
    if (!mf) { fprintf(stderr, "qpair: cannot open %s\n", movespath); return 1; }

    /* replay, following the match loop of analyze.c / match.c */
    int cum[2] = { 0, 0 }, rd = 0;
    lc_deal(&st, &rng);
    st.round = 0;
    st.turn = 0;
    char line[64];
    for (int ply = 1; ply < target; ply++) {
        if (st.over) {
            cum[0] += lc_score(&st, 0);
            cum[1] += lc_score(&st, 1);
            rd++;
            lc_deal(&st, &rng);
            st.round = (uint8_t)rd;
            st.cum[0] = (int16_t)cum[0];
            st.cum[1] = (int16_t)cum[1];
            st.turn = (uint8_t)(rd & 1);
        }
        if (!fgets(line, sizeof line, mf) || sscanf(line, "%15s %15s %15s", cs, as, ws) != 3) {
            fprintf(stderr, "qpair: moves file ends before ply %d\n", target);
            return 1;
        }
        Move m;
        if (!parse_move(&st, st.turn, cs, as, ws, &m)) { fprintf(stderr, "  (at ply %d)\n", ply); return 1; }
        lc_apply(&st, m);
    }
    fclose(mf);
    if (st.over) { fprintf(stderr, "qpair: round already over at ply %d\n", target); return 1; }
    }

    const int p = st.turn;
    char b[8];
    printf("position: %s%s, round %d, player %d to move, deck %d, cum [%d,%d]\nhand:",
           statepath ? "state " : "ply ", statepath ? statepath : (sprintf(b, "%d", target), b),
           st.round, p, st.deck_left, st.cum[0], st.cum[1]);
    uint8_t hc[HAND_SIZE];
    int hn = lc_hand_cards(&st, p, hc);
    for (int i = 0; i < hn; i++) { lc_card_name(hc[i], b); printf(" %s", b); }
    printf("\n");

    if (evalspec) {
        Agent evaluator;
        spec_parse(evalspec, &evaluator);
        if (evaluator.kind != AG_ROLLOUT) {
            fprintf(stderr, "qpair: -E requires a rollout evaluator\n");
            return 1;
        }
        Rng erng;
        rng_seed(&erng, seed);
        SearchStats ss;
        Move selected = rollout_move(&evaluator, &st, &erng, NULL, &ss);
        printf("rollout evaluator: %s\n", evalspec);
        printf("worlds: %d/%d  raw_resolved: %s  confirmation: %d worlds, %s"
               "  prefix: %d trusted, proposed %d, selected %d"
               "  prefix_check: %d worlds, %s\n",
               ss.worlds, ss.max_worlds, ss.resolved ? "yes" : "no",
               ss.confirm_worlds, ss.confirmed ? "passed" : "not passed",
               ss.trusted_candidates, ss.prefix_proposed,
               ss.selection_reference,
               ss.prefix_confirm_worlds,
               ss.prefix_confirmed ? "passed" : "not passed");
        if (ss.prefix_confirm_worlds > 0 && ss.prefix_proposed > 0) {
            int k = ss.prefix_proposed;
            printf("prefix fresh evidence: %+.2f +- %.2f vs baseline; "
                   "numerical agreement: %s; paired gate: %s "
                   "(> %.2f SE and > %.2f objective units%s)\n",
                   ss.prefix_delta[k], ss.prefix_dse[k],
                   ss.prefix_numerical_agreement ? "yes" : "no",
                   ss.prefix_gate_passed ? "passed" : "not passed",
                   evaluator.prefix_confirm_k,
                   evaluator.prefix_confirm_min,
                   evaluator.prefix_confirm_k == 0.0f
                       ? ", disabled" : "");
        }
        printf("%-16s %8s %17s %8s %17s %8s %9s %8s\n",
               "candidate", "prior", "delta vs ref", "p-pass",
               "confirm delta", "c-pass", "guard", "selected");
        for (int i = 0; i < ss.n; i++) {
            char card[8], move[20], draw[8];
            lc_card_name(ss.mv[i].card, card);
            if (ss.mv[i].draw == 0) snprintf(draw, sizeof draw, "deck");
            else {
                static const char suit_ch[NSUIT + 1] = "YBWGR";
                snprintf(draw, sizeof draw, "%c", suit_ch[ss.mv[i].draw - 1]);
            }
            snprintf(move, sizeof move, "%s %c %s", card,
                     ss.mv[i].discard ? 'd' : 'p', draw);
            printf("%-16s %8.4f %+7.2f +- %-6.2f %8s "
                   "%+7.2f +- %-6.2f %8s %9s %8s\n",
                   move, ss.prior[i], ss.rdelta[i], ss.rdse[i],
                   ss.pqualified[i] ? "yes" : "no",
                   ss.cdelta[i], ss.cdse[i],
                   ss.csupported[i] ? "yes" : "no",
                   ss.guard_rejected[i] ? "discard" : "-",
                   MOVE_PACK(ss.mv[i]) == MOVE_PACK(selected) ? "yes" : "");
        }
        free((void *)evaluator.net);
        free(net);
        return 0;
    }

    Move cand[MAXC];
    for (int c = 0; c < ncand; c++) {
        if (sscanf(cand_str[c], "%15s %15s %15s", cs, as, ws) != 3 ||
            !parse_move(&st, p, cs, as, ws, &cand[c])) {
            fprintf(stderr, "qpair: bad candidate '%s'\n", cand_str[c]);
            return 1;
        }
    }

    /* the policy's own opinion of each candidate, for context */
    Move pmv[MAX_MOVES];
    float prob[MAX_MOVES], value;
    int nleg = policy_probs_sym(net, &st, pmv, prob, &value,
                                trajectory_symmetries);
    /* policy_probs already reports the value in points. */
    printf("value head: %+.1f   policy priors:", value);
    for (int c = 0; c < ncand; c++) {
        float pr = 0.0f;
        for (int i = 0; i < nleg; i++)
            if (pmv[i].card == cand[c].card && pmv[i].discard == cand[c].discard &&
                pmv[i].draw == cand[c].draw) pr = prob[i];
        printf("  [%s] %.4f", cand_str[c], pr);
    }
    printf("\n");

    Agent cont;
    if (contspec) {
        spec_parse(contspec, &cont);
        printf("continuations: %s%s\n", contspec, temp > 0 ? " (temp ignored)" : "");
    } else if (temp > 0.0f) {
        printf("continuations: policy sampled at temp %.2f\n", temp);
    } else {
        printf("continuations: exact %d-way policy ensemble\n",
               trajectory_symmetries);
    }
    printf("hidden worlds: %s\n",
           uniform_worlds ? "uniform card-count prior" : "learned belief");

    /* -H CARD: split the report by whether the sampled world put CARD in the
     * opponent's hand -- a direct test of "this move is about what THEY hold" */
    int hold_id = -1;
    if (holdcard) {
        char hb[8];
        for (int c = 0; c < NCARD; c++) {
            lc_card_name(c, hb);
            if (!strcasecmp(hb, holdcard) && !((st.hand[p] >> c) & 1ULL)) { hold_id = c; break; }
        }
        if (hold_id < 0) { fprintf(stderr, "qpair: -H card '%s' not found\n", holdcard); return 1; }
    }

    double *val = (double *)malloc(sizeof(double) * (size_t)ncand * (size_t)worlds);
    uint8_t *held = (uint8_t *)calloc((size_t)worlds, 1);
    uint8_t perms[120][NSUIT];
    int nsym = suit_permutations(trajectory_symmetries, perms);
    for (int d = 0; d < worlds; d++) {
        State world;
        if (uniform_worlds) determinize(&st, p, &rng, &world);
        else determinize_b(&st, p, &rng, net, &world);
        if (hold_id >= 0) held[d] = (uint8_t)((world.hand[p ^ 1] >> hold_id) & 1ULL);
        uint64_t wseed = seed ^ (0x9E3779B97F4A7C15ULL * (uint64_t)(d + 1));
        for (int c = 0; c < ncand; c++) {
            State s = world;               /* same world for every candidate */
            lc_apply(&s, cand[c]);
            if (!contspec && temp <= 0.0f && nsym > 1) {
                /* playout()'s raw path is intentionally small.  An exact
                 * policy Agent supplies the requested ensemble at every
                 * downstream information set. */
                Agent sym;
                agent_default(&sym, AG_POLICY, net);
                sym.symmetries = trajectory_symmetries;
                val[c * worlds + d] = playout(net, &sym, 0.0f, &s, p, wseed);
            } else {
                val[c * worlds + d] =
                    playout(net, contspec ? &cont : NULL, temp, &s, p, wseed);
            }
        }
    }

    printf("\n%-16s %10s %8s     %s\n", "candidate", "Q(round)", "+-SE", "paired diff vs first");
    double m0 = 0.0;
    for (int c = 0; c < ncand; c++) {
        double mean = 0.0, var = 0.0, dmean = 0.0, dvar = 0.0;
        for (int d = 0; d < worlds; d++) mean += val[c * worlds + d];
        mean /= worlds;
        if (c == 0) m0 = mean;
        for (int d = 0; d < worlds; d++) {
            double e = val[c * worlds + d] - mean;
            var += e * e;
            double dd = val[c * worlds + d] - val[0 * worlds + d];
            double de = dd - (mean - m0);
            dvar += de * de;
        }
        var /= (worlds - 1);
        dvar /= (worlds - 1);
        dmean = mean - m0;
        printf("%-16s %+10.2f %8.2f", cand_str[c], mean, sqrt(var / worlds));
        if (c > 0) printf("     %+.2f +- %.2f", dmean, sqrt(dvar / worlds));
        printf("\n");
    }
    if (hold_id >= 0) {
        for (int g = 1; g >= 0; g--) {
            int ng = 0;
            for (int d = 0; d < worlds; d++) if (held[d] == g) ng++;
            printf("\nworlds where opponent %s %s: %d (%.0f%%)\n",
                   g ? "HOLDS" : "does NOT hold", holdcard, ng,
                   100.0 * ng / worlds);
            if (ng < 2) continue;
            for (int c = 1; c < ncand; c++) {
                double dm = 0.0, dv = 0.0;
                for (int d = 0; d < worlds; d++)
                    if (held[d] == g) dm += val[c * worlds + d] - val[0 * worlds + d];
                dm /= ng;
                for (int d = 0; d < worlds; d++)
                    if (held[d] == g) {
                        double x = val[c * worlds + d] - val[0 * worlds + d] - dm;
                        dv += x * x;
                    }
                printf("  [%s] vs [%s]: %+.2f +- %.2f\n",
                       cand_str[c], cand_str[0], dm, sqrt(dv / (ng - 1) / ng));
            }
        }
    }
    free(held);
    free(val);
    free(net);
    return 0;
}
