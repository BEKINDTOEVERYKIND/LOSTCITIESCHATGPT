/* analyze -- play one self-play match and dump a per-ply JSON analysis.
 *
 * A match is -r ROUNDS rounds (default MATCH_ROUNDS): fresh deal per round,
 * st.round / st.cum carrying the match context, and the first player
 * alternating by round, exactly like the reference loop in src/match.c.
 *
 * For every ply, before the chosen move is applied, the dump records the full
 * public state plus the mover's hand(s), the round and cumulative totals, the
 * publicly known cards in each hand, the value head from both perspectives,
 * the policy distribution over legal moves, and the rollout search
 * statistics.  rollout_move is called exactly once per ply and its returned
 * move is the move played, so the dump describes the game that was actually
 * generated.
 *
 * Output is a single JSON object on stdout; redirect to a file.
 */
#define _POSIX_C_SOURCE 200809L /* open_memstream under -std=c11 */
#include "../src/lc.h"
#include "../src/agent.h"
#include "../src/search.h"
#include <math.h>
#include "../src/spec.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static const char SUIT_CH[NSUIT + 1] = "YBWGR";

/* ---- tiny JSON helpers: fp is either stdout or a memstream -------------- */

static void j_string(FILE *fp, const char *s)
{
    fputc('"', fp);
    for (; *s; s++) {
        unsigned char c = (unsigned char)*s;
        switch (c) {
        case '"':  fputs("\\\"", fp); break;
        case '\\': fputs("\\\\", fp); break;
        case '\b': fputs("\\b", fp); break;
        case '\f': fputs("\\f", fp); break;
        case '\n': fputs("\\n", fp); break;
        case '\r': fputs("\\r", fp); break;
        case '\t': fputs("\\t", fp); break;
        default:
            if (c < 0x20) fprintf(fp, "\\u%04x", c);
            else fputc(c, fp);
        }
    }
    fputc('"', fp);
}

static void j_card(FILE *fp, int c)
{
    char b[8];
    lc_card_name(c, b);
    j_string(fp, b);
}

/* ["Y7","Bx",...] */
static void j_card_arr(FILE *fp, const uint8_t *cards, int n)
{
    fputc('[', fp);
    for (int i = 0; i < n; i++) {
        if (i) fputc(',', fp);
        j_card(fp, cards[i]);
    }
    fputc(']', fp);
}

static const char *act_str(Move m) { return m.discard ? "discard" : "play"; }

static void draw_str(Move m, char *b)
{
    if (m.draw == 0) strcpy(b, "deck");
    else { b[0] = SUIT_CH[m.draw - 1]; b[1] = 0; }
}

/* hand sorted by card id */
static void j_hand(FILE *fp, const State *st, int p)
{
    uint8_t cards[HAND_SIZE];
    int n = lc_hand_cards(st, p, cards);
    for (int i = 0; i < n; i++)
        for (int j = i + 1; j < n; j++)
            if (cards[j] < cards[i]) { uint8_t t = cards[i]; cards[i] = cards[j]; cards[j] = t; }
    j_card_arr(fp, cards, n);
}

/* [[cards p0 is publicly known to hold],[same for p1]], sorted by card id
 * (bit-order iteration of st->known is ascending id order already) */
static void j_known(FILE *fp, const State *st)
{
    fputc('[', fp);
    for (int p = 0; p < 2; p++) {
        if (p) fputc(',', fp);
        uint8_t cards[HAND_SIZE];
        int n = 0;
        uint64_t k = st->known[p];
        while (k) { cards[n++] = (uint8_t)__builtin_ctzll(k); k &= k - 1; }
        j_card_arr(fp, cards, n);
    }
    fputc(']', fp);
}

/* one player's five expeditions in play order (card-id order per suit:
 * wagers first, then ascending numbers -- which is the only legal order) */
static void j_exps(FILE *fp, const State *st, int p)
{
    fputc('[', fp);
    for (int s = 0; s < NSUIT; s++) {
        if (s) fputc(',', fp);
        uint8_t cards[NRANK];
        int n = 0;
        for (int c = s * NRANK; c < (s + 1) * NRANK; c++)
            if ((st->played[p] >> c) & 1ULL) cards[n++] = (uint8_t)c;
        j_card_arr(fp, cards, n);
    }
    fputc(']', fp);
}

/* the five discard piles bottom-to-top */
static void j_piles(FILE *fp, const State *st)
{
    fputc('[', fp);
    for (int s = 0; s < NSUIT; s++) {
        if (s) fputc(',', fp);
        j_card_arr(fp, st->pile[s], st->pile_n[s]);
    }
    fputc(']', fp);
}

/* {"card":"R2","act":"play","draw":"deck"  -- shared prefix of policy/search
 * entries and the move object; the caller closes the brace */
static void j_move_open(FILE *fp, Move m)
{
    char d[8];
    draw_str(m, d);
    fprintf(fp, "{\"card\":");
    j_card(fp, m.card);
    fprintf(fp, ",\"act\":\"%s\",\"draw\":\"%s\"", act_str(m), d);
}

static int move_eq(Move a, Move b)
{
    return a.card == b.card && a.discard == b.discard && a.draw == b.draw;
}

int main(int argc, char **argv)
{
    const char *spec = "rollout:data/c8.bin:128:6";
    uint64_t seed = 1;
    int rounds = MATCH_ROUNDS;
    for (int i = 1; i < argc; i++) {
        if (!strcmp(argv[i], "-a") && i + 1 < argc) spec = argv[++i];
        else if (!strcmp(argv[i], "-s") && i + 1 < argc) seed = strtoull(argv[++i], NULL, 10);
        else if (!strcmp(argv[i], "-r") && i + 1 < argc) rounds = atoi(argv[++i]);
        else { fprintf(stderr, "usage: %s [-a SPEC] [-s seed] [-r rounds]\n", argv[0]); return 1; }
    }
    if (rounds < 1) rounds = 1;
    if (rounds > MATCH_ROUNDS) rounds = MATCH_ROUNDS;

    Agent ag;
    spec_parse(spec, &ag);
    if (!ag.net) { fprintf(stderr, "analyze: spec '%s' has no network\n", spec); return 1; }

    Rng rng;
    rng_seed(&rng, seed);

    /* the ply array is streamed into memory while the match is played, because
     * meta (which needs the final scores) comes first in the output */
    char *plybuf = NULL;
    size_t plylen = 0;
    FILE *pf = open_memstream(&plybuf, &plylen);
    if (!pf) { fprintf(stderr, "analyze: open_memstream failed\n"); return 1; }

    char start_hands[2][256];
    int cum[2] = { 0, 0 };
    int round_scores[MATCH_ROUNDS][2];
    int ply = 0;

    for (int rd = 0; rd < rounds; rd++) {
    State st;
    lc_deal(&st, &rng);
    st.round = (uint8_t)rd;
    st.cum[0] = (int16_t)cum[0];
    st.cum[1] = (int16_t)cum[1];
    st.turn = (uint8_t)(rd & 1);

    if (rd == 0) {
        for (int p = 0; p < 2; p++) {
            char *hb = NULL;
            size_t hl = 0;
            FILE *hf = open_memstream(&hb, &hl);
            j_hand(hf, &st, p);
            fclose(hf);
            snprintf(start_hands[p], sizeof start_hands[p], "%s", hb);
            free(hb);
        }
    }

    while (!st.over) {
        int p = st.turn;
        ply++;
        if (ply > 1) fputc(',', pf);
        fprintf(pf, "{\"n\":%d,\"player\":%d,\"round\":%d,\"cum\":[%d,%d],\"deck_left\":%d,",
                ply, p, rd, cum[0], cum[1], st.deck_left);

        fprintf(pf, "\"known\":");
        j_known(pf, &st);
        fprintf(pf, ",\"hands\":[");
        j_hand(pf, &st, 0);
        fputc(',', pf);
        j_hand(pf, &st, 1);
        fprintf(pf, "],\"exps\":[");
        j_exps(pf, &st, 0);
        fputc(',', pf);
        j_exps(pf, &st, 1);
        fprintf(pf, "],\"piles\":");
        j_piles(pf, &st);

        /* value head from each perspective, in points */
        Features feat;
        float v[2];
        for (int q = 0; q < 2; q++) {
            feat_extract(&st, q, &feat);
            v[q] = net_value(ag.net, &feat) * VAL_SCALE;
        }
        fprintf(pf, ",\"values\":[%.1f,%.1f]", v[0], v[1]);

        /* belief head from the MOVER's perspective: for every card whose
         * location the mover cannot pin down, the learned probability that
         * the opponent holds it -- with the omniscient truth beside it so the
         * viewer can show how good the inference actually is */
        {
            int mp = st.turn, mo = mp ^ 1;
            uint8_t bcards[NCARD];
            int nb = 0;
            lc_unseen(&st, mp, bcards, &nb);
            Features bf;
            feat_extract(&st, mp, &bf);
            NetAct bact;
            net_trunk(ag.net, &bf, &bact);
            float blg[NCARD];
            net_belief_act(ag.net, &bact, bcards, nb, blg);
            int bord[NCARD];
            for (int i = 0; i < nb; i++) bord[i] = i;
            for (int i = 0; i < nb; i++)
                for (int j2 = i + 1; j2 < nb; j2++)
                    if (blg[bord[j2]] > blg[bord[i]]) { int t = bord[i]; bord[i] = bord[j2]; bord[j2] = t; }
            int need = (int)st.hand_n[mo] - __builtin_popcountll(st.known[mo]);
            float prior = nb > 0 ? (float)need / (float)nb : 0.0f;
            fprintf(pf, ",\"belief\":{\"persp\":%d,\"unknown_hand\":%d,"
                        "\"unknown_pool\":%d,\"prior\":%.6f,\"cards\":[",
                    mp, need, nb, prior);
            int bkeep = nb < 14 ? nb : 14;
            for (int i = 0; i < bkeep; i++) {
                int ci = bord[i];
                float bp = 1.0f / (1.0f + expf(-blg[ci]));
                if (i) fputc(',', pf);
                fprintf(pf, "{\"card\":");
                j_card(pf, bcards[ci]);
                fprintf(pf, ",\"p\":%.3f,\"held\":%s}",
                        bp, ((st.hand[mo] >> bcards[ci]) & 1ULL) ? "true" : "false");
            }
            fprintf(pf, "],\"all_cards\":[");
            for (int i = 0; i < nb; i++) {
                int ci = bord[i];
                float bp = 1.0f / (1.0f + expf(-blg[ci]));
                if (i) fputc(',', pf);
                fprintf(pf, "{\"card\":");
                j_card(pf, bcards[ci]);
                fprintf(pf, ",\"p\":%.6f,\"held\":%s}",
                        bp, ((st.hand[mo] >> bcards[ci]) & 1ULL)
                                ? "true" : "false");
            }
            fprintf(pf, "]}");
        }

        /* policy head over all legal moves, best first, capped at 10 */
        Move pmv[MAX_MOVES];
        float prob[MAX_MOVES], pv;
        int nleg = policy_probs(ag.net, &st, pmv, prob, &pv);
        int ord[MAX_MOVES];
        for (int i = 0; i < nleg; i++) ord[i] = i;
        for (int i = 0; i < nleg; i++)
            for (int j = i + 1; j < nleg; j++)
                if (prob[ord[j]] > prob[ord[i]]) { int t = ord[i]; ord[i] = ord[j]; ord[j] = t; }
        fprintf(pf, ",\"nlegal\":%d,\"policy\":[", nleg);
        int keep = nleg < 10 ? nleg : 10;
        for (int i = 0; i < keep; i++) {
            if (i) fputc(',', pf);
            j_move_open(pf, pmv[ord[i]]);
            fprintf(pf, ",\"prob\":%.3f}", prob[ord[i]]);
        }
        fputc(']', pf);

        /* one rollout_move call decides the ply and yields the search stats */
        SearchStats ss;
        memset(&ss, 0, sizeof ss);
        float sval = 0.0f;
        Move m = rollout_move(&ag, &st, &rng, &sval, &ss);

        int sord[MAX_MOVES];
        for (int i = 0; i < ss.n; i++) sord[i] = i;
        for (int i = 0; i < ss.n; i++)
            for (int j = i + 1; j < ss.n; j++)
                if (ss.q[sord[j]] > ss.q[sord[i]]) { int t = sord[i]; sord[i] = sord[j]; sord[j] = t; }
        fprintf(pf, ",\"search\":[");
        for (int i = 0; i < ss.n; i++) {
            if (i) fputc(',', pf);
            int k = sord[i];
            j_move_open(pf, ss.mv[k]);
            fprintf(pf, ",\"q\":%.1f,\"se\":%.1f,\"visits\":%.0f,\"chosen\":%s",
                    ss.q[k], ss.se[k], ss.visits[k], move_eq(ss.mv[k], m) ? "true" : "false");
            if (ss.qw[k] >= 0.0) fprintf(pf, ",\"qw\":%.3f", ss.qw[k]);
            fputc('}', pf);
        }
        fputc(']', pf);

        /* the card that will be drawn: read before lc_apply */
        int drawn = m.draw == 0 ? st.deck[st.deck_pos]
                                : st.pile[m.draw - 1][st.pile_n[m.draw - 1] - 1];
        fprintf(pf, ",\"move\":");
        j_move_open(pf, m);
        fprintf(pf, ",\"drawn\":");
        j_card(pf, drawn);
        fprintf(pf, "}}");

        lc_apply(&st, m);
    }

    round_scores[rd][0] = lc_score(&st, 0);
    round_scores[rd][1] = lc_score(&st, 1);
    cum[0] += round_scores[rd][0];
    cum[1] += round_scores[rd][1];
    }   /* rounds */
    fclose(pf);

    printf("{\"meta\":{\"agent\":");
    j_string(stdout, spec);
    printf(",\"seed\":%llu,\"plies\":%d,\"rounds\":%d,\"round_scores\":[",
           (unsigned long long)seed, ply, rounds);
    for (int rd = 0; rd < rounds; rd++)
        printf("%s[%d,%d]", rd ? "," : "", round_scores[rd][0], round_scores[rd][1]);
    printf("],\"final\":[%d,%d],\"generated\":\"analyze\"},\n", cum[0], cum[1]);
    printf("\"start_hands\":[%s,%s],\n", start_hands[0], start_hands[1]);
    printf("\"plies\":[%s]}\n", plybuf);
    free(plybuf);
    return 0;
}
