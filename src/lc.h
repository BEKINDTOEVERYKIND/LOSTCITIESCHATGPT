/* lc.h -- Lost Cities game engine (2 player, Reiner Knizia).
 *
 * Rules implemented (standard 2-player game; competitive play is a match of
 * three rounds with cumulative scoring -- the State carries the match context
 * so agents can condition on it, and the round loops live in the callers):
 *   Deck: 5 suits x 12 cards = 60.  Per suit: three wagers ("handshakes") and
 *   the number cards 2..10.
 *   Setup: each player is dealt 8 cards.
 *   Turn: (1) play a card to one of your own expeditions, or discard it to the
 *   shared discard pile of its suit; (2) draw one card, either the top of the
 *   draw deck or the top of any discard pile -- except the pile you just
 *   discarded to on this same turn.
 *   Expeditions are strictly ascending in value; wagers may only be added
 *   before any number card of that suit.
 *   The game ends immediately after the turn in which the last deck card is
 *   drawn.
 *   Scoring per started expedition: (sum of numbers - 20) * (1 + wagers),
 *   plus a 20 point bonus if the expedition holds 8 or more cards.
 */
#ifndef LC_H
#define LC_H

#include <stdint.h>
#include <string.h>

/* -ffast-math permits the compiler to assume ordinary floating-point values
 * are finite, which can compile isfinite()/isinf() safety checks away.  These
 * bit-level classifiers remain valid under that optimization and should be
 * used anywhere non-finite external or sampled values must be rejected. */
static inline int lc_float_isfinite(float x)
{
    uint32_t bits;
    memcpy(&bits, &x, sizeof bits);
    return (bits & UINT32_C(0x7f800000)) != UINT32_C(0x7f800000);
}

static inline int lc_float_is_pos_inf(float x)
{
    uint32_t bits;
    memcpy(&bits, &x, sizeof bits);
    return bits == UINT32_C(0x7f800000);
}

static inline int lc_double_isfinite(double x)
{
    uint64_t bits;
    memcpy(&bits, &x, sizeof bits);
    return (bits & UINT64_C(0x7ff0000000000000)) !=
           UINT64_C(0x7ff0000000000000);
}

#define NSUIT 5
#define NRANK 12 /* 3 wagers + values 2..10 */
#define NCARD 60
#define HAND_SIZE 8
#define WAGERS_PER_SUIT 3

/* Nothing in the rules forces a game to end: two players who always draw from
 * a discard pile never touch the deck.  Sane play ends a game in 45-70 plies,
 * so this cap only ever fires for degenerate policies (an untrained network
 * early in training) and keeps self-play from stalling. */
#define LC_MAX_PLIES 300

/* card id = suit * 12 + rank_index
 * rank_index 0,1,2  -> wager (value 0)
 * rank_index 3..11  -> number card with value rank_index - 1  (2..10) */
#define CARD_SUIT(c) ((c) / NRANK)
#define CARD_RANK(c) ((c) % NRANK)
#define CARD_IS_WAGER(c) (CARD_RANK(c) < WAGERS_PER_SUIT)
#define CARD_VALUE(c) (CARD_IS_WAGER(c) ? 0 : (CARD_RANK(c) - 1))
#define CARD_MAKE(s, r) ((s) * NRANK + (r))

/* A full turn: which card leaves the hand, whether it is played or discarded,
 * and where the replacement card is drawn from. */
typedef struct {
    uint8_t card;      /* 0..59                                   */
    uint8_t discard;   /* 0 = play to expedition, 1 = discard      */
    uint8_t draw;      /* 0 = deck, 1..5 = discard pile (draw - 1) */
} Move;

#define MOVE_PACK(m) ((uint16_t)((m).card + 60u * (m).discard + 120u * (m).draw))
#define MOVE_CARD(p) ((uint8_t)((p) % 60u))
#define MOVE_DISC(p) ((uint8_t)(((p) / 60u) % 2u))
#define MOVE_DRAW(p) ((uint8_t)((p) / 120u))
#define MOVE_NPACK 720

/* Upper bound on the number of legal turns: 8 cards x 2 dispositions x 6 draw
 * sources. */
#define MAX_MOVES 96

typedef struct {
    uint8_t deck[NCARD];   /* deck[deck_pos] is the next card drawn */
    uint8_t deck_pos;
    uint8_t deck_left;

    uint64_t hand[2];      /* bitmask over card ids */
    uint8_t hand_n[2];
    uint64_t played[2];    /* cards in each player's expeditions */
    uint64_t discarded;    /* cards sitting in the discard piles  */
    /* Cards player p is publicly known to hold: drawing from a discard pile
     * happens face up, so until such a card is played or discarded again the
     * opponent knows exactly where it is.  Hidden (deck) draws never set it. */
    uint64_t known[2];

    uint8_t exp_wager[2][NSUIT]; /* wagers played, 0..3            */
    uint8_t exp_top[2][NSUIT];   /* highest number played, 0=none  */
    uint8_t exp_n[2][NSUIT];     /* cards in expedition incl wager */
    uint8_t exp_sum[2][NSUIT];   /* sum of number values           */

    uint8_t pile[NSUIT][NRANK];  /* discard piles, pile[s][n-1] = top */
    uint8_t pile_n[NSUIT];

    uint8_t turn;          /* player to move: 0 or 1 */
    uint8_t over;
    uint16_t nply;

    /* match context, fixed for the duration of a round */
    uint8_t round;         /* 0-based round of the match           */
    int16_t cum[2];        /* cumulative score from earlier rounds */
} State;

/* competitive Lost Cities is played over three rounds, total score wins */
#define MATCH_ROUNDS 3

/* ---- rng: xoshiro256** ---------------------------------------------- */
typedef struct { uint64_t s[4]; } Rng;

static inline uint64_t rotl64(uint64_t x, int k) { return (x << k) | (x >> (64 - k)); }

static inline uint64_t rng_next(Rng *r)
{
    uint64_t *s = r->s;
    const uint64_t result = rotl64(s[1] * 5, 7) * 9;
    const uint64_t t = s[1] << 17;
    s[2] ^= s[0]; s[3] ^= s[1]; s[1] ^= s[2]; s[0] ^= s[3]; s[2] ^= t;
    s[3] = rotl64(s[3], 45);
    return result;
}

static inline void rng_seed(Rng *r, uint64_t seed)
{
    /* splitmix64 expansion */
    for (int i = 0; i < 4; i++) {
        seed += 0x9E3779B97F4A7C15ULL;
        uint64_t z = seed;
        z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL;
        z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL;
        r->s[i] = z ^ (z >> 31);
    }
    for (int i = 0; i < 16; i++) (void)rng_next(r);
}

/* uniform in [0, n) */
static inline uint32_t rng_below(Rng *r, uint32_t n)
{
    return (uint32_t)(((__uint128_t)rng_next(r) * (__uint128_t)n) >> 64);
}

static inline float rng_float(Rng *r)
{
    return (float)((rng_next(r) >> 40) * (1.0 / 16777216.0));
}

/* ---- engine ---------------------------------------------------------- */
void lc_deal(State *st, Rng *rng);
void lc_deal_from_deck(State *st, const uint8_t deck[NCARD]);
int  lc_moves(const State *st, Move *out);          /* legal turns for st->turn */
void lc_apply(State *st, Move m);
/* Two-phase form.  lc_apply_play performs only the play/discard half, leaving
 * it to the caller to supply the drawn card -- agents use this so that a deck
 * draw can be modelled as an unknown card instead of peeking at the deck.
 * card < 0 means "take the real next deck card" for deck draws. */
void lc_apply_play(State *st, Move m);
void lc_apply_draw(State *st, Move m, int card);
int  lc_exp_score(const State *st, int p, int suit);
uint64_t lc_dead_cards(const State *st);            /* unplayable by anyone  */
int  lc_discard_dominated(const State *st, Move m, uint64_t dead);
int  lc_score(const State *st, int p);
int  lc_hand_cards(const State *st, int p, uint8_t *out);
/* Exact suit-renaming automorphism.  perm[s] is the new name of old suit s.
 * The caller must supply a permutation of 0..NSUIT-1. */
uint8_t lc_permute_card(uint8_t card, const uint8_t perm[NSUIT]);
Move lc_permute_move(Move m, const uint8_t perm[NSUIT]);
void lc_permute_suits(const State *src, State *dst,
                      const uint8_t perm[NSUIT]);
/* Cards whose location p cannot pin down: not in p's hand, not public, and
 * not known-held by the opponent.  This is the sampling pool for the unknown
 * part of the opponent's hand and the deck. */
void lc_unseen(const State *st, int p, uint8_t *out, int *n);
const char *lc_card_name(int card, char *buf);
void lc_move_name(const State *st, Move m, char *buf);

#endif /* LC_H */
