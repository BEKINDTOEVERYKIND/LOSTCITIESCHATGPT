#ifndef HEURISTIC_H
#define HEURISTIC_H

#include "lc.h"

/* Hand-crafted projection evaluation, in points, from p's point of view. */
float heur_eval(const State *st, int p);

/* Value of a move in a perfect-information (determinized) state, where the
 * drawn card is simply taken from the deck.  For real play the deck draw is an
 * unknown card and has to be averaged over the cards the mover has not seen --
 * that is move_value_heur() in agent.h.  Getting it wrong makes deck draws look
 * worse than they are and the agent stalls, recycling discard piles instead of
 * ever finishing the game. */
float heur_move_value_det(const State *st, Move m);

#endif
