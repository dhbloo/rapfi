/*
 *  Rapfi, a Gomoku/Renju playing engine supporting piskvork protocol.
 *  Copyright (C) 2022  Rapfi developers
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "../core/pos.h"
#include "../core/types.h"
#include "board.h"
#include "scopedmove.h"

/// Statically detect a forced win or loss for the side to move from the pattern counts the board
/// already maintains, without searching. Cheap enough to call at every node.
/// @param board The board state of a position to check.
/// @param ply The current search ply, used to encode mate distance into the returned value.
/// @param beta Current beta value (alpha-beta only). Lets Renju skip the deeper, more expensive
///     mate types unless detecting one could still raise beta.
/// @return VALUE_ZERO when nothing conclusive is found. Otherwise a mate score (>= VALUE_MATE or
///     <= -VALUE_MATE) whose magnitude encodes the number of plies to the end of the game.
template <Rule R>
inline Value quickWinCheck(const Board &board, int ply, Value beta = VALUE_INFINITE)
{
    Color self = board.sideToMove(), oppo = ~self;

    // We complete a five with this move.
    if (board.p4Count(self, A_FIVE))
        return mate_in(ply + 1);

    // Opponent threatens a five next move: we can parry a single one, but two or more is a loss.
    switch (board.p4Count(oppo, A_FIVE)) {
    case 0: break;
    case 1: return VALUE_ZERO;
    default: return mated_in(ply + 2);
    }

    // A flex four is an unstoppable double threat: we win in two of our moves.
    if (board.p4Count(self, B_FLEX4))
        return mate_in(ply + 3);

    auto block4Count = [&](Color side) {
        return board.p4Count(side, C_BLOCK4_FLEX3) + board.p4Count(side, D_BLOCK4_PLUS)
               + board.p4Count(side, E_BLOCK4);
    };

    // NOTE: a symmetric "opponent has flex four on multiple lines and we have nothing above a
    // block four, so we lose in three" rule is intentionally absent. The pattern counts cannot
    // distinguish two flex fours sharing one line from two on different lines, so this check
    // would misfire on the shared-line case; detecting it is left to the search.

    // Check the deeper, costlier mate types only for Freestyle/Standard, or in Renju when
    // detecting one could still raise beta.
    if (R != Rule::RENJU || mate_in(ply + 5) >= beta) {
        // C_BLOCK4_FLEX3 winning type: a block four that simultaneously makes a flex three.
        if (int c_count = board.p4Count(self, C_BLOCK4_FLEX3)) {
            // If the opponent has no block-four reply of their own, the threat is unstoppable.
            if ((R != Rule::RENJU || self == WHITE) && block4Count(oppo) == 0)
                return mate_in(ply + 5);

            // Otherwise play out each C_BLOCK4_FLEX3 candidate and check statically whether the
            // opponent's forced block-five reply also defends the follow-up flex three.
            FOR_EVERY_CAND_POS(&board, pos)
            {
                if (board.cell(pos).pattern4[self] != C_BLOCK4_FLEX3)
                    continue;

                bool hasDefend, isFakeCMove;
                {
                    ScopedMove<R> probe(board, pos);
                    Pos           defendMove = board.stateInfo().lastPattern4(self, A_FIVE);
                    Pattern4      defendP4   = board.cell(defendMove).pattern4[oppo];
                    hasDefend                = defendP4 >= E_BLOCK4
                                || R == Rule::RENJU && defendP4 == FORBID
                                       && !board.checkForbiddenPoint(defendMove);
                    // In Renju the flex three may rely on a follow-up flex four that is itself
                    // forbidden, making this C move a false threat.
                    isFakeCMove = R == Rule::RENJU && board.p4Count(self, B_FLEX4) == 0;
                }

                if (!hasDefend && !isFakeCMove)
                    return mate_in(ply + 5);

                if (--c_count == 0)
                    goto check_flex3_2x;
            }
        }

    check_flex3_2x:
        // F_FLEX3_2X winning type: a double flex three is unstoppable if the opponent has no
        // block-four or flex-four reply to interrupt it first.
        if (board.p4Count(self, F_FLEX3_2X)) {
            if (board.p4Count(oppo, B_FLEX4) + block4Count(oppo) == 0)
                return mate_in(ply + 5);
        }
    }

    return VALUE_ZERO;
}

/// Runtime-rule dispatch wrapper around the rule-templated quickWinCheck.
inline Value
quickWinCheck(Rule rule, const Board &board, int ply, Value beta = VALUE_MATE_IN_MAX_PLY)
{
    switch (rule) {
    default:
    case FREESTYLE: return quickWinCheck<FREESTYLE>(board, ply, beta);
    case STANDARD: return quickWinCheck<STANDARD>(board, ply, beta);
    case RENJU: return quickWinCheck<RENJU>(board, ply, beta);
    }
}
