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

/// Quick check if current position is winning or lossing.
/// @param board The board state of a position to check.
/// @param ply The current search ply (ply in the search tree).
/// @param beta Current beta value (only useful in alpha-beta search).
///     This is used to relax some checking conditions in Renju rule.
/// @return VALUE_ZERO if no win/loss is found. Otherwise, a value that larger
///     or smaller than VALUE_MATE/-VALUE_MATE is returned, which also contains
///     the number of plies to end the game.
template <Rule Rule>
inline Value quickWinCheck(const Board &board, int ply, Value beta = VALUE_INFINITE)
{
    Color self = board.sideToMove(), oppo = ~self;

    // We win at this move by five connection
    if (board.p4Count(self, A_FIVE))
        return mate_in(ply + 1);

    // Opponent has a five connection next move, we can defend it if there
    // is only one A_FIVE, while we lose if there are more than one A_FIVE.
    switch (board.p4Count(oppo, A_FIVE)) {
    case 0: break;
    case 1: return VALUE_ZERO;
    default: return mated_in(ply + 2);
    }

    // If we have a B_FLEX4 this move, we win in 2 steps
    if (board.p4Count(self, B_FLEX4))
        return mate_in(ply + 3);

    auto block4Count = [&](Color side) {
        return board.p4Count(side, C_BLOCK4_FLEX3) + board.p4Count(side, D_BLOCK4_PLUS)
               + board.p4Count(side, E_BLOCK4);
    };

    // FIXME: bug when multiple b4 are in the same line!
    // If opponent has more than a line of flex4, and we do not have any attack moves
    // with priority higher than B4, then we can not defend and will lose in 3 steps.
    /*
    if (board.p4Count(oppo, B_FLEX4) > 2 && b4Count(self) == 0)
        return mated_in(ply + 3);
    */

    // Check further mate types only in Freestyle and Standard, or we can improve beta in Renju.
    if (Rule != Rule::RENJU || mate_in(ply + 5) >= beta) {
        // Check C_BLOCK4_FLEX3 winning type
        if (int c_count = board.p4Count(self, C_BLOCK4_FLEX3)) {
            // If opponent has not B4 move, we simply win in 4 steps
            if ((Rule != Rule::RENJU || self == WHITE) && block4Count(oppo) == 0)
                return mate_in(ply + 5);

            // Fast static check for opponent defence at C_BLOCK4_FLEX3 move
            Board &b = const_cast<Board &>(board);  // Get a mutable reference of board
            FOR_EVERY_CAND_POS(&b, pos)
            {
                if (b.cell(pos).pattern4[self] != C_BLOCK4_FLEX3)
                    continue;

                b.move<Rule, Board::MoveType::NO_EVAL>(pos);
                Pos      defendMove  = b.stateInfo().lastPattern4(self, A_FIVE);
                Pattern4 defendP4    = b.cell(defendMove).pattern4[oppo];
                bool     isFakeCMove = Rule == Rule::RENJU && b.p4Count(self, B_FLEX4) == 0;
                b.undo<Rule, Board::MoveType::NO_EVAL>();

                if (!isFakeCMove && defendP4 < E_BLOCK4)
                    return mate_in(ply + 5);

                if (--c_count == 0)
                    goto check_flex3_2x;
            }
        }

    check_flex3_2x:
        // Check F_FLEX3_2X winning type
        if (board.p4Count(self, F_FLEX3_2X)) {
            if (board.p4Count(oppo, B_FLEX4) + block4Count(oppo) == 0)
                return mate_in(ply + 5);
        }
    }

    return VALUE_ZERO;
}

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
