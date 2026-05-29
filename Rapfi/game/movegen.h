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

class Board;

/// GenType is a bitmask selecting which moves a generator emits. The low two bits hold a `Rule`
/// selector (the RULE_* values); the higher bits are independent flags combined with operator|.
enum GenType {
    RULE_ALL       = 0,  ///< Rule selector (low two bits): rule-agnostic.
    RULE_FREESTYLE = 1,  ///< Rule selector: freestyle.
    RULE_STANDARD  = 2,  ///< Rule selector: standard.
    RULE_RENJU     = 3,  ///< Rule selector: renju.

    COMB = 4,  ///< Require a combination (multi-direction) pattern rather than a single line.

    WINNING = 8,    ///< Immediate win or forced-win threat (A_FIVE / B_FLEX4).
    VCF     = 16,   ///< Four-threat moves (block fours).
    VCT     = 32,   ///< Three-threat moves (flex threes).
    VC2     = 64,   ///< Two-threat moves (flex twos).
    TRIVIAL = 128,  ///< Any remaining candidate move.

    DEFEND_FIVE = 256,   ///< Only A_FIVE defence moves.
    DEFEND_FOUR = 512,   ///< Only B_FLEX4 defence moves.
    DEFEND_B4F3 = 1024,  ///< Only C_BLOCK4_FLEX3 defence moves.
    DEFEND      = DEFEND_FIVE | DEFEND_FOUR | DEFEND_B4F3,

    ALL = WINNING | VCF | VCT | VC2 | TRIVIAL,  ///< Every move, ignoring defence obligations.
};

/// Combine two GenType masks into one.
constexpr GenType operator|(GenType a, GenType b)
{
    return GenType(int(a) | int(b));
}

/// A candidate move paired with the scores used to order it during search.
struct ScoredMove
{
    Pos   pos;
    Score score;  ///< Sorting key: raw score blended with history and other heuristics.
    union {
        Score rawScore;  ///< Raw score from the pattern score table or evaluator.
        float policy;    ///< Normalized policy prior from the neural network.
    };

         operator Pos() const { return pos; }
         operator float() const = delete;  // Inhibit unwanted implicit conversions
    void operator=(Pos p) { pos = p; }

    struct ScoreComparator
    {
        bool operator()(const ScoredMove &a, const ScoredMove &b) const
        {
            return a.score > b.score;
        }
    };

    struct PolicyComparator
    {
        bool operator()(const ScoredMove &a, const ScoredMove &b) const
        {
            return a.policy > b.policy;
        }
    };
};

/// Generate every candidate move satisfying the given GenType. Move scores are left uninitialized.
/// @param board The current board state to generate moves from.
/// @param moveList The begin cursor of an empty move list.
/// @tparam Type The set of move categories to emit.
/// @return The end cursor of the move list (one past the last element).
template <GenType Type>
ScoredMove *generate(const Board &board, ScoredMove *moveList);

/// Like generate(), but restricted to a fixed set of neighbor offsets around a center cell.
/// @param board The current board state to generate moves from.
/// @param moveList The begin cursor of an empty move list.
/// @param center The center pos whose neighborhood is scanned.
/// @param neighbors Array of direction offsets relative to `center`.
/// @param numNeighbors Length of the `neighbors` array.
/// @tparam Type The set of move categories to emit.
/// @return The end cursor of the move list (one past the last element).
template <GenType Type>
ScoredMove *generateNeighbors(const Board     &board,
                              ScoredMove      *moveList,
                              Pos              center,
                              const Direction *neighbors,
                              size_t           numNeighbors);

/// Check whether the opponent's C_BLOCK4_FLEX3 move is a genuine threat under Renju (a black
/// flex-three follow-up may be forbidden, making the move only a pseudo-threat).
/// @return True if the threat is real, otherwise false.
bool validateOpponentCMove(const Board &board);
