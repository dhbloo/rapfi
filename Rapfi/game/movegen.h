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

/// GenType specifies which pattern types are allowed to generate.
enum GenType {
    RULE_ALL       = 0,
    RULE_FREESTYLE = 1,
    RULE_STANDARD  = 2,
    RULE_RENJU     = 3,

    COMB = 4,  // Move must be a conbination pattern type

    WINNING = 8,
    VCF     = 16,
    VCT     = 32,
    VC2     = 64,
    TRIVIAL = 128,

    DEFEND_FIVE = 256,   // Only generate A_FIVE defence move
    DEFEND_FOUR = 512,   // Only generate B_FLEX4 defence move
    DEFEND_B4F3 = 1024,  // Only generate C_BLOCK4_FLEX3 defence move
    DEFEND      = DEFEND_FIVE | DEFEND_FOUR | DEFEND_B4F3,

    ALL = WINNING | VCF | VCT | VC2
          | TRIVIAL  // Generate all moves, no matter whether needs to defend
};

/// Combine two GenType mask into one.
constexpr GenType operator|(GenType a, GenType b)
{
    return GenType(int(a) | int(b));
}

/// ScoredMove struct contains a pos and its score, used for move sorting.
struct ScoredMove
{
    Pos   pos;
    Score score;  /// Score with history and other heruistics that is used for sorting
    union {
        Score rawScore;  /// Raw score from score table or evaluator
        float policy;    /// Normalized policy score from neural network
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

/// Generate moves satisfying the given GenType, and stores them
/// in the move list. Scores of generated moves are all set to zero.
/// @param board The current board state to generate moves.
/// @param moveList The begin cursor of an empty move list.
/// @tparam Type ScoredMove generation type.
/// @return The end cursor of move list (next of the last element).
template <GenType Type>
ScoredMove *generate(const Board &board, ScoredMove *moveList);

/// Generate moves satisfying the given GenType, while only generate
/// neighbor moves around the last move of the current side.
/// @param board The current board state to generate moves.
/// @param moveList The begin cursor of an empty move list.
/// @param center The center pos on board to generate moves.
/// @param neighbors A pointer to an array of pos offsets.
/// @param numNeighbors The length of pos offsets array.
/// @tparam Type ScoredMove generation type.
/// @return The end cursor of move list (next of the last element).
template <GenType Type>
ScoredMove *generateNeighbors(const Board     &board,
                              ScoredMove      *moveList,
                              Pos              center,
                              const Direction *neighbors,
                              size_t           numNeighbors);

/// Validate opponent C_BLOCK4_FLEX3 type move is real threat in Renju.
/// @return True if threat is real, otherwise false.
bool validateOpponentCMove(const Board &board);
