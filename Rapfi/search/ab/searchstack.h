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

#include "../../core/pos.h"
#include "../../core/types.h"

#include <cassert>
#include <vector>

namespace Search::AB {

/// SearchStack struct keeps track of the information we need between nodes in the
/// tree during the search. Each search thread has its own array of SearchStack
/// objects, indexed by the current ply.
struct SearchStack
{
    Pos *const pv;
    const int  ply;
    int        moveCount;
    Depth      extraExtension;  /// cumulative extension depth that larger than one ply
    int        dbValueDepth;
    int        statScore;
    Value      staticEval;
    Pos        currentMove;
    Pos        skipMove;
    Pos        killers[2];
    Pattern4   moveP4[SIDE_NB];
    int16_t    numNullMoves;
    bool       ttPv;
    bool       dbChildWritten;

    /// Append current move and to the end of child PV.
    void updatePv(Pos move)
    {
        Pos *thisPv  = this->pv;
        Pos *childPv = (this + 1)->pv;
        assert(thisPv && childPv);

        *thisPv++ = move;
        while (*childPv != Pos::NONE)
            *thisPv++ = *childPv++;
        *thisPv = Pos::NONE;
    }

    /// Check whether a move is killer at this ply.
    bool isKiller(Pos move) const { return move == killers[0] || move == killers[1]; }
    /// Update killer heruistic move.
    void setKiller(Pos move)
    {
        if (killers[0] != move) {
            killers[1] = killers[0];
            killers[0] = move;
        }
    }
};

/// StackArray class allocates and inits the entire stacks and triangular
/// pv-table up to max ply with init static evaluation.
class StackArray : std::vector<SearchStack>
{
public:
    static constexpr int plyBeforeRoot = 4;
    static constexpr int plyAfterMax   = 2;

    StackArray(int maxPly, Value initStaticEval) : triPvTable((maxPly + 1) * (maxPly + 2) / 2)
    {
        auto nextTriPvIndex = [&, idx = 0](int ply) mutable -> Pos * {
            Pos *curPv = nullptr;
            if (ply >= 0 && ply <= maxPly) {
                assert(0 <= idx && idx < triPvTable.size());
                curPv = &triPvTable[idx];
                idx += maxPly + 1 - ply;
            }
            return curPv;
        };

        reserve(maxPly + plyBeforeRoot + plyAfterMax);
        for (int i = -plyBeforeRoot; i < maxPly + plyAfterMax; i++)
            push_back(SearchStack {nextTriPvIndex(i), i});

        // Initialize static evaluation for plies before root
        Value staticEval = initStaticEval;
        for (int i = plyBeforeRoot; i >= 0; i--) {
            (*this)[i].staticEval = staticEval;
            staticEval            = -staticEval;
        }
    }
    SearchStack *rootStack() { return &(*this)[plyBeforeRoot]; }

private:
    std::vector<Pos> triPvTable;
};

}  // namespace Search::AB
