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

#include "../config.h"
#include "../core/types.h"

class Board;

namespace Evaluation {

template <Rule R>
Value evaluate(const Board &board, Value alpha = -VALUE_INFINITE, Value beta = VALUE_INFINITE);
Value evaluate(const Board &board, Rule rule);

class ValueType;
ValueType computeEvaluatorValue(const Board &board);

/// EvalInfo struct contains all information needed to evaluate a position.
struct EvalInfo
{
    struct
    {
        uint16_t pcodeCount[SIDE_NB][PCODE_NB];
    } plyBack[2];
    Color self;
    int   threatMask;

    EvalInfo(const Board &board, Rule rule);
};

}  // namespace Evaluation
