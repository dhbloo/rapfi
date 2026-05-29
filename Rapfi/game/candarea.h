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

#include <algorithm>
#include <cstdint>

/// An axis-aligned bounding rectangle of the cells worth considering as move candidates. Starts
/// empty (inverted bounds) and only ever grows via expand().
struct CandArea
{
    int8_t x0, y0, x1, y1;

    CandArea() : x0(INT8_MAX), y0(INT8_MAX), x1(INT8_MIN), y1(INT8_MIN) {}
    CandArea(int8_t x0, int8_t y0, int8_t x1, int8_t y1) : x0(x0), y0(y0), x1(x1), y1(y1) {}

    /// Grow the area to include the square of half-width `dist` centered on `pos`, clamped to the
    /// board.
    void expand(Pos pos, int boardSize, int dist)
    {
        int x = pos.x(), y = pos.y();

        x0 = std::min((int)x0, std::max(x - dist, 0));
        y0 = std::min((int)y0, std::max(y - dist, 0));
        x1 = std::max((int)x1, std::min(x + dist, boardSize - 1));
        y1 = std::max((int)y1, std::min(y + dist, boardSize - 1));
    }
};
