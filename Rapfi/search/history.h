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
#include "../core/utils.h"

#include <cassert>
#include <limits>
#include <type_traits>

class Board;  // forward declaration

namespace Search {

/// HistTable is a handy multi-demensional array to store history statistics.
/// @tparam ValueT The base value type of the array
/// @tparam Range The range [-Range, Range] of the value
/// @tparam Shape The size list of multiple dimensions, at least have one.
template <typename ValueT, int Range, int... Shape>
struct HistTable
{
    /// Entry stores a single value in the table. It acts as a wrapper around the value
    /// and overloads operator<<() function to ensure that values does not go out of bound.
    struct Entry
    {
        void    operator=(const ValueT &v) { value = v; }
        ValueT *operator&() { return &value; }
                operator const ValueT &() const { return value; }
        ValueT  get() const { return value; }
        void    operator<<(int bonus)
        {
            static_assert(Range <= std::numeric_limits<ValueT>::max());
            assert(std::abs(bonus) <= Range);  // Ensure bonus is in [-Range, Range]
            value += bonus - value * std::abs(bonus) / Range;
            assert(std::abs(value) <= Range);
        }

    private:
        ValueT value;
    };

    auto       &operator[](std::size_t index) { return table[index]; }
    const auto &operator[](std::size_t index) const { return table[index]; }
    void        init(const ValueT &fillValue)
    {
        ValueT *p = reinterpret_cast<ValueT *>(table);
        std::fill_n(p, sizeof(table) / sizeof(ValueT), fillValue);
    }

private:
    MDNativeArray<Entry, Shape...> table;
};

/// The type of a main history record.
enum MoveHistoryType { HIST_ATTACK, HIST_QUIET, MAIN_HIST_TYPE_NB };

/// MainHistory records how often a certain type of move has been successful or unsuccessful
/// (causing a beta cutoff) during the current search. It is indexed by color of the move,
/// move's position, and the move's history type.
typedef HistTable<int16_t, 10692, SIDE_NB, FULL_BOARD_CELL_COUNT, MAIN_HIST_TYPE_NB> MainHistory;

/// CounterMoveHistory records a natural response of moves irrespective of the actual position.
/// It is indexed by color of the previous move, previous move's position and current move's type.
typedef HistTable<std::pair<Pos, Pattern4>, 0, SIDE_NB, MAX_MOVES> CounterMoveHistory;

}  // namespace Search
