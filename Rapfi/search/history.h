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

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <type_traits>
#include <utility>

class Board;  // forward declaration

namespace Search {

namespace detail {

    /// Builds a nested native array type: `MultiDimNativeArray<T, A, B>::type` is `T[A][B]`.
    template <class T, size_t Size, size_t... Sizes>
    struct MultiDimNativeArray
    {
        using Nested = typename MultiDimNativeArray<T, Sizes...>::type;
        using type   = Nested[Size];
    };

    template <class T, size_t Size>
    struct MultiDimNativeArray<T, Size>
    {
        using type = T[Size];
    };

}  // namespace detail

/// Multi-dimensional native array, e.g. `MDNativeArray<int, 3, 4>` is `int[3][4]`. Used as the
/// backing store of HistTable below (its only consumer).
template <class T, size_t... Sizes>
using MDNativeArray = typename detail::MultiDimNativeArray<T, Sizes...>::type;

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
        void   operator=(const ValueT &v) { value = v; }
               operator const ValueT &() const { return value; }
        ValueT get() const { return value; }
        void   operator<<(int bonus)
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
        // `table` is a contiguous block of Entry; fill it through Entry's own assignment
        // operator rather than reinterpreting it as the raw value type.
        Entry *first = reinterpret_cast<Entry *>(table);
        std::fill_n(first, sizeof(table) / sizeof(Entry), fillValue);
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
