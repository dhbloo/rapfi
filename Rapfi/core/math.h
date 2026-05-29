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

#include <cstdint>

/// Number of elements in a C-style array. Compile-time, no runtime cost.
template <typename ElemType, int Size>
constexpr int arraySize(ElemType (&arr)[Size])
{
    return Size;
}

/// `base` raised to a non-negative integer `exponent` via exponentiation by squaring.
template <class T>
constexpr T power(T base, unsigned exponent)
{
    T result = 1;
    while (exponent > 0) {
        if (exponent & 1u)
            result *= base;
        base *= base;
        exponent >>= 1;
    }
    return result;
}

/// True iff `x` is exactly a power of two. `isPowerOfTwo(0)` returns true.
constexpr bool isPowerOfTwo(uint64_t x)
{
    return (x & (x - 1)) == 0;
}

/// Floor of the base-2 logarithm of a positive integer. `floorLog2(1)` is 0.
/// Undefined for `x == 0`.
constexpr uint64_t floorLog2(uint64_t x)
{
    return x == 1 ? 0 : 1 + floorLog2(x >> 1);
}

/// Number of multisubsets of size `m` drawn from `n` kinds, i.e. `C(n + m - 1, m)`.
/// Used to size the per-cell pattern combination table (m=4 directions).
constexpr uint32_t combineNumber(uint32_t n, uint32_t m)
{
    uint64_t r = 1;
    for (unsigned int i = n; i < m + n; i++)
        r *= i;
    for (unsigned int i = 2; i <= m; i++)
        r /= i;
    return (uint32_t)r;
}
