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

#include "time.h"

#include <cstdint>

/// Fast 64-bit PRNG based on SplitMix64. Seedable, cheap, not cryptographic.
/// See <https://xoroshiro.di.unimi.it/splitmix64.c>.
class PRNG
{
public:
    using result_type = uint64_t;

    static constexpr uint64_t min() { return 0; }
    static constexpr uint64_t max() { return UINT64_MAX; }

    /// Construct with an explicit seed. Same seed always yields the same sequence.
    explicit PRNG(uint64_t seed) : x(seed) {}

    /// A generator seeded from the current time, for callers that intentionally want
    /// non-reproducible output. Made explicit so deterministic contexts cannot get a
    /// time seed by accident.
    static PRNG nondeterministic() { return PRNG(now()); }

    uint64_t operator()()
    {
        uint64_t z = (x += 0x9e3779b97f4a7c15);
        z          = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9;
        z          = (z ^ (z >> 27)) * 0x94d049bb133111eb;
        return z ^ (z >> 31);
    }

private:
    uint64_t x;
};
