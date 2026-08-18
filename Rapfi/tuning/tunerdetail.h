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

#include "../core/random.h"
#include "tunedigest.h"
#include "tuner.h"

#include <cmath>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>

namespace Tuning::detail {

inline constexpr Float  CoeffScale = 8;
inline constexpr size_t MiB        = 1024 * 1024;

inline constexpr size_t WorstTermsPerSample = 4 * MAX_MOVES + 1;
inline constexpr size_t WorstPreparedSampleBytes =
    32 + WorstTermsPerSample * sizeof(TuneCoeff)
    + MAX_MOVES * (sizeof(PolicyCandidate) + sizeof(PolicyTargetTerm));
inline constexpr size_t PreparedSampleCredit = 4 * WorstPreparedSampleBytes;

inline size_t checkedProduct(size_t lhs, size_t rhs, const char *what)
{
    if (lhs != 0 && rhs > std::numeric_limits<size_t>::max() / lhs)
        throw std::length_error(std::string(what) + " byte count overflows size_t");
    return lhs * rhs;
}

inline void hashUint64(Sha256 &hasher, uint64_t value)
{
    uint8_t encoded[8];
    for (size_t i = 0; i < 8; i++) {
        encoded[i] = static_cast<uint8_t>(value);
        value >>= 8;
    }
    hasher.update(encoded, sizeof(encoded));
}

inline void hashString(Sha256 &hasher, const std::string &value)
{
    hashUint64(hasher, value.size());
    hasher.update(value.data(), value.size());
}

inline void hashFloat(Sha256 &hasher, float value)
{
    uint32_t bits;
    static_assert(sizeof(bits) == sizeof(value), "float must have a 32-bit representation");
    std::memcpy(&bits, &value, sizeof(bits));
    uint8_t encoded[4];
    for (size_t i = 0; i < 4; i++) {
        encoded[i] = static_cast<uint8_t>(bits);
        bits >>= 8;
    }
    hasher.update(encoded, sizeof(encoded));
}

inline uint64_t domainSeed(uint64_t seed, uint64_t domain)
{
    PRNG mixer(seed ^ domain);
    return mixer();
}

inline TuneParam encodeIntegerForTruncatingExport(Score score, Float scale, Float bias)
{
    TuneParam nearest      = TuneParam((Float(score) - bias) / scale);
    TuneParam candidates[] = {
        nearest,
        std::nextafter(nearest, -std::numeric_limits<TuneParam>::infinity()),
        std::nextafter(nearest, std::numeric_limits<TuneParam>::infinity()),
    };

    for (TuneParam candidate : candidates) {
        Float reconstructed = Float(candidate) * scale + bias;
        if (std::isfinite(candidate) && std::trunc(reconstructed) == Float(score))
            return candidate;
    }

    throw std::runtime_error(
        "move-score scale and bias cannot represent an integer score without drift");
}

}  // namespace Tuning::detail
