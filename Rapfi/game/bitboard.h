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

#include "../core/platform.h"
#include "../core/pos.h"
#include "candarea.h"

#include <cassert>
#include <cstdint>

/// A one-bit-per-cell set over the whole padded board address space, indexed directly by `Pos`
/// (`pos >> 6` is the word, `pos & 63` the bit). Backs the on-board, empty, and candidate cell
/// sets so each can be iterated with pop_lsb instead of a full-board branch scan. Trivially
/// copyable, so the candidate set rides the per-ply StateInfo snapshot.
struct Bitboard
{
    static constexpr int NumWords = FULL_BOARD_CELL_COUNT / 64;
    static_assert(FULL_BOARD_CELL_COUNT % 64 == 0, "address space must be a multiple of 64 bits");
    // A board row is one half of a 64-bit word (two rows per word); the stencil and buildCandSet
    // ops rely on this 32-wide layout.
    static_assert(FULL_BOARD_SIZE == 32, "bitboard ops assume a 32-wide layout");

    uint64_t words[NumWords];

    void zero()
    {
        for (int i = 0; i < NumWords; i++)
            words[i] = 0;
    }
    void set(Pos pos) { words[pos >> 6] |= uint64_t(1) << (pos & 63); }
    void clear(Pos pos) { words[pos >> 6] &= ~(uint64_t(1) << (pos & 63)); }
    bool test(Pos pos) const { return (words[pos >> 6] >> (pos & 63)) & 1; }

    /// Number of set bits (population count). For debug consistency assertions only.
    int count() const
    {
        int n = 0;
        for (int i = 0; i < NumWords; i++)
            n += popcount(words[i]);
        return n;
    }

    /// Set this bitboard to the intersection `a & b`, word by word.
    void setIntersect(const Bitboard &a, const Bitboard &b)
    {
        for (int i = 0; i < NumWords; i++)
            words[i] = a.words[i] & b.words[i];
    }

    /// Build into this bitboard the empty candidate cells inside the candidate box `area`: `cand &
    /// empty`, with bits outside the box (and an inverted/empty box) cleared. Fuses the empty-AND
    /// and the box clip into a single pass, so each word is read from the sources and written here
    /// once.
    void buildCandSet(const Bitboard &cand, const Bitboard &empty, const CandArea &area)
    {
        const int x0 = area.x0, y0 = area.y0, x1 = area.x1, y1 = area.y1;
        if (x0 > x1 || y0 > y1) {  // inverted (empty) box -> no candidates
            zero();
            return;
        }
        // Columns map to a fixed bit run within each row's 32-bit half; rows select word + half.
        const uint64_t colMask = ((uint64_t(1) << (x1 - x0 + 1)) - 1) << (x0 + BOARD_BOUNDARY);
        const int      loRow = y0 + BOARD_BOUNDARY, hiRow = y1 + BOARD_BOUNDARY;
        for (int w = 0; w < NumWords; w++) {
            uint64_t m = 0;
            if (2 * w >= loRow && 2 * w <= hiRow)
                m |= colMask;  // even row -> low half of the word
            if (2 * w + 1 >= loRow && 2 * w + 1 <= hiRow)
                m |= colMask << FULL_BOARD_SIZE;  // odd row -> high half
            words[w] = cand.words[w] & empty.words[w] & m;
        }
    }

    /// A candidate range (the neighbour offsets around a played stone) recast as one dx-bitmask per
    /// board row (index `dy + 4`, bit `dx + 4`), so applyStencil() can OR the whole neighbourhood
    /// in as a few word-wide ops. Each stencil row fits within a single bitboard word.
    struct Stencil
    {
        uint32_t row[9];

        /// Build from a flat offset table, where offset == dy*FULL_BOARD_SIZE + dx, dx/dy in [-4,
        /// 4].
        void build(const Direction *offsets, size_t count)
        {
            for (int k = 0; k < 9; k++)
                row[k] = 0;
            for (size_t i = 0; i < count; i++) {
                int off = int(offsets[i]);
                int t   = off + 4;  // recover dy via floored division, then dx
                int dy =
                    t >= 0 ? t / FULL_BOARD_SIZE : -((-t + FULL_BOARD_SIZE - 1) / FULL_BOARD_SIZE);
                int dx = t - dy * FULL_BOARD_SIZE - 4;
                assert(-4 <= dy && dy <= 4 && -4 <= dx && dx <= 4);
                row[dy + 4] |= 1u << (dx + 4);
            }
        }
    };

    /// OR the stencil's candidate neighbourhood, centred at `center`, into this bitboard.
    void applyStencil(const Stencil &stencil, Pos center)
    {
        const int cx = center.x() + BOARD_BOUNDARY;
        const int ry = center.y() + BOARD_BOUNDARY;
        for (int k = 0; k < 9; k++) {
            uint64_t rowBits = stencil.row[k];
            if (!rowBits)
                continue;
            int r = ry + k - 4;  // board row; r >> 1 selects the word, (r & 1) its half
            words[r >> 1] |= rowBits << ((r & 1) * FULL_BOARD_SIZE + cx - 4);
        }
    }

    /// Forward cursor over the set bits, yielding positions in ascending `Pos` order. `next()`
    /// returns `Pos::NONE` once exhausted, which is unambiguous because real candidate cells are
    /// always inside the playable region (`Pos` well above zero). Holds a pointer into the source
    /// bitboard, so it must not outlive it; the FOR_EVERY_* macros only construct it as a
    /// stack-local over a board member (onBoardBB/emptyBB) or a stack-local snapshot.
    struct Cursor
    {
        const uint64_t *words;
        uint64_t        bits;
        int             wordIdx;

        explicit Cursor(const Bitboard &bb) : words(bb.words), bits(bb.words[0]), wordIdx(0) {}

        Pos next()
        {
            while (bits == 0) {
                if (++wordIdx >= NumWords)
                    return Pos::NONE;
                bits = words[wordIdx];
            }
            return Pos(int16_t(wordIdx * 64 + pop_lsb(bits)));
        }
    };
};
