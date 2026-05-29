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

#include "pattern.h"

#include "../config.h"
#include "../core/utils.h"

#include <array>
#include <cassert>
#include <tuple>

namespace {

using PatternConfig::HalfLineLen;
using PatternConfig::KeyCnt;

/// One cell's state from the perspective of the side whose pattern is being computed.
enum ColorFlag { SELF, OPPO, EMPT };

/// Length of the line examined to classify a pattern: the center cell plus both half-lines
/// (e.g. OOOO_OOOO for freestyle).
template <Rule R>
constexpr int LineLen = HalfLineLen<R> * 2 + 1;

/// A straight line of color flags centered on the cell under consideration.
template <Rule R>
struct Line : std::array<ColorFlag, LineLen<R>>
{
    explicit Line() = default;

    /// Decode a line from a raw 64-bit line key, classifying each cell relative to `self`.
    explicit Line(uint64_t key, Color self);
};

/// Memo table for the (Line -> Pattern) dynamic program, indexed by the line's base-3 encoding.
template <Rule R>
struct PatternMemo : std::array<Pattern, power(3, LineLen<R>)>
{
    static constexpr Pattern NilPat = Pattern(0xff);  ///< Sentinel for a not-yet-computed entry.

    PatternMemo() { this->fill(NilPat); }
    Pattern &get(const Line<R> &line);
};

/// Measure the self-run through the center: returns (realLen, fullLen, start, end), where realLen
/// counts contiguous self stones, fullLen spans up to the first blocking opponent stone on each
/// side, and [start, end] are the inclusive bounds of that span.
template <Rule R>
constexpr std::tuple<int, int, int, int> countLine(const Line<R> &line);

/// Re-center a line so that index `i` becomes the middle cell (off-line cells become OPPO).
template <Rule R>
constexpr Line<R> shiftLine(const Line<R> &line, int i);

/// Classify a line into its `Pattern` via the dynamic program described at the definition.
template <Rule R, Color Side>
constexpr Pattern getPattern(PatternMemo<R> &patMemo, const Line<R> &line);

/// Fill a (fused line key -> Pattern2x) lookup table.
template <Rule R>
void fillPattern2xLUT(Pattern2x pattern2x[KeyCnt<R>]);

/// Combine four directional line patterns into the cell's `Pattern4`.
/// @tparam Forbid Whether to flag renju forbidden points (overline / double-four / double-three).
template <bool Forbid>
Pattern4 getPattern4(Pattern p1, Pattern p2, Pattern p3, Pattern p4);

/// Fill a (pattern code -> Pattern4 score) lookup table.
template <bool Forbid>
void fillPattern4LUT(Pattern4Score p4Score[PCODE_NB]);

/// Fill the (four line patterns -> pattern code) lookup table, assigning each unordered
/// combination of four patterns a dense, order-independent code.
void fillPatternCodeLUT(PatternCode pcode[PATTERN_NB][PATTERN_NB][PATTERN_NB][PATTERN_NB]);

/// Fill a (fused line key + attacker -> defence-move bitmask) lookup table.
template <Rule R>
void fillDefenceLUT(uint8_t defence[KeyCnt<R>][2]);

}  // namespace

namespace PatternConfig {

Pattern2x   PATTERN2x[KeyCnt<FREESTYLE>];
Pattern2x   PATTERN2xStandard[KeyCnt<STANDARD>];
Pattern2x   PATTERN2xRenju[KeyCnt<RENJU>];
PatternCode PCODE[PATTERN_NB][PATTERN_NB][PATTERN_NB][PATTERN_NB];
uint8_t     DEFENCE[KeyCnt<FREESTYLE>][2];
uint8_t     DEFENCEStandard[KeyCnt<STANDARD>][2];
uint8_t     DEFENCERenju[KeyCnt<RENJU>][2];

// Populate every lookup table before main() runs. PCODE must come first because fillPattern4LUT
// reads it to map each pattern combination to its dense code.
[[maybe_unused]] static const bool init = []() {
    fillPatternCodeLUT(PCODE);

    fillPattern2xLUT<FREESTYLE>(PATTERN2x);
    fillPattern2xLUT<STANDARD>(PATTERN2xStandard);
    fillPattern2xLUT<RENJU>(PATTERN2xRenju);

    fillPattern4LUT<false>(Config::P4SCORES[FREESTYLE]);
    fillPattern4LUT<false>(Config::P4SCORES[STANDARD]);
    fillPattern4LUT<true>(Config::P4SCORES[RENJU + BLACK]);
    fillPattern4LUT<false>(Config::P4SCORES[RENJU + WHITE]);

    fillDefenceLUT<FREESTYLE>(DEFENCE);
    fillDefenceLUT<STANDARD>(DEFENCEStandard);
    fillDefenceLUT<RENJU>(DEFENCERenju);
    return true;
}();

}  // namespace PatternConfig

namespace {

template <Rule R>
Line<R>::Line(uint64_t key, Color self)
{
    assert(self == BLACK || self == WHITE);
    constexpr auto Mid = HalfLineLen<R>;

    // Each cell is 2 bits (bit BLACK = bit 0, bit WHITE = bit 1). An empty cell sets both bits;
    // placing a stone of color c clears bit c, so the bit left set marks the opposite color.
    // Hence 00 wall, 01 white, 10 black, 11 empty. Both bits set means empty; otherwise the cell
    // holds `self`'s stone iff the opposite color's bit is set.
    for (int i = 0; i < LineLen<R>; i++) {
        bool c[2];
        if (i < Mid) {
            c[BLACK] = (key >> (2 * i)) & (0x1 + BLACK);
            c[WHITE] = (key >> (2 * i)) & (0x1 + WHITE);
        }
        else if (i > Mid) {
            c[BLACK] = (key >> (2 * (i - 1))) & (0x1 + BLACK);
            c[WHITE] = (key >> (2 * (i - 1))) & (0x1 + WHITE);
        }

        if (i == Mid)
            (*this)[i] = SELF;
        else
            (*this)[i] = c[BLACK] && c[WHITE] ? EMPT : c[~self] ? SELF : OPPO;
    }
}

template <Rule R>
Pattern &PatternMemo<R>::get(const Line<R> &line)
{
    // Each cell holds one of three flags, so the line packs into a base-3 integer in [0,
    // 3^LineLen).
    uint32_t code = 0;
    for (int i = 0; i < LineLen<R>; i++) {
        code = code * 3 + uint32_t(line[i]);
    }
    return (*this)[code];
}

template <Rule R>
constexpr std::tuple<int, int, int, int> countLine(const Line<R> &line)
{
    constexpr auto Mid = LineLen<R> / 2;

    int realLen = 1, fullLen = 1;
    int realLenInc = 1;
    int start = Mid, end = Mid;

    for (int i = Mid - 1; i >= 0; i--) {
        if (line[i] == SELF)
            realLen += realLenInc;
        else if (line[i] == OPPO)
            break;
        else
            realLenInc = 0;

        fullLen++;
        start = i;
    }

    realLenInc = 1;

    for (int i = Mid + 1; i < LineLen<R>; i++) {
        if (line[i] == SELF)
            realLen += realLenInc;
        else if (line[i] == OPPO)
            break;
        else
            realLenInc = 0;

        fullLen++;
        end = i;
    }

    return std::make_tuple(realLen, fullLen, start, end);
}

template <Rule R>
constexpr Line<R> shiftLine(const Line<R> &line, int i)
{
    constexpr auto Len = LineLen<R>;

    Line<R> shiftedLine;
    for (int j = 0; j < Len; j++) {
        int idx        = j + i - Len / 2;
        shiftedLine[j] = idx >= 0 && idx < Len ? line[idx] : OPPO;
    }
    return shiftedLine;
}

template <Rule R, Color Side>
constexpr Pattern getPattern(PatternMemo<R> &patMemo, const Line<R> &line)
{
    // In order to calculate a line's pattern, we can use a dynamic programming solution: continue
    // to put stones at EMPT position, and recursively see what's the next pattern, until it
    // reaches trivial state (i.e. OL[realLen >= 6] or F5[realLen >= 5] or DEAD[fullLen < 5]).
    // Then by tracing back which max pattern we got we can infer current line's pattern.
    //
    // Classification rules:
    // 1. realLen >= 6  -> overline (OL)
    // 2. realLen == 5  -> five (F5)
    // 3. fullLen <  5  -> dead (DEAD)
    // 4. otherwise, classify by what the empty points in the span become with one more self stone:
    //    a. >= 2 five-points       -> open four (F4)
    //    b.    1 five-point        -> block four (B4)
    //    c. >= 2 open-four-points  -> connected open three (F3S)
    //    d.    1 open-four-point   -> plain open three (F3)
    //    e. a block-four-point     -> block three (B3)
    //    f.    4 open-three-points -> connected open two (F2B)
    //    g.    3 open-three-points -> jump open two (F2A)
    //    h.    2 open-three-points -> wide-jump open two (F2)
    //    i. a block-three-point    -> block two (B2)
    //    j. an open-two-point      -> open one (F1)
    //    k. a block-two-point      -> block one (B1)
    //    l. none of the above      -> dead (DEAD)

    constexpr auto Mid           = LineLen<R> / 2;
    constexpr bool CheckOverline = R == Rule::STANDARD || (R == Rule::RENJU && Side == BLACK);

    Pattern &patMemorized = patMemo.get(line);
    if (patMemorized != patMemo.NilPat)
        return patMemorized;

    auto [realLen, fullLen, start, end] = countLine<R>(line);
    Pattern p                           = DEAD;

    if (CheckOverline && realLen >= 6)
        p = OL;
    else if (realLen >= 5)
        p = F5;
    else if (fullLen < 5)
        p = DEAD;
    else {
        int patCnt[PATTERN_NB] = {0};
        int f5Idx[2]           = {0};

        for (int i = start; i <= end; i++) {
            if (line[i] == EMPT) {
                Line<R> sl = shiftLine<R>(line, i);
                sl[Mid]    = SELF;

                Pattern slp = getPattern<R, Side>(patMemo, sl);
                if (slp == F5 && patCnt[F5] < arraySize(f5Idx))
                    f5Idx[patCnt[F5]] = i;
                patCnt[slp]++;
            }
        }

        if (patCnt[F5] >= 2) {
            p = F4;
            if constexpr (R == Rule::RENJU && Side == BLACK) {
                // Check if the line is composed of two B4 pattern.
                //  Eg. OXXX_*_XXXO, OXX_X*_XXXO, OXX_X*X_XXO, ...
                //
                // This is a dirty fix for renju rule: two four in one line is forbidden point.
                // We just set it to the overline pattern, which is also considered as a type
                // of forbidden point. Maybe the best solution is to create a new pattern type
                // for this kind of line, however that would cost some overhead for freestyle
                // and standard rule.
                if (f5Idx[1] - f5Idx[0] < 5)
                    p = OL;
            }
        }
        else if (patCnt[F5])
            p = B4;
        else if (patCnt[F4] >= 2)
            p = F3S;
        else if (patCnt[F4])
            p = F3;
        else if (patCnt[B4])
            p = B3;
        else if (patCnt[F3S] + patCnt[F3] >= 4)
            p = F2B;
        else if (patCnt[F3S] + patCnt[F3] >= 3)
            p = F2A;
        else if (patCnt[F3S] + patCnt[F3])
            p = F2;
        else if (patCnt[B3])
            p = B2;
        else if (patCnt[F2] + patCnt[F2A] + patCnt[F2B])
            p = F1;
        else if (patCnt[B2])
            p = B1;
    }

    return patMemorized = p;
}

template <Rule R>
void fillPattern2xLUT(Pattern2x pattern2x[KeyCnt<R>])
{
    constexpr auto Mid = HalfLineLen<R>;
    PatternMemo<R> memoBlack, memoWhite;

    for (uint32_t key = 0; key < KeyCnt<R>; key++) {
        Line<R> lineBlack(key, BLACK), lineWhite(key, WHITE);

        pattern2x[key].patBlack = getPattern<R, BLACK>(memoBlack, lineBlack);
        pattern2x[key].patWhite = getPattern<R, WHITE>(memoWhite, lineWhite);
    }
}

template <bool Forbid>
Pattern4 getPattern4(Pattern p1, Pattern p2, Pattern p3, Pattern p4)
{
    int n[PATTERN_NB] = {0};
    n[p1]++;
    n[p2]++;
    n[p3]++;
    n[p4]++;

    if (n[F5] >= 1)
        return A_FIVE;  // OOOO_

    // Forbid check for Renju rule
    if constexpr (Forbid) {
        if (n[OL] >= 1)
            return FORBID;  // XOOO_OOX, O_O_O_O
        if (n[F4] + n[B4] >= 2)
            return FORBID;  // XOOO_ * _OOOX
        if (n[F3] + n[F3S] >= 2)
            return FORBID;  // _OO * _OO
    }

    if (n[B4] >= 2)
        return B_FLEX4;  // XOOO_ * _OOOX
    if (n[F4] >= 1)
        return B_FLEX4;  // OOO_, OO_O
    if (n[B4] >= 1) {
        if (n[F3] >= 1 || n[F3S] >= 1)
            return C_BLOCK4_FLEX3;  // XOOO_ * _OO
        if (n[B3] >= 1)
            return D_BLOCK4_PLUS;  // XOOO_ * _OOX
        if (n[F2] + n[F2A] + n[F2B] >= 1)
            return D_BLOCK4_PLUS;  // XOOO_ * _O
        else
            return E_BLOCK4;  // XOOO_
    }
    if (n[F3] >= 1 || n[F3S] >= 1) {
        if (n[F3] + n[F3S] >= 2)
            return F_FLEX3_2X;  // OO_ * _OO
        if (n[B3] >= 1)
            return G_FLEX3_PLUS;  // OO_ * _OOX
        if (n[F2] + n[F2A] + n[F2B] >= 1)
            return G_FLEX3_PLUS;  // OO_ * _O
        else
            return H_FLEX3;  // OO_
    }
    if (n[B3] >= 1) {
        if (n[B3] >= 2)
            return I_BLOCK3_PLUS;  // XOO_ * XOO_
        if (n[F2] + n[F2A] + n[F2B] >= 1)
            return I_BLOCK3_PLUS;  // XOO_ * O_
    }
    if (n[F2] + n[F2A] + n[F2B] >= 2)
        return J_FLEX2_2X;  // O_ * O_
    if (n[B3] >= 1)
        return K_BLOCK3;  // XOO_
    if (n[F2] + n[F2A] + n[F2B] >= 1)
        return L_FLEX2;  // O_

    return NONE;
}

template <bool Forbid>
void fillPattern4LUT(Pattern4Score p4Score[PCODE_NB])
{
    for (int i = 0; i < PATTERN_NB; i++)
        for (int j = 0; j < PATTERN_NB; j++)
            for (int m = 0; m < PATTERN_NB; m++)
                for (int n = 0; n < PATTERN_NB; n++) {
                    PatternCode pcode = PatternConfig::PCODE[i][j][m][n];
                    p4Score[pcode] =
                        getPattern4<Forbid>((Pattern)i, (Pattern)j, (Pattern)m, (Pattern)n);
                }
}

void fillPatternCodeLUT(PatternCode pcode[PATTERN_NB][PATTERN_NB][PATTERN_NB][PATTERN_NB])
{
    constexpr int N = PATTERN_NB;

    // Sort four patterns ascending with a 5-comparator network. Two orderings collide iff they
    // are the same multiset, so the sorted tuple is each combination's canonical representative.
    auto sort4 = [](int &a, int &b, int &c, int &d) {
        if (a > b)
            std::swap(a, b);
        if (c > d)
            std::swap(c, d);
        if (a > c)
            std::swap(a, c);
        if (b > d)
            std::swap(b, d);
        if (b > c)
            std::swap(b, c);
    };

    // Pass 1: hand each non-decreasing 4-tuple a dense code in lexicographic order. This is exactly
    // the order in which canonical representatives first appear when scanning all N^4 tuples by
    // linear index, so the resulting mapping is identical to the historical O(N^8) construction.
    PatternCode code = 0;
    for (int a = 0; a < N; a++)
        for (int b = a; b < N; b++)
            for (int c = b; c < N; c++)
                for (int d = c; d < N; d++)
                    pcode[a][b][c][d] = code++;

    // Pass 2: every ordering of four patterns inherits its sorted representative's code. Safe
    // in place: only sorted slots (finalized in pass 1) are read; a sorted slot just re-stores
    // its own code.
    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int m = 0; m < N; m++)
                for (int n = 0; n < N; n++) {
                    int a = i, b = j, c = m, d = n;
                    sort4(a, b, c, d);
                    pcode[i][j][m][n] = pcode[a][b][c][d];
                }
}

template <Rule R>
void fillDefenceLUT(uint8_t defence[KeyCnt<R>][2])
{
    const auto lookupPattern2x = [](uint32_t key) -> Pattern2x {
        if constexpr (R == Rule::FREESTYLE)
            return PatternConfig::PATTERN2x[key];
        else if constexpr (R == Rule::STANDARD)
            return PatternConfig::PATTERN2xStandard[key];
        else if constexpr (R == Rule::RENJU)
            return PatternConfig::PATTERN2xRenju[key];
    };

    for (uint32_t key = 0; key < KeyCnt<R>; key++) {
        for (Color attacker : {BLACK, WHITE}) {
            uint32_t defenceMask = 0;

            // Check if White need to make any defence (Black has attack moves above three)
            if (Pattern attackPattern = lookupPattern2x(key)[attacker]; attackPattern >= F3) {
                // For each empty cell, we place a block move and see if attacker can not
                // impose any threat now. If so, we regard this move as a defence move.
                for (int i = 0; i < 2 * HalfLineLen<R>; i++) {
                    uint32_t moveMask = 0b11 << 2 * i;
                    if ((key & moveMask) == moveMask
                        && lookupPattern2x(key & ~moveMask)[attacker] < F3)
                        defenceMask |= 1 << i;
                }

                // Make defence mask centered, as the outer one move is not defence move
                // For Standard/Renju rule: |OOOOOXOOOOO| -> O|OOOOXOOOO|O
                defenceMask = (defenceMask >> (HalfLineLen<R> - 4)) & 0xff;
            }

            defence[key][attacker] = defenceMask;
        }
    }
}

}  // namespace
