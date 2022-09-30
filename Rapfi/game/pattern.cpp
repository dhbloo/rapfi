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
#include <ostream>
#include <tuple>

namespace {

using PatternConfig::HalfLineLen;
using PatternConfig::KeyCnt;

/// ColorFlag represents one of the three states in a cell:
/// 1.self stone, 2.opponent stone, 3.empty cell.
enum ColorFlag { SELF, OPPO, EMPT };

/// The length of a line to determine the pattern (OOOO_OOOO).
template <Rule R>
constexpr int LineLen = HalfLineLen<R> * 2 + 1;

/// Line struct is a array representing a stright line of color flags.
template <Rule R>
struct Line : std::array<ColorFlag, LineLen<R>>
{
    explicit Line() = default;

    /// Construct a line from 64bit key and self color.
    explicit Line(uint64_t key, Color self);
};

/// Memorandum for (Line -> Pattern) table used in dynamic programming.
template <Rule R>
struct PatternMemo : std::array<Pattern, power(3, LineLen<R>)>
{
    // Special flag for uninitialized line pat
    static constexpr Pattern NilPat = Pattern(0xff);

    PatternMemo() { this->fill(NilPat); }
    Pattern &get(const Line<R> &line);
};

/// Count a line's SELF length, start/end position.
/// @return (RealLen, FullLen, Start, End)
template <Rule R>
constexpr std::tuple<int, int, int, int> countLine(const Line<R> &line);

/// Shift a line to where i is the middle stone (i' = 4).
template <Rule R>
constexpr Line<R> shiftLine(const Line<R> &line, int i);

/// Calculate line pattern (using dynamic programming).
template <Rule R, Color Side>
constexpr Pattern getPattern(PatternMemo<R> &patMemo, const Line<R> &line);

/// Initialize a (64bit key -> pattern2x) LUT.
template <Rule R>
void fillPattern2xLUT(Pattern2x pattern2x[KeyCnt<R>]);

/// @brief Get pattern4 from four line pattern.
/// @tparam Forbid Whether to mark forbidden point (for renju).
template <bool Forbid>
Pattern4 getPattern4(Pattern p1, Pattern p2, Pattern p3, Pattern p4);

/// Initialize a (pattern code -> pattern4) LUT.
template <bool Forbid>
void fillPattern4LUT(Pattern4Score p4Score[PCODE_NB]);

/// Initialize a (pattern x 4 -> pattern code) LUT.
void fillPatternCodeLUT(PatternCode pcode[PATTERN_NB][PATTERN_NB][PATTERN_NB][PATTERN_NB]);

/// Initialize a (64bit key -> defence moves mask) LUT.
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

// this will force compiler run initialization code before main()
const auto init = []() {
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
    // Encode a line into a unsigned int, used for indexing line pattern memo
    // 0 <= code < pow(3,9) = 19683
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
    // In order to calcuate a line's pattern, we can use a dynamic programming solution: continue
    // to put stones at EMPT position, and recursively see what's the next pattern, until it
    // reaches trivial state (i.e. OL[realLen >= 6] or F5[realLen >= 5] or DEAD[fullLen < 5]).
    // Then by tracing back which max pattern we got we can infer current line's pattern.
    //
    // Here's the description in chinese:
    // 动态规划求解棋型
    // 1. 当 realLen >= 6, 棋型为长连
    // 2. 当 realLen == 5, 棋型为连五
    // 3. 当 fullLen <  5, 棋型为死型
    // 4. 其余情况递归判断棋型:
    //    a. 下一步有两个以上连五点的是活四
    //    b. 下一步有一个连五点的是冲四
    //    c. 下一步有两个以上活四点的是连活三
    //    d. 下一步有一个活四点的是普通活三
    //    e. 下一步有冲四点的是眠三
    //    f. 下一步有四个活三点的是连活二
    //    g. 下一步有三个活三点的是跳活二
    //    h. 下一步有两个活三点的是大跳活二
    //    i. 下一步有眠三点的是眠二
    //    j. 下一步有活二点的是活一
    //    k. 下一步有眠二点的是眠一
    //    l. 其余情况是长连，棋型为死型

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
    constexpr int N     = PATTERN_NB;
    constexpr int N2    = N * N;
    constexpr int N3    = N2 * N;
    constexpr int N4    = N3 * N;
    int           v[N4] = {-1};

    for (int x = 0, i = 0; x < N; x++)
        for (int y = 0; y < N; y++)
            for (int z = 0; z < N; z++)
                for (int w = 0; w < N; w++) {
                    int a = x, b = y, c = z, d = w;
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

                    v[i++] = a * N3 + b * N2 + c * N + d;
                }

    for (int i = 0; i < N4; i++)
        if (v[i] > -1)
            for (int j = i + 1; j < N4; j++)
                if (v[i] == v[j])
                    v[j] = -1;

    for (int i = 0, count = 0; i < N4; i++)
        if (v[i] > -1)
            v[i] = count++;

    for (int i = 0; i < N; i++)
        for (int j = 0; j < N; j++)
            for (int m = 0; m < N; m++)
                for (int n = 0; n < N; n++) {
                    int a = i, b = j, c = m, d = n;
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

                    pcode[i][j][m][n] = v[a * N3 + b * N2 + c * N + d];
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

template <Rule R>
std::ostream &operator<<(std::ostream &os, const Line<R> &line)
{
    for (int i = line.size() - 1; i >= 0; i--) {
        if (i == line.size() / 2)
            os << ".";
        else {
            switch (line[i]) {
            case SELF: os << 'O'; break;
            case OPPO: os << 'X'; break;
            default: os << '_'; break;
            }
        }
    }
    return os;
}

}  // namespace
