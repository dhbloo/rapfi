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

#include "movegen.h"

#include "board.h"
#include "pattern.h"

#include <algorithm>

namespace {

/// Max distance to find a pos in a line.
constexpr int MaxFindDist = 4;

/// Filter move by Pattern4 according to GenType.
template <GenType Type>
inline bool basicPatternFilter(const Board &board, Pos pos, Color side)
{
    const Cell &c  = board.cell(pos);
    Pattern4    p4 = c.pattern4[side];

    if (bool(Type & WINNING)) {
        if (p4 >= B_FLEX4)
            return true;
    }

    if (bool(Type & VCF)) {
        if constexpr (bool(Type & COMB)) {
            if (p4 >= D_BLOCK4_PLUS)
                return true;
        }
        else if constexpr (bool(Type & RULE_RENJU)) {
            if (p4 >= E_BLOCK4
                || p4 == FORBID
                       && (c.pattern(side, 0) >= B4 || c.pattern(side, 1) >= B4
                           || c.pattern(side, 2) >= B4 || c.pattern(side, 3) >= B4))
                return true;
        }
        else {
            if (p4 >= E_BLOCK4)
                return true;
        }
    }

    if ((Type & VCT) && p4 < E_BLOCK4) {
        if constexpr (bool(Type & COMB)) {
            if (p4 >= G_FLEX3_PLUS)
                return true;
        }
        else {
            if (p4 >= H_FLEX3)
                return true;
        }
    }

    if ((Type & VC2) && p4 < H_FLEX3) {
        if constexpr (bool(Type & COMB)) {
            if (p4 >= J_FLEX2_2X)
                return true;
        }
        else {
            if (p4 >= L_FLEX2)
                return true;
        }
    }

    return bool(Type & TRIVIAL);
}

/// A fast check function to skip unnecessary move generation process.
/// @return Whether complete move generation should continue.
template <GenType Type>
constexpr bool preCheckFilter(const Board &board, Color side)
{
    if (bool(Type & VCF)) {
        if constexpr (bool(Type & COMB)) {
            if (board.p4Count(side, D_BLOCK4_PLUS) + board.p4Count(side, C_BLOCK4_FLEX3) == 0)
                return false;
        }
        else {
            if (board.p4Count(side, E_BLOCK4) + board.p4Count(side, D_BLOCK4_PLUS)
                    + board.p4Count(side, C_BLOCK4_FLEX3)
                == 0)
                return false;
        }
    }

    return true;
}

/// Get the first found pos that has the given pattern4.
/// @return Pos::NONE if we cannot find the pattern4 pos.
/// @note Board state must satisfy `board.p4Count(side, p4) > 0`.
Pos findFirstPattern4Pos(const Board &board, Color side, Pattern4 p4)
{
    FOR_EVERY_CAND_POS(&board, pos)
    {
        if (board.cell(pos).pattern4[side] == p4)
            return pos;
    }

    assert(false && "can not find pattern4");
    return Pos::NONE;
}

/// Find all pseudo defence pos of FOUR pattern4.
/// @param side Color of side with a FOUR pattern4.
ScoredMove *findAllPseudoFourDefendPos(const Board &board, Color side, ScoredMove *moveList)
{
    FOR_EVERY_CAND_POS(&board, pos)
    {
        const Cell &c = board.cell(pos);

        if (c.pattern4[side] >= E_BLOCK4)
            *moveList++ = pos;
        else if (c.pattern4[side] == FORBID) {
            assert(side == BLACK);

            for (int dir = 0; dir < 4; dir++) {
                // Check if this pos is a B4 + (B4/F4), but recognized as FORBID in Renju.
                // If true, this is still a defend four pos for white.
                if (c.pattern(BLACK, dir) >= B4) {
                    *moveList++ = pos;
                    break;
                }
            }
        }
    }

    return moveList;
}

/// Find all exact defence pos of opponent FOUR pattern4.
template <bool IncludeLosingMoves>
ScoredMove *findFourDefence(const Board &board, ScoredMove *const moveList)
{
    Color       oppo = ~board.sideToMove();
    ScoredMove *last = moveList;

    assert(board.p4Count(oppo, A_FIVE) == 0);
    assert(board.p4Count(oppo, B_FLEX4) > 0);

    // Find all defend pos for F3 attack line pattern (_*OOO*_, X*OOO**X, _*O*OO*_, _O*O*O*O_)
    auto findF3LineDefence = [=, &board](Pos f4Pos, int dir, ScoredMove *const list) {
        const Cell &f4Cell = board.cell(f4Pos);
        assert(f4Cell.pattern(oppo, dir) == F4);

        list[0] = f4Pos;  // Add first defence

        Pos pos = f4Pos;
        for (int i = 0; i < MaxFindDist; i++) {
            pos -= DIRECTION[dir];

            if (const Cell &c = board.cell(pos); c.piece == oppo)
                continue;
            else if (c.piece == EMPTY) {
                list[1] = pos;  // Second defence
                if (c.pattern(oppo, dir) == F4
                    && (c.pattern4[oppo] != FORBID || !board.checkForbiddenPoint(pos)))
                    return list + 2;
            }
            break;
        }
        pos = f4Pos;
        for (int i = 0; i < MaxFindDist; i++) {
            pos += DIRECTION[dir];

            if (const Cell &c = board.cell(pos); c.piece == oppo)
                continue;
            else if (c.piece == EMPTY) {
                if (c.pattern(oppo, dir) == F4
                    && (c.pattern4[oppo] != FORBID || !board.checkForbiddenPoint(pos))) {
                    list[1] = pos;  // Second defence
                    return list + 2;
                }
                else
                    list[2] = pos;  // Third defence
            }
            break;
        }

        return list + 3;
    };

    // Find all defend pos for double B3 attack line pattern (XOOO**_ + XOOO**_)
    auto findB3Defence = [=, &board](Pos f4Pos, int dir, ScoredMove *list) {
        const Cell &f4Cell = board.cell(f4Pos);
        assert(f4Cell.pattern(oppo, dir) == B4);

        *list++ = f4Pos;

        for (int dir = 0; dir < 4; dir++) {
            int i, j;
            Pos pos = f4Pos;
            for (i = 0; i < MaxFindDist; i++) {
                pos -= DIRECTION[dir];

                if (const Cell &c = board.cell(pos); c.piece == oppo)
                    continue;
                else if (c.piece == EMPTY && c.pattern(oppo, dir) >= B4)
                    *list++ = pos;
                break;
            }
            pos = f4Pos;
            for (j = MaxFindDist - i; j > 0; j--) {
                pos += DIRECTION[dir];

                if (const Cell &c = board.cell(pos); c.piece == oppo)
                    continue;
                else if (c.piece == EMPTY && c.pattern(oppo, dir) >= B4)
                    *list++ = pos;
                break;
            }
        }

        return list;
    };

    // Try to find the last opponent attack move that caused flex4 pattern,
    // then its pos can be used to find the resulted flex4 moves.
    if (Pos lastFlex4AttackPos = board.stateInfo().lastFlex4AttackMove[oppo]) {
        const Cell &attackCell = board.cell(lastFlex4AttackPos);

        // If a pattern in any direction is F3(F3S), then last four is cause
        // by at least one F3. Find all F3 defend moves in every directions.
        for (int dir = 0; dir < 4; dir++) {
            // Cell's pattern is not updated after stone is placed there,
            // so we can look up the pattern before placement
            Pattern lastMovePattern = attackCell.pattern(oppo, dir);
            if (lastMovePattern != F3 && lastMovePattern != F3S)
                continue;

            Pos pos = lastFlex4AttackPos;
            for (int i = 0; i < MaxFindDist; i++) {
                pos -= DIRECTION[dir];

                if (const Cell &c = board.cell(pos); c.piece == oppo)
                    continue;
                else if (c.piece == EMPTY) {
                    if (c.pattern(oppo, dir) == F4 && c.pattern4[oppo] == B_FLEX4) {
                        // If there has already a F3 line, the second F3 line
                        // means double F3 pattern which can not be defended.
                        if (!IncludeLosingMoves && last > moveList)
                            return moveList;

                        last = findF3LineDefence(pos, dir, last);
                        // return last;
                        goto next_F3_dir;
                    }
                    continue;
                }
                break;
            }
            pos = lastFlex4AttackPos;
            for (int i = 0; i < MaxFindDist; i++) {
                pos += DIRECTION[dir];

                if (const Cell &c = board.cell(pos); c.piece == oppo)
                    continue;
                else if (c.piece == EMPTY) {
                    if (c.pattern(oppo, dir) == F4 && c.pattern4[oppo] == B_FLEX4) {
                        // If there has already a F3 line, the second F3 line
                        // means double F3 pattern which can not be defended.
                        if (!IncludeLosingMoves && last > moveList)
                            return moveList;

                        last = findF3LineDefence(pos, dir, last);
                        // return last;
                        goto next_F3_dir;
                    }
                    continue;
                }
                break;
            }

        next_F3_dir:
            continue;
        }

        if (last > moveList)
            return last;

        // If patterns in all directions are not F3, then the B_FLEX4 must
        // be formed by double B3 (in two direction or one direction).
        for (int dir = 0; dir < 4; dir++) {
            if (attackCell.pattern(oppo, dir) != B3)
                continue;

            int i, j, empty;
            Pos pos = lastFlex4AttackPos;
            for (i = 0, empty = 0; i < MaxFindDist; i++) {
                pos -= DIRECTION[dir];

                if (const Cell &c = board.cell(pos); c.piece == oppo)
                    continue;
                else if (c.piece == EMPTY) {
                    if (c.pattern4[oppo] >= B_FLEX4) {
                        Pattern pattern = c.pattern(oppo, dir);
                        if (pattern == F4)
                            return findF3LineDefence(pos, dir, last);
                        else if (pattern == B4)
                            return findB3Defence(pos, dir, last);
                    }
                    if (++empty >= 2)
                        break;
                    continue;
                }
                break;
            }
            pos = lastFlex4AttackPos;
            for (j = MaxFindDist - i; j > 0; j--) {
                pos += DIRECTION[dir];

                if (const Cell &c = board.cell(pos); c.piece == oppo)
                    continue;
                else if (c.piece == EMPTY) {
                    if (c.pattern4[oppo] >= B_FLEX4) {
                        Pattern pattern = c.pattern(oppo, dir);
                        if (pattern == F4)
                            return findF3LineDefence(pos, dir, last);
                        else if (pattern == B4)
                            return findB3Defence(pos, dir, last);
                    }
                    continue;
                }
                break;
            }
        }
    }

    // Some pattern which is impossible in normal game may appear in analysis mode
    return findAllPseudoFourDefendPos(board, oppo, last);
}

/// Find all exact defence pos of opponent B4F3 pattern4.
/// @note If no direct defence is needed, empty move list is returned.
template <Rule R>
ScoredMove *findB4F3Defence(const Board &board, ScoredMove *const moveList)
{
    Color oppo = ~board.sideToMove();

    assert(board.p4Count(oppo, A_FIVE) == 0 && board.p4Count(oppo, B_FLEX4) == 0);
    assert(board.p4Count(oppo, C_BLOCK4_FLEX3) > 0);

    // Find all valid defence move for a F3 line through a LUT
    auto findF3LineDefence = [=, &board](Pos f3Pos, int dir, ScoredMove *list) {
        assert(board.cell(f3Pos).pattern(oppo, dir) == F3
               || board.cell(f3Pos).pattern(oppo, dir) == F3S);

        uint64_t key         = board.getKeyAt<R>(f3Pos, dir);
        uint32_t defenceMask = PatternConfig::lookupDefenceTable<R>(key, oppo);

        const bool checkRenjuDefence   = R == RENJU && oppo == BLACK;
        Pos        pos                 = f3Pos;
        uint32_t   leftMask            = defenceMask >> 4;
        Pos        leftRenjuDefence    = Pos::NONE;
        Pos        rightRenjuDefence   = Pos::NONE;
        bool       prevFound           = false;
        bool       foundLeftForbidden  = false;
        bool       foundRightForbidden = false;

        // In order to check renju defence, we need to put a black move at f3Pos,
        // so that checkForbidden Point will work correctly.
        if (checkRenjuDefence) {
            Board &b = const_cast<Board &>(board);
            b.flipSide();
            b.move<Rule::RENJU, Board::MoveType::NO_EVAL>(f3Pos);
        }

        for (int i = 0; i < 4; i++) {
            pos -= DIRECTION[dir];

            if ((defenceMask >> (3 - i)) & 0x1) {
                assert(board.isEmpty(pos));
                *list++   = pos;
                prevFound = true;

                if (checkRenjuDefence)
                    foundLeftForbidden = foundLeftForbidden || board.checkForbiddenPoint(pos);
            }
            else if (checkRenjuDefence && prevFound && board.isEmpty(pos)) {
                leftRenjuDefence = pos;
                prevFound        = false;
            }
        }
        pos       = f3Pos;
        prevFound = false;
        for (int i = 0; i < 4; i++) {
            pos += DIRECTION[dir];
            if ((defenceMask >> (4 + i)) & 0x1) {
                assert(board.isEmpty(pos));
                *list++   = pos;
                prevFound = true;

                if (checkRenjuDefence)
                    foundRightForbidden = foundRightForbidden || board.checkForbiddenPoint(pos);
            }
            else if (checkRenjuDefence && prevFound && board.isEmpty(pos)) {
                rightRenjuDefence = pos;
                prevFound         = false;
            }
        }

        if (checkRenjuDefence) {
            // If we have found a forbidden move for opponent black, we might have some
            // other possible defences. Normally the defence move should be at the other
            // side of the forbidden point, so we add defence move only in this case.
            if (foundLeftForbidden && rightRenjuDefence)
                *list++ = rightRenjuDefence;
            if (foundRightForbidden && leftRenjuDefence)
                *list++ = leftRenjuDefence;

            // Remember to undo the f3Pos, make sure board is const
            Board &b = const_cast<Board &>(board);
            b.undo<Rule::RENJU, Board::MoveType::NO_EVAL>();
            b.flipSide();
        }

        return list;
    };

    auto findB4InLine = [=, &board](Pos b4Pos, int dir) {
        // F4 in Renju is judged as OL, but it should be a valid defence
        auto checkRenjuF4 = [dir, &board](const Cell &c, Pos pos) {
            auto      lineKey = board.getKeyAt<Rule::STANDARD>(pos, dir);
            Pattern2x pattern = PatternConfig::lookupPattern<Rule::STANDARD>(lineKey);
            return pattern.patBlack >= B4;
        };
        // Detect overline B4 (which is not a valid B4 point) in Standard/Renju
        auto checkNotOverlineB4 = [b4Pos, oppo, dir, &board](const Cell &c, Pos pos) {
            Board &b = const_cast<Board &>(board);
            b.flipSide();
            b.move<R, Board::MoveType::NO_EVAL>(pos);
            bool hasFive = board.cell(b4Pos).pattern(oppo, dir) == F5;
            b.undo<R, Board::MoveType::NO_EVAL>();
            b.flipSide();
            return hasFive;
        };

        int i, j;
        Pos pos = b4Pos;
        for (i = 0; i < MaxFindDist; i++) {
            pos -= DIRECTION[dir];

            if (const Cell &c = board.cell(pos); c.piece == oppo)
                continue;
            else if (c.piece == EMPTY
                     && (c.pattern(oppo, dir) == B4
                         || R == RENJU && c.pattern4[oppo] == FORBID && checkRenjuF4(c, pos))) {
                if (R == FREESTYLE || checkNotOverlineB4(c, pos))
                    return pos;
            }
            break;
        }
        pos = b4Pos;
        for (j = MaxFindDist - i; j > 0; j--) {
            pos += DIRECTION[dir];

            if (const Cell &c = board.cell(pos); c.piece == oppo)
                continue;
            else if (c.piece == EMPTY
                     && (c.pattern(oppo, dir) == B4
                         || R == RENJU && c.pattern4[oppo] == FORBID && checkRenjuF4(c, pos)))
                return pos;
            break;
        }

        assert(false && "did not find B4 pattern pos in C_BLOCK4_FLEX3");
        return Pos::NONE;
    };

    auto findAllB3CounterDefence = [=, &board](Pos b4Pos, int dir, ScoredMove *list) {
        Color      self = ~oppo;
        const bool isPseudoForbiddenB4 =
            R == RENJU && self == BLACK && board.cell(b4Pos).pattern4[self] == FORBID;

        Pos pos = b4Pos;
        for (int i = 0; i < MaxFindDist; i++) {
            pos -= DIRECTION[dir];

            if (const Cell &c = board.cell(pos); c.piece == self)
                continue;
            else if (c.piece == EMPTY && (isPseudoForbiddenB4 || c.pattern(self, dir) >= B3)) {
                *list++ = pos;
                continue;
            }
            break;
        }
        pos = b4Pos;
        for (int i = 0; i < MaxFindDist; i++) {
            pos += DIRECTION[dir];

            if (const Cell &c = board.cell(pos); c.piece == self)
                continue;
            else if (c.piece == EMPTY && (isPseudoForbiddenB4 || c.pattern(self, dir) >= B3)) {
                *list++ = pos;
                continue;
            }
            break;
        }

        return list;
    };

    // Get opponent B4F3 pos from last memorized C type move.
    Pos B4F3Pos = board.stateInfo().lastPattern4(oppo, C_BLOCK4_FLEX3);

    // Make sure we found the right B4F3 pos. If fast query failed in
    // some rare case, we find it by iterating all move candidates.
    if (const Cell &cell = board.cell(B4F3Pos);
        cell.piece != EMPTY || cell.pattern4[oppo] != C_BLOCK4_FLEX3) {
        B4F3Pos = findFirstPattern4Pos(board, oppo, C_BLOCK4_FLEX3);
    }

    const Cell &B4F3Cell = board.cell(B4F3Pos);
    assert(B4F3Cell.piece == EMPTY);
    assert(B4F3Cell.pattern4[oppo] == C_BLOCK4_FLEX3);

    ScoredMove *last = moveList;
    *last++          = B4F3Pos;

    // Iterate all directions to find F3 pattern line and B4 pattern line.
    for (int dir = 0; dir < 4; dir++) {
        Pattern pattern = B4F3Cell.pattern(oppo, dir);
        if (pattern == F3 || pattern == F3S)
            last = findF3LineDefence(B4F3Pos, dir, last);
        else if (pattern == B4) {
            Pos b4Pos = findB4InLine(B4F3Pos, dir);

            // If we have a B4 counter defence move, then direct defence
            // to the opponent move is unnecessary.
            if (board.cell(b4Pos).pattern4[~oppo] >= E_BLOCK4)
                return moveList;

            *last++ = b4Pos;

            for (int d = 0; d < 4; d++)
                last = findAllB3CounterDefence(b4Pos, d, last);
        }
    }

    return last;
}

/// Generates defence moves for opponent B_FLEX4 pattern4.
/// All VCF moves of us is excluded. VCF moves should be generated
/// by other generator.
/// @note Board state must satisfy `board.p4Count(oppo, B_FLEX4) > 0`.
template <bool IncludeLosingMoves>
ScoredMove *generateFourDefence(const Board &board, ScoredMove *moveList)
{
    assert(board.p4Count(~board.sideToMove(), B_FLEX4));
    ScoredMove *last = findFourDefence<IncludeLosingMoves>(board, moveList);

    std::sort(moveList, last, [](ScoredMove m, ScoredMove n) { return m.pos < n.pos; });
    last = std::unique(moveList, last, [](ScoredMove m, ScoredMove n) { return m.pos == n.pos; });

    return std::remove_if(moveList, last, [&](ScoredMove move) {
        assert(board.isEmpty(move));
        assert(board.cell(move).pattern4[~board.sideToMove()] >= E_BLOCK4
               || board.cell(move).pattern4[~board.sideToMove()] == FORBID);

        // only adds non-vcf moves
        return board.cell(move).pattern4[board.sideToMove()] >= E_BLOCK4;
    });
}

/// Generates defence moves for opponent C_BLOCK4_FLEX3 pattern4.
/// All VCF moves of us is excluded. VCF moves should be generated
/// by other generator. If direct defence is not needed since we
/// have some B4 counter defence move, empty move list is returned.
/// @note Board state must satisfy `board.p4Count(oppo, C_BLOCK4_FLEX3) > 0`.
template <Rule R>
ScoredMove *generateB4F3Defence(const Board &board, ScoredMove *moveList)
{
    assert(board.p4Count(~board.sideToMove(), C_BLOCK4_FLEX3));
    ScoredMove *last = findB4F3Defence<R>(board, moveList);

    // If direct defence is not needed, we simply return empty move list.
    if (last == moveList)
        return moveList;

    std::sort(moveList, last, [](ScoredMove m, ScoredMove n) { return m.pos < n.pos; });
    last = std::unique(moveList, last, [](ScoredMove m, ScoredMove n) { return m.pos == n.pos; });

    return std::remove_if(moveList, last, [&](ScoredMove move) {
        assert(board.isEmpty(move));
        return board.cell(move).pattern4[board.sideToMove()] >= E_BLOCK4;
    });
}

}  // namespace

template <GenType Type>
ScoredMove *generate(const Board &board, ScoredMove *moveList)
{
    Color self = board.sideToMove();

    FOR_EVERY_CAND_POS(&board, pos)
    {
        if (basicPatternFilter<Type>(board, pos, self)) {
            *moveList++ = pos;
        }
    }

    return moveList;
}

template ScoredMove *generate<VCF>(const Board &, ScoredMove *);
template ScoredMove *generate<VCF | RULE_RENJU>(const Board &, ScoredMove *);
template ScoredMove *generate<ALL>(const Board &, ScoredMove *);

template <GenType Type>
ScoredMove *generateNeighbors(const Board     &board,
                              ScoredMove      *moveList,
                              Pos              center,
                              const Direction *neighbors,
                              size_t           numNeighbors)
{
    Color self = board.sideToMove();

    if (!preCheckFilter<Type>(board, self))
        return moveList;

    for (size_t i = 0; i < numNeighbors; i++) {
        Pos         pos = center + neighbors[i];
        const Cell &c   = board.cell(pos);

        if (c.piece == EMPTY && c.isCandidate() && basicPatternFilter<Type>(board, pos, self))
            *moveList++ = pos;
    }

    return moveList;
}

template ScoredMove *
generateNeighbors<VCF>(const Board &, ScoredMove *, Pos, const Direction *, size_t);
template ScoredMove *
generateNeighbors<VCF | COMB>(const Board &, ScoredMove *, Pos, const Direction *, size_t);

/// Generate direct winning moves for current side to move.
/// @return The first found winning pos.
/// @note Board state must satisfy `p4Count(self, A_FIVE) + p4Count(self, B_FLEX4) > 0`.
template <>
ScoredMove *generate<WINNING>(const Board &board, ScoredMove *moveList)
{
    Color self = board.sideToMove();

    if (board.p4Count(self, A_FIVE)) {
        *moveList++ = findFirstPattern4Pos(board, self, A_FIVE);
    }
    else if (board.p4Count(self, B_FLEX4)) {
        *moveList++ = findFirstPattern4Pos(board, self, B_FLEX4);
    }
    else {
        assert(false && "no winning moves found");
    }

    return moveList;
}

/// Generate the defence move for opponent A_FIVE pattern4.
/// @return The first found FIVE pattern4 pos.
/// @note Board state must satisfy `board.p4Count(oppo, A_FIVE) > 0`.
template <>
ScoredMove *generate<DEFEND_FIVE>(const Board &board, ScoredMove *moveList)
{
    Color oppo = ~board.sideToMove();
    assert(board.p4Count(oppo, A_FIVE) > 0);

    // Get last opponent A_FIVE directly from state info.
    *moveList     = board.stateInfo().lastPattern4(oppo, A_FIVE);
    const Cell &c = board.cell(*moveList);
    if (LIKELY(c.piece == EMPTY && c.pattern4[oppo] == A_FIVE))
        return moveList + 1;

    // In case of weird history, we find the A_FIVE pos by iterating all move candidates.
    FOR_EVERY_CAND_POS(&board, pos)
    {
        if (board.cell(pos).pattern4[oppo] == A_FIVE) {
            *moveList = pos;
            return moveList + 1;
        }
    }
    return moveList;
}

/// Generates defence moves for opponent B_FLEX4 pattern4.
/// All VCF moves of us is excluded. VCF moves should be generated
/// by other generator.
/// @note Board state must satisfy `board.p4Count(oppo, B_FLEX4) > 0`.
template <>
ScoredMove *generate<DEFEND_FOUR>(const Board &board, ScoredMove *moveList)
{
    return generateFourDefence<false>(board, moveList);
}

/// Generates defence moves for opponent B_FLEX4 pattern4.
/// This version also generates all losing moves.
template <>
ScoredMove *generate<DEFEND_FOUR | ALL>(const Board &board, ScoredMove *moveList)
{
    return generateFourDefence<true>(board, moveList);
}

template <>
ScoredMove *generate<DEFEND_B4F3 | RULE_FREESTYLE>(const Board &board, ScoredMove *moveList)
{
    return generateB4F3Defence<FREESTYLE>(board, moveList);
}

template <>
ScoredMove *generate<DEFEND_B4F3 | RULE_STANDARD>(const Board &board, ScoredMove *moveList)
{
    return generateB4F3Defence<STANDARD>(board, moveList);
}

template <>
ScoredMove *generate<DEFEND_B4F3 | RULE_RENJU>(const Board &board, ScoredMove *moveList)
{
    return generateB4F3Defence<RENJU>(board, moveList);
}

bool validateOpponentCMove(const Board &board)
{
    // If threat is caused by White, it must be real threat
    if (board.sideToMove() == BLACK)
        return true;

    // We check black C_BLOCK4_FLEX3 move by making the move,
    // then check if there is any B_FLEX4 on board.
    Board &b = const_cast<Board &>(board);
    assert(b.p4Count(BLACK, C_BLOCK4_FLEX3) > 0);
    assert(b.p4Count(BLACK, B_FLEX4) == 0);

    Pos lastB4F3Pos = board.stateInfo().lastPattern4(BLACK, C_BLOCK4_FLEX3);
    // Make sure we found the right B4F3 pos. If fast query failed in
    // some rare case, we find it by iterating all move candidates.
    if (const Cell &cell = board.cell(lastB4F3Pos);
        cell.piece != EMPTY || cell.pattern4[BLACK] != C_BLOCK4_FLEX3) {
        lastB4F3Pos = findFirstPattern4Pos(board, BLACK, C_BLOCK4_FLEX3);
    }

    b.flipSide();
    b.move<Rule::RENJU, Board::MoveType::NO_EVAL>(lastB4F3Pos);

    bool hasBMove = b.p4Count(BLACK, B_FLEX4);

    b.undo<Rule::RENJU, Board::MoveType::NO_EVAL>();
    b.flipSide();

    return hasBMove;
}
