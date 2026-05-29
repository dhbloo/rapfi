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

#include "board.h"

/// RAII guards for speculative board mutation. The engine's static threat checks temporarily
/// mutate an otherwise-const Board (play a move, flip the side, rewind a ply), inspect the result,
/// then restore it. These guards encapsulate the const_cast and guarantee the board is restored on
/// every exit path (return, break/goto out of scope, exception), so callers cannot forget to undo.

/// Plays `pos` on construction and undoes it on destruction. The rule is a compile-time parameter,
/// so the underlying move/undo are direct templated calls (no dynamic dispatch) — preferred on the
/// hot search-side probes.
template <Rule R, Board::MoveType MT = Board::MoveType::NO_EVAL>
class ScopedMove
{
public:
    ScopedMove(const Board &board, Pos pos) : board(const_cast<Board &>(board))
    {
        this->board.move<R, MT>(pos);
    }
    ~ScopedMove() { board.undo<R, MT>(); }

    ScopedMove(const ScopedMove &)            = delete;
    ScopedMove &operator=(const ScopedMove &) = delete;

private:
    Board &board;
};

/// Switches the side to move to `side` on construction and restores it on destruction (a no-op
/// when already on that side). For probes that must act as a specific color.
class ScopedSwitchSide
{
public:
    ScopedSwitchSide(const Board &board, Color side)
        : board(const_cast<Board &>(board))
        , flipped(board.sideToMove() != side)
    {
        if (flipped)
            this->board.flipSide();
    }
    ~ScopedSwitchSide()
    {
        if (flipped)
            board.flipSide();
    }

    ScopedSwitchSide(const ScopedSwitchSide &)            = delete;
    ScopedSwitchSide &operator=(const ScopedSwitchSide &) = delete;

private:
    Board &board;
    bool   flipped;
};

/// Like ScopedMove, but with a runtime `rule` (dynamic dispatch). For contexts that only know the
/// rule at runtime, such as the database tooling; not for the templated search hot path.
class ScopedRuleMove
{
public:
    ScopedRuleMove(const Board &board, Rule rule, Pos pos)
        : board(const_cast<Board &>(board))
        , rule(rule)
    {
        this->board.move(rule, pos);
    }
    ~ScopedRuleMove() { board.undo(rule); }

    ScopedRuleMove(const ScopedRuleMove &)            = delete;
    ScopedRuleMove &operator=(const ScopedRuleMove &) = delete;

private:
    Board &board;
    Rule   rule;
};

/// Rewinds the last move on construction and replays it on destruction, for probes that inspect
/// the parent position. Runtime `rule`.
class ScopedRuleUndo
{
public:
    ScopedRuleUndo(const Board &board, Rule rule)
        : board(const_cast<Board &>(board))
        , rule(rule)
        , lastMove_(board.getLastMove())
    {
        this->board.undo(rule);
    }
    ~ScopedRuleUndo() { board.move(rule, lastMove_); }

    ScopedRuleUndo(const ScopedRuleUndo &)            = delete;
    ScopedRuleUndo &operator=(const ScopedRuleUndo &) = delete;

    /// The rewound move (the one that will be replayed on destruction).
    Pos lastMove() const { return lastMove_; }

private:
    Board &board;
    Rule   rule;
    Pos    lastMove_;
};
