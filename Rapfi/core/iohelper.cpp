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

#include "iohelper.h"

#include <cassert>
#include <mutex>

// -------------------------------------------------
// Synchronised output

namespace {

/// Process-wide mutex serialising every SyncOutputStream lifetime. Intentionally heap-
/// allocated and never deleted: other translation units may emit messages from their own
/// static destructors, so the mutex must outlive every other static object.
std::mutex &syncIOMutex()
{
    static std::mutex *m = new std::mutex;
    return *m;
}

}  // namespace

SyncOutputStream::SyncOutputStream(std::ostream &os) : os(os)
{
    syncIOMutex().lock();
}

SyncOutputStream::~SyncOutputStream()
{
    syncIOMutex().unlock();
}

// -------------------------------------------------
// Coordinate conversion

namespace {

constexpr int PASS_COORD_X = -1;
constexpr int PASS_COORD_Y = -1;

}  // namespace

Pos inputCoordConvert(int x, int y, int boardsize, CoordConvertionMode mode)
{
    if (x == PASS_COORD_X && y == PASS_COORD_Y)
        return Pos::PASS;

    switch (mode) {
    case CoordConvertionMode::FLIPY_X: return {y, boardsize - 1 - x};
    case CoordConvertionMode::X_FLIPY: return {x, boardsize - 1 - y};
    default: return {x, y};
    }
}

std::pair<int, int> outputCoordConvert(Pos pos, int boardsize, CoordConvertionMode mode)
{
    if (pos == Pos::PASS)
        return {PASS_COORD_X, PASS_COORD_Y};

    switch (mode) {
    case CoordConvertionMode::FLIPY_X: return {boardsize - 1 - pos.y(), pos.x()};
    case CoordConvertionMode::X_FLIPY: return {pos.x(), boardsize - 1 - pos.y()};
    default: return {pos.x(), pos.y()};
    }
}

// -------------------------------------------------
// Stream formatters

std::ostream &operator<<(std::ostream &out, Pos pos)
{
    assert(pos.valid());

    if (pos == Pos::NONE)
        return out << "None";
    else if (pos == Pos::PASS)
        return out << "Pass";
    else
        return out << char(pos.x() + 'A') << (pos.y() + 1);
}

std::ostream &operator<<(std::ostream &out, Color color)
{
    assert(color >= BLACK && color <= EMPTY);

    static const char *const Names[] = {"Black", "White", "Wall", "Empty"};
    return out << Names[color];
}

std::ostream &operator<<(std::ostream &out, Pattern4 p4)
{
    assert(p4 >= NONE && p4 < PATTERN4_NB);

    // L_FLEX2..A_FIVE -> 'L'..'A' (so stronger threats print as earlier letters).
    if (p4 >= L_FLEX2 && p4 <= A_FIVE)
        return out << char('A' + A_FIVE - p4);
    else if (p4 == FORBID)
        return out << 'X';
    else
        return out << '.';
}

std::ostream &operator<<(std::ostream &out, Value value)
{
    assert(value >= VALUE_NONE && value <= VALUE_INFINITE);

    if (value == VALUE_NONE)
        return out << "VAL_NONE";
    if (value >= VALUE_MATE_IN_MAX_PLY) {
        if (value == VALUE_INFINITE)
            return out << "VAL_INF";
        else if (value == VALUE_MATE_FROM_DATABASE)
            return out << "+M*";
        else
            return out << "+M" << int(VALUE_MATE - value);
    }
    if (value <= VALUE_MATED_IN_MAX_PLY) {
        if (value == -VALUE_INFINITE)
            return out << "-VAL_INF";
        else if (value == VALUE_MATED_FROM_DATABASE)
            return out << "-M*";
        else
            return out << "-M" << int(VALUE_MATE + value);
    }
    else
        return out << int(value);
}

std::ostream &operator<<(std::ostream &out, Rule rule)
{
    assert(rule >= FREESTYLE && rule < RULE_NB);

    static const char *const Names[] = {"Freestyle", "Standard", "Renju"};
    return out << Names[rule];
}

std::ostream &operator<<(std::ostream &out, CoordText coord)
{
    auto [x, y] = outputCoordConvert(coord.pos, coord.boardsize, coord.mode);
    return out << x << ',' << y;
}

std::ostream &operator<<(std::ostream &out, MovesText movesRef)
{
    for (size_t i = 0; i < movesRef.moves.size(); i++) {
        if (movesRef.withSpace && i)
            out << ' ';
        if (movesRef.rawCoords)
            out << CoordText {movesRef.moves[i], movesRef.boardsize, movesRef.mode};
        else
            out << movesRef.moves[i];
    }
    return out;
}
