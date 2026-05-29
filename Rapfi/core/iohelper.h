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

#include "pos.h"
#include "types.h"

#include <iostream>
#include <string>
#include <utility>
#include <vector>

// -------------------------------------------------
// Synchronised output

/// RAII wrapper that locks a process-wide mutex for the duration of one output expression,
/// ensuring multi-threaded `<<` chains do not interleave on stdout/stderr. Use the
/// `sync_cout()` / `sync_cerr()` factories rather than constructing this directly.
struct SyncOutputStream
{
    SyncOutputStream(std::ostream &os);
    ~SyncOutputStream();

    SyncOutputStream(const SyncOutputStream &)            = delete;
    SyncOutputStream &operator=(const SyncOutputStream &) = delete;
    SyncOutputStream(SyncOutputStream &&)                 = delete;
    SyncOutputStream &operator=(SyncOutputStream &&)      = delete;

    template <typename T>
    SyncOutputStream &operator<<(T &&value)
    {
        os << std::forward<T>(value);
        return *this;
    }
    SyncOutputStream &operator<<(std::ostream &(*m)(std::ostream &))
    {
        os << m;
        return *this;
    }

    std::ostream &os;
};

[[nodiscard]] inline SyncOutputStream sync_cout()
{
    return SyncOutputStream {std::cout};
}

/// Emit a piskvork `MESSAGE <text>` line, atomic with respect to other threads.
#define MESSAGEL(message) sync_cout() << "MESSAGE " << message << std::endl

/// Emit a piskvork `ERROR <text>` line, atomic with respect to other threads.
#define ERRORL(message) sync_cout() << "ERROR " << message << std::endl

/// In NDEBUG builds DEBUGL is compiled to a no-op; otherwise it emits `DEBUG <text>`.
#ifdef NDEBUG
    #define DEBUGL(message) ((void)0)
#else
    #define DEBUGL(message) sync_cout() << "DEBUG " << message << std::endl
#endif

// -------------------------------------------------
// Coordinate conversion
//
// The piskvork protocol speaks in (x, y) integer pairs that may need to be remapped before
// being turned into a Pos, so the engine can match different GUIs' coordinate orientations.
// These helpers are pure: the active CoordConvertionMode (held by Config) is passed in.

/// Convert protocol (x, y) to an internal Pos under `mode`. `(x, y) == (-1, -1)` is PASS.
Pos inputCoordConvert(int x, int y, int boardsize, CoordConvertionMode mode);

/// Inverse of `inputCoordConvert`: the protocol `(x, y)` pair for the given Pos under `mode`.
/// Returns `(-1, -1)` for a PASS move.
std::pair<int, int> outputCoordConvert(Pos pos, int boardsize, CoordConvertionMode mode);

// -------------------------------------------------
// Stream formatters

/// Manipulator for streaming a single move as a protocol `"x,y"` pair under a coord mode.
struct CoordText
{
    Pos                 pos;
    int                 boardsize;
    CoordConvertionMode mode = CoordConvertionMode::NONE;
};

/// Manipulator for streaming a list of moves with configurable spacing and coordinate style.
struct MovesText
{
    const std::vector<Pos> &moves;

    bool                withSpace = true;   ///< Insert a space between moves.
    bool                rawCoords = false;  ///< Emit `"x,y"` pairs instead of `"H8"` labels.
    int                 boardsize = 15;     ///< Board size used by raw-coord conversion.
    CoordConvertionMode mode = CoordConvertionMode::NONE;  ///< Mode used by raw-coord conversion.
};

std::ostream &operator<<(std::ostream &out, Pos pos);
std::ostream &operator<<(std::ostream &out, Color color);
std::ostream &operator<<(std::ostream &out, Pattern4 p4);
std::ostream &operator<<(std::ostream &out, Value value);
std::ostream &operator<<(std::ostream &out, Rule rule);
std::ostream &operator<<(std::ostream &out, CoordText coord);
std::ostream &operator<<(std::ostream &out, MovesText movesRef);
