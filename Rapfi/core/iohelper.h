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
#include "time.h"
#include "types.h"

#include <iostream>
#include <string>
#include <vector>

// -------------------------------------------------
// Output Macros / sync utility

/// Thread-safe output stream that locks the mutex during its lifetime.
struct SyncOutputStream
{
    SyncOutputStream(std::ostream &os);
    ~SyncOutputStream();
    // Non-copyable, non-movable
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

/// Create a thread-safe output stream for std::cout.
[[nodiscard]] inline SyncOutputStream sync_cout()
{
    return SyncOutputStream {std::cout};
}

/// Create a thread-safe output stream for std::cerr.
[[nodiscard]] inline SyncOutputStream sync_cerr()
{
    return SyncOutputStream {std::cerr};
}

// Output macros for logging messages, errors, and debug information.

#define MESSAGEL(message) sync_cout() << "MESSAGE " << message << std::endl
#define ERRORL(message)   sync_cout() << "ERROR " << message << std::endl
#ifdef NDEBUG
    #define DEBUGL(message) ((void)0)
#else
    #define DEBUGL(message) sync_cout() << "DEBUG " << message << std::endl
#endif

// -------------------------------------------------
// Coordinate conversion

/// Convert input coordinates (x, y) to a Pos object based on the board size.
Pos inputCoordConvert(int x, int y, int boardsize);
/// Convert output coordinates from a Pos object to x based on the board size.
int outputCoordXConvert(Pos pos, int boardsize);
/// Convert output coordinates from a Pos object to y based on the board size.
int outputCoordYConvert(Pos pos, int boardsize);
/// Convert a coordinate string (e.g., "A1", "h8") to a Pos object.
Pos parseCoord(std::string coordStr);

// -------------------------------------------------
// Formatters

/// A struct to format a list of moves for output with custom options.
struct MovesText
{
    const std::vector<Pos> &moves;

    bool withSpace = true;
    bool rawCoords = false;
    int  boardsize = 15;
};

std::ostream &operator<<(std::ostream &out, Pos pos);
std::ostream &operator<<(std::ostream &out, Color color);
std::ostream &operator<<(std::ostream &out, Pattern p);
std::ostream &operator<<(std::ostream &out, Pattern4 p4);
std::ostream &operator<<(std::ostream &out, Value value);
std::ostream &operator<<(std::ostream &out, Rule rule);
std::ostream &operator<<(std::ostream &out, MovesText movesRef);

// -------------------------------------------------
// Compression helper

class Compressor
{
public:
    enum class Type { NO_COMPRESS, LZ4_DEFAULT, ZIP_DEFAULT };

    /// Create a compressor with the given algorithm type.
    Compressor(std::ostream &ostream, Type type);
    /// Create a decompressor with the given algorithm type.
    Compressor(std::istream &istream, Type type);
    ~Compressor();

    /// Open an output stream by entry name.
    /// Entry name only work for ZIP type. Any type else should leave
    /// the entry name to empty string.
    /// @return Pointer to output stream, or nullptr if failed to open.
    std::ostream *openOutputStream(std::string entryName = "");

    /// Open an input stream by entry name.
    /// Entry name only work for ZIP type. Any type else should leave
    /// the entry name to empty string.
    /// @return Pointer to input stream, or nullptr if failed to open.
    std::istream *openInputStream(std::string entryName = "");

    /// Close current opened stream in advance. Opened stream will
    /// also be closed by the time of Compressor's destructor called.
    void closeStream(std::ios &stream);

private:
    class CompressorData;
    CompressorData *data;
};
