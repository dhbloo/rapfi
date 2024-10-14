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

enum SyncFlag { IO_LOCK, IO_UNLOCK };
std::ostream &operator<<(std::ostream &, SyncFlag);

#define MESSAGEL(message) std::cout << IO_LOCK << "MESSAGE " << message << std::endl << IO_UNLOCK
#define ERRORL(message)   std::cout << IO_LOCK << "ERROR " << message << std::endl << IO_UNLOCK
#ifdef NDEBUG
    #define DEBUGL(message) ((void)0)
#else
    #define DEBUGL(message) std::cout << IO_LOCK << "DEBUG " << message << std::endl << IO_UNLOCK
#endif

// -------------------------------------------------
// Coordinate conversion

Pos inputCoordConvert(int x, int y, int boardsize);
int outputCoordXConvert(Pos pos, int boardsize);
int outputCoordYConvert(Pos pos, int boardsize);

Pos parseCoord(std::string coordStr);

// -------------------------------------------------
// Formatters

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
