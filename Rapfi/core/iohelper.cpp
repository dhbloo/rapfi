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

#include "../config.h"

#include <cassert>
#include <functional>
#include <lz4Stream.hpp>
#include <memory>
#include <mutex>
#include <sstream>
#ifdef COMMAND_MODULES
    #define WITH_ZIP
    #include <zip.h>
#endif

constexpr int PASS_COORD_X = -1;
constexpr int PASS_COORD_Y = -1;

static std::mutex &getSyncIOMutex()
{
    // We never delete this mutex to ensure that we can use it in static destructors,
    // the OS will reclaim the memory when the program exits anyway.
    static std::mutex *syncMutex = new std::mutex;
    return *syncMutex;
}

SyncOutputStream::SyncOutputStream(std::ostream &os) : os(os)
{
    getSyncIOMutex().lock();
}

SyncOutputStream::~SyncOutputStream()
{
    getSyncIOMutex().unlock();
}

// -------------------------------------------------

Pos inputCoordConvert(int x, int y, int boardsize)
{
    if (x == PASS_COORD_X && y == PASS_COORD_Y)
        return Pos::PASS;

    if (Config::IOCoordMode == CoordConvertionMode::FLIPY_X)
        return {y, boardsize - 1 - x};
    else if (Config::IOCoordMode == CoordConvertionMode::X_FLIPY)
        return {x, boardsize - 1 - y};
    else
        return {x, y};
}

int outputCoordXConvert(Pos pos, int boardsize)
{
    if (pos == Pos::PASS)
        return PASS_COORD_X;

    if (Config::IOCoordMode == CoordConvertionMode::FLIPY_X)
        return boardsize - 1 - pos.y();
    else
        return pos.x();
}

int outputCoordYConvert(Pos pos, int boardsize)
{
    if (pos == Pos::PASS)
        return PASS_COORD_Y;

    if (Config::IOCoordMode == CoordConvertionMode::FLIPY_X)
        return pos.x();
    else if (Config::IOCoordMode == CoordConvertionMode::X_FLIPY)
        return boardsize - 1 - pos.y();
    else
        return pos.y();
}

Pos parseCoord(std::string coordStr)
{
    if (coordStr == "Pass" || coordStr == "pass")
        return Pos::PASS;
    if (coordStr == "None" || coordStr == "none")
        return Pos::NONE;

    int x = std::toupper(coordStr[0]) - 'A';
    int y = std::atoi(coordStr.data() + 1) - '1';
    return Pos(x, y);
}

// -------------------------------------------------

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

    const char *ColorName[] = {"Black", "White", "Wall", "Empty"};
    return out << ColorName[color];
}

std::ostream &operator<<(std::ostream &out, Pattern p)
{
    assert(p >= DEAD && p < PATTERN_NB);

    const char *PatternName[] = {"DEAD",
                                 "OVERLINE",
                                 "B1",
                                 "F1",
                                 "B2",
                                 "F2",
                                 "F2A",
                                 "F2B",
                                 "B3",
                                 "F3",
                                 "F3S",
                                 "B4",
                                 "F4",
                                 "F5"};
    return out << PatternName[p];
}

std::ostream &operator<<(std::ostream &out, Pattern4 p4)
{
    assert(p4 >= NONE && p4 < PATTERN4_NB);

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

    const char *RuleName[] = {"Freestyle", "Standard", "Renju"};
    return out << RuleName[rule];
}

std::ostream &operator<<(std::ostream &out, MovesText movesRef)
{
    for (size_t i = 0; i < movesRef.moves.size(); i++) {
        if (movesRef.withSpace && i)
            out << ' ';
        if (movesRef.rawCoords)
            out << outputCoordXConvert(movesRef.moves[i], movesRef.boardsize) << ','
                << outputCoordYConvert(movesRef.moves[i], movesRef.boardsize);
        else
            out << movesRef.moves[i];
    }
    return out;
}

// -------------------------------------------------

class Compressor::CompressorData
{
    friend class Compressor;

    template <typename StreamType>
    struct CStream
    {
        std::unique_ptr<StreamType>                    stream;
        std::string                                    entryName;
        std::function<void(StreamType &, std::string)> finialize;

        CStream(std::unique_ptr<StreamType>                    stream,
                std::string                                    entryName,
                std::function<void(StreamType &, std::string)> finialize = nullptr)
            : stream(std::move(stream))
            , entryName(entryName)
            , finialize(finialize)
        {}
        CStream(CStream &&)            = default;
        CStream &operator=(CStream &&) = default;
        ~CStream()
        {
            if (stream && finialize)
                finialize(*stream, entryName);
        }
    };

    Type                               type        = Type::NO_COMPRESS;
    std::ostream                      *ostreamSink = nullptr;
    std::istream                      *istreamSink = nullptr;
    std::vector<CStream<std::ostream>> openedOutputStreams;
    std::vector<CStream<std::istream>> openedInputStreams;
#ifdef WITH_ZIP
    std::string buffer;
    zip_t      *zip = nullptr;
#endif
};

Compressor::Compressor(std::ostream &ostream, Type type) : data(new CompressorData)
{
    data->type        = type;
    data->ostreamSink = &ostream;

#ifdef WITH_ZIP
    if (type == Type::ZIP_DEFAULT) {
        data->zip = zip_stream_open(nullptr, 0, ZIP_DEFAULT_COMPRESSION_LEVEL, 'w');
    }
#endif
}

Compressor::Compressor(std::istream &istream, Type type) : data(new CompressorData)
{
    data->type        = type;
    data->istreamSink = &istream;

#ifdef WITH_ZIP
    if (type == Type::ZIP_DEFAULT) {
        data->buffer = {std::istreambuf_iterator<char>(istream), {}};
        data->zip    = zip_stream_open(data->buffer.c_str(), data->buffer.size(), 0, 'r');
    }
#endif
}

Compressor::~Compressor()
{
#ifdef WITH_ZIP
    if (data->type == Type::ZIP_DEFAULT) {
        if (data->ostreamSink) {
            char  *outbuf     = nullptr;
            size_t outbufSize = 0;
            zip_stream_copy(data->zip, (void **)(&outbuf), &outbufSize);
            data->ostreamSink->write(outbuf, outbufSize);
            free(outbuf);
        }

        zip_stream_close(data->zip);
    }
#endif

    delete data;
}

std::ostream *Compressor::openOutputStream(std::string entryName)
{
    assert(data->ostreamSink && "can not open output stream for input sink");

    // First find if this entry has been opened
    for (auto &s : data->openedOutputStreams) {
        if (s.entryName == entryName)
            return s.stream.get();
    }

    switch (data->type) {
    case Type::LZ4_DEFAULT: {
        assert(entryName == "");
        static const LZ4F_preferences_t LZ4Perf = {{LZ4F_default,
                                                    LZ4F_blockLinked,
                                                    LZ4F_contentChecksumEnabled,
                                                    LZ4F_frame,
                                                    0ULL,
                                                    0U,
                                                    LZ4F_noBlockChecksum},
                                                   3,
                                                   0u,
                                                   0u,
                                                   {0u, 0u, 0u}};
        data->openedOutputStreams.emplace_back(
            std::make_unique<lz4_stream::ostream>(*data->ostreamSink, LZ4Perf),
            entryName);
    } break;
    case Type::ZIP_DEFAULT:
#ifdef WITH_ZIP
        data->openedOutputStreams.emplace_back(
            std::make_unique<std::stringstream>(),
            entryName,
            [zip = data->zip](std::ostream &os, std::string entryName) {
                assert(zip);
                std::stringstream &ss = static_cast<std::stringstream &>(os);
                std::string        buffer(std::istreambuf_iterator<char>(ss), {});
                zip_entry_open(zip, entryName.c_str());
                zip_entry_write(zip, buffer.c_str(), buffer.size());
                zip_entry_close(zip);
            });
        break;
#else
        throw "Zip is not enabled in this build";
#endif
    default: return data->ostreamSink;
    }
    return data->openedOutputStreams.back().stream.get();
}

std::istream *Compressor::openInputStream(std::string entryName)
{
    assert(data->istreamSink && "can not open input stream for output sink");

    // First find if this entry has been opened
    for (auto &s : data->openedInputStreams) {
        if (s.entryName == entryName)
            return s.stream.get();
    }

    switch (data->type) {
    case Type::LZ4_DEFAULT: {
        assert(entryName == "");
        data->openedInputStreams.emplace_back(
            std::make_unique<lz4_stream::istream>(*data->istreamSink),
            entryName);
    } break;
    case Type::ZIP_DEFAULT:
#ifdef WITH_ZIP
    {
        assert(data->zip);
        auto   ss         = std::make_unique<std::stringstream>();
        char  *outbuf     = nullptr;
        size_t outbufSize = 0;
        if (zip_entry_open(data->zip, entryName.c_str()) < 0)
            return nullptr;
        if (zip_entry_read(data->zip, (void **)(&outbuf), &outbufSize) < 0)
            return nullptr;
        zip_entry_close(data->zip);
        ss->write(outbuf, outbufSize);
        free(outbuf);

        data->openedInputStreams.emplace_back(std::move(ss), entryName);
    } break;
#else
        throw "Zip is not enabled in this build";
#endif
    default: return data->istreamSink;
    }
    return data->openedInputStreams.back().stream.get();
}

void Compressor::closeStream(std::ios &stream)
{
    for (auto it = data->openedOutputStreams.begin(); it != data->openedOutputStreams.end(); it++) {
        if (&stream == static_cast<std::ios *>(it->stream.get())) {
            data->openedOutputStreams.erase(it);
            return;
        }
    }
    for (auto it = data->openedInputStreams.begin(); it != data->openedInputStreams.end(); it++) {
        if (&stream == static_cast<std::ios *>(it->stream.get())) {
            data->openedInputStreams.erase(it);
            return;
        }
    }
    assert(false && "Compressor::closeStream(): invalid stream");
}
