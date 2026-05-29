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

#include "compressor.h"

#include <cassert>
#include <functional>
#include <istream>
#include <lz4Stream.hpp>
#include <memory>
#include <ostream>
#include <sstream>
#include <stdexcept>
#include <vector>

#ifdef COMMAND_MODULES
    #define WITH_ZIP
    #include <zip.h>
#endif

// Multi-entry archives (ZIP) and single-entry frames (LZ4) share a small pimpl: the public
// open*Stream() functions return a pointer to a buffered sub-stream owned by `data`, and the
// finaliser stored alongside the sub-stream is run when the Compressor (or closeStream) drops
// the entry.

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
        // Each open sub-stream's finalizer flushes its entry into `data->zip`. Run them now (by
        // destroying the sub-streams) BEFORE serializing and closing the archive, otherwise the
        // copy below would miss those entries and the finalizers would touch a closed zip.
        data->openedOutputStreams.clear();
        data->openedInputStreams.clear();

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
            // Flush the in-memory entry into the ZIP archive when the sub-stream is dropped.
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
        throw std::runtime_error("Zip is not enabled in this build");
#endif
    default: return data->ostreamSink;
    }
    return data->openedOutputStreams.back().stream.get();
}

std::istream *Compressor::openInputStream(std::string entryName)
{
    assert(data->istreamSink && "can not open input stream for output sink");

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
        throw std::runtime_error("Zip is not enabled in this build");
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
