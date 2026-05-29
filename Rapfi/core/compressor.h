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

#include <iosfwd>
#include <string>

/// Stream wrapper for reading or writing LZ4 frames or ZIP archives. Construct with the
/// underlying byte stream and a backend `Type`, then call `openOutputStream` /
/// `openInputStream` to obtain a sub-stream for one entry (LZ4 has exactly one entry; ZIP
/// supports multiple, addressed by name). Sub-streams are closed automatically by the
/// `Compressor` destructor.
class Compressor
{
public:
    enum class Type {
        NO_COMPRESS,  ///< Pass-through; the sub-stream is the sink/source unchanged.
        LZ4_DEFAULT,  ///< Single-entry LZ4 frame.
        ZIP_DEFAULT,  ///< Multi-entry ZIP archive (requires command modules to be enabled).
    };

    /// Wrap an output byte stream for compressed writing.
    Compressor(std::ostream &ostream, Type type);

    /// Wrap an input byte stream for compressed reading.
    Compressor(std::istream &istream, Type type);

    // Owns a raw pimpl with a freeing destructor; copying would double-free and moving would
    // leave a dangling source, so neither is allowed.
    Compressor(const Compressor &)            = delete;
    Compressor(Compressor &&)                 = delete;
    Compressor &operator=(const Compressor &) = delete;
    Compressor &operator=(Compressor &&)      = delete;

    ~Compressor();

    /// Open (or retrieve, if already open) a writable sub-stream.
    /// `entryName` is only consulted for `ZIP_DEFAULT`; other backends require it to be empty.
    /// Returns nullptr on failure. Throws std::runtime_error if `ZIP_DEFAULT` is requested in
    /// a build without command modules.
    std::ostream *openOutputStream(std::string entryName = "");

    /// Open (or retrieve, if already open) a readable sub-stream.
    /// `entryName` is only consulted for `ZIP_DEFAULT`; other backends require it to be empty.
    /// Returns nullptr on failure (including: ZIP entry not found). Throws std::runtime_error
    /// if `ZIP_DEFAULT` is requested in a build without command modules.
    std::istream *openInputStream(std::string entryName = "");

    /// Eagerly close a sub-stream previously returned by open*Stream(). All still-open
    /// sub-streams are closed automatically by the Compressor destructor.
    void closeStream(std::ios &stream);

private:
    class CompressorData;
    CompressorData *data;
};
