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

#include <cassert>
#include <filesystem>
#include <fstream>
#include <iosfwd>
#include <memory>
#include <string>
#include <vector>

class Compressor;  // forward declaration (core/compressor.h)

namespace Tuning {

/// MultiFileInputStream chains a list of binary dataset files into a sequence
/// of per-file input streams, transparently decompressing LZ4-framed files
/// (detected by the LZ4 frame magic; anything else is read as-is). Zero-byte
/// files open successfully and report EOF on the first read, so callers skip
/// them with their regular end-of-file advance loop.
///
/// Usage contract: the constructor validates every path, opens only the first
/// file, and positions the stream there, so stream() is valid immediately.
/// When a read hits EOF, call nextFile() and retry; nextFile() returns false
/// once the file list is exhausted. reset() rewinds to the first file.
class MultiFileInputStream
{
public:
    /// Validate all dataset paths and open the first file. The list must not be empty.
    /// @throws std::runtime_error when a file cannot be opened.
    explicit MultiFileInputStream(const std::vector<std::string> &filenames);
    explicit MultiFileInputStream(const std::vector<std::filesystem::path> &filenames);
    ~MultiFileInputStream();

    /// Advance to the next file in the list and open its stream.
    /// @return False when the file list is exhausted, otherwise true.
    bool nextFile();

    /// The input stream of the currently opened file.
    std::istream &stream()
    {
        assert(istream_);
        return *istream_;
    }

    /// Whether the current stream has no more bytes to read.
    bool atStreamEnd()
    {
        return stream().eof() || stream().peek() == std::istream::traits_type::eof();
    }

    /// Rewind the whole stream chain back to the first file.
    void reset();

private:
    std::vector<std::filesystem::path> filenames_;
    std::ifstream               file_;
    size_t                      nextFileIdx_;
    std::unique_ptr<Compressor> compressor_;
    std::istream               *istream_;
};

}  // namespace Tuning
