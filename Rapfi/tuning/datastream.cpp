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

#include "datastream.h"

#include "../core/compressor.h"
#include "../core/filesystem.h"

#include <cassert>
#include <stdexcept>

namespace Tuning {

namespace {

std::vector<std::filesystem::path> pathsFromConsoleStrings(const std::vector<std::string> &paths)
{
    std::vector<std::filesystem::path> nativePaths;
    nativePaths.reserve(paths.size());
    for (const std::string &path : paths)
        nativePaths.push_back(pathFromConsoleString(path));
    return nativePaths;
}

}  // namespace

MultiFileInputStream::MultiFileInputStream(const std::vector<std::string> &filenames)
    : MultiFileInputStream(pathsFromConsoleStrings(filenames))
{}

MultiFileInputStream::MultiFileInputStream(
    const std::vector<std::filesystem::path> &filenames)
    : filenames_(filenames)
    , nextFileIdx_(0)
    , istream_(nullptr)
{
    if (filenames_.empty())
        throw std::runtime_error("no file in dataset");

    // Preserve the old fail-fast constructor contract without retaining one
    // file handle and stream buffer for every source file.
    for (const std::filesystem::path &filename : filenames_) {
        std::ifstream fileStream(filename, std::ios::binary);
        if (!fileStream.is_open())
            throw std::runtime_error("unable to open file " + filename.u8string());
    }

    nextFile();
}

// Out-of-line so the header can hold Compressor by unique_ptr as an incomplete type.
MultiFileInputStream::~MultiFileInputStream() = default;

bool MultiFileInputStream::nextFile()
{
    if (nextFileIdx_ == filenames_.size())
        return false;

    // Delete the previous decompressor before closing the stream it references.
    istream_ = nullptr;
    compressor_.reset();
    if (file_.is_open()) {
        file_.exceptions(std::ios::goodbit);
        file_.close();
    }
    file_.clear();

    const std::filesystem::path &filename = filenames_[nextFileIdx_];
    file_.open(filename, std::ios::binary);
    if (!file_.is_open())
        throw std::runtime_error("unable to open file " + filename.u8string());
    file_.exceptions(std::istream::badbit | std::istream::failbit);

    try {
        // Probe the LZ4 frame magic. A zero-byte file skips the probe (the read
        // would throw, while peek only sets eofbit) and is treated as an
        // uncompressed stream that the caller will skip normally.
        int magic = 0;
        if (file_.peek() != std::ifstream::traits_type::eof()) {
            file_.read(reinterpret_cast<char *>(&magic), sizeof(magic));
            file_.seekg(0);
        }

        compressor_ = std::make_unique<Compressor>(
            file_,
            magic == 0x184D2204 ? Compressor::Type::LZ4_DEFAULT
                                : Compressor::Type::NO_COMPRESS);
        istream_ = compressor_->openInputStream();
        nextFileIdx_++;
    }
    catch (const std::exception &e) {
        throw std::runtime_error("unable to open dataset stream " + filename.u8string() + ": "
                                 + e.what());
    }

    if (!istream_)
        throw std::runtime_error("unable to open dataset stream " + filename.u8string());

    return true;
}

void MultiFileInputStream::reset()
{
    istream_ = nullptr;
    compressor_.reset();
    if (file_.is_open()) {
        file_.exceptions(std::ios::goodbit);
        file_.close();
    }
    file_.clear();
    nextFileIdx_ = 0;
    nextFile();
}

}  // namespace Tuning
