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

#include "dataentry.h"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace Tuning {

/// DataWriter class is the base class for an iterable-style dataset writer.
class DataWriter
{
public:
    virtual ~DataWriter() = default;

    /// Writes one data entry into current dataset.
    virtual void writeEntry(const DataEntry &entry) = 0;

    /// Writes one game entry into current dataset. A game entry can be decomposed into
    /// multiple data entries for writing (which is the default behaviour for base class).
    virtual void writeGame(const GameEntry &gameEntry);

    /// Decomposed a game entry into multiple data entries for writing.
    /// Filter can be used to filter out part of data entries.
    void writeEntriesInGame(const GameEntry                       &gameEntry,
                            std::function<bool(const DataEntry &)> filter = nullptr);
};

/// PackedBinaryDataWriter implements DataWriter for binary format in c-gomoku-cli.
class PackedBinaryDataWriter : public DataWriter
{
public:
    PackedBinaryDataWriter(std::string filename, bool compress);
    ~PackedBinaryDataWriter();

    void writeEntry(const DataEntry &entry);

private:
    class DataStream;
    std::unique_ptr<DataStream> dataStream;
};

/// NumpyDataWriter implements DataWriter for numpy format.
/// This numpy format is mainly for neural network training.
/// It contains basic board input, global input, value and policy target, as well as
/// pattern/pattern4 input that can be used as features for efficient NN design.
///
/// Format specification: Each entry in npz file records an ndarray, including:
/// 1. binaryInputNCHWPacked, [N, C, ceil(H*W/8)], uint8
///     Binary spatial inputs are packed bitwise, with each (HW) zero-padded to a round byte.
///     Within each byte, bits are packed bigendianwise.
///         Channel 0: is in board
///         Channel 1: self stones
///         Channel 2: opponent stones
/// 2. sparseInputNCHWU8, [N, C0, H*W], uint8
///     Spatial sparse feature inputs, with each feature recorded in sparse index.
///         Channel 0-3: self pattern[dir], dir in {0,1,2,3}
///         Channel 4-7: opponent pattern[dir], dir in {0,1,2,3}
///         Channel 8: self pattern4
///         Channel 9: opponent pattern4
/// 3. sparseInputNCHWU16, [N, C1, H*W], uint16
///     Spatial sparse feature inputs, with each feature recorded in sparse index.
///         Channel 0: self patternCode
///         Channel 1: opponent patternCode
/// 4. globalInputNC, [N, C], float
///     Global input features.
///         Channel 0: color of side to move (black = -1.0, white = 1.0)
/// 5. globalTargetsNC, [N, C], float
///     Global output targets.
///         Channel 0: win probability of current side to move
///         Channel 1: loss probability of current side to move
///         Channel 2: draw probability
/// 6. policyTargetsNCHW, [N, C, H*W], int16
///         Channel 0: policy target this turn
/// 7. sparseInputDim, [C], uint32, C = C0 + C1
///     Dimension of each spatial sparse inputs, including u8 and u16.
class NumpyDataWriter : public DataWriter
{
public:
    /// Constructs a numpy data writer.
    /// @param dirpath The directory of output npz files
    /// @param maxNumEntriesPerFile The maximum number of entries per file
    /// @param flushCallback Optional callback, called with name of file wrote when flush
    NumpyDataWriter(std::string                      dirpath,
                    size_t                           maxNumEntriesPerFile,
                    std::function<void(std::string)> flushCallback = nullptr);
    ~NumpyDataWriter();

    void writeEntry(const DataEntry &entry);

private:
    class DataBuffer;
    std::unique_ptr<DataBuffer>      buffer;
    std::string                      dirpath;
    size_t                           maxNumEntriesPerFile;
    std::function<void(std::string)> flushCallback;
};

}  // namespace Tuning
