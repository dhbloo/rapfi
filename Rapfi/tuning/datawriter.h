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

class PlainTextDataWriter : public DataWriter
{
public:
    PlainTextDataWriter(std::string filename);
    ~PlainTextDataWriter();

    void writeEntry(const DataEntry &entry);

private:
    class DataStream;
    std::unique_ptr<DataStream> dataStream;
};

/// SimpleBinaryDataWriter implements DataWriter for binary format (.bin) in c-gomoku-cli.
/// struct Entry {
///     uint16_t result : 2;     // game outcome: 0=loss, 1=draw, 2=win (side to move pov)
///     uint16_t ply : 9;        // current number of stones on board
///     uint16_t boardsize : 5;  // board size in [5-22]
///     uint16_t rule : 3;       // game rule: 0=freestyle, 1=standard, 4=renju
///     uint16_t move : 13;      // move output by the engine
///     uint16_t position[ply];  // move sequence that representing a position
/// };
class SimpleBinaryDataWriter : public DataWriter
{
public:
    SimpleBinaryDataWriter(std::string filename, bool compress);
    ~SimpleBinaryDataWriter();

    void writeEntry(const DataEntry &entry);

private:
    class DataStream;
    std::unique_ptr<DataStream> dataStream;
};

/// PackedBinaryDataWriter implements DataWriter for packed binary (.binpack) in c-gomoku-cli.
/// It supports plain binary file and LZ4 compressed binary file.
///
/// Packed binary format is the new training data storage format designed to take advantage of
/// position chains differing by a single move, thus saving plenty of disk space even without
/// compression. It also stores more information compared to the original binary format, such
/// as eval of each move and possible multipv outputs. Each game entry contains a head and a
/// following move sequence. The 8 byte head contains information of one game, such as board
/// size, rule, outcome, total ply, initial opening ply and position. The move sequence contains
/// each 4 byte (multipv) move output with its eval. For multipv mode, each ply can contain
/// multiple moves which are indicated by the bitmask, and the first multipv mode is always
/// played to get the next position.
///
/// struct Entry {
///   uint32_t boardSize : 5;     // board size in [5-22]
///   uint32_t rule : 3;          // game rule: 0=freestyle, 1=standard, 4=renju
///   uint32_t result : 4;        // game outcome: 0=loss, 1=draw, 2=win (first player pov)
///   uint32_t totalPly : 10;     // total number of stones on board after game ended
///   uint32_t initPly : 10;      // initial number of stones on board when game started
///   uint32_t gameTag : 14;      // game tag of this game, reserved for future use
///   uint32_t moveCount : 18;    // the count of move sequence
///   uint16_t position[initPly]; // move sequence that representing an opening position
///   struct Move
///   {
///       uint16_t isFirst : 1;   // is this move the first in multipv?
///       uint16_t isLast : 1;    // is this move the last in multipv?
///       uint16_t isNoEval : 1;  // does this move contain no eval info?
///       uint16_t isPass : 1;    // is this move a pass move (side not changed after this move)?
///       uint16_t reserved : 2;  // reserved for future use
///       uint16_t move : 10;     // move output from engine
///       int16_t  eval;          // eval output from engine
///   } moveSequence[moveCount];  // move sequence that representing the full game
/// };
class PackedBinaryDataWriter : public DataWriter
{
public:
    PackedBinaryDataWriter(std::string filename, bool compress);
    ~PackedBinaryDataWriter();

    void writeEntry(const DataEntry &entry);
    void writeGame(const GameEntry &gameEntry);

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
///         Channel 0-3: win-loss-draw probability from current side to move.
/// 6. policyTargetsNCMove, [N, C, num_moves], int16
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
    /// @param writeSparseInputs Whether to write sparse pattern inputs
    NumpyDataWriter(std::string                      dirpath,
                    size_t                           maxNumEntriesPerFile,
                    std::function<void(std::string)> flushCallback     = nullptr,
                    bool                             writeSparseInputs = true);
    ~NumpyDataWriter();

    void writeEntry(const DataEntry &entry);
    void writeEntryWithSoftValueTarget(const DataEntry &entry,
                                       float            winprob,
                                       float            loseprob,
                                       float            drawprob);

private:
    class DataBuffer;
    std::unique_ptr<DataBuffer>      buffer;
    std::string                      dirpath;
    size_t                           maxNumEntriesPerFile;
    std::function<void(std::string)> flushCallback;
    bool                             writeSparseInputs;
};

}  // namespace Tuning
