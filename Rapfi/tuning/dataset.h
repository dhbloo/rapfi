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

#include <memory>
#include <string>

namespace Tuning {

/// Dataset class is the base class for a sequential iterable-style dataset.
class Dataset
{
public:
    virtual ~Dataset() = default;

    /// Reads one data entry. Returns false if it reaches EOF and there is no more entry to read.
    /// If out is nullptr, data entry will not be read but skipped. This can be used to quickly
    /// check the total number of data entries in the dataset.
    /// Throws exception when stream operation failed or the dataset is corrupted.
    /// This function is not thread-safe.
    virtual bool next(DataEntry *entry) = 0;

    /// Reset the read cursor to the beginning of the dataset.
    virtual void reset() = 0;
};

/// SimpleBinaryDataset implements Dataset for binary format (.bin) in c-gomoku-cli.
/// It supports plain binary file and LZ4 compressed binary file.
///
/// Format specification: Binary format uses variable length encoding show below, which is easy
/// to parse for engines. Each entry has a length of (4+ply) bytes. Position is represented by
/// a move sequence that black plays first. Move sequence is guaranteed to have the same order
/// as the actual game record.
/// struct Entry {
///     uint16_t result : 2;     // game outcome: 0=loss, 1=draw, 2=win (side to move pov)
///     uint16_t ply : 9;        // current number of stones on board
///     uint16_t boardsize : 5;  // board size in [5-22]
///     uint16_t rule : 3;       // game rule: 0=freestyle, 1=standard, 4=renju
///     uint16_t move : 13;      // move output by the engine
///     uint16_t position[ply];  // move sequence that representing a position
/// };
class SimpleBinaryDataset : public Dataset
{
public:
    // Throws expcetion when an error occured when reading the file
    SimpleBinaryDataset(const std::vector<std::string> &filenames);
    ~SimpleBinaryDataset();

    bool next(DataEntry *entry) override;
    void reset() override;

private:
    class DataSource;
    std::unique_ptr<DataSource> dataSource;
};

/// PackedBinaryDataset implements Dataset for packed binary format (.binpack) in c-gomoku-cli.
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
class PackedBinaryDataset : public Dataset
{
public:
    /// Creates a packed binary dataset from a list of filenames.
    /// @note Throws expcetion when an error occured when reading the file.
    PackedBinaryDataset(const std::vector<std::string> &filenames);
    ~PackedBinaryDataset();

    bool next(DataEntry *entry) override;
    void reset() override;

private:
    class DataSource;
    std::unique_ptr<DataSource> dataSource;
};

/// KatagoNumpyDataset implements Dataset for katago training data format (.npz).
///
/// Format specification: Each entry in npz file records an ndarray, only some of them
/// are needed for building a DataEntry, including:
/// 1. binaryInputNCHWPacked, [N, C, ceil(H*W/8)], int8
///     Binary spatial inputs are packed bitwise, with each (HW) zero-padded to a round byte.
///     Within each byte, bits are packed bigendianwise.
///         Channel 0: is in board
///         Channel 1: next player stones
///         Channel 2: opponent stones
/// 2. globalInputNC, [N, C], float
///     Global input features.
///         Channel 5: color of side to move (black = -1.0, white = 1.0)
/// 3. globalTargetsNC, [N, C], float
///     Global output targets.
///         Channel 0-3: win-loss-draw probability of current side to move
/// 4. policyTargetsNCMove, [N, C, Pos], int16
///     Pos dimension length is BoardSize*BoardSize+1, due to the pass input.
///         Channel 0: policy target this turn
///         Channel 1: policy target next turn
class KatagoNumpyDataset : public Dataset
{
public:
    /// Construct a katago numpy dataset from a list of npz file names.
    /// Rule is needed since rule infomation is not recorded in npz files.
    KatagoNumpyDataset(const std::vector<std::string> &filenames, Rule rule);
    ~KatagoNumpyDataset();

    bool next(DataEntry *entry) override;
    void reset() override;

private:
    class DataSource;
    std::unique_ptr<DataSource> dataSource;
    Rule                        defaultRule;
};

}  // namespace Tuning
