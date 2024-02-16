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

#include "datawriter.h"

#include "../config.h"
#include "../core/hash.h"
#include "../core/iohelper.h"
#include "../game/board.h"

#include <cassert>
#include <chrono>
#include <execution>
#include <filesystem>
#include <fstream>
#include <future>
#include <iomanip>
#include <npy.hpp>
#include <sstream>

namespace {

/// Packs a bit array into byte array (in big-endian).
/// @param bits The source of bits array
/// @param numBits number of bits to pack
/// @param bytes Destination of byte array
void packBitsToBytes(const uint8_t *bits, size_t numBits, uint8_t *bytes)
{
    size_t numBytesFloored = numBits / 8;
    size_t numBitsRemained = numBits % 8;

    for (size_t byteIdx = 0; byteIdx < numBytesFloored; byteIdx++) {
        *bytes++ = (bits[0] << 7) | (bits[1] << 6) | (bits[2] << 5) | (bits[3] << 4)
                   | (bits[4] << 3) | (bits[5] << 2) | (bits[6] << 1) | (bits[7] << 0);
        bits += 8;
    }

    // Deals with remaining bits that less than a byte
    *bytes = 0;
    for (size_t bitIdx = 0; bitIdx < numBitsRemained; bitIdx++)
        *bytes |= bits[bitIdx] << (7 - bitIdx);
}

/// Gets the total length of a shape.
size_t lengthOfShape(const std::vector<unsigned long> &shape, size_t startDim = 0)
{
    size_t length = 1;
    for (size_t i = startDim; i < shape.size(); i++)
        length *= shape[i];
    return length;
}

uint64_t entryHash(const Tuning::DataEntry &entry)
{
    Hash::XXHasher hasher;
    hasher(entry.position.data(), entry.position.size());
    hasher << entry.boardsize;
    hasher << entry.rule;
    hasher << entry.result;
    hasher << entry.move;
    hasher << entry.eval;
    if (entry.moveData) {
        int numCells = (int)entry.boardsize * (int)entry.boardsize + 1;
        switch (entry.moveDataTag) {
        case Tuning::DataEntry::POLICY_ARRAY_FLOAT: hasher(entry.policyF32, numCells); break;
        case Tuning::DataEntry::POLICY_ARRAY_INT16: hasher(entry.policyI16, numCells); break;
        default: hasher(entry.multiPvMoves, entry.numMultiPv()); break;
        }
    }

    return hasher;
}

}  // namespace

namespace Tuning {

void DataWriter::writeGame(const GameEntry &gameEntry)
{
    writeEntriesInGame(gameEntry);
}

void DataWriter::writeEntriesInGame(const GameEntry                       &gameEntry,
                                    std::function<bool(const DataEntry &)> filter)
{
    assert(gameEntry.initPly < gameEntry.moves.size());

    Color     startSide = gameEntry.initPly % 2 == 0 ? BLACK : WHITE;
    DataEntry dataEntry {{gameEntry.moves.begin(), gameEntry.moves.begin() + gameEntry.initPly},
                         gameEntry.boardsize,
                         gameEntry.rule,
                         startSide == WHITE ? gameEntry.result
                                            : Result(RESULT_WIN - gameEntry.result)};

    for (size_t i = gameEntry.initPly; i < gameEntry.moves.size(); i++) {
        dataEntry.move   = gameEntry.moves[i];
        dataEntry.eval   = i < gameEntry.evals.size() ? gameEntry.evals[i] : VALUE_NONE;
        dataEntry.result = Result(RESULT_WIN - dataEntry.result);
        if (i < gameEntry.moveDatas.size()) {
            dataEntry.moveDataTag = gameEntry.moveDatas[i].tag;
            dataEntry.moveData    = gameEntry.moveDatas[i].moveData;
        }
        else
            dataEntry.moveData = nullptr;

        if (!filter || filter(dataEntry))
            writeEntry(dataEntry);

        dataEntry.position.push_back(gameEntry.moves[i]);
    }

    // Remember to reset the move data pointer, as we do not own it
    dataEntry.moveData = nullptr;
}

// ==============================================

class PlainTextDataWriter::DataStream
{
public:
    DataStream(std::ofstream ofs) : file(std::move(ofs)) {}
    std::ostream &getStream() { return file; }

private:
    std::ofstream file;
};

PlainTextDataWriter::PlainTextDataWriter(std::string filename)
{
    std::ofstream file(filename);
    if (!file.is_open())
        throw std::runtime_error("can not open output file: " + filename);

    dataStream = std::make_unique<DataStream>(std::move(file));
}

PlainTextDataWriter::~PlainTextDataWriter() {}

void PlainTextDataWriter::writeEntry(const DataEntry &entry)
{
    std::ostream &dst = dataStream->getStream();
    dst << int(entry.boardsize) << ',' << entry.rule << ',' << MoveSeqText {entry.position} << ','
        << int(entry.result) << ',' << entry.move << ',' << entry.eval;

    if (entry.numMultiPv() > 0) {
        dst << ',';
        for (int i = 0; i < entry.numMultiPv(); i++) {
            if (i)
                dst << '|';
            auto multiPvMove = entry.multiPvMoves[i];
            dst << multiPvMove.move << '(' << multiPvMove.eval << ')';
        }
    }

    dst << '\n';
}

// ==============================================

class SimpleBinaryDataWriter::DataStream
{
public:
    DataStream(std::ofstream ofs, bool compress)
        : file(std::move(ofs))
        , compressor(file, compress ? Compressor::Type::LZ4_DEFAULT : Compressor::Type::NO_COMPRESS)
        , ostream(compressor.openOutputStream())
    {
        if (!ostream)
            throw std::runtime_error("failed to open output stream");
    }
    std::ostream &getStream() { return *ostream; }

private:
    std::ofstream file;
    Compressor    compressor;
    std::ostream *ostream;
};

SimpleBinaryDataWriter::SimpleBinaryDataWriter(std::string filename, bool compress)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("can not open output file: " + filename);

    dataStream = std::make_unique<DataStream>(std::move(file), compress);
}

SimpleBinaryDataWriter::~SimpleBinaryDataWriter() {}

void SimpleBinaryDataWriter::writeEntry(const DataEntry &entry)
{
    struct EntryHead
    {
        uint16_t result : 2;     // game outcome: 0=loss, 1=draw, 2=win (side to move pov)
        uint16_t ply : 9;        // current number of stones on board
        uint16_t boardsize : 5;  // board size in [5-22]
        uint16_t rule : 3;       // game rule: 0=freestyle, 1=standard, 4=renju
        uint16_t move : 13;      // move output by the engine
    } ehead;
    uint16_t position[MAX_MOVES];  // move sequence that representing a position

    /// Each move is represented by a 16bit unsigned integer. It's lower 10 bits are
    /// constructed with two index x and y using uint16_t move = (x << 5) | y.
    const auto makeMove = [](Pos move) -> uint16_t { return (move.x() << 5) | move.y(); };

    ehead.result    = entry.result;
    ehead.ply       = (uint16_t)entry.position.size();
    ehead.boardsize = entry.boardsize;
    ehead.rule      = entry.rule == RENJU ? 4 : (uint16_t)entry.rule;
    ehead.move      = makeMove(entry.move);
    for (size_t i = 0; i < ehead.ply; i++) {
        position[i] = makeMove(entry.position[i]);
    }

    std::ostream &dst = dataStream->getStream();
    dst.write(reinterpret_cast<char *>(&ehead), sizeof(EntryHead));
    dst.write(reinterpret_cast<char *>(position), sizeof(uint16_t) * ehead.ply);
}

// ==============================================

class PackedBinaryDataWriter::DataStream
{
public:
    DataStream(std::ofstream ofs, bool compress)
        : file(std::move(ofs))
        , compressor(file, compress ? Compressor::Type::LZ4_DEFAULT : Compressor::Type::NO_COMPRESS)
        , ostream(compressor.openOutputStream())
    {
        if (!ostream)
            throw std::runtime_error("failed to open output stream");
    }

    ~DataStream()
    {
        // Flush previous game entry if any
        flushGameEntry(getStream());
    }

    std::ostream &getStream() { return *ostream; }

    void addDataEntry(const DataEntry &dataEntry)
    {
        // Setup game entry for the first data entry
        if (moveSequence.empty() || !checkDataEntryMatched(dataEntry)) {
            // Flush previous game entry if any
            flushGameEntry(getStream());

            // Setup new game entry
            boardSize     = dataEntry.boardsize;
            rule          = dataEntry.rule;
            result        = dataEntry.result;
            curResult     = dataEntry.result;
            curSideToMove = dataEntry.sideToMove();
            totalPly      = (uint32_t)dataEntry.position.size();
            initPosition  = {dataEntry.position.begin(), dataEntry.position.end()};
        }

        PVList pvList;
        // Add main pv move and eval
        pvList.push_back({dataEntry.move, dataEntry.eval});
        // Add extra multi-pv moves and evals
        if (dataEntry.moveData && dataEntry.moveDataTag > DataEntry::MULTIPV_BEGIN) {
            uint32_t numMultiPv = dataEntry.moveDataTag - DataEntry::MULTIPV_BEGIN + 1;
            for (uint32_t i = 1; i < numMultiPv; i++)
                pvList.push_back(dataEntry.multiPvMoves[i]);
        }
        moveSequence.push_back(std::move(pvList));

        if (dataEntry.move != Pos::PASS) {
            curResult     = Result(RESULT_WIN - curResult);
            curSideToMove = ~curSideToMove;
            totalPly++;
        }
    }

    void addGameEntry(const GameEntry &gameEntry)
    {
        // Flush previous game entry if any
        flushGameEntry(getStream());

        // Write game entry
        boardSize    = gameEntry.boardsize;
        rule         = gameEntry.rule;
        result       = gameEntry.result;
        totalPly     = (uint16_t)gameEntry.moves.size();
        initPosition = {gameEntry.moves.begin(), gameEntry.moves.begin() + gameEntry.initPly};

        for (size_t i = gameEntry.initPly; i < gameEntry.moves.size(); i++) {
            PVList pvList;
            // Add main pv move and eval
            pvList.push_back({gameEntry.moves[i],
                              i < gameEntry.evals.size() ? gameEntry.evals[i] : Eval(VALUE_NONE)});
            // Add extra multi-pv moves and evals
            if (i < gameEntry.moveDatas.size() && gameEntry.moveDatas[i].moveData
                && gameEntry.moveDatas[i].tag >= DataEntry::MULTIPV_BEGIN) {
                uint32_t numMultiPv = gameEntry.moveDatas[i].tag - DataEntry::MULTIPV_BEGIN + 1;
                for (uint32_t j = 1; j < numMultiPv; j++)
                    pvList.push_back(gameEntry.moveDatas[i].multiPvMoves[j]);
            }
            // Add pv list to move sequence
            moveSequence.push_back(std::move(pvList));
        }

        // Flush current game entry as it is completed
        flushGameEntry(getStream());
    }

private:
    std::ofstream file;
    Compressor    compressor;
    std::ostream *ostream;

    // PVList contains all PVs in a move.
    using PVList = std::vector<MultiPvMove>;
    // GameEntry represents a full game in the dataset.
    uint8_t             boardSize;      // board size in [5-22]
    Rule                rule;           // game rule: 0=freestyle, 1=standard, 4=renju
    Result              result;         // first player pov
    Result              curResult;      // current result of the game
    Color               curSideToMove;  // current side to move
    uint32_t            totalPly;       // total number of stones on board after game ended
    std::vector<Pos>    initPosition;   // move sequence that representing an opening position
    std::vector<PVList> moveSequence;   // move sequence that representing a game

    bool checkDataEntryMatched(const DataEntry &dataEntry) const
    {
        if (boardSize != dataEntry.boardsize)
            return false;
        if (rule != dataEntry.rule)
            return false;
        if (curResult != dataEntry.result)
            return false;
        if (curSideToMove != dataEntry.sideToMove())
            return false;
        if (dataEntry.position.size() > totalPly)
            return false;
        if (initPosition.size() + moveSequence.size() < dataEntry.position.size())
            return false;

        size_t i = 0;
        for (; i < initPosition.size(); i++) {
            if (initPosition[i] != dataEntry.position[i])
                return false;
        }
        for (const auto &pvList : moveSequence) {
            if (pvList[0].move != dataEntry.position[i])
                return false;
            i++;
        }

        return true;
    }

    void flushGameEntry(std::ostream &os)
    {
        if (moveSequence.empty())
            return;

        struct EntryHead
        {
            uint32_t boardSize : 5;   // board size in [5-22]
            uint32_t rule : 3;        // game rule: 0=freestyle, 1=standard, 4=renju
            uint32_t result : 4;      // game outcome: 0=loss, 1=draw, 2=win (first player pov)
            uint32_t totalPly : 10;   // total number of stones on board after game ended
            uint32_t initPly : 10;    // initial number of stones on board when game started
            uint32_t gameTag : 14;    // game tag of this game, reserved for future use
            uint32_t moveCount : 18;  // the count of move sequence
        } ehead;
        uint16_t position[MAX_MOVES];  // move sequence that representing an opening position

        ehead.boardSize = boardSize;
        ehead.rule      = rule == RENJU ? 4 : (uint16_t)rule;
        ehead.result    = result;
        ehead.totalPly  = totalPly;
        ehead.initPly   = (uint16_t)initPosition.size();
        ehead.gameTag   = 0;
        // Move count is summed for all pv lists
        uint32_t moveCount = 0;
        for (const auto &pvList : moveSequence)
            moveCount += pvList.size();
        ehead.moveCount = moveCount;

        // Write entry header first
        os.write(reinterpret_cast<char *>(&ehead), sizeof(EntryHead));

        /// Each move is represented by a 16bit unsigned integer. It's lower 10 bits are
        /// constructed with two index x and y using uint16_t move = (x << 5) | y.
        const auto makeMove = [](Pos move) -> uint16_t { return (move.x() << 5) | move.y(); };

        // Write initial position
        for (size_t i = 0; i < ehead.initPly; i++)
            position[i] = makeMove(initPosition[i]);
        os.write(reinterpret_cast<char *>(position), sizeof(uint16_t) * ehead.initPly);

        struct Move
        {
            uint16_t isFirst : 1;   // is this move the first in multipv?
            uint16_t isLast : 1;    // is this move the last in multipv?
            uint16_t isNoEval : 1;  // does this move contain no eval info?
            uint16_t isPass : 1;    // is this move a pass move (side not changed after this move)?
            uint16_t reserved : 2;  // reserved for future use
            uint16_t move : 10;     // move output from engine
            int16_t  eval;          // eval output from engine
        } moveData;
        moveData.reserved = 0;
        for (const auto &pvList : moveSequence) {
            for (size_t i = 0; i < pvList.size(); i++) {
                moveData.isFirst  = i == 0;
                moveData.isLast   = i == pvList.size() - 1;
                moveData.isNoEval = pvList[i].eval == VALUE_NONE;
                moveData.isPass   = pvList[i].move == Pos::PASS;
                moveData.move     = makeMove(pvList[i].move);
                moveData.eval     = moveData.isNoEval ? 0 : pvList[i].eval;
                os.write(reinterpret_cast<char *>(&moveData), sizeof(Move));
            }
        }

        // Mark as flushed, and reset the game entry
        initPosition.clear();
        moveSequence.clear();
    }
};

PackedBinaryDataWriter::PackedBinaryDataWriter(std::string filename, bool compress)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file.is_open())
        throw std::runtime_error("can not open output file: " + filename);

    dataStream = std::make_unique<DataStream>(std::move(file), compress);
}

PackedBinaryDataWriter::~PackedBinaryDataWriter() {}

void PackedBinaryDataWriter::writeEntry(const DataEntry &entry)
{
    dataStream->addDataEntry(entry);
}

void PackedBinaryDataWriter::writeGame(const GameEntry &gameEntry)
{
    dataStream->addGameEntry(gameEntry);
}

// ==============================================

class NumpyDataWriter::DataBuffer
{
public:
    size_t bufferedSize() const { return entryBuffer.size(); }

    void addEntry(const DataEntry &entry)
    {
        // Copy entry to buffer
        entryBuffer.push_back(entry);

        // Update entry hash
        hash ^= entryHash(entry);
    }

    void asyncSaveToDir(std::string dirpath, std::function<void(std::string)> finishedCallback)
    {
        // Get file name from entry hash
        std::ostringstream ss;
        ss << std::setw(16) << std::setfill('0') << std::hex << hash;
        auto filename = dirpath + "/" + ss.str() + ".npz";

        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("can not open output file: " + filename);

        // Add processing to async job list
        results.push_back(
            std::async(std::launch::async,
                       [os               = std::move(file),
                        localEntryBuffer = std::move(entryBuffer),  // clears current entry buffer
                        finishedCallback = std::move(finishedCallback),
                        filename         = filename]() mutable {
                           flushToStream(os, std::move(localEntryBuffer));

                           if (finishedCallback)
                               finishedCallback(filename);
                       }));

        // Cleanup previous finished jobs
        cleanupFinishedResult();
        hash = 0;
    }

private:
    uint64_t                       hash = 0;
    std::vector<DataEntry>         entryBuffer;
    std::vector<std::future<void>> results;

    static void flushToStream(std::ostream &os, std::vector<DataEntry> localEntryBuffer)
    {
        // Find max board size
        int maxBoardSize = std::max_element(localEntryBuffer.begin(),
                                            localEntryBuffer.end(),
                                            [](const DataEntry &e1, const DataEntry &e2) {
                                                return e1.boardsize < e2.boardsize;
                                            })
                               ->boardsize;
        unsigned long maxNumCells = maxBoardSize * maxBoardSize;
        unsigned long numBytes    = (maxNumCells + 7) / 8;
        unsigned long numEntries  = localEntryBuffer.size();

        std::vector<unsigned long> binaryInputNCHWPackedShape {numEntries, 3, numBytes};
        std::vector<unsigned long> sparseInputNCHWU8Shape {numEntries, 10, maxNumCells};
        std::vector<unsigned long> sparseInputNCHWU16Shape {numEntries, 2, maxNumCells};
        std::vector<unsigned long> globalInputNCShape {numEntries, 1};
        std::vector<unsigned long> globalTargetsNCShape {numEntries, 3};
        std::vector<unsigned long> policyTargetsNCHWShape {numEntries, 1, maxNumCells};

        std::vector<uint8_t>  binaryInputNCHWPacked(lengthOfShape(binaryInputNCHWPackedShape));
        std::vector<uint8_t>  sparseInputNCHWU8(lengthOfShape(sparseInputNCHWU8Shape));
        std::vector<uint16_t> sparseInputNCHWU16(lengthOfShape(sparseInputNCHWU16Shape));
        std::vector<float>    globalInputNC(lengthOfShape(globalInputNCShape));
        std::vector<float>    globalTargetsNC(lengthOfShape(globalTargetsNCShape));
        std::vector<uint16_t> policyTargetsNCHW(
            lengthOfShape(policyTargetsNCHWShape));  // Quantitize policy target to int16

        const std::vector<uint32_t> sparseInputDim {
            PATTERN_NB,
            PATTERN_NB,
            PATTERN_NB,
            PATTERN_NB,
            PATTERN_NB,
            PATTERN_NB,
            PATTERN_NB,
            PATTERN_NB,
            PATTERN4_NB,
            PATTERN4_NB,
            PCODE_NB,
            PCODE_NB,
        };
        std::vector<unsigned long> sparseInputDimShape {(unsigned long)sparseInputDim.size()};

        size_t binaryInputNCHWStride    = lengthOfShape(binaryInputNCHWPackedShape, 1);
        size_t sparseInputNCHWU8Stride  = lengthOfShape(sparseInputNCHWU8Shape, 1);
        size_t sparseInputNCHWU16Stride = lengthOfShape(sparseInputNCHWU16Shape, 1);
        size_t globalInputNCStride      = lengthOfShape(globalInputNCShape, 1);
        size_t globalTargetsNCtride     = lengthOfShape(globalTargetsNCShape, 1);
        size_t policyTargetsNCHWStride  = lengthOfShape(policyTargetsNCHWShape, 1);

        std::atomic<uint64_t> hash = 0;
        std::for_each(
            std::execution::par_unseq,
            localEntryBuffer.begin(),
            localEntryBuffer.end(),
            [&](DataEntry &e) {
                // Get index of entry
                size_t i = std::distance(localEntryBuffer.data(), &e);

                // Initialize board from position
                Board board(e.boardsize);
                board.newGame(e.rule);
                for (Pos pos : e.position)
                    board.move(e.rule, pos);

                // Update inboard, self, oppo plane
                std::vector<uint8_t> inBoardPlane(maxNumCells, 0);
                std::vector<uint8_t> selfPlane(maxNumCells, 0);
                std::vector<uint8_t> oppoPlane(maxNumCells, 0);
                Color                self = board.sideToMove(), oppo = ~self;
                FOR_EVERY_POSITION(&board, pos)
                {
                    int         posIndex   = pos.y() * board.size() + pos.x();
                    const Cell &c          = board.cell(pos);
                    inBoardPlane[posIndex] = true;
                    selfPlane[posIndex]    = c.piece == self;
                    oppoPlane[posIndex]    = c.piece == oppo;
                    oppoPlane[posIndex]    = c.piece == oppo;

                    // Write sparseInputNCHWU8 and sparseInputNCHWU16
                    sparseInputNCHWU8[i * sparseInputNCHWU8Stride + 0 * maxNumCells + posIndex] =
                        c.pattern(self, 0);
                    sparseInputNCHWU8[i * sparseInputNCHWU8Stride + 1 * maxNumCells + posIndex] =
                        c.pattern(self, 1);
                    sparseInputNCHWU8[i * sparseInputNCHWU8Stride + 2 * maxNumCells + posIndex] =
                        c.pattern(self, 2);
                    sparseInputNCHWU8[i * sparseInputNCHWU8Stride + 3 * maxNumCells + posIndex] =
                        c.pattern(self, 3);
                    sparseInputNCHWU8[i * sparseInputNCHWU8Stride + 4 * maxNumCells + posIndex] =
                        c.pattern(oppo, 0);
                    sparseInputNCHWU8[i * sparseInputNCHWU8Stride + 5 * maxNumCells + posIndex] =
                        c.pattern(oppo, 1);
                    sparseInputNCHWU8[i * sparseInputNCHWU8Stride + 6 * maxNumCells + posIndex] =
                        c.pattern(oppo, 2);
                    sparseInputNCHWU8[i * sparseInputNCHWU8Stride + 7 * maxNumCells + posIndex] =
                        c.pattern(oppo, 3);
                    sparseInputNCHWU8[i * sparseInputNCHWU8Stride + 8 * maxNumCells + posIndex] =
                        c.pattern4[self];
                    sparseInputNCHWU8[i * sparseInputNCHWU8Stride + 9 * maxNumCells + posIndex] =
                        c.pattern4[oppo];
                    sparseInputNCHWU16[i * sparseInputNCHWU16Stride + 0 * maxNumCells + posIndex] =
                        self == BLACK ? c.pcode<BLACK>() : c.pcode<WHITE>();
                    sparseInputNCHWU16[i * sparseInputNCHWU16Stride + 1 * maxNumCells + posIndex] =
                        oppo == BLACK ? c.pcode<BLACK>() : c.pcode<WHITE>();

                    // Write policyTargetsNCHW
                    uint16_t cellPolicy = uint16_t(e.policyTarget(pos) * 65535);
                    policyTargetsNCHW[i * policyTargetsNCHWStride + 0 * maxNumCells + posIndex] =
                        cellPolicy;
                }

                // Write binaryInputNCHWPacked
                packBitsToBytes(inBoardPlane.data(),
                                maxNumCells,
                                &binaryInputNCHWPacked[i * binaryInputNCHWStride + 0 * numBytes]);
                packBitsToBytes(selfPlane.data(),
                                maxNumCells,
                                &binaryInputNCHWPacked[i * binaryInputNCHWStride + 1 * numBytes]);
                packBitsToBytes(oppoPlane.data(),
                                maxNumCells,
                                &binaryInputNCHWPacked[i * binaryInputNCHWStride + 2 * numBytes]);

                // Write globalInputNC
                globalInputNC[i * globalInputNCStride + 0] = (self == BLACK ? -1.0f : 1.0f);

                // Write globalTargetsNC
                globalTargetsNC[i * globalTargetsNCtride + 0] = e.result == RESULT_WIN;
                globalTargetsNC[i * globalTargetsNCtride + 1] = e.result == RESULT_LOSS;
                globalTargetsNC[i * globalTargetsNCtride + 2] = e.result == RESULT_DRAW;
            });

        // Write npz with ZIP compression (in another thread)
        Compressor compressor(os, Compressor::Type::ZIP_DEFAULT);
        auto openEntryAndWrite = [&](std::string entryName, const auto &data, const auto &shape) {
            std::ostream *os = compressor.openOutputStream(entryName);
            if (!os)
                throw std::runtime_error("unable to write " + entryName + " in zip");
            npy::SaveArrayAsNumpy(*os, false, shape.size(), shape.data(), data);
            compressor.closeStream(*os);
        };

        // Write out all ndarray
        openEntryAndWrite("binaryInputNCHWPacked",
                          binaryInputNCHWPacked,
                          binaryInputNCHWPackedShape);
        openEntryAndWrite("sparseInputNCHWU8", sparseInputNCHWU8, sparseInputNCHWU8Shape);
        openEntryAndWrite("sparseInputNCHWU16", sparseInputNCHWU16, sparseInputNCHWU16Shape);
        openEntryAndWrite("globalInputNC", globalInputNC, globalInputNCShape);
        openEntryAndWrite("globalTargetsNC", globalTargetsNC, globalTargetsNCShape);
        openEntryAndWrite("policyTargetsNCHW", policyTargetsNCHW, policyTargetsNCHWShape);
        openEntryAndWrite("sparseInputDim", sparseInputDim, sparseInputDimShape);

        // Clear all entry buffer
        localEntryBuffer.clear();
    }

    void cleanupFinishedResult()
    {
        for (auto it = results.begin(); it != results.end();) {
            if (it->wait_for(std::chrono::seconds(0)) == std::future_status::ready) {
                it->get();
                it = results.erase(it);
            }
            else
                ++it;
        }
    }
};

NumpyDataWriter::NumpyDataWriter(std::string                      dirpath,
                                 size_t                           maxNumEntriesPerFile,
                                 std::function<void(std::string)> flushCallback)
    : buffer(std::make_unique<DataBuffer>())
    , dirpath(dirpath)
    , maxNumEntriesPerFile(maxNumEntriesPerFile)
    , flushCallback(flushCallback)
{}

NumpyDataWriter::~NumpyDataWriter()
{
    if (buffer->bufferedSize())
        buffer->asyncSaveToDir(dirpath, flushCallback);
}

void NumpyDataWriter::writeEntry(const DataEntry &entry)
{
    buffer->addEntry(entry);
    if (buffer->bufferedSize() >= maxNumEntriesPerFile)
        buffer->asyncSaveToDir(dirpath, flushCallback);
}

}  // namespace Tuning
