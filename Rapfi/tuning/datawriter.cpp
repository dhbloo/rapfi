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
    if (entry.policy) {
        int numCells = (int)entry.boardsize * (int)entry.boardsize;
        hasher(entry.policy.get(), numCells);
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
    size_t startPly = std::max(gameEntry.numOpeningMoves, 0);
    assert(startPly < gameEntry.moves.size());

    Color     startSide = startPly % 2 == 0 ? BLACK : WHITE;
    DataEntry dataEntry {
        {gameEntry.moves.begin(), gameEntry.moves.begin() + startPly},
        gameEntry.boardsize,
        gameEntry.rule,
        startSide == WHITE ? gameEntry.result : Result(RESULT_WIN - gameEntry.result),
    };

    for (size_t i = startPly; i < gameEntry.moves.size(); i++) {
        dataEntry.move   = gameEntry.moves[i];
        dataEntry.result = Result(RESULT_WIN - dataEntry.result);

        if (!filter || filter(dataEntry))
            writeEntry(dataEntry);

        dataEntry.position.push_back(gameEntry.moves[i]);
    }
}

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
    std::ostream &getStream() { return *ostream; }

private:
    std::ofstream file;
    Compressor    compressor;
    std::ostream *ostream;
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

class NumpyDataWriter::DataBuffer
{
public:
    size_t bufferedSize() const { return entryBuffer.size(); }

    void addEntry(const DataEntry &entry)
    {
        // Copy policy array of the source entry if needed
        std::unique_ptr<float[]> policy = nullptr;
        if (entry.policy) {
            int numCells = (int)entry.boardsize * (int)entry.boardsize;
            policy       = std::make_unique<float[]>(numCells);
            std::copy_n(entry.policy.get(), numCells, policy.get());
        }

        entryBuffer.push_back({entry.position,
                               entry.boardsize,
                               entry.rule,
                               entry.result,
                               entry.move,
                               std::move(policy)});

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
