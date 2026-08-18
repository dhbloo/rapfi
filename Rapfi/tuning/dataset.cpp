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

#include "dataset.h"

#include "../core/compressor.h"
#include "../core/iohelper.h"
#include "../core/utils.h"
#include "../eval/evaluator.h"
#include "dataformat.h"
#include "datastream.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <lz4Stream.hpp>
#include <npy.hpp>
#include <unordered_set>

namespace {

/// Unpacks a byte array into bit array (in big-endian).
/// @param bytes The source of byte array
/// @param numBits number of bits to unpack
/// @param bits Destination of bits array
void unpackBytesToBits(const uint8_t *bytes, size_t numBits, uint8_t *bits)
{
    size_t numBytesFloored = numBits / 8;
    size_t numBitsRemained = numBits % 8;

    for (size_t byteIdx = 0; byteIdx < numBytesFloored; byteIdx++) {
        uint8_t byte = *bytes++;
        for (int i = 0; i < 8; i++)
            bits[i] = (byte >> (7 - i)) & 0x1;
        bits += 8;
    }

    // Deals with remaining bits that less than a byte
    uint8_t byte = *bytes;
    for (size_t bitIdx = 0; bitIdx < numBitsRemained; bitIdx++)
        bits[bitIdx] = (byte >> (7 - bitIdx)) & 0x1;
}

/// Converting a board array to a pos sequence (in arbitrary order).
void boardArrayToPosSequence(const std::vector<Color> &boardArray,
                             int                       boardSize,
                             std::vector<Pos>         &posSequence)
{
    std::vector<Pos> blackPos, whitePos;
    for (size_t i = 0; i < boardArray.size(); i++) {
        Pos pos(i % boardSize, i / boardSize);
        switch (boardArray[i]) {
        case BLACK: blackPos.push_back(pos); break;
        case WHITE: whitePos.push_back(pos); break;
        default: break;
        }
    }

    assert((int)blackPos.size() - (int)whitePos.size() <= 1);
    posSequence.clear();
    size_t numCommonPos = std::min(blackPos.size(), whitePos.size());
    for (size_t i = 0; i < numCommonPos; i++) {
        posSequence.push_back(blackPos[i]);
        posSequence.push_back(whitePos[i]);
    }

    // Black might have one more move than white
    if (blackPos.size() > numCommonPos)
        posSequence.push_back(blackPos[numCommonPos]);
}

}  // namespace

namespace Tuning {

SimpleBinaryDataset::SimpleBinaryDataset(const std::vector<std::string> &filenames)
    : dataStream(std::make_unique<MultiFileInputStream>(filenames))
{}

SimpleBinaryDataset::SimpleBinaryDataset(const std::vector<std::filesystem::path> &filenames)
    : dataStream(std::make_unique<MultiFileInputStream>(filenames))
{}

SimpleBinaryDataset::~SimpleBinaryDataset() {}

bool SimpleBinaryDataset::next(DataEntry *entry)
{
    // While the current stream has reached its EOF, proceed to the next file
    // (looping over any empty file in the middle of the file list)
    while (dataStream->atStreamEnd()) {
        if (!dataStream->nextFile())
            return false;
    }

    return readBinEntry(dataStream->stream(), entry, scratch);
}

void SimpleBinaryDataset::reset()
{
    dataStream->reset();
}

// ==============================================

class PackedBinaryDataset::DataSource
{
public:
    template <typename Path>
    DataSource(const std::vector<Path> &filenames) : stream(filenames), nextMoveIdx(0)
    {}

    /// Goto the next file in the file list.
    /// @return False when the file list is exhausted, otherwise true.
    bool nextFile() { return stream.nextFile(); }

    /// Get the next data entry from the currently opened file.
    /// This function does NOT advance to the next file: when it returns false, the
    /// caller decides whether to advance (see PackedBinaryDataset::next), mirroring
    /// the nextEntry/nextFile split of the other DataSource classes.
    /// @param dataEntry An optional dataEntry pointer to receive the data.
    /// @return False when the current file has no more entries, otherwise true.
    bool nextEntry(DataEntry *dataEntry)
    {
        // If all moves of the current game are consumed, read the next game from the
        // current file (looping over games with an empty move sequence, if any).
        while (nextMoveIdx >= game.moveSequence.size()) {
            if (!readPackedGame(stream.stream(), game, scratch, maxRecordBytes, retainExtraPVs))
                return false;
            nextMoveIdx = 0;
        }

        if (dataEntry) {
            clearPayload(dataEntry->payload);  // reset payload of the previous entry, if reused
            dataEntry->position.reserve(game.initPosition.size() + nextMoveIdx);
            dataEntry->position.resize(game.initPosition.size() + nextMoveIdx);
            auto posEnd = std::copy(game.initPosition.begin(),
                                    game.initPosition.end(),
                                    dataEntry->position.begin());
            for (size_t i = 0; i < nextMoveIdx; i++)
                *posEnd++ = game.moveSequence[i].move;

            const GameEntry::MoveData &md = game.moveSequence[nextMoveIdx];
            dataEntry->move               = md.move;
            dataEntry->eval               = md.eval;
            dataEntry->boardsize          = game.boardsize;
            dataEntry->rule               = game.rule;
            // GameEntry::result is white pov; the entry wants its side to move's pov.
            dataEntry->result =
                dataEntry->sideToMove() == WHITE ? game.result : flipResult(game.result);

            // Attach the extra multi-pv moves of this ply, if any
            if (const ExtraPVArray *pvs = extraPVs(md.payload))
                payloadAs<ExtraPVArray>(dataEntry->payload).assign(pvs->begin(), pvs->end());
        }

        nextMoveIdx++;
        return true;
    }

    /// Read the next whole game record from the currently opened file (the
    /// nextGame() fast path; do not interleave with nextEntry()).
    /// @return False when the current file has no more games, otherwise true.
    bool nextGameRecord(GameEntry &outGame)
    {
        return readPackedGame(stream.stream(), outGame, scratch, maxRecordBytes, retainExtraPVs);
    }

    void setMaxRecordBytes(size_t bytes) { maxRecordBytes = bytes; }
    void setRetainExtraPVs(bool retain) { retainExtraPVs = retain; }

    /// Reset the state of data source to its initial state.
    void reset()
    {
        stream.reset();
        nextMoveIdx = 0;
        // Discard the cached game: without this, the first nextEntry() after a reset
        // would serve the moves of the previously cached game again.
        game.initPosition.clear();
        game.moveSequence.clear();
    }

private:
    MultiFileInputStream stream;
    GameEntry            game;
    PackedDecodeScratch  scratch;
    size_t               nextMoveIdx;
    size_t               maxRecordBytes = std::numeric_limits<size_t>::max();
    bool                 retainExtraPVs = true;
};

PackedBinaryDataset::PackedBinaryDataset(const std::vector<std::string> &filenames)
    : dataSource(std::make_unique<DataSource>(filenames))
{}

PackedBinaryDataset::PackedBinaryDataset(const std::vector<std::filesystem::path> &filenames)
    : dataSource(std::make_unique<DataSource>(filenames))
{}

PackedBinaryDataset::~PackedBinaryDataset() {}

bool PackedBinaryDataset::next(DataEntry *entry)
{
    // Check if we reached the end of entry list, if so proceeds to the next file
    while (!dataSource->nextEntry(entry)) {
        // Check if we reached the end of file list, if so we have completed the whole dataset
        if (!dataSource->nextFile())
            return false;
    }
    return true;
}

bool PackedBinaryDataset::nextGame(GameEntry *game)
{
    // A skip-only call still has to decode the record to find the next boundary;
    // decode into a discarded local in that case.
    GameEntry  skipped;
    GameEntry &outGame = game ? *game : skipped;

    // Check if we reached the end of the current file, if so proceed to the next file
    while (!dataSource->nextGameRecord(outGame)) {
        // Check if we reached the end of file list, if so we have completed the whole dataset
        if (!dataSource->nextFile())
            return false;
    }
    return true;
}

void PackedBinaryDataset::reset()
{
    dataSource->reset();
}

void PackedBinaryDataset::setMaxRecordBytes(size_t bytes)
{
    dataSource->setMaxRecordBytes(bytes);
}

void PackedBinaryDataset::setRetainExtraPVs(bool retain)
{
    dataSource->setRetainExtraPVs(retain);
}

// ==============================================

class KatagoNumpyDataset::DataSource
{
public:
    struct RawDataEntry
    {
        Color                sideToMove;
        std::vector<Color>   boardInput;
        std::array<float, 3> valueTarget;
        std::vector<int16_t> policyTarget;
    };

    DataSource(std::vector<std::string> filenames)
        : filenames(std::move(filenames))
        , nextFileIdx(0)
        , nextEntryIdx(0)
    {
        nextFile();
    }
    ~DataSource() = default;

    /// Goto the next file in the file list.
    /// @return False when file list reaches the end, otherwise true.
    bool nextFile()
    {
        nextEntryIdx = 0;

        // If we have reached the end of file list
        if (nextFileIdx == filenames.size())
            return false;

        std::ifstream fileStream(filenames[nextFileIdx], std::ios::binary);
        if (!fileStream.is_open())
            throw std::runtime_error("unable to open file " + filenames[nextFileIdx]);

        // fileStream will be set std::istream::badbit
        fileStream.exceptions(std::istream::badbit | std::istream::failbit);

        // Open .npz with ZIP
        Compressor compressor(fileStream, Compressor::Type::ZIP_DEFAULT);

        auto openEntryThen = [&](std::string entryName,
                                 bool (DataSource::*receiver)(std::istream &is)) {
            std::istream *is = compressor.openInputStream(entryName);
            if (!is)
                throw std::runtime_error("unable to open " + entryName + " in file "
                                         + filenames[nextFileIdx]);
            if (!(this->*receiver)(*is))
                throw std::runtime_error("incorrect data in " + entryName + " in file "
                                         + filenames[nextFileIdx]);
            compressor.closeStream(*is);
        };

        openEntryThen("globalInputNC", &DataSource::readSideToMove);
        openEntryThen("binaryInputNCHWPacked", &DataSource::readBoardInput);
        openEntryThen("globalTargetsNC", &DataSource::readValueTarget);
        openEntryThen("policyTargetsNCMove", &DataSource::readPolicyTarget);

        nextFileIdx++;
        return true;
    }

    /// Get the next raw data entry.
    /// @return False when entry list reaches the end, otherwise true.
    bool nextEntry(RawDataEntry &rawDataEntry)
    {
        if (nextEntryIdx == sideToMove.size())
            return false;

        rawDataEntry.sideToMove   = sideToMove[nextEntryIdx];
        rawDataEntry.boardInput   = std::move(boardInput[nextEntryIdx]);
        rawDataEntry.valueTarget  = std::move(valueTarget[nextEntryIdx]);
        rawDataEntry.policyTarget = std::move(policyTarget[nextEntryIdx]);
        nextEntryIdx++;
        return true;
    }

    /// Reset the state of data source to its initial state.
    void reset()
    {
        nextFileIdx = nextEntryIdx = 0;
        sideToMove.clear();
        boardInput.clear();
        valueTarget.clear();
        policyTarget.clear();

        nextFile();
    }

private:
    std::vector<std::string> filenames;
    size_t                   nextFileIdx;
    size_t                   nextEntryIdx;

    std::vector<Color>                sideToMove;    // [N]
    std::vector<std::vector<Color>>   boardInput;    // [N, HW]
    std::vector<std::array<float, 3>> valueTarget;   // [N, 3] win, loss, draw
    std::vector<std::vector<int16_t>> policyTarget;  // [N, HW]

    // Read globalInputNC into (sideToMove)
    bool readSideToMove(std::istream &is)
    {
        // Read ndarray [N, C] float
        std::vector<unsigned long> shape;
        std::vector<float>         data;
        npy::LoadArrayFromNumpy(is, shape, data);
        if (shape.size() != 2)
            return false;

        size_t length      = shape[0];
        int    numChannels = shape[1];

        sideToMove.resize(length);
        for (std::size_t i = 0; i < length; i++) {
            // Channel 5: side to move (black = -1.0, white = 1.0)
            float stmInput = data[i * numChannels + 5];
            sideToMove[i]  = (stmInput < 0 ? BLACK : WHITE);
        }

        return true;
    }

    // Read binaryInputNCHWPacked
    bool readBoardInput(std::istream &is)
    {
        // Read ndarray [N, C, ceil(H*W/8)] uint8
        std::vector<unsigned long> shape;
        std::vector<uint8_t>       data;
        npy::LoadArrayFromNumpy(is, shape, data);
        if (shape.size() != 3)
            return false;

        size_t length      = shape[0];
        int    numChannels = shape[1];
        int    numBytes    = shape[2];
        int    boardSize   = (int)std::sqrt(numBytes * 8);
        int    numCells    = boardSize * boardSize;

        boardInput.resize(length);
        std::vector<uint8_t> boardSelfBits(numCells);
        std::vector<uint8_t> boardOppoBits(numCells);
        uint8_t             *boardSelfBytes = &data[1 * numBytes];  // Channel 1: next player stones
        uint8_t             *boardOppoBytes = &data[2 * numBytes];  // Channel 2: opponent stones
        size_t               stride         = numChannels * numBytes;
        for (std::size_t i = 0; i < length; i++) {
            unpackBytesToBits(boardSelfBytes, numCells, boardSelfBits.data());
            unpackBytesToBits(boardOppoBytes, numCells, boardOppoBits.data());

            // Fill board input according to unpacked bits
            boardInput[i].resize(numCells);
            Color stm = sideToMove[i];
            for (std::size_t j = 0; j < numCells; j++)
                boardInput[i][j] = boardSelfBits[j] ? stm : boardOppoBits[j] ? ~stm : EMPTY;

            // Goto next entry
            boardSelfBytes += stride;
            boardOppoBytes += stride;
        }

        return true;
    }

    // Read globalTargetsNC
    bool readValueTarget(std::istream &is)
    {
        // Read ndarray [N, C] float
        std::vector<unsigned long> shape;
        std::vector<float>         data;
        npy::LoadArrayFromNumpy(is, shape, data);
        if (shape.size() != 2)
            return false;

        size_t length      = shape[0];
        int    numChannels = shape[1];

        valueTarget.resize(length);
        for (std::size_t i = 0; i < length; i++) {
            valueTarget[i][0] = data[i * numChannels + 0];  // Channel 0: win prob
            valueTarget[i][1] = data[i * numChannels + 1];  // Channel 1: loss prob
            valueTarget[i][2] = data[i * numChannels + 2];  // Channel 2: draw prob
        }

        return true;
    }

    // Read policyTargetsNCMove
    bool readPolicyTarget(std::istream &is)
    {
        // Read ndarray [N, C, Pos] int16
        std::vector<unsigned long> shape;
        std::vector<int16_t>       data;
        npy::LoadArrayFromNumpy(is, shape, data);
        if (shape.size() != 3)
            return false;

        size_t length      = shape[0];
        int    numChannels = shape[1];
        int    numCells    = shape[2] - 1;

        // Read policy target without normalize (do that when actually needed)
        policyTarget.resize(length);
        size_t stride = numChannels * (numCells + 1);
        for (std::size_t i = 0; i < length; i++) {
            policyTarget[i].resize(numCells);
            std::copy_n(&data[i * stride], numCells, policyTarget[i].data());
        }

        return true;
    }
};

KatagoNumpyDataset::KatagoNumpyDataset(const std::vector<std::string> &filenames, Rule rule)
    : defaultRule(rule)
{
    if (filenames.empty())
        throw std::runtime_error("no file in katago numpy dataset");

    // Check all file legality
    for (const std::string &filename : filenames) {
        std::ifstream fileStream(filename, std::ios::binary);
        if (!fileStream.is_open())
            throw std::runtime_error("unable to open file " + filename);
    }

    dataSource = std::make_unique<DataSource>(filenames);
}

KatagoNumpyDataset::~KatagoNumpyDataset() {}

bool KatagoNumpyDataset::next(DataEntry *entry)
{
    KatagoNumpyDataset::DataSource::RawDataEntry rawDataEntry;

    // Check if we reached the end of entry list, if so proceeds to the next file
    while (!dataSource->nextEntry(rawDataEntry)) {
        // Check if we reached the end of file list, if so we have completed the whole dataset
        if (!dataSource->nextFile())
            return false;
    }

    if (entry) {
        int numCells  = rawDataEntry.boardInput.size();
        int boardSize = (int)std::sqrt(numCells);  // square board
        boardArrayToPosSequence(rawDataEntry.boardInput, boardSize, entry->position);

        // Create and normalize policy target (in place, reusing the entry's buffer)
        PolicyArrayF32 &policy = payloadAs<PolicyArrayF32>(entry->payload);
        policy.resize(numCells + 1);
        float policySum      = 0.0f;
        float policyMax      = std::numeric_limits<float>::min();
        int   maxPolicyIndex = 0;
        for (size_t i = 0; i < rawDataEntry.policyTarget.size(); i++) {
            policy[i] = (float)rawDataEntry.policyTarget[i];
            policySum += policy[i];
            if (policy[i] > policyMax) {
                policyMax      = policy[i];
                maxPolicyIndex = i;
            }
        }
        for (size_t i = rawDataEntry.policyTarget.size(); i < numCells + 1; i++) {
            policy[i] = 0.0f;
        }
        float invPolicySum = 1.0f / (policySum + 1e-7);
        for (int i = 0; i < numCells + 1; i++)
            policy[i] *= invPolicySum;

        entry->move = Pos(maxPolicyIndex % boardSize, maxPolicyIndex / boardSize);

        // Create value target from already normalized probailities
        Evaluation::ValueType value {rawDataEntry.valueTarget[0],
                                     rawDataEntry.valueTarget[1],
                                     rawDataEntry.valueTarget[2],
                                     false};
        entry->eval = value.value();

        entry->boardsize = boardSize;
        entry->rule      = defaultRule;
        entry->result    = rawDataEntry.valueTarget[0] > 0   ? RESULT_WIN
                           : rawDataEntry.valueTarget[1] > 0 ? RESULT_LOSS
                                                             : RESULT_DRAW;
    }

    return true;
}

void KatagoNumpyDataset::reset()
{
    dataSource->reset();
}

}  // namespace Tuning
