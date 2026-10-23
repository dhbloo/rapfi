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

/// NumpyDataWriter implementation: batching, background flushing and npz
/// serialization. The training-plane layout itself lives in featureencoder.

#include "../core/compressor.h"
#include "../core/filesystem.h"
#include "../core/hash.h"
#include "../core/iohelper.h"
#include "../game/board.h"
#include "datawriter.h"
#include "featureencoder.h"

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <future>
#include <iomanip>
#include <npy.hpp>
#include <sstream>

namespace {

using namespace Tuning;

/// Gets the total length of a shape.
size_t lengthOfShape(const std::vector<unsigned long> &shape, size_t startDim = 0)
{
    size_t length = 1;
    for (size_t i = startDim; i < shape.size(); i++)
        length *= shape[i];
    return length;
}

void hashPayload(Hash::XXHasher &hasher, const MovePayload &payload, bool hashExtraPVs)
{
    std::visit(
        [&hasher, hashExtraPVs](const auto &v) {
            using PayloadT = std::decay_t<decltype(v)>;
            if constexpr (!std::is_same_v<PayloadT, std::monostate>) {
                if constexpr (std::is_same_v<PayloadT, ExtraPVArray>) {
                    if (!hashExtraPVs)
                        return;
                }
                if (!v.empty())
                    hasher(v.data(), v.size());
            }
        },
        payload);
}

void hashPolicyConfig(Hash::XXHasher &hasher, const PolicyTargetConfig &config)
{
    hasher << config.multiPVTemperature;
    hasher << config.evalScalingFactor;
}

uint64_t entryHash(const DataEntry &entry, const PolicyTargetConfig &policyConfig)
{
    Hash::XXHasher hasher;
    hasher(entry.position.data(), entry.position.size());
    hasher << entry.boardsize;
    hasher << entry.rule;
    hasher << entry.result;
    hasher << entry.move;
    hasher << entry.eval;
    hashPayload(hasher, entry.payload, true);
    if (policyConfig.useMultiPV())
        hashPolicyConfig(hasher, policyConfig);

    return hasher;
}

uint64_t gameHash(const GameEntry &game, const PolicyTargetConfig &policyConfig)
{
    Hash::XXHasher hasher;
    hasher(game.initPosition.data(), game.initPosition.size());
    hasher << game.boardsize;
    hasher << game.rule;
    hasher << game.result;
    for (const auto &m : game.moveSequence) {
        hasher << m.move;
        hasher << m.eval;
        hashPayload(hasher, m.payload, policyConfig.useMultiPV());
    }
    if (policyConfig.useMultiPV())
        hashPolicyConfig(hasher, policyConfig);
    return hasher;
}

}  // namespace

namespace Tuning {

class NumpyDataWriter::DataBuffer
{
public:
    explicit DataBuffer(PolicyTargetConfig policyConfig) : policyConfig(policyConfig) {}

    /// One buffered move of a game (an entry-to-be at flush time).
    struct BufferedMove
    {
        Pos                                 move;
        Eval                                eval;
        MovePayload                         payload;
        std::optional<std::array<float, 3>> softValueTarget;
    };
    /// One buffered game. Feature extraction walks its moves with a single
    /// incrementally-updated board, so consecutive entries of the same game
    /// cost one board.move() each instead of a full position replay.
    struct BufferedGame
    {
        std::vector<Pos>          initPosition;
        std::vector<BufferedMove> moves;
        uint8_t                   boardsize;
        Rule                      rule;
        Result                    result;  // white pov (GameEntry convention)
    };

    size_t bufferedSize() const { return numBufferedEntries; }

    void addEntry(const DataEntry &entry, std::optional<std::array<float, 3>> softValueTarget)
    {
        // A standalone entry buffers as a one-move game
        BufferedGame &game = gameBuffer.emplace_back();
        game.initPosition  = entry.position;
        game.boardsize     = entry.boardsize;
        game.rule          = entry.rule;
        game.result = entry.sideToMove() == WHITE ? entry.result : flipResult(entry.result);
        game.moves.push_back({entry.move, entry.eval, entry.payload, softValueTarget});
        numBufferedEntries++;

        hash ^= entryHash(entry, policyConfig);
    }

    void addGame(const GameEntry &gameEntry)
    {
        BufferedGame &game = gameBuffer.emplace_back();
        game.initPosition  = gameEntry.initPosition;
        game.boardsize     = gameEntry.boardsize;
        game.rule          = gameEntry.rule;
        game.result        = gameEntry.result;
        game.moves.reserve(gameEntry.moveSequence.size());
        for (const auto &m : gameEntry.moveSequence)
            game.moves.push_back({m.move, m.eval, m.payload, std::nullopt});
        numBufferedEntries += gameEntry.moveSequence.size();

        hash ^= gameHash(gameEntry, policyConfig);
    }

    void asyncSaveToDir(std::string                      dirpath,
                        std::function<void(std::string)> finishedCallback,
                        bool                             writeSparseInputs)
    {
        // Get file name from entry hash
        std::ostringstream ss;
        ss << std::setw(16) << std::setfill('0') << std::hex << hash;
        auto filename = dirpath + "/" + ss.str() + ".npz";

        std::ofstream file(filename, std::ios::binary);
        if (!file.is_open())
            throw std::runtime_error("can not open output file: " + filename);

        // Add processing to async job list
        results.push_back(std::async(
            std::launch::async,
            [os                = std::move(file),
             localGameBuffer   = std::move(gameBuffer),  // clears current game buffer
             numEntries        = numBufferedEntries,
             finishedCallback  = std::move(finishedCallback),
             filename          = filename,
             writeSparseInputs = writeSparseInputs,
             policyConfig      = policyConfig]() mutable {
                try {
                    flushToStream(os,
                                  std::move(localGameBuffer),
                                  numEntries,
                                  writeSparseInputs,
                                  policyConfig);
                }
                catch (...) {
                    // A partially written npz is indistinguishable from valid
                    // output; remove it before reporting the failure
                    os.close();
                    std::remove(filename.c_str());
                    throw;
                }

                if (finishedCallback)
                    finishedCallback(filename);
            }));

        // Cleanup previous finished jobs
        cleanupFinishedResult();
        hash               = 0;
        numBufferedEntries = 0;
        gameBuffer.clear();  // moved-from; make the reset explicit
    }

    /// Joins all pending flush jobs, then rethrows the first failure. Without
    /// an explicit join the future destructors would wait but swallow errors.
    void joinAllResults()
    {
        std::exception_ptr firstError;
        for (auto &result : results) {
            try {
                result.get();
            }
            catch (...) {
                if (!firstError)
                    firstError = std::current_exception();
            }
        }
        results.clear();
        if (firstError)
            std::rethrow_exception(firstError);
    }

private:
    uint64_t                       hash               = 0;
    size_t                         numBufferedEntries = 0;
    std::vector<BufferedGame>      gameBuffer;
    std::vector<std::future<void>> results;
    const PolicyTargetConfig       policyConfig;

    static void flushToStream(std::ostream             &os,
                              std::vector<BufferedGame> localGameBuffer,
                              size_t                    numEntries,
                              bool                      writeSparseInputs,
                              const PolicyTargetConfig &policyConfig)
    {
        // Find max board size
        int maxBoardSize = std::max_element(localGameBuffer.begin(),
                                            localGameBuffer.end(),
                                            [](const BufferedGame &g1, const BufferedGame &g2) {
                                                return g1.boardsize < g2.boardsize;
                                            })
                               ->boardsize;
        unsigned long numCells     = maxBoardSize * maxBoardSize;
        unsigned long numPolicy    = numCells + 1;
        unsigned long numBytes     = (numCells + 7) / 8;
        unsigned long numEntriesUL = (unsigned long)numEntries;

        std::vector<unsigned long> binaryInputNCHWPackedShape {numEntriesUL, 3, numBytes};
        std::vector<unsigned long> sparseInputNCHWU8Shape {writeSparseInputs ? numEntriesUL : 0,
                                                           10,
                                                           numCells};
        std::vector<unsigned long> sparseInputNCHWU16Shape {writeSparseInputs ? numEntriesUL : 0,
                                                            2,
                                                            numCells};
        std::vector<unsigned long> globalInputNCShape {numEntriesUL, 1};
        std::vector<unsigned long> globalTargetsNCShape {numEntriesUL, 3};
        std::vector<unsigned long> policyTargetsNCMoveShape {numEntriesUL, 1, numPolicy};

        std::vector<uint8_t>  binaryInputNCHWPacked(lengthOfShape(binaryInputNCHWPackedShape));
        std::vector<uint8_t>  sparseInputNCHWU8(lengthOfShape(sparseInputNCHWU8Shape));
        std::vector<uint16_t> sparseInputNCHWU16(lengthOfShape(sparseInputNCHWU16Shape));
        std::vector<float>    globalInputNC(lengthOfShape(globalInputNCShape));
        std::vector<float>    globalTargetsNC(lengthOfShape(globalTargetsNCShape));
        std::vector<uint16_t> policyTargetsNCMove(
            lengthOfShape(policyTargetsNCMoveShape));  // Quantitize policy target to int16

        size_t binaryInputNCHWStride     = lengthOfShape(binaryInputNCHWPackedShape, 1);
        size_t sparseInputNCHWU8Stride   = lengthOfShape(sparseInputNCHWU8Shape, 1);
        size_t sparseInputNCHWU16Stride  = lengthOfShape(sparseInputNCHWU16Shape, 1);
        size_t globalInputNCStride       = lengthOfShape(globalInputNCShape, 1);
        size_t globalTargetsNCtride      = lengthOfShape(globalTargetsNCShape, 1);
        size_t policyTargetsNCMoveStride = lengthOfShape(policyTargetsNCMoveShape, 1);

        // Walk each game with one incrementally-updated board: entry i's features are
        // extracted from the board state, then the entry's move is played to reach
        // entry i+1's position (instead of replaying every position from scratch).
        FeatureEncodeScratch scratch;
        size_t               i = 0;
        for (const BufferedGame &g : localGameBuffer) {
            Board board(g.boardsize);
            board.newGame(g.rule);
            for (Pos pos : g.initPosition)
                board.move(g.rule, pos);

            for (const BufferedMove &bm : g.moves) {
                // BufferedGame::result is white pov; the entry wants its side to move
                Result result =
                    board.sideToMove() == WHITE ? g.result : flipResult(g.result);

                EntryFeatureDst dst {
                    &binaryInputNCHWPacked[i * binaryInputNCHWStride],
                    writeSparseInputs ? &sparseInputNCHWU8[i * sparseInputNCHWU8Stride] : nullptr,
                    writeSparseInputs ? &sparseInputNCHWU16[i * sparseInputNCHWU16Stride]
                                      : nullptr,
                    &globalInputNC[i * globalInputNCStride],
                    &globalTargetsNC[i * globalTargetsNCtride],
                    &policyTargetsNCMove[i * policyTargetsNCMoveStride],
                };
                encodeEntryFeatures(board,
                                    bm.move,
                                    bm.eval,
                                    bm.payload,
                                    policyConfig,
                                    result,
                                    bm.softValueTarget,
                                    numCells,
                                    dst,
                                    scratch);

                // Advance the board to the next entry's position
                i++;
                board.move(g.rule, bm.move);
            }
        }
        assert(i == numEntries);

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
        if (writeSparseInputs) {
            const std::vector<uint32_t> &sparseInputDim = sparseInputDims();
            std::vector<unsigned long> sparseInputDimShape {(unsigned long)sparseInputDim.size()};

            openEntryAndWrite("sparseInputDim", sparseInputDim, sparseInputDimShape);
            openEntryAndWrite("sparseInputNCHWU8", sparseInputNCHWU8, sparseInputNCHWU8Shape);
            openEntryAndWrite("sparseInputNCHWU16", sparseInputNCHWU16, sparseInputNCHWU16Shape);
        }
        openEntryAndWrite("globalInputNC", globalInputNC, globalInputNCShape);
        openEntryAndWrite("globalTargetsNC", globalTargetsNC, globalTargetsNCShape);
        openEntryAndWrite("policyTargetsNCMove", policyTargetsNCMove, policyTargetsNCMoveShape);
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
                                 std::function<void(std::string)> flushCallback,
                                 bool                             writeSparseInputs)
    : NumpyDataWriter(std::move(dirpath),
                      maxNumEntriesPerFile,
                      {},
                      std::move(flushCallback),
                      writeSparseInputs)
{}

NumpyDataWriter::NumpyDataWriter(std::string                      dirpath,
                                 size_t                           maxNumEntriesPerFile,
                                 PolicyTargetConfig               policyConfig,
                                 std::function<void(std::string)> flushCallback,
                                 bool                             writeSparseInputs)
    : buffer(std::make_unique<DataBuffer>(policyConfig))
    , dirpath(dirpath)
    , maxNumEntriesPerFile(maxNumEntriesPerFile)
    , flushCallback(flushCallback)
    , writeSparseInputs(writeSparseInputs)
{
    if (!std::isfinite(policyConfig.multiPVTemperature) || policyConfig.multiPVTemperature < 0.0f)
        throw std::invalid_argument("multi-pv policy temperature must be finite and nonnegative");
    if (policyConfig.useMultiPV()
        && (!std::isfinite(policyConfig.evalScalingFactor)
            || policyConfig.evalScalingFactor <= 0.0f))
        throw std::invalid_argument(
            "multi-pv evaluation scaling factor must be finite and positive");

    // Create output directory
    ensureDir(dirpath);
}

NumpyDataWriter::~NumpyDataWriter()
{
    // Flush the remaining buffer and join every outstanding save job before
    // the buffer is destroyed: future destructors would only wait, discarding
    // any error from the final (often largest) batch. A destructor cannot
    // propagate, so report the failure here; the failed job has already
    // removed its partial output file. The two steps get separate try blocks
    // so earlier jobs are still joined (and their errors reported) even when
    // the final batch fails to schedule.
    try {
        if (buffer->bufferedSize())
            buffer->asyncSaveToDir(dirpath, flushCallback, writeSparseInputs);
    }
    catch (const std::exception &e) {
        ERRORL("Failed to save numpy data to " << dirpath << ": " << e.what());
    }
    catch (...) {
        ERRORL("Failed to save numpy data to " << dirpath);
    }
    try {
        buffer->joinAllResults();
    }
    catch (const std::exception &e) {
        ERRORL("Failed to save numpy data to " << dirpath << ": " << e.what());
    }
    catch (...) {
        ERRORL("Failed to save numpy data to " << dirpath);
    }
}

void NumpyDataWriter::flush()
{
    if (buffer->bufferedSize())
        buffer->asyncSaveToDir(dirpath, flushCallback, writeSparseInputs);
    buffer->joinAllResults();
}

void NumpyDataWriter::writeEntry(const DataEntry &entry)
{
    buffer->addEntry(entry, std::nullopt);
    if (buffer->bufferedSize() >= maxNumEntriesPerFile)
        buffer->asyncSaveToDir(dirpath, flushCallback, writeSparseInputs);
}

void NumpyDataWriter::writeGame(const GameEntry &gameEntry)
{
    if (gameEntry.moveSequence.empty())
        return;
    buffer->addGame(gameEntry);
    if (buffer->bufferedSize() >= maxNumEntriesPerFile)
        buffer->asyncSaveToDir(dirpath, flushCallback, writeSparseInputs);
}

void NumpyDataWriter::writeEntryWithSoftValueTarget(const DataEntry &entry,
                                                    float            winprob,
                                                    float            loseprob,
                                                    float            drawprob)
{
    buffer->addEntry(entry, std::array<float, 3> {winprob, loseprob, drawprob});
    if (buffer->bufferedSize() >= maxNumEntriesPerFile)
        buffer->asyncSaveToDir(dirpath, flushCallback, writeSparseInputs);
}

}  // namespace Tuning
