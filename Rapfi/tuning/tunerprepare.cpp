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

#include "../config.h"
#include "../core/iohelper.h"
#include "../core/random.h"
#include "../eval/eval.h"
#include "../eval/scoretables.h"
#include "../game/board.h"
#include "dataset.h"
#include "tunedigest.h"
#include "tuner.h"
#include "tunerdetail.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <deque>
#include <future>
#include <limits>
#include <memory>
#include <numeric>
#include <stdexcept>
#include <unordered_map>

namespace {

using Tuning::Float;
using Tuning::Sha256;
using Tuning::TuneCoeff;
using Tuning::TuneParam;
using Tuning::detail::checkedProduct;
using Tuning::detail::CoeffScale;
using Tuning::detail::domainSeed;
using Tuning::detail::hashFloat;
using Tuning::detail::hashString;
using Tuning::detail::hashUint64;
using Tuning::detail::MiB;
using Tuning::detail::PreparedSampleCredit;

inline bool checkEqual(Float a, Float b)
{
    return std::abs(a - b) <= Float(1.0);
}

template <typename Collector>
void collectEvalCoeffs(Rule r, const Evaluation::EvalInfo &evalInfo, Collector collect)
{
    Color self = evalInfo.self, opponent = ~self;
    for (size_t pcode = 0; pcode < PCODE_NB; pcode++) {
        int coeff[2][SIDE_NB] = {{evalInfo.plyBack[0].pcodeCount[BLACK][pcode],
                                  evalInfo.plyBack[0].pcodeCount[WHITE][pcode]},
                                 {evalInfo.plyBack[1].pcodeCount[BLACK][pcode],
                                  evalInfo.plyBack[1].pcodeCount[WHITE][pcode]}};
        if (r == RENJU) {
            collect(coeff[0][self] + coeff[1][self], 2, &Evaluation::EVALS[r + self][pcode]);
            collect(-coeff[0][opponent] - coeff[1][opponent],
                    2,
                    &Evaluation::EVALS[r + opponent][pcode]);
        }
        else
            collect(coeff[0][self] - coeff[0][opponent] + coeff[1][self]
                        - coeff[1][opponent],
                    2,
                    &Evaluation::EVALS[r][pcode]);
    }
    collect(1, 1, &Evaluation::EVALS_THREAT[Evaluation::tableIndex(r, self)][evalInfo.threatMask]);
}

template <typename Collector>
void collectMoveScoreCoeffs(Rule r,
                            const Board &board,
                            Collector    collect,
                            bool         blackWhitePerspective = false)
{
    Color self = board.sideToMove(), opponent = ~self;
    FOR_EVERY_EMPTY_CAND_POS(&board, pos)
    {
        const auto [pcodeBlack, pcodeWhite] = board.pcodePair(pos);
        PatternCode pcodeSelf = blackWhitePerspective ? pcodeBlack
                                : self == BLACK       ? pcodeBlack
                                                      : pcodeWhite;
        PatternCode pcodeOpponent = blackWhitePerspective ? pcodeWhite
                                    : self == BLACK       ? pcodeWhite
                                                          : pcodeBlack;
        collect(pos,
                1,
                1,
                &Evaluation::P4SCORES[Evaluation::tableIndex(r, self)][pcodeSelf],
                &Evaluation::P4SCORES[Evaluation::tableIndex(r, opponent)][pcodeOpponent]);
    }
}

}  // namespace

namespace Tuning {
PreparedCacheKey Tuner::makePreparedCacheKey(const std::vector<std::filesystem::path> &sourcePaths,
                                             const std::string &datasetFormat,
                                             const char        *role) const
{
    if (sourcePaths.empty() || datasetFormat.empty())
        throw std::invalid_argument(
            "prepared caching requires dataset paths and a dataset format");

    Sha256 hasher;
    hashString(hasher, "rapfi-classical-prepared-corpus-v23");
    hashString(hasher, role);
    hashString(hasher, datasetFormat);
    hashUint64(hasher, config.maxTuneEntries);
    hashUint64(hasher, config.batchSize);
    hashUint64(hasher, config.boardSizeMin);
    hashUint64(hasher, config.boardSizeMax);
    hashUint64(hasher, config.minPly);
    hashUint64(hasher, config.minPlyBeforeFull);
    hashUint64(hasher, config.tuneEval);
    hashUint64(hasher, config.tuneMoveScore);
    hashUint64(hasher, static_cast<uint64_t>(config.trainingSemantics));
    hashUint64(hasher, static_cast<uint64_t>(Config::GeneralCfg.defaultCandidateRange));
    for (bool tuneRule : config.tuneRule)
        hashUint64(hasher, tuneRule);
    hashFloat(hasher, policyTargetConfig.multiPVTemperature);
    hashFloat(hasher, policyTargetConfig.evalScalingFactor);
    hashUint64(hasher, static_cast<uint64_t>(CoeffScale));

    hashUint64(hasher, syncRecords.size());
    for (const ParamsSyncRecord &record : syncRecords) {
        hashString(hasher, record.layoutTag);
        hashUint64(hasher, record.baseIndex);
        hashUint64(hasher, record.numElems);
        hashUint64(hasher, record.elemSize);
        hashUint64(hasher, record.paramPerElem);
    }
    hashUint64(hasher, tiedMoveScoreSyncRecords.size());
    for (const TiedMoveScoreSyncRecord &record : tiedMoveScoreSyncRecords) {
        hashString(hasher, record.layoutTag);
        hashUint64(hasher, record.parameterIndices.size());
        for (const auto &indices : record.parameterIndices) {
            hashUint64(hasher, indices[0]);
            hashUint64(hasher, indices[1]);
        }
    }
    hashUint64(hasher, tuneParams.size());
    for (TuneParam param : tuneParams)
        hashFloat(hasher, param);
    if (config.tuneMoveScore)
        hashString(hasher,
                   config.trainingSemantics == TrainingSemantics::Bootstrap
                       ? "bootstrap-untied-two-term-v1"
                       : "tied-global-two-term-v1");

    PreparedCacheKey key;
    key.sourcePaths.reserve(sourcePaths.size());
    for (const std::filesystem::path &sourcePath : sourcePaths) {
        std::error_code       error;
        std::filesystem::path configured =
            std::filesystem::absolute(sourcePath, error).lexically_normal();
        if (error)
            throw std::runtime_error("unable to resolve tuning dataset source: "
                                     + sourcePath.string());
        std::filesystem::path canonical = std::filesystem::weakly_canonical(configured, error);
        if (error)
            throw std::runtime_error("unable to canonicalize tuning dataset source: "
                                     + configured.string());
        std::string configuredPath = configured.generic_u8string();
        std::string canonicalPath  = canonical.generic_u8string();

        hashString(hasher, canonicalPath);
        key.sourcePaths.push_back({std::move(configuredPath), std::move(canonicalPath)});
    }
    key.fingerprint = sha256Hex(hasher.finish());
    return key;
}

void Tuner::appendTuneSample(PreparedCorpus    &tuneEntries,
                             const Board       &board,
                             Rule               rule,
                             uint8_t            resultTimesTwo,
                             Pos                bestMove,
                             Eval               bestEval,
                             const MovePayload &payload,
                             CompileScratch    &scratch) const
{
    Value staticEval = Evaluation::evaluate(board, rule);
    if (staticEval < INT16_MIN || staticEval > INT16_MAX)
        throw std::overflow_error("static evaluation exceeds int16 storage");

    scratch.evalTerms.clear();
    if (config.tuneEval) {
        Evaluation::EvalInfo evalInfo(board, rule);
        collectEvalCoeffs(rule, evalInfo, [this, &scratch](int coeff, int coeffScale, void *addr) {
            if (coeff == 0)
                return;
            int scaledCoeff = int(coeff * CoeffScale) / coeffScale;
            if (scaledCoeff < INT16_MIN || scaledCoeff > INT16_MAX)
                throw std::overflow_error("value coefficient exceeds int16 storage");
            scratch.evalTerms.push_back({static_cast<int16_t>(scaledCoeff), paramIndex(addr)});
        });
    }

    scratch.policyCandidates.clear();
    scratch.policyTargets.clear();
    uint16_t bestCandidate = PreparedCorpus::NoPolicyTarget;
    bool     trainsPolicy  = config.tuneMoveScore && bestMove != Pos {board.size(), board.size()}
                        && bestMove != Pos::NONE && bestMove != Pos::PASS
                        && board.isEmptyCandidate(bestMove);
    if (!config.tuneEval && !trainsPolicy)
        return;
    if (trainsPolicy) {
        bool useSoftTarget = policyTargetConfig.useMultiPV();
        if (useSoftTarget) {
            size_t numCells = size_t(board.size()) * size_t(board.size());
            scratch.policyTarget.resize(numCells + 1);
            encodePolicyTarget(bestMove,
                               bestEval,
                               payload,
                               board.size(),
                               numCells,
                               policyTargetConfig,
                               scratch.policyTarget.data(),
                               scratch.featureEncoder);
        }
        auto appendPolicyCandidate = [this,
                                      bestMove,
                                      boardSize = board.size(),
                                      useSoftTarget,
                                      &scratch,
                                      &bestCandidate](Pos pos, void *addrSelf, void *addrOppo) {
            if (pos == bestMove)
                bestCandidate = static_cast<uint16_t>(scratch.policyCandidates.size());
            uint16_t        candidateIndex = static_cast<uint16_t>(scratch.policyCandidates.size());
            PolicyCandidate candidate {};
            candidate.indices[0] = paramIndex(addrSelf, 0);
            candidate.indices[1] = paramIndex(addrOppo, 1);

            scratch.policyCandidates.push_back(candidate);
            if (useSoftTarget) {
                uint16_t target = scratch.policyTarget[pos.y() * boardSize + pos.x()];
                if (target != 0)
                    scratch.policyTargets.push_back({candidateIndex, target});
            }
        };

        collectMoveScoreCoeffs(
            rule,
            board,
            [&](Pos pos, int coeffSelf, int coeffOppo, void *addrSelf, void *addrOppo) {
                if (rule == RENJU && board.sideToMove() == BLACK
                    && board.pattern4(pos, BLACK) == FORBID && board.checkForbiddenPoint(pos))
                    return;
                if (coeffSelf != 1 || coeffOppo != 1)
                    throw std::logic_error("compact policy storage requires unit coefficients");
                appendPolicyCandidate(pos, addrSelf, addrOppo);
            },
            config.trainingSemantics == TrainingSemantics::Bootstrap);
        if (bestCandidate == PreparedCorpus::NoPolicyTarget)
            throw std::logic_error("best move is missing from policy candidates");
        if (scratch.policyTargets.size() == 1
            && scratch.policyTargets.front().candidate == bestCandidate)
            scratch.policyTargets.clear();
    }

    if (config.tuneEval) {
        Float linearEval = 0;
        for (const TuneCoeff &term : scratch.evalTerms)
            linearEval += term.coeff * tuneParams[term.index];
        linearEval /= CoeffScale;
        if (!checkEqual(Float(staticEval), linearEval))
            throw std::logic_error("prepared linear evaluation differs from engine evaluation");
    }

    tuneEntries.append(static_cast<uint8_t>(board.size()),
                       resultTimesTwo,
                       static_cast<int16_t>(staticEval),
                       scratch.evalTerms,
                       scratch.policyCandidates,
                       scratch.policyTargets,
                       bestCandidate);
}

/// initTuneEntries() inits tuneEntries from dataEntry read from datasets.
/// DataEntry that does not satisfy a certain condition will be skipped.
void Tuner::initTuneEntries(PreparedCorpus   &tuneEntries,
                            FileBackedCorpus *fileCorpus,
                            class Dataset    &dataset,
                            bool              buildShuffleOrder)
{
    tuneEntries.clear();

    struct PreparedJob
    {
        std::future<PreparedCorpus> future;
        size_t                      creditBytes;
    };
    std::deque<PreparedJob> jobs;
    size_t                  pendingCreditBytes = 0;
    const size_t            workerCount        = std::max<size_t>(threadPool.get_thread_count(), 1);
    size_t                  maxPendingJobs     = workerCount;
    size_t                  chunkEntryLimit    = config.batchSize;

    if (fileCorpus) {
        maxPendingJobs  = fileMaxPendingJobs;
        chunkEntryLimit = fileChunkEntryLimit;
        dataset.setMaxRecordBytes(fileRecordLimitBytes);
        dataset.setRetainExtraPVs(policyTargetConfig.useMultiPV());
        MESSAGEL("File-backed preparation chunk = "
                 << chunkEntryLimit << " raw entries, pending jobs = " << maxPendingJobs
                 << ", record limit = " << fileRecordLimitBytes / double(MiB) << " MiB.");
    }

    PreparedCorpus openShard;
    PreparedCorpus batchCorpus;

    auto sealOpenShard = [&]() {
        if (!fileCorpus || openShard.empty())
            return;
        if (openShard.capacityBytes() > fileShardBudgetBytes)
            throw std::logic_error("prepared shard escaped its allocation credit");
        fileCorpus->append(std::move(openShard));
        openShard = PreparedCorpus {};
    };

    auto appendBatchToOpenShard = [&]() {
        if (!fileCorpus || batchCorpus.empty())
            return;

        auto copyFitsCredit = [&]() {
            size_t peakBytes =
                openShard.appendRangePeakCapacityBytes(batchCorpus, 0, batchCorpus.size());
            return batchCorpus.capacityBytes() <= fileShardBudgetBytes
                   && peakBytes <= fileShardBudgetBytes - batchCorpus.capacityBytes();
        };

        if (!copyFitsCredit() && !openShard.empty())
            sealOpenShard();

        if (copyFitsCredit()) {
            openShard.reserveAppendRange(batchCorpus, 0, batchCorpus.size());
            openShard.appendRange(batchCorpus, 0, batchCorpus.size());
            batchCorpus = PreparedCorpus {};
            if (openShard.capacityBytes() >= fileShardTargetBytes)
                sealOpenShard();
            return;
        }

        if (!openShard.empty())
            throw std::logic_error("prepared shard could not be sealed before direct batch write");
        if (batchCorpus.capacityBytes() > fileShardBudgetBytes)
            throw std::runtime_error(
                "one prepared gradient batch exceeds its shard allocation credit; "
                "increase --memory-limit-mb or reduce --batchsize");
        fileCorpus->append(std::move(batchCorpus));
        batchCorpus = PreparedCorpus {};
    };

    auto collectFrontJob = [&]() {
        size_t         creditBytes = jobs.front().creditBytes;
        PreparedCorpus fragment    = jobs.front().future.get();
        jobs.pop_front();
        if (!fileCorpus) {
            tuneEntries.append(std::move(fragment));
            pendingCreditBytes -= creditBytes;
            return;
        }

        for (size_t begin = 0; begin < fragment.size();) {
            size_t batchSpace = config.batchSize - batchCorpus.size();
            size_t count      = std::min(batchSpace, fragment.size() - begin);

            while (true) {
                size_t peakBytes = batchCorpus.appendRangePeakCapacityBytes(fragment, begin, count);
                if (openShard.capacityBytes() <= fileShardBudgetBytes
                    && peakBytes <= fileShardBudgetBytes - openShard.capacityBytes()) {
                    batchCorpus.reserveAppendRange(fragment, begin, count);
                    batchCorpus.appendRange(fragment, begin, count);
                    break;
                }
                if (!openShard.empty()) {
                    sealOpenShard();
                    continue;
                }
                throw std::runtime_error(
                    "one prepared gradient batch exceeds its shard allocation credit; "
                    "increase --memory-limit-mb or reduce --batchsize");
            }
            begin += count;
            if (batchCorpus.size() == config.batchSize)
                appendBatchToOpenShard();
        }
        pendingCreditBytes -= creditBytes;
    };

    auto ensureWorkerCredit = [&](size_t creditBytes) {
        if (!fileCorpus)
            return;
        if (creditBytes > fileJobBudgetBytes)
            throw std::runtime_error(
                "one dataset job exceeds the worker allocation credit; "
                "increase --memory-limit-mb or reduce the record or batch size");
        while (!jobs.empty()
               && (jobs.size() >= maxPendingJobs
                   || pendingCreditBytes > fileJobBudgetBytes - creditBytes))
            collectFrontJob();
        if (pendingCreditBytes > fileJobBudgetBytes - creditBytes)
            throw std::logic_error("worker allocation credit accounting failed");
    };

    // Read dataset and convert bounded batches to compact corpus fragments.
    size_t totalEntriesRead = 0;
    if (dataset.supportsGames()) {
        using GameWork   = std::pair<GameEntry, size_t>;
        auto submitGames = [&](std::vector<GameWork> &&games,
                               size_t                  chunkEntries,
                               size_t                  creditBytes) {
            auto sharedGames = std::make_shared<std::vector<GameWork>>(std::move(games));
            jobs.push_back(PreparedJob {
                threadPool.submit_task(
                    [this, games = std::move(sharedGames), chunkEntries]() -> PreparedCorpus {
                        PreparedCorpus entries;
                        CompileScratch scratch;
                        entries.reserveSamples(chunkEntries);

                        for (const auto &work : *games) {
                            const GameEntry &game      = work.first;
                            size_t           moveLimit = work.second;
                            Board            board(game.boardsize);
                            board.newGame(game.rule);
                            for (Pos pos : game.initPosition)
                                board.move(game.rule, pos);

                            for (size_t moveIndex = 0; moveIndex < moveLimit; moveIndex++) {
                                size_t ply = game.initPosition.size() + moveIndex;
                                if (config.tuneRule[game.rule]
                                    && game.boardsize >= config.boardSizeMin
                                    && game.boardsize <= config.boardSizeMax && ply >= config.minPly
                                    && ply + config.minPlyBeforeFull
                                           <= int(game.boardsize) * int(game.boardsize)) {
                                    Result  result         = board.sideToMove() == WHITE
                                                                 ? game.result
                                                                 : flipResult(game.result);
                                    uint8_t resultTimesTwo = result == RESULT_WIN    ? 2
                                                             : result == RESULT_DRAW ? 1
                                                                                     : 0;
                                    appendTuneSample(entries,
                                                     board,
                                                     game.rule,
                                                     resultTimesTwo,
                                                     game.moveSequence[moveIndex].move,
                                                     game.moveSequence[moveIndex].eval,
                                                     game.moveSequence[moveIndex].payload,
                                                     scratch);
                                }
                                board.move(game.rule, game.moveSequence[moveIndex].move);
                            }
                        }
                        return entries;
                    }),
                creditBytes});
            pendingCreditBytes += creditBytes;
        };

        bool reachedEnd = false;
        while (totalEntriesRead < config.maxTuneEntries && !reachedEnd) {
            std::vector<GameWork> games;
            size_t                chunkEntries = 0;
            bool                  submitted    = false;

            do {
                if (fileCorpus)
                    ensureWorkerCredit(fileRecordLimitBytes);
                GameEntry game;
                if (!dataset.nextGame(&game)) {
                    reachedEnd = true;
                    break;
                }

                size_t gameMoveCount = game.moveSequence.size();
                size_t remaining     = config.maxTuneEntries - totalEntriesRead - chunkEntries;
                size_t moveLimit     = std::min(gameMoveCount, remaining);
                if (moveLimit != 0) {
                    size_t creditBytes = 0;
                    if (fileCorpus) {
                        size_t preparedBytes =
                            checkedProduct(moveLimit, PreparedSampleCredit, "prepared job");
                        if (preparedBytes > fileJobBudgetBytes - fileRecordLimitBytes)
                            throw std::runtime_error(
                                "one game cannot be prepared within the worker allocation credit");
                        creditBytes = fileRecordLimitBytes + preparedBytes;
                        ensureWorkerCredit(creditBytes);
                    }
                    chunkEntries += moveLimit;
                    games.emplace_back(std::move(game), moveLimit);
                    if (fileCorpus) {
                        totalEntriesRead += chunkEntries;
                        submitGames(std::move(games), chunkEntries, creditBytes);
                        chunkEntries = 0;
                        submitted    = true;
                        break;
                    }
                }
                if (moveLimit < gameMoveCount)
                    break;
            } while (chunkEntries < chunkEntryLimit
                     && totalEntriesRead + chunkEntries < config.maxTuneEntries);

            if (submitted || games.empty())
                continue;

            totalEntriesRead += chunkEntries;
            submitGames(std::move(games), chunkEntries, 0);
            if (jobs.size() >= maxPendingJobs)
                collectFrontJob();
        }
    }
    else {
        while (totalEntriesRead < config.maxTuneEntries) {
            size_t entriesToRead =
                std::min(chunkEntryLimit, config.maxTuneEntries - totalEntriesRead);
            size_t creditBytes =
                fileCorpus
                    ? checkedProduct(entriesToRead, PreparedSampleCredit, "prepared dataset chunk")
                    : 0;
            ensureWorkerCredit(creditBytes);
            std::vector<DataEntry> dataEntries;
            dataEntries.reserve(entriesToRead);
            for (size_t i = 0; i < entriesToRead; i++) {
                DataEntry dataEntry;
                if (!dataset.next(&dataEntry))
                    break;
                dataEntries.push_back(std::move(dataEntry));
            }

            if (dataEntries.empty())
                break;
            totalEntriesRead += dataEntries.size();

            jobs.push_back(PreparedJob {
                threadPool.submit_task([this, data = std::move(dataEntries)]() -> PreparedCorpus {
                    std::unordered_map<int, Board> boardObjectCache;
                    PreparedCorpus                 entries;
                    CompileScratch                 scratch;
                    entries.reserveSamples(data.size());

                    for (const DataEntry &dataEntry : data) {
                        if (!config.tuneRule[dataEntry.rule]
                            || dataEntry.boardsize < config.boardSizeMin
                            || dataEntry.boardsize > config.boardSizeMax
                            || dataEntry.position.size() < config.minPly
                            || dataEntry.position.size() + config.minPlyBeforeFull
                                   > int(dataEntry.boardsize) * int(dataEntry.boardsize))
                            continue;

                        auto boardIt = boardObjectCache.find(dataEntry.boardsize);
                        if (boardIt == boardObjectCache.end()) {
                            boardIt = boardObjectCache
                                          .emplace(std::piecewise_construct,
                                                   std::forward_as_tuple(dataEntry.boardsize),
                                                   std::forward_as_tuple(dataEntry.boardsize))
                                          .first;
                        }

                        Board &board = boardIt->second;
                        board.newGame(dataEntry.rule);
                        for (Pos pos : dataEntry.position)
                            board.move(dataEntry.rule, pos);

                        uint8_t resultTimesTwo = dataEntry.result == RESULT_WIN    ? 2
                                                 : dataEntry.result == RESULT_DRAW ? 1
                                                                                   : 0;
                        appendTuneSample(entries,
                                         board,
                                         dataEntry.rule,
                                         resultTimesTwo,
                                         dataEntry.move,
                                         dataEntry.eval,
                                         dataEntry.payload,
                                         scratch);
                    }

                    return entries;
                }),
                creditBytes});
            pendingCreditBytes += creditBytes;

            if (jobs.size() >= maxPendingJobs)
                collectFrontJob();
        }
    }

    MESSAGEL("Read " << totalEntriesRead << " tune entries from dataset, initializing...");

    while (!jobs.empty())
        collectFrontJob();

    if (fileCorpus) {
        appendBatchToOpenShard();
        sealOpenShard();
        MESSAGEL(fileCorpus->size()
                 << " tune entries initialized in " << fileCorpus->shardCount()
                 << " file-backed shards (" << fileCorpus->diskBytes() / (1024.0 * 1024.0)
                 << " MiB) at " << fileCorpus->directory().string() << '.');
    }
    else {
        MESSAGEL(tuneEntries.size() << " tune entries initialized in "
                                    << tuneEntries.capacityBytes() / (1024.0 * 1024.0) << " MiB.");
    }

    if (buildShuffleOrder && config.shuffleTuneEntries) {
        MESSAGEL("Creating logical shuffle order...");

        if (tuneEntries.size() > std::numeric_limits<uint32_t>::max())
            throw std::length_error("shuffle order exceeds 32-bit sample indices");
        trainSampleOrder.resize(tuneEntries.size());
        std::iota(trainSampleOrder.begin(), trainSampleOrder.end(), uint32_t(0));
        PRNG prng(domainSeed(config.seed, 0x73687566666c6500ULL));
        std::shuffle(trainSampleOrder.begin(), trainSampleOrder.end(), prng);
    }
}

}  // namespace Tuning
