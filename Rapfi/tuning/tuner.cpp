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

#include "tuner.h"

#include "../config.h"
#include "../core/iohelper.h"
#include "../core/time.h"
#include "../eval/eval.h"
#include "../eval/evaluator.h"
#include "../game/board.h"
#include "../game/pattern.h"
#include "dataset.h"
#include "optimizer.h"
#include "tunedigest.h"
#include "tunerdetail.h"
#include "tunerexport.h"
#include "tunerobjective.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstring>
#include <deque>
#include <future>
#include <iomanip>
#include <limits>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <thread>

#ifdef _WIN32
    #ifndef NOMINMAX
        #define NOMINMAX
    #endif
    #define WIN32_LEAN_AND_MEAN
    #include <windows.h>
#endif

namespace {

using Tuning::Float;
using Tuning::LossType;
using Tuning::PolicyCandidate;
using Tuning::PolicyTargetTerm;
using Tuning::PreparedCacheKey;
using Tuning::PreparedCorpus;
using Tuning::TuneCoeff;
using Tuning::TuneGradient;
using Tuning::TuneParam;
using Tuning::detail::checkedProduct;
using Tuning::detail::CoeffScale;
using Tuning::detail::encodeIntegerForTruncatingExport;
using Tuning::detail::MiB;
using Tuning::detail::PreparedSampleCredit;

constexpr size_t KiB = 1024;

TuneParam scheduledLearningRate(double                       initialLearningRate,
                                double                       finalLearningRate,
                                Tuning::LearningRateSchedule schedule,
                                size_t                       epoch,
                                size_t                       epochs)
{
    if (schedule == Tuning::LearningRateSchedule::Constant || epoch == 0)
        return TuneParam(initialLearningRate);

    const double progress = epochs > 1 ? double(epoch - 1) / double(epochs - 1) : 0.0;
    switch (schedule) {
    case Tuning::LearningRateSchedule::Exponential:
        return TuneParam(initialLearningRate
                         * std::pow(finalLearningRate / initialLearningRate, progress));
    case Tuning::LearningRateSchedule::Constant: return TuneParam(initialLearningRate);
    }
    throw std::logic_error("unknown learning-rate schedule");
}

TuneParam learningRateForEpoch(const Tuning::TuningConfig &config, size_t epoch, size_t epochs)
{
    return scheduledLearningRate(config.learningRate,
                                 config.finalLearningRate,
                                 config.learningRateSchedule,
                                 epoch,
                                 epochs);
}

struct LossPair
{
    Float value  = 0;
    Float policy = 0;

    LossPair &operator+=(const LossPair &other)
    {
        value += other.value;
        policy += other.policy;
        return *this;
    }
};

struct BoardLossTotals
{
    std::array<LossPair, MAX_BOARD_SIZE + 1> losses;
    std::array<size_t, MAX_BOARD_SIZE + 1>   samples = {};

    BoardLossTotals &operator+=(const BoardLossTotals &other)
    {
        for (size_t boardSize = 0; boardSize < samples.size(); boardSize++) {
            losses[boardSize] += other.losses[boardSize];
            samples[boardSize] += other.samples[boardSize];
        }
        return *this;
    }
};

bool pathComponentEqual(const std::filesystem::path &lhs, const std::filesystem::path &rhs)
{
#ifdef _WIN32
    return CompareStringOrdinal(lhs.c_str(), -1, rhs.c_str(), -1, TRUE) == CSTR_EQUAL;
#else
    return lhs == rhs;
#endif
}

bool isWithinDirectory(const std::filesystem::path &directory,
                       const std::filesystem::path &candidate)
{
    std::filesystem::path normalizedDirectory = directory.lexically_normal();
    std::filesystem::path normalizedCandidate = candidate.lexically_normal();
    auto                  directoryComponent  = normalizedDirectory.begin();
    auto                  candidateComponent  = normalizedCandidate.begin();
    for (; directoryComponent != normalizedDirectory.end();
         ++directoryComponent, ++candidateComponent) {
        if (candidateComponent == normalizedCandidate.end()
            || !pathComponentEqual(*directoryComponent, *candidateComponent))
            return false;
    }
    return true;
}

void validatePreparedCacheRoot(const std::filesystem::path &root,
                               const PreparedCacheKey      &trainKey,
                               const PreparedCacheKey      *validationKey)
{
    std::error_code       error;
    std::filesystem::path absoluteRoot = std::filesystem::absolute(root, error).lexically_normal();
    if (error)
        throw std::runtime_error("unable to resolve prepared-cache root: " + root.string());
    std::filesystem::path canonicalRoot = std::filesystem::weakly_canonical(absoluteRoot, error);
    if (error)
        throw std::runtime_error("unable to canonicalize prepared-cache root: "
                                 + absoluteRoot.string());

    auto validateKey = [&](const PreparedCacheKey &key) {
        for (const Tuning::PreparedSourcePath &source : key.sourcePaths) {
            std::filesystem::path configured =
                std::filesystem::u8path(source.configuredPath).lexically_normal();
            std::filesystem::path aliasParent =
                std::filesystem::weakly_canonical(configured.parent_path(), error);
            if (error)
                throw std::runtime_error("unable to canonicalize tuning dataset parent: "
                                         + configured.string());
            std::filesystem::path aliasLocation =
                (aliasParent / configured.filename()).lexically_normal();
            std::filesystem::path canonicalTarget =
                std::filesystem::u8path(source.canonicalPath).lexically_normal();
            if (isWithinDirectory(canonicalRoot, aliasLocation)
                || isWithinDirectory(canonicalRoot, canonicalTarget))
                throw std::invalid_argument(
                    "tuning dataset sources must be outside the prepared-cache root: "
                    + configured.string());
        }
    };

    validateKey(trainKey);
    if (validationKey)
        validateKey(*validationKey);
}

/// sigmoid(x) = 1/(1+exp(-x))
inline Float sigmoid(Float x)
{
    return Float(1) / (Float(1) + std::exp(-x));
}

inline Float lossFunction(LossType lt, Float logit, Float target)
{
    Float pred = sigmoid(logit);
    switch (lt) {
    case LossType::L1: return std::abs(target - pred);
    case LossType::L2: {
        Float dist = target - pred;
        return dist * dist;
    }
    case LossType::BCE: {
        Float loss =
            std::max(logit, Float(0)) - logit * target + std::log1p(std::exp(-std::abs(logit)));
        Float targetBias = 0;
        if (target > 0)
            targetBias += target * std::log(target);
        if (target < 1)
            targetBias += (1 - target) * std::log1p(-target);
        return (loss + targetBias) / 2;
    }
    }
    throw std::logic_error("unknown loss type");
}

inline Float lossFunctionLogitGrad(LossType lt, Float logit, Float target)
{
    Float pred = sigmoid(logit);
    Float diff = pred - target;
    switch (lt) {
    case LossType::L1: return ((Float(0) < diff) - (diff < Float(0))) * pred * (Float(1) - pred);
    case LossType::L2: return Float(2) * diff * pred * (Float(1) - pred);
    case LossType::BCE: return diff / 2;
    }
    throw std::logic_error("unknown loss type");
}

template <typename T, typename UnaryOp>
T parallelIndexReduce(BS::thread_pool &pool, size_t count, T init, UnaryOp transform)
{
    if (count == 0)
        return init;

    constexpr size_t            LogicalPartitions = 64;
    size_t                      numBlocks         = std::min(count, LogicalPartitions);
    size_t                      blockSize         = (count + numBlocks - 1) / numBlocks;
    std::vector<std::future<T>> futures;
    futures.reserve(numBlocks);

    for (size_t i = 0; i < numBlocks; ++i) {
        size_t blockBegin = i * blockSize;
        size_t blockEnd   = std::min(blockBegin + blockSize, count);
        if (blockBegin == blockEnd)
            break;

        futures.emplace_back(pool.submit_task([blockBegin, blockEnd, transform]() {
            T blockResult = T {};
            for (size_t index = blockBegin; index < blockEnd; ++index)
                blockResult += transform(index);
            return blockResult;
        }));
    }

    T result = init;
    for (auto &future : futures)
        result += future.get();

    return result;
}

Float linearValue(const std::vector<TuneCoeff> &terms,
                  uint32_t                      begin,
                  uint32_t                      end,
                  const std::vector<TuneParam> &params)
{
    Float value = 0;
    for (uint32_t i = begin; i < end; i++)
        value += terms[i].coeff * params[terms[i].index];
    return value;
}

Float computeLinearEval(const PreparedCorpus         &corpus,
                        size_t                        sample,
                        const std::vector<TuneParam> &params)
{
    return linearValue(corpus.evalTerms(),
                       corpus.evalOffsets()[sample],
                       corpus.evalOffsets()[sample + 1],
                       params)
           / CoeffScale;
}

template <bool UseTunedEval>
Float computeEvalLoss(const PreparedCorpus         &corpus,
                      size_t                        sample,
                      const std::vector<TuneParam> &params,
                      Float                         K,
                      LossType                      loss)
{
    if (corpus.evalOffsets()[sample] == corpus.evalOffsets()[sample + 1])
        return 0;

    Float eval   = UseTunedEval ? computeLinearEval(corpus, sample, params)
                                : Float(corpus.staticEvals()[sample]);
    Float result = Float(corpus.results()[sample]) * Float(0.5);
    return lossFunction(loss, eval * K, result);
}

void computeEvalGradient(const PreparedCorpus         &corpus,
                         size_t                        sample,
                         std::vector<TuneGradient>    &grads,
                         const std::vector<TuneParam> &params,
                         Float                         K,
                         LossType                      loss,
                         Float                         sampleWeight)
{
    uint32_t begin = corpus.evalOffsets()[sample];
    uint32_t end   = corpus.evalOffsets()[sample + 1];
    if (begin == end)
        return;

    Float       result    = Float(corpus.results()[sample]) * Float(0.5);
    Float       logit     = computeLinearEval(corpus, sample, params) * K;
    Float       dL_dEval  = lossFunctionLogitGrad(loss, logit, result) * K * sampleWeight;
    const auto &evalTerms = corpus.evalTerms();
    for (uint32_t i = begin; i < end; i++)
        grads[evalTerms[i].index] += evalTerms[i].coeff * dL_dEval;
}

Tuning::TuningConfig configForPhase(Tuning::TuningConfig config, const Tuning::TuningPhase &phase)
{
    config.preparedCachePath       = phase.preparedCachePath;
    config.trainDatasetPaths       = phase.trainDatasetPaths;
    config.validationDatasetPaths  = phase.validationDatasetPaths;
    config.trainDatasetFormat      = phase.trainDatasetFormat;
    config.validationDatasetFormat = phase.validationDatasetFormat;
    config.rebuildPreparedCache    = phase.rebuildPreparedCache;
    config.boardSizeMin            = phase.boardSizeMin;
    config.boardSizeMax            = phase.boardSizeMax;
    if (phase.batchSize)
        config.batchSize = *phase.batchSize;
    if (phase.seed)
        config.seed = *phase.seed;
    if (phase.multiPVPolicyTemperature)
        config.multiPVPolicyTemperature = *phase.multiPVPolicyTemperature;
    if (phase.multiPVPolicyEvalScale)
        config.multiPVPolicyEvalScale = *phase.multiPVPolicyEvalScale;
    if (phase.trainingSemantics)
        config.trainingSemantics = *phase.trainingSemantics;
    config.tuneEval             = phase.tuneEval;
    config.tuneMoveScore        = phase.tuneMoveScore;
    config.learningRate         = phase.learningRate;
    config.finalLearningRate    = phase.finalLearningRate;
    config.weightDecay          = phase.weightDecay;
    config.learningRateSchedule = phase.learningRateSchedule;
    config.recomputeInterval    = phase.recomputeInterval;
    config.boardSizeWeighting   = phase.boardSizeWeighting;
    return config;
}

Tuning::PolicyTargetConfig policyTargetConfigFor(const Tuning::TuningConfig &config)
{
    return {float(config.multiPVPolicyTemperature),
            float(config.multiPVPolicyEvalScale > 0 ? config.multiPVPolicyEvalScale
                                                    : Evaluation::ScalingFactor)};
}

const Tuning::TuningPhase &firstPhase(const std::vector<Tuning::TuningPhase> &phases)
{
    if (phases.empty() || !phases.front().trainingDataset)
        throw std::invalid_argument("tuning curriculum requires at least one training phase");
    return phases.front();
}

std::string normalizedPathKey(const std::filesystem::path &path)
{
    std::string key = std::filesystem::absolute(path).lexically_normal().generic_string();
#ifdef _WIN32
    std::transform(key.begin(), key.end(), key.begin(), [](unsigned char c) {
        return static_cast<char>(std::tolower(c));
    });
#endif
    return key;
}

Tuning::TuningConfig validatedCurriculumConfig(const std::vector<Tuning::TuningPhase> &phases,
                                               Tuning::TuningConfig                    config)
{
    firstPhase(phases);
    if (config.memoryLimitMB == 0)
        throw std::invalid_argument("tuning curricula require file-backed prepared corpora");
    const bool hasTraditionalPhase =
        std::any_of(phases.begin(), phases.end(), [](const Tuning::TuningPhase &phase) {
            return phase.tuneEval || phase.tuneMoveScore;
        });
    size_t coalescingBoundaries = 0;
    for (size_t i = 0; i < phases.size(); i++) {
        const Tuning::TuningPhase &phase = phases[i];
        if (phase.name.empty())
            throw std::invalid_argument("tuning curriculum phase name must not be empty");
        if (!phase.trainingDataset || !phase.validationDataset)
            throw std::invalid_argument(
                "tuning curriculum phases require training and validation datasets");
        if (phase.epochs == 0)
            throw std::invalid_argument("tuning curriculum phase epochs must be positive");
        if (phase.boardSizeMin == 0 || phase.boardSizeMax > MAX_BOARD_SIZE
            || phase.boardSizeMin > phase.boardSizeMax)
            throw std::invalid_argument("tuning curriculum phase board range is invalid");
        if (phase.preparedCachePath.empty() || phase.trainDatasetPaths.empty()
            || phase.validationDatasetPaths.empty() || phase.trainDatasetFormat.empty()
            || phase.validationDatasetFormat.empty())
            throw std::invalid_argument(
                "tuning curriculum phases require complete dataset and cache identities");
        Tuning::TuningConfig phaseConfig = configForPhase(config, phase);
        if (phaseConfig.batchSize == 0)
            throw std::invalid_argument("tuning curriculum phase batch size must be positive");
        if (!std::isfinite(phaseConfig.multiPVPolicyTemperature)
            || phaseConfig.multiPVPolicyTemperature < 0)
            throw std::invalid_argument(
                "tuning curriculum phase MultiPV temperature must be finite and nonnegative");
        if (!std::isfinite(phaseConfig.multiPVPolicyEvalScale)
            || phaseConfig.multiPVPolicyEvalScale < 0)
            throw std::invalid_argument(
                "tuning curriculum phase MultiPV eval scale must be finite and nonnegative");
        if (phaseConfig.multiPVPolicyTemperature > std::numeric_limits<float>::max()
            || phaseConfig.multiPVPolicyEvalScale > std::numeric_limits<float>::max())
            throw std::invalid_argument(
                "tuning curriculum phase MultiPV settings must fit float storage");
        if ((phaseConfig.multiPVPolicyTemperature > 0
             && phaseConfig.multiPVPolicyTemperature < std::numeric_limits<float>::denorm_min())
            || (phaseConfig.multiPVPolicyEvalScale > 0
                && phaseConfig.multiPVPolicyEvalScale < std::numeric_limits<float>::denorm_min()))
            throw std::invalid_argument(
                "tuning curriculum phase positive MultiPV settings must fit float storage");
        if (phaseConfig.multiPVPolicyEvalScale > 0 && phaseConfig.multiPVPolicyTemperature == 0)
            throw std::invalid_argument(
                "tuning curriculum phase MultiPV eval scale requires a positive temperature");
        if (!(phase.tuneEval || phase.tuneMoveScore || phase.trainCompactPolicy))
            throw std::invalid_argument(
                "tuning curriculum phases require an active objective");
        if (hasTraditionalPhase && phase.trainCompactPolicy && !phase.tuneEval
            && !phase.tuneMoveScore && i + 1 != phases.size())
            throw std::invalid_argument(
                "a pure compact-policy phase must end a mixed tuning curriculum");
        if (!std::isfinite(phaseConfig.learningRate) || phaseConfig.learningRate <= 0)
            throw std::invalid_argument(
                "tuning curriculum phase learning rate must be finite and positive");
        if (!std::isfinite(phaseConfig.weightDecay) || phaseConfig.weightDecay < 0)
            throw std::invalid_argument(
                "tuning curriculum phase weight decay must be finite and nonnegative");
        if (phaseConfig.learningRateSchedule != Tuning::LearningRateSchedule::Constant
            && (!std::isfinite(phaseConfig.finalLearningRate) || phaseConfig.finalLearningRate <= 0
                || phaseConfig.finalLearningRate > phaseConfig.learningRate))
            throw std::invalid_argument("tuning curriculum phase final learning rate is invalid");
        if (phaseConfig.multiPVPolicyTemperature > 0 && !phaseConfig.tuneMoveScore)
            throw std::invalid_argument(
                "tuning curriculum phase MultiPV targets require move-score tuning");
        if (phaseConfig.multiPVPolicyTemperature > 0 && config.moveScoreLossGamma != 0)
            throw std::invalid_argument(
                "tuning curriculum phase MultiPV targets require move-score-loss-gamma=0");
        if (phase.projectMoveScoreAtEnd && !phase.tuneMoveScore)
            throw std::invalid_argument(
                "project_move_score_at_end requires an active move_score objective");
        if (phase.projectMoveScoreAtEnd && !phaseConfig.projectMoveScoreScale)
            throw std::invalid_argument(
                "project_move_score_at_end requires --project-move-score-scale");
        if (phase.projectMoveScoreAtEnd && i + 1 == phases.size())
            throw std::invalid_argument(
                "project_move_score_at_end requires a following curriculum phase");
        if (phase.projectMoveScoreAtEnd
            && std::any_of(phases.begin() + i + 1,
                           phases.end(),
                           [](const Tuning::TuningPhase &later) { return later.tuneMoveScore; }))
            throw std::invalid_argument(
                "project_move_score_at_end requires move_score to remain frozen");
        if (phase.coalesceMoveScoreAtEnd) {
            coalescingBoundaries++;
            if (!phase.tuneMoveScore)
                throw std::invalid_argument(
                    "coalesce_move_score_at_end requires an active move_score objective");
            if (phase.projectMoveScoreAtEnd)
                throw std::invalid_argument(
                    "move-score coalescing and projection cannot share a phase boundary");
            if (i + 1 == phases.size())
                throw std::invalid_argument(
                    "coalesce_move_score_at_end requires a following curriculum phase");
            if (phaseConfig.trainingSemantics != Tuning::TrainingSemantics::Bootstrap)
                throw std::invalid_argument(
                    "move-score coalescing requires bootstrap semantics before the boundary");
            Tuning::TuningConfig nextConfig = configForPhase(config, phases[i + 1]);
            if (nextConfig.trainingSemantics != Tuning::TrainingSemantics::Current)
                throw std::invalid_argument(
                    "move-score coalescing requires current semantics after the boundary");
        }
        if (i > 0) {
            Tuning::TuningConfig previousConfig = configForPhase(config, phases[i - 1]);
            if (previousConfig.trainingSemantics != phaseConfig.trainingSemantics
                && !phases[i - 1].coalesceMoveScoreAtEnd)
                throw std::invalid_argument(
                    "curriculum training semantics may change only at a coalescing boundary");
        }
        for (size_t previous = 0; previous < i; previous++) {
            if (normalizedPathKey(phases[previous].preparedCachePath)
                == normalizedPathKey(phase.preparedCachePath))
                throw std::invalid_argument(
                    "tuning curriculum phases require distinct prepared-cache namespaces");
        }
    }
    if (coalescingBoundaries > 1)
        throw std::invalid_argument("a tuning curriculum supports at most one coalescing boundary");
    Tuning::TuningConfig layoutConfig = configForPhase(config, phases.front());
    layoutConfig.tuneEval             = config.tuneEval;
    layoutConfig.tuneMoveScore        = config.tuneMoveScore;
    return layoutConfig;
}

}  // namespace

namespace Tuning {

using detail::makeMoveScoreLossSettings;
using detail::MoveScoreLossSettings;

Tuner::Tuner(Dataset                                &trainDataset,
             Dataset                                *valDataset,
             TuningConfig                            config,
             std::vector<MoveScoreReferencePosition> moveScoreReferencePositions)
    : Tuner(trainDataset,
            valDataset,
            std::move(config),
            std::move(moveScoreReferencePositions),
            DeferCorpusPreparation {})
{
    prepareCorpus(trainDataset, valDataset);
}

Tuner::Tuner(Dataset                                &trainDataset,
             Dataset                                *valDataset,
             TuningConfig                            config,
             std::vector<MoveScoreReferencePosition> moveScoreReferencePositions,
             DeferCorpusPreparation)
    : config(config)
    , policyTargetConfig(policyTargetConfigFor(config))
    , moveScoreReferencePositions(std::move(moveScoreReferencePositions))
    , threadPool(config.numThreads != 0 ? config.numThreads
                                        : std::max<size_t>(std::thread::hardware_concurrency(), 1))
{
    MESSAGEL("Tuner worker threads = " << threadPool.get_thread_count()
                                       << ", seed = " << config.seed << ".");
    MESSAGEL("Start initializing parameters...");
    initParams();
    moveScoreReference = detail::makeMoveScoreReference(config, this->moveScoreReferencePositions);

    if (policyTargetConfig.useMultiPV())
        MESSAGEL("Multi-PV policy targets enabled: temperature = "
                 << policyTargetConfig.multiPVTemperature
                 << ", eval scaling factor = " << policyTargetConfig.evalScalingFactor << ".");
    if (config.tuneMoveScore)
        MESSAGEL(
            "Move-score softmax mode = "
            << (config.trainingSemantics == TrainingSemantics::Bootstrap ? "bootstrap" : "current")
            << ", canonical logit scale = " << Evaluation::PolicyBuffer::ScoreScale << ".");
    if (config.tuneEval && !config.usePreviousScalingFactor && config.nStepsPerIteration < 1)
        throw std::invalid_argument("scaling-factor calibration requires at least one step");
    if (config.memoryLimitMB != 0) {
        if (config.memoryLimitMB < 32
            || config.memoryLimitMB > std::numeric_limits<size_t>::max() / MiB)
            throw std::invalid_argument("file-backed memory limit must be at least 32 MiB");
        if (config.shardSizeMB == 0
            || config.shardSizeMB > std::numeric_limits<size_t>::max() / MiB)
            throw std::invalid_argument("prepared shard target must fit size_t");
        if (config.preparedCachePath.empty())
            throw std::invalid_argument("prepared cache path is required for file-backed tuning");
        if (config.shuffleTuneEntries)
            throw std::invalid_argument("global shuffle is not supported by file-backed tuning");

        constexpr size_t RuntimeAllowance            = 16 * MiB;
        constexpr size_t CalibrationScratchAllowance = RuntimeAllowance / 2;
        if (config.tuneEval && !config.usePreviousScalingFactor) {
            size_t calibrationBytes =
                checkedProduct(checkedProduct(static_cast<size_t>(config.nStepsPerIteration),
                                              sizeof(Float),
                                              "calibration grid"),
                               LogicalPartitions + 2,
                               "calibration grid");
            if (calibrationBytes > CalibrationScratchAllowance)
                throw std::invalid_argument(
                    "scaling-factor calibration grid exceeds its runtime memory allowance; "
                    "reduce --num-steps-per-iteration");
        }
        size_t budgetBytes = checkedProduct(config.memoryLimitMB, MiB, "memory limit");
        size_t paramCopies = LogicalPartitions + 9;
        size_t fixedTrainingBytes =
            checkedProduct(checkedProduct(paramCopies, tuneParams.size(), "optimizer state"),
                           sizeof(TuneParam),
                           "optimizer state");
        if (fixedTrainingBytes > budgetBytes || RuntimeAllowance > budgetBytes - fixedTrainingBytes)
            throw std::invalid_argument(
                "memory limit is too small for optimizer state and runtime allowance");

        size_t variableBytes     = budgetBytes - fixedTrainingBytes - RuntimeAllowance;
        fileWorkerBudgetBytes    = variableBytes / 2;
        fileShardBudgetBytes     = variableBytes - fileWorkerBudgetBytes;
        size_t workerCount       = std::max<size_t>(threadPool.get_thread_count(), 1);
        size_t inputDatasetCount = valDataset ? 2 : 1;
        fileRecordLimitBytes =
            std::min<size_t>(8 * MiB,
                             fileWorkerBudgetBytes / (2 * (workerCount + inputDatasetCount)));
        if (fileRecordLimitBytes < 64 * KiB || fileShardBudgetBytes < 64 * KiB)
            throw std::invalid_argument(
                "memory limit leaves insufficient worker or shard allocation credit");
        fileJobBudgetBytes  = fileWorkerBudgetBytes - inputDatasetCount * fileRecordLimitBytes;
        fileMaxPendingJobs  = workerCount;
        fileChunkEntryLimit = std::max<size_t>(
            1,
            std::min(config.batchSize, fileJobBudgetBytes / workerCount / PreparedSampleCredit));
        size_t requestedShardBytes = checkedProduct(config.shardSizeMB, MiB, "shard target");
        fileShardTargetBytes =
            std::min(requestedShardBytes, fileShardBudgetBytes - fileShardBudgetBytes / 4);

        MESSAGEL("File-backed budget: workers "
                 << fileWorkerBudgetBytes / double(MiB) << " MiB, shard "
                 << fileShardBudgetBytes / double(MiB) << " MiB, shard target "
                 << fileShardTargetBytes / double(MiB) << " MiB.");
    }

    trainableParams.assign(tuneParams.size(), uint8_t(1));
    if (config.tuneMoveScore && moveScoreParamRanges.empty())
        throw std::logic_error("policy tuning requires at least one parameter range");
}

Tuner::Tuner(std::vector<TuningPhase>                phases,
             TuningConfig                            config,
             std::vector<MoveScoreReferencePosition> moveScoreReferencePositions)
    : Tuner(*firstPhase(phases).trainingDataset,
            firstPhase(phases).validationDataset,
            validatedCurriculumConfig(phases, config),
            std::move(moveScoreReferencePositions),
            DeferCorpusPreparation {})
{
    curriculumBaseConfig = std::move(config);
    curriculumPhases     = std::move(phases);
    phaseObjectiveMasking =
        std::any_of(curriculumPhases.begin(),
                    curriculumPhases.end(),
                    [&](const TuningPhase &phase) {
                        return phase.tuneEval != curriculumBaseConfig.tuneEval
                               || phase.tuneMoveScore != curriculumBaseConfig.tuneMoveScore;
                    });
    inactiveCorpusStates  = std::vector<CorpusState>(curriculumPhases.size());
    bool deferPreparation = false;
    for (size_t phaseIndex = 0; phaseIndex < curriculumPhases.size(); phaseIndex++) {
        if (deferPreparation)
            continue;
        applyPhaseConfig(curriculumPhases[phaseIndex]);
        prepareCorpus(*curriculumPhases[phaseIndex].trainingDataset,
                      curriculumPhases[phaseIndex].validationDataset);
        storeActiveCorpus(phaseIndex);
        deferPreparation = curriculumPhases[phaseIndex].coalesceMoveScoreAtEnd;
    }
    applyPhaseConfig(curriculumPhases.front());
    loadActiveCorpus(0);
}

void Tuner::applyPhaseConfig(const TuningPhase &phase)
{
    config             = configForPhase(curriculumBaseConfig, phase);
    policyTargetConfig = policyTargetConfigFor(config);
    updateTrainableParams(phase.trainCompactPolicy);
}

void Tuner::updateTrainableParams(bool allowNoTraditionalParams)
{
    if (trainableParams.empty())
        return;
    std::fill(trainableParams.begin(), trainableParams.end(), uint8_t(0));
    if (config.tuneEval)
        for (const auto &[begin, end] : evalParamRanges)
            std::fill(trainableParams.begin() + begin, trainableParams.begin() + end, uint8_t(1));
    if (config.tuneMoveScore)
        for (const auto &[begin, end] : moveScoreParamRanges)
            std::fill(trainableParams.begin() + begin, trainableParams.begin() + end, uint8_t(1));
    if (!allowNoTraditionalParams
        && std::none_of(trainableParams.begin(), trainableParams.end(), [](uint8_t active) {
               return active != 0;
           }))
        throw std::logic_error("tuning phase activated no parameters");
}

void Tuner::coalesceMoveScoreLayout(AdamOptimizer<TuneParam> &optimizer)
{
    if (config.trainingSemantics != TrainingSemantics::Bootstrap || !config.tuneMoveScore)
        throw std::logic_error("move-score coalescing requires active bootstrap policy training");
    if (moveScoreParamRanges.empty() || moveScoreParamRanges.size() != moveScoreTables.size())
        throw std::logic_error("move-score coalescing requires complete bootstrap table metadata");
    if (!tiedMoveScoreSyncRecords.empty() || !tiedMoveScoreParamIndices.empty())
        throw std::logic_error("move-score parameters are already tied");

    const size_t                          prefixEnd = moveScoreParamRanges.front().first;
    size_t                                oldPolicyParameterCount = 0;
    size_t                                expectedRangeBegin      = prefixEnd;
    std::vector<const ParamsSyncRecord *> policyRecords(moveScoreParamRanges.size(), nullptr);
    for (size_t tableIndex = 0; tableIndex < moveScoreParamRanges.size(); tableIndex++) {
        const auto [rangeBegin, rangeEnd] = moveScoreParamRanges[tableIndex];
        const int table                   = moveScoreTables[tableIndex];
        if (table < 0 || table >= RULE_NB + 1 || rangeBegin != expectedRangeBegin)
            throw std::logic_error("bootstrap move-score ranges are not a terminal partition");
        const size_t scoreCount = arraySize(Evaluation::P4SCORES[table]);
        if (rangeEnd - rangeBegin != scoreCount * 2)
            throw std::logic_error("bootstrap move-score range has an unexpected size");
        expectedRangeBegin = rangeEnd;
        oldPolicyParameterCount += rangeEnd - rangeBegin;

        for (const ParamsSyncRecord &record : syncRecords) {
            const size_t recordEnd = record.baseIndex + record.numElems * record.paramPerElem;
            if (record.baseIndex == rangeBegin && recordEnd == rangeEnd) {
                if (policyRecords[tableIndex])
                    throw std::logic_error("move-score coalescing found multiple policy records");
                policyRecords[tableIndex] = &record;
            }
            else if (record.baseIndex < rangeEnd && recordEnd > rangeBegin)
                throw std::logic_error("move-score parameter range overlaps another sync record");
        }
        const ParamsSyncRecord *record = policyRecords[tableIndex];
        if (!record || record->address != Evaluation::P4SCORES[table]
            || record->numElems != scoreCount || record->paramPerElem != 2)
            throw std::logic_error(
                "move-score coalescing cannot identify a bootstrap policy record");
    }
    if (expectedRangeBegin != tuneParams.size())
        throw std::logic_error("bootstrap move-score ranges are not terminal");

    std::vector<TuneParam>           newParams(tuneParams.begin(), tuneParams.begin() + prefixEnd);
    std::vector<std::vector<size_t>> stateGroups(prefixEnd);
    for (size_t index = 0; index < prefixEnd; index++)
        stateGroups[index].push_back(index);

    struct CoalescedTable
    {
        MoveScorePair             *scores;
        std::vector<MoveScorePair> quantizedScores;
    };
    std::vector<CoalescedTable>            coalescedTables;
    std::vector<TiedMoveScoreSyncRecord>   newTiedRecords;
    std::vector<std::pair<size_t, size_t>> newMoveScoreRanges;
    coalescedTables.reserve(moveScoreParamRanges.size());
    newTiedRecords.reserve(moveScoreParamRanges.size());
    newMoveScoreRanges.reserve(moveScoreParamRanges.size());
    constexpr size_t MaxParameterCount = size_t(std::numeric_limits<ParameterId>::max()) + 1;
    for (size_t tableIndex = 0; tableIndex < moveScoreParamRanges.size(); tableIndex++) {
        const auto [rangeBegin, rangeEnd]     = moveScoreParamRanges[tableIndex];
        const int               table         = moveScoreTables[tableIndex];
        MoveScorePair          *scores        = Evaluation::P4SCORES[table];
        const size_t            scoreCount    = arraySize(Evaluation::P4SCORES[table]);
        const size_t            newRangeBegin = newParams.size();
        TiedMoveScoreSyncRecord tiedRecord;
        tiedRecord.layoutTag = "move-score/coalesced-table-" + std::to_string(table);
        tiedRecord.scores    = scores;
        tiedRecord.parameterIndices.resize(scoreCount);
        std::vector<MoveScorePair>                            quantizedScores(scoreCount);
        std::array<std::unordered_map<Score, ParameterId>, 2> groups;
        for (size_t side = 0; side < 2; side++) {
            for (size_t i = 0; i < scoreCount; i++) {
                const size_t oldIndex    = rangeBegin + i * 2 + side;
                const Score  score       = decodeMoveScoreParam(tuneParams[oldIndex]);
                quantizedScores[i][side] = score;
                auto found               = groups[side].find(score);
                if (found == groups[side].end()) {
                    if (newParams.size() >= MaxParameterCount)
                        throw std::length_error(
                            "coalesced tuning parameter count exceeds ParameterId capacity");
                    const TuneParam encoded =
                        encodeIntegerForTruncatingExport(score,
                                                         config.moveScoreScale,
                                                         config.moveScoreBias);
                    if (decodeMoveScoreParam(encoded) != score)
                        throw std::runtime_error(
                            "coalesced move score is not representable by tuner parameters");
                    const ParameterId newIndex = static_cast<ParameterId>(newParams.size());
                    newParams.push_back(encoded);
                    stateGroups.push_back({oldIndex});
                    found = groups[side].emplace(score, newIndex).first;
                }
                else
                    stateGroups[found->second].push_back(oldIndex);
                tiedRecord.parameterIndices[i][side] = found->second;
            }
        }
        newMoveScoreRanges.emplace_back(newRangeBegin, newParams.size());
        newTiedRecords.push_back(std::move(tiedRecord));
        coalescedTables.push_back({scores, std::move(quantizedScores)});
    }

    std::vector<ParamsSyncRecord> newSyncRecords;
    newSyncRecords.reserve(syncRecords.size() - policyRecords.size());
    std::unordered_map<const void *, ParameterAddress> newParamIndices;
    for (const ParamsSyncRecord &record : syncRecords) {
        if (std::find(policyRecords.begin(), policyRecords.end(), &record) != policyRecords.end())
            continue;
        const size_t recordEnd = record.baseIndex + record.numElems * record.paramPerElem;
        if (recordEnd > prefixEnd)
            throw std::logic_error("non-policy parameter record follows bootstrap policy ranges");
        newSyncRecords.push_back(record);
        for (size_t i = 0; i < record.numElems; i++) {
            const size_t elementBase = record.baseIndex + i * record.paramPerElem;
            if (!newParamIndices
                     .emplace(record[i],
                              ParameterAddress {static_cast<ParameterId>(elementBase),
                                                record.paramPerElem})
                     .second)
                throw std::logic_error("coalesced tuning layout contains duplicate addresses");
        }
    }
    std::unordered_map<const void *, std::array<ParameterId, 2>> newTiedIndices;
    for (const TiedMoveScoreSyncRecord &record : newTiedRecords)
        for (size_t i = 0; i < record.parameterIndices.size(); i++)
            if (!newTiedIndices.emplace(&record.scores[i], record.parameterIndices[i]).second)
                throw std::logic_error("coalesced move-score layout contains duplicate addresses");

    struct StagedElement
    {
        void                *destination;
        std::vector<uint8_t> objectBytes;
    };
    std::vector<StagedElement> stagedElements;
    for (const ParamsSyncRecord &record : newSyncRecords) {
        for (size_t i = 0; i < record.numElems; i++) {
            std::vector<TuneParam> elementParams;
            elementParams.reserve(record.paramPerElem);
            for (size_t j = 0; j < record.paramPerElem; j++)
                elementParams.push_back(newParams[record.baseIndex + i * record.paramPerElem + j]);
            StagedParams staged = record.stager(record[i], elementParams);
            if (staged.params.size() != record.paramPerElem
                || staged.objectBytes.size() != record.elemSize)
                throw std::logic_error("tuning parameter stager returned an invalid element");
            for (size_t j = 0; j < record.paramPerElem; j++) {
                const size_t    index     = record.baseIndex + i * record.paramPerElem + j;
                const TuneParam quantized = staged.params[j];
                if (!std::isfinite(quantized))
                    throw std::runtime_error(
                        "coalesced model boundary produced a non-finite parameter");
                newParams[index] = quantized;
            }
            stagedElements.push_back({record[i], std::move(staged.objectBytes)});
        }
    }

    AdamOptimizer<TuneParam> newOptimizer = optimizer.coalesced(stateGroups);
    std::vector<uint8_t>     newTrainableParams(newParams.size(), uint8_t(1));

    for (const StagedElement &staged : stagedElements)
        std::memcpy(staged.destination, staged.objectBytes.data(), staged.objectBytes.size());
    for (const CoalescedTable &table : coalescedTables)
        for (size_t i = 0; i < table.quantizedScores.size(); i++)
            table.scores[i] = table.quantizedScores[i];
    optimizer.swap(newOptimizer);
    tuneParams.swap(newParams);
    syncRecords.swap(newSyncRecords);
    tiedMoveScoreSyncRecords.swap(newTiedRecords);
    paramIndices.swap(newParamIndices);
    tiedMoveScoreParamIndices.swap(newTiedIndices);
    moveScoreParamRanges.swap(newMoveScoreRanges);
    trainableParams.swap(newTrainableParams);

    MESSAGEL("Coalesced bootstrap move-score layout from "
             << oldPolicyParameterCount << " parameters to " << (tuneParams.size() - prefixEnd)
             << " tied parameters with conservative Adam state transport.");
}

void Tuner::storeActiveCorpus(size_t phaseIndex)
{
    CorpusState &state                 = inactiveCorpusStates.at(phaseIndex);
    state.trainTuneEntries             = std::move(trainTuneEntries);
    state.valTuneEntries               = std::move(valTuneEntries);
    state.trainFileCorpus              = std::move(trainFileCorpus);
    state.valFileCorpus                = std::move(valFileCorpus);
    state.trainSampleOrder             = std::move(trainSampleOrder);
    state.trainBoardSampleCounts       = trainBoardSampleCounts;
    state.trainBoardOptimizationCounts = trainBoardOptimizationCounts;
    state.validationBoardSampleCounts  = validationBoardSampleCounts;
    state.trainBoardSampleWeights      = trainBoardSampleWeights;
    state.prepared = true;
}

void Tuner::loadActiveCorpus(size_t phaseIndex)
{
    CorpusState &state = inactiveCorpusStates.at(phaseIndex);
    if (!state.prepared)
        throw std::logic_error("tuning curriculum phase corpus is not prepared");
    trainTuneEntries             = std::move(state.trainTuneEntries);
    valTuneEntries               = std::move(state.valTuneEntries);
    trainFileCorpus              = std::move(state.trainFileCorpus);
    valFileCorpus                = std::move(state.valFileCorpus);
    trainSampleOrder             = std::move(state.trainSampleOrder);
    trainBoardSampleCounts       = state.trainBoardSampleCounts;
    trainBoardOptimizationCounts = state.trainBoardOptimizationCounts;
    validationBoardSampleCounts  = state.validationBoardSampleCounts;
    trainBoardSampleWeights      = state.trainBoardSampleWeights;
    currentPhaseIndex = phaseIndex;
}

void Tuner::prepareCorpus(Dataset &trainDataset, Dataset *valDataset)
{
    std::optional<PreparedCacheKey> validationCacheKey;
    if (config.memoryLimitMB != 0) {
        PreparedCacheKey trainCacheKey =
            makePreparedCacheKey(config.trainDatasetPaths, config.trainDatasetFormat, "train");
        if (valDataset)
            validationCacheKey.emplace(makePreparedCacheKey(config.validationDatasetPaths,
                                                            config.validationDatasetFormat,
                                                            "validation"));
        validatePreparedCacheRoot(config.preparedCachePath,
                                  trainCacheKey,
                                  validationCacheKey ? &*validationCacheKey : nullptr);
        trainFileCorpus = std::make_unique<FileBackedCorpus>(config.preparedCachePath,
                                                             "train",
                                                             fileShardBudgetBytes,
                                                             std::move(trainCacheKey),
                                                             config.rebuildPreparedCache);
        MESSAGEL(trainFileCorpus->cacheStatus() << '.');
    }

    if (!trainFileCorpus || !trainFileCorpus->reused()) {
        if (trainFileCorpus)
            trainDataset.reset();
        MESSAGEL("Start initializing tune entries from training dataset...");
        initTuneEntries(trainTuneEntries, trainFileCorpus.get(), trainDataset, true);
        if (trainFileCorpus)
            trainFileCorpus->publish();
    }
    else
        MESSAGEL(trainFileCorpus->size() << " training entries restored from prepared cache.");

    if (valDataset) {
        if (config.memoryLimitMB != 0) {
            assert(validationCacheKey);
            valFileCorpus = std::make_unique<FileBackedCorpus>(config.preparedCachePath,
                                                               "validation",
                                                               fileShardBudgetBytes,
                                                               std::move(*validationCacheKey),
                                                               config.rebuildPreparedCache);
            MESSAGEL(valFileCorpus->cacheStatus() << '.');
        }
        if (!valFileCorpus || !valFileCorpus->reused()) {
            if (valFileCorpus)
                valDataset->reset();
            MESSAGEL("Start initializing tune entries from validation dataset...");
            initTuneEntries(valTuneEntries, valFileCorpus.get(), *valDataset, false);
            if (valFileCorpus)
                valFileCorpus->publish();
        }
        else
            MESSAGEL(valFileCorpus->size() << " validation entries restored from prepared cache.");
    }

    trainBoardSampleCounts      = computeTrainingBoardSampleCounts(trainBoardOptimizationCounts);
    validationBoardSampleCounts = computeValidationBoardSampleCounts();
    trainBoardSampleWeights.fill(Float(1));
    if (config.boardSizeWeighting == BoardSizeWeighting::EqualBoard) {
        size_t activeBoards = std::count_if(trainBoardOptimizationCounts.begin(),
                                            trainBoardOptimizationCounts.end(),
                                            [](size_t count) { return count != 0; });
        size_t totalSamples = std::accumulate(trainBoardOptimizationCounts.begin(),
                                              trainBoardOptimizationCounts.end(),
                                              size_t(0));
        if (activeBoards != 0) {
            for (size_t boardSize = 1; boardSize < trainBoardOptimizationCounts.size();
                 boardSize++) {
                size_t count = trainBoardOptimizationCounts[boardSize];
                if (count != 0)
                    trainBoardSampleWeights[boardSize] =
                        Float(totalSamples) / Float(activeBoards * count);
            }
        }
        MESSAGEL("Board-size weighting = equal-board across " << activeBoards
                                                              << " observed sizes.");
    }
    else
        MESSAGEL("Board-size weighting = sample-frequency.");
}

std::vector<BoardSampleStatistic> Tuner::boardSampleStatistics() const
{
    return boardSampleStatistics(currentPhaseIndex);
}

std::vector<BoardSampleStatistic> Tuner::boardSampleStatistics(size_t phaseIndex) const
{
    const auto *sampleCounts       = &trainBoardSampleCounts;
    const auto *optimizationCounts = &trainBoardOptimizationCounts;
    const auto *validationCounts   = &validationBoardSampleCounts;
    const auto *sampleWeights      = &trainBoardSampleWeights;
    if (!curriculumPhases.empty() && phaseIndex != currentPhaseIndex) {
        const CorpusState &state = inactiveCorpusStates.at(phaseIndex);
        if (!state.prepared)
            throw std::logic_error("tuning curriculum phase corpus is not prepared yet");
        sampleCounts       = &state.trainBoardSampleCounts;
        optimizationCounts = &state.trainBoardOptimizationCounts;
        validationCounts   = &state.validationBoardSampleCounts;
        sampleWeights      = &state.trainBoardSampleWeights;
    }
    else if (!curriculumPhases.empty() && phaseIndex >= curriculumPhases.size())
        throw std::out_of_range("tuning curriculum phase index is out of range");

    std::vector<BoardSampleStatistic> statistics;
    for (size_t boardSize = 1; boardSize < sampleCounts->size(); boardSize++) {
        if ((*sampleCounts)[boardSize] == 0 && (*validationCounts)[boardSize] == 0)
            continue;
        statistics.push_back({static_cast<uint8_t>(boardSize),
                              (*sampleCounts)[boardSize],
                              (*optimizationCounts)[boardSize],
                              (*validationCounts)[boardSize],
                              (*sampleWeights)[boardSize]});
    }
    return statistics;
}

std::array<size_t, MAX_BOARD_SIZE + 1> Tuner::computeTrainingBoardSampleCounts(
    std::array<size_t, MAX_BOARD_SIZE + 1> &optimizationCounts) const
{
    std::array<size_t, MAX_BOARD_SIZE + 1> acceptedCounts = {};
    optimizationCounts.fill(0);
    auto boardSizeAt = [](const PreparedCorpus &entries, size_t sample) {
        uint8_t boardSize = entries.boardSizes()[sample];
        if (boardSize == 0 || boardSize > MAX_BOARD_SIZE)
            throw std::runtime_error("prepared sample board size exceeds engine limit");
        return boardSize;
    };

    if (trainFileCorpus) {
        const auto &fileCounts = trainFileCorpus->boardSampleCounts();
        for (size_t boardSize = 0; boardSize < fileCounts.size(); boardSize++) {
            if ((boardSize == 0 || boardSize > MAX_BOARD_SIZE) && fileCounts[boardSize] != 0)
                throw std::runtime_error("prepared sample board size exceeds engine limit");
            if (boardSize <= MAX_BOARD_SIZE)
                acceptedCounts[boardSize] = fileCounts[boardSize];
        }
        for (size_t shard = 0; shard < trainFileCorpus->shardCount(); shard++) {
            PreparedCorpus entries       = trainFileCorpus->load(shard);
            size_t         usableSamples = entries.size() / config.batchSize * config.batchSize;
            for (size_t sample = 0; sample < usableSamples; sample++)
                optimizationCounts[boardSizeAt(entries, sample)]++;
        }
    }
    else {
        for (size_t sample = 0; sample < trainTuneEntries.size(); sample++)
            acceptedCounts[boardSizeAt(trainTuneEntries, sample)]++;
        size_t usableSamples = trainTuneEntries.size() / config.batchSize * config.batchSize;
        for (size_t logicalSample = 0; logicalSample < usableSamples; logicalSample++) {
            size_t sample =
                trainSampleOrder.empty() ? logicalSample : trainSampleOrder[logicalSample];
            optimizationCounts[boardSizeAt(trainTuneEntries, sample)]++;
        }
    }
    return acceptedCounts;
}

std::array<size_t, MAX_BOARD_SIZE + 1> Tuner::computeValidationBoardSampleCounts() const
{
    std::array<size_t, MAX_BOARD_SIZE + 1> counts     = {};
    auto                                   addEntries = [&counts](const PreparedCorpus &entries) {
        for (uint8_t boardSize : entries.boardSizes()) {
            if (boardSize == 0 || boardSize > MAX_BOARD_SIZE)
                throw std::runtime_error("prepared sample board size exceeds engine limit");
            counts[boardSize]++;
        }
    };
    if (valFileCorpus) {
        const auto &fileCounts = valFileCorpus->boardSampleCounts();
        for (size_t boardSize = 0; boardSize < fileCounts.size(); boardSize++) {
            if ((boardSize == 0 || boardSize > MAX_BOARD_SIZE) && fileCounts[boardSize] != 0)
                throw std::runtime_error("prepared sample board size exceeds engine limit");
            if (boardSize <= MAX_BOARD_SIZE)
                counts[boardSize] = fileCounts[boardSize];
        }
    }
    else
        addEntries(valTuneEntries);
    return counts;
}

/// run() runs the tuner for specified epochs. After each epoch completed, callback will be called.
void Tuner::run(size_t epochs, std::function<void(TuningStatistic)> callback)
{
    if (!curriculumPhases.empty())
        throw std::logic_error("run() cannot be used with a tuning curriculum");
    runImpl(epochs, nullptr, std::move(callback));
}

void Tuner::runCurriculum(std::function<void(TuningStatistic)> callback)
{
    if (curriculumPhases.empty())
        throw std::logic_error("runCurriculum() requires a tuning curriculum");
    size_t totalEpochs = 0;
    for (const TuningPhase &phase : curriculumPhases) {
        if (phase.epochs > std::numeric_limits<size_t>::max() - totalEpochs)
            throw std::overflow_error("tuning curriculum epoch sum overflows size_t");
        totalEpochs += phase.epochs;
    }
    runImpl(totalEpochs, &curriculumPhases, std::move(callback));
}

void Tuner::runImpl(size_t                               epochs,
                    const std::vector<TuningPhase>      *phases,
                    std::function<void(TuningStatistic)> callback)
{
    Time initTime = now();

    // Set Float output precision
    std::cout << std::setprecision(std::min(std::numeric_limits<Float>::digits10, 7)) << std::fixed;

    auto emitCompactPolicyStatistic = [&](size_t globalEpoch,
                                          size_t phaseIndex,
                                          size_t localEpoch,
                                          double scalingFactor) {
        if (!callback)
            return;
        TuningStatistic stat {};
        stat.currentEpoch       = globalEpoch;
        stat.currentPhase       = phaseIndex;
        stat.currentPhaseEpoch  = localEpoch;
        stat.phaseName          = (*phases)[phaseIndex].name;
        stat.elapsedSeconds     = 0.001;
        stat.scalingFactor      = scalingFactor;
        stat.learningRate       = 0;
        stat.trainCompactPolicy = true;
        callback(stat);
    };

    const bool compactOnlyCurriculum =
        phases
        && std::all_of(phases->begin(), phases->end(), [](const TuningPhase &phase) {
               return phase.trainCompactPolicy && !phase.tuneEval && !phase.tuneMoveScore;
           });
    if (compactOnlyCurriculum) {
        MESSAGEL("Start compact-policy-only curriculum for " << epochs << " epochs.");
        size_t globalEpoch = 0;
        for (size_t phaseIndex = 0; phaseIndex < phases->size(); phaseIndex++) {
            const TuningPhase &phase = (*phases)[phaseIndex];
            if (phaseIndex != 0)
                applyPhaseConfig(phase);
            const size_t firstLocalEpoch = phaseIndex == 0 ? 0 : 1;
            for (size_t localEpoch = firstLocalEpoch; localEpoch <= phase.epochs; localEpoch++) {
                if (phaseIndex != 0 || localEpoch != 0)
                    globalEpoch++;
                emitCompactPolicyStatistic(
                    globalEpoch, phaseIndex, localEpoch, Evaluation::ScalingFactor);
            }
        }
        if (globalEpoch != epochs)
            throw std::logic_error("compact-policy curriculum epoch count mismatch");
        MESSAGEL("Compact-policy-only training completed in " << ((now() - initTime) / 1000)
                                                                << " seconds.");
        return;
    }

    // Note: the last non-full batch of tune entries will be dropped.
    // Validate this before calibration, which may otherwise scan the corpus
    // many times only to discover that training cannot run.
    auto validateActiveCorpus = [&]() {
        size_t trainSampleCount =
            trainFileCorpus ? trainFileCorpus->size() : trainTuneEntries.size();
        size_t activeBatches = trainSampleCount / config.batchSize;
        if (activeBatches == 0)
            throw std::runtime_error("training dataset has fewer accepted entries than one batch");
        if (trainFileCorpus && trainFileCorpus->maxShardStorageBytes() > fileShardBudgetBytes)
            throw std::logic_error("prepared shard escaped its allocation credit");
        return activeBatches;
    };
    size_t numBatches = validateActiveCorpus();

    // Search a new K or use previous K. Policy-only tuning does not use a
    // value scaling factor and must not scan the corpus for calibration.
    Float K = Float(1.0) / Evaluation::ScalingFactor;
    if (!config.tuneEval) {
        MESSAGEL("Skip scaling factor search for policy-only tuning.");
    }
    else if (config.usePreviousScalingFactor) {
        MESSAGEL("Use previous inv scaling factor = " << K);
    }
    else {
        MESSAGEL("Start seaching for optimal inv scaling factor...");
        K = searchOptimalInvScalingFactor(false);
    }

    // Init gradient array and optimizer
    std::vector<TuneGradient> gradients(tuneParams.size());
    std::vector<TuneParam>    parameterLearningRates;
    AdamOptimizer<TuneParam>  optim(tuneParams.size(),
                                   TuneParam(config.learningRate),
                                   TuneParam(config.weightDecay));
    MESSAGEL("Start training for "
             << epochs << " epochs, initial lr = " << optim.currentLR()
             << ", final lr = " << learningRateForEpoch(config, epochs, epochs) << ", batch size = "
             << config.batchSize << ", number of batches = " << numBatches << ".");

    auto updateBatch = [&](const PreparedCorpus        &entries,
                           size_t                       batchBegin,
                           const std::vector<uint32_t> *sampleOrder) {
        std::fill(gradients.begin(), gradients.end(), TuneGradient(0));
        computeGradientBatch(gradients, K, entries, batchBegin, sampleOrder);
        if (std::any_of(gradients.begin(), gradients.end(), [](TuneGradient gradient) {
                return !std::isfinite(gradient);
            }))
            throw std::runtime_error("non-finite gradient in tuning batch");
        if (phaseObjectiveMasking)
            optim.stepMasked(tuneParams, gradients, parameterLearningRates, trainableParams);
        else
            optim.step(tuneParams, gradients);
    };

    size_t phaseIndex = 0;
    size_t phaseEpoch = 0;
    if (phases) {
        MESSAGEL("Curriculum phase 1/"
                 << phases->size() << " [" << phases->front().name
                 << "] begins at global epoch 0, local epoch 0, train samples "
                 << (trainFileCorpus ? trainFileCorpus->size() : trainTuneEntries.size())
                 << ", validation samples "
                 << (valFileCorpus ? valFileCorpus->size() : valTuneEntries.size()) << ", batch "
                 << config.batchSize << ", cache " << config.preparedCachePath.string()
                 << (config.tuneMoveScore
                         ? ", MultiPV temperature "
                               + std::to_string(config.multiPVPolicyTemperature)
                         : std::string())
                 << ".");
    }

    for (size_t epoch = 0; epoch <= epochs; epoch++) {
        Time startTime              = now();
        bool recalibratedAtBoundary = false;

        if (phases && epoch > 0 && phaseEpoch == (*phases)[phaseIndex].epochs) {
            if (phaseIndex + 1 >= phases->size())
                throw std::logic_error("tuning curriculum ended before the global epoch horizon");
            if ((*phases)[phaseIndex].projectMoveScoreAtEnd)
                synchronizeProjectedMoveScores();
            storeActiveCorpus(phaseIndex);
            if ((*phases)[phaseIndex].coalesceMoveScoreAtEnd) {
                coalesceMoveScoreLayout(optim);
                gradients.assign(tuneParams.size(), TuneGradient(0));
                parameterLearningRates.clear();
                partitionGradients.clear();
                for (size_t future = phaseIndex + 1; future < inactiveCorpusStates.size(); future++)
                    inactiveCorpusStates[future] = CorpusState {};
            }
            phaseIndex++;
            applyPhaseConfig((*phases)[phaseIndex]);
            const TuningPhase &nextPhase = (*phases)[phaseIndex];
            if (nextPhase.trainCompactPolicy && !nextPhase.tuneEval && !nextPhase.tuneMoveScore) {
                if (phaseIndex + 1 != phases->size())
                    throw std::logic_error(
                        "a pure compact-policy phase must end a mixed tuning curriculum");
                MESSAGEL("Terminal compact-policy phase "
                         << (phaseIndex + 1) << '/' << phases->size() << " [" << nextPhase.name
                         << "] begins at global epoch " << epoch << '.');
                for (size_t localEpoch = 1; localEpoch <= nextPhase.epochs; localEpoch++)
                    emitCompactPolicyStatistic(
                        epoch + localEpoch - 1, phaseIndex, localEpoch, 1.0 / double(K));
                if (epoch + nextPhase.epochs - 1 != epochs)
                    throw std::logic_error("terminal compact-policy epoch count mismatch");
                Time totalElapsed = now() - initTime;
                MESSAGEL("Training completed in " << (totalElapsed / 1000) << " seconds.");
                return;
            }
            optim.setWeightDecay(TuneParam(config.weightDecay));
            if (!inactiveCorpusStates[phaseIndex].prepared) {
                prepareCorpus(*(*phases)[phaseIndex].trainingDataset,
                              (*phases)[phaseIndex].validationDataset);
                storeActiveCorpus(phaseIndex);
            }
            loadActiveCorpus(phaseIndex);
            phaseEpoch = 0;
            numBatches = validateActiveCorpus();
            MESSAGEL(
                "Curriculum phase "
                << (phaseIndex + 1) << '/' << phases->size() << " [" << (*phases)[phaseIndex].name
                << "] begins at global epoch " << epoch << ", local epoch 0, train samples "
                << (trainFileCorpus ? trainFileCorpus->size() : trainTuneEntries.size())
                << ", validation samples "
                << (valFileCorpus ? valFileCorpus->size() : valTuneEntries.size()) << ", batch "
                << config.batchSize << ", cache " << config.preparedCachePath.string()
                << (config.tuneMoveScore
                        ? ", MultiPV temperature "
                              + std::to_string(config.multiPVPolicyTemperature)
                        : std::string())
                << ".");
            if (config.tuneEval && !config.usePreviousScalingFactor) {
                MESSAGEL("Recalibrating the current unquantized parameters on the new phase.");
                K                      = searchOptimalInvScalingFactor(true);
                recalibratedAtBoundary = true;
            }
        }

        if (epoch > 0) {
            const bool   localSchedule  = phases && (*phases)[phaseIndex].localLearningRateSchedule;
            const size_t scheduleEpoch  = localSchedule ? phaseEpoch + 1 : epoch;
            const size_t scheduleEpochs = localSchedule ? (*phases)[phaseIndex].epochs : epochs;
            optim.setLR(learningRateForEpoch(config, scheduleEpoch, scheduleEpochs));
            if (phaseObjectiveMasking) {
                parameterLearningRates.assign(tuneParams.size(), optim.currentLR());
            }
            if (trainFileCorpus) {
                size_t batchesProcessed = 0;
                for (size_t shard = 0; shard < trainFileCorpus->shardCount(); shard++) {
                    PreparedCorpus entries = trainFileCorpus->load(shard);
                    for (size_t batchBegin = 0; batchBegin + config.batchSize <= entries.size();
                         batchBegin += config.batchSize) {
                        updateBatch(entries, batchBegin, nullptr);
                        batchesProcessed++;
                    }
                }
                if (batchesProcessed != numBatches)
                    throw std::logic_error(
                        "prepared shard boundaries do not preserve global batches");
            }
            else {
                const std::vector<uint32_t> *order =
                    trainSampleOrder.empty() ? nullptr : &trainSampleOrder;
                for (size_t batch = 0; batch < numBatches; batch++)
                    updateBatch(trainTuneEntries, batch * config.batchSize, order);
            }
            phaseEpoch++;
        }

        // Recalibrate against the parameters updated by this epoch before
        // reporting metrics or exporting a checkpoint with the new scale.
        if (config.tuneEval && !config.usePreviousScalingFactor && epoch > 0
            && config.recomputeInterval
            && ((phases && (*phases)[phaseIndex].localRecomputeSchedule ? phaseEpoch : epoch)
                    % config.recomputeInterval
                == 0)
            && !recalibratedAtBoundary) {
            K = searchOptimalInvScalingFactor(true);
        }

        // Print out current epoch and loss
        auto [valueLoss, policyLoss] = computeLosses(K, false);
        auto     validationByBoard   = computeValidationLossesByBoard(K);
        LossPair validationTotals;
        size_t   validationSamples = 0;
        for (const auto &board : validationByBoard) {
            validationTotals.value += board.valueLoss * Float(board.samples);
            validationTotals.policy += board.policyLoss * Float(board.samples);
            validationSamples += board.samples;
        }
        Float valueValLoss =
            validationSamples != 0 ? validationTotals.value / Float(validationSamples) : Float(0);
        Float policyValLoss =
            validationSamples != 0 ? validationTotals.policy / Float(validationSamples) : Float(0);
        if (!(std::isfinite(valueLoss) && std::isfinite(policyLoss) && std::isfinite(valueValLoss)
              && std::isfinite(policyValLoss)))
            throw std::runtime_error("non-finite tuning metric");
        for (const auto &board : validationByBoard)
            if (!(std::isfinite(board.valueLoss) && std::isfinite(board.policyLoss)))
                throw std::runtime_error("non-finite per-board validation metric");
        Time elapsed = now() - startTime;
        if (valFileCorpus ? !valFileCorpus->empty() : !valTuneEntries.empty())
            MESSAGEL("Epoch " << epoch << " | LR " << optim.currentLR() << " | Value " << valueLoss
                              << " | Policy " << policyLoss << " | ValueVal " << valueValLoss
                              << " | PolicyVal " << policyValLoss << " | Time(ms) " << elapsed);
        else
            MESSAGEL("Epoch " << epoch << " | LR " << optim.currentLR() << " | Value " << valueLoss
                              << " | Policy " << policyLoss << " | Time(ms) " << elapsed);

        // Call callback after each epoch completed
        if (callback) {
            TuningStatistic stat;
            stat.currentEpoch      = epoch;
            stat.currentPhase      = phaseIndex;
            stat.currentPhaseEpoch = phaseEpoch;
            stat.phaseName         = phases ? (*phases)[phaseIndex].name : std::string();
            stat.valueLoss         = valueLoss;
            stat.policyLoss        = policyLoss;
            stat.valueValLoss      = valueValLoss;
            stat.policyValLoss     = policyValLoss;
            stat.elapsedSeconds    = double(elapsed) / 1000.0;
            stat.scalingFactor     = 1.0 / double(K);
            stat.learningRate      = optim.currentLR();
            stat.trainCompactPolicy = !phases || (*phases)[phaseIndex].trainCompactPolicy;
            stat.validationByBoard  = std::move(validationByBoard);
            callback(stat);
        }
    }

    Time totalElapsed = now() - initTime;
    MESSAGEL("Training completed in " << (totalElapsed / 1000) << " seconds.");
}

/// initParams() inits tuneParams according to their value in the live
/// Evaluation:: model tables. It also associates TuneParam index with its
/// table address. Parameters loaded from the tables will be automatically
/// saved back when Tuner is destroyed.
void Tuner::initParams()
{
    std::vector<int> ruleSetIdx;
    if (config.tuneRule[FREESTYLE])
        ruleSetIdx.push_back(FREESTYLE);
    if (config.tuneRule[STANDARD])
        ruleSetIdx.push_back(STANDARD);
    if (config.tuneRule[RENJU]) {
        ruleSetIdx.push_back(RENJU + BLACK);
        ruleSetIdx.push_back(RENJU + WHITE);
    }

    // The tuner intentionally mutates the LIVE Evaluation:: tables in place
    // through these stored addresses: evaluation during tuning always sees the
    // current candidate parameters without a copy/swap step. Evaluation ranges
    // precede policy ranges so bootstrap policy tables form one terminal
    // partition that can be coalesced independently.
    if (config.tuneEval)
        for (int r : ruleSetIdx) {
            size_t rangeBegin = tuneParams.size();
            addArrayParams<Eval>(
                "eval/basic/table-" + std::to_string(r),
                Evaluation::EVALS[r],
                [](const Eval &ev, size_t) { return TuneParam(ev); },
                [](Eval &ev, size_t, TuneParam param) {
                    constexpr Float EvalMin = Float(std::numeric_limits<Eval>::min());
                    constexpr Float EvalMax = Float(std::numeric_limits<Eval>::max());
                    ev = static_cast<Eval>(std::clamp(Float(param), EvalMin, EvalMax));
                });

            addArrayParams<Eval>(
                "eval/threat/table-" + std::to_string(r),
                Evaluation::EVALS_THREAT[r],
                [](const Eval &ev, size_t) { return TuneParam(ev); },
                [](Eval &ev, size_t, TuneParam param) {
                    constexpr Float EvalMin = Float(std::numeric_limits<Eval>::min());
                    constexpr Float EvalMax = Float(std::numeric_limits<Eval>::max());
                    ev = static_cast<Eval>(std::clamp(Float(param), EvalMin, EvalMax));
                });
            evalParamRanges.emplace_back(rangeBegin, tuneParams.size());
        }

    if (config.tuneMoveScore)
        for (int r : ruleSetIdx) {
            size_t rangeBegin = tuneParams.size();
            if (config.trainingSemantics == TrainingSemantics::Bootstrap) {
                addArrayParams<MoveScorePair, arraySize(Evaluation::P4SCORES[0]), 2>(
                    "move-score/bootstrap-table-" + std::to_string(r),
                    Evaluation::P4SCORES[r],
                    [scale = config.moveScoreScale,
                     bias  = config.moveScoreBias](const MoveScorePair &pair, size_t offset) {
                        return encodeIntegerForTruncatingExport(pair[offset], scale, bias);
                    },
                    [scoreMin = Float(config.moveScoreMin),
                     scoreMax = Float(config.moveScoreMax),
                     scale    = config.moveScoreScale,
                     bias     = config.moveScoreBias](MoveScorePair &pair,
                                                  size_t         offset,
                                                  TuneParam      param) {
                        Float score  = Float(param) * scale + bias;
                        pair[offset] = static_cast<Score>(std::clamp(score, scoreMin, scoreMax));
                    });
            }
            else {
                addTiedMoveScoreParams(
                    "move-score/tied-table-" + std::to_string(r),
                    Evaluation::P4SCORES[r],
                    arraySize(Evaluation::P4SCORES[r]),
                    [scale = config.moveScoreScale, bias = config.moveScoreBias](Score score) {
                        return encodeIntegerForTruncatingExport(score, scale, bias);
                    });
            }
            moveScoreParamRanges.emplace_back(rangeBegin, tuneParams.size());
            moveScoreTables.push_back(r);
        }

    MESSAGEL(tuneParams.size() << " parameters initialized.");
}

/// searchOptimalInvScalingFactor() searches the optimal K in
/// winRate = 1 / (1 + exp(-Eval * K)). It scans the scaling-factor space for
/// several iterations and minimizes the error of either the original static
/// evaluation or the current tuned evaluation.
Float Tuner::searchOptimalInvScalingFactor(bool useTunedEval) const
{
    assert(config.nStepsPerIteration);

    Float startK = 1.0 / Float(config.scalingFactorMin);
    Float endK   = 1.0 / Float(config.scalingFactorMax);
    Float stepK  = (endK - startK) / Float(config.nStepsPerIteration);
    Float bestK  = 0;

    for (int iter = 1; iter <= config.nIterations; iter++) {
        size_t candidateCount =
            config.nStepsPerIteration
            + (config.trainingSemantics == TrainingSemantics::Bootstrap ? 0 : 1);
        std::vector<Float> candidates(candidateCount);
        Float              k = startK;
        for (Float &candidate : candidates) {
            candidate = k;
            k += stepK;
        }
        std::vector<Float> losses   = useTunedEval ? computeEvaluationLossGrid<true>(candidates)
                                                   : computeEvaluationLossGrid<false>(candidates);
        Float              bestLoss = std::numeric_limits<Float>::max();

        for (size_t i = 0; i < candidates.size(); i++) {
            if (losses[i] < bestLoss) {
                bestLoss = losses[i];
                bestK    = candidates[i];
            }
        }

        MESSAGEL("Iteration " << iter << " | K " << bestK << " | Loss " << bestLoss);

        startK = bestK - stepK;
        stepK  = stepK * 2 / Float(config.nStepsPerIteration);
    }

    MESSAGEL("Optimal inv scaling factor K = " << bestK << " after " << config.nIterations
                                               << " iteration.");

    return bestK;
}

template <bool UseTunedEval>
std::vector<Float> Tuner::computeEvaluationLossGrid(const std::vector<Float> &candidates) const
{
    std::vector<Float> total(candidates.size(), Float(0));
    auto accumulateEntries = [this, &candidates, &total](const PreparedCorpus &entries,
                                                         size_t                logicalSamples,
                                                         bool                  applyShuffleOrder) {
        if (logicalSamples == 0)
            return;
        size_t numBlocks = std::min(logicalSamples, LogicalPartitions);
        size_t blockSize = (logicalSamples + numBlocks - 1) / numBlocks;
        std::vector<std::future<std::vector<Float>>> futures;
        futures.reserve(numBlocks);
        for (size_t block = 0; block < numBlocks; block++) {
            size_t blockBegin = block * blockSize;
            size_t blockEnd   = std::min(blockBegin + blockSize, logicalSamples);
            if (blockBegin == blockEnd)
                break;
            futures.emplace_back(threadPool.submit_task(
                [this, &entries, &candidates, applyShuffleOrder, blockBegin, blockEnd] {
                    std::vector<Float> blockLosses(candidates.size(), Float(0));
                    for (size_t logicalSample = blockBegin; logicalSample < blockEnd;
                         logicalSample++) {
                        size_t  sample    = applyShuffleOrder && !trainSampleOrder.empty()
                                                ? trainSampleOrder[logicalSample]
                                                : logicalSample;
                        uint8_t boardSize = entries.boardSizes()[sample];
                        if (boardSize == 0 || boardSize > MAX_BOARD_SIZE)
                            throw std::runtime_error(
                                "prepared sample board size exceeds engine limit");
                        Float sampleWeight = trainBoardSampleWeights[boardSize];
                        for (size_t candidate = 0; candidate < candidates.size(); candidate++)
                            blockLosses[candidate] +=
                                ::computeEvalLoss<UseTunedEval>(entries,
                                                                sample,
                                                                tuneParams,
                                                                candidates[candidate],
                                                                config.lossType)
                                * sampleWeight;
                    }
                    return blockLosses;
                }));
        }
        for (auto &future : futures) {
            std::vector<Float> blockLosses = future.get();
            for (size_t candidate = 0; candidate < total.size(); candidate++)
                total[candidate] += blockLosses[candidate];
        }
    };

    size_t sampleCount = 0;
    if (trainFileCorpus) {
        for (size_t shard = 0; shard < trainFileCorpus->shardCount(); shard++) {
            PreparedCorpus entries       = trainFileCorpus->load(shard);
            size_t         usableSamples = entries.size() / config.batchSize * config.batchSize;
            sampleCount += usableSamples;
            accumulateEntries(entries, usableSamples, false);
        }
    }
    else {
        sampleCount = trainTuneEntries.size() / config.batchSize * config.batchSize;
        accumulateEntries(trainTuneEntries, sampleCount, true);
    }
    if (sampleCount == 0)
        return total;
    for (Float &loss : total)
        loss /= Float(sampleCount);
    return total;
}

/// computeEvaluationLoss() computes loss between the current tuned/static
/// evaluation and target win rate in all tune entries using the given K.
template <bool UseTunedEval>
Float Tuner::computeEvaluationLoss(Float K, bool validation) const
{
    if (!config.tuneEval)
        return Float(0.0);

    const FileBackedCorpus *fileCorpus = validation ? valFileCorpus.get() : trainFileCorpus.get();
    if (fileCorpus) {
        if (fileCorpus->empty())
            return Float(0.0);

        Float total = 0;
        for (size_t shard = 0; shard < fileCorpus->shardCount(); shard++) {
            PreparedCorpus entries = fileCorpus->load(shard);
            total += parallelIndexReduce<Float>(threadPool,
                                                entries.size(),
                                                Float(0.0),
                                                [this, &entries, K](size_t sample) {
                                                    return ::computeEvalLoss<UseTunedEval>(
                                                        entries,
                                                        sample,
                                                        tuneParams,
                                                        K,
                                                        config.lossType);
                                                });
        }
        return total / Float(fileCorpus->size());
    }

    const PreparedCorpus &entries = validation ? valTuneEntries : trainTuneEntries;

    if (entries.empty())
        return Float(0.0);

    return parallelIndexReduce<Float>(threadPool,
                                      entries.size(),
                                      Float(0.0),
                                      [this, &entries, K, validation](size_t logicalSample) {
                                          size_t sample = !validation && !trainSampleOrder.empty()
                                                              ? trainSampleOrder[logicalSample]
                                                              : logicalSample;
                                          return ::computeEvalLoss<UseTunedEval>(entries,
                                                                                 sample,
                                                                                 tuneParams,
                                                                                 K,
                                                                                 config.lossType);
                                      })
           / Float(entries.size());
}

/// computeMoveScoreLoss() computes loss of current move scores between
/// the target best move in all tune entries.
Float Tuner::computeMoveScoreLoss(bool validation) const
{
    if (!config.tuneMoveScore)
        return Float(0.0);

    const MoveScoreLossSettings settings = makeMoveScoreLossSettings(config);
    const FileBackedCorpus *fileCorpus   = validation ? valFileCorpus.get() : trainFileCorpus.get();
    if (fileCorpus) {
        if (fileCorpus->empty())
            return Float(0.0);

        Float total = 0;
        for (size_t shard = 0; shard < fileCorpus->shardCount(); shard++) {
            PreparedCorpus entries = fileCorpus->load(shard);
            total += parallelIndexReduce<Float>(
                threadPool,
                entries.size(),
                Float(0.0),
                [this, &entries, &settings](size_t sample) {
                    return detail::computeMoveScoreLoss(entries, sample, tuneParams, settings);
                });
        }
        return total / Float(fileCorpus->size());
    }

    const PreparedCorpus &entries = validation ? valTuneEntries : trainTuneEntries;

    if (entries.empty())
        return Float(0.0);

    return parallelIndexReduce<Float>(
               threadPool,
               entries.size(),
               Float(0.0),
               [this, &entries, &settings, validation](size_t logicalSample) {
                   size_t sample = !validation && !trainSampleOrder.empty()
                                       ? trainSampleOrder[logicalSample]
                                       : logicalSample;
                   return detail::computeMoveScoreLoss(entries, sample, tuneParams, settings);
               })
           / Float(entries.size());
}

std::pair<Float, Float> Tuner::computeLosses(Float K, bool validation) const
{
    const FileBackedCorpus *fileCorpus   = validation ? valFileCorpus.get() : trainFileCorpus.get();
    const MoveScoreLossSettings settings = makeMoveScoreLossSettings(config);
    auto sampleLoss = [this, K, &settings](const PreparedCorpus &entries, size_t sample) {
        return LossPair {
            config.tuneEval
                ? ::computeEvalLoss<true>(entries, sample, tuneParams, K, config.lossType)
                : Float(0),
            config.tuneMoveScore
                ? detail::computeMoveScoreLoss(entries, sample, tuneParams, settings)
                : Float(0),
        };
    };

    if (fileCorpus) {
        if (fileCorpus->empty())
            return {0, 0};
        LossPair total;
        for (size_t shard = 0; shard < fileCorpus->shardCount(); shard++) {
            PreparedCorpus entries = fileCorpus->load(shard);
            total += parallelIndexReduce<LossPair>(
                threadPool,
                entries.size(),
                LossPair {},
                [&entries, &sampleLoss](size_t sample) { return sampleLoss(entries, sample); });
        }
        return {total.value / Float(fileCorpus->size()), total.policy / Float(fileCorpus->size())};
    }

    const PreparedCorpus &entries = validation ? valTuneEntries : trainTuneEntries;
    if (entries.empty())
        return {0, 0};
    LossPair total = parallelIndexReduce<LossPair>(
        threadPool,
        entries.size(),
        LossPair {},
        [this, validation, &entries, &sampleLoss](size_t logicalSample) {
            size_t sample = !validation && !trainSampleOrder.empty()
                                ? trainSampleOrder[logicalSample]
                                : logicalSample;
            return sampleLoss(entries, sample);
        });
    return {total.value / Float(entries.size()), total.policy / Float(entries.size())};
}

std::vector<TuningStatistic::BoardValidationLoss>
Tuner::computeValidationLossesByBoard(Float K) const
{
    const MoveScoreLossSettings settings = makeMoveScoreLossSettings(config);
    auto reduceEntries                   = [this, K, &settings](const PreparedCorpus &entries) {
        size_t numBlocks = std::min(entries.size(), LogicalPartitions);
        if (numBlocks == 0)
            return BoardLossTotals {};
        size_t blockSize = (entries.size() + numBlocks - 1) / numBlocks;
        std::vector<std::future<BoardLossTotals>> futures;
        futures.reserve(numBlocks);
        for (size_t block = 0; block < numBlocks; block++) {
            size_t blockBegin = block * blockSize;
            size_t blockEnd   = std::min(blockBegin + blockSize, entries.size());
            if (blockBegin == blockEnd)
                break;
            futures.emplace_back(
                threadPool.submit_task([this, K, blockBegin, blockEnd, &entries, &settings] {
                    BoardLossTotals total;
                    for (size_t sample = blockBegin; sample < blockEnd; sample++) {
                        uint8_t boardSize = entries.boardSizes()[sample];
                        if (boardSize == 0 || boardSize > MAX_BOARD_SIZE)
                            throw std::runtime_error(
                                "prepared sample board size exceeds engine limit");
                        total.samples[boardSize]++;
                        if (config.tuneEval)
                            total.losses[boardSize].value +=
                                ::computeEvalLoss<true>(entries,
                                                        sample,
                                                        tuneParams,
                                                        K,
                                                        config.lossType);
                        if (config.tuneMoveScore)
                            total.losses[boardSize].policy +=
                                detail::computeMoveScoreLoss(entries, sample, tuneParams, settings);
                    }
                    return total;
                }));
        }
        BoardLossTotals total;
        for (auto &future : futures)
            total += future.get();
        return total;
    };

    BoardLossTotals totals;
    if (valFileCorpus) {
        for (size_t shard = 0; shard < valFileCorpus->shardCount(); shard++) {
            PreparedCorpus entries = valFileCorpus->load(shard);
            totals += reduceEntries(entries);
        }
    }
    else
        totals = reduceEntries(valTuneEntries);

    std::vector<TuningStatistic::BoardValidationLoss> losses;
    for (size_t boardSize = 1; boardSize < totals.samples.size(); boardSize++) {
        size_t samples = totals.samples[boardSize];
        if (samples != validationBoardSampleCounts[boardSize])
            throw std::logic_error("per-board validation sample count changed during training");
        if (samples == 0)
            continue;
        losses.push_back({static_cast<uint8_t>(boardSize),
                          samples,
                          totals.losses[boardSize].value / Float(samples),
                          totals.losses[boardSize].policy / Float(samples)});
    }
    return losses;
}

/// computeGradients() computes gradients of all parameters used in one tune
/// entries batch and accumulates them into gradients vector. These gradients
/// then will be used to tune the parameters with a gradient descent optimizer.
void Tuner::computeGradientBatch(std::vector<TuneGradient>   &grads,
                                 Float                        K,
                                 const PreparedCorpus        &entries,
                                 size_t                       batchBegin,
                                 const std::vector<uint32_t> *sampleOrder)
{
    assert(grads.size() == tuneParams.size());
    const size_t numJobs     = std::min(LogicalPartitions, config.batchSize);
    const size_t baseJobSize = config.batchSize / numJobs;
    const size_t remainder   = config.batchSize % numJobs;

    if (partitionGradients.size() != numJobs)
        partitionGradients.assign(numJobs, std::vector<TuneGradient>(tuneParams.size()));

    std::vector<std::future<void>> gradJobs;
    gradJobs.reserve(numJobs);
    const MoveScoreLossSettings settings = makeMoveScoreLossSettings(config);
    for (size_t jobIdx = 0; jobIdx < numJobs; jobIdx++) {
        // Get range of tune entries for this job
        size_t jobOffset = jobIdx * baseJobSize + std::min(jobIdx, remainder);
        size_t jobSize   = baseJobSize + (jobIdx < remainder);
        size_t jobBegin  = batchBegin + jobOffset;
        size_t jobEnd    = jobBegin + jobSize;

        // Accumulate local gradient asynchronously
        auto job = threadPool.submit_task(
            [this, K, jobIdx, jobBegin, jobEnd, &entries, sampleOrder, settings] {
                std::vector<TuneGradient> &localGrads = partitionGradients[jobIdx];
                std::fill(localGrads.begin(), localGrads.end(), TuneGradient(0));

                for (size_t logicalSample = jobBegin; logicalSample < jobEnd; logicalSample++) {
                    size_t  sample    = sampleOrder ? (*sampleOrder)[logicalSample] : logicalSample;
                    uint8_t boardSize = entries.boardSizes()[sample];
                    if (boardSize == 0 || boardSize > MAX_BOARD_SIZE)
                        throw std::runtime_error("prepared sample board size exceeds engine limit");
                    Float sampleWeight = trainBoardSampleWeights[boardSize];
                    ::computeEvalGradient(entries,
                                          sample,
                                          localGrads,
                                          tuneParams,
                                          K,
                                          config.lossType,
                                          sampleWeight);
                    detail::computeMoveScoreGradient(entries,
                                                     sample,
                                                     localGrads,
                                                     tuneParams,
                                                     settings,
                                                     sampleWeight);
                }

                // Scale gradient according to batch size
                TuneGradient scale = TuneGradient(1 / Float(config.batchSize));
                for (TuneGradient &gradient : localGrads)
                    gradient *= scale;
            });
        gradJobs.push_back(std::move(job));
    }

    for (auto &job : gradJobs)
        job.get();

    // Reduce in logical partition order, independent of physical scheduling.
    for (const std::vector<TuneGradient> &localGrads : partitionGradients) {
        assert(grads.size() == localGrads.size());

        for (size_t i = 0; i < grads.size(); i++)
            grads[i] += localGrads[i];
    }

    for (size_t i = 0; i < grads.size(); i++)
        if (!trainableParams[i])
            grads[i] = 0;
}

/// addParams() adds a continous range of params in config to tuneParams
void Tuner::addParams(std::string   layoutTag,
                      void         *address,
                      size_t        numElems,
                      uint32_t      elemSize,
                      uint32_t      paramPerElem,
                      ParamGetter<> getter,
                      ParamSetter<> setter,
                      ParamStager   stager)
{
    assert(paramPerElem > 0);
    if (layoutTag.empty())
        throw std::invalid_argument("tuning parameter layout tag must not be empty");
    auto duplicateLayout = [&](const auto &record) { return record.layoutTag == layoutTag; };
    if (std::any_of(syncRecords.begin(), syncRecords.end(), duplicateLayout)
        || std::any_of(tiedMoveScoreSyncRecords.begin(),
                       tiedMoveScoreSyncRecords.end(),
                       duplicateLayout))
        throw std::logic_error("duplicate tuning parameter layout tag: " + layoutTag);
    size_t baseIndex = tuneParams.size();

    // Init parameters from getter and add them to tuneParams
    if (numElems > std::numeric_limits<size_t>::max() / paramPerElem)
        throw std::length_error("tuning parameter count overflows size_t");
    size_t           numParams         = numElems * paramPerElem;
    constexpr size_t MaxParameterCount = size_t(std::numeric_limits<ParameterId>::max()) + 1;
    if (numParams > MaxParameterCount - baseIndex)
        throw std::length_error("tuning parameter count exceeds ParameterId capacity");

    syncRecords.push_back(ParamsSyncRecord {std::move(layoutTag),
                                            baseIndex,
                                            numElems,
                                            elemSize,
                                            paramPerElem,
                                            address,
                                            std::move(getter),
                                            std::move(setter),
                                            std::move(stager)});
    const ParamsSyncRecord &record = syncRecords.back();

    tuneParams.reserve(baseIndex + numParams);
    for (size_t i = 0; i < numElems; i++) {
        size_t      elementBase = baseIndex + i * paramPerElem;
        const void *address     = record[i];
        auto [it, inserted]     = paramIndices.emplace(
            address,
            ParameterAddress {static_cast<ParameterId>(elementBase), paramPerElem});
        if (!inserted || tiedMoveScoreParamIndices.find(address) != tiedMoveScoreParamIndices.end())
            throw std::logic_error("duplicate tuning parameter address");

        for (size_t j = 0; j < paramPerElem; j++) {
            TuneParam param = record.getter(record[i], j);
            if (!std::isfinite(param))
                throw std::runtime_error("non-finite initial tuning parameter");
            tuneParams.emplace_back(param);
        }
    }
}

void Tuner::addTiedMoveScoreParams(std::string                            layoutTag,
                                   MoveScorePair                         *scores,
                                   size_t                                 count,
                                   const std::function<TuneParam(Score)> &initializer)
{
    if (layoutTag.empty())
        throw std::invalid_argument("tied move-score layout tag must not be empty");
    if (!scores || count == 0 || !initializer)
        throw std::invalid_argument("tied move-score parameters require a nonempty table");
    auto duplicateLayout = [&](const auto &record) { return record.layoutTag == layoutTag; };
    if (std::any_of(syncRecords.begin(), syncRecords.end(), duplicateLayout)
        || std::any_of(tiedMoveScoreSyncRecords.begin(),
                       tiedMoveScoreSyncRecords.end(),
                       duplicateLayout))
        throw std::logic_error("duplicate tuning parameter layout tag: " + layoutTag);

    TiedMoveScoreSyncRecord record;
    record.layoutTag = std::move(layoutTag);
    record.scores    = scores;
    record.parameterIndices.resize(count);
    std::array<std::unordered_map<Score, ParameterId>, 2> groups;
    constexpr size_t MaxParameterCount = size_t(std::numeric_limits<ParameterId>::max()) + 1;
    for (size_t side = 0; side < 2; side++) {
        for (size_t i = 0; i < count; i++) {
            Score score = scores[i][side];
            auto  found = groups[side].find(score);
            if (found == groups[side].end()) {
                if (tuneParams.size() >= MaxParameterCount)
                    throw std::length_error("tuning parameter count exceeds ParameterId capacity");
                TuneParam param = initializer(score);
                if (!std::isfinite(param))
                    throw std::runtime_error("non-finite initial tied move-score parameter");
                ParameterId index = static_cast<ParameterId>(tuneParams.size());
                tuneParams.push_back(param);
                found = groups[side].emplace(score, index).first;
            }
            record.parameterIndices[i][side] = found->second;
        }
    }

    for (size_t i = 0; i < count; i++) {
        const void *address = &scores[i];
        if (paramIndices.find(address) != paramIndices.end()
            || !tiedMoveScoreParamIndices.emplace(address, record.parameterIndices[i]).second)
            throw std::logic_error("duplicate tuning parameter address");
    }
    tiedMoveScoreSyncRecords.push_back(std::move(record));
}

/// paramIndex() finds tuneParams index according to address of its config value
ParameterId Tuner::paramIndex(const void *addr, size_t offset) const
{
    auto tied = tiedMoveScoreParamIndices.find(addr);
    if (tied != tiedMoveScoreParamIndices.end()) {
        if (offset >= tied->second.size())
            throw std::out_of_range("tied move-score parameter offset is out of range");
        return tied->second[offset];
    }

    auto it = paramIndices.find(addr);
    if (it == paramIndices.end())
        throw std::logic_error("unknown tuning parameter address");

    if (offset >= it->second.parameterCount)
        throw std::out_of_range("tuning parameter offset is out of range");
    size_t index = size_t(it->second.baseIndex) + offset;
    if (index >= tuneParams.size())
        throw std::out_of_range("tuning parameter offset is out of range");
    return static_cast<ParameterId>(index);
}

}  // namespace Tuning
