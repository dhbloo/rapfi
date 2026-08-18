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

#include "../core/pos.h"
#include "../core/types.h"
#include "featureencoder.h"
#include "tunecorpus.h"
#include "tunestore.h"

#include <BS_thread_pool.hpp>
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <utility>
#include <vector>

class Board;
struct MoveScorePair;

namespace Tuning {

template <typename T>
class AdamOptimizer;

class Dataset;

using Float        = double;  // stable scalar intermediates and global reductions
using TuneParam    = float;   // high-cardinality parameter and optimizer state
using TuneGradient = float;   // persistent/per-partition gradient storage

/// LossType represents a type of loss function to use.
enum class LossType { L1, L2, BCE };

/// Parameterization and target interpretation used by a training corpus.
/// Bootstrap preserves the master-era joint value/policy geometry; new
/// training should normally use Current.
enum class TrainingSemantics {
    Current,
    Bootstrap,
};

/// Relative contribution assigned to samples from each board size.
enum class BoardSizeWeighting {
    /// Every accepted position has equal weight.
    SampleFrequency,
    /// Every board size present in the optimized training subset has equal total weight.
    EqualBoard,
};

enum class LearningRateSchedule {
    Constant,
    Exponential,
};

/// ParamGetter gets a TuneParam from a raw parameter given by an address and an offset.
template <typename AddrType = void *>
using ParamGetter = std::function<TuneParam(AddrType, size_t)>;
/// ParamSetter sets a raw parameter given by an address and an offset to a TuneParam.
template <typename AddrType = void *>
using ParamSetter = std::function<void(AddrType, size_t, TuneParam)>;

struct StagedParams
{
    std::vector<TuneParam> params;
    std::vector<uint8_t>   objectBytes;
};
using ParamStager = std::function<StagedParams(const void *, const std::vector<TuneParam> &)>;

/// ParamsRecord struct is a helper to sync tune parameters with the config.
/// It also maps a range of addresses that contains the tunable parameters
/// to its base index in the tuneParams array.
struct ParamsSyncRecord
{
    std::string layoutTag;
    size_t      baseIndex;
    size_t      numElems;
    uint32_t    elemSize;
    uint32_t    paramPerElem;
    void       *address;

    ParamGetter<> getter;
    ParamSetter<> setter;
    ParamStager   stager;

    void *operator[](size_t i) const { return static_cast<char *>(address) + elemSize * i; }
};

/// TuningConfig struct records all configuration settings used in Tuner
struct TuningConfig
{
    // --------------------------------------------
    // General training settings

    size_t                             batchSize      = 8192;
    size_t                             maxTuneEntries = UINT32_MAX;
    size_t                             numThreads     = 0;
    uint64_t                           seed           = 1;
    size_t                             memoryLimitMB  = 0;
    size_t                             shardSizeMB    = 64;
    std::filesystem::path              preparedCachePath;
    std::vector<std::filesystem::path> trainDatasetPaths;
    std::vector<std::filesystem::path> validationDatasetPaths;
    std::string                        trainDatasetFormat;
    std::string                        validationDatasetFormat;
    bool                               rebuildPreparedCache  = false;
    BoardSizeWeighting                 boardSizeWeighting    = BoardSizeWeighting::SampleFrequency;
    double                             learningRate          = 0.01;
    double                             finalLearningRate     = 0.0;
    LearningRateSchedule               learningRateSchedule  = LearningRateSchedule::Constant;
    double                             weightDecay                  = 0.0;
    double                             moveScoreLossGamma           = 0.0;
    double                             multiPVPolicyTemperature     = 0.0;
    double                             multiPVPolicyEvalScale       = 0.0;
    double                             moveScoreScale               = 24.0;
    double                             moveScoreBias                = 24.0;
    Score                              moveScoreMin                 = -999;
    Score                              moveScoreMax                 = 999;
    bool                               projectMoveScoreScale        = false;
    double                             moveScoreReferenceDispersion = 0.0;
    LossType                           lossType                     = LossType::BCE;
    bool                               shuffleTuneEntries           = false;
    bool                               tuneEval                     = true;
    bool                               tuneMoveScore                = false;
    TrainingSemantics                  trainingSemantics            = TrainingSemantics::Current;
    // --------------------------------------------
    // Data entry filter settings

    bool     tuneRule[RULE_NB] = {};
    uint8_t  boardSizeMin      = 5;
    uint8_t  boardSizeMax      = MAX_BOARD_SIZE;
    uint16_t minPly            = 1;
    uint16_t minPlyBeforeFull  = 50;

    // --------------------------------------------
    // Scaling Factor searching settings

    bool   usePreviousScalingFactor = false;
    int    nIterations              = 10;
    int    nStepsPerIteration       = 10;
    double scalingFactorMin         = 100;
    double scalingFactorMax         = 400;
    size_t recomputeInterval        = 0;
};

/// One ordered corpus segment in a multi-dataset training curriculum.
/// Corpus identity, active objectives, and local optimizer schedule may vary
/// between phases while one model and optimizer container remain live.
struct TuningPhase
{
    std::string                        name;
    Dataset                           *trainingDataset   = nullptr;
    Dataset                           *validationDataset = nullptr;
    size_t                             epochs            = 0;
    std::filesystem::path              preparedCachePath;
    std::vector<std::filesystem::path> trainDatasetPaths;
    std::vector<std::filesystem::path> validationDatasetPaths;
    std::string                        trainDatasetFormat;
    std::string                        validationDatasetFormat;
    bool                               rebuildPreparedCache = false;
    uint8_t                            boardSizeMin         = 5;
    uint8_t                            boardSizeMax         = MAX_BOARD_SIZE;
    std::optional<size_t>              batchSize;
    std::optional<uint64_t>            seed;
    std::optional<double>              multiPVPolicyTemperature;
    std::optional<double>              multiPVPolicyEvalScale;
    std::optional<TrainingSemantics>   trainingSemantics;
    bool                               tuneEval                  = true;
    bool                               tuneMoveScore             = false;
    bool                               trainCompactPolicy        = false;
    double                             learningRate              = 0.01;
    double                             finalLearningRate         = 0.0;
    double                             weightDecay               = 0.0;
    LearningRateSchedule               learningRateSchedule      = LearningRateSchedule::Constant;
    size_t                             recomputeInterval         = 0;
    bool                               localLearningRateSchedule = false;
    bool                               localRecomputeSchedule    = false;
    bool                               projectMoveScoreAtEnd     = false;
    bool                               coalesceMoveScoreAtEnd    = false;
    BoardSizeWeighting                 boardSizeWeighting = BoardSizeWeighting::SampleFrequency;
};

/// TuningStatistic struct records all current statistic in tuning process.
/// This can be used to produce a training record for reporting.
struct TuningStatistic
{
    struct BoardValidationLoss
    {
        uint8_t boardSize;
        size_t  samples;
        double  valueLoss;
        double  policyLoss;
    };

    size_t                           currentEpoch;
    size_t                           currentPhase;
    size_t                           currentPhaseEpoch;
    std::string                      phaseName;
    double                           valueLoss, policyLoss;
    double                           valueValLoss, policyValLoss;
    double                           elapsedSeconds;
    double                           scalingFactor;
    double                           learningRate;
    bool                             trainCompactPolicy;
    std::vector<BoardValidationLoss> validationByBoard;
};

struct BoardSampleStatistic
{
    uint8_t boardSize;
    size_t  trainingSamples;
    size_t  optimizedTrainingSamples;
    size_t  validationSamples;
    double  trainingWeight;
};

/// One immutable opening used to measure the loaded and exported policy scale.
/// Every non-empty prefix is included in the quiet-move dispersion reference.
struct MoveScoreReferencePosition
{
    Rule             rule;
    int              boardSize;
    std::vector<Pos> moves;
};

/// State used by the post-training quiet-score projection.
struct MoveScoreScaleReport
{
    bool                                           projectionEnabled = false;
    std::array<bool, RULE_NB + 1>                  projectedTables          = {};
    std::array<size_t, RULE_NB + 1>                referenceLists           = {};
    std::array<size_t, RULE_NB + 1>                referenceObservations    = {};
    std::array<double, RULE_NB + 1>                loadedDispersion         = {};
    std::array<double, RULE_NB + 1>                referenceDispersion      = {};
    std::array<double, RULE_NB + 1>                unprojectedDispersion    = {};
    std::array<double, RULE_NB + 1>                projectionFactors        = {1, 1, 1, 1};
    std::array<double, RULE_NB + 1>                exportedDispersion       = {};
    std::array<std::array<double, 2>, RULE_NB + 1> projectionCenters        = {};
};

/// Tuner runs the whole tuning process for the given dataset and tuning config.
/// Some actions in tuning process will be performed in parallel.
class Tuner
{
public:
    Tuner(class Dataset                          &trainDataset,
          class Dataset                          *valDataset,
          TuningConfig                            config                      = {},
          std::vector<MoveScoreReferencePosition> moveScoreReferencePositions = {});
    Tuner(std::vector<TuningPhase>                phases,
          TuningConfig                            config,
          std::vector<MoveScoreReferencePosition> moveScoreReferencePositions = {});
    Tuner(const Tuner &) = delete;

    void run(size_t epochs, std::function<void(TuningStatistic)> callback = nullptr);
    void runCurriculum(std::function<void(TuningStatistic)> callback = nullptr);
    void saveParams() const;
    std::vector<BoardSampleStatistic> boardSampleStatistics() const;
    std::vector<BoardSampleStatistic> boardSampleStatistics(size_t phaseIndex) const;

private:
    struct DeferCorpusPreparation
    {};
    Tuner(class Dataset                          &trainDataset,
          class Dataset                          *valDataset,
          TuningConfig                            config,
          std::vector<MoveScoreReferencePosition> moveScoreReferencePositions,
          DeferCorpusPreparation);

    static constexpr size_t LogicalPartitions = 64;

    TuningConfig                      config;
    PolicyTargetConfig                policyTargetConfig;
    PreparedCorpus                    trainTuneEntries, valTuneEntries;
    std::unique_ptr<FileBackedCorpus> trainFileCorpus, valFileCorpus;
    std::vector<uint32_t>             trainSampleOrder;
    std::vector<TuneParam>            tuneParams;
    std::vector<ParamsSyncRecord>     syncRecords;
    struct TiedMoveScoreSyncRecord
    {
        std::string                             layoutTag;
        MoveScorePair                          *scores;
        std::vector<std::array<ParameterId, 2>> parameterIndices;
    };
    std::vector<TiedMoveScoreSyncRecord>    tiedMoveScoreSyncRecords;
    std::vector<std::pair<size_t, size_t>>  moveScoreParamRanges;
    std::vector<int>                        moveScoreTables;
    std::vector<std::pair<size_t, size_t>>  evalParamRanges;
    std::vector<MoveScoreReferencePosition> moveScoreReferencePositions;
    MoveScoreScaleReport                    moveScoreReference;
    struct ParameterAddress
    {
        ParameterId baseIndex;
        uint32_t    parameterCount;
    };
    std::unordered_map<const void *, ParameterAddress>           paramIndices;
    std::unordered_map<const void *, std::array<ParameterId, 2>> tiedMoveScoreParamIndices;
    struct CorpusState
    {
        PreparedCorpus                         trainTuneEntries, valTuneEntries;
        std::unique_ptr<FileBackedCorpus>      trainFileCorpus, valFileCorpus;
        std::vector<uint32_t>                  trainSampleOrder;
        std::array<size_t, MAX_BOARD_SIZE + 1> trainBoardSampleCounts       = {};
        std::array<size_t, MAX_BOARD_SIZE + 1> trainBoardOptimizationCounts = {};
        std::array<size_t, MAX_BOARD_SIZE + 1> validationBoardSampleCounts  = {};
        std::array<Float, MAX_BOARD_SIZE + 1>  trainBoardSampleWeights      = {};
        bool                                   prepared                     = false;
    };
    std::vector<TuningPhase>               curriculumPhases;
    TuningConfig                           curriculumBaseConfig;
    std::vector<CorpusState>               inactiveCorpusStates;
    size_t                                 currentPhaseIndex = 0;
    std::vector<std::vector<TuneGradient>> partitionGradients;
    std::vector<uint8_t>                   trainableParams;
    bool                                   phaseObjectiveMasking        = false;
    std::array<size_t, MAX_BOARD_SIZE + 1> trainBoardSampleCounts       = {};
    std::array<size_t, MAX_BOARD_SIZE + 1> trainBoardOptimizationCounts = {};
    std::array<size_t, MAX_BOARD_SIZE + 1> validationBoardSampleCounts  = {};
    std::array<Float, MAX_BOARD_SIZE + 1>  trainBoardSampleWeights      = {};
    size_t                                 fileWorkerBudgetBytes        = 0;
    size_t                                 fileJobBudgetBytes           = 0;
    size_t                                 fileShardBudgetBytes         = 0;
    size_t                                 fileShardTargetBytes         = 0;
    size_t                                 fileRecordLimitBytes         = 0;
    size_t                                 fileChunkEntryLimit          = 0;
    size_t                                 fileMaxPendingJobs           = 0;
    /// Worker pool for dataset transformation and loss/gradient computation.
    /// mutable: the const loss-computation methods submit tasks through it.
    mutable BS::thread_pool threadPool;

    struct CompileScratch
    {
        std::vector<TuneCoeff>        evalTerms;
        std::vector<PolicyCandidate>  policyCandidates;
        std::vector<PolicyTargetTerm> policyTargets;
        std::vector<uint16_t>         policyTarget;
        FeatureEncodeScratch          featureEncoder;
    };

    void             initParams();
    void             prepareCorpus(Dataset &trainDataset, Dataset *valDataset);
    void             applyPhaseConfig(const TuningPhase &phase);
    void             updateTrainableParams(bool allowNoTraditionalParams = false);
    void             synchronizeProjectedMoveScores();
    void             coalesceMoveScoreLayout(AdamOptimizer<TuneParam> &optimizer);
    void             storeActiveCorpus(size_t phaseIndex);
    void             loadActiveCorpus(size_t phaseIndex);
    void             runImpl(size_t                               epochs,
                             const std::vector<TuningPhase>      *phases,
                             std::function<void(TuningStatistic)> callback);
    PreparedCacheKey makePreparedCacheKey(const std::vector<std::filesystem::path> &sourcePaths,
                                          const std::string                        &datasetFormat,
                                          const char                               *role) const;
    void             initTuneEntries(PreparedCorpus   &tuneEntries,
                                     FileBackedCorpus *fileCorpus,
                                     class Dataset    &dataset,
                                     bool              buildShuffleOrder);
    void             appendTuneSample(PreparedCorpus    &tuneEntries,
                                      const Board       &board,
                                      Rule               rule,
                                      uint8_t            resultTimesTwo,
                                      Pos                bestMove,
                                      Eval               bestEval,
                                      const MovePayload &payload,
                                      CompileScratch    &scratch) const;
    Float            searchOptimalInvScalingFactor(bool useTunedEval) const;
    template <bool UseTunedEval>
    std::vector<Float> computeEvaluationLossGrid(const std::vector<Float> &candidates) const;
    template <bool UseTunedEval = true>
    Float                   computeEvaluationLoss(Float K, bool validation) const;
    Float                   computeMoveScoreLoss(bool validation) const;
    std::pair<Float, Float> computeLosses(Float K, bool validation) const;
    std::vector<TuningStatistic::BoardValidationLoss> computeValidationLossesByBoard(Float K) const;
    std::array<size_t, MAX_BOARD_SIZE + 1>            computeTrainingBoardSampleCounts(
                   std::array<size_t, MAX_BOARD_SIZE + 1> &optimizationCounts) const;
    std::array<size_t, MAX_BOARD_SIZE + 1> computeValidationBoardSampleCounts() const;
    Score                                  decodeMoveScoreParam(TuneParam param) const;
    MoveScoreScaleReport                   projectMoveScoreScale() const;
    void                                   computeGradientBatch(std::vector<TuneGradient>   &grads,
                                                                Float                        K,
                                                                const PreparedCorpus        &entries,
                                                                size_t                       batchBegin,
                                                                const std::vector<uint32_t> *sampleOrder);

    void        addParams(std::string   layoutTag,
                          void         *address,
                          size_t        numElems,
                          uint32_t      elemSize,
                          uint32_t      paramPerElem,
                          ParamGetter<> getter,
                          ParamSetter<> setter,
                          ParamStager   stager);
    void        addTiedMoveScoreParams(std::string                            layoutTag,
                                       MoveScorePair                         *scores,
                                       size_t                                 count,
                                       const std::function<TuneParam(Score)> &initializer);
    ParameterId paramIndex(const void *address, size_t offset = 0) const;

    /* helper functions to add typed params to synced tune params */

    template <typename T, size_t ParamPerElem = 1>
    void addSingleParam(std::string            layoutTag,
                        T                     &param,
                        ParamGetter<const T &> getter,
                        ParamSetter<T &>       setter);
    template <typename T, size_t Length, size_t ParamPerElem = 1>
    void addArrayParams(std::string layoutTag,
                        T (&paramArray)[Length],
                        ParamGetter<const T &> getter,
                        ParamSetter<T &>       setter);
};

}  // namespace Tuning

template <typename T, size_t ParamPerElem>
inline void Tuning::Tuner::addSingleParam(std::string            layoutTag,
                                          T                     &param,
                                          ParamGetter<const T &> getter,
                                          ParamSetter<T &>       setter)
{
    static_assert(std::is_trivially_copyable<T>::value,
                  "tuning parameter elements must be trivially copyable");
    auto stagerGetter = getter;
    auto stagerSetter = setter;
    addParams(
        std::move(layoutTag),
        &param,  // std::addressof() might be better
        1,
        sizeof(T),
        ParamPerElem,
        [getter = std::move(getter)](void *addr, size_t offset) -> TuneParam {
            return getter(*static_cast<const T *>(addr), offset);
        },
        [setter = std::move(setter)](void *addr, size_t offset, TuneParam param) -> void {
            setter(*static_cast<T *>(addr), offset, param);
        },
        [getter = std::move(stagerGetter),
         setter = std::move(stagerSetter)](const void                   *addr,
                                           const std::vector<TuneParam> &params) -> StagedParams {
            if (params.size() != ParamPerElem)
                throw std::invalid_argument("staged tuning parameter count mismatch");
            T staged = *static_cast<const T *>(addr);
            for (size_t offset = 0; offset < ParamPerElem; offset++)
                setter(staged, offset, params[offset]);
            StagedParams result;
            result.params.reserve(ParamPerElem);
            for (size_t offset = 0; offset < ParamPerElem; offset++)
                result.params.push_back(getter(staged, offset));
            result.objectBytes.resize(sizeof(T));
            std::memcpy(result.objectBytes.data(), &staged, sizeof(T));
            return result;
        });
}

template <typename T, size_t Length, size_t ParamPerElem>
inline void Tuning::Tuner::addArrayParams(std::string layoutTag,
                                          T (&paramArray)[Length],
                                          ParamGetter<const T &> getter,
                                          ParamSetter<T &>       setter)
{
    static_assert(std::is_trivially_copyable<T>::value,
                  "tuning parameter elements must be trivially copyable");
    auto stagerGetter = getter;
    auto stagerSetter = setter;
    addParams(
        std::move(layoutTag),
        paramArray,
        Length,
        sizeof(T),
        ParamPerElem,
        [getter = std::move(getter)](void *addr, size_t offset) -> TuneParam {
            return getter(*static_cast<const T *>(addr), offset);
        },
        [setter = std::move(setter)](void *addr, size_t offset, TuneParam param) -> void {
            setter(*static_cast<T *>(addr), offset, param);
        },
        [getter = std::move(stagerGetter),
         setter = std::move(stagerSetter)](const void                   *addr,
                                           const std::vector<TuneParam> &params) -> StagedParams {
            if (params.size() != ParamPerElem)
                throw std::invalid_argument("staged tuning parameter count mismatch");
            T staged = *static_cast<const T *>(addr);
            for (size_t offset = 0; offset < ParamPerElem; offset++)
                setter(staged, offset, params[offset]);
            StagedParams result;
            result.params.reserve(ParamPerElem);
            for (size_t offset = 0; offset < ParamPerElem; offset++)
                result.params.push_back(getter(staged, offset));
            result.objectBytes.resize(sizeof(T));
            std::memcpy(result.objectBytes.data(), &staged, sizeof(T));
            return result;
        });
}
