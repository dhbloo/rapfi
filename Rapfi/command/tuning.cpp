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
#include "../core/filesystem.h"
#include "../core/iohelper.h"
#include "../eval/classicalpolicy.h"
#include "../eval/scoretables.h"
#include "../tuning/dataset.h"
#include "../tuning/policytrainer.h"
#include "../tuning/tuner.h"
#include "argutils.h"
#include "command.h"

#define CXXOPTS_NO_REGEX
#include <algorithm>
#include <array>
#include <cmath>
#include <cpptoml.h>
#include <ctime>
#include <cxxopts.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>

using namespace Tuning;

namespace {

void parseTuningRules(TuningConfig &cfg, std::vector<std::string> rules)
{
    for (const std::string &ruleStr : rules) {
        Rule rule          = Command::parseRule(ruleStr);
        cfg.tuneRule[rule] = true;
    }
}

int parsePolicyTable(const std::string &table)
{
    if (table == "freestyle")
        return FREESTYLE;
    if (table == "standard")
        return STANDARD;
    if (table == "renju-black")
        return RENJU + BLACK;
    if (table == "renju-white")
        return RENJU + WHITE;
    throw std::invalid_argument("unknown compact policy table " + table);
}

LossType parseLossType(std::string lossType)
{
    if (lossType == "L1")
        return LossType::L1;
    else if (lossType == "L2")
        return LossType::L2;
    else if (lossType == "BCE")
        return LossType::BCE;
    else
        throw std::invalid_argument("unknown loss type " + lossType);
}

const char *formatLearningRateSchedule(LearningRateSchedule schedule)
{
    switch (schedule) {
    case LearningRateSchedule::Constant: return "constant";
    case LearningRateSchedule::Exponential: return "exponential";
    }
    throw std::logic_error("unknown learning-rate schedule");
}

BoardSizeWeighting parseBoardSizeWeighting(const std::string &weighting)
{
    if (weighting == "sample")
        return BoardSizeWeighting::SampleFrequency;
    if (weighting == "equal")
        return BoardSizeWeighting::EqualBoard;
    throw std::invalid_argument("unknown board size weighting " + weighting);
}

const char *formatBoardSizeWeighting(BoardSizeWeighting weighting)
{
    switch (weighting) {
    case BoardSizeWeighting::SampleFrequency: return "sample";
    case BoardSizeWeighting::EqualBoard: return "equal";
    }
    throw std::logic_error("unknown board-size weighting");
}

LearningRateSchedule parseLearningRateSchedule(const std::string &schedule)
{
    if (schedule == "constant")
        return LearningRateSchedule::Constant;
    if (schedule == "exponential")
        return LearningRateSchedule::Exponential;
    throw std::invalid_argument("unknown learning-rate schedule " + schedule);
}

TrainingSemantics parseTrainingSemantics(const std::string &semantics)
{
    if (semantics == "current")
        return TrainingSemantics::Current;
    if (semantics == "bootstrap")
        return TrainingSemantics::Bootstrap;
    throw std::invalid_argument("unknown training semantics " + semantics);
}

void validateConfig(size_t epochs, const TuningConfig &cfg, bool tuneCompactPolicy)
{
    if (epochs < 1)
        throw std::invalid_argument("epochs must be greater than 0");
    if (!(cfg.tuneRule[FREESTYLE] || cfg.tuneRule[STANDARD] || cfg.tuneRule[RENJU]))
        throw std::invalid_argument("there must be at least one rule to tune");
    if (!(cfg.tuneEval || cfg.tuneMoveScore || tuneCompactPolicy))
        throw std::invalid_argument("at least one tuning objective must be enabled");
    if (cfg.batchSize < 1)
        throw std::invalid_argument("batchsize must be greater than 0");
    constexpr size_t MiB = 1024 * 1024;
    if (cfg.memoryLimitMB > std::numeric_limits<size_t>::max() / MiB)
        throw std::invalid_argument("memory-limit-mb is too large");
    if (cfg.memoryLimitMB != 0 && cfg.memoryLimitMB < 32)
        throw std::invalid_argument("memory-limit-mb must be 0 or at least 32");
    if (cfg.shardSizeMB < 1 || cfg.shardSizeMB > std::numeric_limits<size_t>::max() / MiB)
        throw std::invalid_argument("shard-size-mb must be at least 1 and fit size_t");
    if (!std::isfinite(cfg.learningRate) || cfg.learningRate <= 0)
        throw std::invalid_argument("learning-rate must be finite and greater than 0");
    if (cfg.learningRateSchedule != LearningRateSchedule::Constant
        && (!std::isfinite(cfg.finalLearningRate) || cfg.finalLearningRate <= 0
            || cfg.finalLearningRate > cfg.learningRate))
        throw std::invalid_argument(
            "final-learning-rate must be finite, greater than 0, and not greater than "
            "learning-rate for a decaying schedule");
    if (!std::isfinite(cfg.weightDecay) || cfg.weightDecay < 0)
        throw std::invalid_argument("weight-decay must be finite and not less than 0");
    if (!std::isfinite(cfg.moveScoreLossGamma) || cfg.moveScoreLossGamma < 0)
        throw std::invalid_argument("move-score-loss-gamma must be not less than 0");
    if (!std::isfinite(cfg.multiPVPolicyTemperature) || cfg.multiPVPolicyTemperature < 0)
        throw std::invalid_argument(
            "multipv-policy-temperature must be finite and not less than 0");
    if (cfg.multiPVPolicyTemperature > 0
        && (cfg.multiPVPolicyTemperature > std::numeric_limits<float>::max()
            || cfg.multiPVPolicyTemperature < std::numeric_limits<float>::denorm_min()))
        throw std::invalid_argument("multipv-policy-temperature does not fit float storage");
    if (cfg.multiPVPolicyTemperature > 0 && !cfg.tuneMoveScore)
        throw std::invalid_argument("multipv-policy-temperature requires --tune-move-score");
    if (cfg.multiPVPolicyTemperature > 0 && cfg.moveScoreLossGamma != 0)
        throw std::invalid_argument("multipv-policy-temperature requires move-score-loss-gamma=0");
    if (!std::isfinite(cfg.multiPVPolicyEvalScale) || cfg.multiPVPolicyEvalScale < 0)
        throw std::invalid_argument("multipv-policy-eval-scale must be finite and not less than 0");
    if (cfg.multiPVPolicyEvalScale > 0
        && (cfg.multiPVPolicyEvalScale > std::numeric_limits<float>::max()
            || cfg.multiPVPolicyEvalScale < std::numeric_limits<float>::denorm_min()))
        throw std::invalid_argument("multipv-policy-eval-scale does not fit float storage");
    if (cfg.multiPVPolicyEvalScale > 0 && cfg.multiPVPolicyTemperature == 0)
        throw std::invalid_argument(
            "multipv-policy-eval-scale requires multipv-policy-temperature greater than 0");
    if (cfg.tuneMoveScore) {
        auto tunesTable = [&](int table) {
            return table == FREESTYLE && cfg.tuneRule[FREESTYLE]
                   || table == STANDARD && cfg.tuneRule[STANDARD]
                   || (table == RENJU || table == RENJU + WHITE) && cfg.tuneRule[RENJU];
        };
        for (int table = 0; table < RULE_NB + 1; table++)
            if (tunesTable(table) && Evaluation::POLICY_CROSS_ACTIVE_MASK[table])
                throw std::invalid_argument(
                    "move-score training requires selected tables without compact policy "
                    "residuals");
    }
    if (!std::isfinite(cfg.moveScoreScale) || cfg.moveScoreScale <= 0)
        throw std::invalid_argument("move-score-scale must be finite and greater than 0");
    if (!std::isfinite(cfg.moveScoreBias))
        throw std::invalid_argument("move-score-bias must be finite");
    if (cfg.moveScoreMin > cfg.moveScoreMax)
        throw std::invalid_argument("move-score-max must be not less than move-score-min");
    if (cfg.projectMoveScoreScale && !cfg.tuneMoveScore)
        throw std::invalid_argument("move-score scale projection requires --tune-move-score");
    if (!std::isfinite(cfg.moveScoreReferenceDispersion) || cfg.moveScoreReferenceDispersion < 0)
        throw std::invalid_argument(
            "move-score-reference-dispersion must be finite and not less than 0");
    if (cfg.moveScoreReferenceDispersion > 0 && !cfg.projectMoveScoreScale)
        throw std::invalid_argument(
            "move-score-reference-dispersion requires --project-move-score-scale");
    if (cfg.boardSizeMin == 0 || cfg.boardSizeMax > MAX_BOARD_SIZE)
        throw std::invalid_argument("board-size range must be inside the engine board limits");
    if (cfg.boardSizeMin > cfg.boardSizeMax)
        throw std::invalid_argument("max-boardsize must be greater than min-boardsize");
    if (!cfg.usePreviousScalingFactor) {
        if (cfg.nIterations < 1 || cfg.nStepsPerIteration < 1)
            throw std::invalid_argument(
                "num-iteration and num-steps-per-iteration must be greater than 0");
        if (!std::isfinite(cfg.scalingFactorMin) || !std::isfinite(cfg.scalingFactorMax)
            || cfg.scalingFactorMin <= 0 || cfg.scalingFactorMax <= 0)
            throw std::invalid_argument("scaling factor bounds must be finite and greater than 0");
        if (cfg.scalingFactorMin > cfg.scalingFactorMax)
            throw std::invalid_argument(
                "scaling-factor-lower-bound must be not less than scaling-factor-upper-bound");
    }
}

std::unique_ptr<Dataset> createDataset(Command::DatasetType            datasetType,
                                       const std::vector<std::string> &pathList)
{
    switch (datasetType) {
    case Command::DatasetType::SimpleBinary: return std::make_unique<SimpleBinaryDataset>(pathList);
    case Command::DatasetType::PackedBinary: return std::make_unique<PackedBinaryDataset>(pathList);
    default: throw std::invalid_argument("unsupported dataset type");
    }
}

std::unique_ptr<Dataset> createDataset(Command::DatasetType                      datasetType,
                                       const std::vector<std::filesystem::path> &pathList)
{
    switch (datasetType) {
    case Command::DatasetType::SimpleBinary: return std::make_unique<SimpleBinaryDataset>(pathList);
    case Command::DatasetType::PackedBinary: return std::make_unique<PackedBinaryDataset>(pathList);
    default: throw std::invalid_argument("unsupported dataset type");
    }
}

std::vector<std::filesystem::path>
expandDatasetPaths(const std::vector<std::filesystem::path> &paths,
                   const std::vector<std::string>           &extensions)
{
    auto extensionMatches = [&extensions](const std::filesystem::path &extension) {
        return std::any_of(extensions.begin(), extensions.end(), [&](const std::string &expected) {
            return extension == std::filesystem::u8path(expected);
        });
    };

    std::vector<std::filesystem::path> filenames;
    for (const std::filesystem::path &path : paths) {
        if (std::filesystem::is_directory(path)) {
            std::vector<std::filesystem::path> directoryFiles;
            for (const auto &entry : std::filesystem::recursive_directory_iterator(path)) {
                if (entry.is_regular_file() && extensionMatches(entry.path().extension()))
                    directoryFiles.push_back(entry.path());
            }
            std::sort(directoryFiles.begin(), directoryFiles.end());
            filenames.insert(filenames.end(), directoryFiles.begin(), directoryFiles.end());
        }
        else {
            filenames.push_back(path);
        }
    }
    return filenames;
}

const char *datasetFormatName(Command::DatasetType datasetType)
{
    switch (datasetType) {
    case Command::DatasetType::SimpleBinary: return "bin";
    case Command::DatasetType::PackedBinary: return "binpack";
    default: throw std::invalid_argument("unsupported tuning dataset type");
    }
}

template <typename T>
cpptoml::option<T> getOptionalPhaseValue(const cpptoml::table &table, const std::string &key)
{
    cpptoml::option<T> value = table.get_as<T>(key);
    if (table.contains(key) && !value)
        throw std::invalid_argument("tuning curriculum phase key " + key
                                    + " has the wrong TOML type");
    return value;
}

struct CurriculumPhaseInput
{
    std::string                        name;
    size_t                             epochs;
    std::vector<std::filesystem::path> trainingPaths;
    std::vector<std::filesystem::path> validationPaths;
    Command::DatasetType               trainingType;
    Command::DatasetType               validationType;
    std::filesystem::path              preparedCachePath;
    bool                               rebuildPreparedCache;
    uint8_t                            boardSizeMin;
    uint8_t                            boardSizeMax;
    size_t                             batchSize;
    uint64_t                           seed;
    double                             multiPVPolicyTemperature;
    double                             multiPVPolicyEvalScale;
    std::optional<TrainingSemantics>   trainingSemantics;
    bool                               tuneEval;
    bool                               tuneMoveScore;
    bool                               trainCompactPolicy;
    double                             learningRate;
    double                             finalLearningRate;
    double                             weightDecay;
    LearningRateSchedule               learningRateSchedule;
    size_t                             recomputeInterval;
    bool                               localLearningRateSchedule;
    bool                               localRecomputeSchedule;
    bool                               projectMoveScoreAtEnd;
    bool                               coalesceMoveScoreAtEnd;
    BoardSizeWeighting                 boardSizeWeighting;
    std::unique_ptr<Dataset>           trainingDataset;
    std::unique_ptr<Dataset>           validationDataset;
};

std::vector<CurriculumPhaseInput> parseCurriculum(const std::filesystem::path &manifestPath,
                                                  Command::DatasetType         defaultTrainingType,
                                                  Command::DatasetType defaultValidationType,
                                                  const TuningConfig  &defaults)
{
    std::ifstream manifest(manifestPath);
    if (!manifest)
        throw std::runtime_error("unable to open tuning curriculum " + manifestPath.string());
    auto root = cpptoml::parser(manifest).parse();
    for (const auto &[key, value] : *root)
        if (key != "phase")
            throw std::invalid_argument("unknown tuning curriculum root key " + key);
    auto phaseArray = root->get_table_array("phase");
    if (!phaseArray || phaseArray->begin() == phaseArray->end())
        throw std::invalid_argument("tuning curriculum requires at least one [[phase]] table");

    const std::filesystem::path base        = manifestPath.parent_path();
    auto                        resolvePath = [&base](const std::string &configured) {
        std::filesystem::path path = std::filesystem::u8path(configured);
        if (path.is_relative())
            path = base / path;
        return std::filesystem::absolute(path).lexically_normal();
    };

    std::vector<CurriculumPhaseInput> phases;
    phases.reserve(static_cast<size_t>(std::distance(phaseArray->begin(), phaseArray->end())));
    for (const auto &table : *phaseArray) {
        static constexpr std::array<const char *, 24> PhaseKeys = {
            "name",
            "epochs",
            "training_dataset",
            "validation_dataset",
            "prepared_cache",
            "training_dataset_type",
            "validation_dataset_type",
            "rebuild_prepared_cache",
            "min_boardsize",
            "max_boardsize",
            "batch_size",
            "seed",
            "multipv_policy_temperature",
            "multipv_policy_eval_scale",
            "training_semantics",
            "objectives",
            "learning_rate",
            "learning_rate_schedule",
            "final_learning_rate",
            "recompute_interval",
            "project_move_score_at_end",
            "coalesce_move_score_at_end",
            "board_size_weighting",
            "weight_decay",
        };
        for (const auto &[key, value] : *table)
            if (std::find(PhaseKeys.begin(), PhaseKeys.end(), key) == PhaseKeys.end())
                throw std::invalid_argument("unknown tuning curriculum phase key " + key);
        CurriculumPhaseInput phase;
        phase.name      = table->get_as<std::string>("name").value_or("");
        auto epochValue = table->get_as<int64_t>("epochs");
        auto training   = table->get_array_of<std::string>("training_dataset");
        auto validation = table->get_array_of<std::string>("validation_dataset");
        auto cache      = table->get_as<std::string>("prepared_cache");
        if (phase.name.empty() || !epochValue || *epochValue <= 0 || !training || training->empty()
            || !validation || validation->empty() || !cache || cache->empty())
            throw std::invalid_argument(
                "each curriculum phase requires a name, positive epochs, non-empty training and "
                "validation datasets, and a prepared cache");
        if (uint64_t(*epochValue) > std::numeric_limits<size_t>::max())
            throw std::invalid_argument("curriculum phase epochs do not fit size_t");
        phase.epochs = static_cast<size_t>(*epochValue);
        for (const std::string &path : *training)
            phase.trainingPaths.push_back(resolvePath(path));
        for (const std::string &path : *validation)
            phase.validationPaths.push_back(resolvePath(path));
        phase.trainingType = Command::parseDatasetType(
            getOptionalPhaseValue<std::string>(*table, "training_dataset_type")
                .value_or(datasetFormatName(defaultTrainingType)));
        phase.validationType = Command::parseDatasetType(
            getOptionalPhaseValue<std::string>(*table, "validation_dataset_type")
                .value_or(datasetFormatName(defaultValidationType)));
        phase.preparedCachePath = resolvePath(*cache);
        phase.rebuildPreparedCache =
            getOptionalPhaseValue<bool>(*table, "rebuild_prepared_cache").value_or(false);
        int64_t boardMin =
            getOptionalPhaseValue<int64_t>(*table, "min_boardsize").value_or(defaults.boardSizeMin);
        int64_t boardMax =
            getOptionalPhaseValue<int64_t>(*table, "max_boardsize").value_or(defaults.boardSizeMax);
        if (boardMin < 1 || boardMax > MAX_BOARD_SIZE || boardMin > boardMax)
            throw std::invalid_argument("curriculum phase board-size range is invalid");
        phase.boardSizeMin = static_cast<uint8_t>(boardMin);
        phase.boardSizeMax = static_cast<uint8_t>(boardMax);
        int64_t batchSize =
            getOptionalPhaseValue<int64_t>(*table, "batch_size")
                .value_or(defaults.batchSize <= size_t(std::numeric_limits<int64_t>::max())
                              ? int64_t(defaults.batchSize)
                              : int64_t(-1));
        int64_t seed = getOptionalPhaseValue<int64_t>(*table, "seed")
                           .value_or(defaults.seed <= uint64_t(std::numeric_limits<int64_t>::max())
                                         ? int64_t(defaults.seed)
                                         : int64_t(-1));
        if (batchSize <= 0 || uint64_t(batchSize) > std::numeric_limits<size_t>::max())
            throw std::invalid_argument(
                "curriculum phase batch_size must be positive and fit size_t");
        if (seed < 0)
            throw std::invalid_argument("curriculum phase seed must be nonnegative and fit int64");
        phase.batchSize = static_cast<size_t>(batchSize);
        phase.seed      = static_cast<uint64_t>(seed);
        phase.multiPVPolicyTemperature =
            getOptionalPhaseValue<double>(*table, "multipv_policy_temperature")
                .value_or(defaults.multiPVPolicyTemperature);
        phase.multiPVPolicyEvalScale =
            getOptionalPhaseValue<double>(*table, "multipv_policy_eval_scale")
                .value_or(defaults.multiPVPolicyEvalScale);
        if (auto semantics = getOptionalPhaseValue<std::string>(*table, "training_semantics"))
            phase.trainingSemantics = parseTrainingSemantics(*semantics);
        auto objectives = table->get_array_of<std::string>("objectives");
        if (table->contains("objectives") && !objectives)
            throw std::invalid_argument(
                "tuning curriculum phase key objectives has the wrong TOML type");
        phase.tuneEval           = defaults.tuneEval;
        phase.tuneMoveScore      = defaults.tuneMoveScore;
        phase.trainCompactPolicy = false;
        if (objectives) {
            phase.tuneEval      = false;
            phase.tuneMoveScore = false;
            if (objectives->empty())
                throw std::invalid_argument("tuning curriculum phase objectives must not be empty");
            for (const std::string &objective : *objectives) {
                if (objective == "value")
                    phase.tuneEval = true;
                else if (objective == "move_score")
                    phase.tuneMoveScore = true;
                else if (objective == "compact_policy")
                    phase.trainCompactPolicy = true;
                else
                    throw std::invalid_argument("unknown tuning curriculum objective " + objective);
            }
        }
        if (phase.trainCompactPolicy && phase.tuneMoveScore)
            throw std::invalid_argument(
                "compact_policy and move_score must use separate curriculum phases");
        phase.learningRate =
            getOptionalPhaseValue<double>(*table, "learning_rate").value_or(defaults.learningRate);
        phase.learningRateSchedule = parseLearningRateSchedule(
            getOptionalPhaseValue<std::string>(*table, "learning_rate_schedule")
                .value_or(formatLearningRateSchedule(defaults.learningRateSchedule)));
        phase.finalLearningRate = getOptionalPhaseValue<double>(*table, "final_learning_rate")
                                      .value_or(defaults.finalLearningRate);
        phase.weightDecay =
            getOptionalPhaseValue<double>(*table, "weight_decay").value_or(defaults.weightDecay);
        int64_t recomputeInterval =
            getOptionalPhaseValue<int64_t>(*table, "recompute_interval")
                .value_or(defaults.recomputeInterval <= size_t(std::numeric_limits<int64_t>::max())
                              ? int64_t(defaults.recomputeInterval)
                              : int64_t(-1));
        if (recomputeInterval < 0)
            throw std::invalid_argument(
                "tuning curriculum phase recompute_interval must be nonnegative and fit size_t");
        phase.recomputeInterval         = static_cast<size_t>(recomputeInterval);
        phase.localLearningRateSchedule = table->contains("learning_rate")
                                          || table->contains("learning_rate_schedule")
                                          || table->contains("final_learning_rate");
        phase.localRecomputeSchedule = table->contains("recompute_interval");
        phase.projectMoveScoreAtEnd =
            getOptionalPhaseValue<bool>(*table, "project_move_score_at_end").value_or(false);
        phase.coalesceMoveScoreAtEnd =
            getOptionalPhaseValue<bool>(*table, "coalesce_move_score_at_end").value_or(false);
        phase.boardSizeWeighting = parseBoardSizeWeighting(
            getOptionalPhaseValue<std::string>(*table, "board_size_weighting")
                .value_or(formatBoardSizeWeighting(defaults.boardSizeWeighting)));
        phases.push_back(std::move(phase));
    }
    return phases;
}

}  // namespace

int Command::tuning(int argc, char *argv[])
{
    std::string                        outdir;
    std::string                        trainName;
    size_t                             epochs              = 0;
    size_t                             modelExportInterval = 0;
    TuningConfig                       cfg                 = {};
    PolicyTrainingConfig               policyConfig;
    DatasetType                        trainDatasetType;
    DatasetType                        valDatasetType;
    std::vector<std::string>           trainDatasetPathList;
    std::vector<std::string>           valDatasetPathList;
    std::vector<std::string>           extensions;
    std::string                        curriculumPath;
    std::filesystem::path              parentModelPath;
    std::vector<std::filesystem::path> policyTracePaths;
    std::unique_ptr<Dataset>           trainDataset, valDataset;

    cxxopts::Options options("rapfi tuning");
    options.add_options()                                                        //
        ("o,output", "Output directory", cxxopts::value<std::string>())          //
        ("n,name", "Name of the trained models", cxxopts::value<std::string>())  //
        ("d,training-dataset",
         "Training dataset filename/directory(s), plain or compressed",
         cxxopts::value<std::vector<std::string>>())  //
        ("v,validation-dataset",
         "Validation dataset filename/directory(s), plain or compressed",
         cxxopts::value<std::vector<std::string>>())  //
        ("curriculum",
         "TOML file containing ordered [[phase]] dataset groups",
         cxxopts::value<std::string>())  //
        ("parent-model",
         "Classical parent model used to initialize training",
         cxxopts::value<std::string>())  //
        ("policy-trace-dataset",
         "Policy trace file(s); selects compact policy-residual training",
         cxxopts::value<std::vector<std::string>>())  //
        ("policy-trace-tables",
         "Compact policy destination table(s): freestyle, standard, renju-black, renju-white",
         cxxopts::value<std::vector<std::string>>()->default_value("freestyle"))  //
        ("policy-anchor",
         "L2 anchor strength for compact policy-residual training",
         cxxopts::value<double>())  //
        ("policy-learning-rate",
         "Policy-trace learning rate when value and policy train jointly",
         cxxopts::value<double>())  //
        ("policy-score-scale",
         "Logit scale for compact policy-residual training",
         cxxopts::value<double>())  //
        ("policy-residual-limit",
         "Absolute integer limit for compact policy residuals",
         cxxopts::value<int>())  //
        ("training-dataset-type",
         "Input dataset type, one of [bin, binpack]",
         cxxopts::value<std::string>()->default_value("binpack"))  //
        ("validation-dataset-type",
         "Input dataset type, one of [bin, binpack]",
         cxxopts::value<std::string>()->default_value("binpack"))  //
        ("e,epochs",
         "Number of epochs to train",
         cxxopts::value<size_t>())  //
        ("i,export-interval",
         "Number of epochs between model checkpoint saving (0 for no checkpoint)",
         cxxopts::value<size_t>()->default_value("100"))  //
        ("b,batchsize",
         "Number of samples in one gradient batch",
         cxxopts::value<size_t>()->default_value(std::to_string(cfg.batchSize)))  //
        ("board-size-weighting",
         "Board-size contribution, one of [sample, equal]",
         cxxopts::value<std::string>()->default_value("sample"))  //
        ("training-semantics",
         "Training geometry, one of [current, bootstrap]",
         cxxopts::value<std::string>()->default_value("current"))  //
        ("threads",
         "Number of tuner worker threads (0 uses hardware concurrency)",
         cxxopts::value<size_t>()->default_value(std::to_string(cfg.numThreads)))  //
        ("seed",
         "Seed for reproducible tuner random streams",
         cxxopts::value<uint64_t>()->default_value(std::to_string(cfg.seed)))  //
        ("memory-limit-mb",
         "Enable file-backed tuning with this process memory budget in MiB (0 keeps in memory)",
         cxxopts::value<size_t>()->default_value(std::to_string(cfg.memoryLimitMB)))  //
        ("prepared-cache",
         "Directory for versioned prepared corpus shards",
         cxxopts::value<std::string>())                                          //
        ("rebuild-prepared-cache", "Bypass and replace a valid prepared cache")  //
        ("shard-size-mb",
         "Target size of each prepared shard in MiB",
         cxxopts::value<size_t>()->default_value(std::to_string(cfg.shardSizeMB)))  //
        ("l,learning-rate",
         "Learning rate for gradient descent",
         cxxopts::value<double>()->default_value(std::to_string(cfg.learningRate)))  //
        ("learning-rate-schedule",
         "Learning-rate schedule (one of [constant, exponential])",
         cxxopts::value<std::string>()->default_value("constant"))  //
        ("final-learning-rate",
         "Final learning rate for exponential decay",
         cxxopts::value<double>()->default_value(std::to_string(cfg.finalLearningRate)))  //
        ("w,weight-decay",
         "Weight decay for gradient descent (0.0~1.0)",
         cxxopts::value<double>()->default_value(std::to_string(cfg.weightDecay)))  //
        ("L,loss",
         "Loss type (one of [L1, L2, BCE])",
         cxxopts::value<std::string>()->default_value("BCE"))  //
        ("r,rules-to-tune",
         "Params of which rules [freestyle, standard, renju] that need to be tuned",
         cxxopts::value<std::vector<std::string>>()->default_value("freestyle,standard,renju"))  //
        ("s,shuffle", "Shuffle training datasets")                                               //
        ("m,tune-move-score", "Enable tuning of move scores")                                    //
        ("no-tune-eval", "Disable tuning of evaluation")                                         //
        ("move-score-loss-gamma",
         "Gamma value (>= 0) of move score focal loss",
         cxxopts::value<double>()->default_value(std::to_string(cfg.moveScoreLossGamma)))  //
        ("multipv-policy-temperature",
         "Softmax temperature over Multi-PV win rates (0 uses the best move only)",
         cxxopts::value<double>()->default_value(std::to_string(cfg.multiPVPolicyTemperature)))  //
        ("multipv-policy-eval-scale",
         "Scaling factor used to decode Multi-PV evals (0 uses the loaded model)",
         cxxopts::value<double>()->default_value(std::to_string(cfg.multiPVPolicyEvalScale)))  //
        ("move-score-scale",
         "Parameter-to-integer move-score scale (canonical logits use engine ScoreScale)",
         cxxopts::value<double>()->default_value(std::to_string(cfg.moveScoreScale)))  //
        ("move-score-bias",
         "Bias of move score conversion from float to int",
         cxxopts::value<double>()->default_value(std::to_string(cfg.moveScoreBias)))  //
        ("move-score-min",
         "Minimum of converted move score value",
         cxxopts::value<Score>()->default_value(std::to_string(cfg.moveScoreMin)))  //
        ("move-score-max",
         "Maximum of converted move score value",
         cxxopts::value<Score>()->default_value(std::to_string(cfg.moveScoreMax)))  //
        ("project-move-score-scale",
         "Project exported move-score quiet-band dispersion to its frozen reference")  //
        ("move-score-reference-dispersion",
         "Frozen quiet-band reference dispersion (0 measures the loaded parent)",
         cxxopts::value<double>()->default_value(
             std::to_string(cfg.moveScoreReferenceDispersion)))  //
        ("dataset-file-extensions",
         "Extensions to filter dataset file in a directory",
         cxxopts::value<std::vector<std::string>>()->default_value(".bin,.lz4"))  //
        ("max-entries",
         "Max number of tune entries to read from datasets",
         cxxopts::value<size_t>()->default_value(std::to_string(cfg.maxTuneEntries)))  //
        ("min-boardsize",
         "Minimal board size to accept a tune entry from dataset",
         cxxopts::value<uint8_t>()->default_value(std::to_string(cfg.boardSizeMin)))  //
        ("max-boardsize",
         "Maximal board size to accept a tune entry from dataset",
         cxxopts::value<uint8_t>()->default_value(std::to_string(cfg.boardSizeMax)))  //
        ("min-ply",
         "Minimal game ply to accept a tune entry from dataset",
         cxxopts::value<uint16_t>()->default_value(std::to_string(cfg.minPly)))  //
        ("min-ply-before-full",
         "Minimal game ply from fulfilled board to accept a tune entry from dataset",
         cxxopts::value<uint16_t>()->default_value(std::to_string(cfg.minPlyBeforeFull)))  //
        ("fix-scaling-factor", "Keep scaling factor unchanged during tuning")              //
        ("num-iteration",
         "Number of iterations to find the optimal scaling factor",
         cxxopts::value<int>()->default_value(std::to_string(cfg.nIterations)))  //
        ("num-steps-per-iteration",
         "Number of steps per iteration to find the optimal scaling factor",
         cxxopts::value<int>()->default_value(std::to_string(cfg.nStepsPerIteration)))  //
        ("scaling-factor-lower-bound",
         "Lower bound of scaling factor to search",
         cxxopts::value<double>()->default_value(std::to_string(cfg.scalingFactorMin)))  //
        ("scaling-factor-upper-bound",
         "Upper bound of scaling factor to search",
         cxxopts::value<double>()->default_value(std::to_string(cfg.scalingFactorMax)))  //
        ("recompute-interval",
         "Number of epoches to recompute scaling factor (0 for no recompute)",
         cxxopts::value<size_t>()->default_value(std::to_string(cfg.recomputeInterval)))  //
        ("h,help", "Print tuning usage");

    parseSubcommandArguments(
        options,
        argc,
        argv,
        "tuning argument",
        [&](const cxxopts::ParseResult &args) {
            const bool usesCurriculum = args.count("curriculum") != 0;
            if (args.count("policy-trace-dataset")) {
                if (!args.count("policy-anchor") || !args.count("policy-score-scale")
                    || !args.count("policy-residual-limit") || !args.count("policy-learning-rate"))
                    throw std::invalid_argument(
                        "policy trace training requires --policy-anchor, --policy-score-scale, "
                        "--policy-residual-limit, and --policy-learning-rate");
                if (!usesCurriculum)
                    throw std::invalid_argument(
                        "policy trace training is supported only by curriculum compact_policy "
                        "phases");
                for (const std::string &path :
                     args["policy-trace-dataset"].as<std::vector<std::string>>())
                    policyTracePaths.emplace_back(path);
                policyConfig.epochs        = 0;
                policyConfig.learningRate  = args["policy-learning-rate"].as<double>();
                policyConfig.anchor        = args["policy-anchor"].as<double>();
                policyConfig.scoreScale    = args["policy-score-scale"].as<double>();
                policyConfig.residualLimit = args["policy-residual-limit"].as<int>();
                policyConfig.equalBoardWeighting =
                    parseBoardSizeWeighting(args["board-size-weighting"].as<std::string>())
                    == BoardSizeWeighting::EqualBoard;
                policyConfig.destinationTables.clear();
                for (const std::string &table :
                     args["policy-trace-tables"].as<std::vector<std::string>>())
                    policyConfig.destinationTables.push_back(parsePolicyTable(table));
            }

            parseTuningRules(cfg, args["rules-to-tune"].as<std::vector<std::string>>());
            trainDatasetType = parseDatasetType(args["training-dataset-type"].as<std::string>());
            valDatasetType   = parseDatasetType(args["validation-dataset-type"].as<std::string>());
            if (usesCurriculum) {
                if (args.count("training-dataset") || args.count("validation-dataset")
                    || args.count("epochs") || args.count("prepared-cache")
                    || args.count("rebuild-prepared-cache"))
                    throw std::invalid_argument(
                        "--curriculum replaces dataset, epoch, and prepared-cache arguments");
                curriculumPath = args["curriculum"].as<std::string>();
            }
            else {
                trainDatasetPathList = args["training-dataset"].as<std::vector<std::string>>();
            }
            if (!usesCurriculum && args.count("validation-dataset")) {
                valDatasetType =
                    parseDatasetType(args["validation-dataset-type"].as<std::string>());
                valDatasetPathList = args["validation-dataset"].as<std::vector<std::string>>();
            }
            if (args.count("parent-model"))
                parentModelPath = args["parent-model"].as<std::string>();
            extensions          = args["dataset-file-extensions"].as<std::vector<std::string>>();
            outdir              = args["output"].as<std::string>();
            trainName           = args["name"].as<std::string>();
            epochs              = usesCurriculum ? 0 : args["epochs"].as<size_t>();
            modelExportInterval = usesCurriculum && args.count("export-interval") == 0
                                      ? 0
                                      : args["export-interval"].as<size_t>();
            if (usesCurriculum && modelExportInterval != 0)
                throw std::invalid_argument("tuning curricula prohibit intermediate model export");
            cfg.batchSize = args["batchsize"].as<size_t>();
            cfg.boardSizeWeighting =
                parseBoardSizeWeighting(args["board-size-weighting"].as<std::string>());
            cfg.trainingSemantics =
                parseTrainingSemantics(args["training-semantics"].as<std::string>());
            cfg.maxTuneEntries = args["max-entries"].as<size_t>();
            cfg.numThreads     = args["threads"].as<size_t>();
            cfg.seed           = args["seed"].as<uint64_t>();
            cfg.memoryLimitMB  = args["memory-limit-mb"].as<size_t>();
            cfg.shardSizeMB    = args["shard-size-mb"].as<size_t>();
            if (!usesCurriculum && args.count("prepared-cache"))
                cfg.preparedCachePath = args["prepared-cache"].as<std::string>();
            cfg.rebuildPreparedCache = args.count("rebuild-prepared-cache");
            cfg.learningRate         = args["learning-rate"].as<double>();
            cfg.learningRateSchedule =
                parseLearningRateSchedule(args["learning-rate-schedule"].as<std::string>());
            cfg.finalLearningRate = args["final-learning-rate"].as<double>();
            cfg.weightDecay      = args["weight-decay"].as<double>();
            cfg.lossType                   = parseLossType(args["loss"].as<std::string>());
            cfg.shuffleTuneEntries         = args.count("shuffle");
            cfg.tuneMoveScore              = args.count("tune-move-score");
            cfg.tuneEval                   = !args.count("no-tune-eval");
            cfg.moveScoreLossGamma         = args["move-score-loss-gamma"].as<double>();
            cfg.multiPVPolicyTemperature   = args["multipv-policy-temperature"].as<double>();
            cfg.multiPVPolicyEvalScale     = args["multipv-policy-eval-scale"].as<double>();
            cfg.moveScoreScale               = args["move-score-scale"].as<double>();
            cfg.moveScoreBias                = args["move-score-bias"].as<double>();
            cfg.moveScoreMin                 = args["move-score-min"].as<Score>();
            cfg.moveScoreMax                 = args["move-score-max"].as<Score>();
            cfg.projectMoveScoreScale        = args.count("project-move-score-scale");
            cfg.moveScoreReferenceDispersion = args["move-score-reference-dispersion"].as<double>();
            cfg.boardSizeMin                 = args["min-boardsize"].as<uint8_t>();
            cfg.boardSizeMax                 = args["max-boardsize"].as<uint8_t>();
            cfg.minPly                       = args["min-ply"].as<uint16_t>();
            cfg.minPlyBeforeFull             = args["min-ply-before-full"].as<uint16_t>();
            cfg.usePreviousScalingFactor     = args.count("fix-scaling-factor");
            cfg.nIterations                  = args["num-iteration"].as<int>();
            cfg.nStepsPerIteration           = args["num-steps-per-iteration"].as<int>();
            cfg.scalingFactorMin             = args["scaling-factor-lower-bound"].as<double>();
            cfg.scalingFactorMax             = args["scaling-factor-upper-bound"].as<double>();
            cfg.recomputeInterval            = args["recompute-interval"].as<size_t>();
        });

    try {
        for (std::filesystem::path &path : policyTracePaths)
            path = std::filesystem::absolute(pathFromConsoleString(path.string()));

        if (!parentModelPath.empty()) {
            parentModelPath = getModelFullPath(parentModelPath);
            if (!loadModelFromFile(parentModelPath))
                throw std::runtime_error("unable to load tuning parent model");
            MESSAGEL("Tuning parent loaded from " << parentModelPath.string() << '.');
        }

        std::vector<CurriculumPhaseInput> curriculumInputs;
        if (!curriculumPath.empty()) {
            std::filesystem::path curriculumManifestPath =
                std::filesystem::absolute(pathFromConsoleString(curriculumPath));
            curriculumInputs =
                parseCurriculum(curriculumManifestPath, trainDatasetType, valDatasetType, cfg);
            if (cfg.memoryLimitMB == 0)
                throw std::invalid_argument("dataset curricula require nonzero --memory-limit-mb");
            bool   anyTuneEval      = false;
            bool   anyTuneMoveScore = false;
            size_t compactEpochs    = 0;
            for (const CurriculumPhaseInput &phase : curriculumInputs) {
                if (phase.epochs > std::numeric_limits<size_t>::max() - epochs)
                    throw std::overflow_error("tuning curriculum epoch sum overflows size_t");
                epochs += phase.epochs;
                anyTuneEval |= phase.tuneEval;
                anyTuneMoveScore |= phase.tuneMoveScore;
                if (phase.trainCompactPolicy) {
                    if (phase.epochs > std::numeric_limits<size_t>::max() - compactEpochs)
                        throw std::overflow_error(
                            "compact-policy curriculum epoch sum overflows size_t");
                    compactEpochs += phase.epochs;
                }
            }
            cfg.tuneEval      = anyTuneEval;
            cfg.tuneMoveScore = anyTuneMoveScore;
            if (compactEpochs != 0) {
                if (policyTracePaths.empty())
                    throw std::invalid_argument(
                        "compact_policy curriculum objectives require --policy-trace-dataset");
                policyConfig.epochs = compactEpochs;
            }
            else if (!policyTracePaths.empty())
                throw std::invalid_argument(
                    "curriculum policy traces require a compact_policy phase objective");
        }
        validateConfig(epochs, cfg, !policyTracePaths.empty());

        // Create output directory
        ensureDir(outdir);
        std::filesystem::path outpath = std::filesystem::absolute(pathFromConsoleString(outdir));
        if (curriculumInputs.empty() && cfg.memoryLimitMB != 0 && cfg.preparedCachePath.empty())
            cfg.preparedCachePath = outpath / ".prepared-cache";
        if (curriculumInputs.empty() && cfg.memoryLimitMB == 0 && !cfg.preparedCachePath.empty())
            throw std::invalid_argument("prepared-cache requires a nonzero --memory-limit-mb");
        if (curriculumInputs.empty() && cfg.memoryLimitMB == 0 && cfg.rebuildPreparedCache)
            throw std::invalid_argument(
                "rebuild-prepared-cache requires a nonzero --memory-limit-mb");

        // Make path list
        std::vector<TuningPhase> curriculumPhases;
        if (curriculumInputs.empty()) {
            trainDatasetPathList = makeFileListFromPathList(trainDatasetPathList, extensions);
            cfg.trainDatasetPaths.clear();
            for (const std::string &path : trainDatasetPathList)
                cfg.trainDatasetPaths.emplace_back(path);
            cfg.trainDatasetFormat = datasetFormatName(trainDatasetType);
            trainDataset           = createDataset(trainDatasetType, trainDatasetPathList);
            if (!valDatasetPathList.empty()) {
                valDatasetPathList = makeFileListFromPathList(valDatasetPathList, extensions);
                cfg.validationDatasetPaths.clear();
                for (const std::string &path : valDatasetPathList)
                    cfg.validationDatasetPaths.emplace_back(path);
                cfg.validationDatasetFormat = datasetFormatName(valDatasetType);
                valDataset                  = createDataset(valDatasetType, valDatasetPathList);
            }
        }
        else {
            curriculumPhases.reserve(curriculumInputs.size());
            for (CurriculumPhaseInput &input : curriculumInputs) {
                input.trainingPaths   = expandDatasetPaths(input.trainingPaths, extensions);
                input.validationPaths = expandDatasetPaths(input.validationPaths, extensions);
                input.trainingDataset = createDataset(input.trainingType, input.trainingPaths);
                input.validationDataset =
                    createDataset(input.validationType, input.validationPaths);

                TuningPhase phase;
                phase.name                      = input.name;
                phase.trainingDataset           = input.trainingDataset.get();
                phase.validationDataset         = input.validationDataset.get();
                phase.epochs                    = input.epochs;
                phase.preparedCachePath         = input.preparedCachePath;
                phase.trainDatasetFormat        = datasetFormatName(input.trainingType);
                phase.validationDatasetFormat   = datasetFormatName(input.validationType);
                phase.rebuildPreparedCache      = input.rebuildPreparedCache;
                phase.boardSizeMin              = input.boardSizeMin;
                phase.boardSizeMax              = input.boardSizeMax;
                phase.batchSize                 = input.batchSize;
                phase.seed                      = input.seed;
                phase.multiPVPolicyTemperature  = input.multiPVPolicyTemperature;
                phase.multiPVPolicyEvalScale    = input.multiPVPolicyEvalScale;
                phase.trainingSemantics         = input.trainingSemantics;
                phase.tuneEval                  = input.tuneEval;
                phase.tuneMoveScore             = input.tuneMoveScore;
                phase.trainCompactPolicy        = input.trainCompactPolicy;
                phase.learningRate              = input.learningRate;
                phase.finalLearningRate         = input.finalLearningRate;
                phase.weightDecay               = input.weightDecay;
                phase.learningRateSchedule      = input.learningRateSchedule;
                phase.recomputeInterval         = input.recomputeInterval;
                phase.localLearningRateSchedule = input.localLearningRateSchedule;
                phase.localRecomputeSchedule    = input.localRecomputeSchedule;
                phase.projectMoveScoreAtEnd     = input.projectMoveScoreAtEnd;
                phase.coalesceMoveScoreAtEnd    = input.coalesceMoveScoreAtEnd;
                phase.boardSizeWeighting        = input.boardSizeWeighting;
                for (const std::filesystem::path &path : input.trainingPaths)
                    phase.trainDatasetPaths.emplace_back(path);
                for (const std::filesystem::path &path : input.validationPaths)
                    phase.validationDatasetPaths.emplace_back(path);
                curriculumPhases.push_back(std::move(phase));
            }
        }
        std::vector<MoveScoreReferencePosition> moveScoreReferencePositions;
        if (cfg.projectMoveScoreScale) {
            moveScoreReferencePositions.reserve(benchPositions().size());
            for (const BenchPosition &bench : benchPositions())
                moveScoreReferencePositions.push_back(
                    {bench.rule,
                     bench.boardSize,
                     parsePositionString(bench.position, bench.boardSize, bench.boardSize)});
        }
        std::unique_ptr<PolicyTraceTrainer> policyTrainer;
        if (!policyTracePaths.empty())
            policyTrainer = std::make_unique<PolicyTraceTrainer>(policyTracePaths, policyConfig);

        // Create tuner with dataset and tunerConfig
        std::unique_ptr<Tuner> tuner;
        if (curriculumPhases.empty())
            tuner = std::make_unique<Tuner>(*trainDataset,
                                            valDataset.get(),
                                            cfg,
                                            std::move(moveScoreReferencePositions));
        else
            tuner = std::make_unique<Tuner>(curriculumPhases,
                                            cfg,
                                            std::move(moveScoreReferencePositions));
        // Open and init statistic CSV file
        std::ofstream statFile(outpath / "stat.csv");
        if (!statFile)
            throw std::runtime_error("unable to open tuning statistic output");
        std::ofstream boardStatFile(outpath / "stat-by-board.csv");
        if (!boardStatFile)
            throw std::runtime_error("unable to open per-board tuning statistic output");
        double totalElapsedSeconds    = 0.0;
        bool   compactPolicyBaseReady = curriculumPhases.empty();
        size_t compactPolicyPhase     = std::numeric_limits<size_t>::max();
        if (!curriculumPhases.empty()) {
            statFile << "Phase, PhaseEpoch, PhaseName, ";
            boardStatFile << "Phase, PhaseEpoch, PhaseName, ";
        }
        statFile
            << "Epoch, LearningRate, ValueLoss, PolicyLoss, ValueValLoss, "
               "PolicyValLoss, Elapsed, Epochs/Sec, Timestamp\n";
        boardStatFile << "Epoch, BoardSize, Samples, ValueValLoss, PolicyValLoss, Timestamp\n";
        // Run tuner
        auto recordStatistic = [&](TuningStatistic stat) {
            if (!curriculumPhases.empty() && stat.currentPhase != compactPolicyPhase) {
                compactPolicyPhase     = stat.currentPhase;
                compactPolicyBaseReady = false;
            }
            if (policyTrainer && stat.trainCompactPolicy) {
                if (stat.currentPhaseEpoch > 0) {
                    if (!compactPolicyBaseReady) {
                        policyTrainer->refreshBaseScores();
                        compactPolicyBaseReady = true;
                    }
                    policyTrainer->step();
                }
                else if (!compactPolicyBaseReady) {
                    policyTrainer->refreshBaseScores();
                    compactPolicyBaseReady = true;
                }
            }
            else
                compactPolicyBaseReady = false;
            // Log statistics for current epoch
            totalElapsedSeconds += stat.elapsedSeconds;
            if (!curriculumPhases.empty())
                statFile << stat.currentPhase << ", " << stat.currentPhaseEpoch << ", "
                         << stat.phaseName << ", ";
            statFile << std::setprecision(std::numeric_limits<double>::digits10)  //
                     << stat.currentEpoch << ", "                                 //
                     << stat.learningRate << ", "                                 //
                     << stat.valueLoss << ", "                                    //
                     << stat.policyLoss << ", "                                   //
                     << stat.valueValLoss << ", "                                 //
                     << stat.policyValLoss << ", "                                //
                     << stat.elapsedSeconds << ", "                               //
                     << stat.currentEpoch / totalElapsedSeconds << ", "           //
                     << std::time(0) << std::endl;
            if (!statFile)
                throw std::runtime_error("failed to write tuning statistics");
            for (const auto &board : stat.validationByBoard) {
                if (!curriculumPhases.empty())
                    boardStatFile << stat.currentPhase << ", " << stat.currentPhaseEpoch << ", "
                                  << stat.phaseName << ", ";
                boardStatFile << std::setprecision(std::numeric_limits<double>::digits10)  //
                              << stat.currentEpoch << ", "                                 //
                              << unsigned(board.boardSize) << ", "                         //
                              << board.samples << ", "                                     //
                              << board.valueLoss << ", "                                   //
                              << board.policyLoss << ", "                                  //
                              << std::time(0) << std::endl;
            }
            if (!boardStatFile)
                throw std::runtime_error("failed to write per-board tuning statistics");
            // Export model periodically
            if (modelExportInterval && stat.currentEpoch > 0
                    && stat.currentEpoch % modelExportInterval == 0
                || stat.currentEpoch == epochs) {
                constexpr size_t EpochDigits = 5;
                std::string      epochStr    = std::to_string(stat.currentEpoch);
                if (epochStr.length() < EpochDigits)
                    epochStr = std::string(EpochDigits - epochStr.length(), '0') + epochStr;

                // Save params and scaling factor to config
                tuner->saveParams();
                Evaluation::ScalingFactor = stat.scalingFactor;

                // Export model file
                std::string                 modelFileName = trainName + "-e" + epochStr + ".bin";
                const std::filesystem::path modelPath     = outpath / modelFileName;
                std::optional<PolicyTrainingResult> policyResult;
                if (policyTrainer)
                    policyResult = policyTrainer->saveModel(modelPath);
                else {
                    std::ofstream model(modelPath, std::ios::binary);
                    if (!model)
                        throw std::runtime_error("unable to open model output " + modelFileName);
                    Config::exportModel(model);
                    model.flush();
                    if (!model)
                        throw std::runtime_error("failed to write model output " + modelFileName);
                    model.close();
                }

                if (policyResult)
                    MESSAGEL("Joint policy parameters: " << policyResult->parameterCount
                                                         << ", range " << policyResult->minimum
                                                         << ".." << policyResult->maximum
                                                         << ", RMS " << policyResult->rms << '.');
                MESSAGEL("Model saved to " << modelFileName);
            }
        };
        if (curriculumPhases.empty())
            tuner->run(epochs, recordStatistic);
        else
            tuner->runCurriculum(recordStatistic);

        std::ofstream boardSampleFile(outpath / "board-samples.csv");
        if (!boardSampleFile)
            throw std::runtime_error("unable to open board sample statistic output");
        if (curriculumPhases.empty()) {
            boardSampleFile
                << "BoardSize, TrainingSamples, OptimizedTrainingSamples, ValidationSamples, "
                   "TrainingWeight\n";
            for (const BoardSampleStatistic &board : tuner->boardSampleStatistics())
                boardSampleFile << unsigned(board.boardSize) << ", " << board.trainingSamples
                                << ", " << board.optimizedTrainingSamples << ", "
                                << board.validationSamples << ", "
                                << std::setprecision(std::numeric_limits<double>::digits10)
                                << board.trainingWeight << '\n';
        }
        else {
            boardSampleFile
                << "Phase, PhaseName, BoardSize, TrainingSamples, OptimizedTrainingSamples, "
                   "ValidationSamples, TrainingWeight\n";
            for (size_t phaseIndex = 0; phaseIndex < curriculumPhases.size(); phaseIndex++)
                for (const BoardSampleStatistic &board : tuner->boardSampleStatistics(phaseIndex))
                    boardSampleFile
                        << phaseIndex << ", " << curriculumPhases[phaseIndex].name << ", "
                        << unsigned(board.boardSize) << ", " << board.trainingSamples << ", "
                        << board.optimizedTrainingSamples << ", " << board.validationSamples << ", "
                        << std::setprecision(std::numeric_limits<double>::digits10)
                        << board.trainingWeight << '\n';
        }
        boardSampleFile.flush();
        if (!boardSampleFile)
            throw std::runtime_error("failed to write board sample statistics");
        return EXIT_SUCCESS;
    }
    catch (const std::exception &e) {
        ERRORL("Error occurred when tuning: " << e.what());
        return EXIT_FAILURE;
    }
}
