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
#include "../tuning/dataset.h"
#include "../tuning/tuner.h"
#include "argutils.h"
#include "command.h"

#define CXXOPTS_NO_REGEX
#include <ctime>
#include <cxxopts.hpp>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
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

void validateConfig(size_t epochs, const TuningConfig &cfg)
{
    if (epochs < 1)
        throw std::invalid_argument("epochs must be greater than 0");
    if (!(cfg.tuneRule[FREESTYLE] || cfg.tuneRule[STANDARD] || cfg.tuneRule[RENJU]))
        throw std::invalid_argument("there must be at least one rule to tune");
    if (cfg.batchSize < 1)
        throw std::invalid_argument("batchsize must be greater than 0");
    if (cfg.moveScoreLossGamma < 0)
        throw std::invalid_argument("move-score-loss-gamma must be not less than 0");
    if (cfg.boardSizeMin > cfg.boardSizeMax)
        throw std::invalid_argument("max-boardsize must be greater than min-boardsize");
    if (!cfg.usePreviousScalingFactor) {
        if (cfg.nIterations < 1 || cfg.nStepsPerIteration < 1)
            throw std::invalid_argument(
                "num-iteration and num-steps-per-iteration must be greater than 0");
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

}  // namespace

void Command::tuning(int argc, char *argv[])
{
    std::string              outdir;
    std::string              trainName;
    size_t                   epochs;
    size_t                   modelExportInterval;
    TuningConfig             cfg = {};
    DatasetType              trainDatasetType;
    DatasetType              valDatasetType;
    std::vector<std::string> trainDatasetPathList;
    std::vector<std::string> valDatasetPathList;
    std::vector<std::string> extensions;
    std::unique_ptr<Dataset> trainDataset, valDataset;

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
        ("l,learning-rate",
         "Learning rate for gradient descent",
         cxxopts::value<double>()->default_value(std::to_string(cfg.learningRate)))  //
        ("w,weight-decay",
         "Weight dacay for gradient descent (0.0~1.0)",
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
        ("move-score-scale",
         "Scale of move score conversion from float to int",
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
        ("random-move-score-init", "Init move score parameters to [0,1]")                  //
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

    try {
        auto args = options.parse(argc, argv);

        if (args.count("help")) {
            std::cout << options.help() << std::endl;
            std::exit(EXIT_SUCCESS);
        }

        parseTuningRules(cfg, args["rules-to-tune"].as<std::vector<std::string>>());
        trainDatasetType     = parseDatasetType(args["training-dataset-type"].as<std::string>());
        trainDatasetPathList = args["training-dataset"].as<std::vector<std::string>>();
        if (args.count("validation-dataset")) {
            valDatasetType = parseDatasetType(args["validation-dataset-type"].as<std::string>());
            valDatasetPathList = args["validation-dataset"].as<std::vector<std::string>>();
        }
        extensions             = args["dataset-file-extensions"].as<std::vector<std::string>>();
        outdir                 = args["output"].as<std::string>();
        trainName              = args["name"].as<std::string>();
        epochs                 = args["epochs"].as<size_t>();
        modelExportInterval    = args["export-interval"].as<size_t>();
        cfg.batchSize          = args["batchsize"].as<size_t>();
        cfg.maxTuneEntries     = args["max-entries"].as<size_t>();
        cfg.learningRate       = args["learning-rate"].as<double>();
        cfg.weightDecay        = args["weight-decay"].as<double>();
        cfg.lossType           = parseLossType(args["loss"].as<std::string>());
        cfg.shuffleTuneEntries = args.count("shuffle");
        cfg.tuneMoveScore      = args.count("tune-move-score");
        cfg.tuneEval           = !args.count("no-tune-eval");
        cfg.moveScoreLossGamma = args["move-score-loss-gamma"].as<double>();
        cfg.moveScoreScale     = args["move-score-scale"].as<double>();
        cfg.moveScoreBias      = args["move-score-bias"].as<double>();
        cfg.moveScoreMin       = args["move-score-min"].as<Score>();
        cfg.moveScoreMax       = args["move-score-max"].as<Score>();
        cfg.boardSizeMin       = args["min-boardsize"].as<uint8_t>();
        cfg.boardSizeMax       = args["max-boardsize"].as<uint8_t>();
        cfg.minPly             = args["min-ply"].as<uint16_t>();
        cfg.minPlyBeforeFull   = args["min-ply-before-full"].as<uint16_t>();
        cfg.usePreviousScalingFactor = args.count("fix-scaling-factor");
        cfg.randomMoveScoreInit      = args.count("random-move-score-init");
        cfg.nIterations              = args["num-iteration"].as<int>();
        cfg.nStepsPerIteration       = args["num-steps-per-iteration"].as<int>();
        cfg.scalingFactorMin         = args["scaling-factor-lower-bound"].as<double>();
        cfg.scalingFactorMax         = args["scaling-factor-upper-bound"].as<double>();
        cfg.recomputeInterval        = args["recompute-interval"].as<size_t>();

        validateConfig(epochs, cfg);
    }
    catch (const std::exception &e) {
        ERRORL("tuning argument: " << e.what());
        std::exit(EXIT_FAILURE);
    }

    try {
        // Create output directory
        ensureDir(outdir);
        std::filesystem::path outpath = outdir;

        // Make path list
        trainDatasetPathList = makeFileListFromPathList(trainDatasetPathList, extensions);
        trainDataset         = createDataset(trainDatasetType, trainDatasetPathList);
        if (!valDatasetPathList.empty()) {
            valDatasetPathList = makeFileListFromPathList(valDatasetPathList, extensions);
            valDataset         = createDataset(valDatasetType, valDatasetPathList);
        }

        // Create tuner with dataset and tunerConfig
        Tuner tuner(*trainDataset, valDataset.get(), cfg);

        // Open and init statistic CSV file
        std::ofstream statFile(outpath / "stat.csv");
        double        totalElapsedSeconds = 0.0;
        statFile << "Epoch, ValueLoss, PolicyLoss, ValueValLoss, PolicyValLoss, "
                    "Elapsed, Epochs/Sec, Timestamp\n";

        // Run tuner
        tuner.run(epochs, [&](TuningStatistic stat) {
            // Log statistics for current epoch
            totalElapsedSeconds += stat.elapsedSeconds;
            statFile << std::setprecision(std::numeric_limits<double>::digits10)  //
                     << stat.currentEpoch << ", "                                 //
                     << stat.valueLoss << ", "                                    //
                     << stat.policyLoss << ", "                                   //
                     << stat.valueValLoss << ", "                                 //
                     << stat.policyValLoss << ", "                                //
                     << stat.elapsedSeconds << ", "                               //
                     << stat.currentEpoch / totalElapsedSeconds << ", "           //
                     << std::time(0) << std::endl;

            // Export model periodically
            if (modelExportInterval && stat.currentEpoch > 0
                    && stat.currentEpoch % modelExportInterval == 0
                || stat.currentEpoch == epochs) {
                constexpr size_t EpochDigits = 5;
                std::string      epochStr    = std::to_string(stat.currentEpoch);
                if (epochStr.length() < EpochDigits)
                    epochStr = std::string(EpochDigits - epochStr.length(), '0') + epochStr;

                // Save params and scaling factor to config
                tuner.saveParams();
                Config::ScalingFactor = stat.scalingFactor;

                // Export model file
                std::string   modelFileName = trainName + "-e" + epochStr + ".bin";
                std::ofstream model(outpath / modelFileName, std::ios::binary);
                Config::exportModel(model);

                MESSAGEL("Model saved to " << modelFileName);
            }
        });
    }
    catch (const std::exception &e) {
        ERRORL("Error occurred when tuning: " << e.what());
    }
}
