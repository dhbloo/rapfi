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

#include "../core/filesystem.h"
#include "../core/iohelper.h"
#include "../core/time.h"
#include "../eval/scoretables.h"
#include "../tuning/dataset.h"
#include "../tuning/datawriter.h"
#include "argutils.h"
#include "command.h"

#define CXXOPTS_NO_REGEX
#include <cmath>
#include <cstdlib>
#include <ctime>
#include <cxxopts.hpp>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>

using namespace Tuning;

void Command::dataprep(int argc, char *argv[])
{
    std::unique_ptr<Dataset>    inputDataset;
    std::unique_ptr<DataWriter> dataWriter;
    DatasetType                 datasetType;
    DataWriterType              dataWriterType;
    Rule                        defaultRule;
    std::vector<std::string>    pathList;
    std::vector<std::string>    extensions;
    std::string                 outputPath;
    size_t                      maxNumEntriesPerFile;
    Time                        reportInterval;
    PolicyTargetConfig          policyTargetConfig;

    cxxopts::Options options("rapfi data preparation utility"
                             "\nConverting the input dataset to the target dataset.");
    options.add_options()                                                         //
        ("o,output", "Output filename/directory", cxxopts::value<std::string>())  //
        ("i,input",
         "Input dataset filename/directory(s)",
         cxxopts::value<std::vector<std::string>>())  //
        ("input-type",
         "Input dataset type, one of [bin, binpack, katago]",
         cxxopts::value<std::string>())  //
        ("output-type",
         "Output dataset type, one of [txt, bin, bin_lz4, binpack, binpack_lz4, numpy]",
         cxxopts::value<std::string>()->default_value("numpy"))  //
        ("default-rule",
         "Default rule for dataset type that does not contain rule information",
         cxxopts::value<std::string>()->default_value("freestyle"))  //
        ("dataset-file-extensions",
         "Extensions to filter dataset file in a directory",
         cxxopts::value<std::vector<std::string>>()->default_value(".bin,.binpack,.lz4,.npz"))  //
        ("max-entries-per-file",
         "Max number of entries per NPZ file",
         cxxopts::value<size_t>()->default_value("25000"))  //
        ("multipv-policy-temperature",
         "Temperature for softmax-over-winrates multi-PV policy targets (0 disables)",
         cxxopts::value<double>()->default_value("0"))  //
        ("report-interval",
         "Time (ms) between two progress report message",
         cxxopts::value<Time>()->default_value("10000"))  //
        ("h,help", "Print dataprep usage");

    parseSubcommandArguments(
        options,
        argc,
        argv,
        "dataprep argument",
        [&](const cxxopts::ParseResult &args) {
            if (!args.count("input"))
                throw std::invalid_argument("there must be at least one input dataset");

            datasetType          = parseDatasetType(args["input-type"].as<std::string>());
            dataWriterType       = parseDataWriterType(args["output-type"].as<std::string>());
            defaultRule          = parseRule(args["default-rule"].as<std::string>());
            pathList             = args["input"].as<std::vector<std::string>>();
            extensions           = args["dataset-file-extensions"].as<std::vector<std::string>>();
            outputPath           = args["output"].as<std::string>();
            maxNumEntriesPerFile = args["max-entries-per-file"].as<size_t>();
            reportInterval       = args["report-interval"].as<Time>();

            double multiPVTemperature = args["multipv-policy-temperature"].as<double>();
            if (!std::isfinite(multiPVTemperature) || multiPVTemperature < 0.0)
                throw std::invalid_argument(
                    "multipv-policy-temperature must be finite and nonnegative");
            if (multiPVTemperature > 0.0 && datasetType != DatasetType::PackedBinary)
                throw std::invalid_argument("multipv-policy-temperature requires binpack input");
            if (multiPVTemperature > 0.0 && dataWriterType != DataWriterType::Numpy)
                throw std::invalid_argument("multipv-policy-temperature requires numpy output");

            float storedTemperature = float(multiPVTemperature);
            if (!std::isfinite(storedTemperature)
                || (multiPVTemperature > 0.0 && storedTemperature == 0.0f))
                throw std::invalid_argument(
                    "multipv-policy-temperature is outside the supported range");

            policyTargetConfig.multiPVTemperature = storedTemperature;
            policyTargetConfig.evalScalingFactor  = Evaluation::ScalingFactor;
        });

    try {
        // Make path list
        pathList = makeFileListFromPathList(pathList, extensions);

        // Create input dataset
        switch (datasetType) {
        case DatasetType::SimpleBinary:
            inputDataset = std::make_unique<SimpleBinaryDataset>(pathList);
            break;

        case DatasetType::PackedBinary:
            inputDataset = std::make_unique<PackedBinaryDataset>(pathList);
            break;

        case DatasetType::KatagoNumpy:
            inputDataset = std::make_unique<KatagoNumpyDataset>(pathList, defaultRule);
            break;
        }

        // Create data writer
        if (dataWriterType == DataWriterType::Numpy)
            dataWriter = std::make_unique<NumpyDataWriter>(
                outputPath,
                maxNumEntriesPerFile,
                policyTargetConfig,
                [&, numFilesWrote = 0](std::string filename) mutable {
                    MESSAGEL("Wrote npz to " << filename << ", saved " << ++numFilesWrote
                                             << " npz in total.");
                });
        else
            dataWriter = Tuning::makeDataWriter(dataWriterType, outputPath);

        // Start processing loop
        size_t numEntriesProcessed = 0;
        Time   startTime = now(), lastTime = startTime;

        auto reportProgress = [&]() {
            if (now() - lastTime >= reportInterval) {
                MESSAGEL("Processed " << numEntriesProcessed << " entries, entry/s = "
                                      << numEntriesProcessed / ((now() - startTime) / 1000.0));
                lastTime = now();
            }
        };

        if (inputDataset->supportsGames()) {
            // Game fast path: stream whole games. This preserves game boundaries
            // exactly (no reconstruction from entry chains) and lets game-aware
            // writers process each game incrementally.
            GameEntry game;
            while (inputDataset->nextGame(&game)) {
                dataWriter->writeGame(game);
                numEntriesProcessed += game.moveSequence.size();
                reportProgress();
            }
        }
        else {
            DataEntry entry;
            while (inputDataset->next(&entry)) {
                dataWriter->writeEntry(entry);
                numEntriesProcessed++;
                reportProgress();
            }
        }

        // Flush inside the try so a failed final batch reaches the nonzero-exit
        // path instead of being swallowed by the destructor's logging.
        dataWriter->flush();
        dataWriter.reset();
        MESSAGEL("Finished processing " << numEntriesProcessed << " entries.");
    }
    catch (const std::exception &e) {
        ERRORL("Error occurred when preparing data: " << e.what());
        // Join pending writes before process exit, including batches queued before
        // an input error, so their final filenames cannot contain partial archives.
        dataWriter.reset();
        // Exit nonzero so that pipelines do not mistake partial output for success
        std::exit(EXIT_FAILURE);
    }
}
