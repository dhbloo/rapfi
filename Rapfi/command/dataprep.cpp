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

#include "../core/iohelper.h"
#include "../core/utils.h"
#include "../tuning/dataset.h"
#include "../tuning/datawriter.h"
#include "argutils.h"
#include "command.h"

#define CXXOPTS_NO_REGEX
#include <ctime>
#include <cxxopts.hpp>
#include <filesystem>
#include <fstream>
#include <memory>
#include <stdexcept>

namespace {

enum class DatasetType { PackedBinary, KatagoNumpy };
DatasetType parseDatasetType(std::string dsType)
{
    if (dsType == "binary")
        return DatasetType::PackedBinary;
    else if (dsType == "katago")
        return DatasetType::KatagoNumpy;
    else
        throw std::invalid_argument("unknown dataset type " + dsType);
}

}  // namespace

using namespace Tuning;

void Command::dataprep(int argc, char *argv[])
{
    std::unique_ptr<Dataset>    inputDataset;
    std::unique_ptr<DataWriter> dataWriter;
    DatasetType                 datasetType;
    Rule                        defaultRule;
    std::vector<std::string>    pathList;
    std::vector<std::string>    extensions;
    std::string                 outdir;
    size_t                      maxNumEntriesPerFile;
    Time                        reportInterval;

    cxxopts::Options options("rapfi data preparation utility"
                             "\nConverting input dataset to numpy dataset for external trainer.");
    options.add_options()                                                //
        ("o,output", "Output directory", cxxopts::value<std::string>())  //
        ("d,dataset",
         "Input dataset filename/directory(s)",
         cxxopts::value<std::vector<std::string>>())  //
        ("t,dataset-type",
         "Input dataset type, one of [binary, katago]",
         cxxopts::value<std::string>())  //
        ("r,default-rule",
         "Default rule for dataset type that does not contain rule infomation",
         cxxopts::value<std::string>()->default_value("freestyle"))  //
        ("dataset-file-extensions",
         "Extensions to filter dataset file in a directory",
         cxxopts::value<std::vector<std::string>>()->default_value(".bin,.lz4,.npz"))  //
        ("max-entries-per-file",
         "Max number of entries per npz file",
         cxxopts::value<size_t>()->default_value("25000"))  //
        ("report-interval",
         "Time (ms) between two progress report message",
         cxxopts::value<Time>()->default_value("10000"))  //
        ("h,help", "Print dataprep usage");

    try {
        auto args = options.parse(argc, argv);

        if (args.count("help")) {
            std::cout << options.help() << std::endl;
            std::exit(EXIT_SUCCESS);
        }

        if (!args.count("dataset"))
            throw std::invalid_argument("there must be at least one input dataset");
        if (!args.count("dataset-type"))
            throw std::invalid_argument("must specify the dataset type");

        datasetType          = parseDatasetType(args["dataset-type"].as<std::string>());
        defaultRule          = parseRule(args["default-rule"].as<std::string>());
        pathList             = args["dataset"].as<std::vector<std::string>>();
        extensions           = args["dataset-file-extensions"].as<std::vector<std::string>>();
        outdir               = args["output"].as<std::string>();
        maxNumEntriesPerFile = args["max-entries-per-file"].as<size_t>();
        reportInterval       = args["report-interval"].as<Time>();
    }
    catch (const std::exception &e) {
        ERRORL("dataprep argument: " << e.what());
        std::exit(EXIT_FAILURE);
    }

    try {
        // Create output directory
        ensureDir(outdir);

        // Make path list
        pathList = makeFileListFromPathList(pathList, extensions);

        // Create input dataset
        switch (datasetType) {
        case DatasetType::PackedBinary:
            inputDataset = std::make_unique<PackedBinaryDataset>(pathList);
            break;
        case DatasetType::KatagoNumpy:
            inputDataset = std::make_unique<KatagoNumpyDataset>(pathList, defaultRule);
            break;
        }

        size_t numEntriesProcessed = 0;
        size_t numFilesWrote       = 0;

        // Create numpy data writer
        dataWriter = std::make_unique<NumpyDataWriter>(outdir,
                                                       maxNumEntriesPerFile,
                                                       [&](std::string filename) {
                                                           MESSAGEL("Wrote to " << filename);
                                                           numFilesWrote++;
                                                       });

        // Start processing loop
        Time      startTime = now(), lastTime = startTime;
        DataEntry entry;
        while (inputDataset->next(&entry)) {
            dataWriter->writeEntry(entry);
            numEntriesProcessed++;

            // Print out processing progress over time
            if (now() - lastTime >= reportInterval) {
                MESSAGEL("Processed " << numEntriesProcessed << " entries, wrote " << numFilesWrote
                                      << " files, entry/s = "
                                      << numEntriesProcessed / ((now() - startTime) / 1000.0));
                lastTime = now();
            }
        }

        dataWriter.reset();  // flush entries in buffer
        MESSAGEL("Finished processing " << numEntriesProcessed << " entries.");
    }
    catch (const std::exception &e) {
        ERRORL("Error occurred when preparing data: " << e.what());
    }
}
