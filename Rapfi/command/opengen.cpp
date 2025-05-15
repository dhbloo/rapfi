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
#include "../search/hashtable.h"
#include "../search/opening.h"
#include "../search/searchthread.h"
#include "argutils.h"
#include "command.h"

#define CXXOPTS_NO_REGEX
#include <cxxopts.hpp>
#include <fstream>
#include <stdexcept>

void Command::opengen(int argc, char *argv[])
{
    size_t                    numOpenings;
    size_t                    numThreads;
    size_t                    hashSizeMb;
    int                       boardsize;
    Rule                      rule;
    Time                      reportInterval;
    bool                      silence;
    Opening::OpeningGenConfig cfg;
    std::ostream             *os = &std::cout;
    std::ofstream             outfile;

    cxxopts::Options options("rapfi opengen");
    options.add_options()                                                         //
        ("n,number", "Number of openings to generate", cxxopts::value<size_t>())  //
        ("o,output",
         "Save openings to a text file (default to stdout if not specified)",
         cxxopts::value<std::string>())  //
        ("a,append-to-output",
         "Append results to the output file without overwritting it")  //
        ("report-interval",
         "Time (ms) between two progress report message",
         cxxopts::value<Time>()->default_value("10000"))  //
        ("h,help", "Print opengen usage");
    addPlayOptions(options);
    addOpengenOptions(options, cfg);

    try {
        auto args = options.parse(argc, argv);

        if (args.count("help")) {
            std::cout << options.help() << std::endl;
            std::exit(EXIT_SUCCESS);
        }

        if (args.count("output")) {
            // Open output file and change output stream
            std::string filename   = args["output"].as<std::string>();
            bool        appendMode = args.count("append-to-output");

            outfile.open(filename, appendMode ? std::ios::app : std::ios::out);
            if (!outfile.is_open())
                throw std::invalid_argument("unable to open file " + filename);
            os = &outfile;
        }

        numOpenings    = args["number"].as<size_t>();
        rule           = parseRule(args["rule"].as<std::string>());
        numThreads     = std::max<size_t>(args["thread"].as<size_t>(), 1);
        boardsize      = args["boardsize"].as<int>();
        hashSizeMb     = std::max<size_t>(args["hashsize"].as<size_t>(), 1);
        reportInterval = args["report-interval"].as<Time>();
        silence        = args.count("no-search-message");
        cfg            = parseOpengenConfig(args);

        if (numOpenings < 1)
            throw std::invalid_argument("there must be at least one opening to generate");
        if (boardsize < 5 || boardsize > MAX_BOARD_SIZE)
            throw std::invalid_argument("boardsize must be in range [5,22]");
    }
    catch (const std::exception &e) {
        ERRORL("opengen argument: " << e.what());
        std::exit(EXIT_FAILURE);
    }

    // Set message mode to none if silence search is enabled
    if (silence)
        Config::MessageMode = MsgMode::NONE;

    // Set number of iterations after special search condition
    Config::NumIterationAfterMate         = 0;
    Config::NumIterationAfterSingularRoot = 64;

    // Start generating openings and write position strings to output stream
    Search::Threads.searcher()->setMemoryLimit(hashSizeMb * 1024);
    Search::Threads.setNumThreads(numThreads);
    Search::Threads.clear(false);

    Opening::OpeningGenerator og(boardsize, rule, cfg);
    Time                      startTime = now(), lastTime = startTime;
    for (size_t i = 0; i < numOpenings;) {
        bool balanced = og.next();

        // If we want to generate a balanced opening, abandon those not balanced
        if (!balanced && cfg.balanceWindow > 0 && (cfg.balance1Nodes > 0 || cfg.balance2Nodes > 0))
            continue;

        i++;
        (*os) << og.positionString() << std::endl;

        // Print out generation progress over time
        if (now() - lastTime >= reportInterval) {
            MESSAGEL("Generated " << i << " of " << numOpenings << " openings, opening/min = "
                                  << i / ((now() - startTime) / 60000.0));
            lastTime = now();
        }
    }
    MESSAGEL("Completed generating " << numOpenings << " openings.");
}
