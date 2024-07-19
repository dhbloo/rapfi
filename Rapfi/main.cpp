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

#include "command/command.h"
#include "core/iohelper.h"
#include "search/searchthread.h"

#ifdef COMMAND_MODULES
    #define CXXOPTS_NO_REGEX
    #include <cxxopts.hpp>
#endif

#include <stdexcept>

#ifdef _WIN32
    #include <Windows.h>
#endif

int main(int argc, char *argv[])
{
    Command::CommandLine::init(argc, argv);

#ifdef COMMAND_MODULES
    // The running mode of current Rapfi instance
    enum RunMode {
        GOMOCUP_PROTOCOL,
        BENCHMARK,
        OPENGEN,
        TUNING,
        SELFPLAY,
        DATAPREP,
        DATABASE,
    } runMode = GOMOCUP_PROTOCOL;

    {
        cxxopts::Options options("rapfi");
        options.add_options()  //
            ("mode",
             "One of [gomocup, bench, opengen, tuning, selfplay, dataprep, database] run modes",
             cxxopts::value<std::string>()->default_value("gomocup"))  //
            ("config",
             "Path to the specified config file",
             cxxopts::value<std::string>())  //
            ("force-utf8",
             "Force to use utf-8 encoding for stdin and stdout (for Windows)")  //
            ("h,help", "Print usage");
        options.parse_positional("mode");
        options.positional_help("[mode]");
        options.show_positional_help();
        options.allow_unrecognised_options();

        try {
            auto result = options.parse(argc, argv);

            std::string mode = result["mode"].as<std::string>();
            upperInplace(mode);
            if (mode == "GOMOCUP")
                runMode = GOMOCUP_PROTOCOL;
            else if (mode == "BENCH")
                runMode = BENCHMARK;
            else if (mode == "OPENGEN")
                runMode = OPENGEN;
            else if (mode == "TUNING")
                runMode = TUNING;
            else if (mode == "SELFPLAY")
                runMode = SELFPLAY;
            else if (mode == "DATAPREP")
                runMode = DATAPREP;
            else if (mode == "DATABASE")
                runMode = DATABASE;
            else
                throw std::invalid_argument("unknown mode " + mode);

            if (result.count("help") && (runMode == GOMOCUP_PROTOCOL || runMode == BENCHMARK)) {
                std::cout << options.help() << std::endl;
                std::exit(EXIT_SUCCESS);
            }

            if (result.count("config")) {
                Command::configPath          = result["config"].as<std::string>();
                Command::allowInternalConfig = false;
            }

    #ifdef _WIN32
            if (result.count("force-utf8")) {
                SetConsoleCP(CP_UTF8);
                SetConsoleOutputCP(CP_UTF8);
            }
    #endif
        }
        catch (const std::exception &e) {
            ERRORL("parsing argument: " << e.what());
            return EXIT_FAILURE;
        }
    }
#endif

    if (!Command::loadConfig()) {
        ERRORL("Failed to load config, please check if config is correct.");
        return EXIT_FAILURE;
    }

#ifdef COMMAND_MODULES
    switch (runMode) {
    case BENCHMARK: Command::benchmark(); break;
    case OPENGEN: Command::opengen(argc, argv); break;
    case TUNING: Command::tuning(argc, argv); break;
    case SELFPLAY: Command::selfplay(argc, argv); break;
    case DATAPREP: Command::dataprep(argc, argv); break;
    case DATABASE: Command::database(argc, argv); break;
    default: Command::gomocupLoop(); break;
    }
#else
    Command::gomocupLoop();
#endif

    return EXIT_SUCCESS;
}
