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

#include "argutils.h"

#ifndef NO_COMMAND_MODULES
    #define CXXOPTS_NO_REGEX
    #include <cxxopts.hpp>
#endif
#include <string>

Rule Command::parseRule(std::string_view ruleStr)
{
    if (ruleStr == "freestyle" || ruleStr == "Freestyle" || ruleStr == "f" || ruleStr == "F"
        || ruleStr == "0")
        return Rule::FREESTYLE;
    else if (ruleStr == "standard" || ruleStr == "Standard" || ruleStr == "s" || ruleStr == "S"
             || ruleStr == "1")
        return Rule::STANDARD;
    else if (ruleStr == "renju" || ruleStr == "Renju" || ruleStr == "r" || ruleStr == "R"
             || ruleStr == "2" || ruleStr == "4")
        return Rule::RENJU;
    else
        throw std::invalid_argument("unknown rule " + std::string(ruleStr));
}

Command::DatasetType Command::parseDatasetType(std::string dsType)
{
    if (dsType == "bin")
        return DatasetType::SimpleBinary;
    else if (dsType == "binpack")
        return DatasetType::PackedBinary;
    else if (dsType == "katago")
        return DatasetType::KatagoNumpy;
    else
        throw std::invalid_argument("unknown dataset type " + dsType);
}

Command::DataWriterType Command::parseDataWriterType(std::string dwType)
{
    if (dwType == "txt")
        return DataWriterType::PlainText;
    else if (dwType == "bin")
        return DataWriterType::SimpleBinary;
    else if (dwType == "bin_lz4")
        return DataWriterType::SimpleBinaryLZ4;
    else if (dwType == "binpack")
        return DataWriterType::PackedBinary;
    else if (dwType == "binpack_lz4")
        return DataWriterType::PackedBinaryLZ4;
    else if (dwType == "numpy")
        return DataWriterType::Numpy;
    else
        throw std::invalid_argument("unknown data writer type " + dwType);
}

std::vector<Pos>
Command::parsePositionString(std::string_view positionString, int boardWidth, int boardHeight)
{
    // Convert Pos format to int string
    std::stringstream ss;
    for (char ch : positionString) {
        if (ch >= 'a' && ch <= 'z')
            ss << ' ' << int(ch - 'a' + 1) << ' ';
        else if (ch >= '0' && ch <= '9')
            ss << ch;
        else
            throw std::invalid_argument("invalid position string " + std::string(positionString));
    }

    // Read coordinates from stream
    std::vector<int> coords;
    int              coord = 0;
    while (ss >> coord) {
        int size = coords.size() % 2 == 0 ? boardWidth : boardHeight;
        if (coord <= 0 || coord > size)
            throw std::invalid_argument("invalid position coord " + std::to_string(coord) + " in "
                                        + std::string(positionString));
        coords.push_back(coord - 1);
    }

    // Num coords must be even
    if (coords.size() % 2 != 0)
        throw std::invalid_argument("incomplete position string " + std::string(positionString));

    // Convert coords to pos vector
    std::vector<Pos> position;
    position.reserve(coords.size() / 2);
    for (size_t i = 0; i < coords.size(); i += 2) {
        int x = coords[i], y = coords[i + 1];
        Pos pos {x, y};
        if (std::find(position.begin(), position.end(), pos) != position.end())
            throw std::invalid_argument("duplicate position coord in "
                                        + std::string(positionString));
        position.push_back(pos);
    }

    return position;
}

#ifndef NO_COMMAND_MODULES

void Command::addPlayOptions(cxxopts::Options &options)
{
    options.add_options("play-general")                                                      //
        ("s,boardsize", "Board size in [5,22]", cxxopts::value<int>()->default_value("15"))  //
        ("r,rule",
         "One of [freestyle, standard, renju] rule",
         cxxopts::value<std::string>()->default_value("freestyle"))  //
        ("t,thread",
         "Number of search threads to use for searching balanced moves",
         cxxopts::value<size_t>()->default_value(std::to_string(Config::DefaultThreadNum)))  //
        ("hashsize",
         "Hash size of the transposition table (in MB)",
         cxxopts::value<size_t>()->default_value("128"))  //
        ("q,no-search-message",
         "Disable message output during search")  //
        ;
}

void Command::addOpengenOptions(cxxopts::Options &options, const Opening::OpeningGenConfig &cfg)
{
    options.add_options("opengen-configs")  //
        ("min-move",
         "Minimal number of moves per opening",
         cxxopts::value<int>()->default_value(std::to_string(cfg.minMoves)))  //
        ("max-move",
         "Maximal number of moves per opening",
         cxxopts::value<int>()->default_value(std::to_string(cfg.maxMoves)))  //
        ("min-area-size",
         "Minimal size of local area",
         cxxopts::value<int>()->default_value(std::to_string(cfg.localSizeMin)))  //
        ("max-area-size",
         "Maximal size of local area",
         cxxopts::value<int>()->default_value(std::to_string(cfg.localSizeMax)))  //
        ("balance1-node",
         "Maximal nodes for balance1 search",
         cxxopts::value<uint64_t>()->default_value(std::to_string(cfg.balance1Nodes)))  //
        ("balance1-fast-check-ratio",
         "Spend how much amount of nodes to fast check if this position is balanceable",
         cxxopts::value<double>()->default_value(
             std::to_string((double)cfg.balance1FastCheckNodes / (double)cfg.balance1Nodes)))  //
        ("balance1-fast-check-window",
         "Consider this position as unbalanceable if its initial value falls outside the window",
         cxxopts::value<int>()->default_value(std::to_string(cfg.balance1FastCheckWindow)))  //
        ("balance2-node",
         "Maximal nodes for balance2 search",
         cxxopts::value<uint64_t>()->default_value(std::to_string(cfg.balance2Nodes)))  //
        ("balance-window",
         "Eval in [-window, window] is considered as balanced",
         cxxopts::value<int>()->default_value(std::to_string(cfg.balanceWindow)))  //
        ;
}

Opening::OpeningGenConfig Command::parseOpengenConfig(const cxxopts::ParseResult &result)
{
    Opening::OpeningGenConfig cfg;

    cfg.minMoves      = result["min-move"].as<int>();
    cfg.maxMoves      = result["max-move"].as<int>();
    cfg.localSizeMin  = result["min-area-size"].as<int>();
    cfg.localSizeMax  = result["max-area-size"].as<int>();
    cfg.balance1Nodes = result["balance1-node"].as<size_t>();
    cfg.balance1FastCheckNodes =
        static_cast<uint64_t>(result["balance1-fast-check-ratio"].as<double>() * cfg.balance1Nodes);
    cfg.balance1FastCheckWindow = Value(result["balance1-fast-check-window"].as<int>());
    cfg.balance2Nodes           = result["balance2-node"].as<size_t>();
    cfg.balanceWindow           = Value(result["balance-window"].as<int>());

    if (cfg.minMoves <= 0 || cfg.maxMoves < cfg.minMoves)
        throw std::invalid_argument("condition 0 < minMove <= maxMove does no satisfy");
    if (cfg.localSizeMin < 0 || cfg.localSizeMax < cfg.localSizeMin)
        throw std::invalid_argument("condition 0 < minAreaSize <= maxAreaSize does no satisfy");
    if (cfg.balanceWindow < 0)
        throw std::invalid_argument("balancewindow must be greater than 0");

    return cfg;
}

#endif
