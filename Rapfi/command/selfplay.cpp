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
#include "../search/hashtable.h"
#include "../search/searchthread.h"
#include "../tuning/datawriter.h"
#include "argutils.h"
#include "command.h"

#define CXXOPTS_NO_REGEX
#include <csignal>
#include <cxxopts.hpp>
#include <fstream>
#include <functional>
#include <numeric>
#include <random>
#include <stdexcept>

static std::function<void()> signalFunc;
static void                  setupSignalHandler(std::function<void()> handler)
{
    signalFunc         = std::move(handler);
    auto signalHandler = [](int signal) {
        if (signalFunc)
            signalFunc();
    };

    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);
    std::signal(SIGSEGV, signalHandler);
    std::signal(SIGILL, signalHandler);
    std::signal(SIGABRT, signalHandler);
    std::signal(SIGFPE, signalHandler);
#ifdef SIGHUP
    std::signal(SIGHUP, signalHandler);
#endif
#ifdef SIGQUIT
    std::signal(SIGQUIT, signalHandler);
#endif
}

namespace {

std::vector<std::vector<Pos>> readOpenings(std::istream &is, int boardSize)
{
    std::vector<std::vector<Pos>> ops;
    std::string                   opStr;
    size_t                        lineCount = 0;

    while (std::getline(is, opStr)) {
        lineCount++;
        if (opStr.empty())
            continue;

        try {
            ops.push_back(Command::parsePositionString(opStr, boardSize, boardSize));
        }
        catch (const std::exception &e) {
            throw std::runtime_error("illegal opening [line " + std::to_string(lineCount)
                                     + "]: " + e.what());
        }
    }

    return ops;
}

}  // namespace

using namespace Tuning;

void Command::selfplay(int argc, char *argv[])
{
    size_t                        numGames;
    size_t                        numThreads;
    size_t                        hashSizeMb;
    int                           boardSizeMin, boardSizeMax;
    Rule                          rule;
    double                        meanNodes, varNodes;
    int                           matePly;
    int                           drawValue, drawCount, minDrawPly;
    int                           forceDrawPly, forceDrawPlyLeft;
    double                        samplingFreqRoot, samplingFreqPV;
    Time                          reportInterval;
    bool                          sampleMateOnly;
    bool                          generateOpening;
    bool                          silence;
    bool                          compress;
    std::vector<std::vector<Pos>> openings;
    Opening::OpeningGenConfig     opengenCfg;
    std::unique_ptr<DataWriter>   datasetWriter;

    cxxopts::Options options("rapfi selfplay");
    options.add_options()  //
        ("o,output",
         "Save data entries of played games to a binary file",
         cxxopts::value<std::string>())                                    //
        ("n,number", "Number of games to play", cxxopts::value<size_t>())  //
        ("s,boardsize",
         "Board size in [5,22]. Can be overriden by boardsize-min/boardsize-max",
         cxxopts::value<int>()->default_value("15"))                              //
        ("boardsize-min", "Minimal board size in [5,22]", cxxopts::value<int>())  //
        ("boardsize-max", "Maximal board size in [5,22]", cxxopts::value<int>())  //
        ("r,rule",
         "One of [freestyle, standard, renju] rule",
         cxxopts::value<std::string>()->default_value("freestyle"))  //
        ("t,thread",
         "Number of search threads to use",
         cxxopts::value<size_t>()->default_value(std::to_string(Config::DefaultThreadNum)))  //
        ("opening",
         "Path to the opening book file. If not specified, auto generated openings are used",
         cxxopts::value<std::string>())  //
        ("no-compress",
         "Do not compress for saving binary file")  //
        ("hashsize",
         "Hash size of the transposition table (in MB)",
         cxxopts::value<size_t>()->default_value("128"))  //
        ("mean-nodes",
         "Mean of a normal distribution of num search nodes",
         cxxopts::value<double>()->default_value("100000"))  //
        ("var-nodes",
         "Variance of a normal distribution of num search nodes",
         cxxopts::value<double>()->default_value("10000"))  //
        ("mate-ply",
         "Judge win/loss if there is only mate-ply before mate (must be at least 1)",
         cxxopts::value<int>()->default_value("1"))  //
        ("draw-count",
         "Draw if |value| <= draw-value occurred for [draw-count] consecutive moves, and at least "
         "min-draw-ply stones are on board (0 for not enabled)",
         cxxopts::value<int>()->default_value("7"))  //
        ("draw-value",
         "Draw if |value| <= [draw-value] occurred for draw-count consecutive moves, and at least "
         "min-draw-ply stones are on board",
         cxxopts::value<int>()->default_value("10"))  //
        ("min-draw-ply",
         "Draw if |value| <= draw-value occurred for draw-count consecutive moves, and at least "
         "[min-draw-ply] stones are on board",
         cxxopts::value<int>()->default_value("100"))  //
        ("force-draw-ply",
         "Force draw after this ply (0 for not enabled)",
         cxxopts::value<int>()->default_value("0"))  //
        ("force-draw-plyleft",
         "Force draw when left ply is less than this (0 for not enabled)",
         cxxopts::value<int>()->default_value("25"))  //
        ("sampling-freq-root",
         "Sampling frequency when saving data entry from root position",
         cxxopts::value<double>()->default_value("1.0"))  //
        ("sampling-freq-pv",
         "Sampling frequency when saving data entry from PV position",
         cxxopts::value<double>()->default_value("0.0"))  //
        ("sample-mate-only",
         "Sampling only after mate/mated from root position")  //
        ("min-move",
         "Minimal number of moves per opening",
         cxxopts::value<int>()->default_value(std::to_string(opengenCfg.minMoves)))  //
        ("max-move",
         "Maximal number of moves per opening",
         cxxopts::value<int>()->default_value(std::to_string(opengenCfg.maxMoves)))  //
        ("min-area-size",
         "Minimal size of local area",
         cxxopts::value<int>()->default_value(std::to_string(opengenCfg.localSizeMin)))  //
        ("max-area-size",
         "Maximal size of local area",
         cxxopts::value<int>()->default_value(std::to_string(opengenCfg.localSizeMax)))  //
        ("balance1-node",
         "Maximal nodes for balance1 search",
         cxxopts::value<uint64_t>()->default_value(std::to_string(opengenCfg.balance1Nodes)))  //
        ("balance2-node",
         "Maximal nodes for balance2 search",
         cxxopts::value<uint64_t>()->default_value(std::to_string(opengenCfg.balance2Nodes)))  //
        ("balance-window",
         "Eval in [-window, window] is considered as balanced",
         cxxopts::value<int>()->default_value(std::to_string(opengenCfg.balanceWindow)))  //
        ("q,no-search-message",
         "Disable message output during balance move search")  //
        ("report-interval",
         "Time (ms) between two progress report message",
         cxxopts::value<Time>()->default_value("60000"))  //
        ("h,help", "Print selfplay usage");

    try {
        auto args = options.parse(argc, argv);

        if (args.count("help")) {
            std::cout << options.help() << std::endl;
            std::exit(EXIT_SUCCESS);
        }

        if (args.count("output")) {
            // Open output file and change output stream
            std::string filename = args["output"].as<std::string>();
            compress             = !args.count("no-compress");
            datasetWriter        = std::make_unique<PackedBinaryDataWriter>(filename, compress);
        }

        if (args.count("boardsize-min") || args.count("boardsize-max")) {
            boardSizeMin = args["boardsize-min"].as<int>();
            boardSizeMax = args["boardsize-max"].as<int>();
            if (boardSizeMin > boardSizeMax)
                throw std::invalid_argument("invalid board size range");
        }
        else {
            boardSizeMin = boardSizeMax = args["boardsize"].as<int>();
        }
        if (boardSizeMin < 5 || boardSizeMax > 20)
            throw std::invalid_argument("board size must in range [5,22]");

        if (args.count("opening")) {
            std::string filename = args["opening"].as<std::string>();
            if (boardSizeMin != boardSizeMax)
                throw std::invalid_argument(
                    "opening file can only be used with constant board size");

            // Open output file and change output stream
            std::ifstream openingFile(filename);
            if (!openingFile.is_open())
                throw std::invalid_argument("unable to open opening file " + filename);

            openings        = readOpenings(openingFile, boardSizeMin);
            generateOpening = false;
        }
        else {
            opengenCfg      = parseOpengenConfig(args);
            generateOpening = true;
        }

        numGames         = args["number"].as<size_t>();
        rule             = parseRule(args["rule"].as<std::string>());
        numThreads       = std::max<size_t>(args["thread"].as<size_t>(), 1);
        hashSizeMb       = args["hashsize"].as<size_t>();
        meanNodes        = args["mean-nodes"].as<double>();
        varNodes         = args["var-nodes"].as<double>();
        matePly          = args["mate-ply"].as<int>();
        drawValue        = args["draw-value"].as<int>();
        drawCount        = args["draw-count"].as<int>();
        minDrawPly       = args["min-draw-ply"].as<int>();
        forceDrawPly     = args["force-draw-ply"].as<int>();
        forceDrawPlyLeft = args["force-draw-plyleft"].as<int>();
        samplingFreqRoot = args["sampling-freq-root"].as<double>();
        samplingFreqPV   = args["sampling-freq-pv"].as<double>();
        sampleMateOnly   = args.count("sample-mate-only");
        reportInterval   = args["report-interval"].as<Time>();
        silence          = args.count("no-search-message");

        if (meanNodes <= 0)
            throw std::invalid_argument("mean-nodes must be greater than 0");
        if (matePly < 1)
            throw std::invalid_argument("mate-ply must be at least 1");
        if (matePly >= MAX_MOVES)
            throw std::invalid_argument("mate-ply must be less than " + std::to_string(MAX_MOVES));
        if (drawValue < 0)
            throw std::invalid_argument("draw-value must be at least 0");
        if (drawCount < 0)
            throw std::invalid_argument("draw-count must be at least 0");
        if (minDrawPly < 0)
            throw std::invalid_argument("min-draw-ply must be at least 0");
        if (forceDrawPly < 0)
            throw std::invalid_argument("force-draw-ply must be at least 0");
    }
    catch (const std::exception &e) {
        ERRORL("selfplay argument: " << e.what());
        std::exit(EXIT_FAILURE);
    }

    // Setup signal handler to close dataset file when receiving signal
    setupSignalHandler([&]() {
        if (datasetWriter)
            datasetWriter.reset();
    });

    if (openings.size())
        MESSAGEL("Readed " << openings.size() << " openings for selfplay.");
    else if (generateOpening)
        MESSAGEL("No opening file is specified, will use automatic opening generation.");
    else
        MESSAGEL("No openings for selfplay, will use empty board for opening.");

    // Set message mode to none if silence search is enabled
    if (silence)
        Config::MessageMode = MsgMode::NONE;
    Config::AspirationWindow              = true;
    Config::NumIterationAfterSingularRoot = 10;
    Config::NumIterationAfterMate         = 10;

    // Set num threads and TT size
    Search::Threads.setNumThreads(numThreads);
    Search::TT.resize(hashSizeMb * 1024);

    PRNG                               prng {};
    std::uniform_int_distribution<int> boardSizeDis(boardSizeMin, boardSizeMax);
    std::normal_distribution<double>   nodesDis(meanNodes, varNodes);
    size_t                             totalGamePly = 0;

    Time startTime = now(), lastTime = startTime;
    for (size_t i = 0; i < numGames;) {
        Board board(boardSizeDis(prng));
        board.newGame(rule);

        if (!silence)
            MESSAGEL("Start game " << i << ", boardsize = " << board.size());

        if (generateOpening) {
            Opening::OpeningGenerator og(board.size(), rule, opengenCfg, prng);

            // Generate a valid opening
            for (;;) {
                bool balanced = og.next();

                // If we want to generate a balanced opening, abandon those not balanced
                if (!balanced && opengenCfg.balanceWindow > 0
                    && (opengenCfg.balance1Nodes > 0 || opengenCfg.balance2Nodes > 0))
                    continue;
                else
                    break;
            }

            // Put opening
            for (int i = 0; i < og.getBoard().ply(); i++) {
                board.move(rule, og.getBoard().getHistoryMove(i));
            }

            if (!silence)
                MESSAGEL("Put generated opening " << og.positionString());
        }
        else if (!openings.empty()) {
            std::uniform_int_distribution<size_t> openingDis(0, openings.size() - 1);
            std::uniform_int_distribution<int>    transformDis(0, TRANS_NB - 1);
            size_t                                openingIdx = openingDis(prng);
            TransformType                         transform  = TransformType(transformDis(prng));

            // Apply opening pos
            for (Pos pos : openings[openingIdx]) {
                Pos transformedPos = applyTransform(pos, board.size(), transform);
                board.move(rule, transformedPos);
            }
        }

        // Set search options and init
        Search::SearchOptions options;
        options.rule                = {rule, GameRule::FREEOPEN};
        options.balanceMode         = Search::SearchOptions::BALANCE_NONE;
        options.disableOpeningQuery = true;

        // Clean up search states
        int numOpeningPly = board.ply();
        int firstMatePly  = -1;
        Search::Threads.clear(true);

        // Selfplay loop
        struct PVSample
        {
            int              rootGamePly;
            std::vector<Pos> pv;
        };
        std::vector<PVSample> pvSamples;
        Value                 searchValue = VALUE_ZERO;
        int                   drawCnt     = 0;
        while (board.movesLeft() > 0) {
            // Set search limits
            // Make sure max nodes stays in [0, 10*meanNodes]
            options.maxNodes =
                std::clamp<size_t>((size_t)nodesDis(prng), 0, size_t(10 * meanNodes));

            Search::Threads.startThinking(board, options);
            Search::Threads.waitForIdle();

            // We might have no legal move in Renju mode, which is regarded as loss
            if (Search::Threads.main()->rootMoves.empty()) {
                searchValue = mated_in(0);
                break;
            }

            // Add sampled PV moves
            if (samplingFreqPV >= 1.0
                || std::uniform_real_distribution<> {}(prng) <= samplingFreqPV) {
                auto bestRootMove = std::find(Search::Threads.main()->rootMoves.begin(),
                                              Search::Threads.main()->rootMoves.end(),
                                              Search::Threads.main()->previousPlyBestMove);
                // If previousPlyBestMove is Pos::NONE, we will not find a rootMove
                if (bestRootMove != Search::Threads.main()->rootMoves.end())
                    pvSamples.push_back({board.ply(), bestRootMove->previousPv});
            }

            searchValue = Search::Threads.main()->rootMoves[0].value;
            board.move(rule, Search::Threads.main()->bestMove);

            if (firstMatePly < 0 && std::abs(searchValue) >= VALUE_MATE_IN_MAX_PLY)
                firstMatePly = board.ply();

            // Stop self-play game if win/loss is found, or draw adjudication
            if (std::abs(searchValue) >= mate_in(matePly))
                break;

            if ((forceDrawPly && board.ply() >= forceDrawPly)
                || (forceDrawPlyLeft && board.movesLeft() <= forceDrawPlyLeft))
                break;

            if (drawCount && std::abs(searchValue) <= drawValue) {
                if (++drawCnt >= drawCount && board.ply() >= minDrawPly)
                    break;
            }
            else
                drawCnt = 0;
        }

        // Save game result (root samples) and pv samples
        if (datasetWriter) {
            GameEntry gameEntry;
            for (int ply = 0; ply < board.ply(); ply++)
                gameEntry.moves.push_back(board.getHistoryMove(ply));
            gameEntry.numOpeningMoves = numOpeningPly;
            gameEntry.boardsize       = board.size();
            gameEntry.rule            = rule;

            Result result    = searchValue >= VALUE_MATE_IN_MAX_PLY    ? RESULT_WIN
                               : searchValue <= VALUE_MATED_IN_MAX_PLY ? RESULT_LOSS
                                                                       : RESULT_DRAW;
            gameEntry.result = board.sideToMove() == WHITE ? result : Result(RESULT_WIN - result);

            datasetWriter->writeEntriesInGame(gameEntry, [&](const DataEntry &e) {
                // Exclude non-mate if sampleMateOnly is true
                if (sampleMateOnly && e.position.size() < firstMatePly)
                    return false;

                return samplingFreqRoot >= 1.0
                       || std::uniform_real_distribution<> {}(prng) <= samplingFreqRoot;
            });

            for (const PVSample &pvSample : pvSamples) {
                DataEntry dataEntry = {{}, (uint8_t)board.size(), rule};
                dataEntry.move = Pos {board.size(), board.size()};  // Mark the move as None move

                for (int ply = 0; ply < pvSample.rootGamePly; ply++)
                    dataEntry.position.push_back(board.getHistoryMove(ply));
                for (Pos move : pvSample.pv)
                    dataEntry.position.push_back(move);

                // Set result based on the current side to move
                dataEntry.result = dataEntry.position.size() % 2 == 0
                                       ? gameEntry.result
                                       : Result(RESULT_WIN - gameEntry.result);

                datasetWriter->writeEntry(dataEntry);
            }
        }

        // Print out generation progress over time
        i++;
        totalGamePly += board.ply();
        if (now() - lastTime >= reportInterval) {
            MESSAGEL("Played " << i << " of " << numGames
                               << " games, averagePly = " << totalGamePly / i
                               << ", game/min = " << i / ((now() - startTime) / 60000.0));
            lastTime = now();
        }
    }

    MESSAGEL("Completed playing " << numGames << " games.");
}
