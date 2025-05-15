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
    int                           multipv, multipvDecaySteps;
    double                        meanNodes, varNodes, nodesDecay;
    int                           maxDecaySteps;
    uint64_t                      minNodes;
    int                           matePly;
    int                           drawValue, drawCount, minDrawPly;
    int                           forceDrawPly, forceDrawPlyLeft;
    Time                          reportInterval;
    bool                          noMateMultiPV;
    bool                          generateOpening;
    bool                          silence;
    std::vector<std::vector<Pos>> openings;
    Opening::OpeningGenConfig     opengenCfg;
    DataWriterType                dataWriterType;
    std::string                   outputPath;
    std::unique_ptr<DataWriter>   dataWriter;

    cxxopts::Options options("rapfi selfplay");
    options.add_options()  //
        ("o,output",
         "Save data entries of played games to a binary file",
         cxxopts::value<std::string>())  //
        ("output-type",
         "Output dataset type, one of [txt, bin, bin_lz4, binpack, binpack_lz4]",
         cxxopts::value<std::string>()->default_value("binpack_lz4"))             //
        ("n,number", "Number of games to play", cxxopts::value<size_t>())         //
        ("boardsize-min", "Minimal board size in [5,22]", cxxopts::value<int>())  //
        ("boardsize-max", "Maximal board size in [5,22]", cxxopts::value<int>())  //
        ("opening",
         "Path to the opening book file. If not specified, auto generated openings are used",
         cxxopts::value<std::string>())  //
        ("multipv",
         "The maximum number of multipv to record (must be at least 1)",
         cxxopts::value<int>()->default_value("1"))  //
        ("multipv-decay-steps",
         "The number of steps to decrease multipv by 1 (0 for not enabled)",
         cxxopts::value<int>()->default_value("0"))  //
        ("no-multipv-after-mate",
         "Disable multipv after mate has been found")  //
        ("mean-nodes",
         "Mean of a normal distribution of num search nodes",
         cxxopts::value<double>()->default_value("100000"))  //
        ("var-nodes",
         "Variance of a normal distribution of num search nodes",
         cxxopts::value<double>()->default_value("10000"))  //
        ("nodes-decay",
         "The lambda to decay the number of search nodes",
         cxxopts::value<double>()->default_value("1.0"))  //
        ("max-nodes-decay-steps",
         "The maximum number of steps to decay the number of search nodes",
         cxxopts::value<int>()->default_value("100"))  //
        ("min-nodes",
         "Minimal number of nodes for search",
         cxxopts::value<uint64_t>()->default_value("20000"))  //
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
         "Force draw when left ply is less than this",
         cxxopts::value<int>()->default_value("0"))  //
        ("report-interval",
         "Time (ms) between two progress report message",
         cxxopts::value<Time>()->default_value("60000"))  //
        ("h,help", "Print selfplay usage");
    addPlayOptions(options);
    addOpengenOptions(options, opengenCfg);

    try {
        auto args = options.parse(argc, argv);

        if (args.count("help")) {
            std::cout << options.help() << std::endl;
            std::exit(EXIT_SUCCESS);
        }

        if (args.count("output")) {
            // Open output file and change output stream
            outputPath     = args["output"].as<std::string>();
            dataWriterType = parseDataWriterType(args["output-type"].as<std::string>());
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
        if (boardSizeMin < 5 || boardSizeMax > 22)
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

        numGames          = args["number"].as<size_t>();
        rule              = parseRule(args["rule"].as<std::string>());
        numThreads        = std::max<size_t>(args["thread"].as<size_t>(), 1);
        hashSizeMb        = std::max<size_t>(args["hashsize"].as<size_t>(), 1);
        multipv           = std::max(args["multipv"].as<int>(), 1);
        multipvDecaySteps = std::max(args["multipv-decay-steps"].as<int>(), 0);
        noMateMultiPV     = args.count("no-multipv-after-mate");
        meanNodes         = std::max(args["mean-nodes"].as<double>(), 0.0);
        varNodes          = std::max(args["var-nodes"].as<double>(), 0.0);
        nodesDecay        = std::min(args["nodes-decay"].as<double>(), 1.0);
        maxDecaySteps     = std::max(args["max-nodes-decay-steps"].as<int>(), 0);
        minNodes          = args["min-nodes"].as<uint64_t>();
        matePly           = std::max(args["mate-ply"].as<int>(), 1);
        drawValue         = args["draw-value"].as<int>();
        drawCount         = args["draw-count"].as<int>();
        minDrawPly        = args["min-draw-ply"].as<int>();
        forceDrawPly      = args["force-draw-ply"].as<int>();
        forceDrawPlyLeft  = args["force-draw-plyleft"].as<int>();
        reportInterval    = args["report-interval"].as<Time>();
        silence           = args.count("no-search-message");

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

    if (!outputPath.empty()) {
        // Create data writer
        switch (dataWriterType) {
        case DataWriterType::PlainText:
            dataWriter = std::make_unique<PlainTextDataWriter>(outputPath);
            break;

        case DataWriterType::SimpleBinary:
            dataWriter = std::make_unique<SimpleBinaryDataWriter>(outputPath, false);
            break;

        case DataWriterType::SimpleBinaryLZ4:
            dataWriter = std::make_unique<SimpleBinaryDataWriter>(outputPath, true);
            break;

        case DataWriterType::PackedBinary:
            dataWriter = std::make_unique<PackedBinaryDataWriter>(outputPath, false);
            break;

        case DataWriterType::PackedBinaryLZ4:
            dataWriter = std::make_unique<PackedBinaryDataWriter>(outputPath, true);
            break;

        case DataWriterType::Numpy:
            ERRORL("Numpy data writer is not supported in selfplay.");
            std::exit(EXIT_FAILURE);
            break;
        }
    }

    // Setup signal handler to close dataset file when receiving signal
    setupSignalHandler([&]() {
        MESSAGEL("Gracefully exiting...");
        dataWriter.reset();
        std::exit(0);
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
    else
        Config::MessageMode = MsgMode::BRIEF;
    Config::AspirationWindow = true;

    // Set num threads and TT size
    Search::Threads.setNumThreads(numThreads);
    Search::Threads.searcher()->setMemoryLimit(hashSizeMb * 1024);

    PRNG                               prng {};
    std::uniform_int_distribution<int> boardSizeDis(boardSizeMin, boardSizeMax);
    std::normal_distribution<double>   nodesDis(meanNodes, varNodes);
    size_t                             totalGamePly = 0;

    Time startTime = now(), lastTime = startTime;
    for (size_t i = 0; i < numGames;) {
        Board board(boardSizeDis(prng));
        board.newGame(rule);

        if (!silence)
            MESSAGEL("Start game " << i << ", boardsize = " << board.size() << ", rule = " << rule);

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
        options.multiPV             = multipv;
        options.balanceMode         = Search::SearchOptions::BALANCE_NONE;
        options.disableOpeningQuery = true;
        Search::Threads.clear(true);

        // Setup game entry data
        GameEntry gameEntry;
        for (int i = 0; i < board.ply(); i++)
            gameEntry.initPosition.push_back(board.getHistoryMove(i));
        gameEntry.boardsize = board.size();
        gameEntry.rule      = rule;

        // Selfplay loop
        Value searchValue = VALUE_ZERO;
        int   drawCnt     = 0;
        while (board.movesLeft() > 0) {
            // Stop self-play game if force draw or board is full
            if ((forceDrawPly && board.ply() >= forceDrawPly)
                || (forceDrawPlyLeft && board.movesLeft() <= forceDrawPlyLeft))
                break;

            // Set search limits, make sure max nodes stays in [0, +inf)
            int    steps      = board.ply() - (int)gameEntry.initPosition.size();
            double nodesScale = std::pow(nodesDecay, std::min(steps, maxDecaySteps));
            options.maxNodes =
                std::max<uint64_t>((uint64_t)std::max(nodesDis(prng) * nodesScale, 0.0), minNodes);
            if (multipvDecaySteps > 0 && steps > 0 && steps % multipvDecaySteps == 0)
                options.multiPV = std::max(1, options.multiPV - 1);

            // Start thinking and wait for finish
            Search::Threads.startThinking(board, options);
            Search::Threads.waitForIdle();
            auto mainThread = Search::Threads.main();

            // We might have no legal move in Renju mode, which is regarded as loss
            if (mainThread->rootMoves.empty()) {
                searchValue = mated_in(0);
                break;
            }

            // Record best move result
            searchValue  = mainThread->rootMoves[0].value;
            Pos bestMove = mainThread->bestMove;
            gameEntry.moveSequence.push_back({bestMove, Eval(searchValue)});
            if (options.multiPV > 1) {
                auto &moveData   = gameEntry.moveSequence.back();
                int   numPVMoves = std::min<int>(options.multiPV, mainThread->rootMoves.size());
                moveData.tag = DataEntry::MoveDataTag(DataEntry::MULTIPV_BEGIN + numPVMoves - 2);
                moveData.multiPvMoves = new PVMove[numPVMoves - 1];
                for (int i = 1; i < numPVMoves; i++) {
                    auto &rm = mainThread->rootMoves[i];
                    assert(rm.pv[0] != bestMove);
                    moveData.multiPvMoves[i - 1] = {
                        rm.pv[0],
                        Eval(rm.value != VALUE_NONE ? rm.value : rm.previousValue)};
                }
            }

            // Stop self-play game if win/loss is found
            if (std::abs(searchValue) >= mate_in(matePly))
                break;
            if (noMateMultiPV && std::abs(searchValue) >= VALUE_MATE_IN_MAX_PLY)
                options.multiPV = 1;

            // Stop self-play game if draw adjudication
            if (drawCount && std::abs(searchValue) <= drawValue) {
                if (++drawCnt >= drawCount && board.ply() >= minDrawPly)
                    break;
            }
            else
                drawCnt = 0;

            // Make the move
            board.move(rule, bestMove);
        }

        // Save game result (root samples)
        Result result    = searchValue >= VALUE_MATE_IN_MAX_PLY    ? RESULT_WIN
                           : searchValue <= VALUE_MATED_IN_MAX_PLY ? RESULT_LOSS
                                                                   : RESULT_DRAW;
        gameEntry.result = board.sideToMove() == WHITE ? result : Result(RESULT_WIN - result);
        dataWriter->writeGame(gameEntry);

        // Print out generation progress over time
        i++;
        totalGamePly += board.ply();
        if (now() - lastTime >= reportInterval) {
            MESSAGEL("Played " << i << " of " << numGames
                               << " games, average ply = " << totalGamePly / i
                               << ", game/min = " << i / ((now() - startTime) / 60000.0));
            lastTime = now();
        }
    }

    MESSAGEL("Completed playing " << numGames << " games.");
}
