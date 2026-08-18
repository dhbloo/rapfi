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
#include "../core/utils.h"
#include "../search/hashtable.h"
#include "../search/searchconfig.h"
#include "../search/searchthread.h"
#include "../tuning/datawriter.h"
#ifdef POLICY_TRAINING
    #include "../search/ab/searcher.h"
    #include "../search/movepick.h"
    #include "../tuning/policytrace.h"
#endif
#include "argutils.h"
#include "command.h"

#define CXXOPTS_NO_REGEX
#include <cmath>
#include <csignal>
#include <cxxopts.hpp>
#include <fstream>
#include <functional>
#include <iterator>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <stdexcept>

static std::function<void(int)> signalFunc;
static void                     setupSignalHandler(std::function<void(int)> handler)
{
    signalFunc         = std::move(handler);
    auto signalHandler = [](int signal) {
        if (signalFunc)
            signalFunc(signal);
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

static void resetSignalHandler()
{
    signalFunc = nullptr;
    std::signal(SIGINT, SIG_DFL);
    std::signal(SIGTERM, SIG_DFL);
    std::signal(SIGSEGV, SIG_DFL);
    std::signal(SIGILL, SIG_DFL);
    std::signal(SIGABRT, SIG_DFL);
    std::signal(SIGFPE, SIG_DFL);
#ifdef SIGHUP
    std::signal(SIGHUP, SIG_DFL);
#endif
#ifdef SIGQUIT
    std::signal(SIGQUIT, SIG_DFL);
#endif
}

class SignalHandlerScope
{
public:
    explicit SignalHandlerScope(std::function<void(int)> handler)
    {
        setupSignalHandler(std::move(handler));
    }
    ~SignalHandlerScope() { resetSignalHandler(); }

    SignalHandlerScope(const SignalHandlerScope &)            = delete;
    SignalHandlerScope &operator=(const SignalHandlerScope &) = delete;
};

using namespace Tuning;

namespace {

enum class BudgetUnit { NODES, TIME_MS };
enum class BudgetDistribution { FIXED, UNIFORM, NORMAL, LOGNORMAL, LOGUNIFORM };
constexpr uint64_t MAX_EXACT_BUDGET = (uint64_t(1) << 53) - 1;

// For self-play step s after the opening, the new CLI's effective budget is:
// clamp(round(sample * decayFactor ^ min(s, decayPlies)), min, max).
// The legacy CLI uses truncation to preserve its existing output byte-for-byte.
struct SearchBudgetConfig
{
    BudgetUnit         unit         = BudgetUnit::NODES;
    BudgetDistribution distribution = BudgetDistribution::NORMAL;
    uint64_t           value        = 0;
    double             mean         = 100000.0;
    double             stddev       = 10000.0;
    uint64_t           min          = 20000;
    uint64_t           max          = MAX_EXACT_BUDGET;
    double             decayFactor  = 1.0;
    int                decayPlies   = 100;
    bool               legacy       = true;
};

class SearchBudgetSampler
{
public:
    explicit SearchBudgetSampler(const SearchBudgetConfig &config)
        : config(config)
        , uniformDis(uniformMin(config), uniformMax(config))
        , normalDis(normalMean(config), normalStddev(config))
        , lognormalDis(lognormalMu(config), lognormalStddev(config))
        , loguniformDis(loguniformMin(config), loguniformMax(config))
    {}

    uint64_t sample(PRNG &prng, int step)
    {
        double rawBudget;
        switch (config.distribution) {
        case BudgetDistribution::FIXED: rawBudget = config.value; break;
        case BudgetDistribution::UNIFORM: rawBudget = uniformDis(prng); break;
        case BudgetDistribution::NORMAL: rawBudget = normalDis(prng); break;
        case BudgetDistribution::LOGNORMAL: rawBudget = lognormalDis(prng); break;
        case BudgetDistribution::LOGUNIFORM: rawBudget = std::exp(loguniformDis(prng)); break;
        default: throw std::logic_error("unknown search budget distribution");
        }

        double scale = std::pow(config.decayFactor, std::min(std::max(step, 0), config.decayPlies));
        // Preserve the old node-only CLI's truncation for reproducibility. The new
        // unit-neutral CLI rounds before enforcing its integral hard bounds.
        double scaledBudget   = rawBudget * scale;
        double integralBudget = config.legacy ? std::trunc(scaledBudget) : std::round(scaledBudget);
        if (std::isnan(integralBudget) || integralBudget <= static_cast<double>(config.min))
            return config.min;
        if (!std::isfinite(integralBudget) || integralBudget >= static_cast<double>(config.max))
            return config.max;
        return static_cast<uint64_t>(integralBudget);
    }

private:
    static double uniformMin(const SearchBudgetConfig &config)
    {
        return config.distribution == BudgetDistribution::UNIFORM ? static_cast<double>(config.min)
                                                                  : 0.0;
    }

    static double uniformMax(const SearchBudgetConfig &config)
    {
        return config.distribution == BudgetDistribution::UNIFORM ? static_cast<double>(config.max)
                                                                  : 1.0;
    }

    static double normalMean(const SearchBudgetConfig &config)
    {
        return config.distribution == BudgetDistribution::NORMAL ? config.mean : 0.0;
    }

    static double normalStddev(const SearchBudgetConfig &config)
    {
        return config.distribution == BudgetDistribution::NORMAL ? config.stddev : 1.0;
    }

    static double lognormalSigma(const SearchBudgetConfig &config)
    {
        double cv = config.stddev / config.mean;
        if (cv <= 1.0)
            return std::sqrt(std::log1p(cv * cv));

        double inverseCv = 1.0 / cv;
        return std::sqrt(2.0 * std::log(cv) + std::log1p(inverseCv * inverseCv));
    }

    static double lognormalMu(const SearchBudgetConfig &config)
    {
        double sigma = lognormalSigma(config);
        return std::log(config.mean) - sigma * sigma / 2.0;
    }

    static double lognormalStddev(const SearchBudgetConfig &config)
    {
        return config.distribution == BudgetDistribution::LOGNORMAL ? lognormalSigma(config) : 1.0;
    }

    static double loguniformMin(const SearchBudgetConfig &config)
    {
        return config.distribution == BudgetDistribution::LOGUNIFORM
                   ? std::log(static_cast<double>(config.min))
                   : 0.0;
    }

    static double loguniformMax(const SearchBudgetConfig &config)
    {
        return config.distribution == BudgetDistribution::LOGUNIFORM
                   ? std::log(static_cast<double>(config.max))
                   : 1.0;
    }

    SearchBudgetConfig                     config;
    std::uniform_real_distribution<double> uniformDis;
    std::normal_distribution<double>       normalDis;
    std::lognormal_distribution<double>    lognormalDis;
    std::uniform_real_distribution<double> loguniformDis;
};

uint64_t maximumBudget(BudgetUnit unit)
{
    if (unit == BudgetUnit::TIME_MS) {
        int64_t  communicationReserve = std::max(Search::TimeCfg.turnTimeReserved, 0);
        int64_t  moveHorizon          = Search::TimeCfg.moveHorizon;
        int64_t  maxMovesInProduct    = moveHorizon > 0 ? std::min<int64_t>(moveHorizon, MAX_MOVES)
                                                        : std::max<int64_t>(-moveHorizon, 1);
        uint64_t safeTimeBudget       = static_cast<uint64_t>(
            std::numeric_limits<int64_t>::max() / maxMovesInProduct - communicationReserve);
        return std::min(MAX_EXACT_BUDGET, safeTimeBudget);
    }
    return MAX_EXACT_BUDGET;
}

BudgetUnit parseBudgetUnit(const std::string &value)
{
    if (value == "nodes")
        return BudgetUnit::NODES;
    if (value == "time-ms")
        return BudgetUnit::TIME_MS;
    throw std::invalid_argument("budget-unit must be one of [nodes, time-ms]");
}

BudgetDistribution parseBudgetDistribution(const std::string &value)
{
    if (value == "fixed")
        return BudgetDistribution::FIXED;
    if (value == "uniform")
        return BudgetDistribution::UNIFORM;
    if (value == "normal")
        return BudgetDistribution::NORMAL;
    if (value == "lognormal")
        return BudgetDistribution::LOGNORMAL;
    if (value == "loguniform")
        return BudgetDistribution::LOGUNIFORM;
    throw std::invalid_argument(
        "budget-distribution must be one of [fixed, uniform, normal, lognormal, loguniform]");
}

void validateFinitePositive(double value, const char *name)
{
    if (!std::isfinite(value) || value <= 0.0)
        throw std::invalid_argument(std::string(name) + " must be finite and greater than 0");
}

uint64_t domainSeed(uint64_t seed, uint64_t domain)
{
    PRNG mixer(seed ^ domain);
    return mixer();
}

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

/// All knobs of one selfplay run, parsed from the command line.
struct SelfplayConfig
{
    size_t                        numGames;
    size_t                        numThreads;
    size_t                        hashSizeMb;
    int                           boardSizeMin, boardSizeMax;
    Rule                          rule;
    int                           multipv, multipvDecaySteps;
    SearchBudgetConfig            searchBudget;
    int                           matePly;
    int                           drawValue, drawCount, minDrawPly;
    int                           forceDrawPly, forceDrawPlyLeft;
    Time                          reportInterval;
    std::optional<uint64_t>       randomSeed;
    bool                          noMateMultiPV;
    bool                          generateOpening;
    bool                          silence;
    std::vector<std::vector<Pos>> openings;
    Opening::OpeningGenConfig     opengenCfg;
    Command::DataWriterType       dataWriterType;
    std::string                   outputPath;
#ifdef POLICY_TRAINING
    std::filesystem::path policyTracePath;
    std::filesystem::path policyModelPath;
    uint64_t              policySplitSalt          = 0;
    uint16_t              policyValidationPermille = 200;
#endif
};

void validateOpeningBook(const std::vector<std::vector<Pos>> &openings,
                         int                                  boardSize,
                         Rule                                 rule)
{
    Board board(boardSize);
    for (size_t openingIndex = 0; openingIndex < openings.size(); openingIndex++) {
        board.newGame(rule);
        for (size_t ply = 0; ply < openings[openingIndex].size(); ply++) {
            Pos move = openings[openingIndex][ply];
            if (!board.isInBoard(move) || !board.isEmpty(move)
                || (rule == RENJU && board.sideToMove() == BLACK
                    && board.checkForbiddenPoint(move)))
                throw std::invalid_argument("illegal opening move at opening "
                                            + std::to_string(openingIndex + 1) + ", ply "
                                            + std::to_string(ply + 1));
            board.move(rule, move);
        }
    }
}

/// Put an opening on the fresh `board`: either an auto-generated (balanced) opening,
/// or a random book opening under a random symmetry transform. No-op when neither
/// source is configured (game starts from the empty board).
void applyOpening(Board &board,
                  const SelfplayConfig &cfg,
                  PRNG &prng,
                  uint64_t *openingFamily)
{
#ifdef POLICY_TRAINING
    if (openingFamily)
        *openingFamily = policyOpeningFamily({});
#endif
    if (cfg.generateOpening) {
        Opening::OpeningGenerator og(board.size(), cfg.rule, cfg.opengenCfg, prng);

        // Generate a valid opening
        for (;;) {
            bool balanced = og.next();

            // If we want to generate a balanced opening, abandon those not balanced
            if (!balanced && cfg.opengenCfg.balanceWindow > 0
                && (cfg.opengenCfg.balance1Nodes > 0 || cfg.opengenCfg.balance2Nodes > 0))
                continue;
            else
                break;
        }

        // Put opening
        for (int i = 0; i < og.getBoard().ply(); i++) {
            board.move(cfg.rule, og.getBoard().getHistoryMove(i));
        }
#ifdef POLICY_TRAINING
        if (openingFamily) {
            std::vector<Pos> opening;
            opening.reserve(og.getBoard().ply());
            for (int i = 0; i < og.getBoard().ply(); i++)
                opening.push_back(og.getBoard().getHistoryMove(i));
            *openingFamily = policyOpeningFamily(opening);
        }
#endif

        if (!cfg.silence)
            MESSAGEL("Put generated opening " << og.positionString());
    }
    else if (!cfg.openings.empty()) {
        std::uniform_int_distribution<size_t> openingDis(0, cfg.openings.size() - 1);
        std::uniform_int_distribution<int>    transformDis(0, TRANS_NB - 1);
        size_t                                openingIdx = openingDis(prng);
        TransformType                         transform  = TransformType(transformDis(prng));
#ifdef POLICY_TRAINING
        if (openingFamily)
            *openingFamily = policyOpeningFamily(cfg.openings[openingIdx]);
#endif

        // Apply opening pos
        for (Pos pos : cfg.openings[openingIdx]) {
            Pos transformedPos = applyTransform(pos, board.size(), transform);
            board.move(cfg.rule, transformedPos);
        }
    }
}

/// Play one selfplay game on the prepared board (opening already applied) until the
/// win/loss/draw adjudication triggers, and return the recorded game entry.
GameEntry playOneGame(Board                &board,
                      const SelfplayConfig &cfg,
                      PRNG                 &budgetPrng,
                      SearchBudgetSampler  &budgetSampler
#ifdef POLICY_TRAINING
                      ,
                      PolicyTraceSession *policyTraceSession
#endif
)
{
    // Set search options and init
    Search::SearchOptions options;
    options.rule                = {cfg.rule, GameRule::FREEOPEN};
    options.multiPV             = cfg.multipv;
    options.balanceMode         = Search::SearchOptions::BALANCE_NONE;
    options.disableOpeningQuery = true;
    Search::Engine.clear(true);

    // Setup game entry data
    GameEntry gameEntry;
    for (int i = 0; i < board.ply(); i++)
        gameEntry.initPosition.push_back(board.getHistoryMove(i));
    gameEntry.boardsize = board.size();
    gameEntry.rule      = cfg.rule;

    // Selfplay loop
    Value searchValue = VALUE_ZERO;
    int   drawCnt     = 0;
#ifdef POLICY_TRAINING
    uint64_t rootSearchId = 0;
#endif
    while (board.movesLeft() > 0) {
        // Stop self-play game if force draw or board is full
        if ((cfg.forceDrawPly && board.ply() >= cfg.forceDrawPly)
            || (cfg.forceDrawPlyLeft && board.movesLeft() <= cfg.forceDrawPlyLeft))
            break;

        // Sample the base budget, decay it once per self-play ply after the opening,
        // and finally enforce the configured hard limits.
        int      step   = board.ply() - (int)gameEntry.initPosition.size();
        uint64_t budget = budgetSampler.sample(budgetPrng, step);
        if (cfg.searchBudget.unit == BudgetUnit::NODES) {
            options.setTimeControl(0, 0);
            options.maxNodes = budget;
        }
        else {
            options.maxNodes = 0;
            // A self-play time budget is search time, so compensate for the GUI
            // communication reserve that normal protocol time controls subtract.
            int64_t communicationReserve = std::max(Search::TimeCfg.turnTimeReserved, 0);
            options.setTimeControl(static_cast<int64_t>(budget) + communicationReserve, 0);
        }
        if (cfg.multipvDecaySteps > 0 && step > 0 && step % cfg.multipvDecaySteps == 0)
            options.multiPV = std::max(1, options.multiPV - 1);

        // Start thinking and wait for finish
#ifdef POLICY_TRAINING
        if (policyTraceSession)
            policyTraceSession->beginRootSearch(++rootSearchId);
#endif
        Search::Engine.startThinking(board, options);
        Search::Engine.waitForIdle();
#ifdef POLICY_TRAINING
        if (policyTraceSession)
            policyTraceSession->rethrowFailure();
#endif
        auto mainThread = Search::Engine.main();

        // We might have no legal move in Renju mode, which is regarded as loss
        if (mainThread->rootMoves.empty()) {
            searchValue = mated_in(0);
            break;
        }

        // Record best move result
        searchValue  = mainThread->rootMoves[0].value;
        Pos bestMove = Search::Engine.ctx.bestMove;
        gameEntry.moveSequence.push_back({bestMove, Eval(searchValue)});
        // A multipv search can still yield a single root move (a forced defence, or
        // renju forbidden points leaving one legal move); such a ply carries no
        // extra-PV payload.
        int numPVMoves = std::min<int>(options.multiPV, mainThread->rootMoves.size());
        if (options.multiPV > 1 && numPVMoves > 1) {
            auto &pvs = payloadAs<ExtraPVArray>(gameEntry.moveSequence.back().payload);
            pvs.reserve(numPVMoves - 1);
            for (int i = 1; i < numPVMoves; i++) {
                auto &rm = mainThread->rootMoves[i];
                assert(rm.pv[0] != bestMove);
                pvs.push_back(
                    {rm.pv[0], Eval(rm.value != VALUE_NONE ? rm.value : rm.previousValue)});
            }
        }

        // Stop self-play game if win/loss is found
        if (std::abs(searchValue) >= mate_in(cfg.matePly))
            break;
        if (cfg.noMateMultiPV && std::abs(searchValue) >= VALUE_MATE_IN_MAX_PLY)
            options.multiPV = 1;

        // Stop self-play game if draw adjudication
        if (cfg.drawCount && std::abs(searchValue) <= cfg.drawValue) {
            if (++drawCnt >= cfg.drawCount && board.ply() >= cfg.minDrawPly)
                break;
        }
        else
            drawCnt = 0;

        // Make the move
        board.move(cfg.rule, bestMove);
    }

    // Save game result (root samples)
    Result result = searchValue >= VALUE_MATE_IN_MAX_PLY    ? RESULT_WIN
                    : searchValue <= VALUE_MATED_IN_MAX_PLY ? RESULT_LOSS
                                                            : RESULT_DRAW;
    // `result` is from the final side to move's pov; GameEntry::result is white pov.
    gameEntry.result = board.sideToMove() == WHITE ? result : flipResult(result);
    return gameEntry;
}

/// Parse the selfplay command line into a SelfplayConfig (reading the opening book
/// file if one is given). Exits after printing usage on --help or argument errors.
SelfplayConfig parseSelfplayArguments(int argc, char *argv[])
{
    using namespace Command;

    SelfplayConfig cfg;

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
        ("budget-unit",
         "Search budget unit, one of [nodes, time-ms]",
         cxxopts::value<std::string>())  //
        ("budget-distribution",
         "Search budget distribution, one of [fixed, uniform, normal, lognormal, loguniform]",
         cxxopts::value<std::string>())  //
        ("budget-value",
         "Integral base budget for the fixed distribution",
         cxxopts::value<uint64_t>())  //
        ("budget-mean",
         "Arithmetic mean for the normal or lognormal distribution",
         cxxopts::value<double>())  //
        ("budget-stddev",
         "Arithmetic standard deviation for the normal or lognormal distribution",
         cxxopts::value<double>())  //
        ("budget-min",
         "Hard minimum budget after decay (required for non-fixed distributions)",
         cxxopts::value<uint64_t>())  //
        ("budget-max",
         "Hard maximum budget after decay (required for non-fixed distributions)",
         cxxopts::value<uint64_t>())  //
        ("budget-decay-factor",
         "Budget multiplier per self-play ply after the opening, in (0,1]",
         cxxopts::value<double>())  //
        ("budget-decay-plies",
         "Maximum number of per-ply budget decay multiplications",
         cxxopts::value<int>())  //
        ("mean-nodes",
         "Legacy: mean of the normal node-budget distribution",
         cxxopts::value<double>()->default_value("100000"))  //
        ("stddev-nodes",
         "Legacy: standard deviation of the normal node-budget distribution",
         cxxopts::value<double>()->default_value("10000"))  //
        ("var-nodes",
         "Deprecated alias for --stddev-nodes (this value was never a variance)",
         cxxopts::value<double>())  //
        ("nodes-decay",
         "Legacy: node-budget multiplier per self-play ply after the opening",
         cxxopts::value<double>()->default_value("1.0"))  //
        ("max-nodes-decay-steps",
         "Legacy: maximum number of per-ply node-budget decay multiplications",
         cxxopts::value<int>()->default_value("100"))  //
        ("min-nodes",
         "Legacy: hard minimum node budget after decay",
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
        ("seed",
         "Seed for reproducible self-play (omitted uses the current time)",
         cxxopts::value<uint64_t>())  //
        ("report-interval",
         "Time (ms) between two progress report message",
         cxxopts::value<Time>()->default_value("60000"))  //
        ("h,help", "Print selfplay usage");
#ifdef POLICY_TRAINING
    options.add_options()  //
        ("policy-trace", "Write a natural alpha-beta policy trace", cxxopts::value<std::string>())  //
        ("policy-model",
         "Explicit collection/teacher classical model for policy tracing",
         cxxopts::value<std::string>())  //
        ("policy-split-salt", "Nonzero opening split salt", cxxopts::value<uint64_t>())  //
        ("policy-validation-permille",
         "Validation opening families per 1000",
         cxxopts::value<uint16_t>()->default_value("200"));
#endif
    addPlayOptions(options);
    addOpengenOptions(options, cfg.opengenCfg);

    parseSubcommandArguments(
        options,
        argc,
        argv,
        "selfplay argument",
        [&](const cxxopts::ParseResult &args) {
            if (args.count("output")) {
                // Open output file and change output stream
                cfg.outputPath     = args["output"].as<std::string>();
                cfg.dataWriterType = parseDataWriterType(args["output-type"].as<std::string>());
            }

            if (args.count("boardsize-min") || args.count("boardsize-max")) {
                cfg.boardSizeMin = args["boardsize-min"].as<int>();
                cfg.boardSizeMax = args["boardsize-max"].as<int>();
                if (cfg.boardSizeMin > cfg.boardSizeMax)
                    throw std::invalid_argument("invalid board size range");
            }
            else {
                cfg.boardSizeMin = cfg.boardSizeMax = args["boardsize"].as<int>();
            }
            if (cfg.boardSizeMin < 5 || cfg.boardSizeMax > 22)
                throw std::invalid_argument("board size must in range [5,22]");

            if (args.count("opening")) {
                std::string filename = args["opening"].as<std::string>();
                if (cfg.boardSizeMin != cfg.boardSizeMax)
                    throw std::invalid_argument(
                        "opening file can only be used with constant board size");

                // Open output file and change output stream
                std::ifstream openingFile(filename);
                if (!openingFile.is_open())
                    throw std::invalid_argument("unable to open opening file " + filename);

                cfg.openings        = readOpenings(openingFile, cfg.boardSizeMin);
                cfg.generateOpening = false;
            }
            else {
                cfg.opengenCfg      = parseOpengenConfig(args);
                cfg.generateOpening = true;
            }

            cfg.numGames          = args["number"].as<size_t>();
            cfg.rule              = parseRule(args["rule"].as<std::string>());
            if (!cfg.openings.empty())
                validateOpeningBook(cfg.openings, cfg.boardSizeMin, cfg.rule);
            cfg.numThreads        = std::max<size_t>(args["thread"].as<size_t>(), 1);
            cfg.hashSizeMb        = std::max<size_t>(args["hashsize"].as<size_t>(), 1);
            cfg.multipv           = std::max(args["multipv"].as<int>(), 1);
            cfg.multipvDecaySteps = std::max(args["multipv-decay-steps"].as<int>(), 0);
            cfg.noMateMultiPV     = args.count("no-multipv-after-mate");

            const char *newBudgetOptions[] = {
                "budget-unit",
                "budget-distribution",
                "budget-value",
                "budget-mean",
                "budget-stddev",
                "budget-min",
                "budget-max",
                "budget-decay-factor",
                "budget-decay-plies",
            };
            const char *legacyBudgetOptions[] = {
                "mean-nodes",
                "stddev-nodes",
                "var-nodes",
                "nodes-decay",
                "max-nodes-decay-steps",
                "min-nodes",
            };
            auto hasAnyOption = [&](const auto &optionNames) {
                return std::any_of(std::begin(optionNames),
                                   std::end(optionNames),
                                   [&](const char *name) { return args.count(name) != 0; });
            };
            bool hasNewBudgetOption    = hasAnyOption(newBudgetOptions);
            bool hasLegacyBudgetOption = hasAnyOption(legacyBudgetOptions);

            if (hasNewBudgetOption) {
                if (hasLegacyBudgetOption)
                    throw std::invalid_argument(
                        "budget-* options cannot be combined with legacy node-budget options");
                if (!args.count("budget-unit"))
                    throw std::invalid_argument("budget-unit is required with budget-* options");
                if (!args.count("budget-distribution"))
                    throw std::invalid_argument(
                        "budget-distribution is required with budget-* options");

                auto &budget  = cfg.searchBudget;
                budget.legacy = false;
                budget.unit   = parseBudgetUnit(args["budget-unit"].as<std::string>());
                budget.distribution =
                    parseBudgetDistribution(args["budget-distribution"].as<std::string>());
                budget.min = args.count("budget-min") ? args["budget-min"].as<uint64_t>() : 1;
                budget.max = args.count("budget-max") ? args["budget-max"].as<uint64_t>()
                                                      : maximumBudget(budget.unit);
                budget.decayFactor = args.count("budget-decay-factor")
                                         ? args["budget-decay-factor"].as<double>()
                                         : 1.0;
                budget.decayPlies =
                    args.count("budget-decay-plies") ? args["budget-decay-plies"].as<int>() : 100;

                if (budget.min == 0)
                    throw std::invalid_argument("budget-min must be greater than 0");
                if (budget.max == 0)
                    throw std::invalid_argument("budget-max must be greater than 0");
                if (budget.max > maximumBudget(budget.unit))
                    throw std::invalid_argument(
                        "budget-max is too large for the selected budget-unit");
                if (budget.min > budget.max)
                    throw std::invalid_argument("budget-min must not be greater than budget-max");
                if (!std::isfinite(budget.decayFactor) || budget.decayFactor <= 0.0
                    || budget.decayFactor > 1.0)
                    throw std::invalid_argument("budget-decay-factor must be in (0,1]");
                if (budget.decayPlies < 0)
                    throw std::invalid_argument("budget-decay-plies must be at least 0");

                auto requireOption = [&](const char *name) {
                    if (!args.count(name))
                        throw std::invalid_argument(std::string(name)
                                                    + " is required for the selected distribution");
                };
                auto rejectOption = [&](const char *name) {
                    if (args.count(name))
                        throw std::invalid_argument(
                            std::string(name) + " is not valid for the selected distribution");
                };

                switch (budget.distribution) {
                case BudgetDistribution::FIXED:
                    requireOption("budget-value");
                    rejectOption("budget-mean");
                    rejectOption("budget-stddev");
                    budget.value = args["budget-value"].as<uint64_t>();
                    if (budget.value == 0)
                        throw std::invalid_argument("budget-value must be greater than 0");
                    if (budget.value < budget.min || budget.value > budget.max)
                        throw std::invalid_argument(
                            "budget-value must be within [budget-min, budget-max]");
                    break;
                case BudgetDistribution::UNIFORM:
                case BudgetDistribution::LOGUNIFORM:
                    requireOption("budget-min");
                    requireOption("budget-max");
                    rejectOption("budget-value");
                    rejectOption("budget-mean");
                    rejectOption("budget-stddev");
                    if (budget.min >= budget.max)
                        throw std::invalid_argument(
                            "budget-min must be less than budget-max for this distribution");
                    break;
                case BudgetDistribution::NORMAL:
                case BudgetDistribution::LOGNORMAL:
                    requireOption("budget-mean");
                    requireOption("budget-stddev");
                    requireOption("budget-min");
                    requireOption("budget-max");
                    rejectOption("budget-value");
                    budget.mean   = args["budget-mean"].as<double>();
                    budget.stddev = args["budget-stddev"].as<double>();
                    validateFinitePositive(budget.mean, "budget-mean");
                    validateFinitePositive(budget.stddev, "budget-stddev");
                    if (budget.min >= budget.max)
                        throw std::invalid_argument(
                            "budget-min must be less than budget-max for this distribution");
                    if (budget.mean < static_cast<double>(budget.min)
                        || budget.mean > static_cast<double>(budget.max))
                        throw std::invalid_argument(
                            "budget-mean must be within [budget-min, budget-max]");
                    break;
                default: throw std::logic_error("unknown search budget distribution");
                }
            }
            else {
                auto &budget = cfg.searchBudget;
                if (args.count("stddev-nodes") && args.count("var-nodes"))
                    throw std::invalid_argument(
                        "stddev-nodes and deprecated var-nodes cannot be specified together");
                if (args.count("var-nodes"))
                    MESSAGEL("Warning: --var-nodes is deprecated; use --stddev-nodes instead.");

                budget.mean = std::max(args["mean-nodes"].as<double>(), 0.0);
                budget.stddev =
                    std::max(args.count("var-nodes") ? args["var-nodes"].as<double>()
                                                     : args["stddev-nodes"].as<double>(),
                             0.0);
                budget.min         = args["min-nodes"].as<uint64_t>();
                budget.max         = std::numeric_limits<uint64_t>::max();
                budget.decayFactor = std::min(args["nodes-decay"].as<double>(), 1.0);
                budget.decayPlies  = std::max(args["max-nodes-decay-steps"].as<int>(), 0);

                if (!std::isfinite(budget.mean) || budget.mean <= 0.0)
                    throw std::invalid_argument("mean-nodes must be greater than 0");
                if (!std::isfinite(budget.stddev))
                    throw std::invalid_argument("stddev-nodes must be finite");
                if (!std::isfinite(budget.decayFactor))
                    throw std::invalid_argument("nodes-decay must be finite");
            }
            cfg.matePly          = std::max(args["mate-ply"].as<int>(), 1);
            cfg.drawValue        = args["draw-value"].as<int>();
            cfg.drawCount        = args["draw-count"].as<int>();
            cfg.minDrawPly       = args["min-draw-ply"].as<int>();
            cfg.forceDrawPly     = args["force-draw-ply"].as<int>();
            cfg.forceDrawPlyLeft = args["force-draw-plyleft"].as<int>();
            if (args.count("seed"))
                cfg.randomSeed = args["seed"].as<uint64_t>();
            cfg.reportInterval = args["report-interval"].as<Time>();
            cfg.silence        = args.count("no-search-message");

#ifdef POLICY_TRAINING
            if (args.count("policy-trace")) {
                if (!args.count("policy-model") || !args.count("policy-split-salt"))
                    throw std::invalid_argument(
                        "policy tracing requires --policy-model and --policy-split-salt");
                cfg.policyTracePath = args["policy-trace"].as<std::string>();
                cfg.policyModelPath = args["policy-model"].as<std::string>();
                cfg.policySplitSalt = args["policy-split-salt"].as<uint64_t>();
                cfg.policyValidationPermille =
                    args["policy-validation-permille"].as<uint16_t>();

                if (cfg.policyTracePath.empty() || cfg.policySplitSalt == 0
                    || cfg.policyValidationPermille == 0
                    || cfg.policyValidationPermille >= 1000)
                    throw std::invalid_argument("invalid policy trace settings");
                if (cfg.generateOpening)
                    throw std::invalid_argument(
                        "policy tracing requires an explicit immutable opening file");
                if ((cfg.boardSizeMin != 15 && cfg.boardSizeMin != 20)
                    || cfg.boardSizeMin != cfg.boardSizeMax)
                    throw std::invalid_argument(
                        "policy tracing currently supports f15 or f20 per run");
                if (cfg.numThreads != 1 || cfg.multipv != 1)
                    throw std::invalid_argument(
                        "policy tracing currently requires one search thread and MultiPV 1");
                if (!cfg.randomSeed)
                    throw std::invalid_argument("policy tracing requires an explicit seed");
                if (cfg.searchBudget.unit != BudgetUnit::NODES
                    || cfg.searchBudget.distribution != BudgetDistribution::FIXED
                    || cfg.searchBudget.decayFactor != 1.0)
                    throw std::invalid_argument(
                        "policy tracing currently requires a fixed non-decaying node budget");
            }
#endif

            if (cfg.matePly < 1)
                throw std::invalid_argument("mate-ply must be at least 1");
            if (cfg.matePly >= MAX_MOVES)
                throw std::invalid_argument("mate-ply must be less than "
                                            + std::to_string(MAX_MOVES));
            if (cfg.drawValue < 0)
                throw std::invalid_argument("draw-value must be at least 0");
            if (cfg.drawCount < 0)
                throw std::invalid_argument("draw-count must be at least 0");
            if (cfg.minDrawPly < 0)
                throw std::invalid_argument("min-draw-ply must be at least 0");
            if (cfg.forceDrawPly < 0)
                throw std::invalid_argument("force-draw-ply must be at least 0");
        });

    return cfg;
}

}  // namespace

void Command::selfplay(int argc, char *argv[]) try
{
    SelfplayConfig              cfg = parseSelfplayArguments(argc, argv);
    std::unique_ptr<DataWriter> dataWriter;
#ifdef POLICY_TRAINING
    std::unique_ptr<PolicyTraceSession> policyTraceSession;
#endif

    if (!cfg.outputPath.empty()) {
        // Create data writer
        if (cfg.dataWriterType == DataWriterType::Numpy) {
            ERRORL("Numpy data writer is not supported in selfplay.");
            std::exit(EXIT_FAILURE);
        }
        dataWriter = Tuning::makeDataWriter(cfg.dataWriterType, cfg.outputPath);
    }

#ifdef POLICY_TRAINING
    if (!cfg.policyTracePath.empty()) {
        if (!dynamic_cast<Search::AB::ABSearcher *>(Search::Engine.searcher()))
            throw std::runtime_error("policy tracing requires the alpha-beta searcher");
        if (Search::Engine.hasEvaluatorMaker())
            throw std::runtime_error("policy tracing requires classical fallback without evaluator");
        cfg.policyTracePath =
            std::filesystem::absolute(pathFromConsoleString(cfg.policyTracePath.string()));
        cfg.policyModelPath = getModelFullPath(cfg.policyModelPath);
        if (!loadModelFromFile(cfg.policyModelPath))
            throw std::runtime_error("unable to load policy trace model");

        if (!cfg.policyTracePath.parent_path().empty())
            ensureDir(cfg.policyTracePath.parent_path().string());
        policyTraceSession = std::make_unique<PolicyTraceSession>(cfg.policyTracePath);
    }
#endif

    // Setup signal handler to close dataset file when receiving signal. Termination
    // requests exit cleanly; fault signals (SIGSEGV etc.) still flush the writer but
    // must report the crash and exit nonzero instead of masquerading as success.
    SignalHandlerScope signalHandler([&](int signal) {
        bool requested = signal == SIGINT || signal == SIGTERM;
        if (requested)
            MESSAGEL("Gracefully exiting...");
        else
            ERRORL("Terminated by signal " << signal << ", flushing dataset...");
        dataWriter.reset();
        // Do not destroy the trace session here: a search worker may still reference it.
        // Process exit cannot publish its staged corpus, so the partial trace remains invalid.
        std::exit(requested ? 0 : EXIT_FAILURE);
    });

    if (cfg.openings.size())
        MESSAGEL("Read " << cfg.openings.size() << " openings for selfplay.");
    else if (cfg.generateOpening)
        MESSAGEL("No opening file is specified, will use automatic opening generation.");
    else
        MESSAGEL("No openings for selfplay, will use empty board for opening.");

    // Set message mode to none if silence search is enabled
    if (cfg.silence)
        Config::GeneralCfg.messageMode = MsgMode::NONE;
    else
        Config::GeneralCfg.messageMode = MsgMode::BRIEF;
    Search::SearchCfg.aspirationWindow = true;

    // Set num threads and TT size
    Search::Engine.setNumThreads(cfg.numThreads);
    Search::Engine.searcher()->setMemoryLimit(cfg.hashSizeMb * 1024);
#ifdef POLICY_TRAINING
    if (policyTraceSession)
        Search::Engine.setPolicyTraceSession(policyTraceSession.get());
#endif

    PRNG  prng       = cfg.randomSeed ? PRNG(*cfg.randomSeed) : PRNG::nondeterministic();
    PRNG  budgetPrng = cfg.randomSeed ? PRNG(domainSeed(*cfg.randomSeed, 0x7365617263682d62ULL))
                                      : PRNG::nondeterministic();
    PRNG &activeBudgetPrng = cfg.searchBudget.legacy ? prng : budgetPrng;
    SearchBudgetSampler                budgetSampler(cfg.searchBudget);
    std::uniform_int_distribution<int> boardSizeDis(cfg.boardSizeMin, cfg.boardSizeMax);
    size_t                             totalGamePly = 0;

    Time startTime = now(), lastTime = startTime;
    for (size_t i = 0; i < cfg.numGames;) {
        Board board(boardSizeDis(prng));
        board.newGame(cfg.rule);

        if (!cfg.silence)
            MESSAGEL("Start game " << i << ", boardsize = " << board.size()
                                   << ", rule = " << cfg.rule);

#ifdef POLICY_TRAINING
        uint64_t openingFamily = 0;
        applyOpening(board, cfg, prng, policyTraceSession ? &openingFamily : nullptr);
        if (policyTraceSession)
            policyTraceSession->beginGame(
                i + 1,
                openingFamily,
                assignPolicyTraceSplit(
                    openingFamily, cfg.policySplitSalt, cfg.policyValidationPermille));
#else
        applyOpening(board, cfg, prng, nullptr);
#endif

        GameEntry gameEntry =
            playOneGame(board,
                        cfg,
                        activeBudgetPrng,
                        budgetSampler
#ifdef POLICY_TRAINING
                        ,
                        policyTraceSession.get()
#endif
            );
        if (dataWriter)
            dataWriter->writeGame(gameEntry);

        // Print out generation progress over time
        i++;
        totalGamePly += board.ply();
        if (now() - lastTime >= cfg.reportInterval) {
            MESSAGEL("Played " << i << " of " << cfg.numGames
                               << " games, average ply = " << totalGamePly / i
                               << ", game/min = " << i / ((now() - startTime) / 60000.0));
            lastTime = now();
        }
    }

#ifdef POLICY_TRAINING
    if (policyTraceSession) {
        Search::Engine.setPolicyTraceSession(nullptr);
        Search::Engine.setNumThreads(0);
        const PolicyTraceSummary written = policyTraceSession->finalize();
        if (written.events == 0)
            throw std::runtime_error("policy trace contains no events");
        policyTraceSession->publish();
        MESSAGEL("Policy trace PASS: " << written.events << " events, "
                                       << written.candidates << " candidates.");
    }
#endif

    // Release rule-specific worker state while the command's signal callback and
    // writer are still alive. This avoids deferring evaluator/thread destruction to
    // static shutdown, after the callback's captured locals no longer exist.
    Search::Engine.setNumThreads(0);
    dataWriter.reset();

    MESSAGEL("Completed playing " << cfg.numGames << " games.");
}
catch (const std::exception &e) {
#ifdef POLICY_TRAINING
    Search::Engine.setPolicyTraceSession(nullptr);
#endif
    ERRORL("selfplay failed: " << e.what());
    std::exit(EXIT_FAILURE);
}
