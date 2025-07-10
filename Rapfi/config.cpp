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

#include "config.h"

#include "command/command.h"
#include "core/iohelper.h"
#include "database/dbstorage.h"
#include "database/yxdbstorage.h"
#include "eval/evaluator.h"
#include "eval/mix10nnue.h"
#include "eval/mix9svqnnue.h"
#include "game/pattern.h"
#include "search/ab/searcher.h"
#include "search/hashtable.h"
#include "search/mcts/searcher.h"
#include "search/searchthread.h"

#ifdef USE_ORT_EVALUATOR
    #include "eval/onnxevaluator.h"
#endif

#include <cpptoml.h>
#include <fstream>
#include <functional>
#include <limits>
#ifdef MULTI_THREADING
    #include <thread>
#endif

namespace {

static constexpr uint32_t N            = PATTERN_NB;
static constexpr uint32_t TABLE1_COUNT = combineNumber(N, 1);
static constexpr uint32_t TABLE2_COUNT = combineNumber(N, 2);

constexpr uint32_t I(uint32_t y, uint32_t x)
{
    // x >= y
    assert(x >= y);
    return y * N + x - combineNumber(y, 2);
}

}  // namespace

namespace Config {

// -------------------------------------------------
// Model configs

/// Scaling Factor is used for conversion between eval value and win rate.
/// Formula: win rate = sigmoid(eval / ScalingFactor)
float ScalingFactor                  = 200.0f;
float EvaluatorMarginWinLossScale    = 1.18f;
float EvaluatorMarginWinLossExponent = 3.07f;
float EvaluatorMarginScale           = 395.0f;
float EvaluatorDrawBlackWinRate      = 0.5f;
float EvaluatorDrawRatio             = 1.0f;

// Classical evaluation and score tables
// Note that Renju has asymmetry eval and score

Eval          EVALS[RULE_NB + 1][PCODE_NB];
Eval          EVALS_THREAT[RULE_NB + 1][THREAT_NB];
Pattern4Score P4SCORES[RULE_NB + 1][PCODE_NB];

// -------------------------------------------------
// General options

/// Should we reload config file before searching each move.
bool ReloadConfigEachMove = false;
/// Should we clear hash after each time config file is loaded.
bool ClearHashAfterConfigLoaded = true;
/// Default number of therads if not specified (0 means max hardware concurrency).
size_t DefaultThreadNum = 1;
/// Message output mode.
MsgMode MessageMode = MsgMode::BRIEF;
/// Coordinate convertion mode.
CoordConvertionMode IOCoordMode = CoordConvertionMode::NONE;
/// Default candidate range mode if not specified when creating board.
CandidateRange DefaultCandidateRange = CandidateRange::SQUARE3_LINE4;
/// Memory reserved for stuff other than hash table in max_memory option.
size_t MemoryReservedMB[RULE_NB] = {0};
/// Default hash table size (zero for not setting).
size_t DefaultTTSizeKB = 0;

// -------------------------------------------------
// Search options

const char *DefaultSearcherName = "alphabeta";

/// Whether to enable aspiration window.
bool AspirationWindow = true;
/// Whether to filter redundant symmetry moves at root.
bool FilterSymmetryRootMoves = true;
/// Number of iterations after we found a mate.
int NumIterationAfterMate = 6;
/// Number of iterations after we found a singular root.
int NumIterationAfterSingularRoot = 4;
/// Max depth to search.
int MaxSearchDepth = 99;
/// Expand node (evaluating policy) when first evaluate a node (evaluating value).
bool ExpandWhenFirstEvaluate = false;
/// The maximum number of visits per playout in MCTS search.
int MaxNumVisitsPerPlayout = 100;
/// How many nodes to print root moves in MCTS search. (Positive number to enable)
int NodesToPrintMCTSRootmoves = 0;
/// How much milliseconds to print root moves in MCTS search. (Positive number to enable)
int TimeToPrintMCTSRootmoves = 1000;
/// Maximum number of non-pv root moves to print in MCTS search.
int MaxNonPVRootmovesToPrint = 10;
/// Maximum number of search nodes after we found that we are in singular root.
int NumNodesAfterSingularRoot = 100;
/// The power of two number of shards that the node table has.
int NumNodeTableShardsPowerOfTwo = 10;
/// The ratio to decrase utility when child draw rate is high.
float DrawUtilityPenalty = 0.35f;

// Time management options

/// Time reserved for delay in communication between engine and GUI.
int TurnTimeReserved = 30;
/// Number of moves spared for the rest of game
float MatchSpace = 22.0f;
/// Minimum number of moves spared for the rest of game
float MatchSpaceMin = 7.0f;
/// Average branch factor to whether next depth has enough time
float AverageBranchFactor = 1.7f;
/// Exit search if turn time is used more than this ratio (even given ample match time)
float AdvancedStopRatio = 0.9f;
/// Plan time management at most this many moves ahead
int MoveHorizon = 64;
/// Bias of time divisor factor to depth
float TimeDivisorBias = 1.25f;
/// Scale of time divisor factor to depth
float TimeDivisorScale = 0.02f;
/// Pow to depth in time divisor factor
float TimeDivisorDepthPow = 1.4f;
/// Scale of score to falling factor
float FallingFactorScale = 0.0032f;
/// Offset of score to falling factor
float FallingFactorBias = 0.544f;
/// Scale of best move stable ply to reduction factor
float BestmoveStableReductionScale = 0.0125f;
/// Power of previous time reduction factor to get current factor
float BestmoveStablePrevReductionPow = 0.528f;

// -------------------------------------------------
// Database options

/// Whether to enable database by default
bool DatabaseDefaultEnabled;
/// Legacy code page to use for early database files and imported library files.
uint16_t DatabaseLegacyFileCodePage;
/// The type of database storage
std::string DatabaseType;
/// The URL of database storage (in utf-8 encoding)
std::string DatabaseURL;
/// Database storage factory, which takes the url (in utf-8 encoding)
/// and returns a unique pointer to an instance of DBStorage.
std::function<std::unique_ptr<::Database::DBStorage>(std::string)> DatabaseMaker;
/// Database client cache sizes
size_t DatabaseCacheSize       = 4096;
size_t DatabaseRecordCacheSize = 32768;

// Library import options

/// Mapping of marks in library file
char DatabaseLibBlackWinMark  = 'a';
char DatabaseLibWhiteWinMark  = 'a';
char DatabaseLibBlackLoseMark = 'c';
char DatabaseLibWhiteLoseMark = 'c';
/// Ignore all comments in imported library file
bool DatabaseLibIgnoreComment = false;
/// Ignore all board texts in imported library file
bool DatabaseLibIgnoreBoardText = false;

// Database search options

/// Whether to write/update the database in search
bool DatabaseReadonlyMode = false;
/// Whether to always write parent node if any of the children are written
bool DatabaseMandatoryParentWrite = true;

/// Search before this ply is required to query the database
int DatabaseQueryPly = 3;
/// How many iteration needed to increase one database query ply
int DatabaseQueryPVIterPerPlyIncrement = 1;
/// How many iteration needed to increase one database query ply
int DatabaseQueryNonPVIterPerPlyIncrement = 2;

/// PV node before this ply is required to write the database
int DatabasePVWritePly = 1;
/// How many depth needed to add a new record in PV node
int DatabasePVWriteMinDepth = 25;
/// NonPV node before this ply is required to write the database
int DatabaseNonPVWritePly = 0;
/// How many depth needed to add a new record in NonPV node
int DatabaseNonPVWriteMinDepth = 25;
/// The range of value allowed to write in PV/NonPV node
int DatabaseWriteValueRange = 800;
/// Mate node before this ply is required to write the database
int DatabaseMateWritePly = 2;
/// How many depth needed to add a new record in exact Mate node
int DatabaseMateWriteMinDepthExact = 20;
/// How many depth needed to add a new record in non-exact Mate node
int DatabaseMateWriteMinDepthNonExact = 40;
/// For mate longer than this step, we will try to write the record
int DatabaseMateWriteMinStep = 10;

/// For record found less then this ply, it will try to overwrite it with exact record
int DatabaseExactOverwritePly = 100;
/// For record found less then this ply, it will try to overwrite it with non-exact record
int DatabaseNonExactOverwritePly = 0;
/// The overwrite rule to write the database
::Database::OverwriteRule DatabaseOverwriteRule = ::Database::OverwriteRule::BetterValueDepthBound;
/// The bias added to the exact bound when comparing
int DatabaseOverwriteExactBias = 3;
/// The bias added to the old depth bound when comparing
int DatabaseOverwriteDepthBoundBias = -1;
/// The bias added to the queried depth bound when comparing
int DatabaseQueryResultDepthBoundBias = 0;

// -------------------------------------------------

void readRequirement(const cpptoml::table &t);
void readGeneral(const cpptoml::table &t);
void readSearch(const cpptoml::table &t);
void readModel(const cpptoml::table &t);
void readEvaluator(const cpptoml::table &t);
void readDatabase(const cpptoml::table &t);
template <typename ValueType,
          ValueType MinVal    = std::numeric_limits<ValueType>::lowest(),
          ValueType MaxVal    = std::numeric_limits<ValueType>::max(),
          typename SetterType = void(PatternCode, ValueType)>
void readValueModel(const cpptoml::table &t, SetterType setter);

}  // namespace Config

/// Load config from a stream.
/// @param configStream A input stream that contains a config file.
/// @param skipModelLoading Whether to skip model loading. Can be useful when
///     model is loaded separately from a binary file.
/// @return Returns true if loading succeeded, otherwise returns false.
bool Config::loadConfig(std::istream &configStream)
{
    Search::Threads.setupEvaluator(nullptr);
    Search::Threads.setupDatabase(nullptr);

    try {
        auto c = cpptoml::parser(configStream).parse();

        if (auto requirement = c->get_table("requirement"))
            readRequirement(*requirement);

        if (auto general = c->get_table("general"))
            readGeneral(*general);

        if (auto search = c->get_table("search"))
            readSearch(*search);

        if (auto database = c->get_table("database"))
            readDatabase(*database);

        if (auto model = c->get_table("model"))
            readModel(*model);
    }
    catch (const std::exception &e) {
        ERRORL("Failed to load config: " << e.what());
        return false;
    }

    return true;
}

/// Read requirement table of the config.
/// This is used to check if the config file is suitable for current version of Rapfi.
void Config::readRequirement(const cpptoml::table &t)
{
    auto [major, minor, revision] = getVersionNumbers();
    uint64_t rapVer = ((uint64_t)major << 32) | ((uint64_t)minor << 16) | (uint64_t)revision;
    if (auto minVer = t.get_array_of<int64_t>("min_version")) {
        if (minVer->size() != 3)
            throw std::runtime_error("illegal min_version");
        uint64_t cfgVer = ((*minVer)[0] << 32) | ((*minVer)[1] << 16) | (*minVer)[2];
        if (cfgVer > rapVer)
            throw std::runtime_error("config requires newer version of rapfi");
    }
    if (auto maxVer = t.get_array_of<int64_t>("max_version")) {
        if (maxVer->size() != 3)
            throw std::runtime_error("illegal max_version");
        uint64_t cfgVer = ((*maxVer)[0] << 32) | ((*maxVer)[1] << 16) | (*maxVer)[2];
        if (cfgVer < rapVer)
            throw std::runtime_error("config requires older version of rapfi");
    }
}

/// Read general table of the config.
void Config::readGeneral(const cpptoml::table &t)
{
    ReloadConfigEachMove = t.get_as<bool>("reload_config_each_move").value_or(ReloadConfigEachMove);
    ClearHashAfterConfigLoaded =
        t.get_as<bool>("clear_hash_after_config_loaded").value_or(ClearHashAfterConfigLoaded);

    // Read Default Thread Num
    DefaultThreadNum = t.get_as<uint64_t>("default_thread_num").value_or(DefaultThreadNum);
    if (DefaultThreadNum == 0) {
#ifdef MULTI_THREADING
        DefaultThreadNum = std::thread::hardware_concurrency();
        MESSAGEL("Setting default thread num to " << DefaultThreadNum << ".");
#else
        DefaultThreadNum = 1;
#endif
    }

    // Read Message Mode
    if (t.get_as<std::string>("message_mode")) {
        std::string msgModeStr = *t.get_as<std::string>("message_mode");
        if (msgModeStr == "normal")
            MessageMode = MsgMode::NORMAL;
        else if (msgModeStr == "brief")
            MessageMode = MsgMode::BRIEF;
        else if (msgModeStr == "ucilike")
            MessageMode = MsgMode::UCILIKE;
        else {
            if (msgModeStr != "none")
                MESSAGEL("Warning: unknown message mode [" << msgModeStr << "], reset to [none].");
            MessageMode = MsgMode::NONE;
        }
    }

    // Read Coord Conversion Mode
    if (t.get_as<std::string>("coord_conversion_mode")) {
        std::string coordModeStr = *t.get_as<std::string>("coord_conversion_mode");
        if (coordModeStr == "X_flipY")
            IOCoordMode = CoordConvertionMode::X_FLIPY;
        else if (coordModeStr == "flipY_X")
            IOCoordMode = CoordConvertionMode::FLIPY_X;
        else {
            if (coordModeStr != "none")
                MESSAGEL("Warning: unknown coordinate conversion mode [" << coordModeStr
                                                                         << "], reset to [none].");
            IOCoordMode = CoordConvertionMode::NONE;
        }
    }

    // Read Default Condidate Range Mode
    if (t.get_as<std::string>("default_candidate_range")) {
        std::string candRangeModeStr = *t.get_as<std::string>("default_candidate_range");
        if (candRangeModeStr == "square2")
            DefaultCandidateRange = CandidateRange::SQUARE2;
        else if (candRangeModeStr == "square2_line3")
            DefaultCandidateRange = CandidateRange::SQUARE2_LINE3;
        else if (candRangeModeStr == "square3")
            DefaultCandidateRange = CandidateRange::SQUARE3;
        else if (candRangeModeStr == "square3_line4")
            DefaultCandidateRange = CandidateRange::SQUARE3_LINE4;
        else if (candRangeModeStr == "square4")
            DefaultCandidateRange = CandidateRange::SQUARE4;
        else if (candRangeModeStr == "full_board")
            DefaultCandidateRange = CandidateRange::FULL_BOARD;
        else {
            MESSAGEL("Warning: unknown candidate range [" << candRangeModeStr
                                                          << "], reset to [square2_line3].");
            DefaultCandidateRange = CandidateRange::SQUARE2_LINE3;
        }
    }

    // Read memory reserved for each rule
    if (auto table = t.get_array("memory_reserved_mb")) {
        if (auto array = table->get_array_of<int64_t>()) {
            for (int i = 0; i < RULE_NB; i++)
                MemoryReservedMB[i] = array->at(std::min<size_t>(i, array->size() - 1));
        }
    }
    else {
        auto v = t.get_as<uint64_t>("memory_reserved_mb");
        for (int i = 0; i < RULE_NB; i++)
            MemoryReservedMB[i] = v.value_or(MemoryReservedMB[i]);
    }

    DefaultTTSizeKB = t.get_as<uint64_t>("default_tt_size_kb").value_or(DefaultTTSizeKB);
    // Resize TT according to default TT size (overriding previous size)
    if (DefaultTTSizeKB > 0)
        Search::Threads.searcher()->setMemoryLimit(DefaultTTSizeKB);
}

/// Read search table of the config.
void Config::readSearch(const cpptoml::table &t)
{
    if (auto v = t.get_as<std::string>("default_searcher"); v)
        Search::Threads.setupSearcher(createSearcher(*v));

    // Parameters for alpha-beta search
    AspirationWindow = t.get_as<bool>("aspiration_window").value_or(AspirationWindow);
    FilterSymmetryRootMoves =
        t.get_as<bool>("filter_symmetry_root_moves").value_or(FilterSymmetryRootMoves);
    NumIterationAfterMate =
        t.get_as<int>("num_iteration_after_mate").value_or(NumIterationAfterMate);
    NumIterationAfterSingularRoot =
        t.get_as<int>("num_iteration_after_singular_root").value_or(NumIterationAfterSingularRoot);
    MaxSearchDepth = t.get_as<int>("max_search_depth").value_or(MaxSearchDepth);

    // Parameters for MCTS search
    ExpandWhenFirstEvaluate =
        t.get_as<bool>("expand_when_first_evaluate").value_or(ExpandWhenFirstEvaluate);
    MaxNumVisitsPerPlayout =
        t.get_as<int>("max_num_visits_per_playout").value_or(MaxNumVisitsPerPlayout);
    NodesToPrintMCTSRootmoves =
        t.get_as<int>("nodes_to_print_mcts_rootmoves").value_or(NodesToPrintMCTSRootmoves);
    TimeToPrintMCTSRootmoves =
        t.get_as<int>("time_to_print_mcts_rootmoves").value_or(TimeToPrintMCTSRootmoves);
    MaxNonPVRootmovesToPrint =
        t.get_as<int>("max_non_pv_rootmoves_to_print").value_or(MaxNonPVRootmovesToPrint);
    NumNodesAfterSingularRoot =
        t.get_as<int>("num_nodes_after_singular_root").value_or(NumNodesAfterSingularRoot);
    NumNodeTableShardsPowerOfTwo =
        t.get_as<int>("num_node_table_shards_power_of_two").value_or(NumNodeTableShardsPowerOfTwo);
    DrawUtilityPenalty = t.get_as<double>("draw_utility_penalty").value_or(DrawUtilityPenalty);

    // Read time management options
    if (auto tm = t.get_table("timectl")) {
        TurnTimeReserved = tm->get_as<int>("turn_time_reserved").value_or(TurnTimeReserved);
        MatchSpace       = tm->get_as<double>("match_space").value_or(MatchSpace);
        MatchSpaceMin    = tm->get_as<double>("match_space_min").value_or(MatchSpaceMin);
        AverageBranchFactor =
            tm->get_as<double>("average_branch_factor").value_or(AverageBranchFactor);
        AdvancedStopRatio = tm->get_as<double>("advanced_stop_ratio").value_or(AdvancedStopRatio);
        MoveHorizon       = tm->get_as<int>("move_horizon").value_or(MoveHorizon);

        TimeDivisorScale = tm->get_as<double>("time_divisor_scale").value_or(TimeDivisorScale);
        TimeDivisorBias  = tm->get_as<double>("time_divisor_bias").value_or(TimeDivisorBias);
        TimeDivisorDepthPow =
            tm->get_as<double>("time_divisor_depth_pow").value_or(TimeDivisorDepthPow);

        FallingFactorScale =
            tm->get_as<double>("falling_factor_scale").value_or(FallingFactorScale);
        FallingFactorBias = tm->get_as<double>("falling_factor_bias").value_or(FallingFactorBias);

        BestmoveStableReductionScale = tm->get_as<double>("bestmove_stable_reduction_scale")
                                           .value_or(BestmoveStableReductionScale);
        BestmoveStablePrevReductionPow = tm->get_as<double>("bestmove_stable_prev_reduction_pow")
                                             .value_or(BestmoveStablePrevReductionPow);
    }
}

/// Read model table of all rules in the config.
void Config::readModel(const cpptoml::table &t)
{
    const Rule  Rules[]    = {FREESTYLE, STANDARD, RENJU};
    const char *RuleName[] = {"freestyle", "standard", "renju"};

    std::string modelPath = t.get_as<std::string>("binary_file").value_or("");
    if (!modelPath.empty()) {
        if (!Command::loadModelFromFile(modelPath))
            throw std::runtime_error("failed to load classic model file");
    }
    else {
        // Read Eval & Score
        if (auto eval = t.get_table("eval")) {
            bool hasAsymmetryRenjuEval = false;
            for (Rule r : Rules) {
                auto setEvalBlack = [r](PatternCode pcode, Eval ev) {
                    EVALS[r + BLACK][pcode] = ev;
                };
                auto setEvalWhite = [r](PatternCode pcode, Eval ev) {
                    EVALS[r + WHITE][pcode] = ev;
                };
                auto ruleEval = eval->get_table(RuleName[r]);
                if (!ruleEval)  // fallback
                    ruleEval = eval;
                else if (r == RENJU) {
                    auto blackEval    = ruleEval->get_table("black");
                    auto whiteEval    = ruleEval->get_table("white");
                    auto fallbackEval = ruleEval->get_as<int64_t>("model_type") ? ruleEval : eval;
                    if (hasAsymmetryRenjuEval = blackEval || whiteEval; hasAsymmetryRenjuEval) {
                        readValueModel<Eval, -16384, 16383>(*(blackEval ? blackEval : fallbackEval),
                                                            setEvalBlack);
                        readValueModel<Eval, -16384, 16383>(*(whiteEval ? whiteEval : fallbackEval),
                                                            setEvalWhite);
                    }
                }

                if (!hasAsymmetryRenjuEval) {
                    if (r == RENJU) {
                        readValueModel<Eval, -16384, 16383>(*ruleEval, setEvalBlack);
                        readValueModel<Eval, -16384, 16383>(*ruleEval, setEvalWhite);
                    }
                    else {
                        readValueModel<Eval, -16384, 16383>(*ruleEval, setEvalBlack);
                    }
                }
            }
        }

        if (auto score = t.get_table("score")) {
            auto readScore = [](const cpptoml::table &t, int tableIdx) {
                auto selfTable = t.get_table("self");
                auto oppoTable = t.get_table("oppo");
                readValueModel<Score, -8192, 8191>(selfTable ? *selfTable : t,
                                                   [tableIdx](PatternCode pcode, Score score) {
                                                       P4SCORES[tableIdx][pcode][0] = score;
                                                   });
                readValueModel<Score, -8192, 8191>(oppoTable ? *oppoTable : t,
                                                   [tableIdx](PatternCode pcode, Score score) {
                                                       P4SCORES[tableIdx][pcode][1] = score;
                                                   });
            };
            bool hasAsymmetryRenjuScore = false;
            for (Rule r : Rules) {
                auto ruleScore = score->get_table(RuleName[r]);
                if (!ruleScore)  // fallback
                    ruleScore = score;
                else if (r == RENJU) {
                    auto blackScore = ruleScore->get_table("black");
                    auto whiteScore = ruleScore->get_table("white");
                    auto fallbackScore =
                        ruleScore->get_as<int64_t>("model_type") ? ruleScore : score;
                    if (hasAsymmetryRenjuScore = blackScore || whiteScore; hasAsymmetryRenjuScore) {
                        readScore(*(blackScore ? blackScore : fallbackScore), r + BLACK);
                        readScore(*(whiteScore ? whiteScore : fallbackScore), r + WHITE);
                    }
                }

                if (!hasAsymmetryRenjuScore) {
                    if (r == RENJU) {
                        readScore(*ruleScore, r + BLACK);
                        readScore(*ruleScore, r + WHITE);
                    }
                    else {
                        readScore(*ruleScore, r);
                    }
                }
            }
        }
    }

    // Read scalingFactor
    ScalingFactor = (float)t.get_as<double>("scaling_factor").value_or(ScalingFactor);

    // Read evaluator
    if (auto evaluator = t.get_table("evaluator"))
        readEvaluator(*evaluator);
}

/// Read evaluator table in the config.
void Config::readEvaluator(const cpptoml::table &t)
{
    using namespace std::filesystem;

    auto evaluatorType = t.get_as<std::string>("type");
    auto weights       = t.get_table_array("weights");
    if (!evaluatorType || !weights || weights->begin() == weights->end())
        return;

    auto warpEvaluatorMaker = [weights,
                               evaluatorName = *evaluatorType](auto maker,
                                                               bool seperateBlackAndWhiteWeights) {
        return [=](int              boardSize,
                   Rule             rule,
                   Numa::NumaNodeId numaId) -> std::unique_ptr<Evaluation::Evaluator> {
            try {
                for (auto weightCfg : *weights) {
                    path weightPath;
                    path blackWeightPath, whiteWeightPath;

                    if (auto weightFile = weightCfg->get_as<std::string>("weight_file"))
                        weightPath = Command::getModelFullPath(u8path(*weightFile));
                    else if (!seperateBlackAndWhiteWeights)
                        throw std::runtime_error("must specify weight_file in weight configs.");

                    if (seperateBlackAndWhiteWeights && weightPath.empty()) {
                        auto weightFileBlack = weightCfg->get_as<std::string>("weight_file_black");
                        auto weightFileWhite = weightCfg->get_as<std::string>("weight_file_white");

                        blackWeightPath = Command::getModelFullPath(u8path(*weightFileBlack));
                        whiteWeightPath = Command::getModelFullPath(u8path(*weightFileWhite));
                        if (!weightFileBlack || !weightFileWhite)
                            throw std::runtime_error(
                                "must specify weight_file or weight_file_black "
                                "and weight_file_white in weight configs.");
                    }
                    else {
                        blackWeightPath = weightPath;
                        whiteWeightPath = weightPath;
                    }

                    try {
                        return maker(boardSize,
                                     rule,
                                     numaId,
                                     weightPath,
                                     std::make_pair(blackWeightPath, whiteWeightPath),
                                     *weightCfg);
                    }
                    catch (const Evaluation::UnsupportedRuleError &e) {
                    }
                    catch (const Evaluation::UnsupportedBoardSizeError &e) {
                    }
                    catch (const std::exception &e) {
                        if (MessageMode != MsgMode::NONE)
                            MESSAGEL("Failed to load from "
                                     << (!weightPath.empty()
                                             ? pathToConsoleString(weightPath)
                                             : pathToConsoleString(blackWeightPath) + " and "
                                                   + pathToConsoleString(whiteWeightPath))
                                     << " due to error: " << e.what());
                    }
                }

                if (MessageMode != MsgMode::NONE)
                    MESSAGEL("Evaluator " << evaluatorName
                                          << " disabled: no compatible weight config found.");
                return nullptr;
            }
            catch (const std::exception &e) {
                ERRORL("Evaluator " << evaluatorName << " failed to initialized: " << e.what());
                return nullptr;
            }
        };
    };

    if (*evaluatorType == "mix9svq") {
        Search::Threads.setupEvaluator(warpEvaluatorMaker(
            [=](int                   boardSize,
                Rule                  rule,
                Numa::NumaNodeId      numaId,
                path                  weightPath,
                std::pair<path, path> blackAndWhiteWeightPath,
                const cpptoml::table &weightCfg) {
                return std::make_unique<Evaluation::mix9svq::Evaluator>(
                    boardSize,
                    rule,
                    numaId,
                    blackAndWhiteWeightPath.first,
                    blackAndWhiteWeightPath.second);
            },
            true));
    }
    else if (*evaluatorType == "mix10") {
        Search::Threads.setupEvaluator(warpEvaluatorMaker(
            [=](int                   boardSize,
                Rule                  rule,
                Numa::NumaNodeId      numaId,
                path                  weightPath,
                std::pair<path, path> blackAndWhiteWeightPath,
                const cpptoml::table &weightCfg) {
                return std::make_unique<Evaluation::mix10::Evaluator>(
                    boardSize,
                    rule,
                    numaId,
                    blackAndWhiteWeightPath.first,
                    blackAndWhiteWeightPath.second);
            },
            true));
    }
#ifdef USE_ORT_EVALUATOR
    else if (*evaluatorType == "ort") {
        std::string deviceName = t.get_as<std::string>("ort_device").value_or("");

        Search::Threads.setupEvaluator(warpEvaluatorMaker(
            [=](int                   boardSize,
                Rule                  rule,
                Numa::NumaNodeId      numaId,
                path                  weightPath,
                std::pair<path, path> blackAndWhiteWeightPath,
                const cpptoml::table &weightCfg) {
                return std::make_unique<Evaluation::onnx::OnnxEvaluator>(boardSize,
                                                                         rule,
                                                                         weightPath,
                                                                         deviceName);
            },
            false));
    }
#endif
    else {
        throw std::runtime_error("unsupported evaluator type " + *evaluatorType);
    }

    // Read classical/evaluator switching margin
    EvaluatorMarginWinLossScale =
        (float)t.get_as<double>("margin_winloss_scale").value_or(EvaluatorMarginWinLossScale);
    EvaluatorMarginWinLossExponent =
        (float)t.get_as<double>("margin_winloss_exp").value_or(EvaluatorMarginWinLossExponent);
    EvaluatorMarginScale = (float)t.get_as<double>("margin_scale").value_or(EvaluatorMarginScale);
    EvaluatorDrawBlackWinRate =
        (float)t.get_as<double>("draw_black_winrate").value_or(EvaluatorDrawBlackWinRate);
    EvaluatorDrawRatio        = (float)t.get_as<double>("draw_ratio").value_or(EvaluatorDrawRatio);
    EvaluatorDrawBlackWinRate = std::clamp(EvaluatorDrawBlackWinRate, 0.0f, 1.0f);
    EvaluatorDrawRatio        = std::clamp(EvaluatorDrawRatio, 0.0f, 1.0f);

    MESSAGEL("Evaluator set to " << *evaluatorType << ".");
}

/// Read database table in the config.
void Config::readDatabase(const cpptoml::table &t)
{
    DatabaseDefaultEnabled = t.get_as<bool>("enable_by_default").value_or(DatabaseDefaultEnabled);
    DatabaseType           = t.get_as<std::string>("type").value_or(DatabaseType);
    DatabaseURL            = t.get_as<std::string>("url").value_or(DatabaseURL);
    DatabaseCacheSize      = t.get_as<size_t>("cache_size").value_or(DatabaseCacheSize);
    DatabaseRecordCacheSize =
        t.get_as<size_t>("record_cache_size").value_or(DatabaseRecordCacheSize);
    DatabaseLegacyFileCodePage =
        t.get_as<int>("legacy_file_code_page").value_or(DatabaseLegacyFileCodePage);
    DatabaseMaker = nullptr;

    if (DatabaseType == "yixindb") {
        if (DatabaseURL.empty())
            DatabaseURL = "rapfi.db";

        bool compressedSave   = true;
        bool saveOnClose      = true;
        int  numBackupsOnSave = 1;
        bool ignoreCorrupted  = false;
        if (auto args = t.get_table("yixindb")) {
            compressedSave   = args->get_as<bool>("compressed_save").value_or(compressedSave);
            saveOnClose      = args->get_as<bool>("save_on_close").value_or(saveOnClose);
            numBackupsOnSave = args->get_as<int>("num_backups_on_save").value_or(numBackupsOnSave);
            ignoreCorrupted  = args->get_as<bool>("ignore_corrupted").value_or(ignoreCorrupted);
        }

        DatabaseMaker = [=](std::string utf8URL) -> std::unique_ptr<::Database::DBStorage> {
            try {
                auto dbPath    = std::filesystem::u8path(utf8URL);
                bool existing  = std::filesystem::exists(dbPath);
                auto startTime = now();
                MESSAGEL("Opening yixin database at " << pathToConsoleString(dbPath) << " ...");
                auto dbStorage = std::make_unique<::Database::YXDBStorage>(dbPath,
                                                                           compressedSave,
                                                                           saveOnClose,
                                                                           numBackupsOnSave,
                                                                           ignoreCorrupted);
                if (existing)
                    MESSAGEL("Loaded Yixin database (" << dbStorage->size() << " records) using "
                                                       << (now() - startTime) << " ms.");
                return std::move(dbStorage);
            }
            catch (const std::exception &e) {
                ERRORL("Failed to create yixin database: " << e.what());
                return nullptr;
            }
        };
    }
    else if (!DatabaseType.empty()) {
        throw std::runtime_error("unsupported database type " + DatabaseType);
    }

    if (auto s = t.get_table("search")) {
        DatabaseReadonlyMode         = s->get_as<bool>("readonly_mode").value_or(false);
        DatabaseMandatoryParentWrite = s->get_as<bool>("mandatory_parent_write").value_or(true);
        DatabaseQueryPly             = s->get_as<int>("query_ply").value_or(DatabaseQueryPly);
        DatabaseQueryPVIterPerPlyIncrement = s->get_as<int>("pv_iter_per_ply_increment")
                                                 .value_or(DatabaseQueryPVIterPerPlyIncrement);
        DatabaseQueryNonPVIterPerPlyIncrement =
            s->get_as<int>("nonpv_iter_per_ply_increment")
                .value_or(DatabaseQueryNonPVIterPerPlyIncrement);

        DatabasePVWritePly = s->get_as<int>("pv_write_ply").value_or(DatabasePVWritePly);
        DatabasePVWriteMinDepth =
            s->get_as<int>("pv_write_min_depth").value_or(DatabasePVWriteMinDepth);

        DatabaseNonPVWritePly = s->get_as<int>("nonpv_write_ply").value_or(DatabaseNonPVWritePly);
        DatabaseNonPVWriteMinDepth =
            s->get_as<int>("nonpv_write_min_depth").value_or(DatabaseNonPVWriteMinDepth);

        DatabaseWriteValueRange =
            s->get_as<int>("write_value_range").value_or(DatabaseWriteValueRange);

        DatabaseMateWritePly = s->get_as<int>("mate_write_ply").value_or(DatabaseMateWritePly);
        DatabaseMateWriteMinDepthExact =
            s->get_as<int>("mate_write_min_depth_exact").value_or(DatabaseMateWriteMinDepthExact);
        DatabaseMateWriteMinDepthNonExact = s->get_as<int>("mate_write_min_depth_nonexact")
                                                .value_or(DatabaseMateWriteMinDepthNonExact);
        DatabaseMateWriteMinStep =
            s->get_as<int>("mate_write_min_step").value_or(DatabaseMateWriteMinStep);

        DatabaseExactOverwritePly =
            s->get_as<int>("exact_overwrite_ply").value_or(DatabaseExactOverwritePly);
        DatabaseNonExactOverwritePly =
            s->get_as<int>("nonexact_overwrite_ply").value_or(DatabaseNonExactOverwritePly);

        if (auto overwriteRule =
                s->get_as<std::string>("overwrite_rule").value_or("better_value_depth_bound");
            overwriteRule == "better_value_depth_bound")
            DatabaseOverwriteRule = ::Database::OverwriteRule::BetterValueDepthBound;
        else if (overwriteRule == "better_depth_bound")
            DatabaseOverwriteRule = ::Database::OverwriteRule::BetterDepthBound;
        else if (overwriteRule == "better_value")
            DatabaseOverwriteRule = ::Database::OverwriteRule::BetterValue;
        else if (overwriteRule == "better_label")
            DatabaseOverwriteRule = ::Database::OverwriteRule::BetterLabel;
        else if (overwriteRule == "always")
            DatabaseOverwriteRule = ::Database::OverwriteRule::Always;
        else if (overwriteRule == "disabled")
            DatabaseOverwriteRule = ::Database::OverwriteRule::Disabled;
        else
            MESSAGEL("unknown database overwrite rule " << overwriteRule << ", keep it unchanged.");

        DatabaseOverwriteExactBias =
            s->get_as<int>("overwrite_exact_bias").value_or(DatabaseOverwriteExactBias);
        DatabaseOverwriteDepthBoundBias =
            s->get_as<int>("overwrite_depth_bound_bias").value_or(DatabaseOverwriteDepthBoundBias);
        DatabaseQueryResultDepthBoundBias = s->get_as<int>("query_result_depth_bound_bias")
                                                .value_or(DatabaseQueryResultDepthBoundBias);
    }

    if (auto s = t.get_table("libfile")) {
        DatabaseLibBlackWinMark    = t.get_as<std::string>("black_win_mark").value_or("a")[0];
        DatabaseLibWhiteWinMark    = t.get_as<std::string>("white_win_mark").value_or("a")[0];
        DatabaseLibBlackLoseMark   = t.get_as<std::string>("black_lose_mark").value_or("c")[0];
        DatabaseLibWhiteLoseMark   = t.get_as<std::string>("white_lose_mark").value_or("c")[0];
        DatabaseLibIgnoreComment   = t.get_as<bool>("ignore_comment").value_or(false);
        DatabaseLibIgnoreBoardText = t.get_as<bool>("ignore_board_text").value_or(false);
    }

    if (DatabaseDefaultEnabled)
        Search::Threads.setupDatabase(createDefaultDBStorage());
}

/// Read a value model from a model table.
/// @tparam ValueType Type of read values.
/// @tparam MinVal Minimal of the read values. Values smaller than this will be clamped.
/// @tparam MaxVal Maximal of the read values. Values greater than this will be clamped.
/// @tparam SetterType Type of value setter, must be compatible with void(PatternCode, ValueType).
/// @param setter setter(pcode, value) is called when saving values.
template <typename ValueType, ValueType MinVal, ValueType MaxVal, typename SetterType>
void Config::readValueModel(const cpptoml::table &t, SetterType setter)
{
    enum ComposeFunc { SUM, MAX };
    auto getComposeFunc = [](const std::string &key) -> ComposeFunc {
        if (key == "sum")
            return SUM;
        else if (key == "max")
            return MAX;
        else
            throw std::runtime_error("unknown value compose function");
    };

    auto modelType = t.get_as<int64_t>("model_type");
    if (!modelType)
        throw std::runtime_error("model_type not specified");

    size_t overflowCount = 0;
    switch (*modelType) {
    case 0:
        if (auto raw = t.get_array_of<int64_t>("raw")) {
            if (raw->size() != PCODE_NB)
                throw std::runtime_error("number of values in raw model is not correct");
            for (size_t pcode = 0; pcode < PCODE_NB; pcode++) {
                int64_t val = (*raw)[pcode];
                overflowCount += val < MinVal || val > MaxVal;
                setter(pcode, ValueType(val));
            }
        }
        else
            throw std::runtime_error("raw values are missing or incorrect");
        break;
    case 1:
        if (auto table1 = t.get_array_of<int64_t>("table1")) {
            if (table1->size() != TABLE1_COUNT)
                throw std::runtime_error("size of values of table1 not correct");

            double      valScale = t.get_as<double>("table1_scale").value_or(1.0);
            int64_t     valMin   = t.get_as<int64_t>("table1_min").value_or(MinVal);
            int64_t     valMax   = t.get_as<int64_t>("table1_max").value_or(MaxVal);
            ComposeFunc compose =
                getComposeFunc(t.get_as<std::string>("table1_compose_func").value_or("sum"));

            for (int a = 0; a < N; a++)
                for (int b = a; b < N; b++)
                    for (int c = b; c < N; c++)
                        for (int d = c; d < N; d++) {
                            int64_t val = 0;
                            switch (compose) {
                            case SUM:
                                val = (*table1)[a] + (*table1)[b] + (*table1)[c] + (*table1)[d];
                                break;
                            case MAX:
                                val = std::max(
                                    {(*table1)[a], (*table1)[b], (*table1)[c], (*table1)[d]});
                                break;
                            }
                            val = std::clamp((int64_t)std::round(val * valScale), valMin, valMax);
                            overflowCount += val < MinVal || val > MaxVal;
                            setter(PatternConfig::PCODE[a][b][c][d], ValueType(val));
                        }
        }
        else
            throw std::runtime_error("table1 is missing or incorrect");
        break;
    case 2:
        if (auto table2 = t.get_array_of<int64_t>("table2")) {
            if (table2->size() != TABLE2_COUNT)
                throw std::runtime_error("size of values of table2 not correct");

            double      valScale = t.get_as<double>("table2_scale").value_or(1.0);
            int64_t     valMin   = t.get_as<int64_t>("table2_min").value_or(MinVal);
            int64_t     valMax   = t.get_as<int64_t>("table2_max").value_or(MaxVal);
            ComposeFunc compose =
                getComposeFunc(t.get_as<std::string>("table2_compose_func").value_or("sum"));

            for (int a = 0; a < N; a++)
                for (int b = a; b < N; b++)
                    for (int c = b; c < N; c++)
                        for (int d = c; d < N; d++) {
                            int64_t val = 0;
                            switch (compose) {
                            case SUM:
                                val = (*table2)[I(a, b)] + (*table2)[I(a, c)] + (*table2)[I(a, d)]
                                      + (*table2)[I(b, c)] + (*table2)[I(b, d)]
                                      + (*table2)[I(c, d)];
                                break;
                            case MAX:
                                val = std::max({(*table2)[I(a, b)],
                                                (*table2)[I(a, c)],
                                                (*table2)[I(a, d)],
                                                (*table2)[I(b, c)],
                                                (*table2)[I(b, d)],
                                                (*table2)[I(c, d)]});
                                break;
                            }
                            val = std::clamp((int64_t)std::round(val * valScale), valMin, valMax);
                            overflowCount += val < MinVal || val > MaxVal;
                            setter(PatternConfig::PCODE[a][b][c][d], ValueType(val));
                        }
        }
        else
            throw std::runtime_error("table2 is missing or incorrect");
        break;
    default: throw std::runtime_error("unknown valuation model");
    }

    if (overflowCount)
        MESSAGEL("Warning: " << overflowCount
                             << " values in (converted) raw model overflows. "
                                "Value should be in range ["
                             << MinVal << ", " << MaxVal
                             << "]. "
                                "This will cause incorrect evaluation and move sorting. "
                                "Please try to turn down large values or adding limits.");
}

bool Config::loadModel(std::istream &inStream)
{
    Compressor    compressor(inStream, Compressor::Type::LZ4_DEFAULT);
    std::istream *in = compressor.openInputStream();
    if (!in)
        return false;

    double scalingFactorF64;
    in->read(reinterpret_cast<char *>(&scalingFactorF64), sizeof(scalingFactorF64));
    ScalingFactor = scalingFactorF64;

    in->read(reinterpret_cast<char *>(EVALS), sizeof(EVALS));
    in->read(reinterpret_cast<char *>(EVALS_THREAT), sizeof(EVALS_THREAT));

    Score scores[PCODE_NB][2];
    for (int rule = 0; rule < RULE_NB + 1; rule++) {
        in->read(reinterpret_cast<char *>(scores), sizeof(scores));

        // Set score table to P4SCORES
        for (size_t pcode = 0; pcode < PCODE_NB; pcode++) {
            P4SCORES[rule][pcode][0] = scores[pcode][0];
            P4SCORES[rule][pcode][1] = scores[pcode][1];
        }
    }

    return *in && in->peek() == std::ios::traits_type::eof();
}

void Config::exportModel(std::ostream &outStream)
{
    Compressor    compressor(outStream, Compressor::Type::LZ4_DEFAULT);
    std::ostream *out = compressor.openOutputStream();
    assert(out);

    double scalingFactorF64 = ScalingFactor;
    out->write(reinterpret_cast<char *>(&scalingFactorF64), sizeof(scalingFactorF64));
    out->write(reinterpret_cast<char *>(EVALS), sizeof(EVALS));
    out->write(reinterpret_cast<char *>(EVALS_THREAT), sizeof(EVALS_THREAT));

    Score scores[PCODE_NB][2];
    for (int rule = 0; rule < RULE_NB + 1; rule++) {
        // Get score table out of P4SCORES
        for (size_t pcode = 0; pcode < PCODE_NB; pcode++) {
            scores[pcode][0] = (Score)P4SCORES[rule][pcode][0];
            scores[pcode][1] = (Score)P4SCORES[rule][pcode][1];
        }

        out->write(reinterpret_cast<char *>(scores), sizeof(scores));
    }
}

std::unique_ptr<::Search::Searcher> Config::createSearcher(std::string searcherName)
{
    if (searcherName.empty())
        searcherName = DefaultSearcherName;

    upperInplace(searcherName);

    if (searcherName == "ALPHABETA")
        return std::make_unique<Search::AB::ABSearcher>();
    if (searcherName == "MCTS")
        return std::make_unique<Search::MCTS::MCTSSearcher>();

    ERRORL("Unknown search type: " << searcherName
                                   << ", must be one of [alphabeta, mcts]."
                                      " Use alphabeta searcher as default.");
    return std::make_unique<Search::AB::ABSearcher>();
}

std::unique_ptr<::Database::DBStorage> Config::createDefaultDBStorage(std::string utf8URL)
{
    return DatabaseMaker ? DatabaseMaker(utf8URL.empty() ? DatabaseURL : utf8URL) : nullptr;
}
