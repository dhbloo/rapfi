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
#include "core/compressor.h"
#include "core/iohelper.h"
#include "database/dbconfig.h"
#include "eval/classicalpolicy.h"
#include "eval/evalconfig.h"
#include "eval/scoretables.h"
#include "game/pattern.h"
#include "search/hashtable.h"
#include "search/searchconfig.h"
#include "search/searcher.h"
#include "search/searchthread.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cpptoml.h>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <limits>
#include <optional>
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

constexpr std::array<char, 8> ClassicalModelExtensionMagic =
    {'R', 'F', 'C', 'L', 'M', 'O', 'D', '1'};
constexpr uint32_t ClassicalModelCompactP3Version = 7;
constexpr uint32_t ModelPolicyTableCount          = 1;
constexpr uint32_t ModelPolicyContextCount        = Evaluation::PolicyStoredContextCount;
constexpr uint32_t ModelBlendComponentCount       = 0;
constexpr uint32_t PolicyCrossPatternCount        = Evaluation::PolicyStoredPatternCount;
constexpr uint32_t PolicyCrossPayloadBytes =
    ModelPolicyContextCount * PolicyCrossPatternCount * PolicyCrossPatternCount * sizeof(Score);
constexpr uint32_t ClassicalModelCompactP3PayloadBytes = PolicyCrossPayloadBytes;
static_assert(ModelPolicyContextCount == Evaluation::QUIET);
static_assert(Evaluation::QUIET + 1 == Evaluation::POLICY_CONTEXT_NB);
static_assert(ModelPolicyContextCount < Evaluation::PolicyContextCount);

using StoredPolicyCross =
    Score[ModelPolicyContextCount][PolicyCrossPatternCount][PolicyCrossPatternCount];

}  // namespace

namespace Evaluation {

/// Scaling Factor is used for conversion between eval value and win rate.
/// Formula: win rate = sigmoid(eval / ScalingFactor)
float ScalingFactor = 200.0f;

// Classical evaluation and score tables
// Note that Renju has asymmetry eval and score

Eval                                                   EVALS[RULE_NB + 1][PCODE_NB];
Eval                                                   EVALS_THREAT[RULE_NB + 1][THREAT_NB];
MoveScorePair                                          P4SCORES[RULE_NB + 1][PCODE_NB];
ClassicalValueReadout                                  CLASSICAL_VALUE_READOUT;
std::array<Value, VALUE_EVAL_MAX - VALUE_EVAL_MIN + 1> CLASSICAL_VALUE_READOUT_CACHE {};
void refreshClassicalValueReadoutCache()
{
    static constexpr std::array<double, ClassicalValueReadout::KnotCount> XKnots =
        {0.0, 0.5, 1.0, 2.0, 4.0, 8.0};
    const auto  &yKnots = CLASSICAL_VALUE_READOUT.knots;
    const double scale  = ScalingFactor;

    for (int raw = VALUE_EVAL_MIN; raw <= VALUE_EVAL_MAX; raw++) {
        double x = std::abs(double(raw)) / scale;
        double y;
        if (x >= XKnots.back())
            y = yKnots.back() + x - XKnots.back();
        else {
            size_t hi = 1;
            while (x > XKnots[hi])
                hi++;
            size_t lo = hi - 1;
            double t  = (x - XKnots[lo]) / (XKnots[hi] - XKnots[lo]);
            y         = yKnots[lo] + t * (yKnots[hi] - yKnots[lo]);
        }
        double mapped = std::copysign(y * scale, double(raw));
        CLASSICAL_VALUE_READOUT_CACHE[raw - VALUE_EVAL_MIN] =
            Value(std::clamp<long long>(std::llround(mapped), VALUE_EVAL_MIN, VALUE_EVAL_MAX));
    }
}

Value mapClassicalValue(Value rawValue)
{
    assert(VALUE_EVAL_MIN <= rawValue && rawValue <= VALUE_EVAL_MAX);
    return CLASSICAL_VALUE_READOUT_CACHE[int(rawValue) - VALUE_EVAL_MIN];
}

}  // namespace Evaluation

namespace Config {

// -------------------------------------------------
// General options

GeneralConfig GeneralCfg;

// -------------------------------------------------

/// PendingConfig stages one loadConfig pass. The struct copies are seeded
/// from the live globals so that absent keys inherit the current values and
/// present keys override them, exactly as the old in-place writes did; the
/// remaining members record the engine effects requested by the parsed keys.
/// Nothing is published until loadConfig's commit step.
struct PendingConfig
{
    GeneralConfig                     general      = GeneralCfg;
    Search::SearchConfig              search       = Search::SearchCfg;
    Search::TimeConfig                time         = Search::TimeCfg;
    Database::DatabaseConfig          database     = Database::DatabaseCfg;
    Evaluation::EvaluatorConfig       eval         = Evaluation::EvalCfg;
    Evaluation::ClassicalValueReadout valueReadout = Evaluation::CLASSICAL_VALUE_READOUT;

    /// "[search] default_searcher" was present: switch the searcher at commit.
    std::optional<std::string> searcherName;
    /// "[general]" was present with a nonzero (inherited or new)
    /// default_tt_size_kb: resize the TT at commit.
    bool applyTTSize = false;
    /// "[database]" was present with enable_by_default resolved true: attach
    /// a storage at commit.
    bool attachDatabase = false;
    /// "[model.evaluator]" parsed a valid type + weights: the maker to install
    /// at commit. Left null otherwise, so the commit resets to classical eval
    /// (mirroring the old reset-then-maybe-install sequence).
    Evaluation::EvaluatorMakerFunc evaluatorMaker;
    /// Evaluator type name for the commit-time "Evaluator set to" message.
    std::string evaluatorName;
};

void readRequirement(const cpptoml::table &t);
void readGeneral(const cpptoml::table &t, PendingConfig &pending);
void readSearch(const cpptoml::table &t, PendingConfig &pending);
void readModel(const cpptoml::table &t, PendingConfig &pending);
void readEvaluator(const cpptoml::table &t, PendingConfig &pending);
void readDatabase(const cpptoml::table &t, PendingConfig &pending);
template <typename ValueType,
          ValueType MinVal    = std::numeric_limits<ValueType>::lowest(),
          ValueType MaxVal    = std::numeric_limits<ValueType>::max(),
          typename SetterType = void(PatternCode, ValueType)>
void readValueModel(const cpptoml::table &t, SetterType setter);

}  // namespace Config

/// Load config from a stream, transactionally.
///
/// The readers parse the whole document into a staged PendingConfig (building
/// the one fallible resource, the evaluator maker, along the way); only after
/// everything parsed does the commit step publish the structs to the live
/// globals and apply the engine effects: searcher switch (default_searcher),
/// TT resize (default_tt_size_kb), database attach (enable_by_default),
/// evaluator maker install. A load that fails in any way leaves the previous
/// configuration and engine state fully intact. The one exception is the
/// classical model surface written by readModel (EVALS / P4SCORES /
/// ScalingFactor, and binary_file via loadModelFromFile): those tables are
/// still written in place during parse.
/// @param configStream A input stream that contains a config file.
/// @return Returns true if loading succeeded, otherwise returns false.
bool Config::loadConfig(std::istream &configStream)
{
    // Callers signal stopThinking() but do not wait, so quiesce the search
    // threads before touching any state they read: the classical model tables
    // are written in place during parse, and the commit publishes the config
    // structs that search reads directly.
    Search::Engine.waitForIdle();

    PendingConfig pending;

    try {
        auto c = cpptoml::parser(configStream).parse();

        if (auto requirement = c->get_table("requirement"))
            readRequirement(*requirement);

        if (auto general = c->get_table("general"))
            readGeneral(*general, pending);

        if (auto search = c->get_table("search"))
            readSearch(*search, pending);

        if (auto database = c->get_table("database"))
            readDatabase(*database, pending);

        if (auto model = c->get_table("model"))
            readModel(*model, pending);
    }
    catch (const std::exception &e) {
        ERRORL("Failed to load config: " << e.what());
        return false;
    }

    // Commit: publish the parsed structs, then apply the engine effects.
    GeneralCfg                          = pending.general;
    Search::SearchCfg                   = pending.search;
    Search::TimeCfg                     = pending.time;
    Database::DatabaseCfg               = pending.database;
    Evaluation::EvalCfg                 = pending.eval;
    Evaluation::CLASSICAL_VALUE_READOUT = pending.valueReadout;
    if (Evaluation::CLASSICAL_VALUE_READOUT.knotsActive)
        Evaluation::refreshClassicalValueReadoutCache();

    // The searcher switch precedes the TT resize: setupSearcher carries the
    // old searcher's memory limit onto the new one, and the resize then
    // overrides it - the same end state the old resize-then-switch produced.
    if (pending.searcherName)
        Search::Engine.setupSearcher(::Search::createSearcher(*pending.searcherName));
    if (pending.applyTTSize)
        Search::Engine.searcher()->setMemoryLimit(GeneralCfg.defaultTTSizeKB);
    // Detach (and thereby flush) any previous database before creating the
    // new storage: YXDBStorage reads the file on construction, and dirty
    // thread DBClients write back into the old storage only when it detaches,
    // so a same-path reload must not open the file ahead of that flush.
    Search::Engine.setupDatabase(nullptr);
    if (pending.attachDatabase)
        Search::Engine.setupDatabase(Database::createDBStorage(Database::DatabaseCfg));
    Search::Engine.setupEvaluator(std::move(pending.evaluatorMaker));
    if (!pending.evaluatorName.empty())
        MESSAGEL("Evaluator set to " << pending.evaluatorName << ".");

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
void Config::readGeneral(const cpptoml::table &t, PendingConfig &pending)
{
    auto &cfg = pending.general;

    cfg.reloadConfigEachMove =
        t.get_as<bool>("reload_config_each_move").value_or(cfg.reloadConfigEachMove);
    cfg.clearHashAfterConfigLoaded =
        t.get_as<bool>("clear_hash_after_config_loaded").value_or(cfg.clearHashAfterConfigLoaded);

    // Read Default Thread Num
    cfg.defaultThreadNum = t.get_as<uint64_t>("default_thread_num").value_or(cfg.defaultThreadNum);
    if (cfg.defaultThreadNum == 0) {
#ifdef MULTI_THREADING
        cfg.defaultThreadNum = std::thread::hardware_concurrency();
        MESSAGEL("Setting default thread num to " << cfg.defaultThreadNum << ".");
#else
        cfg.defaultThreadNum = 1;
#endif
    }

    // Read Message Mode
    if (t.get_as<std::string>("message_mode")) {
        std::string msgModeStr = *t.get_as<std::string>("message_mode");
        if (msgModeStr == "normal")
            cfg.messageMode = MsgMode::NORMAL;
        else if (msgModeStr == "brief")
            cfg.messageMode = MsgMode::BRIEF;
        else if (msgModeStr == "ucilike")
            cfg.messageMode = MsgMode::UCILIKE;
        else {
            if (msgModeStr != "none")
                MESSAGEL("Warning: unknown message mode [" << msgModeStr << "], reset to [none].");
            cfg.messageMode = MsgMode::NONE;
        }
    }

    // Read Coord Conversion Mode
    if (t.get_as<std::string>("coord_conversion_mode")) {
        std::string coordModeStr = *t.get_as<std::string>("coord_conversion_mode");
        if (coordModeStr == "X_flipY")
            cfg.ioCoordMode = CoordConversionMode::X_FLIPY;
        else if (coordModeStr == "flipY_X")
            cfg.ioCoordMode = CoordConversionMode::FLIPY_X;
        else {
            if (coordModeStr != "none")
                MESSAGEL("Warning: unknown coordinate conversion mode [" << coordModeStr
                                                                         << "], reset to [none].");
            cfg.ioCoordMode = CoordConversionMode::NONE;
        }
    }

    // Read Default Condidate Range Mode
    if (t.get_as<std::string>("default_candidate_range")) {
        std::string candRangeModeStr = *t.get_as<std::string>("default_candidate_range");
        if (candRangeModeStr == "square2")
            cfg.defaultCandidateRange = CandidateRange::SQUARE2;
        else if (candRangeModeStr == "square2_line3")
            cfg.defaultCandidateRange = CandidateRange::SQUARE2_LINE3;
        else if (candRangeModeStr == "square3")
            cfg.defaultCandidateRange = CandidateRange::SQUARE3;
        else if (candRangeModeStr == "square3_line4")
            cfg.defaultCandidateRange = CandidateRange::SQUARE3_LINE4;
        else if (candRangeModeStr == "square4")
            cfg.defaultCandidateRange = CandidateRange::SQUARE4;
        else if (candRangeModeStr == "full_board")
            cfg.defaultCandidateRange = CandidateRange::FULL_BOARD;
        else {
            MESSAGEL("Warning: unknown candidate range [" << candRangeModeStr
                                                          << "], reset to [square2_line3].");
            cfg.defaultCandidateRange = CandidateRange::SQUARE2_LINE3;
        }
    }

    // Read memory reserved for each rule
    if (auto table = t.get_array("memory_reserved_mb")) {
        if (auto array = table->get_array_of<int64_t>()) {
            for (int i = 0; i < RULE_NB; i++)
                cfg.memoryReservedMB[i] = array->at(std::min<size_t>(i, array->size() - 1));
        }
    }
    else {
        auto v = t.get_as<uint64_t>("memory_reserved_mb");
        for (int i = 0; i < RULE_NB; i++)
            cfg.memoryReservedMB[i] = v.value_or(cfg.memoryReservedMB[i]);
    }

    cfg.defaultTTSizeKB = t.get_as<uint64_t>("default_tt_size_kb").value_or(cfg.defaultTTSizeKB);
    // Request a TT resize at commit according to default TT size (overriding previous size)
    pending.applyTTSize = cfg.defaultTTSizeKB > 0;
}

/// Read search table of the config.
void Config::readSearch(const cpptoml::table &t, PendingConfig &pending)
{
    if (auto v = t.get_as<std::string>("default_searcher"); v)
        pending.searcherName = *v;

    auto &cfg     = pending.search;
    auto &timeCfg = pending.time;

    // Parameters for alpha-beta search
    cfg.aspirationWindow = t.get_as<bool>("aspiration_window").value_or(cfg.aspirationWindow);
    cfg.filterSymmetryRootMoves =
        t.get_as<bool>("filter_symmetry_root_moves").value_or(cfg.filterSymmetryRootMoves);
    cfg.numIterationAfterMate =
        t.get_as<int>("num_iteration_after_mate").value_or(cfg.numIterationAfterMate);
    cfg.numIterationAfterSingularRoot = t.get_as<int>("num_iteration_after_singular_root")
                                            .value_or(cfg.numIterationAfterSingularRoot);
    cfg.maxSearchDepth = t.get_as<int>("max_search_depth").value_or(cfg.maxSearchDepth);

    // Parameters for MCTS search
    cfg.expandWhenFirstEvaluate =
        t.get_as<bool>("expand_when_first_evaluate").value_or(cfg.expandWhenFirstEvaluate);
    cfg.maxNumVisitsPerPlayout =
        t.get_as<int>("max_num_visits_per_playout").value_or(cfg.maxNumVisitsPerPlayout);
    cfg.nodesToPrintMCTSRootmoves =
        t.get_as<int>("nodes_to_print_mcts_rootmoves").value_or(cfg.nodesToPrintMCTSRootmoves);
    cfg.timeToPrintMCTSRootmoves =
        t.get_as<int>("time_to_print_mcts_rootmoves").value_or(cfg.timeToPrintMCTSRootmoves);
    cfg.maxNonPVRootmovesToPrint =
        t.get_as<int>("max_non_pv_rootmoves_to_print").value_or(cfg.maxNonPVRootmovesToPrint);
    cfg.numNodesAfterSingularRoot =
        t.get_as<int>("num_nodes_after_singular_root").value_or(cfg.numNodesAfterSingularRoot);
    cfg.numNodeTableShardsPowerOfTwo = t.get_as<int>("num_node_table_shards_power_of_two")
                                           .value_or(cfg.numNodeTableShardsPowerOfTwo);
    cfg.drawUtilityPenalty =
        t.get_as<double>("draw_utility_penalty").value_or(cfg.drawUtilityPenalty);
    cfg.mctsVCFTTMaxSizeKB =
        t.get_as<int>("mcts_vcf_tt_max_size_kb").value_or(cfg.mctsVCFTTMaxSizeKB);
    cfg.mctsVCFTTBudgetDivisor =
        t.get_as<int>("mcts_vcf_tt_budget_divisor").value_or(cfg.mctsVCFTTBudgetDivisor);

    // Read time management options
    if (auto tm = t.get_table("timectl")) {
        timeCfg.turnTimeReserved =
            tm->get_as<int>("turn_time_reserved").value_or(timeCfg.turnTimeReserved);
        timeCfg.matchSpace = tm->get_as<double>("match_space").value_or(timeCfg.matchSpace);
        timeCfg.matchSpaceMin =
            tm->get_as<double>("match_space_min").value_or(timeCfg.matchSpaceMin);
        timeCfg.averageBranchFactor =
            tm->get_as<double>("average_branch_factor").value_or(timeCfg.averageBranchFactor);
        timeCfg.advancedStopRatio =
            tm->get_as<double>("advanced_stop_ratio").value_or(timeCfg.advancedStopRatio);
        timeCfg.moveHorizon = tm->get_as<int>("move_horizon").value_or(timeCfg.moveHorizon);

        timeCfg.timeDivisorScale =
            tm->get_as<double>("time_divisor_scale").value_or(timeCfg.timeDivisorScale);
        timeCfg.timeDivisorBias =
            tm->get_as<double>("time_divisor_bias").value_or(timeCfg.timeDivisorBias);
        timeCfg.timeDivisorDepthPow =
            tm->get_as<double>("time_divisor_depth_pow").value_or(timeCfg.timeDivisorDepthPow);

        timeCfg.fallingFactorScale =
            tm->get_as<double>("falling_factor_scale").value_or(timeCfg.fallingFactorScale);
        timeCfg.fallingFactorBias =
            tm->get_as<double>("falling_factor_bias").value_or(timeCfg.fallingFactorBias);

        timeCfg.bestmoveStableReductionScale = tm->get_as<double>("bestmove_stable_reduction_scale")
                                                   .value_or(timeCfg.bestmoveStableReductionScale);
        timeCfg.bestmoveStablePrevReductionPow =
            tm->get_as<double>("bestmove_stable_prev_reduction_pow")
                .value_or(timeCfg.bestmoveStablePrevReductionPow);
    }
}

/// Read model table of all rules in the config. The classical model values
/// (EVALS / P4SCORES / ScalingFactor and optional binary context-policy rows)
/// are written in place, not staged.
void Config::readModel(const cpptoml::table &t, PendingConfig &pending)
{
    const Rule  Rules[]    = {FREESTYLE, STANDARD, RENJU};
    const char *RuleName[] = {"freestyle", "standard", "renju"};

    pending.valueReadout = {};

    std::string modelPath = t.get_as<std::string>("binary_file").value_or("");
    if (!modelPath.empty()) {
        if (!Command::loadModelFromFile(modelPath))
            throw std::runtime_error("failed to load classic model file");
    }
    else {
        // Inline/legacy TOML models carry neither threat tables nor optional
        // binary extensions. Reset both before applying inline values.
        Evaluation::resetPolicyCross();
        std::memset(Evaluation::EVALS_THREAT, 0, sizeof(Evaluation::EVALS_THREAT));

        // Read Eval & Score
        if (auto eval = t.get_table("eval")) {
            for (Rule r : Rules) {
                bool hasAsymmetryRenjuEval = false;
                auto setEvalBlack          = [r](PatternCode pcode, Eval ev) {
                    Evaluation::EVALS[r + BLACK][pcode] = ev;
                };
                auto setEvalWhite = [r](PatternCode pcode, Eval ev) {
                    Evaluation::EVALS[r + WHITE][pcode] = ev;
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
                                                       Evaluation::P4SCORES[tableIdx][pcode][0] =
                                                           score;
                                                   });
                readValueModel<Score, -8192, 8191>(oppoTable ? *oppoTable : t,
                                                   [tableIdx](PatternCode pcode, Score score) {
                                                       Evaluation::P4SCORES[tableIdx][pcode][1] =
                                                           score;
                                                   });
            };
            for (Rule r : Rules) {
                bool hasAsymmetryRenjuScore = false;
                auto ruleScore              = score->get_table(RuleName[r]);
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
    double configuredScalingFactor =
        t.get_as<double>("scaling_factor").value_or(Evaluation::ScalingFactor);
    float runtimeScalingFactor = static_cast<float>(configuredScalingFactor);
    auto  readoutKnots         = t.get_array_of<double>("value_readout_knots");
    if (readoutKnots) {
        if (readoutKnots->size() != Evaluation::ClassicalValueReadout::KnotCount)
            throw std::runtime_error("value_readout_knots must contain 6 values");
        for (size_t i = 0; i < readoutKnots->size(); i++) {
            double value = (*readoutKnots)[i];
            if (!std::isfinite(value) || value < 0.0 || value > 64.0 || (i == 0 && value != 0.0)
                || (i != 0 && value < (*readoutKnots)[i - 1]))
                throw std::runtime_error("value_readout_knots must be finite, monotone, start at "
                                         "zero, and not exceed 64");
            pending.valueReadout.knots[i] = value;
        }
        pending.valueReadout.knotsActive = true;
    }

    if (pending.valueReadout.knotsActive
        && (!std::isfinite(runtimeScalingFactor) || runtimeScalingFactor <= 0.0f))
        throw std::runtime_error(
            "classical value readout requires a finite positive scaling_factor");
    Evaluation::ScalingFactor = runtimeScalingFactor;

    // Read evaluator
    if (auto evaluator = t.get_table("evaluator"))
        readEvaluator(*evaluator, pending);
}

/// Read evaluator table in the config.
void Config::readEvaluator(const cpptoml::table &t, PendingConfig &pending)
{
    auto evaluatorType = t.get_as<std::string>("type");
    auto weights       = t.get_table_array("weights");
    // A present-but-partial evaluator table is always a config mistake (an intentional
    // classical setup omits the table entirely); silently falling back to classical
    // eval here would corrupt any match or test built on this config.
    if (!evaluatorType)
        throw std::runtime_error("evaluator table must specify a type");
    if (!weights || weights->begin() == weights->end())
        throw std::runtime_error("evaluator " + *evaluatorType + " has no weights entries");

    Evaluation::EvaluatorWeightsConfig weightsCfg;
    weightsCfg.type      = *evaluatorType;
    weightsCfg.ortDevice = t.get_as<std::string>("ort_device").value_or("");
    for (auto weightCfg : *weights) {
        Evaluation::EvaluatorWeightsConfig::WeightFiles wf;
        if (auto v = weightCfg->get_as<std::string>("weight_file"))
            wf.file = *v;
        if (auto v = weightCfg->get_as<std::string>("weight_file_black"))
            wf.fileBlack = *v;
        if (auto v = weightCfg->get_as<std::string>("weight_file_white"))
            wf.fileWhite = *v;
        weightsCfg.weights.push_back(std::move(wf));
    }

    pending.evaluatorMaker =
        Evaluation::makeEvaluatorMaker(std::move(weightsCfg), Command::getModelFullPath);
    pending.evaluatorName = *evaluatorType;

    // Read classical/evaluator switching margin
    auto &evalCfg = pending.eval;
    evalCfg.marginWinLossScale =
        (float)t.get_as<double>("margin_winloss_scale").value_or(evalCfg.marginWinLossScale);
    evalCfg.marginWinLossExponent =
        (float)t.get_as<double>("margin_winloss_exp").value_or(evalCfg.marginWinLossExponent);
    evalCfg.marginScale = (float)t.get_as<double>("margin_scale").value_or(evalCfg.marginScale);
    evalCfg.drawBlackWinRate =
        (float)t.get_as<double>("draw_black_winrate").value_or(evalCfg.drawBlackWinRate);
    evalCfg.drawRatio        = (float)t.get_as<double>("draw_ratio").value_or(evalCfg.drawRatio);
    evalCfg.drawBlackWinRate = std::clamp(evalCfg.drawBlackWinRate, 0.0f, 1.0f);
    evalCfg.drawRatio        = std::clamp(evalCfg.drawRatio, 0.0f, 1.0f);
}

/// Read database table in the config.
void Config::readDatabase(const cpptoml::table &t, PendingConfig &pending)
{
    auto &cfg = pending.database;

    cfg.defaultEnabled  = t.get_as<bool>("enable_by_default").value_or(cfg.defaultEnabled);
    cfg.type            = t.get_as<std::string>("type").value_or(cfg.type);
    cfg.url             = t.get_as<std::string>("url").value_or(cfg.url);
    cfg.cacheSize       = t.get_as<size_t>("cache_size").value_or(cfg.cacheSize);
    cfg.recordCacheSize = t.get_as<size_t>("record_cache_size").value_or(cfg.recordCacheSize);
    cfg.legacyFileCodePage =
        t.get_as<int>("legacy_file_code_page").value_or(cfg.legacyFileCodePage);
    cfg.factoryEnabled = false;

    if (cfg.type == "yixindb") {
        if (cfg.url.empty())
            cfg.url = "rapfi.db";

        cfg.yixindb = {};
        if (auto args = t.get_table("yixindb")) {
            cfg.yixindb.compressedSave =
                args->get_as<bool>("compressed_save").value_or(cfg.yixindb.compressedSave);
            cfg.yixindb.saveOnClose =
                args->get_as<bool>("save_on_close").value_or(cfg.yixindb.saveOnClose);
            cfg.yixindb.numBackupsOnSave =
                args->get_as<int>("num_backups_on_save").value_or(cfg.yixindb.numBackupsOnSave);
            cfg.yixindb.ignoreCorrupted =
                args->get_as<bool>("ignore_corrupted").value_or(cfg.yixindb.ignoreCorrupted);
        }
        cfg.factoryEnabled = true;
    }
    else if (!cfg.type.empty()) {
        throw std::runtime_error("unsupported database type " + cfg.type);
    }

    if (auto s = t.get_table("search")) {
        auto &dbs                = cfg.search;
        dbs.readonlyMode         = s->get_as<bool>("readonly_mode").value_or(false);
        dbs.mandatoryParentWrite = s->get_as<bool>("mandatory_parent_write").value_or(true);
        dbs.queryPly             = s->get_as<int>("query_ply").value_or(dbs.queryPly);
        dbs.queryPVIterPerPlyIncrement =
            s->get_as<int>("pv_iter_per_ply_increment").value_or(dbs.queryPVIterPerPlyIncrement);
        dbs.queryNonPVIterPerPlyIncrement = s->get_as<int>("nonpv_iter_per_ply_increment")
                                                .value_or(dbs.queryNonPVIterPerPlyIncrement);

        dbs.pvWritePly      = s->get_as<int>("pv_write_ply").value_or(dbs.pvWritePly);
        dbs.pvWriteMinDepth = s->get_as<int>("pv_write_min_depth").value_or(dbs.pvWriteMinDepth);

        dbs.nonPVWritePly = s->get_as<int>("nonpv_write_ply").value_or(dbs.nonPVWritePly);
        dbs.nonPVWriteMinDepth =
            s->get_as<int>("nonpv_write_min_depth").value_or(dbs.nonPVWriteMinDepth);

        dbs.writeValueRange = s->get_as<int>("write_value_range").value_or(dbs.writeValueRange);

        dbs.mateWritePly = s->get_as<int>("mate_write_ply").value_or(dbs.mateWritePly);
        dbs.mateWriteMinDepthExact =
            s->get_as<int>("mate_write_min_depth_exact").value_or(dbs.mateWriteMinDepthExact);
        dbs.mateWriteMinDepthNonExact =
            s->get_as<int>("mate_write_min_depth_nonexact").value_or(dbs.mateWriteMinDepthNonExact);
        dbs.mateWriteMinStep = s->get_as<int>("mate_write_min_step").value_or(dbs.mateWriteMinStep);

        dbs.exactOverwritePly =
            s->get_as<int>("exact_overwrite_ply").value_or(dbs.exactOverwritePly);
        dbs.nonExactOverwritePly =
            s->get_as<int>("nonexact_overwrite_ply").value_or(dbs.nonExactOverwritePly);

        if (auto overwriteRule =
                s->get_as<std::string>("overwrite_rule").value_or("better_value_depth_bound");
            overwriteRule == "better_value_depth_bound")
            dbs.overwriteRule = ::Database::OverwriteRule::BetterValueDepthBound;
        else if (overwriteRule == "better_depth_bound")
            dbs.overwriteRule = ::Database::OverwriteRule::BetterDepthBound;
        else if (overwriteRule == "better_value")
            dbs.overwriteRule = ::Database::OverwriteRule::BetterValue;
        else if (overwriteRule == "better_label")
            dbs.overwriteRule = ::Database::OverwriteRule::BetterLabel;
        else if (overwriteRule == "always")
            dbs.overwriteRule = ::Database::OverwriteRule::Always;
        else if (overwriteRule == "disabled")
            dbs.overwriteRule = ::Database::OverwriteRule::Disabled;
        else
            MESSAGEL("unknown database overwrite rule " << overwriteRule << ", keep it unchanged.");

        dbs.overwriteExactBias =
            s->get_as<int>("overwrite_exact_bias").value_or(dbs.overwriteExactBias);
        dbs.overwriteDepthBoundBias =
            s->get_as<int>("overwrite_depth_bound_bias").value_or(dbs.overwriteDepthBoundBias);
        dbs.queryResultDepthBoundBias =
            s->get_as<int>("query_result_depth_bound_bias").value_or(dbs.queryResultDepthBoundBias);
    }

    if (auto s = t.get_table("libfile")) {
        auto &lib           = cfg.libfile;
        lib.blackWinMark    = s->get_as<std::string>("black_win_mark").value_or("a")[0];
        lib.whiteWinMark    = s->get_as<std::string>("white_win_mark").value_or("a")[0];
        lib.blackLoseMark   = s->get_as<std::string>("black_lose_mark").value_or("c")[0];
        lib.whiteLoseMark   = s->get_as<std::string>("white_lose_mark").value_or("c")[0];
        lib.ignoreComment   = s->get_as<bool>("ignore_comment").value_or(false);
        lib.ignoreBoardText = s->get_as<bool>("ignore_board_text").value_or(false);
    }

    pending.attachDatabase = cfg.defaultEnabled;
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
                            setter(PatternConfig::PCODE[a][b][c][d].pcode(), ValueType(val));
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
                            setter(PatternConfig::PCODE[a][b][c][d].pcode(), ValueType(val));
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
    Eval   evals[RULE_NB + 1][PCODE_NB];
    Eval   evalsThreat[RULE_NB + 1][THREAT_NB];
    Score  scores[RULE_NB + 1][PCODE_NB][2];
    in->read(reinterpret_cast<char *>(&scalingFactorF64), sizeof(scalingFactorF64));
    in->read(reinterpret_cast<char *>(evals), sizeof(evals));
    in->read(reinterpret_cast<char *>(evalsThreat), sizeof(evalsThreat));
    in->read(reinterpret_cast<char *>(scores), sizeof(scores));

    if (!*in)
        return false;

    auto publishBaseTables = [&]() {
        Evaluation::ScalingFactor = static_cast<float>(scalingFactorF64);
        std::memcpy(Evaluation::EVALS, evals, sizeof(Evaluation::EVALS));
        std::memcpy(Evaluation::EVALS_THREAT, evalsThreat, sizeof(Evaluation::EVALS_THREAT));
        std::memcpy(Evaluation::P4SCORES, scores, sizeof(Evaluation::P4SCORES));
    };

    // Files produced before the context-policy extension end after the score
    // tables. They remain valid and use the engine's compiled blend rows.
    if (in->peek() == std::ios::traits_type::eof()) {
        if (in->bad())
            return false;
        publishBaseTables();
        Evaluation::resetPolicyCross();
        return true;
    }

    std::array<char, ClassicalModelExtensionMagic.size()> magic;
    uint32_t version, payloadBytes, tableCount, contextCount, componentCount, patternCount;
    in->read(magic.data(), magic.size());
    in->read(reinterpret_cast<char *>(&version), sizeof(version));
    in->read(reinterpret_cast<char *>(&payloadBytes), sizeof(payloadBytes));
    in->read(reinterpret_cast<char *>(&tableCount), sizeof(tableCount));
    in->read(reinterpret_cast<char *>(&contextCount), sizeof(contextCount));
    in->read(reinterpret_cast<char *>(&componentCount), sizeof(componentCount));
    in->read(reinterpret_cast<char *>(&patternCount), sizeof(patternCount));
    if (!*in || magic != ClassicalModelExtensionMagic || version != ClassicalModelCompactP3Version
        || tableCount != ModelPolicyTableCount || contextCount != ModelPolicyContextCount
        || patternCount != PolicyCrossPatternCount
        || payloadBytes != ClassicalModelCompactP3PayloadBytes
        || componentCount != ModelBlendComponentCount)
        return false;

    StoredPolicyCross policyCross;
    in->read(reinterpret_cast<char *>(policyCross), sizeof(policyCross));
    if (!*in)
        return false;

    if (in->peek() != std::ios::traits_type::eof() || in->bad())
        return false;
    publishBaseTables();
    Evaluation::resetPolicyCross();
    for (size_t context = 0; context < ModelPolicyContextCount; context++)
        for (size_t self = 0; self < PolicyCrossPatternCount; self++)
            for (size_t opponent = 0; opponent < PolicyCrossPatternCount; opponent++)
                Evaluation::POLICY_CROSS[FREESTYLE][context]
                                        [Evaluation::policyPatternFromStorageIndex(self)]
                                        [Evaluation::policyPatternFromStorageIndex(opponent)] =
                                            policyCross[context][self][opponent];
    Evaluation::activatePolicyCross();
    return true;
}

void Config::exportModel(std::ostream &outStream)
{
    Compressor    compressor(outStream, Compressor::Type::LZ4_DEFAULT);
    std::ostream *out = compressor.openOutputStream();
    assert(out);

    double scalingFactorF64 = Evaluation::ScalingFactor;
    out->write(reinterpret_cast<char *>(&scalingFactorF64), sizeof(scalingFactorF64));
    out->write(reinterpret_cast<char *>(Evaluation::EVALS), sizeof(Evaluation::EVALS));
    out->write(reinterpret_cast<char *>(Evaluation::EVALS_THREAT),
               sizeof(Evaluation::EVALS_THREAT));

    Score scores[PCODE_NB][2];
    for (int rule = 0; rule < RULE_NB + 1; rule++) {
        // Get score table out of P4SCORES
        for (size_t pcode = 0; pcode < PCODE_NB; pcode++) {
            scores[pcode][0] = (Score)Evaluation::P4SCORES[rule][pcode][0];
            scores[pcode][1] = (Score)Evaluation::P4SCORES[rule][pcode][1];
        }

        out->write(reinterpret_cast<char *>(scores), sizeof(scores));
    }

    bool writePolicyCross =
        Evaluation::isPolicyCrossPresent() || !Evaluation::hasDefaultPolicyCross();
    if (!writePolicyCross)
        return;
    out->write(ClassicalModelExtensionMagic.data(), ClassicalModelExtensionMagic.size());
    uint32_t version        = ClassicalModelCompactP3Version;
    uint32_t payloadBytes   = ClassicalModelCompactP3PayloadBytes;
    uint32_t tableCount     = ModelPolicyTableCount;
    uint32_t contextCount   = ModelPolicyContextCount;
    uint32_t componentCount = ModelBlendComponentCount;
    uint32_t patternCount   = PolicyCrossPatternCount;
    out->write(reinterpret_cast<const char *>(&version), sizeof(version));
    out->write(reinterpret_cast<const char *>(&payloadBytes), sizeof(payloadBytes));
    out->write(reinterpret_cast<const char *>(&tableCount), sizeof(tableCount));
    out->write(reinterpret_cast<const char *>(&contextCount), sizeof(contextCount));
    out->write(reinterpret_cast<const char *>(&componentCount), sizeof(componentCount));
    out->write(reinterpret_cast<const char *>(&patternCount), sizeof(patternCount));

    StoredPolicyCross policyCross;
    for (size_t context = 0; context < ModelPolicyContextCount; context++)
        for (size_t self = 0; self < PolicyCrossPatternCount; self++)
            for (size_t opponent = 0; opponent < PolicyCrossPatternCount; opponent++)
                policyCross[context][self][opponent] =
                    Evaluation::POLICY_CROSS[FREESTYLE][context]
                                            [Evaluation::policyPatternFromStorageIndex(self)]
                                            [Evaluation::policyPatternFromStorageIndex(opponent)];
    out->write(reinterpret_cast<const char *>(policyCross), sizeof(policyCross));
}
