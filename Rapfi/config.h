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

#pragma once

#include "core/types.h"
#include "core/utils.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <memory>

/// Total count of patterncode (pattern combination for 4 directions)
constexpr uint32_t PCODE_NB = combineNumber(PATTERN_NB, 4);

constexpr uint32_t THREAT_NB = power(2, 11);

/// MsgMode represents the message mode that controls how messages are outputed in search.
enum class MsgMode { NONE, BRIEF, NORMAL, UCILIKE };

/// CoordConvertionMode represents the coordinate convertion mode for protocol IO.
enum class CoordConvertionMode { NONE, X_FLIPY, FLIPY_X };

/// Pattern4Score struct packs score and pattern4 into a struct of 2 bytes.
struct Pattern4Score
{
    int32_t  _scoreSelf : 14;
    int32_t  _scoreOppo : 14;
    uint32_t _pattern4 : 4;

    struct ScoreProxy
    {
        Pattern4Score &v;
        bool           isOppo;

        ScoreProxy &operator=(Score score)
        {
            if (isOppo)
                v._scoreOppo = score;
            else
                v._scoreSelf = score;
            return *this;
        }
        operator Score() const { return Score(isOppo ? v._scoreOppo : v._scoreSelf); }
    };
    ScoreProxy operator[](size_t idx)
    {
        assert(idx < 2);
        return ScoreProxy {*this, idx != 0};
    }
    Score operator[](size_t idx) const
    {
        assert(idx < 2);
        return Score(idx != 0 ? _scoreOppo : _scoreSelf);
    }
    Score          scoreSelf() const { return (Score)_scoreSelf; }
    Score          scoreOppo() const { return (Score)_scoreOppo; }
    Pattern4Score &operator=(Pattern4 pattern4) { return _pattern4 = pattern4, *this; }
    operator Pattern4() const { return Pattern4(_pattern4); }
};
static_assert(sizeof(Pattern4Score) == sizeof(int32_t));

namespace Search {
class Searcher;  // forward declaration
}

namespace Database {
class DBStorage;           // forward declaration
enum class OverwriteRule;  // forward declaration
}  // namespace Database

namespace Config {

extern const std::string InternalConfig;

// -------------------------------------------------
// Model configs
extern float         ScalingFactor;
extern float         EvaluatorMarginWinLossScale;
extern float         EvaluatorMarginWinLossExponent;
extern float         EvaluatorMarginScale;
extern float         EvaluatorDrawBlackWinRate;
extern float         EvaluatorDrawRatio;
extern Eval          EVALS[RULE_NB + 1][PCODE_NB];
extern Eval          EVALS_THREAT[RULE_NB + 1][THREAT_NB];
extern Pattern4Score P4SCORES[RULE_NB + 1][PCODE_NB];

// -------------------------------------------------
// General options
extern bool                ReloadConfigEachMove;
extern bool                ClearHashAfterConfigLoaded;
extern size_t              DefaultThreadNum;
extern MsgMode             MessageMode;
extern CoordConvertionMode IOCoordMode;
extern CandidateRange      DefaultCandidateRange;
extern size_t              MemoryReservedMB[RULE_NB];
extern size_t              DefaultTTSizeKB;

// -------------------------------------------------
// Search options
extern bool AspirationWindow;
extern bool FilterSymmetryRootMoves;
extern int  NumIterationAfterMate;
extern int  NumIterationAfterSingularRoot;
extern int  MaxSearchDepth;

extern bool  ExpandWhenFirstEvaluate;
extern int   MaxNumVisitsPerPlayout;
extern int   NodesToPrintMCTSRootmoves;
extern int   TimeToPrintMCTSRootmoves;
extern int   MaxNonPVRootmovesToPrint;
extern int   NumNodesAfterSingularRoot;
extern int   NumNodeTableShardsPowerOfTwo;
extern float DrawUtilityPenalty;

// -------------------------------------------------
// Time management options
extern int   TurnTimeReserved;
extern float MatchSpace;
extern float MatchSpaceMin;
extern float AverageBranchFactor;
extern float AdvancedStopRatio;
extern int   MoveHorizon;
extern float TimeDivisorBias;
extern float TimeDivisorScale;
extern float TimeDivisorDepthPow;
extern float FallingFactorScale;
extern float FallingFactorBias;
extern float BestmoveStableReductionScale;
extern float BestmoveStablePrevReductionPow;

// -------------------------------------------------
// Database options
extern bool        DatabaseDefaultEnabled;
extern uint16_t    DatabaseLegacyFileCodePage;
extern std::string DatabaseType;
extern std::string DatabaseURL;
extern size_t      DatabaseCacheSize;
extern size_t      DatabaseRecordCacheSize;

// Library import options
extern char DatabaseLibBlackWinMark;
extern char DatabaseLibWhiteWinMark;
extern char DatabaseLibBlackLoseMark;
extern char DatabaseLibWhiteLoseMark;
extern bool DatabaseLibIgnoreComment;
extern bool DatabaseLibIgnoreBoardText;

// Database search options
extern bool                      DatabaseReadonlyMode;
extern bool                      DatabaseMandatoryParentWrite;
extern int                       DatabaseQueryPly;
extern int                       DatabaseQueryPVIterPerPlyIncrement;
extern int                       DatabaseQueryNonPVIterPerPlyIncrement;
extern int                       DatabasePVWritePly;
extern int                       DatabasePVWriteMinDepth;
extern int                       DatabaseNonPVWritePly;
extern int                       DatabaseNonPVWriteMinDepth;
extern int                       DatabaseWriteValueRange;
extern int                       DatabaseMateWritePly;
extern int                       DatabaseMateWriteMinDepthExact;
extern int                       DatabaseMateWriteMinDepthNonExact;
extern int                       DatabaseMateWriteMinStep;
extern int                       DatabaseExactOverwritePly;
extern int                       DatabaseNonExactOverwritePly;
extern ::Database::OverwriteRule DatabaseOverwriteRule;
extern int                       DatabaseOverwriteExactBias;
extern int                       DatabaseOverwriteDepthBoundBias;
extern int                       DatabaseQueryResultDepthBoundBias;

// -------------------------------------------------
// Eval/Score/Pattern4 Query

/// Get table index for rule and color.
constexpr int tableIndex(Rule r, Color c)
{
    return r + (r == Rule::RENJU ? c : 0);
}

/// Lookup eval table with color and pcode of rule R.
inline Value getValueBlack(Rule R, PatternCode pcodeBlack, PatternCode pcodeWhite)
{
    Value valueBlack = (Value)EVALS[tableIndex(R, BLACK)][pcodeBlack];
    Value valueWhite = (Value)EVALS[tableIndex(R, WHITE)][pcodeWhite];
    return valueBlack - valueWhite;
}

/// Lookup pattern4 & score table with color and pcode of rule R.
inline Pattern4Score getP4Score(Rule R, Color C, PatternCode pcode)
{
    return P4SCORES[tableIndex(R, C)][pcode];
}

/// Converts a evaluation value to winning rate (in [0, 1]) using current ScalingFactor.
template <bool Strict = true>
inline float valueToWinRate(Value eval)
{
    if (eval >= (Strict ? VALUE_MATE_IN_MAX_PLY : VALUE_EVAL_MAX))
        return 1.0f;
    if (eval <= (Strict ? VALUE_MATED_IN_MAX_PLY : VALUE_EVAL_MIN))
        return 0.0f;
    return 1.0f / (1.0f + ::expf(-float(eval) / ScalingFactor));
}

/// Converts a winning rate in [0, 1] to a evaluation value using current ScalingFactor.
inline Value winRateToValue(float winRate)
{
    float valueF32 = ScalingFactor * ::logf(winRate / (1.0f - winRate));
    valueF32       = std::clamp<float>(valueF32, VALUE_EVAL_MIN, VALUE_EVAL_MAX);
    return Value(valueF32);
}

// -------------------------------------------------
// Config loading & exporting

/// Loads TOML config from a text stream.
/// @param configStream Input config text stream.
/// @param skipModelLoading Whether to skip model loading. This can be set when
/// model is overrided in command line.
/// @return True if config is loaded successfully.
bool loadConfig(std::istream &configStream);

/// Load a LZ4 compressed classical evaluation model from a binary stream.
/// @return Returns true if loaded successfully, otherwise returns false.
bool loadModel(std::istream &inStream);

/// Exports current classic evaluation model to a binary stream.
void exportModel(std::ostream &outStream);

// -------------------------------------------------
// Searcher loading

/// Create a searcher instance from config.
/// @param searcherName The name of the searcher. Empty string to use default.
/// @return The popinter to the searcher. Can not be nullptr.
std::unique_ptr<::Search::Searcher> createSearcher(std::string searcherName = "");

// -------------------------------------------------
// Database loading

/// Create a default database storage instance from config.
/// @param url URL of the database, empty for default url from config.
/// @return The pointer to DBStorage instance, or nullptr if can not create.
std::unique_ptr<::Database::DBStorage> createDefaultDBStorage(std::string url = "");

}  // namespace Config
