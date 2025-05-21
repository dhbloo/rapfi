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

#include "../../core/types.h"
#include "../../tuning/tunemap.h"
#include "history.h"

#include <array>
#include <cmath>

namespace Search::AB {

// -------------------------------------------------
// Search limits

inline int MAX_DEPTH = 200;
inline int MAX_PLY   = 256;

// -------------------------------------------------
// Depth & Value Constants

// Search Constants
inline Value MARGIN_INFINITE  = Value(INT16_MAX);
inline Depth ASPIRATION_DEPTH = 5.0f;
inline Depth IID_DEPTH        = 14.0f;

TUNE(ASPIRATION_DEPTH);
TUNE(IID_DEPTH);

// Reductions

inline Depth IIR_REDUCTION                 = 0.65f;
inline Depth IIR_REDUCTION_PV              = 0.30f;
inline Depth IIR_REDUCTION_TT              = 0.31f;
inline Depth IIR_REDUCTION_TT_MAX          = 3.6f;
inline Depth TRIVIAL_PRUN_DEPTH            = 4.5f;
inline Depth LMR_EXTRA_MAX_DEPTH           = 5.0f;
inline Depth TTPV_NEG_REDUCTION            = 1.0f;
inline Depth NO_ALPHA_IMPROVING_REDUCTION  = 1.0f;
inline Depth NOKILLER_CUTNODE_REDUCTION    = 1.73f;
inline Depth OPPO_USELESS_DEFEND_REDUCTION = 2.0f;
inline Depth SELF_USELESS_DEFEND_REDUCTION = 1.0f;
inline Depth FALSE_FORBID_LESS_REDUCTION   = 1.0f;
inline Depth ALPHA_IMPROVEMENT_REDUCTION   = 1.0f;

TUNE(IIR_REDUCTION);
TUNE(IIR_REDUCTION_PV);
TUNE(IIR_REDUCTION_TT);
TUNE(IIR_REDUCTION_TT_MAX);
TUNE(TRIVIAL_PRUN_DEPTH);
TUNE(LMR_EXTRA_MAX_DEPTH);
TUNE(TTPV_NEG_REDUCTION);
TUNE(NO_ALPHA_IMPROVING_REDUCTION);
TUNE(NOKILLER_CUTNODE_REDUCTION);
TUNE(OPPO_USELESS_DEFEND_REDUCTION);
TUNE(SELF_USELESS_DEFEND_REDUCTION);
TUNE(FALSE_FORBID_LESS_REDUCTION);
TUNE(ALPHA_IMPROVEMENT_REDUCTION);

// Extensions

inline Depth OPPO5_EXT             = 1.33f;
inline Depth SE_DEPTH              = 7.0f;
inline Depth SE_TTE_DEPTH          = 1.96f;
inline Depth SE_EXTRA_MAX_DEPTH    = 12.0f;
inline Depth SE_REDUCTION_FH       = 1.57f;
inline Depth TTM_EXT_PV            = 0.24f;
inline Depth TTM_EXT_NONPV         = 0.08f;
inline Depth NEARB4_EXT_DIST4      = 0.23f;
inline Depth NEARB4_EXT_DIST6      = 0.05f;
inline Depth CONTINUOUS_ATTACK_EXT = 0.55f;

TUNE(OPPO5_EXT);
TUNE(SE_DEPTH);
TUNE(SE_TTE_DEPTH);
TUNE(SE_EXTRA_MAX_DEPTH);
TUNE(SE_REDUCTION_FH);
TUNE(TTM_EXT_PV);
TUNE(TTM_EXT_NONPV);
TUNE(NEARB4_EXT_DIST4);
TUNE(NEARB4_EXT_DIST6);
TUNE(CONTINUOUS_ATTACK_EXT);

// -------------------------------------------------
// Dynamic margin & reduction functions/LUTs

/// Aspiration window delta. When prevDelta is zero, returns the initial aspiration
/// window size. Otherwise returns the next expanded window size for the given prevDelta.
/// Window will expand faster for large absolute previous value.
inline Value nextAspirationWindowDelta(Value prevValue, Value prevDelta = VALUE_ZERO)
{
    return prevDelta ? prevDelta * (3 + std::abs(prevValue) / 1024) / 4 + 6 : Value(16);
}

inline float Razor1 = 0.18f;
inline int   Razor2 = 16;
inline int   Razor3 = 18;
TUNE(Razor1);
TUNE(Razor2);
TUNE(Razor3);

/// Razoring depth & margins
template <Rule R>
inline Value razorMargin(Depth d)
{
    return d < 2.6f ? Value(std::max(int(Razor1 * d * d + Razor2 * d) + Razor3, 0))
                    : MARGIN_INFINITE;
}

inline int FutilityScale[RULE_NB] = {63, 61, 78};
inline int FutilityNoTTCutNode    = 13;
TUNE(FutilityScale);
TUNE(FutilityNoTTCutNode);

/// Static futility pruning depth & margins
template <Rule R>
inline Value futilityMargin(Depth d, bool noTTCutNode, bool improving)
{
    return Value(
        std::max(int((FutilityScale[R] - FutilityNoTTCutNode * noTTCutNode) * (d - improving)), 0));
}

inline int NMMDepth[RULE_NB] = {10, 11, 8};
inline int NMMBias[RULE_NB]  = {585, 538, 652};
inline int NMMScale[RULE_NB] = {27, 27, 29};
inline int NMMMax[RULE_NB]   = {315, 345, 242};
TUNE(NMMDepth);
TUNE(NMMBias);
TUNE(NMMScale);
TUNE(NMMMax);

/// Null move pruning margin
template <Rule R>
inline Value nullMoveMargin(Depth d)
{
    return d >= NMMDepth[R] ? Value(NMMBias[R] - std::min(NMMScale[R] * int(d), NMMMax[R]))
                            : MARGIN_INFINITE;
}

inline float NMRBias[RULE_NB]  = {3.67f, 4.07f, 4.30f};
inline float NMRScale[RULE_NB] = {0.135f, 0.134f, 0.174f};
TUNE(NMRBias);
TUNE(NMRScale);

/// Null move search depth reduction. The result of a null move will be
/// tested using reduced depth search.
template <Rule R>
inline Depth nullMoveReduction(Depth d)
{
    return NMRBias[R] + NMRScale[R] * d;
}

inline Depth IDRBias[RULE_NB]  = {7.6f, 8.2f, 9.8f};
inline Depth IDRScale[RULE_NB] = {0.67f, 0.60f, 0.64f};
TUNE(IDRBias);
TUNE(IDRScale);

/// Internal iterative deepening depth reduction.
template <Rule R>
inline Depth iidDepthReduction(Depth d)
{
    return IDRBias[R] + IDRScale[R] * d;
}

inline int FHBias[RULE_NB]  = {-10, -12, -9};
inline int FHScale[RULE_NB] = {18, 15, 14};
inline int FHOppo4[RULE_NB] = {88, 93, 87};
TUNE(FHBias);
TUNE(FHScale);
TUNE(FHOppo4);

/// Fail high reduction margin
template <Rule R>
inline Value failHighMargin(Depth d, int oppo4)
{
    return Value(FHBias[R] + FHScale[R] * int(d) + FHOppo4[R] * bool(oppo4));
}

// Lookup tables used for move count based pruning, initialized at startup
inline const auto FutilityMC = []() {
    std::array<int, MAX_MOVES + 1> MC {0};  // [depth]
    for (size_t i = 1; i < MC.size(); i++)
        MC[i] = 3 + int(std::pow(i, 1.32));
    return MC;
}();

/// Move count based pruning. When we already have a non-losing move,
/// and opponent is not making a four at last step, moves that exceeds
/// futility move count will be directly pruned.
inline int futilityMoveCount(Depth d, bool improving)
{
    return FutilityMC[std::max(int(d), 0)] / (2 - improving);
}

inline float SEScale[RULE_NB] = {0.91f, 0.86f, 0.95f};
TUNE(SEScale);

/// Singular extension margin
template <Rule R>
inline Value singularMargin(Depth d, bool formerPv)
{
    return Value((SEScale[R] + formerPv) * d);
}

inline float SRScale = 0.57f;
TUNE(SRScale);

/// Depth reduction for singular move test search
inline Depth singularReduction(Depth d, bool formerPv)
{
    return d * SRScale - formerPv;
}

inline int   DSEBias[RULE_NB]  = {29, 32, 26};
inline float DSEScale[RULE_NB] = {0.56f, 0.55f, 0.53f};
inline int   DSEMax[RULE_NB]   = {9, 8, 9};
TUNE(DSEBias);
TUNE(DSEScale);
TUNE(DSEMax);

/// Margin for double singular extension
template <Rule R>
inline Value doubleSEMargin(Depth d)
{
    return Value(DSEBias[R] - std::min(int(d * DSEScale[R]), DSEMax[R]));
}

inline int QVCFBias[RULE_NB]  = {2086, 1818, 2163};
inline int QVCFScale[RULE_NB] = {60, 64, 58};
TUNE(QVCFBias);
TUNE(QVCFScale);

/// Delta pruning margin for QVCF search
template <Rule R>
inline Value qvcfDeltaMargin(Depth d)  // note: d <= 0
{
    return Value(std::max(QVCFBias[R] + QVCFScale[R] * int(d), 768));
}

inline int LMRExt1Bias[RULE_NB] = {37, 28, 36};
inline int LMRExt2Bias[RULE_NB] = {349, 311, 290};
TUNE(LMRExt1Bias);
TUNE(LMRExt2Bias);

inline int LMRExt1Diff = 11;
inline int LMRExt2Diff = 29;
TUNE(LMRExt1Diff);
TUNE(LMRExt2Diff);

/// Extension for full-depth search when reduced LMR search fails high
template <Rule R>
inline int
lmrExtension(Depth newDepth, Depth searchedDepth, Value value, Value alpha, Value bestValue)
{
    bool doDeeperSearch =
        value > (alpha + LMRExt1Bias[R] + Value(LMRExt1Diff * (newDepth - searchedDepth)));
    bool doEvenDeeperSearch =
        value > (alpha + LMRExt2Bias[R] + Value(LMRExt2Diff * (newDepth - searchedDepth)));
    bool doShallowerSearch = value < bestValue + Value(newDepth);
    return doDeeperSearch + doEvenDeeperSearch - doShallowerSearch;
}

inline double Factor[RULE_NB] = {1.0, 0.78, 0.75};
TUNE(Factor);

/// Init Reductions table according to num threads.
inline void initReductionLUT(std::array<Depth, MAX_MOVES + 1> (&lut)[RULE_NB], int numThreads = 1)
{
    double threadBias = 0.1 * std::log(numThreads);
    for (int r = 0; r < RULE_NB; r++) {
        lut[r][0] = 0.0f;
        for (size_t i = 1; i < lut[r].size(); i++)
            lut[r][i] = float(Factor[r] * (std::log(i) + threadBias));
    }
}

/// Basic depth reduction in LMR search
template <Rule R, bool PvNode>
inline Depth reduction(const std::array<Depth, MAX_MOVES + 1> (&lut)[RULE_NB],
                       Depth d,
                       int   moveCount,
                       int   improvement,
                       Value delta,
                       Value rootDelta)
{
    assert(d > 0.0f);
    assert(moveCount > 0 && moveCount < lut[R].size());
    Depth r = lut[R][(int)d] * lut[R][moveCount];
    if constexpr (PvNode)
        return std::max(r - Depth(delta) / Depth(rootDelta), 0.0f);
    else
        return r + (improvement <= 0 && r > 1.0f);
}

inline Depth CR1[RULE_NB] = {0.0968f, 0.075f, 0.069f};
inline Depth CR2[RULE_NB] = {0.0385f, 0.024f, 0.029f};
inline Depth CR3[RULE_NB] = {0.0222f, 0.020f, 0.023f};
inline Depth CR4[RULE_NB] = {0.0055f, 0.007f, 0.007f};
TUNE(CR1);
TUNE(CR2);
TUNE(CR3);
TUNE(CR4);

/// Complexity reduction factor based on move type.
template <Rule R>
inline Depth complexityReduction(bool trivialMove, bool importantMove, bool distract)
{
    return (trivialMove ? (distract ? CR1 : CR2) : !importantMove ? CR3 : CR4)[R];
}

inline Depth PolicyReductionScale[RULE_NB] = {3.14f, 2.95f, 3.39f};
inline Depth PolicyReductionBias[RULE_NB]  = {2.21f, 3.81f, 4.40f};
inline Depth PolicyReductionMax[RULE_NB]   = {5.00f, 3.76f, 5.04f};
TUNE(PolicyReductionScale);
TUNE(PolicyReductionBias);
TUNE(PolicyReductionMax);

/// Policy depth reduction based on normalized policy score.
template <Rule R>
inline Depth policyReduction(float normalizedPolicyScore)
{
    Depth r = PolicyReductionBias[R] - PolicyReductionScale[R] * normalizedPolicyScore;
    return std::min(std::max(r, 0.0f), PolicyReductionMax[R]);
}

inline int PPBias[RULE_NB]  = {394, 370, 403};
inline int PPScale[RULE_NB] = {46, 55, 60};
TUNE(PPBias);
TUNE(PPScale);

/// Policy pruning score at given depth. Moves lower than this are pruned at low depth.
template <Rule R>
inline int policyPruningScore(Depth d)
{
    return PPBias[R] - int(d * PPScale[R]);
}

inline int StatScoreBias[RULE_NB] = {3253, 3253, 3253};
inline int SSMainHistoryScale     = 819;
TUNE(StatScoreBias);
TUNE(SSMainHistoryScale);

/// Compute stat score of current move from history table.
template <Rule R>
inline int statScore(const MainHistory &mainHistory, Color stm, Pos move)
{
    return mainHistory[stm][move][HIST_ATTACK]                               // history attack score
           + mainHistory[stm][move][HIST_QUIET] * SSMainHistoryScale / 1024  // history quiet score
           - StatScoreBias[R];
}

inline int   ExtStatScore1 = 12518;
inline int   ExtStatScore2 = 4088;
inline float ExtStatScore3 = 5.55f;
TUNE(ExtStatScore1);
TUNE(ExtStatScore2);
TUNE(ExtStatScore3);

/// Compute depth extension from statScore of current move.
inline Depth extensionFromStatScore(int statScore, Depth depth)
{
    // Use less stat score at higher depths
    return statScore * (1.0f / (ExtStatScore1 + ExtStatScore2 * (depth > ExtStatScore3)));
}

}  // namespace Search::AB
