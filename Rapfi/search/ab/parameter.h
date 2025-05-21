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
#include "history.h"

#include <array>
#include <cmath>

namespace Search::AB {

// -------------------------------------------------
// Search limits

constexpr int MAX_DEPTH = 200;
constexpr int MAX_PLY   = 256;

// -------------------------------------------------
// Depth & Value Constants

// Search Constants
constexpr Value MARGIN_INFINITE  = Value(INT16_MAX);
constexpr Depth ASPIRATION_DEPTH = 5.0f;
constexpr Depth IID_DEPTH        = 14.0f;

// Reductions

constexpr Depth IIR_REDUCTION                = 0.65f;
constexpr Depth IIR_REDUCTION_PV             = 0.30f;
constexpr Depth IIR_REDUCTION_TT             = 0.31f;
constexpr Depth IIR_REDUCTION_TT_MAX         = 3.6f;
constexpr Depth TRIVIAL_PRUN_DEPTH           = 4.5f;
constexpr Depth LMR_EXTRA_MAX_DEPTH          = 5.0f;
constexpr Depth TTPV_NEG_REDUCTION           = 1.0f;
constexpr Depth NO_ALPHA_IMPROVING_REDUCTION = 1.0f;
constexpr Depth NOKILLER_CUTNODE_REDUCTION   = 1.73f;
constexpr Depth FALSE_FORBID_LESS_REDUCTION  = 1.0f;
constexpr Depth ALPHA_IMPROVEMENT_REDUCTION  = 1.0f;

// Extensions

constexpr Depth OPPO5_EXT             = 1.33f;
constexpr Depth SE_DEPTH              = 7.0f;
constexpr Depth SE_TTE_DEPTH          = 1.96f;
constexpr Depth SE_EXTRA_MAX_DEPTH    = 12.0f;
constexpr Depth SE_REDUCTION_FH       = 1.57f;
constexpr Depth TTM_EXT_PV            = 0.24f;
constexpr Depth TTM_EXT_NONPV         = 0.08f;
constexpr Depth NEARB4_EXT_DIST4      = 0.23f;
constexpr Depth NEARB4_EXT_DIST6      = 0.05f;
constexpr Depth CONTINUOUS_ATTACK_EXT = 0.55f;

// -------------------------------------------------
// Dynamic margin & reduction functions/LUTs

/// Aspiration window delta. When prevDelta is zero, returns the initial aspiration
/// window size. Otherwise returns the next expanded window size for the given prevDelta.
/// Window will expand faster for large absolute previous value.
constexpr Value nextAspirationWindowDelta(Value prevValue, Value prevDelta = VALUE_ZERO)
{
    return prevDelta ? prevDelta * (3 + std::abs(prevValue) / 1024) / 4 + 6 : Value(16);
}

/// Razoring depth & margins
template <Rule R>
constexpr Value razorMargin(Depth d)
{
    return d < 2.6f ? Value(std::max(int(0.18f * d * d + 16 * d) + 18, 0)) : MARGIN_INFINITE;
}

/// Static futility pruning depth & margins
template <Rule R>
constexpr Value futilityMargin(Depth d, bool noTTCutNode, bool improving)
{
    constexpr int FutilityScale[RULE_NB] = {63, 61, 78};
    return Value(std::max(int((FutilityScale[R] - 13 * noTTCutNode) * (d - improving)), 0));
}

/// Null move pruning margin
template <Rule R>
constexpr Value nullMoveMargin(Depth d)
{
    constexpr int NMMDepth[RULE_NB] = {10, 11, 8};
    constexpr int NMMBias[RULE_NB]  = {585, 538, 652};
    constexpr int NMMScale[RULE_NB] = {27, 27, 29};
    constexpr int NMMMax[RULE_NB]   = {315, 345, 242};
    return d >= NMMDepth[R] ? Value(NMMBias[R] - std::min(NMMScale[R] * int(d), NMMMax[R]))
                            : MARGIN_INFINITE;
}

/// Null move search depth reduction. The result of a null move will be
/// tested using reduced depth search.
template <Rule R>
constexpr Depth nullMoveReduction(Depth d)
{
    constexpr float NMRBias[RULE_NB]  = {3.67f, 4.07f, 4.30f};
    constexpr float NMRScale[RULE_NB] = {0.135f, 0.134f, 0.174f};
    return NMRBias[R] + NMRScale[R] * d;
}

/// Internal iterative deepening depth reduction.
template <Rule R>
constexpr Depth iidDepthReduction(Depth d)
{
    constexpr Depth IDRBias[RULE_NB]  = {7.6f, 8.2f, 9.8f};
    constexpr Depth IDRScale[RULE_NB] = {0.67f, 0.60f, 0.64f};
    return IDRBias[R] + IDRScale[R] * d;
}

/// Fail high reduction margin
template <Rule R>
constexpr Value failHighMargin(Depth d, int oppo4)
{
    constexpr int FHBias[RULE_NB]  = {-10, -12, -9};
    constexpr int FHScale[RULE_NB] = {18, 15, 14};
    constexpr int FHOppo4[RULE_NB] = {88, 93, 87};
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
constexpr int futilityMoveCount(Depth d, bool improving)
{
    return FutilityMC[std::max(int(d), 0)] / (2 - improving);
}

/// Singular extension margin
template <Rule R>
constexpr Value singularMargin(Depth d, bool formerPv)
{
    constexpr float SEScale[RULE_NB] = {0.91f, 0.86f, 0.95f};
    return Value((SEScale[R] + formerPv) * d);
}

/// Depth reduction for singular move test search
constexpr Depth singularReduction(Depth d, bool formerPv)
{
    return d * 0.57f - formerPv;
}

/// Margin for double singular extension
template <Rule R>
constexpr Value doubleSEMargin(Depth d)
{
    constexpr int   DSEBias[RULE_NB]  = {29, 32, 26};
    constexpr float DSEScale[RULE_NB] = {0.56f, 0.55f, 0.53f};
    constexpr int   DSEMax[RULE_NB]   = {9, 8, 9};
    return Value(DSEBias[R] - std::min(int(d * DSEScale[R]), DSEMax[R]));
}

/// Delta pruning margin for QVCF search
template <Rule R>
constexpr Value qvcfDeltaMargin(Depth d)  // note: d <= 0
{
    constexpr int QVCFBias[RULE_NB]  = {2086, 1818, 2163};
    constexpr int QVCFScale[RULE_NB] = {60, 64, 58};

    return Value(std::max(QVCFBias[R] + QVCFScale[R] * int(d), 768));
}

/// Extension for full-depth search when reduced LMR search fails high
template <Rule R>
constexpr int
lmrExtension(Depth newDepth, Depth searchedDepth, Value value, Value alpha, Value bestValue)
{
    constexpr int LMRExt1Bias[RULE_NB] = {37, 28, 36};
    constexpr int LMRExt2Bias[RULE_NB] = {349, 311, 290};

    bool doDeeperSearch = value > (alpha + LMRExt1Bias[R] + Value(11 * (newDepth - searchedDepth)));
    bool doEvenDeeperSearch =
        value > (alpha + LMRExt2Bias[R] + Value(29 * (newDepth - searchedDepth)));
    bool doShallowerSearch = value < bestValue + Value(newDepth);
    return doDeeperSearch + doEvenDeeperSearch - doShallowerSearch;
}

/// Init Reductions table according to num threads.
inline void initReductionLUT(std::array<Depth, MAX_MOVES + 1> (&lut)[RULE_NB], int numThreads = 1)
{
    constexpr double Factor[RULE_NB] = {1.0, 0.78, 0.75};
    double           threadBias      = 0.1 * std::log(numThreads);
    for (int r = 0; r < RULE_NB; r++) {
        lut[r][0] = 0.0f;
        for (size_t i = 1; i < lut[r].size(); i++)
            lut[r][i] = float(Factor[r] * (std::log(i) + threadBias));
    }
}

/// Basic depth reduction in LMR search
template <Rule R, bool PvNode>
constexpr Depth reduction(const std::array<Depth, MAX_MOVES + 1> (&lut)[RULE_NB],
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

/// Complexity reduction factor based on move type.
template <Rule R>
constexpr Depth complexityReduction(bool trivialMove, bool importantMove, bool distract)
{
    constexpr Depth CR1[RULE_NB] = {0.0968f, 0.075f, 0.069f};
    constexpr Depth CR2[RULE_NB] = {0.0385f, 0.024f, 0.029f};
    constexpr Depth CR3[RULE_NB] = {0.0222f, 0.020f, 0.023f};
    constexpr Depth CR4[RULE_NB] = {0.0055f, 0.007f, 0.007f};
    return (trivialMove ? (distract ? CR1 : CR2) : !importantMove ? CR3 : CR4)[R];
}

/// Policy depth reduction based on normalized policy score.
template <Rule R>
constexpr Depth policyReduction(float normalizedPolicyScore)
{
    constexpr Depth PolicyReductionScale[RULE_NB] = {3.14f, 2.95f, 3.39f};
    constexpr Depth PolicyReductionBias[RULE_NB]  = {2.21f, 3.81f, 4.40f};
    constexpr Depth PolicyReductionMax[RULE_NB]   = {5.00f, 3.76f, 5.04f};

    Depth r = PolicyReductionBias[R] - PolicyReductionScale[R] * normalizedPolicyScore;
    return std::min(std::max(r, 0.0f), PolicyReductionMax[R]);
}

/// Policy pruning score at given depth. Moves lower than this are pruned at low depth.
template <Rule R>
constexpr int policyPruningScore(Depth d)
{
    constexpr int PPBias[RULE_NB]  = {394, 370, 403};
    constexpr int PPScale[RULE_NB] = {46, 55, 60};
    return PPBias[R] - int(d * PPScale[R]);
}

/// Compute stat score of current move from history table.
inline int statScore(const MainHistory &mainHistory, Color stm, Pos move)
{
    return mainHistory[stm][move][HIST_ATTACK]           // history attack score
           + mainHistory[stm][move][HIST_QUIET] * 4 / 5  // history quiet score
           - 3253;
}

/// Compute depth extension from statScore of current move.
constexpr Depth extensionFromStatScore(int statScore, Depth depth)
{
    // Use less stat score at higher depths
    return statScore * (1.0f / (12518 + 4088 * (depth > 5.55f)));
}

}  // namespace Search::AB
