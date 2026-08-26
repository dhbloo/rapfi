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
constexpr Depth ASPIRATION_DEPTH = 5.3f;
constexpr Depth IID_DEPTH        = 14.5f;

// Reductions

inline constexpr Depth IIR_REDUCTION[RULE_NB]         = {0.71f, 0.65f, 0.65f};
inline constexpr Depth IIR_REDUCTION_PV[RULE_NB]      = {0.45f, 0.43f, 0.37f};
inline constexpr Depth IIR_REDUCTION_TT[RULE_NB]      = {0.19f, 0.20f, 0.17f};
inline constexpr Depth IIR_REDUCTION_TT_MAX[RULE_NB]  = {3.6f, 3.8f, 3.3f};
constexpr Depth        TRIVIAL_PRUN_DEPTH             = 4.4f;
constexpr Depth        LMR_EXTRA_MAX_DEPTH            = 5.1f;
inline constexpr Depth LMRTtPvSubtract[RULE_NB]       = {0.98f, 0.81f, 0.71f};
inline constexpr Depth LMRNoAlphaAdd[RULE_NB]         = {0.91f, 0.96f, 0.99f};
inline constexpr Depth LMRNoKillerCutNodeAdd[RULE_NB] = {1.8f, 2.2f, 1.8f};
constexpr Depth        FALSE_FORBID_LESS_REDUCTION    = 1.03f;
constexpr Depth        ALPHA_IMPROVEMENT_REDUCTION    = 1.05f;
constexpr Depth        OPPO_USELESS_DEFEND_REDUCTION  = 2.07f;
constexpr Depth        SELF_USELESS_DEFEND_REDUCTION  = 0.93f;

// Extensions

constexpr Depth        OPPO5_EXT              = 1.1f;
constexpr Depth        SE_DEPTH               = 6.85f;
inline constexpr Depth SE_TTE_DEPTH[RULE_NB]  = {2.09f, 2.03f, 1.95f};
constexpr Depth        SE_EXTRA_MAX_DEPTH     = 10.8f;
constexpr Depth        SE_REDUCTION_FH        = 1.48f;
inline constexpr Depth TTM_EXT_PV[RULE_NB]    = {0.16f, 0.24f, 0.14f};
inline constexpr Depth TTM_EXT_NONPV[RULE_NB] = {0.08f, 0.12f, 0.10f};
constexpr Depth        NEARB4_EXT_DIST4       = 0.27f;
constexpr Depth        NEARB4_EXT_DIST6       = 0.05f;
constexpr Depth        CONTINUOUS_ATTACK_EXT  = 0.41f;

// -------------------------------------------------
// Dynamic margin & reduction functions/LUTs

inline constexpr int AspirationInitialDelta[RULE_NB] = {20, 10, 11};
inline constexpr int AspirationAddDelta[RULE_NB]     = {8, 3, 8};

inline constexpr float RazorQuadratic[RULE_NB] = {0.17f, 0.20f, 0.17f};
inline constexpr int   RazorLinear[RULE_NB]    = {15, -1, 8};
inline constexpr int   RazorBias[RULE_NB]      = {11, -1, 9};

inline constexpr int FutilityScale[RULE_NB]       = {68, 43, 63};
inline constexpr int FutilityNoTTPenalty[RULE_NB] = {16, 13, 13};

inline constexpr int NMMDepth[RULE_NB]    = {6, 15, 11};
inline constexpr int NMMSlope[RULE_NB]    = {24, 32, 24};
inline constexpr int NMMFloor[RULE_NB]    = {277, 197, 399};
inline constexpr int NMMHeadroom[RULE_NB] = {38, 38, 6};

inline constexpr float NMRBias[RULE_NB]  = {3.70f, 4.37f, 4.21f};
inline constexpr float NMRScale[RULE_NB] = {0.141f, 0.156f, 0.160f};

inline constexpr int FailHighDepthScale[RULE_NB] = {20, 12, 16};
inline constexpr int FailHighOppo4Bonus[RULE_NB] = {80, 91, 80};

inline constexpr float SEScale[RULE_NB] = {0.85f, 0.85f, 0.90f};

inline constexpr int LMRExt1Bias[RULE_NB] = {22, 12, 20};
inline constexpr int LMRExtGap[RULE_NB]   = {321, 278, 253};

/// Aspiration window delta. When prevDelta is zero, returns the initial aspiration
/// window size. Otherwise returns the next expanded window size for the given prevDelta.
/// Window will expand faster for large absolute previous value.
constexpr Value nextAspirationWindowDelta(Rule rule, Value prevValue, Value prevDelta = VALUE_ZERO)
{
    return prevDelta ? prevDelta * (3 + std::abs(prevValue) / 1024) / 4 + AspirationAddDelta[rule]
                     : Value(AspirationInitialDelta[rule]);
}

/// Razoring depth & margins
template <Rule R>
constexpr Value razorMargin(Depth d)
{
    return d < 2.6f
               ? Value(std::max(int(RazorQuadratic[R] * d * d + RazorLinear[R] * d) + RazorBias[R],
                                0))
               : MARGIN_INFINITE;
}

/// Static futility pruning depth & margins
template <Rule R>
constexpr Value futilityMargin(Depth d, bool noTTCutNode, bool improving)
{
    return Value(
        std::max(int((FutilityScale[R] - FutilityNoTTPenalty[R] * noTTCutNode) * (d - improving)),
                 0));
}

/// Null move pruning margin
template <Rule R>
constexpr Value nullMoveMargin(Depth d)
{
    constexpr int nmmMax  = NMMSlope[R] * NMMDepth[R] + NMMHeadroom[R];
    constexpr int nmmBias = nmmMax + NMMFloor[R];
    return d >= NMMDepth[R] ? Value(nmmBias - std::min(NMMSlope[R] * int(d), nmmMax))
                            : MARGIN_INFINITE;
}

/// Null move search depth reduction. The result of a null move will be
/// tested using reduced depth search.
template <Rule R>
constexpr Depth nullMoveReduction(Depth d)
{
    return NMRBias[R] + NMRScale[R] * d;
}

/// Internal iterative deepening depth reduction.
template <Rule R>
constexpr Depth iidDepthReduction(Depth d)
{
    constexpr Depth IDRBias[RULE_NB]  = {8.4f, 8.2f, 9.7f};
    constexpr Depth IDRScale[RULE_NB] = {0.74f, 0.56f, 0.52f};
    return IDRBias[R] + IDRScale[R] * d;
}

/// Fail high reduction margin
template <Rule R>
constexpr Value failHighMargin(Depth d, int oppo4)
{
    constexpr int FailHighBias[RULE_NB] = {-3, -11, -4};
    return Value(FailHighBias[R] + FailHighDepthScale[R] * int(d)
                 + FailHighOppo4Bonus[R] * bool(oppo4));
}

// Lookup tables used for move count based pruning, initialized at startup
inline const auto FutilityMC = []() {
    std::array<int, MAX_MOVES + 1> MC {0};  // [depth]
    for (size_t i = 1; i < MC.size(); i++)
        MC[i] = 3 + int(std::pow(i, 1.30));
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
    return Value((SEScale[R] + formerPv) * d);
}

/// Depth reduction for singular move test search
constexpr Depth singularReduction(Depth d, bool formerPv)
{
    return d * 0.55f - formerPv;
}

/// Margin for double singular extension
template <Rule R>
constexpr Value doubleSEMargin(Depth d)
{
    constexpr float DoubleSEScale[RULE_NB] = {0.71f, 0.60f, 0.58f};
    constexpr int   DoubleSEMax[RULE_NB]   = {4, 6, 8};
    return Value(29 - std::min(int(d * DoubleSEScale[R]), DoubleSEMax[R]));
}

/// Delta pruning margin for QVCF search
template <Rule R>
constexpr Value qvcfDeltaMargin(Depth d)  // note: d <= 0
{
    constexpr int QVCFBias[RULE_NB]  = {2084, 1817, 2154};
    constexpr int QVCFScale[RULE_NB] = {62, 66, 53};
    return Value(std::max(QVCFBias[R] + QVCFScale[R] * int(d), 768));
}

/// Extension for full-depth search when reduced LMR search fails high
template <Rule R>
constexpr int
lmrExtension(Depth newDepth, Depth searchedDepth, Value value, Value alpha, Value bestValue)
{
    constexpr int LMRExt1Diff[RULE_NB] = {10, 3, 10};
    bool          doDeeperSearch =
        value > (alpha + LMRExt1Bias[R] + Value(LMRExt1Diff[R] * (newDepth - searchedDepth)));
    bool doEvenDeeperSearch =
        value > (alpha + LMRExt1Bias[R] + LMRExtGap[R] + Value(27 * (newDepth - searchedDepth)));
    bool doShallowerSearch = value < bestValue + Value(newDepth);
    return doDeeperSearch + doEvenDeeperSearch - doShallowerSearch;
}

/// Init Reductions table according to num threads.
inline void initReductionLUT(std::array<Depth, MAX_MOVES + 1> (&lut)[RULE_NB], int numThreads = 1)
{
    constexpr double Factor[RULE_NB] = {0.95, 0.78, 0.83};
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
    constexpr Depth CR1[RULE_NB] = {0.0920f, 0.0697f, 0.0600f};
    constexpr Depth CR2[RULE_NB] = {0.0333f, 0.0199f, 0.0280f};
    constexpr Depth CR3[RULE_NB] = {0.0250f, 0.0206f, 0.0229f};
    constexpr Depth CR4[RULE_NB] = {0.0053f, 0.0070f, 0.0077f};
    return (trivialMove ? (distract ? CR1 : CR2) : !importantMove ? CR3 : CR4)[R];
}

/// Policy depth reduction based on normalized policy score.
template <Rule R>
constexpr Depth policyReduction(float normalizedPolicyScore)
{
    constexpr Depth PolicyReductionScale[RULE_NB] = {3.60f, 1.76f, 3.15f};
    constexpr Depth PolicyReductionBias[RULE_NB]  = {2.53f, 2.64f, 3.39f};
    constexpr Depth PolicyReductionMax[RULE_NB]   = {5.18f, 3.07f, 4.90f};

    Depth r = PolicyReductionBias[R] - PolicyReductionScale[R] * normalizedPolicyScore;
    return std::min(std::max(r, 0.0f), PolicyReductionMax[R]);
}

/// Policy pruning score at given depth. Moves lower than this are pruned at low depth.
template <Rule R>
constexpr int policyPruningScore(Depth d)
{
    constexpr int PPBias[RULE_NB]  = {372, 363, 377};
    constexpr int PPScale[RULE_NB] = {48, 63, 62};
    return PPBias[R] - int(d * PPScale[R]);
}

/// Compute stat score of current move from history table.
inline int statScore(const MainHistory &mainHistory, Color stm, Pos move)
{
    return mainHistory[stm][move][HIST_ATTACK]                // history attack score
           + mainHistory[stm][move][HIST_QUIET] * 778 / 1024  // history quiet score
           - 3323;
}

/// Compute depth extension from statScore of current move.
template <Rule R>
constexpr Depth extensionFromStatScore(int statScore, Depth depth)
{
    constexpr Depth ExtStatDepth[RULE_NB] = {5.3f, 5.4f, 5.4f};
    // Use less stat score at higher depths
    return statScore * (1.0f / (12266 + 4380 * (depth > ExtStatDepth[R])));
}

}  // namespace Search::AB
