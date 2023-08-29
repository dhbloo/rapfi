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

#include <array>
#include <cmath>

namespace Search::AB {

// -------------------------------------------------
// Search limits

constexpr int MAX_DEPTH = 200;
constexpr int MAX_PLY   = 256;

// -------------------------------------------------
// Depth & Value Constants

constexpr Value MARGIN_INFINITE     = Value(INT16_MAX);
constexpr Depth ASPIRATION_DEPTH    = 5.0f;
constexpr Depth IID_DEPTH           = 14.8f;
constexpr Depth IIR_REDUCTION       = 0.9f;
constexpr Depth IIR_REDUCTION_PV    = 0.35f;
constexpr Depth SE_DEPTH            = 7.8f;
constexpr Depth SE_TTE_DEPTH        = 1.9f;
constexpr Depth TRIVIAL_PRUN_DEPTH  = 3.9f;
constexpr Depth SE_EXTRA_MAX_DEPTH  = 12.0f;
constexpr Depth LMR_EXTRA_MAX_DEPTH = 5.0f;

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
    return d < 2.7f ? Value(std::max(int(0.18f * d * d + 17 * d) + 18, 0)) : MARGIN_INFINITE;
}

/// Static futility pruning depth & margins
template <Rule R>
constexpr Value futilityMargin(Depth d, bool noTTCutNode, bool improving)
{
    constexpr int FutilityScale[RULE_NB] = {63, 69, 73};
    return Value(std::max(int((FutilityScale[R] - 13 * noTTCutNode) * (d - improving)), 0));
}

/// Null move pruning margin
template <Rule R>
constexpr Value nullMoveMargin(Depth d)
{
    constexpr int NMMBias[RULE_NB]  = {585, 484, 573};
    constexpr int NMMScale[RULE_NB] = {27, 26, 29};
    return d >= 10 ? Value(NMMBias[R] - std::min(NMMScale[R] * int(d), 315)) : MARGIN_INFINITE;
}

/// Null move search depth reduction. The result of a null move will be
/// tested using reduced depth search.
template <Rule R>
constexpr Depth nullMoveReduction(Depth d)
{
    constexpr float NMRBias[RULE_NB]  = {3.67f, 3.66f, 4.44f};
    constexpr float NMRScale[RULE_NB] = {0.135f, 0.134f, 0.157f};
    return NMRBias[R] + NMRScale[R] * d;
}

/// Internal iterative deepening depth reduction.
template <Rule R>
constexpr Depth iidDepthReduction(Depth d)
{
    constexpr Depth IDRBias[RULE_NB]  = {7.6f, 7.9f, 9.1f};
    constexpr Depth IDRScale[RULE_NB] = {0.67f, 0.57f, 0.64f};
    return IDRBias[R] + IDRScale[R] * d;
}

/// Fail high reduction margin
template <Rule R>
constexpr Value failHighMargin(Depth d, int oppo4)
{
    constexpr int FHBias[RULE_NB]  = {-10, -12, -8};
    constexpr int FHScale[RULE_NB] = {18, 16, 16};
    return Value(FHBias[R] + FHScale[R] * int(d) + 88 * bool(oppo4));
}

/// Fail low reduction margin
template <Rule R>
constexpr Value failLowMargin(Depth d)
{
    constexpr int FLBias[RULE_NB]  = {89, 82, 104};
    constexpr int FLScale[RULE_NB] = {58, 54, 43};
    return Value(FLBias[R] + int(FLScale[R] * d));
}

// Lookup tables used for move count based pruning, initialized at startup
inline const auto FutilityMC = []() {
    std::array<int, MAX_MOVES + 1> MC {0};  // [depth]
    for (size_t i = 1; i < MC.size(); i++)
        MC[i] = 3 + int(std::pow(i, 1.39));
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
constexpr Value singularMargin(Depth d, bool formerPv)
{
    return Value((0.95f + formerPv) * d);
}

/// Depth reduction for singular move test search
constexpr Depth singularReduction(Depth d, bool formerPv)
{
    return d * 0.53f - formerPv;
}

/// Margin for double singular extension
constexpr Value doubleSEMargin(Depth d)
{
    return Value(33 - std::min(int(d * 0.7f), 8));
}

/// Delta pruning margin for QVCF search
constexpr Value qvcfDeltaMargin(Rule rule, Depth d)  // note: d <= 0
{
    return Value(std::max(2526 + 61 * int(d), 700));
}

/// LMR move count. For non-PV all node, moves that exceeds late move count
/// will be searched will late move reduction even without other condition.
template <Rule R>
constexpr int lateMoveCount(Depth d, bool improving)
{
    constexpr float LMC[RULE_NB][2] = {
        {0.82f, 1.48f},
        {0.91f, 1.06f},
        {0.66f, 1.45f},
    };
    return 1 + 2 * improving + int(LMC[R][improving] * d);
}

/// Extension for full-depth search when reduced LMR search fails high
template <Rule R>
constexpr int
lmrExtension(Depth newDepth, Depth searchedDepth, Value value, Value alpha, Value bestValue)
{
    constexpr int LMRExt1Bias[RULE_NB] = {37, 31, 37};
    constexpr int LMRExt2Bias[RULE_NB] = {349, 344, 258};

    bool doDeeperSearch = value > (alpha + LMRExt1Bias[R] + Value(12 * (newDepth - searchedDepth)));
    bool doEvenDeeperSearch =
        value > (alpha + LMRExt2Bias[R] + Value(32 * (newDepth - searchedDepth)));
    bool doShallowerSearch = value < bestValue + Value(newDepth);
    return doDeeperSearch + doEvenDeeperSearch - doShallowerSearch;
}

/// Init Reductions table according to num threads.
inline void initReductionLUT(std::array<Depth, MAX_MOVES + 1> (&lut)[RULE_NB], int numThreads = 1)
{
    constexpr double Factor[RULE_NB] = {1.0, 0.97, 0.90};
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
    assert(moveCount > 0 && moveCount < lut.size());
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
    constexpr Depth CR1[RULE_NB] = {0.0968f, 0.079f, 0.083f};
    constexpr Depth CR2[RULE_NB] = {0.0385f, 0.026f, 0.032f};
    constexpr Depth CR3[RULE_NB] = {0.0222f, 0.022f, 0.021f};
    constexpr Depth CR4[RULE_NB] = {0.0055f, 0.007f, 0.007f};
    return (trivialMove ? (distract ? CR1 : CR2) : !importantMove ? CR3 : CR4)[R];
}

/// Policy depth reduction based on normalized policy score.
template <Rule R>
constexpr Depth policyReduction(float normalizedPolicyScore)
{
    constexpr Depth PolicyReductionScale[RULE_NB] = {3.14f, 3.63f, 3.40f};
    constexpr Depth PolicyReductionBias[RULE_NB]  = {2.21f, 3.08f, 4.25f};
    constexpr Depth PolicyReductionMax[RULE_NB]   = {5.00f, 3.77f, 4.84f};

    Depth r = PolicyReductionBias[R] - PolicyReductionScale[R] * normalizedPolicyScore;
    return std::min(std::max(r, 0.0f), PolicyReductionMax[R]);
}

/// Policy pruning score at given depth. Moves lower than this are pruned at low depth.
template <Rule R>
inline int policyPruningScore(Depth d)
{
    constexpr int PPBias[RULE_NB]  = {394, 454, 484};
    constexpr int PPScale[RULE_NB] = {46, 51, 53};
    return PPBias[R] - int(d * PPScale[R]);
}

/// Policy reduction score at given depth. Moves lower than this will do lmr.
template <Rule R>
inline int policyReductionScore(Depth d)
{
    constexpr int PRBias[RULE_NB] = {489, 506, 581};
    return PRBias[R];
}

}  // namespace Search::AB
