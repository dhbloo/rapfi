/*
 *  Rapfi, a Gomoku/Renju playing engine supporting piskvork protocol.
 *  Copyright (C) 2024  Rapfi developers
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

#include "evaluator.h"
#include "simdops.h"

#include <array>
#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace Evaluation::mix10 {

using namespace Evaluation;

constexpr uint32_t ArchHashBase  = 0xb53b0258;
constexpr int      ShapeNum      = 442503;
constexpr int      FeatureDim    = 64;
constexpr int      FeatDWConvDim = 32;
constexpr int      ValueDim      = 64;
constexpr int      NumHeadBucket = 1;

static_assert(FeatDWConvDim <= FeatureDim);
static_assert(ValueDim <= FeatureDim);

constexpr int PolicySInDim  = std::max(FeatDWConvDim / 2, 16);
constexpr int PolicySOutDim = std::max(FeatDWConvDim / 4, 16);
constexpr int PolicyLInDim  = std::max(FeatDWConvDim, 16);
constexpr int PolicyLMidDim = std::max(FeatDWConvDim / 2, 16);
constexpr int PolicyLOutDim = std::max(FeatDWConvDim / 4, 16);

template <int OutSize, int InSize>
struct FCWeight
{
    int8_t  weight[OutSize * InSize];
    int32_t bias[OutSize];
};

struct alignas(64) Weight
{
    // 1  mapping layer
    int16_t mapping[2][ShapeNum][FeatureDim];

    // 2  Depthwise conv
    int16_t feature_dwconv_weight[9][FeatDWConvDim];
    int16_t feature_dwconv_bias[FeatDWConvDim];

    struct HeadBucket
    {
        // 3  Small Policy dynamic pointwise conv
        FCWeight<ValueDim, ValueDim>                          policy_small_pwconv_weight_1;
        FCWeight<PolicySOutDim *(PolicySInDim + 1), ValueDim> policy_small_pwconv_weight_2;

        // 4  Large Policy dynamic pointwise conv
        FCWeight<ValueDim, ValueDim>                           policy_large_pwconv_weight_0;
        FCWeight<PolicyLMidDim *(PolicyLInDim + 1), ValueDim>  policy_large_pwconv_weight_1;
        FCWeight<PolicyLOutDim *(PolicyLMidDim + 1), ValueDim> policy_large_pwconv_weight_2;

        // 5  Small Value Head MLP (layer 1,2,3)
        FCWeight<ValueDim, FeatureDim> value_small_l1;
        FCWeight<ValueDim, ValueDim>   value_small_l2;
        FCWeight<4, ValueDim>          value_small_l3;

        // 6  Large Value Gate & Group MLP
        char                               __padding_to_64bytes_0[48];
        FCWeight<FeatureDim * 2, ValueDim> value_gate;
        FCWeight<ValueDim, FeatureDim>     value_corner;
        FCWeight<ValueDim, FeatureDim>     value_edge;
        FCWeight<ValueDim, FeatureDim>     value_center;
        FCWeight<ValueDim, ValueDim>       value_quad;

        // 7  Large Value Head MLP (layer 1,2,3)
        FCWeight<ValueDim, ValueDim * 5> value_l1;
        FCWeight<ValueDim, ValueDim>     value_l2;
        FCWeight<4, ValueDim>            value_l3;

        // 8  Policy output linear
        char  __padding_to_64bytes_1[48];
        float policy_small_output_weight[PolicySOutDim];
        float policy_large_output_weight[PolicyLOutDim];
        float policy_small_output_bias;
        float policy_large_output_bias;

        char __padding_to_64bytes_2[56];
    } buckets[NumHeadBucket];
};

// Make sure we have proper alignment for SIMD operations
static_assert(offsetof(Weight, feature_dwconv_weight) % 64 == 0);
static_assert(offsetof(Weight, buckets) % 64 == 0);
static_assert(offsetof(Weight::HeadBucket, value_gate) % 64 == 0);
static_assert(offsetof(Weight::HeadBucket, policy_small_output_weight) % 64 == 0);
static_assert(offsetof(Weight::HeadBucket, policy_large_output_weight) % 64 == 0);
static_assert(sizeof(Weight::HeadBucket) % 64 == 0);

class Accumulator
{
public:
    struct alignas(64) ValueSumType
    {
        static constexpr int NGroup = 3;

        std::array<int32_t, FeatureDim> global;
        std::array<int32_t, FeatureDim> group[NGroup][NGroup];
        std::array<int8_t, ValueDim>    small_value_feature;
        std::array<int8_t, ValueDim>    large_value_feature;
        bool                            small_value_feature_valid;
        bool                            large_value_feature_valid;
    };

    // Make sure we have proper alignment for SIMD operations
    static_assert(offsetof(ValueSumType, global) % 64 == 0);
    static_assert(offsetof(ValueSumType, group) % 64 == 0);
    static_assert(offsetof(ValueSumType, small_value_feature) % 64 == 0);
    static_assert(offsetof(ValueSumType, large_value_feature) % 64 == 0);

    Accumulator(int boardSize);
    ~Accumulator();

    /// Init accumulator state to empty board.
    void clear(const Weight &w);
    /// Incremental update mix6 network state.
    void move(const Weight &w, Color pieceColor, int x, int y);
    void undo() { currentVersion--; }

    void updateSharedSmallHead(const Weight &w);
    void updateSharedLargeHead(const Weight &w);

    /// Calculate value (win/loss/draw) and (relative) uncertainty of current network state.
    std::tuple<float, float, float, float> evaluateValueSmall(const Weight &w);
    /// Calculate value (win/loss/draw) and (relative) uncertainty of current network state.
    std::tuple<float, float, float, float> evaluateValueLarge(const Weight &w);
    /// Calculate policy value of current network state.
    void evaluatePolicySmall(const Weight &w, PolicyBuffer &policyBuffer);
    /// Calculate policy value of current network state.
    void evaluatePolicyLarge(const Weight &w, PolicyBuffer &policyBuffer);

private:
    friend class Mix8Evaluator;
    struct ChangeNum
    {
        uint16_t inner, outer;
    };
    //=============================================================
    // Network states

    /// Value feature sum of the full board
    ValueSumType *valueSumTable;          // [H*W+1, FeatureDim] (aligned)
    ChangeNum    *versionChangeNumTable;  // [H*W+1] (unaligned) num inner changes and outer changes
    uint16_t     *versionInnerIndexTable;  // [H*W+1, H*W] (unaligned)
    uint16_t     *versionOuterIndexTable;  // [H*W+1, (H+2)*(W+2)] (unaligned)
    /// Index table to convert line shape to map feature
    std::array<uint32_t, 4> *indexTable;  // [N_inner, 4] (unaligned)
    /// Sumed map feature of four directions
    std::array<int16_t, FeatureDim> *mapSum;  // [N_inner, FeatureDim] (aligned)
    /// Map feature after depth wise conv
    std::array<int16_t, FeatDWConvDim> *mapConv;  // [N_outer, DWConvDim] (aligned)

    //=============================================================
    int    boardSize;
    int    outerBoardSize;  // (boardSize + 2)
    int    currentVersion;
    int8_t groupIndex[32];

    void initIndexTable();
    int  getBucketIndex() { return 0; }
};

class Evaluator : public Evaluation::Evaluator
{
public:
    Evaluator(int                   boardSize,
              Rule                  rule,
              Numa::NumaNodeId      numaNodeId,
              std::filesystem::path blackWeightPath,
              std::filesystem::path whiteWeightPath);
    ~Evaluator();

    void initEmptyBoard();
    void beforeMove(const Board &board, Pos pos);
    void afterUndo(const Board &board, Pos pos);

    ValueType evaluateValue(const Board &board, AccLevel level);
    void      evaluatePolicy(const Board &board, PolicyBuffer &policyBuffer, AccLevel level);

private:
    struct MoveCache
    {
        Color  oldColor, newColor;
        int8_t x, y;

        friend bool isContraryMove(MoveCache a, MoveCache b)
        {
            bool isSameCoord = a.x == b.x && a.y == b.y;
            bool isContrary  = a.oldColor == b.newColor && a.newColor == b.oldColor;
            return isSameCoord && isContrary;
        }
    };

    /// Clear all caches to sync accumulator state with current board state.
    void clearCache(Color side);
    /// Record new board action, but not update accumulator instantly.
    void addCache(Color side, int x, int y, bool isUndo);

    Weight /* non-owning ptr */ *weight[2];
    std::unique_ptr<Accumulator> accumulator[2];
    std::vector<MoveCache>       moveCache[2];
};

}  // namespace Evaluation::mix10
