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

#include "evaluator.h"
#include "simdops.h"

#include <array>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace Evaluation::mix9lite {

using namespace Evaluation;

constexpr uint32_t ArchHashBase    = 0x247e6c70;
constexpr int      ShapeNum        = 442503;
constexpr int      FeatureDim      = 16;
constexpr int      PolicyDim       = 8;
constexpr int      ValueDim        = 16;
constexpr int      PolicyPWConvDim = 16;
constexpr int      NumHeadBucket   = 1;

template <int OutSize, int InSize>
struct StarBlockWeight
{
    int8_t  value_corner_up1_weight[(OutSize * 2) * InSize];
    int32_t value_corner_up1_bias[(OutSize * 2)];
    int8_t  value_corner_up2_weight[(OutSize * 2) * InSize];
    int32_t value_corner_up2_bias[(OutSize * 2)];
    int8_t  value_corner_down_weight[OutSize * OutSize];
    int32_t value_corner_down_bias[OutSize];
};

struct alignas(simd::NativeAlignment) Mix9LiteWeight
{
    // 1  mapping layer
    int16_t mapping[ShapeNum][FeatureDim];

    // 2  Depthwise conv
    int16_t feature_dwconv_weight[9][FeatureDim];
    int16_t feature_dwconv_bias[FeatureDim];

    struct HeadBucket
    {
        // 3  Policy dynamic pointwise conv
        int8_t  policy_pwconv_layer_l1_weight[(PolicyDim * 2) * FeatureDim];
        int32_t policy_pwconv_layer_l1_bias[PolicyDim * 2];
        int8_t  policy_pwconv_layer_l2_weight[(PolicyPWConvDim * PolicyDim + PolicyPWConvDim)
                                             * (PolicyDim * 2)];
        int32_t policy_pwconv_layer_l2_bias[(PolicyPWConvDim * PolicyDim + PolicyPWConvDim)];

        // 4  Value Group MLP (layer 1,2)
        StarBlockWeight<ValueDim, FeatureDim> value_corner;
        StarBlockWeight<ValueDim, FeatureDim> value_edge;
        StarBlockWeight<ValueDim, FeatureDim> value_center;
        StarBlockWeight<ValueDim, ValueDim>   value_quad;

        // 5  Value MLP (layer 1,2,3)
        int8_t  value_l1_weight[ValueDim * (FeatureDim + ValueDim * 4)];
        int32_t value_l1_bias[ValueDim];
        int8_t  value_l2_weight[ValueDim * ValueDim];
        int32_t value_l2_bias[ValueDim];
        int8_t  value_l3_weight[4 * ValueDim];
        int32_t value_l3_bias[4];

        // 6  Policy output linear
        float policy_output_weight[16];
        float policy_output_bias;
        char  __padding_to_64bytes_1[44];
    } buckets[NumHeadBucket];
};

class Mix9LiteAccumulator
{
public:
    struct alignas(simd::NativeAlignment) ValueSumType
    {
        static constexpr int NGroup = 3;

        std::array<int32_t, FeatureDim> global;
        std::array<int32_t, FeatureDim> group[NGroup][NGroup];
    };

    Mix9LiteAccumulator(int boardSize);
    ~Mix9LiteAccumulator();

    /// Init accumulator state to empty board.
    void clear(const Mix9LiteWeight &w);
    /// Incremental update mix6 network state.
    void move(const Mix9LiteWeight &w, Color pieceColor, int x, int y);
    void undo() { currentVersion--; }

    /// Calculate value (win/loss/draw tuple) of current network state.
    std::tuple<float, float, float> evaluateValue(const Mix9LiteWeight &w);
    /// Calculate policy value of current network state.
    void evaluatePolicy(const Mix9LiteWeight &w, PolicyBuffer &policyBuffer);

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
    std::array<int16_t, FeatureDim> *mapConv;  // [N_outer, DWConvDim] (aligned)

    //=============================================================
    int    boardSize;
    int    outerBoardSize;  // (boardSize + 2)
    int    currentVersion;
    int8_t groupIndex[32];

    void initIndexTable();
    int  getBucketIndex() { return 0; }
};

class Mix9LiteEvaluator : public Evaluator
{
public:
    Mix9LiteEvaluator(int                   boardSize,
                      Rule                  rule,
                      std::filesystem::path blackWeightPath,
                      std::filesystem::path whiteWeightPath);
    ~Mix9LiteEvaluator();

    void initEmptyBoard();
    void beforeMove(const Board &board, Pos pos);
    void afterUndo(const Board &board, Pos pos);

    ValueType evaluateValue(const Board &board);
    void      evaluatePolicy(const Board &board, PolicyBuffer &policyBuffer);

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

    Mix9LiteWeight /* non-owning ptr */ *weight[2];
    std::unique_ptr<Mix9LiteAccumulator> accumulator[2];
    std::vector<MoveCache>               moveCache[2];
};

}  // namespace Evaluation::mix9lite
