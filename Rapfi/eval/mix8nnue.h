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

namespace Evaluation::mix8 {

using namespace Evaluation;

constexpr uint32_t ArchHashBase     = 0x9cc20b15;
constexpr size_t   Alignment        = 64;
constexpr int      ShapeNum         = 708588;
constexpr int      FeatureDim       = 64;
constexpr int      PolicyDim        = 32;
constexpr int      ValueDim         = 64;
constexpr int      ValueGroupDim    = 32;
constexpr int      FeatureDWConvDim = 32;
constexpr int      MaxNumBuckets    = 1;

struct alignas(Alignment) Mix8Weight
{
    // 1  mapping layer
    int16_t mapping[ShapeNum][FeatureDim];

    // 2  PReLU after mapping
    int16_t map_prelu_weight[FeatureDim];

    // 3  Depthwise conv
    int16_t feature_dwconv_weight[9][FeatureDWConvDim];
    int16_t feature_dwconv_bias[FeatureDWConvDim];

    // 4  Value sum scale
    float value_sum_scale_after_conv;
    float value_sum_scale_direct;

    int32_t num_head_buckets;  // used to validate the number of head buckets
    char    __padding_to_64bytes_0[52];

    struct HeadBucket
    {
        // 5  Policy dynamic pointwise conv
        float policy_pwconv_layer_l1_weight[FeatureDim][PolicyDim];
        float policy_pwconv_layer_l1_bias[PolicyDim];
        float policy_pwconv_layer_l1_prelu[PolicyDim];
        float policy_pwconv_layer_l2_weight[PolicyDim][4 * PolicyDim];
        float policy_pwconv_layer_l2_bias[4 * PolicyDim];

        // 6  Value Group MLP (layer 1,2)
        float value_corner_weight[FeatureDim][ValueGroupDim];
        float value_corner_bias[ValueGroupDim];
        float value_corner_prelu[ValueGroupDim];
        float value_edge_weight[FeatureDim][ValueGroupDim];
        float value_edge_bias[ValueGroupDim];
        float value_edge_prelu[ValueGroupDim];
        float value_center_weight[FeatureDim][ValueGroupDim];
        float value_center_bias[ValueGroupDim];
        float value_center_prelu[ValueGroupDim];
        float value_quad_weight[ValueGroupDim][ValueGroupDim];
        float value_quad_bias[ValueGroupDim];
        float value_quad_prelu[ValueGroupDim];

        // 7  Value MLP (layer 1,2,3)
        float value_l1_weight[FeatureDim + ValueGroupDim * 4][ValueDim];
        float value_l1_bias[ValueDim];
        float value_l2_weight[ValueDim][ValueDim];
        float value_l2_bias[ValueDim];
        float value_l3_weight[ValueDim][3];
        float value_l3_bias[3];

        // 8  Policy PReLU
        float policy_output_pos_weight[4];
        float policy_output_neg_weight[4];
        float policy_output_bias;
        char  __padding_to_64bytes_1[16];
    } buckets[MaxNumBuckets];
};

class Mix8Accumulator
{
public:
    struct alignas(Alignment) ValueSumType
    {
        static constexpr int NGroup = 3;

        std::array<int32_t, FeatureDim> global;
        std::array<int32_t, FeatureDim> group[NGroup][NGroup];
    };

    Mix8Accumulator(int boardSize);
    ~Mix8Accumulator();

    /// Init accumulator state to empty board.
    void clear(const Mix8Weight &w);
    /// Incremental update mix6 network state.
    void move(const Mix8Weight &w, Color pieceColor, int x, int y);
    void undo() { currentVersion--; }

    /// Calculate value (win/loss/draw tuple) of current network state.
    std::tuple<float, float, float> evaluateValue(const Mix8Weight &w);
    /// Calculate policy value of current network state.
    void evaluatePolicy(const Mix8Weight &w, PolicyBuffer &policyBuffer);

private:
    friend class Mix8Evaluator;
    struct ChangeNum
    {
        uint16_t inner, outer;
    };
    //=============================================================
    // Mix8 network states

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
    std::array<int16_t, FeatureDWConvDim> *mapConv;  // [N_outer, DWConvDim] (aligned)

    //=============================================================
    int    boardSize;
    int    outerBoardSize;  // (boardSize + 2)
    int    currentVersion;
    float  boardSizeScale;
    float  groupSizeScale[ValueSumType::NGroup][ValueSumType::NGroup];
    int8_t groupIndex[32];

    void initIndexTable();
    int  getBucketIndex() { return 0; }
};

class Mix8Evaluator : public Evaluator
{
public:
    Mix8Evaluator(int                   boardSize,
                  Rule                  rule,
                  std::filesystem::path blackWeightPath,
                  std::filesystem::path whiteWeightPath);
    ~Mix8Evaluator();

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

    Mix8Weight /* non-owning ptr */ *weight[2];
    std::unique_ptr<Mix8Accumulator> accumulator[2];
    std::vector<MoveCache>           moveCache[2];
};

}  // namespace Evaluation::mix8
