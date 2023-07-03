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

constexpr uint32_t ArchHashBase     = 0x2171bc0b;
constexpr size_t   Alignment        = 64;
constexpr int      ShapeNum         = 708588;
constexpr int      FeatureDim       = 64;
constexpr int      PolicyDim        = 32;
constexpr int      ValueDim         = 64;
constexpr int      ValueGroupDim    = 16;
constexpr int      FeatureDWConvDim = 32;
constexpr int      KernelMultiplier = 2;
constexpr int      ValueSumDim      = FeatureDim + FeatureDWConvDim * (KernelMultiplier - 1);
constexpr int      NumBuckets       = 1;

struct alignas(Alignment) Mix8Weight
{
    // 1  mapping layer
    int16_t mapping[ShapeNum][FeatureDim];

    // 2  PReLU after mapping
    int16_t map_prelu_weight[FeatureDim];

    // 3  Depthwise conv
    int16_t feature_dwconv_weight[9][KernelMultiplier * FeatureDWConvDim];
    int16_t feature_dwconv_bias[KernelMultiplier * FeatureDWConvDim];

    // 4  Value sum scale
    float value_sum_scale_after_conv;
    float value_sum_scale_direct;

    int32_t numHeadBuckets;  // used to validate the number of head buckets
    char    __padding_to_64bytes_0[52];

    struct HeadBucket
    {
        // 5  Policy depthwise conv
        int16_t policy_dwconv_weight[41][PolicyDim];
        int16_t policy_dwconv_bias[PolicyDim];

        // 6  Policy dynamic pointwise conv
        float policy_pwconv_layer_l1_weight[ValueSumDim][PolicyDim];
        float policy_pwconv_layer_l1_bias[PolicyDim];
        float policy_pwconv_layer_l2_weight[PolicyDim][PolicyDim];

        // 7  Value Group MLP (layer 1,2)
        float value_corner_weight[ValueSumDim][ValueGroupDim];
        float value_corner_bias[ValueGroupDim];
        float value_edge_weight[ValueSumDim][ValueGroupDim];
        float value_edge_bias[ValueGroupDim];
        float value_center_weight[ValueSumDim][ValueGroupDim];
        float value_center_bias[ValueGroupDim];
        float value_quadrant_weight[ValueGroupDim][ValueGroupDim];
        float value_quadrant_bias[ValueGroupDim];

        // 7  Value MLP (layer 1,2,3,4)
        float value_l1_weight[ValueSumDim + ValueGroupDim * 4][ValueDim];
        float value_l1_bias[ValueDim];
        float value_l2_weight[ValueDim][ValueDim];
        float value_l2_bias[ValueDim];
        float value_l3_weight[ValueDim][ValueDim];
        float value_l3_bias[ValueDim];
        float value_l4_weight[ValueDim][3];
        float value_l4_bias[3];

        // 8  Policy PReLU
        float policy_neg_weight;
        float policy_pos_weight;
        char  __padding_to_64bytes_1[44];
    } buckets[NumBuckets];
};

class alignas(Alignment) Mix8Accumulator
{
public:
    struct ValueSumType
    {
        static constexpr int NGroup = 3;

        std::array<int32_t, ValueSumDim> global;
        std::array<int32_t, ValueSumDim> group[NGroup][NGroup];
    };

    Mix8Accumulator(int boardSize);
    ~Mix8Accumulator();

    /// Init accumulator state to empty board.
    void clear(const Mix8Weight &w);
    /// Incremental update mix6 network state.
    enum UpdateType { MOVE, UNDO };
    template <UpdateType UT>
    void update(const Mix8Weight &w, Color pieceColor, int x, int y, ValueSumType *valueSumBackup);

    /// Calculate value (win/loss/draw tuple) of current network state.
    std::tuple<float, float, float> evaluateValue(const Mix8Weight &w);
    /// Calculate policy value of current network state.
    void evaluatePolicy(const Mix8Weight &w, PolicyBuffer &policyBuffer);

private:
    friend class Mix8Evaluator;
    //=============================================================
    // Mix8 network states

    /// Value feature sum of the full board
    ValueSumType valueSum;  // [ValueSumDim] (aligned to 64)
    /// Index table to convert line shape to map feature
    std::array<uint32_t, 4> *indexTable;  // [H*W, 4] (unaligned)
    /// Sumed map feature of four directions
    std::array<int16_t, FeatureDim> *mapSum;  // [H*W, FeatureDim] (aligned)
    /// Map feature after depth wise conv
    std::array<int16_t, KernelMultiplier * FeatureDWConvDim>
        *mapAfterDWConv;  // [(H+2)*(W+2), KernelMultiplier*DWConvDim] (aligned)

    //=============================================================
    int    boardSize;
    int    fullBoardSize;  // (boardSize + 2)
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

    Mix8Weight /* non-owning ptr */           *weight[2];
    std::unique_ptr<Mix8Accumulator>           accumulator[2];
    std::vector<MoveCache>                     moveCache[2];
    std::vector<Mix8Accumulator::ValueSumType> valueSumBoardHistory[2];
};

}  // namespace Evaluation::mix8
