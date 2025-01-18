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
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace Evaluation::mix10 {

using namespace Evaluation;

constexpr uint32_t ArchHashBase    = 0xb53b0258;
constexpr int      ShapeNum        = 442503;
constexpr int      FeatureDim      = 128;
constexpr int      PolicyDim       = 64;
constexpr int      ValueDim        = 64;
constexpr int      FeatDWConvDim   = 64;
constexpr int      PolicyPWConvDim = 16;
constexpr int      NumHeadBucket   = 1;

template <int OutSize, int InSize>
struct LinearWeight
{
    int8_t  weight[OutSize * InSize];
    int32_t bias[OutSize];
};

struct alignas(simd::NativeAlignment) Mix10Weight
{
    // 1  mapping layer
    int16_t mapping[2][ShapeNum][FeatureDim];

    // 2  Depthwise conv
    int16_t feature_dwconv_weight[9][FeatDWConvDim];
    int16_t feature_dwconv_bias[FeatDWConvDim];

    struct HeadBucket
    {
        // 3  Policy dynamic pointwise conv
        LinearWeight<PolicyDim * 2, FeatureDim> policy_pwconv_layer_l1;
        LinearWeight<PolicyPWConvDim * PolicyDim + PolicyPWConvDim, PolicyDim * 2>
            policy_pwconv_layer_l2;

        // 4  Small Value Head MLP (layer 1,2,3)
        LinearWeight<FeatureDim, FeatureDim> value_small_l1;
        LinearWeight<FeatureDim, FeatureDim> value_small_l2;
        LinearWeight<4, FeatureDim>          value_small_l3;

        char __padding_to_64bytes_0[48];

        // 5  Large Value Gate & Group MLP
        LinearWeight<FeatureDim, FeatureDim> value_gate;
        LinearWeight<ValueDim, ValueDim>     value_corner;
        LinearWeight<ValueDim, ValueDim>     value_edge;
        LinearWeight<ValueDim, ValueDim>     value_center;
        LinearWeight<ValueDim, ValueDim>     value_quad;

        // 6  Large Value Head MLP (layer 1,2,3)
        LinearWeight<ValueDim, FeatureDim + ValueDim * 4> value_l1;
        LinearWeight<ValueDim, ValueDim>                  value_l2;
        LinearWeight<4, ValueDim>                         value_l3;

        // 7  Policy output linear
        float policy_output_weight[16];
        float policy_output_bias;

        char __padding_to_64bytes_1[44];
    } buckets[NumHeadBucket];
};

class Mix10Accumulator
{
public:
    struct alignas(simd::NativeAlignment) ValueSumType
    {
        static constexpr int NGroup = 3;

        std::array<int32_t, FeatureDim> global;
        std::array<int32_t, FeatureDim> group[NGroup][NGroup];

        std::array<int8_t, FeatureDim> small_value_feature;
        bool                           small_value_feature_valid;
    };

    Mix10Accumulator(int boardSize);
    ~Mix10Accumulator();

    /// Init accumulator state to empty board.
    void clear(const Mix10Weight &w);
    /// Incremental update mix6 network state.
    void move(const Mix10Weight &w, Color pieceColor, int x, int y);
    void undo() { currentVersion--; }

    void updateSharedSmallHead(const Mix10Weight &w);

    /// Calculate value (win/loss/draw) and (relative) uncertainty of current network state.
    std::tuple<float, float, float, float> evaluateValueSmall(const Mix10Weight &w);
    /// Calculate value (win/loss/draw) and (relative) uncertainty of current network state.
    std::tuple<float, float, float, float> evaluateValueLarge(const Mix10Weight &w);
    /// Calculate policy value of current network state.
    void evaluatePolicy(const Mix10Weight &w, PolicyBuffer &policyBuffer);

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

class Mix10Evaluator : public Evaluator
{
public:
    Mix10Evaluator(int                   boardSize,
                   Rule                  rule,
                   std::filesystem::path blackWeightPath,
                   std::filesystem::path whiteWeightPath);
    ~Mix10Evaluator();

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

    Mix10Weight /* non-owning ptr */ *weight[2];
    std::unique_ptr<Mix10Accumulator> accumulator[2];
    std::vector<MoveCache>            moveCache[2];
};

}  // namespace Evaluation::mix10
