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
#include <cstddef>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace Evaluation::mix9svq {

using namespace Evaluation;

constexpr uint32_t ArchHashBase    = 0x84a071fe;
constexpr int      ShapeNum        = 442503;
constexpr int      FeatureDim      = 64;
constexpr int      PolicyDim       = 32;
constexpr int      ValueDim        = 64;
constexpr int      FeatDWConvDim   = 32;
constexpr int      PolicyPWConvDim = 16;
constexpr int      NumHeadBucket   = 1;

template <int OutSize, int InSize>
struct FCWeight
{
    int8_t  weight[OutSize * InSize];
    int32_t bias[OutSize];
};

template <int OutSize, int InSize>
struct StarBlockWeight
{
    FCWeight<OutSize * 2, InSize> value_corner_up1;
    FCWeight<OutSize * 2, InSize> value_corner_up2;
    FCWeight<OutSize, OutSize>    value_corner_down;
};

struct alignas(64) Weight
{
    // 1  mapping layer
    int16_t  codebook[2][65536][FeatureDim];
    uint16_t mapping_index[2][ShapeNum];
    char     __padding_to_64bytes_0[36];

    // 2  Depthwise conv
    int16_t feature_dwconv_weight[9][FeatDWConvDim];
    int16_t feature_dwconv_bias[FeatDWConvDim];

    struct HeadBucket
    {
        // 3  Policy dynamic pointwise conv
        FCWeight<PolicyDim * 2, FeatureDim> policy_pwconv_layer_l1;
        FCWeight<PolicyPWConvDim * PolicyDim + PolicyPWConvDim, PolicyDim * 2>
            policy_pwconv_layer_l2;

        // 4  Value Group MLP (layer 1,2)
        StarBlockWeight<ValueDim, FeatureDim> value_corner;
        StarBlockWeight<ValueDim, FeatureDim> value_edge;
        StarBlockWeight<ValueDim, FeatureDim> value_center;
        StarBlockWeight<ValueDim, ValueDim>   value_quad;

        // 5  Value MLP (layer 1,2,3)
        FCWeight<ValueDim, FeatureDim + ValueDim * 4> value_l1;
        FCWeight<ValueDim, ValueDim>                  value_l2;
        FCWeight<4, ValueDim>                         value_l3;

        // 6  Policy output linear
        float policy_output_weight[16];
        float policy_output_bias;
        char  __padding_to_64bytes_1[44];
    } buckets[NumHeadBucket];
};

// Make sure we have proper alignment for SIMD operations
static_assert(offsetof(Weight, feature_dwconv_weight) % 64 == 0);
static_assert(offsetof(Weight, buckets) % 64 == 0);
static_assert(offsetof(Weight::HeadBucket, value_corner) % 64 == 0);
static_assert(offsetof(Weight::HeadBucket, value_l1) % 64 == 0);
static_assert(offsetof(Weight::HeadBucket, policy_output_weight) % 16 == 0);
static_assert(sizeof(Weight::HeadBucket) % 64 == 0);

class Accumulator
{
public:
    struct alignas(64) ValueSumType
    {
        static constexpr int NGroup = 3;

        std::array<int32_t, FeatureDim> global;
        std::array<int32_t, FeatureDim> group[NGroup][NGroup];
    };

    // Make sure we have proper alignment for SIMD operations
    static_assert(offsetof(ValueSumType, global) % 64 == 0);
    static_assert(offsetof(ValueSumType, group) % 64 == 0);

    Accumulator(int boardSize);
    ~Accumulator();

    /// Init accumulator state to empty board.
    void clear(const Weight &w);
    /// Incremental update mix6 network state.
    void move(const Weight &w, Color pieceColor, int x, int y);
    void undo() { currentVersion--; }

    /// Calculate value (win/loss/draw tuple) of current network state.
    std::tuple<float, float, float> evaluateValue(const Weight &w);
    /// Calculate policy value of current network state.
    void evaluatePolicy(const Weight &w, PolicyBuffer &policyBuffer);

private:
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

}  // namespace Evaluation::mix9svq
