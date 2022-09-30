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

#include <array>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace Evaluation::mix6 {

using namespace Evaluation;

const uint32_t ArchHash        = 0x78f3c05b;
const uint32_t ArchHashTwoSide = 0xd9d61924;
constexpr int  ShapeNum        = 708588;
constexpr int  ValueDim        = 32;
constexpr int  PolicyDim       = 16;
constexpr int  FeatureDim      = ValueDim + PolicyDim;
constexpr int  BatchSize       = 16;
constexpr int  BatchSize32     = 8;
constexpr int  ValueBatches    = ValueDim / BatchSize;
constexpr int  ValueBatches32  = ValueDim / BatchSize32;
constexpr int  PolicyBatches   = PolicyDim / BatchSize;

struct RawValue
{
    float win, loss, draw;
};

struct Mix6Weight
{
    // 1  map=weight.map(board), shape=H*W*4*c
    int16_t map[ShapeNum][FeatureDim];

    // 2  mapsum=map.sum(2), shape=H*W*c
    // 3  mapAfterLR=leakyRelu(mapsum)
    int16_t map_lr_slope_sub1div8[FeatureDim];
    int16_t map_lr_bias[FeatureDim];

    // 4  update policyBeforeConv and valueSumBoard
    // 5  policyAfterConv=depthwise_conv2d(policyBeforeConv)
    int16_t policy_conv_weight[9][PolicyDim];
    int16_t policy_conv_bias[PolicyDim];

    // 6  policy=conv1x1(relu(policyAfterConv))
    int16_t policy_final_conv[PolicyDim];
    // 7  policy=leakyRelu(policyAfterConv)
    float policy_neg_slope, policy_pos_slope;

    // 8  value leakyRelu
    float scale_before_mlp;
    float value_lr_slope_sub1[ValueDim];

    // 9  mlp
    float mlp_w1[ValueDim][ValueDim];  // shape=(inc，outc)
    float mlp_b1[ValueDim];
    float mlp_w2[ValueDim][ValueDim];
    float mlp_b2[ValueDim];
    float mlp_w3[ValueDim][3];
    float _mlp_w3_padding[5];

    float mlp_b3[3];
    float _mlp_b3_padding[5];
};

class Mix6Accumulator
{
public:
    Mix6Accumulator(int boardSize);
    ~Mix6Accumulator();

    /// Init accumulator state to empty board.
    void clear(const Mix6Weight &w, int alignBoardSize = 0);
    /// Incremental update mix6 network state.
    void update(const Mix6Weight              &w,
                Color                          oldColor,
                Color                          newColor,
                int                            x,
                int                            y,
                std::array<int32_t, ValueDim> &valueSumBoardBackup);

    /// Calculate value of current network state.
    RawValue evaluateValue(const Mix6Weight &w);
    /// Calculate policy value of current network state.
    void evaluatePolicy(const Mix6Weight &w, PolicyBuffer &policyBuffer);

private:
    int   boardSize;
    int   numCells;
    float boardSizeScale;

    //=============================================================
    // Mix6 network states

    // 0  convert board to shape (4 direction on every board loc)
    std::array<uint32_t, 4> *shapeTable;

    // 1  map=weight.map(shape), shape=H*W*4*c
    // 2  mapsum=map.sum(2), shape=H*W*c
    std::array<int16_t, FeatureDim> *mapSum;

    // 3  mapAfterLR=leakyRelu(mapsum)
    std::array<int16_t, FeatureDim> *mapAfterLR;

    // 4  policyBeforeConv=mapAfterLR[policy channels]
    // 5  policyAfterConv=depthwise_conv2d(policyBeforeConv)
    std::array<int16_t, PolicyDim> *policyAfterConv;

    // 6  policy=conv1x1(Relu(policyAfterConv))
    // 7  valueSumBoard=mapAfterLR[value channels].sum(0,1)
    std::array<int32_t, ValueDim> valueSumBoard;  // int32 to avoid overflow

    // 8  global mlp

    //=============================================================
    friend class Mix6Evaluator;
};

class Mix6Evaluator : public Evaluator
{
public:
    Mix6Evaluator(int                   boardSize,
                  Rule                  rule,
                  std::filesystem::path blackWeightPath,
                  std::filesystem::path whiteWeightPath,
                  int                   alignBoardSize = 0);
    ~Mix6Evaluator();

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

    Mix6Weight /* non-owning ptr */           *weight[2];
    std::unique_ptr<Mix6Accumulator>           accumulator[2];
    std::vector<MoveCache>                     moveCache[2];
    std::vector<std::array<int32_t, ValueDim>> valueSumBoardHistory[2];
    int                                        alignBoardSize;
};

}  // namespace Evaluation::mix6
