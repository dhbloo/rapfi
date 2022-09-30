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

namespace Evaluation::mix7 {

using namespace Evaluation;

constexpr uint32_t ArchHashBase = 0x45ea890;
constexpr size_t   Alignment    = 32;
constexpr int      ShapeNum     = 708588;
constexpr int      PolicyDim    = 32;
constexpr int      ValueDim     = 64;
constexpr int      MapDim       = std::max(PolicyDim, ValueDim);
constexpr int      DWConvDim    = 32;

struct alignas(Alignment) Mix7Weight
{
    // 1  mapping layer
    int16_t mapping[ShapeNum][MapDim];

    // 2  PReLU after mapping
    int16_t map_prelu_weight[MapDim];

    // 3  Depthwise conv
    int16_t dw_conv_weight[9][DWConvDim];
    int16_t dw_conv_bias[DWConvDim];

    // 4  Policy pointwise conv
    int16_t policy_pw_conv_weight[PolicyDim];

    // 7  Value MLP (layer 1,2,3)
    float value_l1_weight[ValueDim][ValueDim];  // shape=(in，out)
    float value_l1_bias[ValueDim];
    float value_l2_weight[ValueDim][ValueDim];
    float value_l2_bias[ValueDim];
    float value_l3_weight[ValueDim][3];
    float value_l3_bias[3];

    // 5  Policy PReLU
    float policy_neg_weight;
    float policy_pos_weight;

    // 6  Value sum scale
    float value_sum_scale_after_conv;
    float value_sum_scale_direct;

    char __padding_to_32bytes[4];
};

class alignas(Alignment) Mix7Accumulator
{
public:
    Mix7Accumulator(int boardSize);
    ~Mix7Accumulator();

    /// Init accumulator state to empty board.
    void clear(const Mix7Weight &w, int alignBoardSize = 0);
    /// Incremental update mix6 network state.
    enum UpdateType { MOVE, UNDO };
    template <UpdateType UT>
    void update(const Mix7Weight              &w,
                Color                          pieceColor,
                int                            x,
                int                            y,
                std::array<int32_t, ValueDim> *valueSumBoardBackup);

    /// Calculate value (win/loss/draw tuple) of current network state.
    std::tuple<float, float, float> evaluateValue(const Mix7Weight &w);
    /// Calculate policy value of current network state.
    void evaluatePolicy(const Mix7Weight &w, PolicyBuffer &policyBuffer);

private:
    friend class Mix7Evaluator;
    //=============================================================
    // Mix7 network states

    /// Value feature sum of the full board
    std::array<int32_t, ValueDim> valueSum;  // [ValueDim] (aligned to 32)
    /// Index table to convert line shape to map feature
    std::array<uint32_t, 4> *indexTable;  // [H*W, 4] (unaligned)
    /// Sumed map feature of four directions
    std::array<int16_t, MapDim> *mapSum;  // [H*W, MapDim] (aligned)
    /// Map feature after depth wise conv
    std::array<int16_t, DWConvDim> *mapAfterDWConv;  // [(H+2)*(W+2), DWConvDim] (aligned)

    //=============================================================
    int   boardSize;
    int   fullBoardSize;  // (boardSize + 2)
    float boardSizeScale;

    void initIndexTable();
};

class Mix7Evaluator : public Evaluator
{
public:
    Mix7Evaluator(int                   boardSize,
                  Rule                  rule,
                  std::filesystem::path blackWeightPath,
                  std::filesystem::path whiteWeightPath,
                  int                   alignBoardSize = 0);
    ~Mix7Evaluator();

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

    Mix7Weight /* non-owning ptr */           *weight[2];
    std::unique_ptr<Mix7Accumulator>           accumulator[2];
    std::vector<MoveCache>                     moveCache[2];
    std::vector<std::array<int32_t, ValueDim>> valueSumBoardHistory[2];
    int                                        alignBoardSize;
};

}  // namespace Evaluation::mix7
