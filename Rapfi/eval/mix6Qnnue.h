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

namespace Evaluation::mix6q {

using namespace Evaluation;

const uint32_t  ArchHash         = 0x3eede3db;
constexpr int   ShapeNum         = 708588;
constexpr int   PolicyDim        = 32;
constexpr int   ValueDim         = 64;
constexpr int   FeatureDim       = PolicyDim + ValueDim;
constexpr int   ScaleWeight      = 64;
constexpr int   ScaleBoardFactor = 3600;

/// Feature mapping table(s).
struct Mix6QWeightMapping
{
    struct Mapping
    {
        int8_t feature[ShapeNum][FeatureDim];
    };

    // 1  mapping layer
    std::vector<Mapping> mappings;
};

/// Weights after feature mapping table (POD struct).
struct Mix6QWeightAfterMapping
{
    // 2  PReLU after mapping
    int16_t prelu_weight[FeatureDim];

    // 3  Policy depthwise conv
    int8_t  policy_dw_conv_weight[9][PolicyDim];
    int16_t policy_dw_conv_bias[PolicyDim];

    // 4  Policy pointwise conv
    int8_t policy_pw_conv_weight[PolicyDim];

    // 5  Value MLP (layer 1,2,3)
    int8_t  value_linear1_weight[ValueDim][ValueDim];  // shape=(out channel, in channel)
    int32_t value_linear1_bias[ValueDim];
    int8_t  value_linear2_weight[ValueDim][ValueDim];
    int32_t value_linear2_bias[ValueDim];
    int8_t  value_linear3_weight[3 + 1][ValueDim];  // add one for padding
    int32_t value_linear3_bias[3 + 1];              // add one for padding

    float policy_output_scale;
    float value_output_scale;
};

/// The full mix6Qnnue weight struct.
struct Mix6QWeight
    : Mix6QWeightMapping
    , Mix6QWeightAfterMapping
{};

class Mix6QAccumulator
{
public:
    Mix6QAccumulator(int boardSize, int16_t mapIndex);
    ~Mix6QAccumulator();

    /// Init accumulator state to empty board.
    void clear(const Mix6QWeight &w, int alignBoardSize = 0);
    /// Incremental update mix6v2 network state.
    void update(const Mix6QWeight &w, Color oldColor, Color newColor, int x, int y);

    /// Calculate value of current network state.
    std::tuple<float, float, float> evaluateValue(const Mix6QWeight &w);
    /// Calculate policy value of current network state.
    void evaluatePolicy(const Mix6QWeight &w, PolicyBuffer &policyBuffer);

private:
    int     boardSize;
    int     numCells;
    float   boardSizeScale;
    int16_t mapIndex;
    int16_t boardSizeFactor;

    //=============================================================
    // Mix6 network states

    // 0  convert board to shape (4 direction on every board loc)
    std::array<uint32_t, 4> *shapeTable;

    // 1  feature=mapping(shape), shape=H*W*4*c
    // 2  featureSum=feature.sum(dim=2), shape=H*W*c
    std::array<int16_t, FeatureDim> *featureSum;

    // 3  featureAfterPReLU=PReLU(featureSum)/4
    std::array<int8_t, FeatureDim> *featureAfterPReLU;

    // 4  policyAfterDWConv=policy_dw_conv(featureAfterPReLU[:PolicyDim])/4
    std::array<int16_t, PolicyDim> *policyAfterDWConv;

    // 5  valueSumBoard=featureAfterPReLU[PolicyDim:].sum(dim=(2,3))
    std::array<int16_t, ValueDim> valueSumBoard;
    //=============================================================

    void initShapeTable();
};

class Mix6QEvaluator : public Evaluator
{
public:
    Mix6QEvaluator(int                   boardSize,
                   Rule                  rule,
                   std::filesystem::path weightPath,
                   int                   alignBoardSize = 0);
    ~Mix6QEvaluator();

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

    Mix6QWeight /* non-owning ptr */ *weight;
    std::unique_ptr<Mix6QAccumulator> accumulator[2];
    std::vector<MoveCache>            moveCache[2];
    int                               alignBoardSize;
};

}  // namespace Evaluation::mix6q
