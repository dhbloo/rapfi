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

/*
    Mix6 NNUE Architecture is from @hzyhhzy:
    https://github.com/hzyhhzy/gomoku_nnue
*/

#include "mix6nnue.h"

#include "../core/iohelper.h"
#include "../core/platform.h"
#include "../core/utils.h"
#include "../game/board.h"
#include "simdops.h"
#include "weightloader.h"

#include <algorithm>
#include <cmath>

namespace {

using namespace Evaluation::mix6;

constexpr auto Power3 = []() {
    auto pow3 = std::array<int, 16> {};
    for (size_t i = 0; i < pow3.size(); i++)
        pow3[i] = power(3, i);
    return pow3;
}();

constexpr int DX[4] = {1, 0, 1, 1};
constexpr int DY[4] = {0, 1, 1, -1};

/// DPower3[oldColor][newColor]
constexpr int DPower3[4][4] = {
    {0, 1, 0, -1},
    {-1, 0, 0, -2},
    {0, 0, 0, 0},
    {1, 2, 0, 0},
};

static Evaluation::WeightRegistry<Mix6Weight> Mix6WeightRegistry;

inline simde__m256i mapLeakyRelu(simde__m256i x, simde__m256i lrWeight, simde__m256i lrBias)
{
    auto xOver4 = simde_mm256_srai_epi16(x, 2);                           // 0.25x
    auto y      = simde_mm256_min_epi16(simde_mm256_setzero_si256(), x);  // -relu(-x)
    y           = simde_mm256_mulhrs_epi16(lrWeight, y);  // slopeSub1Div8*(-relu(-x))
    y           = simde_mm256_slli_epi16(y, 1);           // 2*slopeSub1Div8*(-relu(-x))
    auto z      = simde_mm256_add_epi16(lrBias, xOver4);  // 0.25leakyrelu(x)
    y           = simde_mm256_add_epi16(y, z);            // 0.25leakyrelu(x)+bias
    return y;
};

}  // namespace

namespace Evaluation::mix6 {

Mix6Accumulator::Mix6Accumulator(int boardSize)
    : boardSize(boardSize)
    , numCells(boardSize * boardSize)
    , boardSizeScale(1.0f / (boardSize * boardSize))
{
    shapeTable      = new std::array<uint32_t, 4>[numCells];
    mapSum          = new std::array<int16_t, FeatureDim>[numCells];
    mapAfterLR      = new std::array<int16_t, FeatureDim>[numCells];
    policyAfterConv = new std::array<int16_t, PolicyDim>[numCells];
}

Mix6Accumulator::~Mix6Accumulator()
{
    delete[] shapeTable;
    delete[] mapSum;
    delete[] mapAfterLR;
    delete[] policyAfterConv;
}

void Mix6Accumulator::clear(const Mix6Weight &w, int alignBoardSize)
{
    // Clear shape table
    std::fill_n(&shapeTable[0][0], 4 * numCells, 0);
    // Init shape table
    for (int thick = 1; thick <= 5; thick++) {
        for (int i = 0; i < boardSize; i++) {
            int c = 0;
            for (int j = 0; j < thick; j++)
                c += Power3[11 - j];
            shapeTable[(boardSize - 6 + thick) + i * boardSize][0] = c;
            shapeTable[i + (boardSize - 6 + thick) * boardSize][1] = c;
            shapeTable[(boardSize - 6 + thick) + i * boardSize][2] = c;
            shapeTable[i + (boardSize - 6 + thick) * boardSize][2] = c;
            shapeTable[(boardSize - 6 + thick) + i * boardSize][3] = c;
            shapeTable[i + (6 - 1 - thick) * boardSize][3]         = c;
        }
    }

    for (int thick = 1; thick <= 5; thick++) {
        for (int i = 0; i < boardSize; i++) {
            int c = 2 * Power3[11];
            for (int j = 0; j < thick - 1; j++)
                c += Power3[j];
            shapeTable[(6 - 1 - thick) + i * boardSize][0]         = c;
            shapeTable[i + (6 - 1 - thick) * boardSize][1]         = c;
            shapeTable[(6 - 1 - thick) + i * boardSize][2]         = c;
            shapeTable[i + (6 - 1 - thick) * boardSize][2]         = c;
            shapeTable[(6 - 1 - thick) + i * boardSize][3]         = c;
            shapeTable[i + (boardSize - 6 + thick) * boardSize][3] = c;
        }
    }

    for (int a = 1; a <= 5; a++)
        for (int b = 1; b <= 5; b++) {
            int c = 3 * Power3[11];
            for (int i = 0; i < a - 1; i++)
                c += Power3[10 - i];
            for (int i = 0; i < b - 1; i++)
                c += Power3[i];
            shapeTable[(boardSize - 6 + a) + (5 - b) * boardSize][2]             = c;
            shapeTable[(boardSize - 6 + a) * boardSize + (5 - b)][2]             = c;
            shapeTable[(5 - b) + (5 - a) * boardSize][3]                         = c;
            shapeTable[(boardSize - 6 + a) + (boardSize - 6 + b) * boardSize][3] = c;
        }

    // Clear policyAfterConv and valueSumBoard
    for (int b = 0; b < PolicyBatches; b++) {
        auto bias = simde_mm256_loadu_si256(w.policy_conv_bias + b * BatchSize);
        for (int i = 0; i < numCells; i++)
            simde_mm256_storeu_si256(policyAfterConv[i].data() + b * BatchSize, bias);
    }

    for (int b = 0; b < ValueBatches32; b++) {
        simde_mm256_storeu_si256(valueSumBoard.data() + b * BatchSize32,
                                 simde_mm256_setzero_si256());
    }

    // Init mapSum, mapAfterLR, policyAfterConv, valueSumBoard
    for (int i = 0; i < numCells; i++) {
        int y = i / boardSize, x = i % boardSize;

        for (int b = 0; b < ValueBatches + PolicyBatches; b++) {
            // mapSum
            auto sumw = simde_mm256_setzero_si256();
            for (int dir = 0; dir < 4; dir++) {
                auto dw = simde_mm256_loadu_si256(w.map[shapeTable[i][dir]] + b * BatchSize);
                sumw    = simde_mm256_add_epi16(sumw, dw);
            }
            void *wp = mapSum[i].data() + b * BatchSize;
            simde_mm256_storeu_si256(wp, sumw);

            // leaky relu
            auto lrw = simde_mm256_loadu_si256(w.map_lr_slope_sub1div8 + b * BatchSize);
            auto lrb = simde_mm256_loadu_si256(w.map_lr_bias + b * BatchSize);
            sumw     = mapLeakyRelu(sumw, lrw, lrb);
            wp       = mapAfterLR[i].data() + b * BatchSize;
            simde_mm256_storeu_si256(wp, sumw);

            // policy conv
            if (b < PolicyBatches) {
                for (int dy = -1; dy <= 1; dy++) {
                    int yi = y + dy;
                    if ((unsigned)yi >= boardSize)
                        continue;  // zero padding
                    for (int dx = -1; dx <= 1; dx++) {
                        int xi = x + dx;
                        if ((unsigned)xi >= boardSize)
                            continue;  // zero padding

                        auto convw = simde_mm256_loadu_si256(w.policy_conv_weight[4 - dy * 3 - dx]
                                                             + b * BatchSize);
                        wp         = policyAfterConv[boardSize * yi + xi].data() + b * BatchSize;
                        auto oldw  = simde_mm256_loadu_si256(wp);
                        oldw = simde_mm256_add_epi16(oldw, simde_mm256_mulhrs_epi16(sumw, convw));
                        simde_mm256_storeu_si256(wp, oldw);
                    }
                }
            }
            // value sum
            else {
                int vb = 2 * (b - PolicyBatches);

                wp          = valueSumBoard.data() + vb * BatchSize32;
                auto neww32 = simde_mm256_cvtepi16_epi32(simde_mm256_extracti128_si256(sumw, 0));
                auto oldw   = simde_mm256_loadu_si256(wp);
                oldw        = simde_mm256_add_epi32(oldw, neww32);
                simde_mm256_storeu_si256(wp, oldw);

                wp     = valueSumBoard.data() + (vb + 1) * BatchSize32;
                neww32 = simde_mm256_cvtepi16_epi32(simde_mm256_extracti128_si256(sumw, 1));
                oldw   = simde_mm256_loadu_si256(wp);
                oldw   = simde_mm256_add_epi32(oldw, neww32);
                simde_mm256_storeu_si256(wp, oldw);
            }
        }
    }

    // Align valueSumBoard to base board size version
    if (alignBoardSize) {
        Mix6Accumulator accmulatorToAlign(alignBoardSize);
        accmulatorToAlign.clear(w);
        valueSumBoard = accmulatorToAlign.valueSumBoard;
    }
}

void Mix6Accumulator::update(const Mix6Weight              &w,
                             Color                          oldColor,
                             Color                          newColor,
                             int                            x,
                             int                            y,
                             std::array<int32_t, ValueDim> &valueSumBoardBackup)
{
    struct OnePointChange
    {
        int8_t   x;
        int8_t   y;
        int16_t  index;
        int16_t  dir;
        uint32_t oldShape;
        uint32_t newShape;
    } changeTable[4 * 11];
    int changeCount = 0;
    int dPower3     = DPower3[oldColor][newColor];

    // Update shape table and record changes
    for (int dir = 0; dir < 4; dir++) {
        for (int dist = -5; dist <= 5; dist++) {
            int xi = x - dist * DX[dir];
            int yi = y - dist * DY[dir];

            // less-branch test: xi < 0 || xi >= boardSize || yi < 0 || yi >= boardSize
            if ((xi | (boardSize - 1 - xi) | yi | (boardSize - 1 - yi)) < 0)
                continue;

            OnePointChange &c        = changeTable[changeCount++];
            c.x                      = xi;
            c.y                      = yi;
            c.dir                    = dir;
            c.index                  = boardSize * yi + xi;
            c.oldShape               = shapeTable[c.index][dir];
            c.newShape               = c.oldShape + dPower3 * Power3[dist + 5];
            shapeTable[c.index][dir] = c.newShape;
            assert(0 <= c.newShape && c.newShape < ShapeNum);
        }
    }

    // Load value sum
    DEF_BATCH256(int32_t, ValueDim, RegWidth32, ValueBatches32);
    simde__m256i vSum[ValueBatches32];
    for (int b = 0; b < ValueBatches32; b++)
        vSum[b] = simde_mm256_loadu_si256(
            reinterpret_cast<simde__m256i *>(valueSumBoard.data() + b * RegWidth32));

    // Incremental update policy feature and value feature sum
    for (int i = 0; i < changeCount; i++) {
        const OnePointChange &c = changeTable[i];

        for (int b = 0; b < ValueBatches + PolicyBatches; b++) {
            // mapSum
            auto  oldw = simde_mm256_loadu_si256(w.map[c.oldShape] + b * BatchSize);
            auto  neww = simde_mm256_loadu_si256(w.map[c.newShape] + b * BatchSize);
            void *wp   = mapSum[c.index].data() + b * BatchSize;
            auto  sumw = simde_mm256_loadu_si256(wp);
            sumw       = simde_mm256_sub_epi16(sumw, oldw);
            sumw       = simde_mm256_add_epi16(sumw, neww);
            simde_mm256_storeu_si256(wp, sumw);

            // leaky relu
            auto lrw = simde_mm256_loadu_si256(w.map_lr_slope_sub1div8 + b * BatchSize);
            auto lrb = simde_mm256_loadu_si256(w.map_lr_bias + b * BatchSize);
            neww     = mapLeakyRelu(sumw, lrw, lrb);
            wp       = mapAfterLR[c.index].data() + b * BatchSize;
            oldw     = simde_mm256_loadu_si256(wp);
            simde_mm256_storeu_si256(wp, neww);

            // policy conv
            if (b < PolicyBatches) {
                for (int dy = -1; dy <= 1; dy++) {
                    int yi = c.y + dy;
                    if ((unsigned)yi >= boardSize)
                        continue;  // zero padding
                    for (int dx = -1; dx <= 1; dx++) {
                        int xi = c.x + dx;
                        if ((unsigned)xi >= boardSize)
                            continue;  // zero padding

                        auto convw = simde_mm256_loadu_si256(w.policy_conv_weight[4 - dy * 3 - dx]
                                                             + b * BatchSize);
                        wp         = policyAfterConv[boardSize * yi + xi].data() + b * BatchSize;

                        sumw = simde_mm256_loadu_si256(wp);
                        sumw = simde_mm256_sub_epi16(sumw, simde_mm256_mulhrs_epi16(oldw, convw));
                        sumw = simde_mm256_add_epi16(sumw, simde_mm256_mulhrs_epi16(neww, convw));
                        simde_mm256_storeu_si256(wp, sumw);
                    }
                }
            }
            // value sum
            else if (newColor != EMPTY) {
                int vb = 2 * (b - PolicyBatches);

                auto oldw0   = simde_mm256_cvtepi16_epi32(simde_mm256_extracti128_si256(oldw, 0));
                auto neww0   = simde_mm256_cvtepi16_epi32(simde_mm256_extracti128_si256(neww, 0));
                auto oldw1   = simde_mm256_cvtepi16_epi32(simde_mm256_extracti128_si256(oldw, 1));
                auto neww1   = simde_mm256_cvtepi16_epi32(simde_mm256_extracti128_si256(neww, 1));
                vSum[vb + 0] = simde_mm256_sub_epi32(vSum[vb + 0], oldw0);
                vSum[vb + 1] = simde_mm256_sub_epi32(vSum[vb + 1], oldw1);
                vSum[vb + 0] = simde_mm256_add_epi32(vSum[vb + 0], neww0);
                vSum[vb + 1] = simde_mm256_add_epi32(vSum[vb + 1], neww1);
            }
        }
    }

    if (newColor == EMPTY) {
        valueSumBoard = valueSumBoardBackup;  // just copy it
    }
    else {
        // Store value sum
        for (int b = 0; b < ValueBatches32; b++)
            simde_mm256_storeu_si256(
                reinterpret_cast<simde__m256i *>(valueSumBoard.data() + b * RegWidth32),
                vSum[b]);
    }
}

RawValue Mix6Accumulator::evaluateValue(const Mix6Weight &w)
{
    auto scale_before_mlp = simde_mm256_set1_ps(w.scale_before_mlp * boardSizeScale);

    // layer 0 leakyrelu
    float layer0[ValueDim];
    for (int b = 0; b < ValueBatches32; b++) {
        auto x = simde_mm256_loadu_si256(valueSumBoard.data() + b * BatchSize32);  // load
        auto y = simde_mm256_cvtepi32_ps(x);
        auto s = simde_mm256_loadu_ps(w.value_lr_slope_sub1 + b * BatchSize32);  // load
        y      = simde_mm256_mul_ps(y, scale_before_mlp);                        // scale
        y      = simde_mm256_fmadd_ps(s,
                                 simde_mm256_min_ps(simde_mm256_setzero_ps(), y),
                                 y);  // leaky relu
        simde_mm256_storeu_ps(layer0 + b * BatchSize32, y);
    }

    // linear 1
    float layer1[ValueDim];
    simd::linearLayer<simd::Activation::Relu>(layer1, layer0, w.mlp_w1, w.mlp_b1);

    // linear 2
    float layer2[ValueDim];
    simd::linearLayer<simd::Activation::Relu>(layer2, layer1, w.mlp_w2, w.mlp_b2);
    // simd::add<ValueDim, float, alignof(float)>(layer2, layer2, layer0);
    for (int b = 0; b < ValueBatches32; b++) {
        auto y   = simde_mm256_loadu_ps(layer2 + b * BatchSize32);
        auto res = simde_mm256_loadu_ps(layer0 + b * BatchSize32);
        y        = simde_mm256_add_ps(y, res);  // residual connection
        simde_mm256_storeu_ps(layer2 + b * BatchSize32, y);
    }

    // final linear
    float value[8];
    simd::linearLayer<simd::Activation::None>(value, layer2, w.mlp_w3, w.mlp_b3);

    return {value[0], value[1], value[2]};
}

void Mix6Accumulator::evaluatePolicy(const Mix6Weight &w, PolicyBuffer &policyBuffer)
{
    static_assert(PolicyBatches == 1,
                  "Assume there's only one policy batch, or we need to calculate policy by batch");

    for (int i = 0; i < numCells; i++) {
        if (!policyBuffer.getComputeFlag(i))
            continue;

        void *wp     = policyAfterConv[i].data();
        auto  t      = simde_mm256_loadu_si256(wp);
        t            = simde_mm256_max_epi16(simde_mm256_setzero_si256(), t);  // relu
        auto convw   = simde_mm256_loadu_si256(w.policy_final_conv);
        t            = simde_mm256_mulhrs_epi16(t, convw);
        float policy = static_cast<float>(simd::regop::hsumI16(t));
        policy *= (policy < 0 ? w.policy_neg_slope : w.policy_pos_slope);

        policyBuffer(i) = policy;
    }
}

Mix6Evaluator::Mix6Evaluator(int                   boardSize,
                             Rule                  rule,
                             std::filesystem::path blackWeightPath,
                             std::filesystem::path whiteWeightPath,
                             int                   alignBoardSize)
    : Evaluator(boardSize, rule)
    , weight {nullptr, nullptr}
    , alignBoardSize(alignBoardSize)
{
    std::filesystem::path currentWeightPath;

    CompressedWrapper<StandardHeaderParserWarpper<BinaryPODWeightLoader<Mix6Weight>>> loader(
        Compressor::Type::LZ4_DEFAULT);
    loader.setHeaderValidator([&](StandardHeader header) -> bool {
        if (header.archHash != ArchHash && header.archHash != ArchHashTwoSide)
            throw IncompatibleWeightFileError("incompatible architecture in weight file.");

        if (!contains(header.supportedRules, rule))
            throw UnsupportedRuleError(rule);

        if (!contains(header.supportedBoardSizes, boardSize))
            throw UnsupportedBoardSizeError(boardSize);

        if (Config::MessageMode != MsgMode::NONE)
            MESSAGEL("mix6nnue: load weight from " << pathToString(currentWeightPath));
        return true;
    });

    for (const auto &[weightSide, weightPath] : {
             std::make_pair(BLACK, blackWeightPath),
             std::make_pair(WHITE, whiteWeightPath),
         }) {
        currentWeightPath  = weightPath;
        weight[weightSide] = Mix6WeightRegistry.loadWeightFromFile(weightPath, loader);
        if (!weight[weightSide])
            throw std::runtime_error("failed to load nnue weight from " + pathToString(weightPath));
    }

    accumulator[BLACK] = std::make_unique<Mix6Accumulator>(boardSize);
    accumulator[WHITE] = std::make_unique<Mix6Accumulator>(boardSize);

    int numCells = boardSize * boardSize;
    moveCache[BLACK].reserve(numCells);
    moveCache[WHITE].reserve(numCells);

    valueSumBoardHistory[BLACK].reserve(numCells);
    valueSumBoardHistory[WHITE].reserve(numCells);
}

Mix6Evaluator::~Mix6Evaluator()
{
    if (weight[BLACK])
        Mix6WeightRegistry.unloadWeight(weight[BLACK]);
    if (weight[WHITE])
        Mix6WeightRegistry.unloadWeight(weight[WHITE]);
}

void Mix6Evaluator::initEmptyBoard()
{
    moveCache[BLACK].clear();
    moveCache[WHITE].clear();
    accumulator[BLACK]->clear(*weight[BLACK], alignBoardSize);
    accumulator[WHITE]->clear(*weight[WHITE], alignBoardSize);
}

void Mix6Evaluator::beforeMove(const Board &board, Pos pos)
{
    addCache(board.sideToMove(), pos.x(), pos.y(), false);
}

void Mix6Evaluator::afterUndo(const Board &board, Pos pos)
{
    addCache(board.sideToMove(), pos.x(), pos.y(), true);
}

ValueType Mix6Evaluator::evaluateValue(const Board &board)
{
    Color self = board.sideToMove();

    // Apply all incremental update and calculate value
    clearCache(self);
    RawValue rawValue = accumulator[self]->evaluateValue(*weight[self]);

    return ValueType(rawValue.win, rawValue.loss, rawValue.draw, true);
}

void Mix6Evaluator::evaluatePolicy(const Board &board, PolicyBuffer &policyBuffer)
{
    Color self = board.sideToMove();

    // Apply all incremental update and calculate policy
    clearCache(self);
    accumulator[self]->evaluatePolicy(*weight[self], policyBuffer);
}

void Mix6Evaluator::clearCache(Color side)
{
    constexpr Color opponentMap[4] = {WHITE, BLACK, WALL, EMPTY};

    for (MoveCache &mc : moveCache[side]) {
        if (side == WHITE) {
            mc.oldColor = opponentMap[mc.oldColor];
            mc.newColor = opponentMap[mc.newColor];
        }

        if (mc.oldColor == EMPTY)
            valueSumBoardHistory[side].push_back(accumulator[side]->valueSumBoard);

        accumulator[side]->update(*weight[side],
                                  mc.oldColor,
                                  mc.newColor,
                                  mc.x,
                                  mc.y,
                                  valueSumBoardHistory[side].back());

        if (mc.newColor == EMPTY)
            valueSumBoardHistory[side].pop_back();
    }
    moveCache[side].clear();
}

void Mix6Evaluator::addCache(Color side, int x, int y, bool isUndo)
{
    Color oldColor = EMPTY;
    Color newColor = side;
    if (isUndo)
        std::swap(oldColor, newColor);

    MoveCache newCache {oldColor, newColor, (int8_t)x, (int8_t)y};

    for (Color c : {BLACK, WHITE}) {
        if (moveCache[c].empty() || !isContraryMove(newCache, moveCache[c].back()))
            moveCache[c].push_back(newCache);
        else
            moveCache[c].pop_back();  // cancel out the last move cache

        assert(moveCache[c].size() < boardSize * boardSize);
    }
}

}  // namespace Evaluation::mix6
