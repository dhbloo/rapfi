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

#include "mix7nnue.h"

#include "../core/iohelper.h"
#include "../core/platform.h"
#include "../core/utils.h"
#include "../game/board.h"
#include "simdops.h"
#include "weightloader.h"

#include <algorithm>
#include <cmath>

namespace {

using namespace Evaluation::mix7;

constexpr auto Power3 = []() {
    auto pow3 = std::array<int, 16> {};
    for (size_t i = 0; i < pow3.size(); i++)
        pow3[i] = power(3, i);
    return pow3;
}();

constexpr int DX[4] = {1, 0, 1, 1};
constexpr int DY[4] = {0, 1, 1, -1};

constexpr bool ConvChangeMask[13][13] = {
    {1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1},
    {1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0},
    {0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0},
    {0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0},
    {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1},
    {1, 1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1},
    {1, 1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 1, 1},
};

static Evaluation::WeightRegistry<Mix7Weight> Mix7WeightRegistry;

}  // namespace

namespace Evaluation::mix7 {

Mix7Accumulator::Mix7Accumulator(int boardSize)
    : boardSize(boardSize)
    , fullBoardSize(boardSize + 2)
    , boardSizeScale(1.0f / (boardSize * boardSize))
{
    int nCells = boardSize * boardSize;
    indexTable = new std::array<uint32_t, 4>[nCells];
    mapSum     = MemAlloc::alignedArrayAlloc<std::array<int16_t, MapDim>, Alignment>(nCells);

    mapAfterDWConv = MemAlloc::alignedArrayAlloc<std::array<int16_t, DWConvDim>, Alignment>(
        fullBoardSize * fullBoardSize);
}

Mix7Accumulator::~Mix7Accumulator()
{
    delete[] indexTable;
    MemAlloc::alignedFree(mapSum);
    MemAlloc::alignedFree(mapAfterDWConv);
}

void Mix7Accumulator::initIndexTable()
{
    // Clear shape table
    std::fill_n(&indexTable[0][0], 4 * boardSize * boardSize, 0);

    auto getIndex = [&](int x, int y) -> std::array<uint32_t, 4> & {
        return indexTable[x + y * boardSize];
    };

    // Init shape table
    for (int thick = 1; thick <= 5; thick++) {
        for (int i = 0; i < boardSize; i++) {
            int c = 0;
            for (int j = 0; j < thick; j++)
                c += Power3[11 - j];

            getIndex(boardSize - 6 + thick, i)[0] = c;
            getIndex(i, boardSize - 6 + thick)[1] = c;
            getIndex(boardSize - 6 + thick, i)[2] = c;
            getIndex(i, boardSize - 6 + thick)[2] = c;
            getIndex(boardSize - 6 + thick, i)[3] = c;
            getIndex(i, 6 - 1 - thick)[3]         = c;
        }
    }

    for (int thick = 1; thick <= 5; thick++) {
        for (int i = 0; i < boardSize; i++) {
            int c = 2 * Power3[11];
            for (int j = 0; j < thick - 1; j++)
                c += Power3[j];

            getIndex(6 - 1 - thick, i)[0]         = c;
            getIndex(i, 6 - 1 - thick)[1]         = c;
            getIndex(6 - 1 - thick, i)[2]         = c;
            getIndex(i, 6 - 1 - thick)[2]         = c;
            getIndex(6 - 1 - thick, i)[3]         = c;
            getIndex(i, boardSize - 6 + thick)[3] = c;
        }
    }

    for (int a = 1; a <= 5; a++)
        for (int b = 1; b <= 5; b++) {
            int c = 3 * Power3[11];
            for (int i = 0; i < a - 1; i++)
                c += Power3[10 - i];
            for (int i = 0; i < b - 1; i++)
                c += Power3[i];

            getIndex(boardSize - 6 + a, 5 - b)[2]             = c;
            getIndex(5 - b, boardSize - 6 + a)[2]             = c;
            getIndex(5 - b, 5 - a)[3]                         = c;
            getIndex(boardSize - 6 + a, boardSize - 6 + b)[3] = c;
        }
}

void Mix7Accumulator::clear(const Mix7Weight &w, int alignBoardSize)
{
    initIndexTable();

    // Init mapAfterDWConv to bias
    for (int i = 0; i < fullBoardSize * fullBoardSize; i++)
        simd::copy<DWConvDim, int16_t, Alignment>(mapAfterDWConv[i].data(), w.dw_conv_bias);
    // Init valueSum to zeros
    simd::zero<ValueDim, int32_t, Alignment>(valueSum.data());

    for (int y = 0, innerIdx = 0; y < boardSize; y++)
        for (int x = 0; x < boardSize; x++, innerIdx++) {
            // Init mapSum from four directions
            simd::zero<MapDim, int16_t, Alignment>(mapSum[innerIdx].data());
            for (int dir = 0; dir < 4; dir++)
                simd::add<MapDim, int16_t, Alignment>(mapSum[innerIdx].data(),
                                                      mapSum[innerIdx].data(),
                                                      w.mapping[indexTable[innerIdx][dir]]);

            // Init mapAfterDWConv from mapSum
            DEF_BATCH256(int16_t, MapDim, RegWidth, MapBatches);
            DEF_BATCH256(int16_t, ValueDim, RegWidth16, ValueBatches16);
            DEF_BATCH256(int16_t, DWConvDim, ConvRegWidth, ConvBatches);
            for (int b = 0; b < MapBatches; b++) {
                // Apply PReLU for mapSum
                auto feature = simde_mm256_load_si256(
                    reinterpret_cast<const simde__m256i *>(mapSum[innerIdx].data() + b * RegWidth));
                auto negWeight = simde_mm256_load_si256(
                    reinterpret_cast<const simde__m256i *>(w.map_prelu_weight + b * RegWidth));
                feature =
                    simde_mm256_max_epi16(feature, simde_mm256_mulhrs_epi16(feature, negWeight));

                // Apply depthwise conv
                if (b < ConvBatches) {
                    for (int dy = 0; dy <= 2; dy++) {
                        int yi = y + dy;
                        for (int dx = 0; dx <= 2; dx++) {
                            int xi       = x + dx;
                            int outerIdx = xi + yi * fullBoardSize;

                            auto convWeight =
                                simde_mm256_load_si256(reinterpret_cast<const simde__m256i *>(
                                    w.dw_conv_weight[8 - dy * 3 - dx] + b * RegWidth));
                            auto *convPtr = reinterpret_cast<simde__m256i *>(
                                mapAfterDWConv[outerIdx].data() + b * RegWidth);
                            auto convFeat  = simde_mm256_load_si256(convPtr);
                            auto deltaFeat = simde_mm256_mulhrs_epi16(feature, convWeight);
                            convFeat       = simde_mm256_add_epi16(convFeat, deltaFeat);
                            simde_mm256_store_si256(convPtr, convFeat);
                        }
                    }
                }
                else if (b < ValueBatches16) {
                    auto *valueSumPtr =
                        reinterpret_cast<simde__m256i *>(valueSum.data() + b * RegWidth16);
                    auto valueSum0 = simde_mm256_load_si256(valueSumPtr);
                    auto valueSum1 = simde_mm256_load_si256(valueSumPtr + 1);

                    auto v0 = simde_mm256_cvtepi16_epi32(simde_mm256_castsi256_si128(feature));
                    auto v1 = simde_mm256_cvtepi16_epi32(simde_mm256_extracti128_si256(feature, 1));
                    valueSum0 = simde_mm256_add_epi32(valueSum0, v0);
                    valueSum1 = simde_mm256_add_epi32(valueSum1, v1);

                    simde_mm256_store_si256(valueSumPtr, valueSum0);
                    simde_mm256_store_si256(valueSumPtr + 1, valueSum1);
                }
            }
        }

    // Init valueSum by adding all value features
    for (int y = 0, outerIdx = fullBoardSize + 1; y < boardSize; y++, outerIdx += 2)
        for (int x = 0; x < boardSize; x++, outerIdx++) {
            DEF_BATCH256(int16_t, DWConvDim, RegWidth16, ConvBatches);
            DEF_BATCH256(int32_t, ValueDim, RegWidth32, ValueBatches32);
            for (int b = 0; b < ConvBatches; b++) {
                auto  feature = simde_mm256_load_si256(reinterpret_cast<simde__m256i *>(
                    mapAfterDWConv[outerIdx].data() + b * RegWidth16));
                auto *valueSumPtr =
                    reinterpret_cast<simde__m256i *>(valueSum.data() + b * 2 * RegWidth32);
                auto valueSum0 = simde_mm256_load_si256(valueSumPtr);
                auto valueSum1 = simde_mm256_load_si256(valueSumPtr + 1);

                auto v0 = simde_mm256_cvtepi16_epi32(simde_mm256_castsi256_si128(feature));
                auto v1 = simde_mm256_cvtepi16_epi32(simde_mm256_extracti128_si256(feature, 1));
                v0      = simde_mm256_max_epi16(v0, simde_mm256_setzero_si256());  // relu
                v1      = simde_mm256_max_epi16(v1, simde_mm256_setzero_si256());  // relu

                valueSum0 = simde_mm256_add_epi32(valueSum0, v0);
                valueSum1 = simde_mm256_add_epi32(valueSum1, v1);
                simde_mm256_store_si256(valueSumPtr, valueSum0);
                simde_mm256_store_si256(valueSumPtr + 1, valueSum1);
            }
        }

    // Align valueSum to base board size version
    if (alignBoardSize && boardSize != alignBoardSize) {
        Mix7Accumulator accmulatorToAlign(alignBoardSize);
        accmulatorToAlign.clear(w);
        valueSum = accmulatorToAlign.valueSum;
    }
}

template <Mix7Accumulator::UpdateType UT>
void Mix7Accumulator::update(const Mix7Weight              &w,
                             Color                          pieceColor,
                             int                            x,
                             int                            y,
                             std::array<int32_t, ValueDim> *valueSumBoardBackup)
{
    assert(pieceColor == BLACK || pieceColor == WHITE);
    struct OnePointChange
    {
        int8_t   x;
        int8_t   y;
        int16_t  dir;
        int16_t  innerIdx;
        uint32_t oldShape;
        uint32_t newShape;
    } changeTable[4 * 11];
    int changeCount = 0;
    int dPower3     = UT == MOVE ? pieceColor + 1 : -1 - pieceColor;

    // Update shape table and record changes
    for (int dir = 0; dir < 4; dir++) {
        for (int dist = -5; dist <= 5; dist++) {
            int xi = x - dist * DX[dir];
            int yi = y - dist * DY[dir];

            // less-branch test: xi < 0 || xi >= boardSize || yi < 0 || yi >= boardSize
            if ((xi | (boardSize - 1 - xi) | yi | (boardSize - 1 - yi)) < 0)
                continue;

            OnePointChange &c           = changeTable[changeCount++];
            c.x                         = xi;
            c.y                         = yi;
            c.dir                       = dir;
            c.innerIdx                  = boardSize * yi + xi;
            c.oldShape                  = indexTable[c.innerIdx][dir];
            c.newShape                  = c.oldShape + dPower3 * Power3[dist + 5];
            indexTable[c.innerIdx][dir] = c.newShape;
            assert(0 <= c.newShape && c.newShape < ShapeNum);
        }
    }

    // Load value sum
    DEF_BATCH256(int32_t, ValueDim, RegWidth32, ValueBatches32);
    simde__m256i vSum[ValueBatches32];
    int          x0, y0, x1, y1;
    if constexpr (UT == MOVE) {
        for (int b = 0; b < ValueBatches32; b++)
            vSum[b] = simde_mm256_load_si256(
                reinterpret_cast<simde__m256i *>(valueSum.data() + b * RegWidth32));

        x0 = std::max(x - 6 + 1, 1);
        y0 = std::max(y - 6 + 1, 1);
        x1 = std::min(x + 6 + 1, boardSize);
        y1 = std::min(y + 6 + 1, boardSize);

        // Subtract value feature sum
        for (int yi = y0, outerIdxBase = y0 * fullBoardSize; yi <= y1;
             yi++, outerIdxBase += fullBoardSize) {
            for (int xi = x0; xi <= x1; xi++) {
                int outerIdx = xi + outerIdxBase;

                DEF_BATCH256(int16_t, DWConvDim, RegWidth, ConvBatches);
                for (int b = 0; b < ConvBatches; b++) {
                    auto convF = simde_mm256_load_si256(reinterpret_cast<simde__m256i *>(
                        mapAfterDWConv[outerIdx].data() + b * RegWidth));
                    convF      = simde_mm256_max_epi16(convF, simde_mm256_setzero_si256());  // relu

                    auto value0 = simde_mm256_cvtepi16_epi32(simde_mm256_castsi256_si128(convF));
                    auto value1 =
                        simde_mm256_cvtepi16_epi32(simde_mm256_extracti128_si256(convF, 1));
                    vSum[2 * b + 0] = simde_mm256_sub_epi32(vSum[2 * b + 0], value0);
                    vSum[2 * b + 1] = simde_mm256_sub_epi32(vSum[2 * b + 1], value1);
                }
            }
        }
    }

    // Incremental update policy feature and value feature sum
    for (int i = 0; i < changeCount; i++) {
        const OnePointChange &c = changeTable[i];

        // Update mapSum and mapAfterDWConv
        DEF_BATCH256(int16_t, MapDim, RegWidth, MapBatches);
        simde__m256i oldFeats[MapBatches];
        simde__m256i newFeats[MapBatches];
        for (int b = 0; b < MapBatches; b++) {
            // Update mapSum
            auto newMapFeat = simde_mm256_load_si256(
                reinterpret_cast<const simde__m256i *>(w.mapping[c.newShape] + b * RegWidth));
            auto oldMapFeat = simde_mm256_load_si256(
                reinterpret_cast<const simde__m256i *>(w.mapping[c.oldShape] + b * RegWidth));
            auto deltaFeat = simde_mm256_sub_epi16(newMapFeat, oldMapFeat);

            auto *mapSumPtr =
                reinterpret_cast<simde__m256i *>(mapSum[c.innerIdx].data() + b * RegWidth);
            oldFeats[b] = simde_mm256_load_si256(mapSumPtr);
            newFeats[b] = simde_mm256_add_epi16(oldFeats[b], deltaFeat);
            simde_mm256_store_si256(mapSumPtr, newFeats[b]);

            // Apply PReLU for mapSum
            auto negW = simde_mm256_load_si256(
                reinterpret_cast<const simde__m256i *>(w.map_prelu_weight + b * RegWidth));
            oldFeats[b] =
                simde_mm256_max_epi16(oldFeats[b], simde_mm256_mulhrs_epi16(oldFeats[b], negW));
            newFeats[b] =
                simde_mm256_max_epi16(newFeats[b], simde_mm256_mulhrs_epi16(newFeats[b], negW));
        }

        // Update mapAfterDWConv
        DEF_BATCH256(int16_t, DWConvDim, ConvRegWidth, ConvBatches);
        for (int dy = 0, outerIdxBase = c.y * fullBoardSize + c.x; dy <= 2;
             dy++, outerIdxBase += fullBoardSize) {
            for (int dx = 0; dx <= 2; dx++) {
                auto *convWeightBase = w.dw_conv_weight[8 - dy * 3 - dx];
                auto *convBase       = mapAfterDWConv[dx + outerIdxBase].data();

                for (int b = 0; b < ConvBatches; b++) {
                    auto convW = simde_mm256_load_si256(
                        reinterpret_cast<const simde__m256i *>(convWeightBase + b * RegWidth));
                    auto *convPtr   = reinterpret_cast<simde__m256i *>(convBase + b * RegWidth);
                    auto  oldConvF  = simde_mm256_load_si256(convPtr);
                    auto  deltaOldF = simde_mm256_mulhrs_epi16(oldFeats[b], convW);
                    auto  deltaNewF = simde_mm256_mulhrs_epi16(newFeats[b], convW);
                    auto  newConvF  = simde_mm256_sub_epi16(oldConvF, deltaOldF);
                    newConvF        = simde_mm256_add_epi16(newConvF, deltaNewF);
                    simde_mm256_store_si256(convPtr, newConvF);
                }
            }
        }

        // Update valueSum
        DEF_BATCH256(int16_t, ValueDim, RegWidth16, ValueBatches16);
        for (int b = ConvBatches; b < MapBatches; b++) {
            auto oldv0 = simde_mm256_cvtepi16_epi32(simde_mm256_castsi256_si128(oldFeats[b]));
            auto oldv1 = simde_mm256_cvtepi16_epi32(simde_mm256_extracti128_si256(oldFeats[b], 1));
            auto newv0 = simde_mm256_cvtepi16_epi32(simde_mm256_castsi256_si128(newFeats[b]));
            auto newv1 = simde_mm256_cvtepi16_epi32(simde_mm256_extracti128_si256(newFeats[b], 1));
            vSum[2 * b + 0] = simde_mm256_sub_epi32(vSum[2 * b + 0], oldv0);
            vSum[2 * b + 1] = simde_mm256_sub_epi32(vSum[2 * b + 1], oldv1);
            vSum[2 * b + 0] = simde_mm256_add_epi32(vSum[2 * b + 0], newv0);
            vSum[2 * b + 1] = simde_mm256_add_epi32(vSum[2 * b + 1], newv1);
        }
    }

    if constexpr (UT == MOVE) {
        // Add value feature sum
        for (int yi = y0, outerIdxBase = y0 * fullBoardSize; yi <= y1;
             yi++, outerIdxBase += fullBoardSize) {
            for (int xi = x0; xi <= x1; xi++) {
                int outerIdx = xi + outerIdxBase;

                DEF_BATCH256(int16_t, DWConvDim, RegWidth, ConvBatches);
                for (int b = 0; b < ConvBatches; b++) {
                    auto convF = simde_mm256_load_si256(reinterpret_cast<simde__m256i *>(
                        mapAfterDWConv[outerIdx].data() + b * RegWidth));
                    convF      = simde_mm256_max_epi16(convF, simde_mm256_setzero_si256());  // relu

                    auto value0 = simde_mm256_cvtepi16_epi32(simde_mm256_castsi256_si128(convF));
                    auto value1 =
                        simde_mm256_cvtepi16_epi32(simde_mm256_extracti128_si256(convF, 1));
                    vSum[2 * b + 0] = simde_mm256_add_epi32(vSum[2 * b + 0], value0);
                    vSum[2 * b + 1] = simde_mm256_add_epi32(vSum[2 * b + 1], value1);
                }
            }
        }

        // Store value sum
        for (int b = 0; b < ValueBatches32; b++)
            simde_mm256_store_si256(
                reinterpret_cast<simde__m256i *>(valueSum.data() + b * RegWidth32),
                vSum[b]);
    }
    else {
        valueSum = *valueSumBoardBackup;  // just copy it
    }
}

std::tuple<float, float, float> Mix7Accumulator::evaluateValue(const Mix7Weight &w)
{
    auto scale_after_conv = simde_mm256_set1_ps(w.value_sum_scale_after_conv * boardSizeScale);
    auto scale_direct     = simde_mm256_set1_ps(w.value_sum_scale_direct * boardSizeScale);

    DEF_BATCH256(float, ValueDim, RegWidth, ValueBatches);
    DEF_BATCH256(float, DWConvDim, ConvRegWidth, ConvBatches);
    static_assert(ValueDim >= DWConvDim, "Assume ValueDim >= DWConvDim in evaluateValue()!");

    // layer 0 convert int32 to float
    alignas(Alignment) float layer0[ValueDim];
    for (int b = 0; b < ConvBatches; b++) {
        auto valueI32 = simde_mm256_load_si256(
            reinterpret_cast<const simde__m256i *>(valueSum.data() + b * RegWidth));
        auto valueF32 = simde_mm256_cvtepi32_ps(valueI32);
        valueF32      = simde_mm256_mul_ps(valueF32, scale_after_conv);
        simde_mm256_store_ps(layer0 + b * RegWidth, valueF32);
    }
    for (int b = ConvBatches; b < ValueBatches; b++) {
        auto valueI32 = simde_mm256_load_si256(
            reinterpret_cast<const simde__m256i *>(valueSum.data() + b * RegWidth));
        auto valueF32 = simde_mm256_cvtepi32_ps(valueI32);
        valueF32      = simde_mm256_mul_ps(valueF32, scale_direct);
        simde_mm256_store_ps(layer0 + b * RegWidth, valueF32);
    }

    // linear 1
    alignas(Alignment) float layer1[ValueDim];
    simd::linearLayer<simd::Activation::Relu, ValueDim, ValueDim, float, Alignment>(
        layer1,
        layer0,
        w.value_l1_weight,
        w.value_l1_bias);

    // linear 2
    alignas(Alignment) float layer2[ValueDim];
    simd::linearLayer<simd::Activation::Relu, ValueDim, ValueDim, float, Alignment>(
        layer2,
        layer1,
        w.value_l2_weight,
        w.value_l2_bias);

    // final linear
    alignas(Alignment) float value[16];
    simd::linearLayer<simd::Activation::None, 3, ValueDim, float, Alignment>(value,
                                                                             layer2,
                                                                             w.value_l3_weight,
                                                                             w.value_l3_bias);

    return {value[0], value[1], value[2]};
}

void Mix7Accumulator::evaluatePolicy(const Mix7Weight &w, PolicyBuffer &policyBuffer)
{
    for (int y = 0, innerIdx = 0, outerIdx = fullBoardSize + 1; y < boardSize; y++, outerIdx += 2)
        for (int x = 0; x < boardSize; x++, innerIdx++, outerIdx++) {
            if (!policyBuffer.getComputeFlag(innerIdx))
                continue;

            float policy = 0.0f;

            DEF_BATCH256(int16_t, PolicyDim, RegWidth, PolicyBatches);
            static_assert(PolicyDim <= DWConvDim,
                          "Assume PolicyDim <= DWConvDim in evaluatePolicy()!");
            for (int b = 0; b < PolicyBatches; b++) {
                auto feature = simde_mm256_load_si256(reinterpret_cast<const simde__m256i *>(
                    mapAfterDWConv[outerIdx].data() + b * RegWidth));
                auto convW   = simde_mm256_load_si256(
                    reinterpret_cast<const simde__m256i *>(w.policy_pw_conv_weight + b * RegWidth));
                feature = simde_mm256_max_epi16(feature, simde_mm256_setzero_si256());  // relu
                feature = simde_mm256_mulhrs_epi16(feature, convW);
                policy += static_cast<float>(
                    simd::detail::VecOp<int16_t, simd::AVX2>::reduceadd(feature));
            }

            policyBuffer(innerIdx) =
                policy * (policy < 0 ? w.policy_neg_weight : w.policy_pos_weight);
        }
}

Mix7Evaluator::Mix7Evaluator(int                   boardSize,
                             Rule                  rule,
                             std::filesystem::path blackWeightPath,
                             std::filesystem::path whiteWeightPath,
                             int                   alignBoardSize)
    : Evaluator(boardSize, rule)
    , weight {nullptr, nullptr}
    , alignBoardSize(alignBoardSize)
{
    std::filesystem::path currentWeightPath;

    CompressedWrapper<StandardHeaderParserWarpper<BinaryPODWeightLoader<Mix7Weight>>> loader(
        Compressor::Type::LZ4_DEFAULT);
    loader.setHeaderValidator([&](StandardHeader header) -> bool {
        constexpr uint32_t ArchHash =
            ArchHashBase ^ ((DWConvDim << 24) | (PolicyDim << 16) | ValueDim);
        if (header.archHash != ArchHash)
            throw IncompatibleWeightFileError("incompatible architecture in weight file.");

        if (!contains(header.supportedRules, rule))
            throw UnsupportedRuleError(rule);

        if (!contains(header.supportedBoardSizes, boardSize))
            throw UnsupportedBoardSizeError(boardSize);

        if (Config::MessageMode != MsgMode::NONE)
            MESSAGEL("mix7nnue: load weight from " << currentWeightPath);
        return true;
    });

    for (const auto &[weightSide, weightPath] : {
             std::make_pair(BLACK, blackWeightPath),
             std::make_pair(WHITE, whiteWeightPath),
         }) {
        currentWeightPath  = weightPath;
        weight[weightSide] = Mix7WeightRegistry.loadWeightFromFile(weightPath, loader);
        if (!weight[weightSide])
            throw std::runtime_error("failed to load nnue weight from " + weightPath.string());
    }

    accumulator[BLACK] = std::make_unique<Mix7Accumulator>(boardSize);
    accumulator[WHITE] = std::make_unique<Mix7Accumulator>(boardSize);

    int numCells = boardSize * boardSize;
    moveCache[BLACK].reserve(numCells);
    moveCache[WHITE].reserve(numCells);

    valueSumBoardHistory[BLACK].reserve(numCells);
    valueSumBoardHistory[WHITE].reserve(numCells);
}

Mix7Evaluator::~Mix7Evaluator()
{
    if (weight[BLACK])
        Mix7WeightRegistry.unloadWeight(weight[BLACK]);
    if (weight[WHITE])
        Mix7WeightRegistry.unloadWeight(weight[WHITE]);
}

void Mix7Evaluator::initEmptyBoard()
{
    moveCache[BLACK].clear();
    moveCache[WHITE].clear();
    valueSumBoardHistory[BLACK].clear();
    valueSumBoardHistory[WHITE].clear();
    accumulator[BLACK]->clear(*weight[BLACK], alignBoardSize);
    accumulator[WHITE]->clear(*weight[WHITE], alignBoardSize);
}

void Mix7Evaluator::beforeMove(const Board &board, Pos pos)
{
    addCache(board.sideToMove(), pos.x(), pos.y(), false);
}

void Mix7Evaluator::afterUndo(const Board &board, Pos pos)
{
    addCache(board.sideToMove(), pos.x(), pos.y(), true);
}

ValueType Mix7Evaluator::evaluateValue(const Board &board)
{
    Color self = board.sideToMove();

    // Apply all incremental update and calculate value
    clearCache(self);
    auto [win, loss, draw] = accumulator[self]->evaluateValue(*weight[self]);

    return ValueType(win, loss, draw, true);
}

void Mix7Evaluator::evaluatePolicy(const Board &board, PolicyBuffer &policyBuffer)
{
    Color self = board.sideToMove();

    // Apply all incremental update and calculate policy
    clearCache(self);
    accumulator[self]->evaluatePolicy(*weight[self], policyBuffer);

    policyBuffer.setScoreBias(300);
}

void Mix7Evaluator::clearCache(Color side)
{
    constexpr Color opponentMap[4] = {WHITE, BLACK, WALL, EMPTY};

    for (MoveCache &mc : moveCache[side]) {
        if (side == WHITE) {
            mc.oldColor = opponentMap[mc.oldColor];
            mc.newColor = opponentMap[mc.newColor];
        }

        if (mc.oldColor == EMPTY) {
            valueSumBoardHistory[side].push_back(accumulator[side]->valueSum);
            accumulator[side]->update<Mix7Accumulator::MOVE>(*weight[side],
                                                             mc.newColor,
                                                             mc.x,
                                                             mc.y,
                                                             nullptr);
        }
        else {
            accumulator[side]->update<Mix7Accumulator::UNDO>(*weight[side],
                                                             mc.oldColor,
                                                             mc.x,
                                                             mc.y,
                                                             &valueSumBoardHistory[side].back());
            valueSumBoardHistory[side].pop_back();
        }
    }
    moveCache[side].clear();
}

void Mix7Evaluator::addCache(Color side, int x, int y, bool isUndo)
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

        assert(moveCache[c].size() <= boardSize * boardSize);
    }
}

}  // namespace Evaluation::mix7
