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

#include "mix6Qnnue.h"

#include "../core/platform.h"
#include "../core/utils.h"
#include "../game/board.h"
#include "simdops.h"
#include "weightloader.h"

#include <algorithm>
#include <cmath>

namespace {

constexpr auto Power3 = []() {
    auto pow3 = std::array<int, 16> {};
    for (size_t i = 0; i < pow3.size(); i++)
        pow3[i] = power(3, i);
    return pow3;
}();

constexpr int DX[4]                  = {1, 0, 1, 1};
constexpr int DY[4]                  = {0, 1, 1, -1};
constexpr int ConvKernelOffset[9][2] = {
    {-1, -1},
    {-1, 0},
    {-1, 1},
    {0, -1},
    {0, 0},
    {0, 1},
    {1, -1},
    {1, 0},
    {1, 1},
};

/// DPower3[oldColor][newColor]
constexpr int DPower3[4][4] = {
    {0, 1, 0, -1},
    {-1, 0, 0, -2},
    {0, 0, 0, 0},
    {1, 2, 0, 0},
};

using namespace Evaluation;
using namespace Evaluation::mix6q;

static WeightRegistry<Mix6QWeight> Mix6v2WeightRegistry;

struct Mix6QWeightLoader : WeightLoader<Mix6QWeight>
{
    std::unique_ptr<Mix6QWeight> load(std::istream &in) override
    {
        auto weight = std::make_unique<Mix6QWeight>();

        int32_t numMappings;
        in.read(reinterpret_cast<char *>(&numMappings), sizeof(numMappings));
        weight->mappings.resize(numMappings);
        for (int i = 0; i < numMappings; i++) {
            in.read(reinterpret_cast<char *>(&weight->mappings[i]),
                    sizeof(Mix6QWeightMapping::Mapping));
        }

        Mix6QWeightAfterMapping *weightAfterMapping =
            static_cast<Mix6QWeightAfterMapping *>(weight.get());
        in.read(reinterpret_cast<char *>(weightAfterMapping), sizeof(Mix6QWeightAfterMapping));

        if (in && in.peek() == std::ios::traits_type::eof()) {
            preprocessWeight(*weight);
            return weight;
        }
        else
            return nullptr;
    }

    void preprocessWeight(Mix6QWeight &w)
    {
        // Shifting pw conv weight from signed to unsigned int8
        simd::add<PolicyDim, int8_t, alignof(int8_t), simd::AVX2>(w.policy_pw_conv_weight,
                                                                  w.policy_pw_conv_weight,
                                                                  (int8_t)128);
    }
};

}  // namespace

namespace Evaluation::mix6q {

Mix6QAccumulator::Mix6QAccumulator(int boardSize, int16_t mapIndex)
    : boardSize(boardSize)
    , numCells(boardSize * boardSize)
    , boardSizeScale(1.0f / numCells)
    , mapIndex(mapIndex)
    , boardSizeFactor(ScaleBoardFactor / numCells)
{
    shapeTable        = new std::array<uint32_t, 4>[numCells];
    featureSum        = new std::array<int16_t, FeatureDim>[numCells];
    featureAfterPReLU = new std::array<int8_t, FeatureDim>[numCells];
    policyAfterDWConv = new std::array<int16_t, PolicyDim>[numCells];
}

Mix6QAccumulator::~Mix6QAccumulator()
{
    delete[] shapeTable;
    delete[] featureSum;
    delete[] featureAfterPReLU;
    delete[] policyAfterDWConv;
}

void Mix6QAccumulator::initShapeTable()
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
}

void Mix6QAccumulator::clear(const Mix6QWeight &w, int alignBoardSize)
{
    // Clear and init shape table
    initShapeTable();

    // Init featureSum, featureAfterPReLU, policyAfterDWConv, valueSumBoard
    for (int cellIdx = 0; cellIdx < numCells; cellIdx++) {
        simd::copy<PolicyDim, int16_t, alignof(int16_t)>(policyAfterDWConv[cellIdx].data(),
                                                         w.policy_dw_conv_bias);
    }
    simd::zero<ValueDim, int16_t, alignof(int16_t)>(valueSumBoard.data());

    for (int cellIdx = 0; cellIdx < numCells; cellIdx++) {
        int y = cellIdx / boardSize, x = cellIdx % boardSize;

        DEF_BATCH256(int8_t, FeatureDim, FeatRegWidth, FeatBatches);
        DEF_BATCH256(int8_t, PolicyDim, PolicyRegWidth, PolicyBatches);
        DEF_BATCH256(int8_t, ValueDim, ValueRegWidth, ValueBatches);
        simde__m256i featureRegs[FeatBatches];

        // Init featureSum, featureAfterPReLU
        {
            DEF_BATCH256(int16_t, FeatureDim, FeatI16RegWidth, FeatureI16Batches);
            simde__m256i featI16Regs[FeatureI16Batches];
            const auto  &mapping = w.mappings[mapIndex];

            // Init featureSum
            for (int i = 0; i < FeatBatches; i++) {
                featI16Regs[i * 2]     = simde_mm256_setzero_si256();
                featI16Regs[i * 2 + 1] = simde_mm256_setzero_si256();

                for (int dir = 0; dir < 4; dir++) {
                    uint32_t shapeIdx = shapeTable[cellIdx][dir];
                    auto f = simde_mm256_loadu_si256(mapping.feature[shapeIdx] + i * FeatRegWidth);
                    auto [f0i16, f1i16]    = simd::regop::unpackI8ToI16(f);
                    featI16Regs[i * 2]     = simde_mm256_add_epi16(featI16Regs[i * 2], f0i16);
                    featI16Regs[i * 2 + 1] = simde_mm256_add_epi16(featI16Regs[i * 2 + 1], f1i16);
                }

                simde_mm256_storeu_si256(featureSum[cellIdx].data() + (i * 2) * FeatI16RegWidth,
                                         featI16Regs[i * 2]);
                simde_mm256_storeu_si256(featureSum[cellIdx].data() + (i * 2 + 1) * FeatI16RegWidth,
                                         featI16Regs[i * 2 + 1]);
            }

            simd::debug::assertInRange<int16_t>(featI16Regs, -508, 508);

            // Init featureAfterPReLU
            simd::regop::prelu16<FeatureDim, 64>(featI16Regs, w.prelu_weight);

            simd::debug::assertInRange<int16_t>(featI16Regs, -508, 508);

            for (int i = 0; i < FeatBatches; i++) {
                featureRegs[i] = simd::regop::divideAndPackI16ToI8<4>(featI16Regs[i * 2],
                                                                      featI16Regs[i * 2 + 1]);

                simde_mm256_storeu_si256(featureAfterPReLU[cellIdx].data() + i * FeatRegWidth,
                                         featureRegs[i]);
            }

            simd::debug::assertInRange<int8_t>(featureRegs, -127, 127);
        }

        // Init policyAfterDWConv
        {
            DEF_BATCH256(int16_t, PolicyDim, PolicyAccRegWidth, PolicyAccBatches);

            for (auto [dy, dx] : ConvKernelOffset) {
                int yi = y + dy, xi = x + dx;
                if (yi < 0 || yi >= boardSize || xi < 0 || xi >= boardSize)
                    continue;  // zero padding

                // adding policy feature to neighbor 3x3 cells
                auto pBase = policyAfterDWConv[boardSize * yi + xi].data();
                for (int i = 0; i < PolicyBatches; i++) {
                    auto pAddr = pBase + (i * 2) * PolicyAccRegWidth;
                    auto convW = simde_mm256_loadu_si256(w.policy_dw_conv_weight[4 - dy * 3 - dx]
                                                         + i * PolicyRegWidth);
                    auto p0    = simde_mm256_loadu_si256(pAddr);
                    auto p1    = simde_mm256_loadu_si256(pAddr + PolicyAccRegWidth);

                    auto [convY0, convY1] = simd::regop::mulI8(featureRegs[i], convW);
                    convY0                = simde_mm256_srai_epi16(convY0, 4);  // div 16
                    convY1                = simde_mm256_srai_epi16(convY1, 4);  // div 16

                    p0 = simde_mm256_add_epi16(p0, convY0);
                    p1 = simde_mm256_add_epi16(p1, convY1);

                    simde_mm256_storeu_si256(pAddr, p0);
                    simde_mm256_storeu_si256(pAddr + PolicyAccRegWidth, p1);

                    simd::debug::assertInRange<int16_t>(convY0, -1008, 1008);
                    simd::debug::assertInRange<int16_t>(convY1, -1008, 1008);
                    simd::debug::assertInRange<int16_t>(p0, -29072, 29072);
                    simd::debug::assertInRange<int16_t>(p1, -29072, 29072);
                }
            }
        }

        // Init valueSumBoard
        {
            DEF_BATCH256(int16_t, ValueDim, ValueSumRegWidth, ValueSumBatches);
            const int    ValueBatchOffset = PolicyBatches;
            simde__m256i valueSum[2];

            for (int i = 0; i < ValueBatches; i++) {
                // unpack i8 to i16, then divide by 2 to avoid overflow valueSumBoard
                auto [v0, v1] = simd::regop::unpackI8ToI16(featureRegs[ValueBatchOffset + i]);
                v0            = simde_mm256_srai_epi16(v0, 1);
                v1            = simde_mm256_srai_epi16(v1, 1);

                simd::debug::assertInRange<int16_t>(v0, -63, 63);
                simd::debug::assertInRange<int16_t>(v1, -63, 63);

                auto vAddr  = valueSumBoard.data() + (i * 2) * ValueSumRegWidth;
                valueSum[0] = simde_mm256_loadu_si256(vAddr);
                valueSum[1] = simde_mm256_loadu_si256(vAddr + ValueSumRegWidth);

                valueSum[0] = simde_mm256_add_epi16(valueSum[0], v0);
                valueSum[1] = simde_mm256_add_epi16(valueSum[1], v1);

                simde_mm256_storeu_si256(vAddr, valueSum[0]);
                simde_mm256_storeu_si256(vAddr + ValueSumRegWidth, valueSum[1]);

                simd::debug::assertInRange<int16_t>(valueSum,
                                                    -63 * boardSize * boardSize,
                                                    63 * boardSize * boardSize);
            }
        }
    }

    // Align valueSumBoard to base board size version
    if (alignBoardSize) {
        Mix6QAccumulator accmulatorToAlign(alignBoardSize, mapIndex);
        accmulatorToAlign.clear(w);
        valueSumBoard = accmulatorToAlign.valueSumBoard;
    }
}

void Mix6QAccumulator::update(const Mix6QWeight &w, Color oldColor, Color newColor, int x, int y)
{
    struct OnePointChange
    {
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
            if (xi < 0 || xi >= boardSize || yi < 0 || yi >= boardSize)
                continue;

            OnePointChange &c        = changeTable[changeCount++];
            c.dir                    = dir;
            c.index                  = boardSize * yi + xi;
            c.oldShape               = shapeTable[c.index][dir];
            c.newShape               = c.oldShape + dPower3 * Power3[dist + 5];
            shapeTable[c.index][dir] = c.newShape;
            assert(0 <= c.newShape && c.newShape < ShapeNum);
        }
    }

    // Incremental update policy feature and value feature sum
    for (int cIdx = 0; cIdx < changeCount; cIdx++) {
        OnePointChange &c  = changeTable[cIdx];
        int             y0 = c.index / boardSize, x0 = c.index % boardSize;

        DEF_BATCH256(int8_t, FeatureDim, FeatRegWidth, FeatBatches);
        DEF_BATCH256(int8_t, PolicyDim, PolicyRegWidth, PolicyBatches);
        DEF_BATCH256(int8_t, ValueDim, ValueRegWidth, ValueBatches);
        simde__m256i oldFeatRegs[FeatBatches], newFeatRegs[FeatBatches];

        // Update featureSum, featureAfterPReLU
        {
            DEF_BATCH256(int16_t, FeatureDim, FeatI16RegWidth, FeatureI16Batches);
            simde__m256i featI16Regs[FeatureI16Batches];
            const auto  &mapping = w.mappings[mapIndex];

            // Update featureSum
            for (int i = 0; i < FeatBatches; i++) {
                auto oldf = simde_mm256_loadu_si256(mapping.feature[c.oldShape] + i * FeatRegWidth);
                auto newf = simde_mm256_loadu_si256(mapping.feature[c.newShape] + i * FeatRegWidth);

                auto [oldf0i16, oldf1i16] = simd::regop::unpackI8ToI16(oldf);
                auto [newf0i16, newf1i16] = simd::regop::unpackI8ToI16(newf);

                auto featSumAddr       = featureSum[c.index].data() + (i * 2) * FeatI16RegWidth;
                featI16Regs[i * 2]     = simde_mm256_loadu_si256(featSumAddr);
                featI16Regs[i * 2 + 1] = simde_mm256_loadu_si256(featSumAddr + FeatI16RegWidth);

                featI16Regs[i * 2]     = simde_mm256_sub_epi16(featI16Regs[i * 2], oldf0i16);
                featI16Regs[i * 2 + 1] = simde_mm256_sub_epi16(featI16Regs[i * 2 + 1], oldf1i16);
                featI16Regs[i * 2]     = simde_mm256_add_epi16(featI16Regs[i * 2], newf0i16);
                featI16Regs[i * 2 + 1] = simde_mm256_add_epi16(featI16Regs[i * 2 + 1], newf1i16);

                simde_mm256_storeu_si256(featSumAddr, featI16Regs[i * 2]);
                simde_mm256_storeu_si256(featSumAddr + FeatI16RegWidth, featI16Regs[i * 2 + 1]);
            }

            simd::debug::assertInRange<int16_t>(featI16Regs, -508, 508);

            // Update featureAfterPReLU
            simd::regop::prelu16<FeatureDim, 64>(featI16Regs, w.prelu_weight);

            simd::debug::assertInRange<int16_t>(featI16Regs, -508, 508);

            for (int i = 0; i < FeatBatches; i++) {
                auto fAddr     = featureAfterPReLU[c.index].data() + i * FeatRegWidth;
                oldFeatRegs[i] = simde_mm256_loadu_si256(fAddr);
                newFeatRegs[i] = simd::regop::divideAndPackI16ToI8<4>(featI16Regs[i * 2],
                                                                      featI16Regs[i * 2 + 1]);

                simde_mm256_storeu_si256(fAddr, newFeatRegs[i]);
            }

            simd::debug::assertInRange<int8_t>(oldFeatRegs, -127, 127);
            simd::debug::assertInRange<int8_t>(newFeatRegs, -127, 127);
        }

        // Update policyAfterDWConv
        {
            DEF_BATCH256(int16_t, PolicyDim, PolicyAccRegWidth, PolicyAccBatches);

            for (auto [dy, dx] : ConvKernelOffset) {
                int yi = y0 + dy, xi = x0 + dx;
                if (yi < 0 || yi >= boardSize || xi < 0 || xi >= boardSize)
                    continue;  // zero padding

                // adding policy feature delta to neighbor 3x3 cells
                auto pBase = policyAfterDWConv[boardSize * yi + xi].data();
                for (int i = 0; i < PolicyBatches; i++) {
                    auto pAddr = pBase + (i * 2) * PolicyAccRegWidth;
                    auto convW = simde_mm256_loadu_si256(w.policy_dw_conv_weight[4 - dy * 3 - dx]
                                                         + i * PolicyRegWidth);
                    auto p0    = simde_mm256_loadu_si256(pAddr);
                    auto p1    = simde_mm256_loadu_si256(pAddr + PolicyAccRegWidth);

                    auto [oldConvY0, oldConvY1] = simd::regop::mulI8(oldFeatRegs[i], convW);
                    auto [newConvY0, newConvY1] = simd::regop::mulI8(newFeatRegs[i], convW);
                    oldConvY0                   = simde_mm256_srai_epi16(oldConvY0, 4);  // div 16
                    oldConvY1                   = simde_mm256_srai_epi16(oldConvY1, 4);  // div 16
                    newConvY0                   = simde_mm256_srai_epi16(newConvY0, 4);  // div 16
                    newConvY1                   = simde_mm256_srai_epi16(newConvY1, 4);  // div 16
                    p0                          = simde_mm256_sub_epi16(p0, oldConvY0);
                    p1                          = simde_mm256_sub_epi16(p1, oldConvY1);
                    p0                          = simde_mm256_add_epi16(p0, newConvY0);
                    p1                          = simde_mm256_add_epi16(p1, newConvY1);

                    simde_mm256_storeu_si256(pAddr, p0);
                    simde_mm256_storeu_si256(pAddr + PolicyAccRegWidth, p1);

                    simd::debug::assertInRange<int16_t>(oldConvY0, -1008, 1008);
                    simd::debug::assertInRange<int16_t>(oldConvY1, -1008, 1008);
                    simd::debug::assertInRange<int16_t>(newConvY0, -1008, 1008);
                    simd::debug::assertInRange<int16_t>(newConvY1, -1008, 1008);
                    simd::debug::assertInRange<int16_t>(p0, -29072, 29072);
                    simd::debug::assertInRange<int16_t>(p1, -29072, 29072);
                }
            }
        }

        // Update valueSumBoard
        {
            DEF_BATCH256(int16_t, ValueDim, ValueSumRegWidth, ValueSumBatches);
            const int    ValueBatchOffset = PolicyBatches;
            simde__m256i valueSum[2];

            for (int i = 0; i < ValueBatches; i++) {
                // unpack i8 to i16, then divide by 2 to avoid overflow valueSumBoard
                auto [oldv0i16, oldv1i16] =
                    simd::regop::unpackI8ToI16(oldFeatRegs[ValueBatchOffset + i]);
                auto [newv0i16, newv1i16] =
                    simd::regop::unpackI8ToI16(newFeatRegs[ValueBatchOffset + i]);
                oldv0i16 = simde_mm256_srai_epi16(oldv0i16, 1);
                oldv1i16 = simde_mm256_srai_epi16(oldv1i16, 1);
                newv0i16 = simde_mm256_srai_epi16(newv0i16, 1);
                newv1i16 = simde_mm256_srai_epi16(newv1i16, 1);

                simd::debug::assertInRange<int16_t>(oldv0i16, -63, 63);
                simd::debug::assertInRange<int16_t>(oldv1i16, -63, 63);
                simd::debug::assertInRange<int16_t>(newv0i16, -63, 63);
                simd::debug::assertInRange<int16_t>(newv1i16, -63, 63);

                auto vAddr  = valueSumBoard.data() + (i * 2) * ValueSumRegWidth;
                valueSum[0] = simde_mm256_loadu_si256(vAddr);
                valueSum[1] = simde_mm256_loadu_si256(vAddr + ValueSumRegWidth);

                valueSum[0] = simde_mm256_sub_epi16(valueSum[0], oldv0i16);
                valueSum[1] = simde_mm256_sub_epi16(valueSum[1], oldv1i16);
                valueSum[0] = simde_mm256_add_epi16(valueSum[0], newv0i16);
                valueSum[1] = simde_mm256_add_epi16(valueSum[1], newv1i16);

                simde_mm256_storeu_si256(vAddr, valueSum[0]);
                simde_mm256_storeu_si256(vAddr + ValueSumRegWidth, valueSum[1]);

                simd::debug::assertInRange<int16_t>(valueSum,
                                                    -63 * boardSize * boardSize,
                                                    63 * boardSize * boardSize);
            }
        }
    }
}

std::tuple<float, float, float> Mix6QAccumulator::evaluateValue(const Mix6QWeight &w)
{
    // layer 0 Global Mean + PReLU
    int8_t layer0[ValueDim];
    {
        DEF_BATCH256(int8_t, ValueDim, ValueRegWidth, ValueNumBatches);
        DEF_BATCH256(int16_t, ValueDim, ValueSumRegWidth, ValueSumNumBatches);
        const auto bsFactor = simde_mm256_set1_epi16(boardSizeFactor);

        for (int i = 0; i < ValueNumBatches; i++) {
            simde__m256i valueI16Reg[2];
            auto         vAddr = valueSumBoard.data() + (i * 2) * ValueSumRegWidth;
            valueI16Reg[0]     = simde_mm256_loadu_si256(vAddr);
            valueI16Reg[1]     = simde_mm256_loadu_si256(vAddr + ValueSumRegWidth);

            simd::debug::assertInRange<int16_t>(valueI16Reg,
                                                -63 * boardSize * boardSize,
                                                63 * boardSize * boardSize);

            // 1. div 8  2. mul 3600/(bs*bs)  3. div 256  4. to int8
            valueI16Reg[0] = simde_mm256_srai_epi16(valueI16Reg[0], 3);
            valueI16Reg[1] = simde_mm256_srai_epi16(valueI16Reg[1], 3);
            valueI16Reg[0] = simde_mm256_mullo_epi16(valueI16Reg[0], bsFactor);
            valueI16Reg[1] = simde_mm256_mullo_epi16(valueI16Reg[1], bsFactor);

            simd::debug::assertInRange<int16_t>(valueI16Reg, -28350, 28350);

            auto valueI8 = simd::regop::divideAndPackI16ToI8<256>(valueI16Reg[0], valueI16Reg[1]);

            simd::debug::assertInRange<int8_t>(valueI8, -110, 110);

            // apply i8 relu
            valueI8 = simde_mm256_max_epi8(valueI8, simde_mm256_setzero_si256());

            simd::debug::assertInRange<int8_t>(valueI8, 0, 110);

            simde_mm256_storeu_si256(layer0 + i * ValueRegWidth, valueI8);
        }
    }

    // layer 1 Linear1
    int32_t layer1i32[ValueDim];
    int8_t  layer1[ValueDim];
    simd::linear<ValueDim, ValueDim, ScaleWeight, alignof(int8_t), simd::AVX2>(
        layer1i32,
        layer0,
        w.value_linear1_weight,
        w.value_linear1_bias);
    simd::crelu32<ValueDim, alignof(int8_t), simd::AVX2>(layer1, layer1i32);

    // layer 2 Linear2
    int32_t layer2i32[ValueDim];
    int8_t  layer2[ValueDim];
    simd::linear<ValueDim, ValueDim, ScaleWeight, alignof(int8_t), simd::AVX2>(
        layer2i32,
        layer1,
        w.value_linear2_weight,
        w.value_linear2_bias);
    simd::crelu32<ValueDim, alignof(int8_t), simd::AVX2>(layer2, layer2i32);

    // layer 3 Linear3
    int32_t layer3i32[4];
    simd::linear<4, ValueDim, ScaleWeight, alignof(int8_t), simd::AVX2>(layer3i32,
                                                                        layer2,
                                                                        w.value_linear3_weight,
                                                                        w.value_linear3_bias);

    return {
        layer3i32[0] * w.value_output_scale,
        layer3i32[1] * w.value_output_scale,
        layer3i32[2] * w.value_output_scale,
    };
}

void Mix6QAccumulator::evaluatePolicy(const Mix6QWeight &w, PolicyBuffer &policyBuffer)
{
    DEF_BATCH256(int8_t, PolicyDim, PolicyRegWidth, PolicyBatches);
    DEF_BATCH256(int16_t, PolicyDim, PolicyI16RegWidth, PolicyI16Batches);
    static_assert(ScaleWeight % 16 == 0, "ScaleWeight must be a multiply of 16");
    constexpr int Log2DivisorAfterDWConv = floorLog2(ScaleWeight / 16);
    constexpr int Log2ScaleWeight        = floorLog2(ScaleWeight);

    for (int cellIdx = 0; cellIdx < numCells; cellIdx++) {
        if (!policyBuffer.getComputeFlag(cellIdx))
            continue;

        auto pBase     = policyAfterDWConv[cellIdx].data();
        int  policySum = 0;

        for (int i = 0; i < PolicyBatches; i++) {
            simde__m256i policyI16Regs[PolicyI16Batches];
            auto         pAddr = pBase + (i * 2) * PolicyI16RegWidth;
            policyI16Regs[0]   = simde_mm256_loadu_si256(pAddr);
            policyI16Regs[1]   = simde_mm256_loadu_si256(pAddr + PolicyI16RegWidth);

            simd::debug::assertInRange<int16_t>(policyI16Regs, -29072, 29072);

            // apply depth-wise conv lrelu/16
            simd::regop::lrelu16<PolicyI16RegWidth * 2, 16>(policyI16Regs);

            simd::debug::assertInRange<int16_t>(policyI16Regs, -29072, 29072);

            auto pwConvW = simde_mm256_loadu_si256(w.policy_pw_conv_weight + i * PolicyBatches);
            // 1. div (64 / 16)  2. clamp to int8
            auto policy = simd::regop::divideAndPackI16ToI8<ScaleWeight / 16>(policyI16Regs[0],
                                                                              policyI16Regs[1]);

            simd::debug::assertInRange<int8_t>(policy, -127, 127);
            simd::debug::assertInRange<uint8_t>(pwConvW, 1, 255);

            // Point-wise conv
            policy = simde_mm256_sub_epi8(
                simde_mm256_maddubs_epi16(pwConvW, policy),
                simde_mm256_maddubs_epi16(simde_mm256_set1_epi8(-128), policy));

            simd::debug::assertInRange<int16_t>(policy, -32258, 32258);

            policy = simde_mm256_srai_epi16(policy, Log2ScaleWeight);  // div ScaleFeature

            simd::debug::assertInRange<int16_t>(policy, -504, 504);

            auto partialPolicySum = simd::detail::VecOp<int16_t, simd::AVX2>::reduceadd(policy);

            assert(-8064 <= partialPolicySum && partialPolicySum <= 8064);

            policySum += partialPolicySum;
        }

        // apply point-wise conv lrelu/16
        int policyFinal       = std::max(policySum, policySum / 16);
        policyBuffer(cellIdx) = w.policy_output_scale * policyFinal;
    }
}

Mix6QEvaluator::Mix6QEvaluator(int                   boardSize,
                               Rule                  rule,
                               std::filesystem::path weightPath,
                               int                   alignBoardSize)
    : Evaluator(boardSize, rule)
    , weight(nullptr)
    , alignBoardSize(alignBoardSize)
{
    CompressedWrapper<StandardHeaderParserWarpper<Mix6QWeightLoader>> loader(
        Compressor::Type::LZ4_DEFAULT);
    loader.setHeaderValidator([&](StandardHeader header) -> bool {
        if (header.archHash != ArchHash)
            throw IncompatibleWeightFileError("incompatible architecture in weight file.");

        if (!contains(header.supportedRules, rule))
            throw UnsupportedRuleError(rule);

        if (!contains(header.supportedBoardSizes, boardSize))
            throw UnsupportedBoardSizeError(boardSize);

        MESSAGEL("mix6qnnue: load weight from " << weightPath);
        return true;
    });
    weight = Mix6v2WeightRegistry.loadWeightFromFile(weightPath, loader);
    if (!weight)
        throw std::runtime_error("failed to load mix6v2nnue weight from " + weightPath.string());

    int mapIndexBlack  = 0;
    int mapIndexWhite  = weight->mappings.size() > 1;
    accumulator[BLACK] = std::make_unique<Mix6QAccumulator>(boardSize, mapIndexBlack);
    accumulator[WHITE] = std::make_unique<Mix6QAccumulator>(boardSize, mapIndexWhite);

    int numCells = boardSize * boardSize;
    moveCache[BLACK].reserve(numCells);
    moveCache[WHITE].reserve(numCells);
}

Mix6QEvaluator::~Mix6QEvaluator()
{
    if (weight)
        Mix6v2WeightRegistry.unloadWeight(weight);
}

void Mix6QEvaluator::initEmptyBoard()
{
    moveCache[BLACK].clear();
    moveCache[WHITE].clear();
    accumulator[BLACK]->clear(*weight, alignBoardSize);
    accumulator[WHITE]->clear(*weight, alignBoardSize);
}

void Mix6QEvaluator::beforeMove(const Board &board, Pos pos)
{
    addCache(board.sideToMove(), pos.x(), pos.y(), false);
}

void Mix6QEvaluator::afterUndo(const Board &board, Pos pos)
{
    addCache(board.sideToMove(), pos.x(), pos.y(), true);
}

ValueType Mix6QEvaluator::evaluateValue(const Board &board)
{
    Color self = board.sideToMove();

    // Apply all incremental update and calculate value
    clearCache(self);
    auto [win, loss, draw] = accumulator[self]->evaluateValue(*weight);

    return ValueType(win, loss, draw, true);
}

void Mix6QEvaluator::evaluatePolicy(const Board &board, PolicyBuffer &policyBuffer)
{
    Color self = board.sideToMove();

    // Apply all incremental update and calculate policy
    clearCache(self);
    accumulator[self]->evaluatePolicy(*weight, policyBuffer);

    policyBuffer.setScoreBias(300);
}

void Mix6QEvaluator::clearCache(Color side)
{
    constexpr Color opponentMap[4] = {WHITE, BLACK, WALL, EMPTY};

    for (MoveCache &mc : moveCache[side]) {
        if (side == WHITE) {
            mc.oldColor = opponentMap[mc.oldColor];
            mc.newColor = opponentMap[mc.newColor];
        }
        accumulator[side]->update(*weight, mc.oldColor, mc.newColor, mc.x, mc.y);
    }
    moveCache[side].clear();
}

void Mix6QEvaluator::addCache(Color side, int x, int y, bool isUndo)
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

}  // namespace Evaluation::mix6q
