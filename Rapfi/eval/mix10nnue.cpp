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

#include "mix10nnue.h"

#include "mixcommon.h"

#include "../core/compressor.h"
#include "../core/iohelper.h"
#include "../core/platform.h"
#include "../core/utils.h"
#include "../game/board.h"
#include "simdops.h"
#include "weightloader.h"

#include <algorithm>
#include <cmath>
#include <cstring>


namespace Evaluation::mix10 {

using namespace Evaluation;
using namespace Evaluation::mixnet;

constexpr uint32_t ArchHashBase  = 0xb53b0258;
constexpr int      FeatureDim    = 64;
constexpr int      FeatDWConvDim = 32;
constexpr int      ValueDim      = 64;
constexpr int      NumHeadBucket = 1;

static_assert(FeatDWConvDim <= FeatureDim);
static_assert(ValueDim <= FeatureDim);

constexpr int PolicySInDim  = std::max(FeatDWConvDim / 2, 16);
constexpr int PolicySOutDim = std::max(FeatDWConvDim / 4, 16);
constexpr int PolicyLInDim  = std::max(FeatDWConvDim, 16);
constexpr int PolicyLMidDim = std::max(FeatDWConvDim / 2, 16);
constexpr int PolicyLOutDim = std::max(FeatDWConvDim / 4, 16);

struct alignas(64) Weight
{
    // 1  mapping layer
    int16_t mapping[2][ShapeNum][FeatureDim];

    // 2  Depthwise conv
    int16_t feature_dwconv_weight[9][FeatDWConvDim];
    int16_t feature_dwconv_bias[FeatDWConvDim];

    struct HeadBucket
    {
        // 3  Small Policy dynamic pointwise conv
        FCWeight<ValueDim, ValueDim>                          policy_small_pwconv_weight_1;
        FCWeight<PolicySOutDim *(PolicySInDim + 1), ValueDim> policy_small_pwconv_weight_2;

        // 4  Large Policy dynamic pointwise conv
        FCWeight<ValueDim, ValueDim>                           policy_large_pwconv_weight_0;
        FCWeight<PolicyLMidDim *(PolicyLInDim + 1), ValueDim>  policy_large_pwconv_weight_1;
        FCWeight<PolicyLOutDim *(PolicyLMidDim + 1), ValueDim> policy_large_pwconv_weight_2;

        // 5  Small Value Head MLP (layer 1,2,3)
        FCWeight<ValueDim, FeatureDim> value_small_l1;
        FCWeight<ValueDim, ValueDim>   value_small_l2;
        FCWeight<4, ValueDim>          value_small_l3;

        // 6  Large Value Gate & Group MLP
        char                               __padding_to_64bytes_0[48];
        FCWeight<FeatureDim * 2, ValueDim> value_gate;
        FCWeight<ValueDim, FeatureDim>     value_corner;
        FCWeight<ValueDim, FeatureDim>     value_edge;
        FCWeight<ValueDim, FeatureDim>     value_center;
        FCWeight<ValueDim, ValueDim>       value_quad;

        // 7  Large Value Head MLP (layer 1,2,3)
        FCWeight<ValueDim, ValueDim * 5> value_l1;
        FCWeight<ValueDim, ValueDim>     value_l2;
        FCWeight<4, ValueDim>            value_l3;

        // 8  Policy output linear
        char  __padding_to_64bytes_1[48];
        float policy_small_output_weight[PolicySOutDim];
        float policy_large_output_weight[PolicyLOutDim];
        float policy_small_output_bias;
        float policy_large_output_bias;

        char __padding_to_64bytes_2[56];
    } buckets[NumHeadBucket];
};

// Make sure we have proper alignment for SIMD operations
static_assert(offsetof(Weight, feature_dwconv_weight) % 64 == 0);
static_assert(offsetof(Weight, buckets) % 64 == 0);
static_assert(offsetof(Weight::HeadBucket, value_gate) % 64 == 0);
static_assert(offsetof(Weight::HeadBucket, policy_small_output_weight) % 64 == 0);
static_assert(offsetof(Weight::HeadBucket, policy_large_output_weight) % 64 == 0);
static_assert(sizeof(Weight::HeadBucket) % 64 == 0);

struct alignas(64) ValueSumType
{
    static constexpr int NGroup = 3;

    std::array<int32_t, FeatureDim> global;
    std::array<int32_t, FeatureDim> group[NGroup][NGroup];
    std::array<int8_t, ValueDim>    small_value_feature;
    std::array<int8_t, ValueDim>    large_value_feature;
    bool                            small_value_feature_valid;
    bool                            large_value_feature_valid;
};

// Make sure we have proper alignment for SIMD operations
static_assert(offsetof(ValueSumType, global) % 64 == 0);
static_assert(offsetof(ValueSumType, group) % 64 == 0);
static_assert(offsetof(ValueSumType, small_value_feature) % 64 == 0);
static_assert(offsetof(ValueSumType, large_value_feature) % 64 == 0);

class Accumulator : private MixAccumulatorState<ValueSumType, FeatureDim, FeatDWConvDim>
{
public:
    explicit Accumulator(int boardSize) : MixAccumulatorState(boardSize) {}

    /// Init accumulator state to empty board.
    void clear(const Weight &w);
    /// Incrementally update the network state for a stone of pieceColor placed at (x, y).
    void move(const Weight &w, Color pieceColor, int x, int y);
    using MixAccumulatorState::undo;

    void updateSharedSmallHead(const Weight &w);
    void updateSharedLargeHead(const Weight &w);

    /// Calculate value (win/loss/draw) and (relative) uncertainty of current network state.
    std::tuple<float, float, float, float> evaluateValueSmall(const Weight &w);
    /// Calculate value (win/loss/draw) and (relative) uncertainty of current network state.
    std::tuple<float, float, float, float> evaluateValueLarge(const Weight &w);
    /// Calculate policy value of current network state.
    void evaluatePolicySmall(const Weight &w, PolicyBuffer &policyBuffer);
    /// Calculate policy value of current network state.
    void evaluatePolicyLarge(const Weight &w, PolicyBuffer &policyBuffer);

private:
    int getBucketIndex() { return 0; }
};

}  // namespace Evaluation::mix10

namespace {

using namespace Evaluation::mix10;

struct Mix10WeightLoader : WeightLoader<mix10::Weight>
{
    LargePagePtr<Weight> load(std::istream &in, Evaluation::EmptyLoadArgs args)
    {
        auto w = make_unique_large_page<Weight>();

        read_compressed_mapping(in, *w);
        in.read(reinterpret_cast<char *>(&w->feature_dwconv_weight[0][0]),
                sizeof(w->feature_dwconv_weight));
        in.read(reinterpret_cast<char *>(&w->feature_dwconv_bias[0]),
                sizeof(w->feature_dwconv_bias));
        for (int headIdx = 0; headIdx < NumHeadBucket; headIdx++)
            in.read(reinterpret_cast<char *>(&w->buckets[headIdx]), sizeof(w->buckets[headIdx]));

        if (in && in.peek() == std::ios::traits_type::eof()) {
            preprocess(*w);
            return w;
        }
        else
            return nullptr;
    }

    void read_compressed_mapping(std::istream &in, Weight &w)
    {
        // Both mapping tables are encoded as one continuous 10-bit packed stream
        PackedSignedIntStream<10> stream;
        for (int mappingIdx = 0; mappingIdx < arraySize(w.mapping); mappingIdx++) {
            auto &mapping = w.mapping[mappingIdx];

            for (int i = 0; i < ShapeNum; i++)
                for (int j = 0; j < FeatureDim; j++)
                    mapping[i][j] = stream.next(in);
        }
    }

    void preprocess(Weight &w)
    {
        for (int bucketIdx = 0; bucketIdx < NumHeadBucket; bucketIdx++) {
            auto &b = w.buckets[bucketIdx];
            simd::preprocessLinear<ValueDim, ValueDim>(b.policy_small_pwconv_weight_1.weight);
            simd::preprocessDynamicWeightLinear<PolicySOutDim, PolicySInDim, int16_t, ValueDim, 0>(
                b.policy_small_pwconv_weight_2.weight,
                b.policy_small_pwconv_weight_2.bias);
            simd::preprocessLinear<PolicySOutDim *(PolicySInDim + 1), ValueDim>(
                b.policy_small_pwconv_weight_2.weight);

            simd::preprocessLinear<ValueDim, ValueDim>(b.policy_large_pwconv_weight_0.weight);
            simd::preprocessDynamicWeightLinear<PolicyLMidDim, PolicyLInDim, int16_t, ValueDim, 0>(
                b.policy_large_pwconv_weight_1.weight,
                b.policy_large_pwconv_weight_1.bias);
            simd::preprocessLinear<PolicyLMidDim *(PolicyLInDim + 1), ValueDim>(
                b.policy_large_pwconv_weight_1.weight);
            simd::preprocessDynamicWeightLinear<PolicyLOutDim, PolicyLMidDim, int16_t, ValueDim, 0>(
                b.policy_large_pwconv_weight_2.weight,
                b.policy_large_pwconv_weight_2.bias);
            simd::preprocessLinear<PolicyLOutDim *(PolicyLMidDim + 1), ValueDim>(
                b.policy_large_pwconv_weight_2.weight);

            simd::preprocessLinear<ValueDim, FeatureDim>(b.value_small_l1.weight);
            simd::preprocessLinear<ValueDim, ValueDim>(b.value_small_l2.weight);
            simd::preprocessLinear<4, ValueDim>(b.value_small_l3.weight);

            simd::preprocessLinear<FeatureDim * 2, ValueDim>(b.value_gate.weight);
            simd::preprocessLinear<ValueDim, FeatureDim>(b.value_corner.weight);
            simd::preprocessLinear<ValueDim, FeatureDim>(b.value_edge.weight);
            simd::preprocessLinear<ValueDim, FeatureDim>(b.value_center.weight);
            simd::preprocessLinear<ValueDim, ValueDim>(b.value_quad.weight);

            simd::preprocessLinear<ValueDim, ValueDim * 5>(b.value_l1.weight);
            simd::preprocessLinear<ValueDim, ValueDim>(b.value_l2.weight);
            simd::preprocessLinear<4, ValueDim>(b.value_l3.weight);
        }
    }
};

static Evaluation::WeightRegistry<StandardHeaderLoader<Mix10WeightLoader>> WeightReg;

template <bool SignedInput = false,
          bool NoReLU      = false,
          int  Divisor     = 128,
          int  OutSize     = 0,
          int  InSize      = 0>
inline void
linearBlock(int8_t output[], const int8_t input[], const FCWeight<OutSize, InSize> &layerWeight)
{
    alignas(Alignment) int32_t outputi32[OutSize];
    simd::linear<OutSize, InSize, SignedInput>(outputi32,
                                               input,
                                               layerWeight.weight,
                                               layerWeight.bias);

    constexpr auto InstType = getInstTypeOfWidth(IT, 8 * OutSize);
    static_assert(InstType != simd::InstructionType::SCALAR,
                  "Failed to find a supported instruction set");
    simd::crelu<OutSize, Divisor, NoReLU, Alignment, InstType>(output, outputi32);
}

}  // namespace

namespace Evaluation::mix10 {

void Accumulator::clear(const Weight &w)
{
    if (currentVersion == -1) {
        // Init mapConv to bias
        for (int i = 0; i < outerBoardSize * outerBoardSize; i++)
            simd::copy<FeatDWConvDim>(mapConv[i].data(), w.feature_dwconv_bias);
        // Init valueSum to zeros
        auto &valueSum = valueSumTable[0];
        simd::zero<FeatureDim>(valueSum.global.data());
        for (int i = 0; i < ValueSumType::NGroup; i++)
            for (int j = 0; j < ValueSumType::NGroup; j++)
                simd::zero<FeatureDim>(valueSum.group[i][j].data());

        typedef Batch<FeatureDim, int16_t>    FeatB;
        typedef Batch<FeatDWConvDim, int16_t> ConvB;
        typedef Batch<FeatureDim, int32_t>    VSumB;

        auto addToAccumulator = [](std::array<int32_t, FeatureDim> &vSum, auto v0, auto v1, int b) {
            auto vSumPtr = vSum.data() + b * 2 * VSumB::RegWidth;
            auto vSum0   = I32LS::load(vSumPtr);
            auto vSum1   = I32LS::load(vSumPtr + VSumB::RegWidth);
            vSum0        = I32Op::add(vSum0, v0);
            vSum1        = I32Op::add(vSum1, v1);
            I32LS::store(vSumPtr, vSum0);
            I32LS::store(vSumPtr + VSumB::RegWidth, vSum1);
        };

        for (int y = 0, innerIdx = 0; y < boardSize; y++) {
            for (int x = 0; x < boardSize; x++, innerIdx++) {
                // Init mapSum from four directions
                simd::zero<FeatureDim>(mapSum[innerIdx].data());
                for (int dir = 0; dir < 4; dir++)
                    simd::add<FeatureDim>(mapSum[innerIdx].data(),
                                          mapSum[innerIdx].data(),
                                          w.mapping[dir / 2][indexTable[innerIdx][dir]]);

                // Init mapConv from mapSum
                for (int b = 0; b < ConvB::NumBatch; b++) {
                    auto feature = I16LS::load(mapSum[innerIdx].data() + b * FeatB::RegWidth);
                    feature      = I16Op::max(feature, I16Op::setzero());
                    feature      = I16Op::slli<2>(feature);  // mul 4
                    // Apply feature depthwise conv
                    for (int dy = 0; dy <= 2; dy++) {
                        int yi = y + dy;
                        for (int dx = 0; dx <= 2; dx++) {
                            int xi       = x + dx;
                            int outerIdx = xi + yi * outerBoardSize;

                            auto *convWeightBase = w.feature_dwconv_weight[8 - dy * 3 - dx];
                            auto  convW     = I16LS::load(convWeightBase + b * ConvB::RegWidth);
                            auto  deltaFeat = I16Op::mulhi(convW, feature);
                            auto  convPtr   = mapConv[outerIdx].data() + b * FeatB::RegWidth;
                            auto  convFeat  = I16LS::load(convPtr);
                            convFeat        = I16Op::add(convFeat, deltaFeat);
                            I16LS::store(convPtr, convFeat);
                        }
                    }
                }

                // Add map feature to map value sum
                for (int b = ConvB::NumBatch; b < FeatB::NumBatch; b++) {
                    auto feature  = I16LS::load(mapSum[innerIdx].data() + b * FeatB::RegWidth);
                    feature       = I16Op::max(feature, I16Op::setzero());
                    auto [v0, v1] = Convert<int16_t, int32_t>::convert(feature);

                    addToAccumulator(valueSum.global, v0, v1, b);
                    addToAccumulator(valueSum.group[groupIndex[y]][groupIndex[x]], v0, v1, b);
                }
            }
        }

        // Init valueSum by adding all dwconv value features
        for (int y = 0, outerIdx = outerBoardSize + 1; y < boardSize; y++, outerIdx += 2) {
            for (int x = 0; x < boardSize; x++, outerIdx++) {
                for (int b = 0; b < ConvB::NumBatch; b++) {
                    auto feature  = I16LS::load(mapConv[outerIdx].data() + b * ConvB::RegWidth);
                    feature       = I16Op::max(feature, I16Op::setzero());  // relu
                    auto [v0, v1] = Convert<int16_t, int32_t>::convert(feature);

                    addToAccumulator(valueSum.global, v0, v1, b);
                    addToAccumulator(valueSum.group[groupIndex[y]][groupIndex[x]], v0, v1, b);
                }
            }
        }

        valueSum.small_value_feature_valid = false;
        valueSum.large_value_feature_valid = false;
    }

    // Reset version and init version table to be zeros
    currentVersion = 0;
}

void Accumulator::move(const Weight &w, Color pieceColor, int x, int y)
{
    assert(pieceColor == BLACK || pieceColor == WHITE);

    // Copy version info to the next ply
    const int       innerBoardSizeSqr       = boardSize * boardSize;
    const int       outerBoardSizeSqr       = outerBoardSize * outerBoardSize;
    const int       innerVersionIdxBasePrev = currentVersion * innerBoardSizeSqr;
    const int       outerVersionIdxBasePrev = currentVersion * outerBoardSizeSqr;
    const int       innerVersionIdxBase     = innerVersionIdxBasePrev + innerBoardSizeSqr;
    const int       outerVersionIdxBase     = outerVersionIdxBasePrev + outerBoardSizeSqr;
    const ChangeNum changeNum               = versionChangeNumTable[currentVersion];
    std::copy_n(versionInnerIndexTable + innerVersionIdxBasePrev,
                innerBoardSizeSqr,
                versionInnerIndexTable + innerVersionIdxBase);
    std::copy_n(versionOuterIndexTable + outerVersionIdxBasePrev,
                outerBoardSizeSqr,
                versionOuterIndexTable + outerVersionIdxBase);

    typedef Batch<FeatureDim, int16_t>    FeatB;
    typedef Batch<FeatDWConvDim, int16_t> ConvB;
    typedef Batch<FeatureDim, int32_t>    VSumB;

    // Subtract value feature sum
    int x0            = std::max(x - 6 + 1, 1);
    int y0            = std::max(y - 6 + 1, 1);
    int x1            = std::min(x + 6 + 1, boardSize);
    int y1            = std::min(y + 6 + 1, boardSize);
    int newMapConvIdx = changeNum.outer;
    for (int yi = y0, outerIdxBase = y0 * outerBoardSize; yi <= y1;
         yi++, outerIdxBase += outerBoardSize) {
        for (int xi = x0; xi <= x1; xi++) {
            int outerIdx                                           = xi + outerIdxBase;
            versionOuterIndexTable[outerVersionIdxBase + outerIdx] = newMapConvIdx;
            for (int b = 0; b < ConvB::NumBatch; b++)
                I16LS::store(mapConv[newMapConvIdx].data() + b * ConvB::RegWidth, I16Op::setzero());
            newMapConvIdx++;
        }
    }

    struct OnePointChange
    {
        int8_t   x;
        int8_t   y;
        int16_t  mappingIdx;
        int16_t  oldMapIdx;
        int16_t  newMapIdx;
        uint32_t oldShape;
        uint32_t newShape;
    } changeTable[4 * 11];
    int changeCount = 0;
    int dPower3     = pieceColor + 1;

    // Update shape table and record changes
    const int boardSizeSub1 = boardSize - 1;
    int       newMapIdx     = changeNum.inner;
    for (int dir = 0; dir < 4; dir++) {
        for (int dist = -5; dist <= 5; dist++) {
            int xi = x + dist * DX[dir];
            int yi = y + dist * DY[dir];

            // branchless test: xi < 0 || xi >= boardSize || yi < 0 || yi >= boardSize
            if ((xi | (boardSizeSub1 - xi) | yi | (boardSizeSub1 - yi)) < 0)
                continue;

            int             innerIdx   = boardSize * yi + xi;
            OnePointChange &c          = changeTable[changeCount++];
            c.x                        = xi;
            c.y                        = yi;
            c.mappingIdx               = dir / 2;  // 0,1 -> 0; 2,3 -> 1
            c.oldMapIdx                = versionInnerIndexTable[innerVersionIdxBase + innerIdx];
            c.newMapIdx                = newMapIdx;
            c.oldShape                 = indexTable[c.oldMapIdx][dir];
            c.newShape                 = c.oldShape + dPower3 * Power3[dist + 5];
            indexTable[newMapIdx]      = indexTable[c.oldMapIdx];
            indexTable[newMapIdx][dir] = c.newShape;
            assert(c.newShape < ShapeNum);

            versionInnerIndexTable[innerVersionIdxBase + innerIdx] = newMapIdx++;
        }
    }

    // Init value sum accumulator
    I32Op::R vSumGlobal[VSumB::NumBatch];
    I32Op::R vSumGroup[ValueSumType::NGroup][ValueSumType::NGroup][VSumB::NumBatch];
    for (int b = 0; b < VSumB::NumBatch; b++)
        vSumGlobal[b] = I32Op::setzero();
    for (int i = 0; i < ValueSumType::NGroup; i++)
        for (int j = 0; j < ValueSumType::NGroup; j++)
            for (int b = 0; b < VSumB::NumBatch; b++)
                vSumGroup[i][j][b] = I32Op::setzero();

    // Incremental update feature sum
    for (int i = 0; i < changeCount; i++) {
        const OnePointChange &c = changeTable[i];
        if (i + 1 < changeCount) {
            const OnePointChange &cnext = changeTable[i + 1];
            multiPrefetch<FeatureDim * sizeof(int16_t)>(
                w.mapping[cnext.mappingIdx][cnext.oldShape]);
            multiPrefetch<FeatureDim * sizeof(int16_t)>(
                w.mapping[cnext.mappingIdx][cnext.newShape]);
        }

        // Update mapSum
        I16Op::R oldFeats[FeatB::NumBatch];
        I16Op::R newFeats[FeatB::NumBatch];
        for (int b = 0; b < FeatB::NumBatch; b++) {
            auto newMapFeat =
                I16LS::load(w.mapping[c.mappingIdx][c.newShape] + b * FeatB::RegWidth);
            auto oldMapFeat =
                I16LS::load(w.mapping[c.mappingIdx][c.oldShape] + b * FeatB::RegWidth);
            oldFeats[b] = I16LS::load(mapSum[c.oldMapIdx].data() + b * FeatB::RegWidth);
            newFeats[b] = I16Op::sub(oldFeats[b], oldMapFeat);
            newFeats[b] = I16Op::add(newFeats[b], newMapFeat);
            I16LS::store(mapSum[c.newMapIdx].data() + b * FeatB::RegWidth, newFeats[b]);
            oldFeats[b] = I16Op::max(oldFeats[b], I16Op::setzero());
            newFeats[b] = I16Op::max(newFeats[b], I16Op::setzero());
        }

        // Update mapConv
        for (int b = 0; b < ConvB::NumBatch; b++) {
            oldFeats[b] = I16Op::slli<2>(oldFeats[b]);  // mul 4
            newFeats[b] = I16Op::slli<2>(newFeats[b]);  // mul 4
        }
        for (int dy = 0, outerIdxBase = c.y * outerBoardSize + c.x; dy <= 2;
             dy++, outerIdxBase += outerBoardSize) {
            for (int dx = 0; dx <= 2; dx++) {
                int   outerIdx       = dx + outerIdxBase;
                int   mapConvIdx     = versionOuterIndexTable[outerVersionIdxBase + outerIdx];
                auto *convWeightBase = w.feature_dwconv_weight[8 - dy * 3 - dx];
                auto *convBase       = mapConv[mapConvIdx].data();

                for (int b = 0; b < ConvB::NumBatch; b++) {
                    auto convW      = I16LS::load(convWeightBase + b * ConvB::RegWidth);
                    auto deltaConvF = I16Op::sub(I16Op::mulhi(convW, newFeats[b]),
                                                 I16Op::mulhi(convW, oldFeats[b]));

                    auto convPtr  = convBase + b * ConvB::RegWidth;
                    auto oldConvF = I16LS::load(convPtr);
                    auto newConvF = I16Op::add(oldConvF, deltaConvF);
                    I16LS::store(convPtr, newConvF);
                }
            }
        }

        // Update valueSum
        for (int b = ConvB::NumBatch; b < FeatB::NumBatch; b++) {
            auto deltaF             = I16Op::sub(newFeats[b], oldFeats[b]);
            auto [deltaF0, deltaF1] = Convert<int16_t, int32_t>::convert(deltaF);

            const int offset       = 2 * b;
            vSumGlobal[offset + 0] = I32Op::add(vSumGlobal[offset + 0], deltaF0);
            vSumGlobal[offset + 1] = I32Op::add(vSumGlobal[offset + 1], deltaF1);
            auto &vGroup           = vSumGroup[groupIndex[c.y]][groupIndex[c.x]];
            vGroup[offset + 0]     = I32Op::add(vGroup[offset + 0], deltaF0);
            vGroup[offset + 1]     = I32Op::add(vGroup[offset + 1], deltaF1);
        }
    }

    // Add value feature sum
    newMapConvIdx = changeNum.outer;
    for (int yi = y0, outerIdxBase = y0 * outerBoardSize; yi <= y1;
         yi++, outerIdxBase += outerBoardSize) {
        int i = groupIndex[yi - 1];
        for (int xi = x0; xi <= x1; xi++) {
            int j             = groupIndex[xi - 1];
            int outerIdx      = xi + outerIdxBase;
            int oldMapConvIdx = versionOuterIndexTable[outerVersionIdxBasePrev + outerIdx];
            for (int b = 0; b < ConvB::NumBatch; b++) {
                auto oldConvF   = I16LS::load(mapConv[oldMapConvIdx].data() + b * ConvB::RegWidth);
                auto deltaConvF = I16LS::load(mapConv[newMapConvIdx].data() + b * ConvB::RegWidth);
                auto newConvF   = I16Op::add(oldConvF, deltaConvF);
                I16LS::store(mapConv[newMapConvIdx].data() + b * ConvB::RegWidth, newConvF);
                oldConvF      = I16Op::max(oldConvF, I16Op::setzero());  // relu
                newConvF      = I16Op::max(newConvF, I16Op::setzero());  // relu
                auto deltaF   = I16Op::sub(newConvF, oldConvF);
                auto [v0, v1] = Convert<int16_t, int32_t>::convert(deltaF);

                const int offset            = 2 * b;
                vSumGlobal[offset + 0]      = I32Op::add(vSumGlobal[offset + 0], v0);
                vSumGlobal[offset + 1]      = I32Op::add(vSumGlobal[offset + 1], v1);
                vSumGroup[i][j][offset + 0] = I32Op::add(vSumGroup[i][j][offset + 0], v0);
                vSumGroup[i][j][offset + 1] = I32Op::add(vSumGroup[i][j][offset + 1], v1);
            }
            newMapConvIdx++;
        }
    }

    // Move to next version
    currentVersion++;
    versionChangeNumTable[currentVersion] = {uint16_t(newMapIdx), uint16_t(newMapConvIdx)};

    // Store value sum
    auto &valueSumOld = valueSumTable[currentVersion - 1];
    auto &valueSumNew = valueSumTable[currentVersion];
    for (int b = 0; b < VSumB::NumBatch; b++) {
        auto vOld = I32LS::load(valueSumOld.global.data() + b * VSumB::RegWidth);
        auto vNew = I32Op::add(vOld, vSumGlobal[b]);
        I32LS::store(valueSumNew.global.data() + b * VSumB::RegWidth, vNew);
    }
    for (int i = 0; i < ValueSumType::NGroup; i++)
        for (int j = 0; j < ValueSumType::NGroup; j++)
            for (int b = 0; b < VSumB::NumBatch; b++) {
                auto vOld = I32LS::load(valueSumOld.group[i][j].data() + b * VSumB::RegWidth);
                auto vNew = I32Op::add(vOld, vSumGroup[i][j][b]);
                I32LS::store(valueSumNew.group[i][j].data() + b * VSumB::RegWidth, vNew);
            }
    valueSumNew.small_value_feature_valid = false;
    valueSumNew.large_value_feature_valid = false;
}

void Accumulator::updateSharedSmallHead(const Weight &w)
{
    auto       &valueSum = valueSumTable[currentVersion];
    const auto &bucket   = w.buckets[getBucketIndex()];

    if (valueSum.small_value_feature_valid)
        return;

    // global feature sum
    alignas(Alignment) int8_t layer0[FeatureDim];
    simd::crelu<FeatureDim, 256, true>(layer0, valueSum.global.data());

    // small value head layer 1
    alignas(Alignment) int8_t layer1[FeatureDim];
    linearBlock(layer1, layer0, bucket.value_small_l1);

    // small value head layer 2
    linearBlock(valueSum.small_value_feature.data(), layer1, bucket.value_small_l2);
    valueSum.small_value_feature_valid = true;
}

void Accumulator::updateSharedLargeHead(const Weight &w)
{
    auto       &valueSum = valueSumTable[currentVersion];
    const auto &bucket   = w.buckets[getBucketIndex()];

    if (valueSum.large_value_feature_valid)
        return;

    updateSharedSmallHead(w);

    // group feature sum
    alignas(Alignment) int8_t group0_in[ValueSumType::NGroup][ValueSumType::NGroup][FeatureDim];
    for (int i = 0; i < ValueSumType::NGroup; i++)
        for (int j = 0; j < ValueSumType::NGroup; j++)
            simd::crelu<FeatureDim, 32, true>(group0_in[i][j], valueSum.group[i][j].data());

    // value gate
    alignas(Alignment) int8_t gate[FeatureDim * 2];
    linearBlock<false, true>(gate, valueSum.small_value_feature.data(), bucket.value_gate);

    alignas(Alignment) int8_t group0[ValueSumType::NGroup][ValueSumType::NGroup][FeatureDim];
    for (int i = 0; i < ValueSumType::NGroup; i++)
        for (int j = 0; j < ValueSumType::NGroup; j++) {
            simd::dot2<FeatureDim / 2, 128>(group0[i][j], group0_in[i][j], gate);
            simd::dot2<FeatureDim / 2, 128>(group0[i][j] + FeatureDim / 2,
                                            group0_in[i][j],
                                            gate + FeatureDim);
        }

    // group linear layer
    alignas(Alignment) int8_t group1[ValueSumType::NGroup][ValueSumType::NGroup][ValueDim];

    linearBlock<true>(group1[0][0], group0[0][0], bucket.value_corner);
    linearBlock<true>(group1[0][2], group0[0][2], bucket.value_corner);
    linearBlock<true>(group1[2][0], group0[2][0], bucket.value_corner);
    linearBlock<true>(group1[2][2], group0[2][2], bucket.value_corner);
    linearBlock<true>(group1[0][1], group0[0][1], bucket.value_edge);
    linearBlock<true>(group1[1][0], group0[1][0], bucket.value_edge);
    linearBlock<true>(group1[1][2], group0[1][2], bucket.value_edge);
    linearBlock<true>(group1[2][1], group0[2][1], bucket.value_edge);
    linearBlock<true>(group1[1][1], group0[1][1], bucket.value_center);

    // average pooling
    alignas(Alignment) int8_t group2[2][2][ValueDim];
    using I8B = Batch<ValueDim, int8_t>;
    for (int b = 0; b < I8B::NumBatch; b++) {
        auto v00 = I8LS::load(group1[0][0] + b * I8B::RegWidth);
        auto v01 = I8LS::load(group1[0][1] + b * I8B::RegWidth);
        auto v02 = I8LS::load(group1[0][2] + b * I8B::RegWidth);
        auto v10 = I8LS::load(group1[1][0] + b * I8B::RegWidth);
        auto v11 = I8LS::load(group1[1][1] + b * I8B::RegWidth);
        auto v12 = I8LS::load(group1[1][2] + b * I8B::RegWidth);
        auto v20 = I8LS::load(group1[2][0] + b * I8B::RegWidth);
        auto v21 = I8LS::load(group1[2][1] + b * I8B::RegWidth);
        auto v22 = I8LS::load(group1[2][2] + b * I8B::RegWidth);

        auto q00 = I8Op::avg(I8Op::avg(v00, v01), I8Op::avg(v10, v11));
        auto q01 = I8Op::avg(I8Op::avg(v01, v02), I8Op::avg(v11, v12));
        auto q10 = I8Op::avg(I8Op::avg(v10, v11), I8Op::avg(v20, v21));
        auto q11 = I8Op::avg(I8Op::avg(v11, v12), I8Op::avg(v21, v22));

        I8LS::store(group2[0][0] + b * I8B::RegWidth, q00);
        I8LS::store(group2[0][1] + b * I8B::RegWidth, q01);
        I8LS::store(group2[1][0] + b * I8B::RegWidth, q10);
        I8LS::store(group2[1][1] + b * I8B::RegWidth, q11);
    }

    // quadrant linear layer
    alignas(Alignment) int8_t layer0[ValueDim * 5];
    simd::copy<ValueDim>(layer0, valueSum.small_value_feature.data());
    linearBlock(layer0 + 1 * ValueDim, group2[0][0], bucket.value_quad);
    linearBlock(layer0 + 2 * ValueDim, group2[0][1], bucket.value_quad);
    linearBlock(layer0 + 3 * ValueDim, group2[1][0], bucket.value_quad);
    linearBlock(layer0 + 4 * ValueDim, group2[1][1], bucket.value_quad);

    // linear 1, 2
    alignas(Alignment) int8_t layer1[ValueDim];
    linearBlock(layer1, layer0, bucket.value_l1);
    linearBlock(valueSum.large_value_feature.data(), layer1, bucket.value_l2);
    valueSum.large_value_feature_valid = true;
}

std::tuple<float, float, float, float> Accumulator::evaluateValueSmall(const Weight &w)
{
    updateSharedSmallHead(w);

    const auto &valueSum = valueSumTable[currentVersion];
    const auto &bucket   = w.buckets[getBucketIndex()];

    // small value head layer 3 final
    alignas(Alignment) int32_t layer3i32[4];
    simd::linear<4, ValueDim>(layer3i32,
                              valueSum.small_value_feature.data(),
                              bucket.value_small_l3.weight,
                              bucket.value_small_l3.bias);

    constexpr float OutScale = 1.0f / (128 * 128);
    return {
        layer3i32[0] * OutScale,
        layer3i32[1] * OutScale,
        layer3i32[2] * OutScale,
        layer3i32[3] * OutScale,
    };
}

std::tuple<float, float, float, float> Accumulator::evaluateValueLarge(const Weight &w)
{
    updateSharedLargeHead(w);

    const auto &valueSum = valueSumTable[currentVersion];
    const auto &bucket   = w.buckets[getBucketIndex()];

    // linear 3 final
    alignas(Alignment) int32_t layer3i32[4];
    simd::linear<4, ValueDim>(layer3i32,
                              valueSum.large_value_feature.data(),
                              bucket.value_l3.weight,
                              bucket.value_l3.bias);

    constexpr float OutScale = 1.0f / (128 * 128);
    return {
        layer3i32[0] * OutScale,
        layer3i32[1] * OutScale,
        layer3i32[2] * OutScale,
        layer3i32[3] * OutScale,
    };
}

void Accumulator::evaluatePolicySmall(const Weight &w, PolicyBuffer &policyBuffer)
{
    updateSharedSmallHead(w);

    const auto &valueSum = valueSumTable[currentVersion];
    const auto &bucket   = w.buckets[getBucketIndex()];

    // policy pwconv weight layer
    alignas(Alignment) int8_t layer1[ValueDim];
    linearBlock(layer1, valueSum.small_value_feature.data(), bucket.policy_small_pwconv_weight_1);

    alignas(Alignment) int32_t layer2i32[PolicySOutDim * (PolicySInDim + 1)];
    alignas(Alignment) int16_t pwconvWeighti16[PolicySOutDim * PolicySInDim];
    alignas(Alignment) int32_t pwconvBiasi32[PolicySOutDim];
    simd::linear<PolicySOutDim *(PolicySInDim + 1), ValueDim>(
        layer2i32,
        layer1,
        bucket.policy_small_pwconv_weight_2.weight,
        bucket.policy_small_pwconv_weight_2.bias);
    simd::crelu<PolicySOutDim * PolicySInDim, 1, true>(pwconvWeighti16, layer2i32);
    // To get pwconv bias, we need to scale the output of layer2 by 128
    {
        typedef Batch<PolicySOutDim, int32_t> B;
        for (int i = 0; i < B::NumBatch; i++) {
            auto data = I32LS::load(layer2i32 + PolicySOutDim * PolicySInDim + i * B::RegWidth);
            data      = I32Op::slli<7>(data);  // scale by 128
            I32LS::store(pwconvBiasi32 + i * B::RegWidth, data);
        }
    }

    const int outerVersionIdxBase = currentVersion * outerBoardSize * outerBoardSize;
    for (int y = 0, innerIdx = 0, outerIdx = outerBoardSize + 1; y < boardSize;
         y++, outerIdx += 2) {
        for (int x = 0; x < boardSize; x++, innerIdx++, outerIdx++) {
            if (!policyBuffer.getComputeFlag(innerIdx))
                continue;

            // Get mapConv index of current version at this point
            int mapConvIdx = versionOuterIndexTable[outerVersionIdxBase + outerIdx];

            // Compute dynamic point-wise policy conv
            alignas(Alignment) int32_t policyLayer1i32[PolicySOutDim];
            simd::linear<PolicySOutDim, PolicySInDim, false, true, true>(policyLayer1i32,
                                                                         mapConv[mapConvIdx].data(),
                                                                         pwconvWeighti16,
                                                                         pwconvBiasi32);

            // Apply relu, convert to float and accumulate all channels of pwconv feature
            typedef Batch<PolicySOutDim, float> PWConvB;
            auto                                policyAccum = F32Op::setzero();
            for (int i = 0; i < PWConvB::NumBatch; i++) {
                auto featI32 = I32LS::load(policyLayer1i32 + i * PWConvB::RegWidth);
                featI32      = I32Op::max(featI32, I32Op::setzero());
                auto featF32 = Convert<int32_t, float>::convert1(featI32);
                auto outputW =
                    F32LS::load(bucket.policy_small_output_weight + i * PWConvB::RegWidth);
                policyAccum = F32Op::fmadd(featF32, outputW, policyAccum);
            }

            float policy = F32Op::reduceadd(policyAccum) + bucket.policy_small_output_bias;
            policyBuffer(innerIdx) = policy;
        }
    }
}

void Accumulator::evaluatePolicyLarge(const Weight &w, PolicyBuffer &policyBuffer)
{
    updateSharedLargeHead(w);

    const auto &valueSum = valueSumTable[currentVersion];
    const auto &bucket   = w.buckets[getBucketIndex()];

    // policy pwconv weight layer
    alignas(Alignment) int8_t layer1[ValueDim];
    linearBlock(layer1, valueSum.large_value_feature.data(), bucket.policy_large_pwconv_weight_0);

    alignas(Alignment) int32_t layer2i32[PolicyLMidDim * (PolicyLInDim + 1)];
    alignas(Alignment) int16_t pwconv1Weighti16[PolicyLMidDim * PolicyLInDim];
    alignas(Alignment) int32_t pwconv1Biasi32[PolicyLMidDim];
    simd::linear<PolicyLMidDim *(PolicyLInDim + 1), ValueDim>(
        layer2i32,
        layer1,
        bucket.policy_large_pwconv_weight_1.weight,
        bucket.policy_large_pwconv_weight_1.bias);
    simd::crelu<PolicyLMidDim * PolicyLInDim, 1, true>(pwconv1Weighti16, layer2i32);
    // To get pwconv bias, we need to scale the output of layer2 by 128
    {
        typedef Batch<PolicyLMidDim, int32_t> B;
        for (int i = 0; i < B::NumBatch; i++) {
            auto data = I32LS::load(layer2i32 + PolicyLMidDim * PolicyLInDim + i * B::RegWidth);
            data      = I32Op::slli<7>(data);  // scale by 128
            I32LS::store(pwconv1Biasi32 + i * B::RegWidth, data);
        }
    }

    alignas(Alignment) int32_t layer3i32[PolicyLOutDim * (PolicyLMidDim + 1)];
    alignas(Alignment) int16_t pwconv2Weighti16[PolicyLOutDim * PolicyLMidDim];
    alignas(Alignment) int32_t pwconv2Biasi32[PolicyLOutDim];
    simd::linear<PolicyLOutDim *(PolicyLMidDim + 1), ValueDim>(
        layer3i32,
        layer1,
        bucket.policy_large_pwconv_weight_2.weight,
        bucket.policy_large_pwconv_weight_2.bias);
    simd::crelu<PolicyLOutDim * PolicyLMidDim, 1, true>(pwconv2Weighti16, layer3i32);
    // To get pwconv bias, we need to scale the output of layer3 by 128
    {
        typedef Batch<PolicyLOutDim, int32_t> B;
        for (int i = 0; i < B::NumBatch; i++) {
            auto data = I32LS::load(layer3i32 + PolicyLOutDim * PolicyLMidDim + i * B::RegWidth);
            data      = I32Op::slli<7>(data);  // scale by 128
            I32LS::store(pwconv2Biasi32 + i * B::RegWidth, data);
        }
    }

    const int outerVersionIdxBase = currentVersion * outerBoardSize * outerBoardSize;
    for (int y = 0, innerIdx = 0, outerIdx = outerBoardSize + 1; y < boardSize;
         y++, outerIdx += 2) {
        for (int x = 0; x < boardSize; x++, innerIdx++, outerIdx++) {
            if (!policyBuffer.getComputeFlag(innerIdx))
                continue;

            // Get mapConv index of current version at this point
            int mapConvIdx = versionOuterIndexTable[outerVersionIdxBase + outerIdx];

            // Compute dynamic point-wise policy conv
            alignas(Alignment) int32_t policyLayer1i32[PolicyLMidDim];
            alignas(Alignment) int16_t policyLayer1i16[PolicyLMidDim];
            simd::linear<PolicyLMidDim, PolicyLInDim, false, true, true>(policyLayer1i32,
                                                                         mapConv[mapConvIdx].data(),
                                                                         pwconv1Weighti16,
                                                                         pwconv1Biasi32);
            // Size of policyLayer1i16 may be less than 512bit width when PolicyLMidDim is only 16,
            // so we choose the maximum available instruction here for AVX512 platforms.
            constexpr auto ITPolicyLMid =
                simd::getInstTypeOfWidth(IT, PolicyLMidDim * sizeof(int16_t) * 8);
            simd::crelu<PolicyLMidDim, 128 * 128, false, Alignment, ITPolicyLMid>(policyLayer1i16,
                                                                                  policyLayer1i32);

            alignas(Alignment) int32_t policyLayer2i32[PolicyLOutDim];
            simd::linear<PolicyLOutDim, PolicyLMidDim>(policyLayer2i32,
                                                       policyLayer1i16,
                                                       pwconv2Weighti16,
                                                       pwconv2Biasi32);

            // Apply relu, convert to float and accumulate all channels of pwconv feature
            typedef Batch<PolicyLOutDim, float> PWConvB;
            auto                                policyAccum = F32Op::setzero();
            for (int i = 0; i < PWConvB::NumBatch; i++) {
                auto featI32 = I32LS::load(policyLayer2i32 + i * PWConvB::RegWidth);
                featI32      = I32Op::max(featI32, I32Op::setzero());
                auto featF32 = Convert<int32_t, float>::convert1(featI32);
                auto outputW =
                    F32LS::load(bucket.policy_large_output_weight + i * PWConvB::RegWidth);
                policyAccum = F32Op::fmadd(featF32, outputW, policyAccum);
            }

            float policy = F32Op::reduceadd(policyAccum) + bucket.policy_large_output_bias;
            policyBuffer(innerIdx) = policy;
        }
    }
}

Evaluator::Evaluator(int                   boardSize,
                     Rule                  rule,
                     Numa::NumaNodeId      numaNodeId,
                     std::filesystem::path blackWeightPath,
                     std::filesystem::path whiteWeightPath)
    : Evaluation::Evaluator(boardSize, rule)
    , weight {nullptr, nullptr}
{
    constexpr uint32_t ArchHash =
        ArchHashBase ^ (((ValueDim / 8) << 16) | ((FeatDWConvDim / 8) << 8) | (FeatureDim / 8));
    loadWeightPair<CompressedWrapper<StandardHeaderLoader<Mix10WeightLoader>>>(WeightReg,
                                                                               weight,
                                                                               "mix10 nnue",
                                                                               ArchHash,
                                                                               boardSize,
                                                                               rule,
                                                                               numaNodeId,
                                                                               blackWeightPath,
                                                                               whiteWeightPath);

    accumulator[BLACK] = std::make_unique<Accumulator>(boardSize);
    accumulator[WHITE] = std::make_unique<Accumulator>(boardSize);

    int numCells = boardSize * boardSize;
    moveCache[BLACK].reserve(numCells);
    moveCache[WHITE].reserve(numCells);
}

Evaluator::~Evaluator()
{
    if (weight[BLACK])
        WeightReg.unloadWeight(weight[BLACK]);
    if (weight[WHITE])
        WeightReg.unloadWeight(weight[WHITE]);
}

void Evaluator::initEmptyBoard()
{
    moveCache[BLACK].clear();
    moveCache[WHITE].clear();
    accumulator[BLACK]->clear(*weight[BLACK]);
    accumulator[WHITE]->clear(*weight[WHITE]);
}

void Evaluator::beforeMove(const Board &board, Pos pos)
{
    addCache(board.sideToMove(), pos.x(), pos.y(), false);
}

void Evaluator::afterUndo(const Board &board, Pos pos)
{
    addCache(board.sideToMove(), pos.x(), pos.y(), true);
}

ValueType Evaluator::evaluateValue(const Board &board, AccLevel level)
{
    Color self = board.sideToMove();

    // Apply all incremental update for both sides and calculate value
    flushMoveCache(self);
    auto [win, loss, draw, _] = accumulator[self]->evaluateValueLarge(*weight[self]);

    return ValueType(win, loss, draw, true);
}

void Evaluator::evaluatePolicy(const Board &board, PolicyBuffer &policyBuffer, AccLevel level)
{
    Color self = board.sideToMove();

    // Apply all incremental update and calculate policy
    flushMoveCache(self);
    accumulator[self]->evaluatePolicyLarge(*weight[self], policyBuffer);
}

void Evaluator::flushMoveCache(Color side)
{
    applyMoveCache(side, moveCache[side], *accumulator[side], *weight[side]);
}

void Evaluator::addCache(Color side, int x, int y, bool isUndo)
{
    queueMoveCache(moveCache, side, x, y, isUndo, boardSize * boardSize);
}

}  // namespace Evaluation::mix10
