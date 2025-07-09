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

#include "../core/iohelper.h"
#include "../core/platform.h"
#include "../core/utils.h"
#include "../game/board.h"
#include "simdops.h"
#include "weightloader.h"

#include <algorithm>
#include <cmath>
#include <cstring>

namespace {

using namespace Evaluation::mix10;

constexpr auto Power3 = []() {
    auto pow3 = std::array<int, 16> {};
    for (size_t i = 0; i < pow3.size(); i++)
        pow3[i] = power(3, i);
    return pow3;
}();

constexpr int DX[4] = {1, 0, 1, 1};
constexpr int DY[4] = {0, 1, -1, 1};

// Max inner and outer point changes, indexed by board size
constexpr int MaxInnerChanges[23] = {1,    6,     33,    102,   233,   446,   761,  1166,
                                     1661, 2246,  2921,  3686,  4541,  5486,  6521, 7646,
                                     8861, 10166, 11561, 13046, 14621, 16286, 18041};
constexpr int MaxOuterChanges[23] = {5,     11,    33,    107,   293,   675,   1361,  2483,
                                     3945,  5747,  7889,  10371, 13193, 16355, 19857, 23699,
                                     27881, 32403, 37265, 42467, 48009, 53891, 60113};

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
        constexpr int      MappingBits   = 10;
        constexpr uint64_t MappingMask   = (1 << MappingBits) - 1;
        constexpr uint16_t ExtensionMask = 1 << (MappingBits - 1);
        constexpr uint16_t ExtensionBits = ~static_cast<uint16_t>(MappingMask);

        uint64_t u64val    = 0;
        int      bits_left = 0;

        for (int mappingIdx = 0; mappingIdx < arraySize(w.mapping); mappingIdx++) {
            auto &mapping = w.mapping[mappingIdx];

            for (int i = 0; i < ShapeNum; i++)
                for (int j = 0; j < FeatureDim; j++) {
                    int16_t feature = 0;

                    if (bits_left >= MappingBits) {
                        uint16_t masked    = static_cast<uint16_t>(u64val & MappingMask);
                        uint16_t extension = (masked & ExtensionMask) ? 0xfc00 : 0;
                        feature            = static_cast<int16_t>(extension | masked);

                        u64val >>= MappingBits;
                        bits_left -= MappingBits;
                    }
                    else {
                        uint64_t u64val2;
                        in.read(reinterpret_cast<char *>(&u64val2), sizeof(u64val2));

                        u64val |= u64val2 << bits_left;
                        uint16_t masked    = static_cast<uint16_t>(u64val & MappingMask);
                        uint16_t extension = (masked & ExtensionMask) ? 0xfc00 : 0;
                        feature            = static_cast<int16_t>(extension | masked);

                        u64val    = u64val2 >> (MappingBits - bits_left);
                        bits_left = 64 - (MappingBits - bits_left);
                    }

                    mapping[i][j] = feature;
                }
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

constexpr int                   Alignment = simd::NativeAlignment;
constexpr simd::InstructionType IT        = simd::NativeInstType;
template <size_t Size, typename T>
using Batch = simd::detail::VecBatch<Size, T, IT>;
template <typename FT, typename TT>
using Convert = simd::detail::VecCvt<FT, TT, IT>;
using I8LS    = simd::detail::VecLoadStore<int8_t, Alignment, IT>;
using I16LS   = simd::detail::VecLoadStore<int16_t, Alignment, IT>;
using I32LS   = simd::detail::VecLoadStore<int32_t, Alignment, IT>;
using F32LS   = simd::detail::VecLoadStore<float, Alignment, IT>;
using I8Op    = simd::detail::VecOp<int8_t, IT>;
using I16Op   = simd::detail::VecOp<int16_t, IT>;
using I32Op   = simd::detail::VecOp<int32_t, IT>;
using F32Op   = simd::detail::VecOp<float, IT>;

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

Accumulator::Accumulator(int boardSize)
    : boardSize(boardSize)
    , outerBoardSize(boardSize + 2)
    , currentVersion(-1)
{
    int nCells        = boardSize * boardSize;
    int nInnerChanges = MaxInnerChanges[boardSize];
    int nOuterChanges = MaxOuterChanges[boardSize];

    valueSumTable          = MemAlloc::alignedArrayAlloc<ValueSumType, Alignment>(nCells + 1);
    versionChangeNumTable  = new ChangeNum[nCells + 1];
    versionInnerIndexTable = new uint16_t[(nCells + 1) * nCells];
    versionOuterIndexTable = new uint16_t[(nCells + 2) * outerBoardSize * outerBoardSize];
    indexTable             = new std::array<uint32_t, 4>[nInnerChanges];
    mapSum = MemAlloc::alignedArrayAlloc<std::array<int16_t, FeatureDim>, Alignment>(nInnerChanges);
    mapConv =
        MemAlloc::alignedArrayAlloc<std::array<int16_t, FeatDWConvDim>, Alignment>(nOuterChanges);

    // Compute group index based on board pos
    std::fill_n(groupIndex, arraySize(groupIndex), 0);
    int size1 = (boardSize / 3) + (boardSize % 3 == 2);
    int size2 = (boardSize / 3) * 2 + (boardSize % 3 > 0);
    for (int i = 0; i < boardSize; i++)
        groupIndex[i] += (i >= size1) + (i >= size2);
    // Setup version change num of the first layer (version 0)
    versionChangeNumTable[0] = {uint16_t(boardSize * boardSize),
                                uint16_t((boardSize + 2) * (boardSize + 2))};
    // Setup the first layer of version index table to be identity mapping
    for (int i = 0; i < nCells; i++)
        versionInnerIndexTable[i] = i;
    for (int i = 0; i < outerBoardSize * outerBoardSize; i++)
        versionOuterIndexTable[i] = i;

    // Init index table at the first layer (version 0)
    initIndexTable();
}

Accumulator::~Accumulator()
{
    MemAlloc::alignedFree(valueSumTable);
    delete[] versionChangeNumTable;
    delete[] versionInnerIndexTable;
    delete[] versionOuterIndexTable;
    delete[] indexTable;
    MemAlloc::alignedFree(mapSum);
    MemAlloc::alignedFree(mapConv);
}

void Accumulator::initIndexTable()
{
    constexpr int length = 11;  // length of line shape
    constexpr int half   = length / 2;

    // Clear shape table
    std::fill_n(&indexTable[0][0], 4 * boardSize * boardSize, 0);

    /// Get an empty line encoding with the given boarder distance.
    /// @param left The distance to the left boarder, in range [0, length/2].
    /// @param right The distance to the right boarder, in range [0, length/2].
    auto get_boarder_encoding = [=](int left, int right) -> uint32_t {
        assert(0 <= left && left <= half);
        assert(0 <= right && right <= half);

        if (left == half && right == half)
            return 0;
        else if (right == half)  // (left < half)
        {
            uint32_t code      = 2 * Power3[length];
            int      left_dist = half - left;
            for (int i = 1; i < left_dist; i++)
                code += 1 * Power3[length - i];
            return code;
        }
        else  // (right < half && left <= half)
        {
            uint32_t code       = 1 * Power3[length];
            int      left_dist  = half - left;
            int      right_dist = half - right;
            int      right_twos = std::min(left_dist, right_dist);
            int      left_twos  = std::min(left_dist, right_dist - 1);

            for (int i = 0; i < right_twos; i++)
                code += 2 * Power3[i];
            for (int i = 0; i < left_twos; i++)
                code += 2 * Power3[length - 1 - i];

            for (int i = right_twos; i < right_dist - 1; i++)
                code += 1 * Power3[i];
            for (int i = left_twos; i < left_dist - 1; i++)
                code += 1 * Power3[length - 1 - i];

            return code;
        }
    };

    for (int y = 0; y < boardSize; y++)
        for (int x = 0; x < boardSize; x++) {
            auto &idxs   = indexTable[y * boardSize + x];
            int   distx0 = std::min(x - 0, half);
            int   distx1 = std::min(boardSize - 1 - x, half);
            int   disty0 = std::min(y - 0, half);
            int   disty1 = std::min(boardSize - 1 - y, half);

            // DX[0]=1, DY[0]=0
            idxs[0] = get_boarder_encoding(distx0, distx1);
            // DX[1]=0, DY[1]=1
            idxs[1] = get_boarder_encoding(disty0, disty1);
            // DX[2]=1, DY[2]=-1
            idxs[2] = get_boarder_encoding(std::min(distx0, disty1), std::min(distx1, disty0));
            // DX[3]=1, DY[3]=1
            idxs[3] = get_boarder_encoding(std::min(distx0, disty0), std::min(distx1, disty1));

            assert(idxs[0] < ShapeNum);
            assert(idxs[1] < ShapeNum);
            assert(idxs[2] < ShapeNum);
            assert(idxs[3] < ShapeNum);
        }
}

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
            multiPrefetch<FeatureDim * sizeof(int16_t)>(
                w.mapping[c.mappingIdx][changeTable[i + 1].oldShape]);
            multiPrefetch<FeatureDim * sizeof(int16_t)>(
                w.mapping[c.mappingIdx][changeTable[i + 1].newShape]);
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

                    // auto deltaConvF = I16Op::sub(newFeats[b], oldFeats[b]);
                    // deltaConvF      = I16Op::mulhi(convW, deltaConvF);

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
    CompressedWrapper<StandardHeaderLoader<Mix10WeightLoader>> loader(
        Compressor::Type::LZ4_DEFAULT);

    if (boardSize > 22)
        throw UnsupportedBoardSizeError(boardSize);

    Time startTime     = now();
    bool printLoadInfo = false;
    loader.setHeaderValidator([&](StandardHeader header, auto &args) -> bool {
        constexpr uint32_t ArchHash =
            ArchHashBase ^ (((ValueDim / 8) << 16) | ((FeatDWConvDim / 8) << 8) | (FeatureDim / 8));
        if (header.archHash != ArchHash)
            throw IncompatibleWeightFileError("incompatible architecture in weight file.");

        if (!contains(header.supportedRules, args.rule))
            throw UnsupportedRuleError(args.rule);

        if (!contains(header.supportedBoardSizes, args.boardSize))
            throw UnsupportedBoardSizeError(args.boardSize);

        if (Config::MessageMode != MsgMode::NONE) {
            MESSAGEL("mix10 nnue: load weight from " << pathToConsoleString(args.weightPath));
            printLoadInfo = true;
        }
        return true;
    });

    for (const auto &[weightSide, weightPath] : {
             std::make_pair(BLACK, blackWeightPath),
             std::make_pair(WHITE, whiteWeightPath),
         }) {
        weight[weightSide] = WeightReg.loadWeightFromFile(loader,
                                                          weightPath,
                                                          numaNodeId,
                                                          {{}, boardSize, rule, weightPath});
        if (!weight[weightSide])
            throw std::runtime_error("failed to load nnue weight from "
                                     + pathToConsoleString(weightPath));
    }

    if (printLoadInfo)
        MESSAGEL("mix10 nnue: weight loaded in " << timeText(now() - startTime));

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
    Color self = board.sideToMove(), oppo = ~self;

    // Apply all incremental update for both sides and calculate value
    clearCache(self);
    auto [win, loss, draw, _] = accumulator[self]->evaluateValueLarge(*weight[self]);

    return ValueType(win, loss, draw, true);
}

void Evaluator::evaluatePolicy(const Board &board, PolicyBuffer &policyBuffer, AccLevel level)
{
    Color self = board.sideToMove();

    // Apply all incremental update and calculate policy
    clearCache(self);
    accumulator[self]->evaluatePolicyLarge(*weight[self], policyBuffer);
}

void Evaluator::clearCache(Color side)
{
    constexpr Color opponentMap[4] = {WHITE, BLACK, WALL, EMPTY};

    for (MoveCache &mc : moveCache[side]) {
        if (side == WHITE) {
            mc.oldColor = opponentMap[mc.oldColor];
            mc.newColor = opponentMap[mc.newColor];
        }

        if (mc.oldColor == EMPTY)
            accumulator[side]->move(*weight[side], mc.newColor, mc.x, mc.y);
        else
            accumulator[side]->undo();
    }
    moveCache[side].clear();
}

void Evaluator::addCache(Color side, int x, int y, bool isUndo)
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

}  // namespace Evaluation::mix10
