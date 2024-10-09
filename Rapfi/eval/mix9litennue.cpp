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

#include "mix9litennue.h"

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

using namespace Evaluation::mix9lite;

constexpr auto Power3 = []() {
    auto pow3 = std::array<int, 16> {};
    for (size_t i = 0; i < pow3.size(); i++)
        pow3[i] = power(3, i);
    return pow3;
}();

constexpr int DX[4] = {1, 0, 1, 1};
constexpr int DY[4] = {0, 1, 1, -1};

// Max inner and outer point changes, indexed by board size
constexpr int MaxInnerChanges[23] = {1,    6,     33,    102,   233,   446,   761,  1166,
                                     1661, 2246,  2921,  3686,  4541,  5486,  6521, 7646,
                                     8861, 10166, 11561, 13046, 14621, 16286, 18041};
constexpr int MaxOuterChanges[23] = {5,     11,    33,    107,   293,   675,   1361,  2483,
                                     3945,  5747,  7889,  10371, 13193, 16355, 19857, 23699,
                                     27881, 32403, 37265, 42467, 48009, 53891, 60113};

constexpr int                   Alignment = 16;
constexpr simd::InstructionType IT256     = getInstTypeOfWidth(simd::NativeInstType, 256);
constexpr simd::InstructionType IT128     = getInstTypeOfWidth(simd::NativeInstType, 128);

template <size_t Size, typename T>
using Batch = std::conditional_t<simd::detail::VecBatch<Size, T, IT256, true>::NumExtra == 0,
                                 simd::detail::VecBatch<Size, T, IT256>,
                                 simd::detail::VecBatch<Size, T, IT128>>;

template <typename FT, typename TT, typename Batch>
using Convert = simd::detail::VecCvt<FT, TT, Batch::Inst>;

template <typename Batch>
using I8LS = simd::detail::VecLoadStore<int8_t, Alignment, Batch::Inst>;
template <typename Batch>
using I16LS = simd::detail::VecLoadStore<int16_t, Alignment, Batch::Inst>;
template <typename Batch>
using I32LS = simd::detail::VecLoadStore<int32_t, Alignment, Batch::Inst>;
template <typename Batch>
using F32LS = simd::detail::VecLoadStore<float, Alignment, Batch::Inst>;

template <typename Batch>
using I8Op = simd::detail::VecOp<int8_t, Batch::Inst>;
template <typename Batch>
using I16Op = simd::detail::VecOp<int16_t, Batch::Inst>;
template <typename Batch>
using I32Op = simd::detail::VecOp<int32_t, Batch::Inst>;
template <typename Batch>
using F32Op = simd::detail::VecOp<float, Batch::Inst>;

struct Mix9LiteBinaryWeightLoader : WeightLoader<Mix9LiteWeight>
{
    std::unique_ptr<Mix9LiteWeight> load(std::istream &in, Evaluation::EmptyLoadArgs args)
    {
        auto w = std::make_unique<Mix9LiteWeight>();

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

    void read_compressed_mapping(std::istream &in, Mix9LiteWeight &w)
    {
        constexpr int      MappingBits   = 10;
        constexpr uint64_t MappingMask   = (1 << MappingBits) - 1;
        constexpr uint16_t ExtensionMask = 1 << (MappingBits - 1);
        constexpr uint16_t ExtensionBits = ~static_cast<uint16_t>(MappingMask);

        uint64_t u64val    = 0;
        int      bits_left = 0;
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

                w.mapping[i][j] = feature;
            }
    }

    void preprocess(Mix9LiteWeight &w)
    {
        for (int bucketIdx = 0; bucketIdx < NumHeadBucket; bucketIdx++) {
            auto &b = w.buckets[bucketIdx];
            preprocessLinear<PolicyDim * 2, FeatureDim>(b.policy_pwconv_layer_l1_weight);
            simd::preprocessDynamicWeightLinear<PolicyPWConvDim,
                                                PolicyDim,
                                                int16_t,
                                                PolicyDim * 2,
                                                0,
                                                Alignment,
                                                Batch<PolicyPWConvDim, int32_t>::Inst>(
                b.policy_pwconv_layer_l2_weight,
                b.policy_pwconv_layer_l2_bias);
            preprocessLinear<PolicyPWConvDim * PolicyDim + PolicyPWConvDim, PolicyDim * 2>(
                b.policy_pwconv_layer_l2_weight);
            preprocess(b.value_corner);
            preprocess(b.value_edge);
            preprocess(b.value_center);
            preprocess(b.value_quad);
            preprocessLinear<ValueDim, FeatureDim + ValueDim * 4>(b.value_l1_weight);
            preprocessLinear<ValueDim, ValueDim>(b.value_l2_weight);
            preprocessLinear<4, ValueDim>(b.value_l3_weight);
        }
    }

    template <int OutSize, int InSize>
    void preprocess(StarBlockWeight<OutSize, InSize> &b)
    {
        preprocessLinear<OutSize * 2, InSize>(b.value_corner_up1_weight);
        preprocessLinear<OutSize * 2, InSize>(b.value_corner_up2_weight);
        preprocessLinear<OutSize, OutSize>(b.value_corner_down_weight);
    }

    template <int OutSize, int InSize, typename InputType = int8_t>
    inline void preprocessLinear(InputType weight[OutSize * InSize])
    {
        simd::preprocessLinear<OutSize, InSize, Alignment, Batch<OutSize, int32_t>::Inst>(weight);
    }
};

static Evaluation::WeightRegistry<Mix9LiteBinaryWeightLoader> Mix9LiteWeightRegistry;

template <int  OutSize,
          int  InSize,
          bool SignedInput   = false,
          bool Bias          = true,
          bool PreReLU       = false,
          bool PostReLU      = false,
          typename AccType   = int32_t,
          typename InputType = int8_t>
FORCE_INLINE AccType *linear(AccType         *output,
                             const InputType *input,
                             const InputType  weight[OutSize * InSize],
                             const AccType    bias[OutSize])
{
    return simd::linear<OutSize,
                        InSize,
                        SignedInput,
                        Bias,
                        PreReLU,
                        PostReLU,
                        Alignment,
                        Batch<OutSize, AccType>::Inst,
                        AccType,
                        InputType>(output, input, weight, bias);
}

template <int  Size,
          int  Divisor,
          bool NoReLU         = false,
          typename OutputType = int8_t,
          typename InputType  = int32_t>
FORCE_INLINE OutputType *crelu(OutputType output[Size], const InputType input[Size])
{
    return simd::crelu<Size, Divisor, NoReLU, Alignment, Batch<Size, OutputType>::Inst>(output,
                                                                                        input);
}

template <int OutSize, int InSize>
inline void
starBlock(int8_t output[OutSize], int8_t input[InSize], const StarBlockWeight<OutSize, InSize> &w)
{
    alignas(Alignment) int32_t upi32[OutSize * 2];
    alignas(Alignment) int8_t  up1[OutSize * 2], up2[OutSize * 2];
    linear<OutSize * 2, InSize>(upi32, input, w.value_corner_up1_weight, w.value_corner_up1_bias);
    crelu<OutSize * 2, 128>(up1, upi32);

    linear<OutSize * 2, InSize>(upi32, input, w.value_corner_up2_weight, w.value_corner_up2_bias);
    crelu<OutSize * 2, 128, true>(up2, upi32);

    alignas(Alignment) int8_t                               dotsum[OutSize];
    typedef Batch<OutSize, int8_t>                          B;
    typedef simd::detail::VecPack<int16_t, int8_t, B::Inst> I16Pack;
    for (int i = 0; i < B::NumBatch; i++) {
        auto in10 = I8LS<B>::load(up1 + (2 * i + 0) * B::RegWidth);  // unsigned
        auto in11 = I8LS<B>::load(up1 + (2 * i + 1) * B::RegWidth);  // unsigned
        auto in20 = I8LS<B>::load(up2 + (2 * i + 0) * B::RegWidth);  // signed
        auto in21 = I8LS<B>::load(up2 + (2 * i + 1) * B::RegWidth);  // signed

        auto dotsum0i16 = I8Op<B>::dot2_u7i8(in10, in20);
        auto dotsum1i16 = I8Op<B>::dot2_u7i8(in11, in21);
        dotsum0i16      = I16Op<B>::template srai<floorLog2(128)>(dotsum0i16);
        dotsum1i16      = I16Op<B>::template srai<floorLog2(128)>(dotsum1i16);
        auto dotsumi8   = I16Pack::packs_permuted(dotsum0i16, dotsum1i16);

        I8LS<B>::store(dotsum + i * B::RegWidth, dotsumi8);
    }

    alignas(Alignment) int32_t outputi32[OutSize];
    linear<OutSize, OutSize, true>(outputi32,
                                   dotsum,
                                   w.value_corner_down_weight,
                                   w.value_corner_down_bias);
    crelu<OutSize, 128>(output, outputi32);
}

}  // namespace

namespace Evaluation::mix9lite {

Mix9LiteAccumulator::Mix9LiteAccumulator(int boardSize)
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
        MemAlloc::alignedArrayAlloc<std::array<int16_t, FeatureDim>, Alignment>(nOuterChanges);

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

Mix9LiteAccumulator::~Mix9LiteAccumulator()
{
    MemAlloc::alignedFree(valueSumTable);
    delete[] versionChangeNumTable;
    delete[] versionInnerIndexTable;
    delete[] versionOuterIndexTable;
    delete[] indexTable;
    MemAlloc::alignedFree(mapSum);
    MemAlloc::alignedFree(mapConv);
}

void Mix9LiteAccumulator::initIndexTable()
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

            // DX[1]=0, DY[1]=1
            idxs[0] = get_boarder_encoding(distx0, distx1);
            // DX[1]=0, DY[1]=1
            idxs[1] = get_boarder_encoding(disty0, disty1);
            // DX[2]=1, DY[2]=1
            idxs[2] = get_boarder_encoding(std::min(distx0, disty0), std::min(distx1, disty1));
            // DX[3]=1, DY[3]=-1
            idxs[3] = get_boarder_encoding(std::min(distx0, disty1), std::min(distx1, disty0));

            assert(idxs[0] < ShapeNum);
            assert(idxs[1] < ShapeNum);
            assert(idxs[2] < ShapeNum);
            assert(idxs[3] < ShapeNum);
        }
}

void Mix9LiteAccumulator::clear(const Mix9LiteWeight &w)
{
    if (currentVersion == -1) {
        // Init mapConv to bias
        for (int i = 0; i < outerBoardSize * outerBoardSize; i++)
            simd::copy<FeatureDim, int16_t, Alignment, Batch<FeatureDim, int16_t>::Inst>(
                mapConv[i].data(),
                w.feature_dwconv_bias);
        // Init valueSum to zeros
        auto &valueSum = valueSumTable[0];
        simd::zero<FeatureDim, int32_t, Alignment, Batch<FeatureDim, int32_t>::Inst>(
            valueSum.global.data());
        for (int i = 0; i < ValueSumType::NGroup; i++)
            for (int j = 0; j < ValueSumType::NGroup; j++)
                simd::zero<FeatureDim, int32_t, Alignment, Batch<FeatureDim, int32_t>::Inst>(
                    valueSum.group[i][j].data());

        typedef Batch<FeatureDim, int16_t> FeatB;
        typedef Batch<FeatureDim, int32_t> VSumB;

        for (int y = 0, innerIdx = 0; y < boardSize; y++) {
            for (int x = 0; x < boardSize; x++, innerIdx++) {
                // Init mapSum from four directions
                simd::zero<FeatureDim, int16_t, Alignment, Batch<FeatureDim, int16_t>::Inst>(
                    mapSum[innerIdx].data());
                for (int dir = 0; dir < 4; dir++)
                    simd::add<FeatureDim, int16_t, Alignment, Batch<FeatureDim, int16_t>::Inst>(
                        mapSum[innerIdx].data(),
                        mapSum[innerIdx].data(),
                        w.mapping[indexTable[innerIdx][dir]]);

                // Init mapConv from mapSum
                for (int b = 0; b < FeatB::NumBatch; b++) {
                    auto feature =
                        I16LS<FeatB>::load(mapSum[innerIdx].data() + b * FeatB::RegWidth);
                    feature = I16Op<FeatB>::max(feature, I16Op<FeatB>::setzero());
                    feature = I16Op<FeatB>::slli<2>(feature);  // mul 4
                    // Apply feature depthwise conv
                    for (int dy = 0; dy <= 2; dy++) {
                        int yi = y + dy;
                        for (int dx = 0; dx <= 2; dx++) {
                            int xi       = x + dx;
                            int outerIdx = xi + yi * outerBoardSize;

                            auto *convWeightBase = w.feature_dwconv_weight[8 - dy * 3 - dx];
                            auto  convW = I16LS<FeatB>::load(convWeightBase + b * FeatB::RegWidth);
                            auto  deltaFeat = I16Op<FeatB>::mulhi(convW, feature);
                            auto  convPtr   = mapConv[outerIdx].data() + b * FeatB::RegWidth;
                            auto  convFeat  = I16LS<FeatB>::load(convPtr);
                            convFeat        = I16Op<FeatB>::add(convFeat, deltaFeat);
                            I16LS<FeatB>::store(convPtr, convFeat);
                        }
                    }
                }
            }
        }

        // Init valueSum by adding all dwconv value features
        for (int y = 0, outerIdx = outerBoardSize + 1; y < boardSize; y++, outerIdx += 2) {
            for (int x = 0; x < boardSize; x++, outerIdx++) {
                for (int b = 0; b < FeatB::NumBatch; b++) {
                    auto feature =
                        I16LS<FeatB>::load(mapConv[outerIdx].data() + b * FeatB::RegWidth);
                    feature       = I16Op<FeatB>::max(feature, I16Op<FeatB>::setzero());  // relu
                    auto [v0, v1] = Convert<int16_t, int32_t, FeatB>::convert(feature);

                    auto addToAccumulator =
                        [&, v0_ = v0, v1_ = v1](std::array<int32_t, FeatureDim> &vSum) {
                            auto vSumPtr = vSum.data() + b * 2 * VSumB::RegWidth;
                            auto vSum0   = I32LS<FeatB>::load(vSumPtr);
                            auto vSum1   = I32LS<FeatB>::load(vSumPtr + VSumB::RegWidth);
                            vSum0        = I32Op<FeatB>::add(vSum0, v0_);
                            vSum1        = I32Op<FeatB>::add(vSum1, v1_);
                            I32LS<FeatB>::store(vSumPtr, vSum0);
                            I32LS<FeatB>::store(vSumPtr + VSumB::RegWidth, vSum1);
                        };
                    addToAccumulator(valueSum.global);
                    addToAccumulator(valueSum.group[groupIndex[y]][groupIndex[x]]);
                }
            }
        }
    }

    // Reset version and init version table to be zeros
    currentVersion = 0;
}

void Mix9LiteAccumulator::move(const Mix9LiteWeight &w, Color pieceColor, int x, int y)
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

    typedef Batch<FeatureDim, int16_t> FeatB;
    typedef Batch<FeatureDim, int32_t> VSumB;

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
            for (int b = 0; b < FeatB::NumBatch; b++)
                I16LS<FeatB>::store(mapConv[newMapConvIdx].data() + b * FeatB::RegWidth,
                                    I16Op<FeatB>::setzero());
            newMapConvIdx++;
        }
    }

    struct OnePointChange
    {
        int8_t   x;
        int8_t   y;
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
    I32Op<VSumB>::R vSumGlobal[VSumB::NumBatch];
    I32Op<VSumB>::R vSumGroup[ValueSumType::NGroup][ValueSumType::NGroup][VSumB::NumBatch];
    for (int b = 0; b < VSumB::NumBatch; b++)
        vSumGlobal[b] = I32Op<VSumB>::setzero();
    for (int i = 0; i < ValueSumType::NGroup; i++)
        for (int j = 0; j < ValueSumType::NGroup; j++)
            for (int b = 0; b < VSumB::NumBatch; b++)
                vSumGroup[i][j][b] = I32Op<VSumB>::setzero();

    // Incremental update feature sum
    for (int i = 0; i < changeCount; i++) {
        const OnePointChange &c = changeTable[i];
        if (i + 1 < changeCount) {
            multiPrefetch<FeatureDim * sizeof(int16_t)>(w.mapping[changeTable[i + 1].oldShape]);
            multiPrefetch<FeatureDim * sizeof(int16_t)>(w.mapping[changeTable[i + 1].newShape]);
        }

        // Update mapSum
        I16Op<FeatB>::R oldFeats[FeatB::NumBatch];
        I16Op<FeatB>::R newFeats[FeatB::NumBatch];
        for (int b = 0; b < FeatB::NumBatch; b++) {
            auto newMapFeat = I16LS<FeatB>::load(w.mapping[c.newShape] + b * FeatB::RegWidth);
            auto oldMapFeat = I16LS<FeatB>::load(w.mapping[c.oldShape] + b * FeatB::RegWidth);
            oldFeats[b]     = I16LS<FeatB>::load(mapSum[c.oldMapIdx].data() + b * FeatB::RegWidth);
            newFeats[b]     = I16Op<FeatB>::sub(oldFeats[b], oldMapFeat);
            newFeats[b]     = I16Op<FeatB>::add(newFeats[b], newMapFeat);
            I16LS<FeatB>::store(mapSum[c.newMapIdx].data() + b * FeatB::RegWidth, newFeats[b]);
            oldFeats[b] = I16Op<FeatB>::max(oldFeats[b], I16Op<FeatB>::setzero());
            newFeats[b] = I16Op<FeatB>::max(newFeats[b], I16Op<FeatB>::setzero());
        }

        // Update mapConv
        for (int b = 0; b < FeatB::NumBatch; b++) {
            oldFeats[b] = I16Op<FeatB>::slli<2>(oldFeats[b]);  // mul 4
            newFeats[b] = I16Op<FeatB>::slli<2>(newFeats[b]);  // mul 4
        }
        for (int dy = 0, outerIdxBase = c.y * outerBoardSize + c.x; dy <= 2;
             dy++, outerIdxBase += outerBoardSize) {
            for (int dx = 0; dx <= 2; dx++) {
                int   outerIdx       = dx + outerIdxBase;
                int   mapConvIdx     = versionOuterIndexTable[outerVersionIdxBase + outerIdx];
                auto *convWeightBase = w.feature_dwconv_weight[8 - dy * 3 - dx];
                auto *convBase       = mapConv[mapConvIdx].data();

                for (int b = 0; b < FeatB::NumBatch; b++) {
                    auto convW      = I16LS<FeatB>::load(convWeightBase + b * FeatB::RegWidth);
                    auto deltaConvF = I16Op<FeatB>::sub(I16Op<FeatB>::mulhi(convW, newFeats[b]),
                                                        I16Op<FeatB>::mulhi(convW, oldFeats[b]));

                    auto convPtr  = convBase + b * FeatB::RegWidth;
                    auto oldConvF = I16LS<FeatB>::load(convPtr);
                    auto newConvF = I16Op<FeatB>::add(oldConvF, deltaConvF);
                    I16LS<FeatB>::store(convPtr, newConvF);
                }
            }
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
            for (int b = 0; b < FeatB::NumBatch; b++) {
                auto oldConvF =
                    I16LS<FeatB>::load(mapConv[oldMapConvIdx].data() + b * FeatB::RegWidth);
                auto deltaConvF =
                    I16LS<FeatB>::load(mapConv[newMapConvIdx].data() + b * FeatB::RegWidth);
                auto newConvF = I16Op<FeatB>::add(oldConvF, deltaConvF);
                I16LS<FeatB>::store(mapConv[newMapConvIdx].data() + b * FeatB::RegWidth, newConvF);
                oldConvF      = I16Op<FeatB>::max(oldConvF, I16Op<FeatB>::setzero());  // relu
                newConvF      = I16Op<FeatB>::max(newConvF, I16Op<FeatB>::setzero());  // relu
                auto deltaF   = I16Op<FeatB>::sub(newConvF, oldConvF);
                auto [v0, v1] = Convert<int16_t, int32_t, FeatB>::convert(deltaF);

                const int offset            = 2 * b;
                vSumGlobal[offset + 0]      = I32Op<FeatB>::add(vSumGlobal[offset + 0], v0);
                vSumGlobal[offset + 1]      = I32Op<FeatB>::add(vSumGlobal[offset + 1], v1);
                vSumGroup[i][j][offset + 0] = I32Op<FeatB>::add(vSumGroup[i][j][offset + 0], v0);
                vSumGroup[i][j][offset + 1] = I32Op<FeatB>::add(vSumGroup[i][j][offset + 1], v1);
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
        auto vOld = I32LS<VSumB>::load(valueSumOld.global.data() + b * VSumB::RegWidth);
        auto vNew = I32Op<VSumB>::add(vOld, vSumGlobal[b]);
        I32LS<VSumB>::store(valueSumNew.global.data() + b * VSumB::RegWidth, vNew);
    }
    for (int i = 0; i < ValueSumType::NGroup; i++)
        for (int j = 0; j < ValueSumType::NGroup; j++)
            for (int b = 0; b < VSumB::NumBatch; b++) {
                auto vOld =
                    I32LS<VSumB>::load(valueSumOld.group[i][j].data() + b * VSumB::RegWidth);
                auto vNew = I32Op<VSumB>::add(vOld, vSumGroup[i][j][b]);
                I32LS<VSumB>::store(valueSumNew.group[i][j].data() + b * VSumB::RegWidth, vNew);
            }
}

std::tuple<float, float, float> Mix9LiteAccumulator::evaluateValue(const Mix9LiteWeight &w)
{
    const auto &valueSum = valueSumTable[currentVersion];
    const auto &bucket   = w.buckets[getBucketIndex()];

    // convert value sum from int32 to int8
    // global feature sum
    alignas(Alignment) int8_t layer0[FeatureDim + ValueDim * 4];
    crelu<FeatureDim, 256, true>(layer0, valueSum.global.data());
    // group feature sum
    alignas(Alignment) int8_t group0[ValueSumType::NGroup][ValueSumType::NGroup][FeatureDim];
    for (int i = 0; i < ValueSumType::NGroup; i++)
        for (int j = 0; j < ValueSumType::NGroup; j++)
            crelu<FeatureDim, 32, true>(group0[i][j], valueSum.group[i][j].data());

    // group linear layer
    alignas(Alignment) int8_t group1[ValueSumType::NGroup][ValueSumType::NGroup][ValueDim];

    starBlock<ValueDim, FeatureDim>(group1[0][0], group0[0][0], bucket.value_corner);
    starBlock<ValueDim, FeatureDim>(group1[0][2], group0[0][2], bucket.value_corner);
    starBlock<ValueDim, FeatureDim>(group1[2][0], group0[2][0], bucket.value_corner);
    starBlock<ValueDim, FeatureDim>(group1[2][2], group0[2][2], bucket.value_corner);

    starBlock<ValueDim, FeatureDim>(group1[0][1], group0[0][1], bucket.value_edge);
    starBlock<ValueDim, FeatureDim>(group1[1][0], group0[1][0], bucket.value_edge);
    starBlock<ValueDim, FeatureDim>(group1[1][2], group0[1][2], bucket.value_edge);
    starBlock<ValueDim, FeatureDim>(group1[2][1], group0[2][1], bucket.value_edge);

    starBlock<ValueDim, FeatureDim>(group1[1][1], group0[1][1], bucket.value_center);

    // average pooling
    alignas(Alignment) int8_t group2[2][2][ValueDim];
    using B = Batch<ValueDim, int8_t>;
    for (int b = 0; b < B::NumBatch; b++) {
        auto v00 = I8LS<B>::load(group1[0][0] + b * B::RegWidth);
        auto v01 = I8LS<B>::load(group1[0][1] + b * B::RegWidth);
        auto v02 = I8LS<B>::load(group1[0][2] + b * B::RegWidth);
        auto v10 = I8LS<B>::load(group1[1][0] + b * B::RegWidth);
        auto v11 = I8LS<B>::load(group1[1][1] + b * B::RegWidth);
        auto v12 = I8LS<B>::load(group1[1][2] + b * B::RegWidth);
        auto v20 = I8LS<B>::load(group1[2][0] + b * B::RegWidth);
        auto v21 = I8LS<B>::load(group1[2][1] + b * B::RegWidth);
        auto v22 = I8LS<B>::load(group1[2][2] + b * B::RegWidth);

        auto q00 = I8Op<B>::avg(I8Op<B>::avg(v00, v01), I8Op<B>::avg(v10, v11));
        auto q01 = I8Op<B>::avg(I8Op<B>::avg(v01, v02), I8Op<B>::avg(v11, v12));
        auto q10 = I8Op<B>::avg(I8Op<B>::avg(v10, v11), I8Op<B>::avg(v20, v21));
        auto q11 = I8Op<B>::avg(I8Op<B>::avg(v11, v12), I8Op<B>::avg(v21, v22));

        I8LS<B>::store(group2[0][0] + b * B::RegWidth, q00);
        I8LS<B>::store(group2[0][1] + b * B::RegWidth, q01);
        I8LS<B>::store(group2[1][0] + b * B::RegWidth, q10);
        I8LS<B>::store(group2[1][1] + b * B::RegWidth, q11);
    }

    // quadrant linear layer
    starBlock<ValueDim, ValueDim>(layer0 + FeatureDim + 0 * ValueDim,
                                  group2[0][0],
                                  bucket.value_quad);
    starBlock<ValueDim, ValueDim>(layer0 + FeatureDim + 1 * ValueDim,
                                  group2[0][1],
                                  bucket.value_quad);
    starBlock<ValueDim, ValueDim>(layer0 + FeatureDim + 2 * ValueDim,
                                  group2[1][0],
                                  bucket.value_quad);
    starBlock<ValueDim, ValueDim>(layer0 + FeatureDim + 3 * ValueDim,
                                  group2[1][1],
                                  bucket.value_quad);

    // linear 1
    alignas(Alignment) int32_t layer1i32[ValueDim];
    alignas(Alignment) int8_t  layer1[ValueDim];
    linear<ValueDim, FeatureDim + ValueDim * 4>(layer1i32,
                                                layer0,
                                                bucket.value_l1_weight,
                                                bucket.value_l1_bias);
    crelu<ValueDim, 128>(layer1, layer1i32);

    // linear 2
    alignas(Alignment) int32_t layer2i32[ValueDim];
    alignas(Alignment) int8_t  layer2[ValueDim];
    linear<ValueDim, ValueDim>(layer2i32, layer1, bucket.value_l2_weight, bucket.value_l2_bias);
    crelu<ValueDim, 128>(layer2, layer2i32);

    // linear 3 final
    alignas(Alignment) int32_t layer3i32[4];
    linear<4, ValueDim>(layer3i32, layer2, bucket.value_l3_weight, bucket.value_l3_bias);

    const float scale = 1.0f / (128 * 128);
    return {layer3i32[0] * scale, layer3i32[1] * scale, layer3i32[2] * scale};
}

void Mix9LiteAccumulator::evaluatePolicy(const Mix9LiteWeight &w, PolicyBuffer &policyBuffer)
{
    const auto &valueSum = valueSumTable[currentVersion];
    const auto &bucket   = w.buckets[getBucketIndex()];

    alignas(Alignment) int8_t layer0[FeatureDim];
    crelu<FeatureDim, 256, true>(layer0, valueSum.global.data());

    // policy pwconv weight layer
    alignas(Alignment) int32_t layer1i32[PolicyDim * 2];
    alignas(Alignment) int8_t  layer1[PolicyDim * 2];
    linear<PolicyDim * 2, FeatureDim>(layer1i32,
                                      layer0,
                                      bucket.policy_pwconv_layer_l1_weight,
                                      bucket.policy_pwconv_layer_l1_bias);
    crelu<PolicyDim * 2, 128>(layer1, layer1i32);

    alignas(Alignment) int32_t layer2i32[PolicyPWConvDim * PolicyDim + PolicyPWConvDim];
    alignas(Alignment) int16_t pwconvWeighti16[PolicyPWConvDim * PolicyDim];
    alignas(Alignment) int32_t pwconvBiasi32[PolicyPWConvDim];
    linear<PolicyPWConvDim * PolicyDim + PolicyPWConvDim, PolicyDim * 2>(
        layer2i32,
        layer1,
        bucket.policy_pwconv_layer_l2_weight,
        bucket.policy_pwconv_layer_l2_bias);
    crelu<PolicyPWConvDim * PolicyDim, 1, true>(pwconvWeighti16, layer2i32);
    // To get pwconv bias, we need to scale the output of layer2 by 128
    {
        typedef Batch<PolicyPWConvDim, int32_t> B;
        for (int i = 0; i < B::NumBatch; i++) {
            auto data = I32LS<B>::load(layer2i32 + PolicyPWConvDim * PolicyDim + i * B::RegWidth);
            data      = I32Op<B>::slli<7>(data);  // scale by 128
            I32LS<B>::store(pwconvBiasi32 + i * B::RegWidth, data);
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
            static_assert(PolicyDim <= FeatureDim,
                          "Assume PolicyDim <= FeatureDim in evaluatePolicy()!");
            alignas(Alignment) int32_t policyLayer1i32[PolicyPWConvDim];
            linear<PolicyPWConvDim, PolicyDim, false, true, true>(policyLayer1i32,
                                                                  mapConv[mapConvIdx].data(),
                                                                  pwconvWeighti16,
                                                                  pwconvBiasi32);

            // Apply relu, convert to float and accumulate all channels of pwconv feature
            typedef Batch<PolicyPWConvDim, float> PWConvB;
            auto                                  policyAccum = F32Op<PWConvB>::setzero();
            for (int i = 0; i < PWConvB::NumBatch; i++) {
                auto featI32 = I32LS<PWConvB>::load(policyLayer1i32 + i * PWConvB::RegWidth);
                featI32      = I32Op<PWConvB>::max(featI32, I32Op<PWConvB>::setzero());
                auto featF32 = Convert<int32_t, float, PWConvB>::convert1(featI32);
                auto outputW =
                    F32LS<PWConvB>::load(bucket.policy_output_weight + i * PWConvB::RegWidth);
                policyAccum = F32Op<PWConvB>::fmadd(featF32, outputW, policyAccum);
            }

            float policy = F32Op<PWConvB>::reduceadd(policyAccum) + bucket.policy_output_bias;
            policyBuffer(innerIdx) = policy;
        }
    }
}

Mix9LiteEvaluator::Mix9LiteEvaluator(int                   boardSize,
                                     Rule                  rule,
                                     std::filesystem::path blackWeightPath,
                                     std::filesystem::path whiteWeightPath)
    : Evaluator(boardSize, rule)
    , weight {nullptr, nullptr}
{
    CompressedWrapper<StandardHeaderParserWarpper<Mix9LiteBinaryWeightLoader>> loader(
        Compressor::Type::LZ4_DEFAULT);

    if (boardSize > 22)
        throw UnsupportedBoardSizeError(boardSize);

    std::filesystem::path currentWeightPath;
    loader.setHeaderValidator([&](StandardHeader header) -> bool {
        constexpr uint32_t ArchHash = ArchHashBase
                                      ^ (((FeatureDim / 8) << 20) | ((ValueDim / 8) << 14)
                                         | ((PolicyDim / 8) << 8) | (FeatureDim / 8));
        if (header.archHash != ArchHash)
            throw IncompatibleWeightFileError("incompatible architecture in weight file.");

        if (!contains(header.supportedRules, rule))
            throw UnsupportedRuleError(rule);

        if (!contains(header.supportedBoardSizes, boardSize))
            throw UnsupportedBoardSizeError(boardSize);

        if (Config::MessageMode != MsgMode::NONE)
            MESSAGEL("mix9litennue: load weight from " << pathToConsoleString(currentWeightPath));
        return true;
    });

    for (const auto &[weightSide, weightPath] : {
             std::make_pair(BLACK, blackWeightPath),
             std::make_pair(WHITE, whiteWeightPath),
         }) {
        currentWeightPath  = weightPath;
        weight[weightSide] = Mix9LiteWeightRegistry.loadWeightFromFile(loader, weightPath);
        if (!weight[weightSide])
            throw std::runtime_error("failed to load nnue weight from "
                                     + pathToConsoleString(weightPath));
    }

    accumulator[BLACK] = std::make_unique<Mix9LiteAccumulator>(boardSize);
    accumulator[WHITE] = std::make_unique<Mix9LiteAccumulator>(boardSize);

    int numCells = boardSize * boardSize;
    moveCache[BLACK].reserve(numCells);
    moveCache[WHITE].reserve(numCells);
}

Mix9LiteEvaluator::~Mix9LiteEvaluator()
{
    if (weight[BLACK])
        Mix9LiteWeightRegistry.unloadWeight(weight[BLACK]);
    if (weight[WHITE])
        Mix9LiteWeightRegistry.unloadWeight(weight[WHITE]);
}

void Mix9LiteEvaluator::initEmptyBoard()
{
    moveCache[BLACK].clear();
    moveCache[WHITE].clear();
    accumulator[BLACK]->clear(*weight[BLACK]);
    accumulator[WHITE]->clear(*weight[WHITE]);
}

void Mix9LiteEvaluator::beforeMove(const Board &board, Pos pos)
{
    addCache(board.sideToMove(), pos.x(), pos.y(), false);
}

void Mix9LiteEvaluator::afterUndo(const Board &board, Pos pos)
{
    addCache(board.sideToMove(), pos.x(), pos.y(), true);
}

ValueType Mix9LiteEvaluator::evaluateValue(const Board &board)
{
    Color self = board.sideToMove(), oppo = ~self;

    // Apply all incremental update for both sides and calculate value
    clearCache(self);
    auto [win, loss, draw] = accumulator[self]->evaluateValue(*weight[self]);

    return ValueType(win, loss, draw, true);
}

void Mix9LiteEvaluator::evaluatePolicy(const Board &board, PolicyBuffer &policyBuffer)
{
    Color self = board.sideToMove();

    // Apply all incremental update and calculate policy
    clearCache(self);
    accumulator[self]->evaluatePolicy(*weight[self], policyBuffer);
}

void Mix9LiteEvaluator::clearCache(Color side)
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

void Mix9LiteEvaluator::addCache(Color side, int x, int y, bool isUndo)
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

}  // namespace Evaluation::mix9lite
