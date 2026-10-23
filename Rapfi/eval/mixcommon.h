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

#include "../config.h"
#include "../core/compressor.h"
#include "../core/iohelper.h"
#include "../core/platform.h"
#include "../core/types.h"
#include "../core/utils.h"
#include "evaluator.h"
#include "simdops.h"
#include "weightloader.h"

#include <algorithm>
#include <array>
#include <cassert>
#include <cstdint>
#include <filesystem>
#include <istream>
#include <vector>

/// Machinery shared by the mix-family NNUE implementations (mix9svq, mix10, ...).
/// All mix networks share the same board representation: per-cell line-shape codes in
/// four directions, mapped to int16 features, summed and depthwise-convolved, with a
/// version-stacked incremental state so that undo is O(1). Architectures differ in how
/// the mapping table is stored and in their value/policy heads, which stay per-arch.
namespace Evaluation::mixnet {

/// Number of distinct line-shape codes: base-3 codes of a length-11 line, plus the
/// extended codes that encode distance to the board boundary (see initIndexTable()).
inline constexpr int ShapeNum = 442503;

/// Power3[i] = 3^i, used to compose base-3 line-shape codes.
inline constexpr auto Power3 = []() {
    auto pow3 = std::array<int, 16> {};
    for (size_t i = 0; i < pow3.size(); i++)
        pow3[i] = power(3, i);
    return pow3;
}();

/// Direction steps of the four update lines (right, down, up-right, down-right).
inline constexpr int DX[4] = {1, 0, 1, 1};
inline constexpr int DY[4] = {0, 1, -1, 1};

/// Max total inner (board cell) and outer (board cell plus one-cell apron) point
/// changes over a whole game, indexed by board size. These bound the number of rows
/// ever appended to the mapSum/mapConv tables of MixAccumulatorState.
inline constexpr int MaxInnerChanges[23] = {1,    6,     33,    102,   233,   446,   761,  1166,
                                            1661, 2246,  2921,  3686,  4541,  5486,  6521, 7646,
                                            8861, 10166, 11561, 13046, 14621, 16286, 18041};
inline constexpr int MaxOuterChanges[23] = {5,     11,    33,    107,   293,   675,   1361,  2483,
                                            3945,  5747,  7889,  10371, 13193, 16355, 19857, 23699,
                                            27881, 32403, 37265, 42467, 48009, 53891, 60113};

/// Weights of one fully-connected layer (int8 weights + int32 biases).
template <int OutSize, int InSize>
struct FCWeight
{
    int8_t  weight[OutSize * InSize];
    int32_t bias[OutSize];
};

/// Reads sign-extended Bits-wide integers from a stream of packed little-endian
/// uint64 words, as produced by the mix weight-file mapping compressor.
template <int Bits>
struct PackedSignedIntStream
{
    static_assert(0 < Bits && Bits <= 16);

    uint64_t buffer   = 0;
    int      bitsLeft = 0;

    int16_t next(std::istream &in)
    {
        constexpr uint64_t Mask    = (1 << Bits) - 1;
        constexpr uint16_t SignBit = 1 << (Bits - 1);
        constexpr uint16_t SignExt = static_cast<uint16_t>(~Mask);

        if (bitsLeft < Bits) {
            uint64_t next64;
            in.read(reinterpret_cast<char *>(&next64), sizeof(next64));

            buffer |= next64 << bitsLeft;
            uint16_t masked = static_cast<uint16_t>(buffer & Mask);

            buffer   = next64 >> (Bits - bitsLeft);
            bitsLeft = 64 - (Bits - bitsLeft);
            return static_cast<int16_t>(((masked & SignBit) ? SignExt : 0) | masked);
        }
        else {
            uint16_t masked = static_cast<uint16_t>(buffer & Mask);

            buffer >>= Bits;
            bitsLeft -= Bits;
            return static_cast<int16_t>(((masked & SignBit) ? SignExt : 0) | masked);
        }
    }
};

// SIMD shorthands shared by the mix implementations, fixed to the native instruction
// set and alignment this binary is compiled for.
inline constexpr int                   Alignment = simd::NativeAlignment;
inline constexpr simd::InstructionType IT        = simd::NativeInstType;
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

/// Version-stacked incremental accumulator state shared by all mix networks.
///
/// Features are never updated in place. Instead, move() appends the changed rows to
/// mapSum (per-cell feature sums) and mapConv (depthwise-convolved features) and
/// bumps the version; each version keeps a full copy-on-write snapshot of which row
/// every cell currently points to:
///   versionInnerIndexTable[v*H*W + cell]        -> row in mapSum/indexTable
///   versionOuterIndexTable[v*(H+2)*(W+2) + cell] -> row in mapConv
/// versionChangeNumTable[v] records how many inner/outer rows are in use after
/// reaching version v, i.e. where a move() from version v starts appending.
/// undo() is then just a version decrement.
template <typename ValueSumT, int FeatureDim, int FeatDWConvDim>
struct MixAccumulatorState
{
    struct ChangeNum
    {
        uint16_t inner, outer;
    };

    /// Value feature sum of the full board
    ValueSumT *valueSumTable;             // [H*W+1] (aligned)
    ChangeNum *versionChangeNumTable;     // [H*W+1] (unaligned) num inner and outer changes
    uint16_t  *versionInnerIndexTable;    // [H*W+1, H*W] (unaligned)
    uint16_t  *versionOuterIndexTable;    // [H*W+1, (H+2)*(W+2)] (unaligned)
    /// Index table to convert line shape to map feature
    std::array<uint32_t, 4> *indexTable;  // [N_inner, 4] (unaligned)
    /// Summed map feature of four directions
    std::array<int16_t, FeatureDim> *mapSum;  // [N_inner, FeatureDim] (aligned)
    /// Map feature after depthwise conv
    std::array<int16_t, FeatDWConvDim> *mapConv;  // [N_outer, DWConvDim] (aligned)

    int    boardSize;
    int    outerBoardSize;  // (boardSize + 2)
    int    currentVersion;
    int8_t groupIndex[32];  // row/column -> 3x3 value group index

    explicit MixAccumulatorState(int boardSize)
        : boardSize(boardSize)
        , outerBoardSize(boardSize + 2)
        , currentVersion(-1)
    {
        assert(0 < boardSize && boardSize <= 22);
        int nCells        = boardSize * boardSize;
        int nInnerChanges = MaxInnerChanges[boardSize];
        int nOuterChanges = MaxOuterChanges[boardSize];

        valueSumTable          = MemAlloc::alignedArrayAlloc<ValueSumT, Alignment>(nCells + 1);
        versionChangeNumTable  = new ChangeNum[nCells + 1];
        versionInnerIndexTable = new uint16_t[(nCells + 1) * nCells];
        versionOuterIndexTable = new uint16_t[(nCells + 2) * outerBoardSize * outerBoardSize];
        indexTable             = new std::array<uint32_t, 4>[nInnerChanges];
        mapSum =
            MemAlloc::alignedArrayAlloc<std::array<int16_t, FeatureDim>, Alignment>(nInnerChanges);
        mapConv = MemAlloc::alignedArrayAlloc<std::array<int16_t, FeatDWConvDim>, Alignment>(
            nOuterChanges);

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

    ~MixAccumulatorState()
    {
        MemAlloc::alignedFree(valueSumTable);
        delete[] versionChangeNumTable;
        delete[] versionInnerIndexTable;
        delete[] versionOuterIndexTable;
        delete[] indexTable;
        MemAlloc::alignedFree(mapSum);
        MemAlloc::alignedFree(mapConv);
    }

    // Owns raw arrays; copying is never needed.
    MixAccumulatorState(const MixAccumulatorState &)            = delete;
    MixAccumulatorState &operator=(const MixAccumulatorState &) = delete;

    /// Reverts the accumulator to the state before the last move().
    void undo() { currentVersion--; }

    /// Init the line-shape codes of all cells for an empty board (version 0).
    void initIndexTable()
    {
        constexpr int length = 11;  // length of line shape
        constexpr int half   = length / 2;

        // Clear shape table
        std::fill_n(&indexTable[0][0], 4 * boardSize * boardSize, 0);

        /// Get an empty line encoding with the given border distance.
        /// @param left The distance to the left border, in range [0, length/2].
        /// @param right The distance to the right border, in range [0, length/2].
        auto get_border_encoding = [=](int left, int right) -> uint32_t {
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
                idxs[0] = get_border_encoding(distx0, distx1);
                // DX[1]=0, DY[1]=1
                idxs[1] = get_border_encoding(disty0, disty1);
                // DX[2]=1, DY[2]=-1
                idxs[2] = get_border_encoding(std::min(distx0, disty1), std::min(distx1, disty0));
                // DX[3]=1, DY[3]=1
                idxs[3] = get_border_encoding(std::min(distx0, disty0), std::min(distx1, disty1));

                assert(idxs[0] < ShapeNum);
                assert(idxs[1] < ShapeNum);
                assert(idxs[2] < ShapeNum);
                assert(idxs[3] < ShapeNum);
            }
    }
};

/// A board change queued for lazy application to an accumulator. Changes are recorded
/// per evaluation side and only applied when an evaluation is actually requested, so
/// that a move immediately followed by its undo never touches the accumulator.
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

/// Queue a new board change into both sides' move caches, cancelling out the last
/// pending change if the new one exactly reverts it.
inline void queueMoveCache(std::vector<MoveCache> (&moveCache)[2],
                           Color side,
                           int   x,
                           int   y,
                           bool  isUndo,
                           [[maybe_unused]] int numCells)
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

        // A cache can span undoing a full board and replaying a different continuation.
        assert(moveCache[c].size() <= 2 * (size_t)numCells);
    }
}

/// Apply all queued board changes of one side to its accumulator, syncing it with the
/// current board state. The WHITE accumulator evaluates a color-flipped board, so
/// piece colors are flipped before being applied.
template <typename AccumulatorT, typename WeightT>
inline void applyMoveCache(Color                   side,
                           std::vector<MoveCache> &moveCache,
                           AccumulatorT           &accumulator,
                           const WeightT          &weight)
{
    constexpr Color opponentMap[4] = {WHITE, BLACK, WALL, EMPTY};

    for (MoveCache &mc : moveCache) {
        if (side == WHITE) {
            mc.oldColor = opponentMap[mc.oldColor];
            mc.newColor = opponentMap[mc.newColor];
        }

        if (mc.oldColor == EMPTY)
            accumulator.move(weight, mc.newColor, mc.x, mc.y);
        else
            accumulator.undo();
    }
    moveCache.clear();
}

// -------------------------------------------------
// Weight-loading scaffolding shared by the mix evaluator constructors.

/// Load the black/white weight pair of a mix-family evaluator through its weight
/// registry, using a fresh LZ4 `Loader`. Installs the standard header validator
/// (arch-hash, rule and board-size checks, plus one load message per file when
/// messages are enabled) and prints the total load time afterwards. Throws on any
/// failure; on success `weight[BLACK]` / `weight[WHITE]` hold registry-owned pointers.
template <typename Loader, typename Registry>
void loadWeightPair(Registry &registry,
                    typename Registry::WeightType *(&weight)[SIDE_NB],
                    const char           *archName,
                    uint32_t              archHash,
                    int                   boardSize,
                    Rule                  rule,
                    Numa::NumaNodeId      numaNodeId,
                    std::filesystem::path blackWeightPath,
                    std::filesystem::path whiteWeightPath)
{
    if (boardSize > MAX_BOARD_SIZE)
        throw UnsupportedBoardSizeError(boardSize);

    Loader loader(Compressor::Type::LZ4_DEFAULT);
    Time   startTime     = now();
    bool   printLoadInfo = false;
    loader.setHeaderValidator([&, archName, archHash](StandardHeader header, auto &args) -> bool {
        if (header.archHash != archHash)
            throw IncompatibleWeightFileError("incompatible architecture in weight file.");

        if (!contains(header.supportedRules, args.rule))
            throw UnsupportedRuleError(args.rule);

        if (!contains(header.supportedBoardSizes, args.boardSize))
            throw UnsupportedBoardSizeError(args.boardSize);

        if (Config::GeneralCfg.messageMode != MsgMode::NONE) {
            MESSAGEL(archName << ": load weight from " << pathToConsoleString(args.weightPath));
            printLoadInfo = true;
        }
        return true;
    });

    for (const auto &[weightSide, weightPath] : {
             std::make_pair(BLACK, blackWeightPath),
             std::make_pair(WHITE, whiteWeightPath),
         }) {
        weight[weightSide] = registry.loadWeightFromFile(loader,
                                                         weightPath,
                                                         numaNodeId,
                                                         {{}, boardSize, rule, weightPath});
        if (!weight[weightSide])
            throw std::runtime_error("failed to load nnue weight from "
                                     + pathToConsoleString(weightPath));
    }

    if (printLoadInfo)
        MESSAGEL(archName << ": weight loaded in " << timeText(now() - startTime));
}

}  // namespace Evaluation::mixnet
