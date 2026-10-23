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

#include "featureencoder.h"

#include "../eval/scoretables.h"
#include "../game/board.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace {

/// Packs a bit array into byte array (in big-endian).
/// @param bits The source of bits array
/// @param numBits number of bits to pack
/// @param bytes Destination of byte array
void packBitsToBytes(const uint8_t *bits, size_t numBits, uint8_t *bytes)
{
    size_t numBytesFloored = numBits / 8;
    size_t numBitsRemained = numBits % 8;

    for (size_t byteIdx = 0; byteIdx < numBytesFloored; byteIdx++) {
        *bytes++ = (bits[0] << 7) | (bits[1] << 6) | (bits[2] << 5) | (bits[3] << 4)
                   | (bits[4] << 3) | (bits[5] << 2) | (bits[6] << 1) | (bits[7] << 0);
        bits += 8;
    }

    // Deals with remaining bits that less than a byte; when numBits is
    // byte-aligned there is no tail byte and the destination ends here
    if (numBitsRemained) {
        *bytes = 0;
        for (size_t bitIdx = 0; bitIdx < numBitsRemained; bitIdx++)
            *bytes |= bits[bitIdx] << (7 - bitIdx);
    }
}

uint16_t quantizePolicy(float policy)
{
    return uint16_t(std::clamp<int>(policy * UINT16_MAX, 0, UINT16_MAX));
}

bool isUnavailableOrMate(Eval eval)
{
    // Keep mate-distance scores categorical on either sign, using Rapfi's
    // canonical mate band rather than the Trainer reader's configurable cutoff.
    return eval == VALUE_NONE || eval >= VALUE_MATE_IN_MAX_PLY || eval <= VALUE_MATED_IN_MAX_PLY;
}

size_t policyIndex(Pos move, int boardsize, size_t numCells)
{
    return move == Pos::PASS ? numCells : move.y() * boardsize + move.x();
}

}  // namespace

namespace Tuning {

void encodePolicyTarget(Pos                       bestMove,
                        Eval                      bestEval,
                        const MovePayload        &payload,
                        int                       boardsize,
                        size_t                    numCells,
                        const PolicyTargetConfig &config,
                        uint16_t                 *dst,
                        FeatureEncodeScratch     &scratch)
{
    std::fill_n(dst, numCells + 1, uint16_t(0));

    // Preserve dense targets exactly, including their own-board pass index.
    size_t ownCells = boardsize * boardsize;
    if (auto *policy = std::get_if<PolicyArrayF32>(&payload);
        policy && policy->size() >= ownCells + 1) {
        for (size_t i = 0; i < ownCells; i++)
            dst[i] = quantizePolicy((*policy)[i]);
        dst[numCells] = quantizePolicy((*policy)[ownCells]);
        return;
    }
    if (auto *policy = std::get_if<PolicyArrayI16>(&payload);
        policy && policy->size() >= ownCells + 1) {
        constexpr float InvScale = 1.0f / 32767;
        for (size_t i = 0; i < ownCells; i++)
            dst[i] = quantizePolicy((*policy)[i] * InvScale);
        dst[numCells] = quantizePolicy((*policy)[ownCells] * InvScale);
        return;
    }

    auto writeOneHot = [&]() {
        if (bestMove == Pos::PASS || bestMove.isInBoard(boardsize, boardsize))
            dst[policyIndex(bestMove, boardsize, numCells)] = UINT16_MAX;
    };

    const ExtraPVArray *pvs = extraPVs(payload);
    if (!config.useMultiPV() || !pvs || isUnavailableOrMate(bestEval)
        || !std::isfinite(config.multiPVTemperature) || !std::isfinite(config.evalScalingFactor)
        || config.evalScalingFactor <= 0.0f) {
        writeOneHot();
        return;
    }

    const size_t numMoves = pvs->size() + 1;
    scratch.policyWeights.resize(numMoves);

    float maxWinRate = -std::numeric_limits<float>::infinity();
    for (size_t i = 0; i < numMoves; i++) {
        Pos  move = i == 0 ? bestMove : (*pvs)[i - 1].move;
        Eval eval = i == 0 ? bestEval : (*pvs)[i - 1].eval;

        if ((move != Pos::PASS && !move.isInBoard(boardsize, boardsize))
            || isUnavailableOrMate(eval)) {
            writeOneHot();
            return;
        }

        // Duplicate PV moves are malformed; retain a normalized, deterministic
        // target by falling back instead of overwriting part of the softmax mass.
        for (size_t j = 0; j < i; j++) {
            Pos previousMove = j == 0 ? bestMove : (*pvs)[j - 1].move;
            if (move == previousMove) {
                writeOneHot();
                return;
            }
        }

        float winRate = 1.0f / (1.0f + std::exp(-float(eval) / config.evalScalingFactor));
        scratch.policyWeights[i] = winRate;
        maxWinRate               = std::max(maxWinRate, winRate);
    }

    float weightSum = 0.0f;
    for (float &weight : scratch.policyWeights) {
        weight = std::exp((weight - maxWinRate) / config.multiPVTemperature);
        weightSum += weight;
    }

    for (size_t i = 0; i < numMoves; i++) {
        Pos move = i == 0 ? bestMove : (*pvs)[i - 1].move;
        dst[policyIndex(move, boardsize, numCells)] =
            quantizePolicy(scratch.policyWeights[i] / weightSum);
    }
}

void encodeEntryFeatures(const Board                               &board,
                         Pos                                        bestMove,
                         Eval                                       bestEval,
                         const MovePayload                         &payload,
                         const PolicyTargetConfig                  &policyConfig,
                         Result                                     result,
                         const std::optional<std::array<float, 3>> &softValueTarget,
                         size_t                                     numCells,
                         const EntryFeatureDst                     &dst,
                         FeatureEncodeScratch                      &scratch)
{
    size_t numBytes  = (numCells + 7) / 8;
    int    boardsize = board.size();

    // Update inboard, self, oppo plane
    scratch.inBoardPlane.assign(numCells, 0);
    scratch.selfPlane.assign(numCells, 0);
    scratch.oppoPlane.assign(numCells, 0);
    Color self = board.sideToMove(), oppo = ~self;
    encodePolicyTarget(bestMove,
                       bestEval,
                       payload,
                       boardsize,
                       numCells,
                       policyConfig,
                       dst.policyTarget,
                       scratch);
    FOR_EVERY_POSITION(&board, pos)
    {
        int posIdx                   = pos.y() * boardsize + pos.x();
        scratch.inBoardPlane[posIdx] = true;
        scratch.selfPlane[posIdx]    = board.get(pos) == self;
        scratch.oppoPlane[posIdx]    = board.get(pos) == oppo;

        if (dst.sparseU8) {
            // Write the sparse pattern/pattern4 features
            dst.sparseU8[0 * numCells + posIdx] = board.pattern(pos, self, 0);
            dst.sparseU8[1 * numCells + posIdx] = board.pattern(pos, self, 1);
            dst.sparseU8[2 * numCells + posIdx] = board.pattern(pos, self, 2);
            dst.sparseU8[3 * numCells + posIdx] = board.pattern(pos, self, 3);
            dst.sparseU8[4 * numCells + posIdx] = board.pattern(pos, oppo, 0);
            dst.sparseU8[5 * numCells + posIdx] = board.pattern(pos, oppo, 1);
            dst.sparseU8[6 * numCells + posIdx] = board.pattern(pos, oppo, 2);
            dst.sparseU8[7 * numCells + posIdx] = board.pattern(pos, oppo, 3);
            dst.sparseU8[8 * numCells + posIdx] = board.pattern4(pos, self);
            dst.sparseU8[9 * numCells + posIdx] = board.pattern4(pos, oppo);
        }
        if (dst.sparseU16) {
            // Write the sparse pattern-code features
            dst.sparseU16[0 * numCells + posIdx] =
                self == BLACK ? board.pcode<BLACK>(pos) : board.pcode<WHITE>(pos);
            dst.sparseU16[1 * numCells + posIdx] =
                oppo == BLACK ? board.pcode<BLACK>(pos) : board.pcode<WHITE>(pos);
        }
    }

    // Write the packed binary planes
    packBitsToBytes(scratch.inBoardPlane.data(), numCells, dst.binaryPacked + 0 * numBytes);
    packBitsToBytes(scratch.selfPlane.data(), numCells, dst.binaryPacked + 1 * numBytes);
    packBitsToBytes(scratch.oppoPlane.data(), numCells, dst.binaryPacked + 2 * numBytes);

    // Write the global input (side to move)
    dst.globalInput[0] = (self == BLACK ? -1.0f : 1.0f);

    // Write the global value target
    if (softValueTarget.has_value()) {
        dst.globalTargets[0] = (*softValueTarget)[0];
        dst.globalTargets[1] = (*softValueTarget)[1];
        dst.globalTargets[2] = (*softValueTarget)[2];
    }
    else {
        dst.globalTargets[0] = result == RESULT_WIN;
        dst.globalTargets[1] = result == RESULT_LOSS;
        dst.globalTargets[2] = result == RESULT_DRAW;
    }
}

const std::vector<uint32_t> &sparseInputDims()
{
    static const std::vector<uint32_t> dims {
        PATTERN_NB,
        PATTERN_NB,
        PATTERN_NB,
        PATTERN_NB,
        PATTERN_NB,
        PATTERN_NB,
        PATTERN_NB,
        PATTERN_NB,
        PATTERN4_NB,
        PATTERN4_NB,
        PCODE_NB,
        PCODE_NB,
    };
    return dims;
}

}  // namespace Tuning
