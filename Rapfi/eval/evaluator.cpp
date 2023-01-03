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

#include "evaluator.h"

#include "../game/board.h"

#include <algorithm>
#include <cmath>
#include <limits>
#include <sstream>

namespace Evaluation {

ValueType::ValueType(float winLogits, float lossLogits, float drawLogits, bool applySoftmax)
    : winRate(winLogits)
    , lossRate(lossLogits)
    , drawRate(drawLogits)
{
    if (applySoftmax) {
        float maxValue = std::max(std::max(winLogits, lossLogits), drawLogits);
        winRate        = std::exp(winLogits - maxValue);
        lossRate       = std::exp(lossLogits - maxValue);
        drawRate       = std::exp(drawLogits - maxValue);
        float invSum   = 1.0f / (winRate + lossRate + drawRate);
        winRate *= invSum;
        lossRate *= invSum;
        drawRate *= invSum;
    }

    constexpr float WLRMateValue = 0.999f;
    float           wlr          = winLossRate();
    if (wlr > WLRMateValue)
        val = VALUE_EVAL_MAX;
    else if (wlr < -WLRMateValue)
        val = VALUE_EVAL_MIN;
    else {
        val = Value(Config::ScalingFactor * std::log((1 + wlr) / (1 - wlr)));
        val = std::clamp(val, VALUE_EVAL_MIN, VALUE_EVAL_MAX);
    }
}

ValueType ValueType::valueOfDrawWinRate(float drawWinRate, float newDrawRate)
{
    assert(0 <= drawWinRate && drawWinRate <= 1);
    assert(newDrawRate <= drawRate);

    float extraDrawRate = drawRate - newDrawRate;
    float drawWin       = extraDrawRate * drawWinRate;
    float drawLoss      = extraDrawRate * (1.0f - drawWinRate);

    return ValueType(winRate + drawWin, lossRate + drawLoss, newDrawRate, false);
}

PolicyBuffer::PolicyBuffer(int boardWidth, int boardHeight)
    : boardWidth(boardWidth)
    , bufferSize(boardWidth * boardHeight)
    , policyScoreBias(ScoreBias)
{
    std::fill_n(computeFlag, bufferSize, false);
}

void PolicyBuffer::setComputeFlagForAllEmptyCell(const Board &board, bool enabled)
{
    FOR_EVERY_EMPTY_POS(&board, pos) { setComputeFlag(pos, enabled); }
}

void PolicyBuffer::setComputeFlagForAllCandidateCell(const Board &board, bool enabled)
{
    FOR_EVERY_CAND_POS(&board, pos) { setComputeFlag(pos, enabled); }
}

void PolicyBuffer::applySoftmax()
{
    // Find max computed policy
    PolicyType maxPolicy = std::numeric_limits<PolicyType>::lowest();
    for (size_t i = 0; i < bufferSize; i++) {
        if (computeFlag[i] && policy[i] > maxPolicy)
            maxPolicy = policy[i];
    }

    // Apply exponent function and sum
    PolicyType sumPolicy = 0;
    for (size_t i = 0; i < bufferSize; i++) {
        if (computeFlag[i])
            sumPolicy += policy[i] = std::exp(policy[i] - maxPolicy);
    }

    // Divide sum policy to normalize
    PolicyType invSumPolicy = 1 / sumPolicy;
    for (size_t i = 0; i < bufferSize; i++) {
        policy[i] *= invSumPolicy;
    }
}

Evaluator::Evaluator(int boardSize, Rule rule) : boardSize(boardSize), rule(rule) {}

void Evaluator::syncWithBoard(const Board &board)
{
    initEmptyBoard();

    // Clone an empty board for replay moves.
    Board boardClone {board.size()};
    boardClone.newGame(rule);

    for (int i = 0; i < board.ply(); i++) {
        Pos pos = board.getHistoryMove(i);

        if (pos == Pos::PASS)
            boardClone.doPassMove();
        else {
            // Ensure side to move is the same as the stone on the original board
            if (boardClone.sideToMove() != board.get(pos))
                boardClone.flipSide();

            beforeMove(boardClone, pos);
            boardClone.move(rule, pos);
            afterMove(boardClone, pos);
        }
    }
}

}  // namespace Evaluation
