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

#include "eval.h"

#include "../game/board.h"
#include "../search/searchthread.h"
#include "evaluator.h"

#include <algorithm>
#include <cmath>

namespace {

/// Makes threat mask according current pattern4 counts on board.
int makeThreatMask(const StateInfo &st, Color self)
{
    Color oppo = ~self;

    bool oppoFive      = st.p4Count[oppo][A_FIVE];
    bool selfFlexFour  = st.p4Count[self][B_FLEX4];
    bool oppoFlexFour  = st.p4Count[oppo][B_FLEX4];
    bool selfFourPlus  = st.p4Count[self][D_BLOCK4_PLUS] + st.p4Count[self][C_BLOCK4_FLEX3];
    bool selfFour      = st.p4Count[self][E_BLOCK4];
    bool selfThreePlus = st.p4Count[self][G_FLEX3_PLUS] + st.p4Count[self][F_FLEX3_2X];
    bool selfThree     = st.p4Count[self][H_FLEX3];
    bool oppoFourPlus  = st.p4Count[oppo][D_BLOCK4_PLUS] + st.p4Count[oppo][C_BLOCK4_FLEX3];
    bool oppoFour      = st.p4Count[oppo][E_BLOCK4];
    bool oppoThreePlus = st.p4Count[oppo][G_FLEX3_PLUS] + st.p4Count[oppo][F_FLEX3_2X];
    bool oppoThree     = st.p4Count[oppo][H_FLEX3];

    int mask = 0;
    mask |= 0b1 & -int(oppoFive);
    mask |= 0b10 & -int(selfFlexFour);
    mask |= 0b100 & -int(oppoFlexFour);
    mask |= 0b1000 & -int(selfFourPlus);
    mask |= 0b10000 & -int(selfFour);
    mask |= 0b100000 & -int(selfThreePlus);
    mask |= 0b1000000 & -int(selfThree);
    mask |= 0b10000000 & -int(oppoFourPlus);
    mask |= 0b100000000 & -int(oppoFour);
    mask |= 0b1000000000 & -int(oppoThreePlus);
    mask |= 0b10000000000 & -int(oppoThree);

    assert(0 <= mask && mask < THREAT_NB);
    return mask;
}

/// Evaluates threats.
/// Threat indicates dynamic first-move status which causes highly non-linear eval changes.
template <Rule R>
inline Value evaluateThreat(const StateInfo &st, Color self)
{
    return (Value)Config::EVALS_THREAT[Config::tableIndex(R, self)][makeThreatMask(st, self)];
}

/// Evaluates basic patterns on board.
inline Value evaluateBasic(const StateInfo &st, Color self)
{
    return self == BLACK ? st.valueBlack : -st.valueBlack;
}

/// Finds a margin for switching to classical evaluation if
/// it falls outside alpha-beta window with this margin.
inline int classicalEvalMargin(Value bound)
{
    float winLossRate = 2 * (Config::valueToWinRate(bound) - 0.5f);
    float x           = Config::EvaluatorMarginWinLossScale * winLossRate;
    float x2          = x * x;
    return (int)(Config::EvaluatorMarginScale
                 * ::expf(-::powf(x2, Config::EvaluatorMarginWinLossExponent)));
}

}  // namespace

namespace Evaluation {

/// Calculates the final evaluation of a board.
/// @note Board must have at least one stone placed (`ply() > 0`).
template <Rule R>
Value evaluate(const Board &board, Value alpha, Value beta)
{
    assert(board.ply() > 0);
    Color self = board.sideToMove();

    const StateInfo &st0 = board.stateInfo();
    const StateInfo &st1 = board.stateInfo(1);

    Value basicEval  = (evaluateBasic(st0, self) + evaluateBasic(st1, self)) / 2;
    Value threatEval = evaluateThreat<R>(st0, self);
    Value eval       = std::clamp(basicEval + threatEval, VALUE_EVAL_MIN, VALUE_EVAL_MAX);

    if (board.evaluator()) {
        // Use evaluator eval if classical eval are in alpha-beta window margin
        int margin = classicalEvalMargin(eval);
        if (eval >= alpha - margin && eval <= beta + margin)
            return computeEvaluatorValue(board).value();
    }

    return eval;
}

template Value evaluate<FREESTYLE>(const Board &, Value, Value);
template Value evaluate<STANDARD>(const Board &, Value, Value);
template Value evaluate<RENJU>(const Board &, Value, Value);

/// A safe but slower version of evaluate().
/// Similar to evaluate(), but can be called for all board states.
Value evaluate(const Board &board, Rule rule)
{
    if (board.ply() > 0) {
        switch (rule) {
        default:
        case Rule::FREESTYLE: return evaluate<Rule::FREESTYLE>(board);
        case Rule::STANDARD: return evaluate<Rule::STANDARD>(board);
        case Rule::RENJU: return evaluate<Rule::RENJU>(board);
        }
    }
    else {
        Color            self = board.sideToMove();
        const StateInfo &st   = board.stateInfo();

        Value basicEval  = evaluateBasic(st, self);
        Value threatEval = VALUE_ZERO;
        switch (rule) {
        default:
        case Rule::FREESTYLE: threatEval = evaluateThreat<Rule::FREESTYLE>(st, self); break;
        case Rule::STANDARD: threatEval = evaluateThreat<Rule::STANDARD>(st, self); break;
        case Rule::RENJU: threatEval = evaluateThreat<Rule::RENJU>(st, self); break;
        }

        Value eval = basicEval + threatEval;

        return std::clamp(eval, VALUE_EVAL_MIN, VALUE_EVAL_MAX);
    }
}

ValueType computeEvaluatorValue(const Board &board)
{
    Color     self = board.sideToMove();
    ValueType v    = board.evaluator()->evaluateValue(board);

    // Adjust draw rate according to draw ratio and draw black win rate
    if (Config::EvaluatorDrawRatio < 1.0) {
        float newDrawRate = Config::EvaluatorDrawRatio * v.draw();
        float drawWinRate = Config::EvaluatorDrawBlackWinRate;
        drawWinRate       = self == BLACK ? drawWinRate : 1.0f - drawWinRate;
        v                 = v.valueOfDrawWinRate(drawWinRate, newDrawRate);
    }

    return v;
}

/// Trace all evaluation info from a board state with rule.
EvalInfo::EvalInfo(const Board &board, Rule rule)
    : plyBack {0}
    , self(board.sideToMove())
    , threatMask(makeThreatMask(board.stateInfo(), self))
{
    Board &b = const_cast<Board &>(board);
    Pos    undoHistory[2];
    int    undoCount = 0;

    for (size_t backIndex = 0; backIndex < arraySize(plyBack); backIndex++) {
        auto &info = plyBack[backIndex];
        if (backIndex > 0 && b.ply() > 0) {
            undoHistory[undoCount++] = b.getLastMove();
            b.undo(rule);
        }

        FOR_EVERY_EMPTY_POS(&b, pos)
        {
            const Cell &c = b.cell(pos);
            info.pcodeCount[BLACK][c.pcode<BLACK>()]++;
            info.pcodeCount[WHITE][c.pcode<WHITE>()]++;
        }
    }

    // Recover board state
    while (undoCount > 0) {
        b.move(rule, undoHistory[--undoCount]);
    }
}

}  // namespace Evaluation
