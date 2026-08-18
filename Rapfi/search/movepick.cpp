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

#include "movepick.h"

#include "../eval/classicalpolicy.h"
#include "../eval/evaluator.h"
#include "../eval/scoretables.h"
#include "../game/board.h"
#include "../game/movegen.h"
#include "searchthread.h"

#include <algorithm>
#include <optional>
#include <array>
#include <limits>

namespace {

/// Move picking stages.
/// Usual procedure: X_TT -> X_MOVES -> ALLMOVES.
enum Stages {
    MAIN_TT,
    MAIN_MOVES,
    DEFENDFIVE_TT,
    DEFENDFIVE_MOVES,
    DEFENDFOUR_TT,
    DEFENDFOUR_MOVES,
    DEFENDB4F3_TT,
    DEFENDB4F3_MOVES,
    QVCF_TT,
    QVCF_MOVES,
    ALLMOVES,
};

/// Append VCF moves using the rule-correct variant of the generator.
ScoredMove *generateVCFMoves(Rule rule, const Board &board, ScoredMove *moveList)
{
    return (rule == RENJU ? generate<VCF | RULE_RENJU> : generate<VCF>)(board, moveList);
}

/// Append defence moves against an opponent B4F3 threat using the rule-correct
/// variant of the generator.
ScoredMove *generateDefendB4F3Moves(Rule rule, const Board &board, ScoredMove *moveList)
{
    switch (rule) {
    default:
    case FREESTYLE: return generate<DEFEND_B4F3 | RULE_FREESTYLE>(board, moveList);
    case STANDARD: return generate<DEFEND_B4F3 | RULE_STANDARD>(board, moveList);
    case RENJU: return generate<DEFEND_B4F3 | RULE_RENJU>(board, moveList);
    }
}

/// Partial sort the move list up to the score limit. It dynamiclly decides
/// which sorting algorithm to use based on how many moves are in the list.
template <typename Comparator>
void fastPartialSort(ScoredMove *begin, ScoredMove *end, Score limit, Comparator comp)
{
    // heuristic values
    constexpr size_t InsertionSortLimit = MAX_MOVES / 4;
    constexpr size_t SortLimit          = MAX_MOVES * 2 / 3;

    size_t nMoves = end - begin;
    if (nMoves <= InsertionSortLimit) {
        // Sorts moves in descending order up to and including a given limit.
        // The order of moves smaller than the limit is left unspecified.
        for (ScoredMove *sortedEnd = begin, *p = begin + 1; p < end; ++p)
            if (p->score >= limit) {
                ScoredMove tmp = *p, *q;
                *p             = *++sortedEnd;
                for (q = sortedEnd; q != begin && comp(tmp, *(q - 1)); --q)
                    *q = *(q - 1);
                *q = tmp;
            }
    }
    else if (nMoves <= SortLimit) {
        std::sort(begin, end, comp);
    }
    else {
        std::partial_sort(begin, begin + SortLimit, end, comp);
    }
}

/// Make the first current move to have policy = 1.0f and the rest to have policy = 0.0f.
void markFirstMoveOneHot(ScoredMove *curMove, ScoredMove *endMove)
{
    curMove->policy = 1.0f;
    for (auto *m = curMove + 1; m < endMove; ++m)
        m->policy = 0.0f;
}

}  // namespace

namespace Search {

/// MovePicker constructor at the root node.
/// Generate all legal moves for current side to move.
template <>
MovePicker::MovePicker(Rule rule, const Board &board, ExtraArgs<MovePicker::ROOT> args)
    : board(board)
    , historyScoring()
    , stage(ALLMOVES)
    , rule(rule)
    , ttMove(Pos::NONE)
    , allowPlainB4InVCF(false)
    , hasPolicy(false)
    , useNormalizedPolicy(args.useNormalizedPolicy)
    , normalizedPolicyTemp(args.normalizedPolicyTemp)
#ifdef POLICY_TRAINING
    , traceEligible(false)
#endif
{
    Color self = board.sideToMove(), oppo = ~self;
    curMove = moves;

    if (board.p4Count(self, A_FIVE)) {
        endMove = generate<WINNING>(board, curMove);
        if (useNormalizedPolicy)
            markFirstMoveOneHot(curMove, endMove);
    }
    else if (board.p4Count(oppo, A_FIVE)) {
        endMove = generate<DEFEND_FIVE>(board, curMove);
        if (useNormalizedPolicy)
            markFirstMoveOneHot(curMove, endMove);
    }
    else if (board.p4Count(self, B_FLEX4)) {
        endMove = generate<WINNING>(board, curMove);
        if (useNormalizedPolicy)
            markFirstMoveOneHot(curMove, endMove);
    }
    else if (board.p4Count(oppo, B_FLEX4)) {
        endMove = generate<DEFEND_FOUR | ALL>(board, curMove);
        endMove = generateVCFMoves(rule, board, endMove);
        if (useNormalizedPolicy)
            scoreAllMoves<ScoreType(CLASSICAL | POLICY)>();
    }
    else if (board.p4Count(oppo, C_BLOCK4_FLEX3)
             && (rule != Rule::RENJU || validateOpponentCMove(board))) {
        endMove = generateDefendB4F3Moves(rule, board, curMove);

        if (endMove == curMove)
            endMove = generate<ALL>(board, curMove);
        else
            endMove = generateVCFMoves(rule, board, endMove);

        if (useNormalizedPolicy)
            scoreAllMoves<ScoreType(CLASSICAL | POLICY)>();
    }
    else {
        endMove = generate<ALL>(board, curMove);
        if (useNormalizedPolicy)
            scoreAllMoves<ScoreType(CLASSICAL | POLICY)>();
    }

    if (useNormalizedPolicy)
        fastPartialSort(curMove, endMove, 0, ScoredMove::PolicyComparator {});
}

/// MovePicker constructor for the main search.
template <>
MovePicker::MovePicker(Rule rule, const Board &board, ExtraArgs<MovePicker::MAIN> args)
    : board(board)
    , historyScoring(args.historyScoring)
    , rule(rule)
    , allowPlainB4InVCF(false)
    , hasPolicy(false)
    , useNormalizedPolicy(args.useNormalizedPolicy)
    , normalizedPolicyTemp(args.normalizedPolicyTemp)
#ifdef POLICY_TRAINING
    , traceEligible(true)
#endif
{
    Color self = board.sideToMove(), oppo = ~self;
    bool  ttmValid;

    if (board.p4Count(oppo, A_FIVE)) {
        stage    = DEFENDFIVE_TT;
        ttmValid = board.pattern4(args.ttMove, oppo) == A_FIVE;
    }
    else if (board.p4Count(oppo, B_FLEX4)) {
        stage = DEFENDFOUR_TT;

        ttmValid = board.pattern4(args.ttMove, BLACK) >= E_BLOCK4
                   || board.pattern4(args.ttMove, BLACK) == FORBID
                   || board.pattern4(args.ttMove, WHITE) >= E_BLOCK4;
    }
    else if (board.p4Count(oppo, C_BLOCK4_FLEX3)
             && (rule != Rule::RENJU || validateOpponentCMove(board))) {
        stage    = DEFENDB4F3_TT;
        ttmValid = true;
    }
    else {
        stage    = MAIN_TT;
        ttmValid = true;
    }

    // check legality for defence ttmove
    ttmValid = ttmValid && board.isLegal(args.ttMove);

    stage += !ttmValid;
    ttMove = ttmValid ? args.ttMove : Pos::NONE;
}

/// MovePicker constructor for quick vcf search.
template <>
MovePicker::MovePicker(Rule rule, const Board &board, ExtraArgs<MovePicker::QVCF> args)
    : board(board)
    , historyScoring()
    , rule(rule)
    , allowPlainB4InVCF(
          args.depth >= DEPTH_QVCF_FULL
          || (args.previousSelfP4[0] >= D_BLOCK4_PLUS && args.previousSelfP4[1] >= D_BLOCK4_PLUS))
    , hasPolicy(false)
    , useNormalizedPolicy(false)
    , normalizedPolicyTemp(1.0f)
#ifdef POLICY_TRAINING
    , traceEligible(true)
#endif
{
    Color self = board.sideToMove(), oppo = ~self;
    bool  ttmValid;

    if (board.p4Count(oppo, A_FIVE)) {
        stage    = DEFENDFIVE_TT;
        ttmValid = board.pattern4(args.ttMove, oppo) == A_FIVE;
    }
    else {
        stage    = QVCF_TT;
        ttmValid = board.pattern4(args.ttMove, self) >= E_BLOCK4;
    }

    // check legality for defence ttmove
    ttmValid = ttmValid && board.isLegal(args.ttMove);

    stage += !ttmValid;
    ttMove = ttmValid ? args.ttMove : Pos::NONE;
}

/// Return the next move satisfying a predicate function.
/// Selected move is recorded in curMoves. It never returns the TT move.
/// If there is no move left, it returns Pos::NONE.
template <MovePicker::PickType T, typename Pred>
Pos MovePicker::pickNextMove(Pred filter)
{
    bool forbidden = rule == Rule::RENJU && board.sideToMove() == BLACK;

    while (curMove < endMove) {
        if constexpr (T == Best)
            // ScoreComparator orders best-first, so the best remaining move is the
            // *minimum* under it. (max_element here would deliver worst-first.)
            std::swap(*curMove,
                      *std::min_element(curMove, endMove, ScoredMove::ScoreComparator {}));

        const bool isTtMove    = curMove->pos == ttMove;
        const bool isForbidden = forbidden && board.checkForbiddenPoint(curMove->pos);
        const bool accepted    = !isTtMove && !isForbidden && filter();
#ifdef POLICY_TRAINING
        if (Tuning::PolicyTraceCandidate *candidate = policyTraceCandidate(curMove->pos)) {
            if (candidate->disposition == Tuning::PolicyTraceDisposition::Pending) {
                if (isTtMove) {
                    candidate->filter      = Tuning::PolicyTraceFilter::TtMove;
                    candidate->disposition = Tuning::PolicyTraceDisposition::Filtered;
                }
                else if (isForbidden) {
                    candidate->filter      = Tuning::PolicyTraceFilter::Forbidden;
                    candidate->disposition = Tuning::PolicyTraceDisposition::Filtered;
                }
                else if (accepted) {
                    if (candidate->policyOrdinal == std::numeric_limits<uint16_t>::max())
                        candidate->policyOrdinal = tracePolicyOrdinal++;
                }
                else {
                    candidate->filter      = Tuning::PolicyTraceFilter::RootExcluded;
                    candidate->disposition = Tuning::PolicyTraceDisposition::Filtered;
                }
            }
        }
#endif
        if (accepted) {
            curScore = curMove->score;
            if (useNormalizedPolicy)
                curPolicy = curMove->policy;
            return *curMove++;
        }

        curMove++;
    }

    return Pos::NONE;
}

/// Score all remaining moves according to score type.
template <MovePicker::ScoreType Type>
void MovePicker::scoreAllMoves()
{
    static_assert(bool(Type & CLASSICAL));

    using Evaluation::Evaluator;
    using Evaluation::PolicyBuffer;
    using PolicyBufferStorage = std::aligned_storage_t<sizeof(PolicyBuffer), alignof(PolicyBuffer)>;

    PolicyBufferStorage policyBufferStorage;
    Color               self = board.sideToMove(), oppo = ~self;
    PolicyBuffer       *policyBuf = reinterpret_cast<PolicyBuffer *>(&policyBufferStorage);
    Evaluator *evaluator = board.thisThread() ? board.thisThread()->evaluator.get() : nullptr;
    const bool useClassicalFallback = !(bool(Type & POLICY) && evaluator);
    std::optional<Evaluation::ClassicalPolicyScorer> classicalPolicy;
    if (useClassicalFallback)
        classicalPolicy.emplace(rule, board);
#ifdef POLICY_TRAINING
    const bool traceRequested =
        traceEligible && board.thisThread() && board.thisThread()->engine.policyTraceSession();
#else
    constexpr bool traceRequested = false;
#endif
    const Evaluation::PolicyContext policyContext =
        useClassicalFallback && traceRequested ? Evaluation::classifyPolicyContext(board)
                                               : Evaluation::QUIET;
    const bool policyCrossActive =
        Evaluation::policyCrossActiveMask(rule, self) & uint8_t(1U << policyContext);
#ifdef POLICY_TRAINING
    if (traceEligible && useClassicalFallback && curMove != endMove)
        beginPolicyTrace(policyContext, policyCrossActive);
#endif

    if (bool(Type & POLICY) && evaluator) {
        new (policyBuf) Evaluation::PolicyBuffer(board.size());

        // Set compute flag for all moves in move list
        for (auto &m : *this)
            policyBuf->setComputeFlag(m.pos);

        evaluator->evaluatePolicy(board, *policyBuf);
        hasPolicy      = true;
        maxPolicyScore = std::numeric_limits<Score>::lowest() / 2;
    }

    for (auto &m : *this) {
        int score;
#ifdef POLICY_TRAINING
        int positionScore = 0, p3Residual = 0;
#endif
        if (bool(Type & POLICY) && evaluator) {
            score = m.rawScore = policyBuf->score(m.pos);
            maxPolicyScore     = std::max(maxPolicyScore, m.rawScore);
        }
        else {
            const auto [pcodeBlack, pcodeWhite] = board.pcodePair(m.pos);
#ifdef POLICY_TRAINING
            positionScore = Evaluation::classicalPolicyBaseScore(rule,
                                                                  self,
                                                                  pcodeBlack,
                                                                  pcodeWhite);
#endif
            score = m.rawScore = classicalPolicy->score(pcodeBlack,
                                                        pcodeWhite,
                                                        board.pattern4(m.pos, self),
                                                        board.pattern4(m.pos, oppo));
#ifdef POLICY_TRAINING
            p3Residual = score - positionScore;
#endif
        }

        int historyBonus = 0;
        if (bool(Type & MAIN_HISTORY) && historyScoring.mainHistory) {
            if (board.pattern4(m.pos, self) >= H_FLEX3) {
                historyBonus = (*historyScoring.mainHistory)[self][m.pos][HIST_ATTACK]
                               * historyScoring.attackWeight / MoveHistoryScoring::WeightScale;
            }
            else {
                historyBonus = (*historyScoring.mainHistory)[self][m.pos][HIST_QUIET]
                               * historyScoring.quietWeight / MoveHistoryScoring::WeightScale;
            }
        }

        int counterMoveBonus = 0;
        if (bool(Type & COUNTER_MOVE) && historyScoring.counterMoveHistory) {
            if (Pos lastMove = board.getLastMove(); board.isInBoard(lastMove)) {
                auto [counterMove, counterMoveP4] =
                    (*historyScoring.counterMoveHistory)[oppo][lastMove.moveIndex()].get();

                if (counterMove == m.pos && counterMoveP4 <= board.pattern4(m.pos, self)) {
                    counterMoveBonus = historyScoring.counterMoveBonus;
                }
            }
        }

        m.score = Evaluation::clampMoveScore(score + historyBonus + counterMoveBonus);
#ifdef POLICY_TRAINING
        if (traceEvent && !policyTraceCandidate(m.pos)) {
            Tuning::PolicyTraceCandidate candidate {};
            candidate.move             = m.pos;
            candidate.generatedOrdinal = static_cast<uint16_t>(traceEvent->candidates.size());
            candidate.selectionOrdinal = std::numeric_limits<uint16_t>::max();
            candidate.policyOrdinal    = std::numeric_limits<uint16_t>::max();
            candidate.filter           = Tuning::PolicyTraceFilter::None;
            candidate.disposition      = Tuning::PolicyTraceDisposition::Pending;
            candidate.selfPattern      = board.pattern4(m.pos, self);
            candidate.opponentPattern  = board.pattern4(m.pos, oppo);
            candidate.positionScore    = positionScore;
            candidate.p3Residual       = p3Residual;
            candidate.mainHistory      = historyBonus;
            candidate.counterMove      = counterMoveBonus;
            candidate.finalScore       = m.score;
            candidate.teacherLogit     = m.rawScore;
            candidate.bound            = BOUND_NONE;
            traceEvent->candidates.push_back(candidate);
        }
#endif
    }

    // Compute normalized policy score if needed
    if (useNormalizedPolicy) {
        // Use the normalized policy if we do have policy from evaluator
        if (hasPolicy) {
            assert(evaluator);
            // BUG: should pass normalizedPolicyTemp here (and use it in the raw-score
            // fallback below); currently the softmax always runs at temperature 1.0.
            // Wiring it changes search behavior, so it needs its own SPRT-gated change.
            policyBuf->applySoftmax();
            for (auto &m : *this)
                m.policy = (*policyBuf)[m.pos];
        }
        // Otherwise, use the normalized raw score as policy
        else {
            const float scale     = 1.0f / PolicyBuffer::ScoreScale;
            float       sumPolicy = 0;
            for (auto &m : *this)
                sumPolicy += m.policy = std::exp((m.rawScore - maxPolicyScore) * scale);

            // Divide sum policy to normalize
            float invSumPolicy = 1 / sumPolicy;
            for (auto &m : *this)
                m.policy *= invSumPolicy;
        }
    }
}

template <MovePicker::ScoreType ExtraFlags>
void MovePicker::scoreAndSortMoves()
{
    if (useNormalizedPolicy) {
        scoreAllMoves<ScoreType(CLASSICAL | POLICY)>();
        fastPartialSort(curMove, endMove, 0, ScoredMove::PolicyComparator {});
    }
    else {
        scoreAllMoves<ScoreType(CLASSICAL | POLICY | ExtraFlags)>();
        fastPartialSort(curMove, endMove, 0, ScoredMove::ScoreComparator {});
    }
#ifdef POLICY_TRAINING
    capturePolicyTraceOrder();
#endif
}

/// Pick the next legal move until there is no legal move left.
/// @return Next legal move, or Pos::NONE if there is no legal move left.
Pos MovePicker::operator()()
{
top:
    switch (stage) {
    case MAIN_TT:
    case DEFENDFIVE_TT:
    case DEFENDFOUR_TT:
    case DEFENDB4F3_TT:
    case QVCF_TT: ++stage; return ttMove;

    case MAIN_MOVES:
        assert(!board.p4Count(~board.sideToMove(), A_FIVE));
        assert(!board.p4Count(~board.sideToMove(), B_FLEX4));

        curMove = moves;
        endMove = generate<ALL>(board, curMove);
        scoreAndSortMoves<ScoreType(MAIN_HISTORY | COUNTER_MOVE)>();

        stage = ALLMOVES;
        goto top;

    case DEFENDFIVE_MOVES:
        assert(board.p4Count(~board.sideToMove(), A_FIVE));

        curMove = moves;
        endMove = !ttMove ? generate<DEFEND_FIVE>(board, moves) : moves;
        if (useNormalizedPolicy) {
            markFirstMoveOneHot(curMove, endMove);
            hasPolicy = true;
        }

        stage = ALLMOVES;
        goto top;

    case DEFENDFOUR_MOVES:
        assert(board.p4Count(~board.sideToMove(), B_FLEX4));

        curMove = moves;
        endMove = generate<DEFEND_FOUR>(board, curMove);
        endMove = generateVCFMoves(rule, board, endMove);
        scoreAndSortMoves<MAIN_HISTORY>();

        stage = ALLMOVES;
        goto top;

    case DEFENDB4F3_MOVES:
        assert(board.p4Count(~board.sideToMove(), C_BLOCK4_FLEX3));

        curMove = moves;
        endMove = generateDefendB4F3Moves(rule, board, curMove);

        if (endMove == curMove) {
            stage = MAIN_MOVES;
            goto top;
        }

        endMove = generateVCFMoves(rule, board, endMove);
        scoreAndSortMoves<MAIN_HISTORY>();

        stage = ALLMOVES;
        goto top;

    case QVCF_MOVES:
        curMove = moves;
        {
            Pos selfLast = board.getLastActualMoveOfSide(board.sideToMove());
            endMove =
                (allowPlainB4InVCF ? generateNeighbors<VCF>
                                   : generateNeighbors<VCF | COMB>)(board,
                                                                    curMove,
                                                                    selfLast,
                                                                    RANGE_SQUARE2_LINE4.data(),
                                                                    RANGE_SQUARE2_LINE4.size());
        }

        scoreAllMoves<CLASSICAL>();
        fastPartialSort(curMove, endMove, 0, ScoredMove::ScoreComparator {});
#ifdef POLICY_TRAINING
        capturePolicyTraceOrder();
#endif

        stage = ALLMOVES;
        [[fallthrough]];

    case ALLMOVES: return pickNextMove<Next>([]() { return true; });
    }

    // This should never be reached, unless a bug occurs
    assert(false && "unknown MovePicker stage occurred");
    return Pos::NONE;
}

#ifdef POLICY_TRAINING
void MovePicker::beginPolicyTrace(uint8_t context, bool p3Active)
{
    if (traceEvent || !board.thisThread())
        return;
    Tuning::PolicyTraceSession *session = board.thisThread()->engine.policyTraceSession();
    if (!session)
        return;

    const Tuning::PolicyTraceEventContext eventContext =
        session->beginEvent(board.thisThread()->id);
    traceEvent                = std::make_unique<Tuning::PolicyTraceEvent>();
    traceEvent->identity      = eventContext.identity;
    traceEvent->openingFamily = eventContext.openingFamily;
    traceEvent->split         = eventContext.split;
    traceEvent->rule          = rule;
    traceEvent->boardSize     = static_cast<uint8_t>(board.size());
    traceEvent->sideToMove    = board.sideToMove();
    traceEvent->policyContext = context;
    traceEvent->p3Active      = p3Active;
    traceEvent->positionKey   = board.zobristKey();
    traceEvent->ttMove        = ttMove;
    traceEvent->bestMove      = Pos::NONE;
    traceEvent->bestValue     = VALUE_NONE;
    traceEvent->bestBound     = BOUND_NONE;
    traceEvent->completion    = Tuning::PolicyTraceCompletion::Terminated;
    traceEvent->domain        = stage == DEFENDFOUR_MOVES   ? Tuning::PolicyTraceDomain::DefendFour
                                : stage == DEFENDB4F3_MOVES ? Tuning::PolicyTraceDomain::DefendB4F3
                                : stage == QVCF_MOVES       ? Tuning::PolicyTraceDomain::Qvcf
                                                            : Tuning::PolicyTraceDomain::Main;
    traceEvent->history.reserve(board.ply());
    for (int i = 0; i < board.ply(); i++)
        traceEvent->history.push_back(board.getHistoryMove(i));
}

void MovePicker::capturePolicyTraceOrder()
{
    for (uint16_t ordinal = 0; ordinal < uint16_t(endMove - curMove); ordinal++) {
        if (traceEvent) {
            if (Tuning::PolicyTraceCandidate *candidate = policyTraceCandidate(curMove[ordinal].pos);
                candidate
                && candidate->selectionOrdinal == std::numeric_limits<uint16_t>::max())
                candidate->selectionOrdinal = ordinal;
        }
    }
}

Tuning::PolicyTraceCandidate *MovePicker::policyTraceCandidate(Pos move)
{
    if (!traceEvent)
        return nullptr;
    auto candidate = std::find_if(
        traceEvent->candidates.begin(),
        traceEvent->candidates.end(),
        [move](const Tuning::PolicyTraceCandidate &entry) { return entry.move == move; });
    return candidate == traceEvent->candidates.end() ? nullptr : &*candidate;
}

void MovePicker::commitPolicyTrace()
{
    if (!traceEvent)
        return;
    Tuning::PolicyTraceSession *session =
        board.thisThread() ? board.thisThread()->engine.policyTraceSession() : nullptr;
    if (!session)
        throw std::logic_error("policy trace session detached before event commit");
    session->commit(std::move(*traceEvent));
    traceEvent.reset();
}

#endif

}  // namespace Search
