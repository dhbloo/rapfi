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

#include "../../config.h"
#include "../../core/hash.h"
#include "../../core/iohelper.h"
#include "../../core/platform.h"
#include "../../eval/eval.h"
#include "../../game/board.h"
#include "../../game/movegen.h"
#include "../../game/wincheck.h"
#include "../hashtable.h"
#include "../movepick.h"
#include "../opening.h"
#include "../searchthread.h"
#include "../skill.h"
#include "parameter.h"
#include "searcher.h"
#include "searchstack.h"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <iomanip>
#include <random>

using namespace Search;
using namespace Search::AB;

namespace {

enum NodeType { Root, PV, NonPV };

void iterativeDeepingLoop(Board &board);
void aspirationSearch(Rule rule, Board &board, SearchStack *ss, Value prevValue, Depth depth);
template <NodeType NT = PV>
Value search(Rule         rule,
             Board       &board,
             SearchStack *ss,
             Value        alpha,
             Value        beta,
             Depth        depth,
             bool         cutNode);
template <Rule Rule, NodeType NT>
Value search(Board &board, SearchStack *ss, Value alpha, Value beta, Depth depth, bool cutNode);
template <Rule Rule, NodeType NT>
Value vcfsearch(Board &board, SearchStack *ss, Value alpha, Value beta, Depth depth = 0.0f);
template <Rule Rule, NodeType NT>
Value vcfdefend(Board &board, SearchStack *ss, Value alpha, Value beta, Depth depth = 0.0f);

}  // namespace

void ABSearchData::clearData(SearchThread &th)
{
    multiPv         = 1;
    pvIdx           = 0;
    rootDepth       = 0;
    completedDepth  = 0;
    bestMoveChanges = 0;
    singularRoot    = false;
    mainHistory.init(0);
    counterMoveHistory.init(std::make_pair(Pos::NONE, NONE));
}

void ABSearcher::setMemoryLimit(size_t memorySizeKB)
{
    TT.resize(memorySizeKB);
}

size_t ABSearcher::getMemoryLimit() const
{
    return TT.hashSizeKB();
}

void ABSearcher::clear(ThreadPool &pool, bool clearAllMemory)
{
    // Clear time control variables for one game
    previousTimeReduction = 1.0f;
    previousBestValue     = VALUE_NONE;

    initReductionLUT(reductions, pool.size());

    if (clearAllMemory)
        TT.clear();
}

void ABSearcher::searchMain(MainSearchThread &th)
{
    SearchOptions &opts = th.options();

    // Probe opening database and find if there is a prepared opening
    if (!opts.disableOpeningQuery
        && Opening::probeOpening(*th.board, opts.rule, th.resultAction, th.bestMove)) {
        th.markPonderingAvailable();
        return;
    }

    // Check for immediate move
    if (th.rootMoves.empty()) {
        // If there is no stones on board, it is possible that the opponent played a pass
        // move at the start of one game. We just choose the center location to play.
        if (th.board->nonPassMoveCount() == 0) {
            th.bestMove = th.board->centerPos();
            return;
        }

        // Return the first empty position if we might find a forced forbidden
        // point mate in Renju, or all legal points have been blocked.
        FOR_EVERY_EMPTY_POS(th.board, pos)
        {
            th.bestMove = pos;
            printer.printBestmoveWithoutSearch(th, pos, mated_in(0), 0, nullptr);
            return;
        }

        return;  // abnormal case: GUI might have a bug
    }
    // If we are winning, return directly
    else if (th.board->p4Count(th.board->sideToMove(), A_FIVE)) {
        assert(th.board->cell(th.rootMoves[0].pv[0]).pattern4[th.board->sideToMove()] == A_FIVE);
        th.rootMoves[0].value = mate_in(1);
        th.bestMove           = th.rootMoves[0].pv[0];

        printer.printBestmoveWithoutSearch(th,
                                           th.rootMoves[0].pv[0],
                                           th.rootMoves[0].value,
                                           1,
                                           &th.rootMoves[0].pv);
        return;
    }

    // Check for forced database move
    Pos   dbWinMove  = Pos::NONE;
    Value dbWinValue = VALUE_NONE;
    int   dbWinDepth = 0;
    if (th.dbClient) {
        auto childRecords = th.dbClient->queryChildren(*th.board, opts.rule);

        Pos   bestMove      = Pos::NONE;
        Value bestMoveValue = VALUE_NONE;
        for (auto &[pos, record] : childRecords) {
            if (record.label == Database::LABEL_WIN && Value(-record.value) > dbWinValue) {
                dbWinMove  = pos;
                dbWinValue = std::max(Value(-record.value), VALUE_MATE_FROM_DATABASE);
                dbWinDepth = record.depth();
            }
            else if (record.label == Database::LABEL_FORCEMOVE  // A forced best move
                     && Value(record.value) > bestMoveValue) {
                bestMove      = pos;
                bestMoveValue = Value(record.value);
            }
        }

        if (bestMove) {
            th.bestMove = bestMove;
            printer.printBestmoveWithoutSearch(th, bestMove, bestMoveValue, 0, nullptr);
            return;
        }
    }

    // Init time management and transposition table
    timectl.init(opts.turnTime,
                 opts.matchTime,
                 opts.timeLeft,
                 {th.board->ply(), th.board->movesLeft()});
    TT.incGeneration();

    // Starts worker threads, then starts main thread
    printer.printSearchStarts(th, timectl);
    th.runCustomTaskAndWait([this](SearchThread &t) { search(t); }, true);

    // Select best thread according to eval and completed depth when needed
    SearchThread *bestThread = &th;
    if (opts.multiPV == 1 && !SkillMovePicker(opts.strengthLevel).enabled() && !opts.balanceMode)
        bestThread = pickBestThread(th.threads);

    if (opts.balanceMode == SearchOptions::BALANCE_NONE && dbWinMove
        && bestThread->rootMoves[0].value < VALUE_MATE_IN_MAX_PLY) {
        // Try to move the winning move to front
        auto rm = std::find(bestThread->rootMoves.begin(), bestThread->rootMoves.end(), dbWinMove);
        if (rm != bestThread->rootMoves.end()) {
            rm->value = dbWinValue;
            std::stable_sort(bestThread->rootMoves.begin(),
                             bestThread->rootMoves.end(),
                             RootMoveValueComparator {});
        }
        // If database winning move is not in rootmoves, return it directly
        else {
            th.bestMove = dbWinMove;
            printer.printBestmoveWithoutSearch(th, dbWinMove, dbWinValue, dbWinDepth, nullptr);
            return;
        }
    }

    // Output search statistic infomation
    printer.printSearchEnds(th,
                            timectl,
                            bestThread->searchDataAs<ABSearchData>()->completedDepth,
                            *bestThread);

    // Do not record bestmove in pondering
    if (th.inPonder)
        return;

    // Record best move
    th.bestMove = bestThread->rootMoves[0].pv[0];

    // If swap check is needed, make swap decision according to the rule
    if (opts.swapable)
        th.resultAction =
            Opening::decideAction(*th.board, opts.rule, bestThread->rootMoves[0].value);
    else if (opts.balanceMode == SearchOptions::BalanceMode::BALANCE_TWO)
        th.resultAction = ActionType::Move2;
    else
        th.resultAction = ActionType::Move;
}

void ABSearcher::search(SearchThread &th)
{
    ABSearchData     &sd        = *th.searchDataAs<ABSearchData>();
    SearchOptions    &options   = th.options();
    Value             initValue = Evaluation::evaluate(*th.board, options.rule);
    StackArray        stackArray(MAX_PLY, initValue);
    Value             bestValue           = -VALUE_INFINITE;
    Pos               lastBestMove        = Pos::NONE;
    int               lastMoveChangeDepth = 0;
    float             timeReduction = 1.0f, totalBestMoveChanges = 0.0f;
    int               firstMateDepth = 0, firstSingularDepth = 0;
    MainSearchThread *mainThread = (&th == th.threads.main() ? th.threads.main() : nullptr);

    // Init search depth range
    int maxDepth   = std::min(options.maxDepth, std::clamp(Config::MaxSearchDepth, 2, MAX_DEPTH));
    int startDepth = std::clamp(options.startDepth, 1, maxDepth);

    // Init random move picker and adjust max depth and multiPV
    SkillMovePicker rmp(options.strengthLevel);
    sd.multiPv = options.multiPV;
    if (rmp.enabled()) {
        maxDepth   = std::min(maxDepth, rmp.pickDepth());
        sd.multiPv = std::max(sd.multiPv, rmp.minMultiPv());
    }

    // Limit multiPV to the size of root moves
    sd.multiPv = std::min<uint32_t>(sd.multiPv, th.rootMoves.size());

    for (sd.rootDepth = startDepth; sd.rootDepth <= maxDepth && !th.threads.isTerminating();
         sd.rootDepth = pickNextDepth(th.threads, th.id, sd.rootDepth)) {
        // Sync modifications in database client to database storage
        if (th.dbClient && timectl.elapsed() > 5000)
            th.dbClient->sync(false);

        // Age out PV variability metric when depth increases
        totalBestMoveChanges *= 0.5;

        // Save the last iteration's values and PVs before first PV line is searched
        // and all the move values except the (new) PV are set to VALUE_NONE.
        for (RootMove &rm : th.rootMoves) {
            rm.previousValue = rm.value;
            rm.previousPv    = rm.pv;
        }

        // MultiPV loop. We perform a full root search for each PV line
        for (sd.pvIdx = 0; sd.pvIdx < sd.multiPv && !th.threads.isTerminating(); ++sd.pvIdx) {
            // Reset selDepth for each depth and each PV line
            th.selDepth = 0;

            // Do (aspiration window) search using previous value of current pv
            aspirationSearch(options.rule,
                             *th.board,
                             stackArray.rootStack(),
                             th.rootMoves[sd.pvIdx].previousValue,
                             Depth(sd.rootDepth));

            // Send out various infomation to GUI
            if (mainThread)
                printer.printPvCompletes(*mainThread, timectl, sd.rootDepth, sd.pvIdx, sd.multiPv);

            // Sort the PV lines searched so far. When we are in balance move mode,
            // sort according to negetive absolute value rather than its original value.
            if (options.balanceMode)
                std::stable_sort(th.rootMoves.begin(),
                                 th.rootMoves.begin() + sd.pvIdx + 1,
                                 BalanceMoveValueComparator {options.balanceBias});
            else
                std::stable_sort(th.rootMoves.begin(),
                                 th.rootMoves.begin() + sd.pvIdx + 1,
                                 RootMoveValueComparator {});
        }

        // If search is complete, update completed depth.
        if (!th.threads.isTerminating()) {
            sd.completedDepth = sd.rootDepth;
        }

        // Update the best eval and best move.
        bestValue = th.rootMoves[0].value;
        if (th.rootMoves[0].pv[0] != lastBestMove) {
            lastBestMove        = th.rootMoves[0].pv[0];
            lastMoveChangeDepth = sd.rootDepth;
        }

        if (!mainThread)
            continue;

        // Check if we have won or lost, or having only one legal move.
        bool isMate = std::abs(th.rootMoves[0].value) >= VALUE_MATE_IN_MAX_PLY;
        if (isMate && !firstMateDepth)
            firstMateDepth = sd.rootDepth;
        if (sd.singularRoot && !firstSingularDepth)
            firstSingularDepth = sd.rootDepth;

        // Stop thinking if we are not in analysis mode, inPonder mode, or random move mode
        if (!options.isAnalysisMode() && !th.threads.main()->inPonder && !rmp.enabled()) {
            if (isMate && sd.rootDepth - firstMateDepth >= Config::NumIterationAfterMate)
                th.threads.stopThinking();
            else if (sd.singularRoot
                     && sd.rootDepth - firstSingularDepth
                            >= Config::NumIterationAfterSingularRoot) {
                th.threads.main()->markPonderingAvailable();
                th.threads.stopThinking();
            }
        }

        // Send out search result of this depth
        if (mainThread)
            printer.printDepthCompletes(*mainThread, timectl, sd.completedDepth);

        // Check do we have time for the next iteration?
        if (options.timeLimit && !th.threads.isTerminating()) {
            // Accumulate all best move changes across threads
            for (const auto &th : th.threads) {
                totalBestMoveChanges += sd.bestMoveChanges;
                sd.bestMoveChanges = 0;
            }

            // Stop the search if exceeds time limit (do not stop in inPonder mode)
            if (timectl.checkStop({sd.rootDepth,
                                   lastMoveChangeDepth,
                                   bestValue,
                                   previousBestValue,
                                   previousTimeReduction,
                                   totalBestMoveChanges / th.threads.size()},
                                  timeReduction)
                && !mainThread->inPonder) {
                // Start pondering after foreground searching naturally ended
                mainThread->markPonderingAvailable();
                th.threads.stopThinking();
            }
        }
    }

    // Sync modifications in database client to database storage
    if (th.dbClient)
        th.dbClient->sync();

    if (!mainThread)
        return;

    // Save time adjustment infomation for next search
    previousBestValue     = bestValue;
    previousTimeReduction = timeReduction;

    // If random move picker is enabled, swap best PV line with a sub-optimal one
    if (rmp.enabled()) {
        std::swap(
            th.rootMoves[0],
            *std::find(th.rootMoves.begin(), th.rootMoves.end(), rmp.pick(th.threads, sd.multiPv)));
    }
}

bool ABSearcher::checkTimeupCondition()
{
    return timectl.elapsed() >= timectl.maximum();
}

int ABSearcher::pickNextDepth(ThreadPool &threads, uint32_t thisId, int lastDepth) const
{
    if (thisId == 0 || threads.size() < 3)
        return lastDepth + 1;

    for (int nextDepth = lastDepth + 1;; nextDepth++) {
        size_t numThreadsAboveNextDepth = 0;

        for (const auto &th : threads) {
            if (th->id != thisId
                && th->searchDataAs<ABSearchData>()->completedDepth + 1 >= nextDepth)
                numThreadsAboveNextDepth++;
        }

        // If more than half of all threads are already searching depth above
        // the next depth, skip and find another next depth.
        if (numThreadsAboveNextDepth > threads.size() / 2)
            continue;
        else
            return nextDepth;
    }
}

SearchThread *ABSearcher::pickBestThread(ThreadPool &threads) const
{
    SearchThread *bestThread = threads.main();

    // In case of database recording mode, only use main thread result.
    if (bestThread->dbClient && !Config::DatabaseReadonlyMode)
        return bestThread;

    // Find minimum value of all threads
    Value minValue = bestThread->rootMoves[0].value;
    for (size_t i = 1; i < threads.size(); i++)
        minValue = std::min(minValue, threads[i]->rootMoves[0].value);

    // Vote moves according to value and depth
    std::unordered_map<Pos, int64_t> votes;
    constexpr int                    DepthBias = 15;
    for (const auto &th : threads) {
        Pos   move  = th->rootMoves[0].pv[0];
        Value value = th->rootMoves[0].value;

        votes[move] +=
            (value - minValue + DepthBias) * th->searchDataAs<ABSearchData>()->completedDepth;

        // If best thread has a winning move, other threads must have a shorter
        // mate to replace it.
        if (std::abs(bestThread->rootMoves[0].value) >= VALUE_MATE_IN_MAX_PLY) {
            if (value > bestThread->rootMoves[0].value)
                bestThread = th.get();
        }
        // Else if current thread has a winning/lossing move while best thread
        // has not, replace best thread with current thread.
        else if (std::abs(value) >= VALUE_MATE_IN_MAX_PLY && value != VALUE_NONE) {
            bestThread = th.get();
        }
        // Else if current thread has a non-lossing move, select best thread
        // according to votes.
        else if (value > VALUE_MATED_IN_MAX_PLY
                 && votes[move] > votes[bestThread->rootMoves[0].pv[0]]) {
            bestThread = th.get();
        }
    }

    return bestThread;
}

namespace {

/// The aspiration window search loop. First start with a small aspiration window, in the case
/// of a fail high/low, re-search with a bigger window until we don't fail high/low anymore.
void aspirationSearch(Rule rule, Board &board, SearchStack *ss, Value prevValue, Depth depth)
{
    Value         delta, alpha, beta;
    SearchThread *thisThread  = board.thisThread();
    ABSearchData *searchData  = thisThread->searchDataAs<ABSearchData>();
    ABSearcher   *searcher    = static_cast<ABSearcher *>(thisThread->threads.searcher());
    int           failHighCnt = 0;

    // Reset aspiration window starting size if aspiration windows is enabled and
    // we are not in balance move mode. (no aspiration window for balance move mode).
    if (depth >= ASPIRATION_DEPTH && Config::AspirationWindow
        && !thisThread->options().balanceMode) {
        delta = nextAspirationWindowDelta(prevValue);
        alpha = std::max(prevValue - delta, -VALUE_INFINITE);
        beta  = std::min(prevValue + delta, VALUE_INFINITE);
    }
    else {
        alpha = delta = -VALUE_INFINITE;
        beta          = VALUE_INFINITE;
    }

    // Loop until we got a search value that lies in the aspiration window.
    while (true) {
        searchData->rootAlpha = alpha;

        // Decrease search depth if multiple fail high occurs
        Depth adjustedDepth = std::max(1.0f, depth - failHighCnt / 3);

        // Search at root node with rule
        Value value = ::search<Root>(rule, board, ss, alpha, beta, adjustedDepth, false);

        // Bring the best move to the front. Stable sorting is used as all the value but the first
        // are set to -VALUE_INFINITE and we want to keep the same order for all the moves except
        // the new PV that goes to the front.
        // In case of MultiPV search, the already searched PV lines are preserved.
        // When in balance mode, sort according to negetive absolute value.
        if (thisThread->options().balanceMode)
            std::stable_sort(thisThread->rootMoves.begin() + searchData->pvIdx,
                             thisThread->rootMoves.end(),
                             BalanceMoveValueComparator {});
        else
            std::stable_sort(thisThread->rootMoves.begin() + searchData->pvIdx,
                             thisThread->rootMoves.end(),
                             RootMoveValueComparator {});

        // If search has been stopped, break immediately. Sorting result is safe to use.
        if (thisThread->threads.isTerminating())
            break;

        // Update current move's sel depth
        thisThread->rootMoves[searchData->pvIdx].selDepth = thisThread->selDepth;

        // Print value out of window result
        if (value <= alpha || value >= beta)
            searcher->printer.printOutOfWindowResult(*static_cast<MainSearchThread *>(thisThread),
                                                     searcher->timectl,
                                                     searchData->rootDepth,
                                                     searchData->pvIdx,
                                                     searchData->multiPv,
                                                     alpha,
                                                     beta);

        // In case of failing low/high increase aspiration window and re-search,
        // otherwise exit the loop.
        if (value <= alpha) {
            beta        = (alpha + beta) / 2;
            alpha       = std::max(value - delta, -VALUE_INFINITE);
            failHighCnt = 0;
        }
        else if (value >= beta) {
            beta = std::min(value + delta, VALUE_INFINITE);
            failHighCnt++;
        }
        else
            break;

        delta = nextAspirationWindowDelta(value, delta);
        assert(alpha >= -VALUE_INFINITE && beta <= VALUE_INFINITE);
    }
}

/// Dynamic dispatch templated search() with correct rule. It is also
/// responsible for iterating the first move in balance2 mode.
template <NodeType NT>
Value search(Rule         rule,
             Board       &board,
             SearchStack *ss,
             Value        alpha,
             Value        beta,
             Depth        depth,
             bool         cutNode)
{
    assert(rule < RULE_NB);

    auto searchWithRule = [&](Rule r) {
        switch (r) {
        default:
        case Rule::FREESTYLE:
            return search<Rule::FREESTYLE, NT>(board, ss, alpha, beta, depth, cutNode);
        case Rule::STANDARD:
            return search<Rule::STANDARD, NT>(board, ss, alpha, beta, depth, cutNode);
        case Rule::RENJU: return search<Rule::RENJU, NT>(board, ss, alpha, beta, depth, cutNode);
        }
    };

    // Loop for first move in balance move mode
    SearchThread *thisThread = board.thisThread();
    if (NT == Root && thisThread->options().balanceMode == SearchOptions::BALANCE_TWO) {
        Value bestValue = -VALUE_INFINITE;
        ss->moveCount   = 0;
        MovePicker mp(rule,
                      board,
                      MovePicker::ExtraArgs<MovePicker::MAIN> {
                          thisThread->rootMoves[0].pv[0],
                          &thisThread->searchDataAs<ABSearchData>()->mainHistory,
                          &thisThread->searchDataAs<ABSearchData>()->counterMoveHistory,
                      });

        // Refresh root move index in balance2Moves
        for (size_t i = 0; i < thisThread->rootMoves.size(); i++) {
            Balance2Move b2move {thisThread->rootMoves[i].pv[0], thisThread->rootMoves[i].pv[1]};
            thisThread->balance2Moves[b2move] = i;
        }

        // Iterate first move in balance2 move pair
        while (Pos move = mp()) {
            board.move(rule, move);
            Value value = searchWithRule(rule);
            board.undo(rule);
            assert(value <= 0);

            // Check for a new best value and update alpha, beta
            if (value > bestValue) {
                bestValue = value;
                if (value > alpha) {
                    alpha = value;
                    beta  = -alpha;
                    // Break if search window size reaches zero
                    if (alpha >= beta)
                        return bestValue;
                }
            }
        }
        return bestValue;
    }
    else {
        return searchWithRule(rule);
    }
}

/// The main search function for both PV and non-PV nodes
template <Rule Rule, NodeType NT>
Value search(Board &board, SearchStack *ss, Value alpha, Value beta, Depth depth, bool cutNode)
{
    constexpr bool PvNode   = NT == PV || NT == Root;
    constexpr bool RootNode = NT == Root;

    // Do some sanity check over input arguments
    assert(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);
    assert(PvNode || (alpha == beta - 1));
    assert(DEPTH_LOWER_BOUND <= depth && depth < DEPTH_UPPER_BOUND);
    assert(!(PvNode && cutNode));
    assert(0 <= ss->ply && ss->ply < MAX_PLY);

    // Step 1. Initialize node
    SearchThread  *thisThread = board.thisThread();
    ABSearchData  *searchData = thisThread->searchDataAs<ABSearchData>();
    SearchOptions &options    = thisThread->options();
    thisThread->numNodes.fetch_add(1, std::memory_order_relaxed);

    Color    self = board.sideToMove(), oppo = ~self;
    uint16_t oppo5 = board.p4Count(oppo, A_FIVE);           // opponent five
    uint16_t oppo4 = oppo5 + board.p4Count(oppo, B_FLEX4);  // opponent straight four and five

    // Dive into vcf search when the depth reaches zero (~17 elo)
    if (depth <= 0.0f) {
        return oppo5 ? vcfdefend<Rule, NT>(board, ss, alpha, beta)
                     : vcfsearch<Rule, NT>(board, ss, alpha, beta);
    }

    int            moveCount = 0, nonMatedCount = 0;
    Value          bestValue = -VALUE_INFINITE, maxValue = VALUE_INFINITE, value;
    Pos            bestMove = Pos::NONE;
    HistoryTracker histTracker(board, ss);

    // Update selDepth (selDepth counts from 1, ply from 0)
    if (PvNode && thisThread->selDepth <= ss->ply)
        thisThread->selDepth = ss->ply + 1;

    if (!RootNode) {
        // Step 2. Check for aborted search and immediate draw
        // Check if we reach the time limit
        if (thisThread->isMainThread())
            static_cast<MainSearchThread *>(thisThread)->checkExit();

        // Check if the board has been filled or we have reached the max game ply.
        if (board.movesLeft() == 0 || board.nonPassMoveCount() >= options.maxMoves)
            return getDrawValue(board, options, ss->ply);

        // Check if we have reached the max ply
        if (ss->ply >= MAX_PLY)
            return Evaluation::evaluate<Rule>(board, alpha, beta);

        // Check for immediate winning
        if ((value = quickWinCheck<Rule>(board, ss->ply, beta)) != VALUE_ZERO) {
            // Do not return mate that longer than maxMoves option
            if (board.nonPassMoveCount() + mate_step(value, ss->ply) > options.maxMoves)
                value = getDrawValue(board, options, ss->ply);

            return value;
        }

        // Step 3. Mate distance pruning.
        alpha = std::max(mated_in(ss->ply), alpha);
        beta  = std::min(mate_in(ss->ply + 1), beta);
        if (alpha >= beta)
            return alpha;

        // Initialize statScore to zero for the grandchildren of the current position.
        // So statScore is shared between all grandchildren and only the first grandchild
        // starts with statScore = 0. Later grandchildren start with the last calculated
        // statScore of the previous grandchild.
        (ss + 2)->statScore = 0;

        // Pass current number of null moves to next ply
        (ss + 1)->numNullMoves = ss->numNullMoves;
    }
    else
        searchData->rootDelta = beta - alpha;

    // Step 4. Transposition table lookup.
    // Use a different hash key in case of an skip move to avoid overriding full search result.
    Pos     skipMove = ss->skipMove;
    HashKey posKey   = board.zobristKey() ^ (skipMove ? Hash::LCHash(skipMove) : 0);
    Value   ttValue  = VALUE_NONE;
    Value   ttEval   = VALUE_NONE;
    bool    ttIsPv   = false;
    Bound   ttBound  = BOUND_NONE;
    Pos     ttMove   = Pos::NONE;
    int     ttDepth  = 0;
    bool    ttHit    = TT.probe(posKey, ttValue, ttEval, ttIsPv, ttBound, ttMove, ttDepth, ss->ply);
    if (RootNode && searchData->completedDepth.load(std::memory_order_relaxed))
        ttMove = thisThread->rootMoves[0].pv[options.balanceMode == SearchOptions::BALANCE_TWO];
    if (!skipMove)
        ss->ttPv = PvNode || ttHit && ttIsPv;
    (ss + 1)->ttPv = false;

    // At non-PV nodes we check for an early TT cutoff
    if (!PvNode && ttHit && ttDepth >= depth
        && (ttBound & (ttValue >= beta ? BOUND_LOWER : BOUND_UPPER))) {
        // Update move heruistics for ttMove
        histTracker.updateTTMoveStats(depth, ttMove, ttValue, beta);
        return ttValue;
    }

    // Step 5. Database query
    Database::DBRecord dbRecord;
    bool               dbHit = false, dbCheckChild = false;
    Bound              dbLabelBound = BOUND_NONE, dbBound = BOUND_NONE;
    Value              dbValue;
    if (thisThread->dbClient) {
        Database::DBClient &dbClient          = *thisThread->dbClient;
        int                 queryPlyIncrement = searchData->rootDepth
                                / (PvNode ? Config::DatabaseQueryPVIterPerPlyIncrement
                                          : Config::DatabaseQueryNonPVIterPerPlyIncrement);
        int queryPly = Config::DatabaseQueryPly + queryPlyIncrement;

        if (!skipMove               // Skip query in singular extension
            && ss->ply <= queryPly  // Only query in the first plies to avoid large speed loss
            && dbClient.query(board, Rule, dbRecord)) {
            dbHit        = true;
            dbValue      = storedValueToSearchValue(dbRecord.value, ss->ply);
            dbBound      = dbRecord.bound();
            dbCheckChild = ss->ply < queryPly;

            switch (dbRecord.label) {
            case Database::LABEL_NULL: break;
            case Database::LABEL_BLOCKMOVE:  // Block this move
                return -VALUE_BLOCKED;
            case Database::LABEL_WIN:  // Win for opponent, loss for self
                dbValue = std::min(dbValue, VALUE_MATED_FROM_DATABASE);
                if (dbBound != BOUND_EXACT)
                    dbBound = BOUND_UPPER;
                dbLabelBound = BOUND_UPPER;
                goto try_database_cut;
            case Database::LABEL_LOSE:  // Loss for opponent, win for self
                dbValue = std::max(dbValue, VALUE_MATE_FROM_DATABASE);
                if (dbBound != BOUND_EXACT)
                    dbBound = BOUND_LOWER;
                dbLabelBound = BOUND_LOWER;
                goto try_database_cut;
            case Database::LABEL_DRAW:  // Draw for both side
                dbValue      = getDrawValue(board, options, ss->ply);
                dbBound      = BOUND_EXACT;
                dbLabelBound = BOUND_EXACT;
                goto try_database_cut;
            default:  // Read depth, bound, value
                goto try_database_cut;

            try_database_cut:
                if (RootNode)
                    break;

                int  dbDepth = dbRecord.depth() + Config::DatabaseQueryResultDepthBoundBias;
                bool dbCut =
                    dbDepth > depth
                    && (dbValue >= beta ? (dbBound & BOUND_LOWER) : (dbBound & BOUND_UPPER));
                if (!PvNode && dbCut
                    || dbLabelBound == BOUND_LOWER && dbValue >= beta   // Win-score-cut
                    || dbLabelBound == BOUND_UPPER && dbValue <= alpha  // Loss-score-cut
                    || dbLabelBound == BOUND_EXACT                      // Draw-score-cut
                ) {
                    TT.store(posKey,
                             dbValue,
                             ss->staticEval,
                             ss->ttPv,
                             dbBound,
                             Pos::NONE,
                             dbLabelBound == BOUND_EXACT
                                 ? (int)DEPTH_UPPER_BOUND
                                 : std::min(dbDepth, (int)DEPTH_UPPER_BOUND),
                             ss->ply);
                    return dbValue;
                }

                // Expand search window for database cut in PvNode
                if (PvNode && (dbCut || dbLabelBound != BOUND_NONE))
                    alpha = -VALUE_INFINITE, beta = VALUE_INFINITE;
            }
        }
    }

    // Step 6. Static evaluation
    Value eval        = VALUE_NONE;
    int   improvement = 0;  // Static eval change in the last two ply

    // Reset killer of grand-children
    (ss + 2)->killers[0] = Pos::NONE;
    (ss + 2)->killers[1] = Pos::NONE;

    if (oppo4) {
        // Use static evaluation from previous ply if opponent makes a four/five attack
        ss->staticEval = -(ss - 1)->staticEval;

        // Skip early pruning when we only have one possible move
        if (oppo5)
            goto moves_loop;
    }
    else if (!RootNode) {
        if (ttHit) {
            // Never assume anything about values stored in TT
            ss->staticEval = eval = ttEval;
            if (eval == VALUE_NONE)
                ss->staticEval = eval = Evaluation::evaluate<Rule>(board, alpha, beta);

            // Try to use ttValue as a better eval estimation
            if (ttValue != VALUE_NONE && (ttBound & (ttValue > eval ? BOUND_LOWER : BOUND_UPPER)))
                eval = ttValue;
        }
        else {
            ss->staticEval = eval = Evaluation::evaluate<Rule>(board, alpha, beta);

            // Save static evaluation into transposition table
            if (!skipMove)
                TT.store(posKey,
                         VALUE_NONE,
                         ss->staticEval,
                         ss->ttPv,
                         BOUND_NONE,
                         Pos::NONE,
                         (int)DEPTH_NONE,
                         ss->ply);
        }

        improvement = ss->staticEval - (ss - 2)->staticEval;
    }

    // Step 7. Razoring with VCF (~68 elo)
    if (!PvNode
        && (alpha < VALUE_MATE_IN_MAX_PLY || !ttHit)  // We are not searching for a short win
        && eval + razorMargin<Rule>(depth) < alpha) {
        return vcfsearch<Rule, NonPV>(board, ss, alpha, alpha + 1);
    }

    // Step 8. Futility pruning: child node (~121 elo)
    if (!PvNode && eval < VALUE_MATE_IN_MAX_PLY  // Do not return unproven wins
        && beta > VALUE_MATED_IN_MAX_PLY         // Confirm non-losing move exists
        && eval - futilityMargin<Rule>(depth - 1, cutNode && !ttHit, improvement > 0) >= beta
        && !((ss - 2)->moveP4[self] >= E_BLOCK4 && (ss - 4)->moveP4[self] >= E_BLOCK4))
        return eval;

    // Step 9. Null move pruning (~3 elo)
    if (!PvNode && !oppo4 && !skipMove && eval >= beta
        && board.getLastMove() != Pos::PASS  // No consecutive pass moves
        && ss->staticEval >= beta + nullMoveMargin<Rule>(depth)) {
        Depth r         = nullMoveReduction<Rule>(depth);
        ss->currentMove = Pos::PASS;

        (ss + 1)->numNullMoves++;
        board.move<Rule>(Pos::PASS);
        TT.prefetch(board.zobristKey());
        value = -search<Rule, NonPV>(board, ss + 1, -beta, -beta + 1, depth - r, !cutNode);
        board.undo<Rule>();
        (ss + 1)->numNullMoves--;

        if (value >= beta) {
            // Do not return unproven mate scores
            if (value >= VALUE_MATE_IN_MAX_PLY)
                value = beta;

            if (std::abs(beta) < VALUE_MATE_IN_MAX_PLY)
                return value;
            else {
                // Do verification search at high depths, with null move pruning disabled
                Value v = search<Rule, NonPV>(board, ss, beta - 1, beta, depth - r, false);
                if (v >= beta)
                    return value;
            }
        }
    }

    // Step 10. Internal iterative reduction (~3 elo)
    if (!RootNode && PvNode && !ttMove)
        depth -= IIR_REDUCTION_PV;

    // Reduce for pv ttMove that has not been chosen for a few iterations (~32 elo)
    if (PvNode && depth > 1 && ttMove)
        depth -= std::clamp((depth - ttDepth) * IIR_REDUCTION_TT, 0.0f, IIR_REDUCTION_TT_MAX);

    // Drop to vcfsearch if depth is below zero
    if (depth <= 0)
        return vcfsearch<Rule, NT>(board, ss, alpha, beta);

    // Internal iterative deepening on higher depth (~1 elo)
    if (depth >= IID_DEPTH && !ttMove) {
        depth -= IIR_REDUCTION;

        // We only need best move from the iid search, so we just discard its result
        search<Rule, NT>(board, ss, alpha, beta, depth - iidDepthReduction<Rule>(depth), cutNode);

        // Get best move from transposition table, which should be just written by the search.
        Value tmpEval;
        bool  tmpIsPv;
        Bound tmpBound;
        int   tmpDepth;
        ttHit = TT.probe(posKey, ttValue, tmpEval, tmpIsPv, tmpBound, ttMove, tmpDepth, ss->ply);
    }

    // When opponent is doing A_FIVE attack, search starts from here
moves_loop:
    ABSearcher  *searcher    = static_cast<ABSearcher *>(thisThread->threads.searcher());
    TimeControl &timectl     = searcher->timectl;
    uint64_t     curNumNodes = 0;
    ss->dbChildWritten       = false;

    // Calculate a complexity metric for current position
    uint16_t complexCount = 1;
    for (int pat4 = L_FLEX2; pat4 <= D_BLOCK4_PLUS; pat4++) {
        complexCount += board.p4Count(self, (Pattern4)pat4);
        complexCount += board.p4Count(oppo, (Pattern4)pat4);
    }
    float complexity = ::logf(complexCount);

    // Fail-High reduction (~50 elo)
    // Indicate cutNode that will probably fail high if current eval is far above beta
    bool likelyFailHigh = !PvNode && cutNode && eval >= beta + failHighMargin(depth, oppo4);

    MovePicker mp(Rule,
                  board,
                  MovePicker::ExtraArgs<MovePicker::MAIN> {
                      ttMove,
                      &searchData->mainHistory,
                      &searchData->counterMoveHistory,
                  });

    // Step 11. Loop through all legal moves until no moves remain
    // or a beta cutoff occurs.
    while (Pos move = mp()) {
        assert(board.isLegal(move));

        // Skip excluded move when in Singular extension search
        if (!RootNode && move == skipMove)
            continue;

        if (RootNode) {
            if (options.balanceMode == SearchOptions::BALANCE_TWO) {
                Balance2Move b2move {board.getLastMove(), move};

                // Skip balance move pair not listed in Root Move List
                if (thisThread->balance2Moves.find(b2move) == thisThread->balance2Moves.end())
                    continue;

                // Skip moves not in root move list and PV moves that have been already searched
                if (std::count(thisThread->rootMoves.begin(),
                               thisThread->rootMoves.begin() + searchData->pvIdx,
                               b2move))
                    continue;

                // Restore previous movecount from stack
                if (!moveCount)
                    moveCount = ss->moveCount;

                // Prevent moveCount overflow in reduction()
                moveCount = std::min(moveCount, MAX_MOVES - 1);
            }
            // Skip moves not in root move list and PV moves that have been already searched
            else if (!std::count(thisThread->rootMoves.begin() + searchData->pvIdx,
                                 thisThread->rootMoves.end(),
                                 move))
                continue;

            curNumNodes = thisThread->numNodes.load(std::memory_order_relaxed);
        }

        if (RootNode && thisThread->isMainThread())
            searcher->printer.printEnteringMove(*static_cast<MainSearchThread *>(thisThread),
                                                timectl,
                                                searchData->pvIdx,
                                                searchData->rootDepth,
                                                move);

        // Initialize heruistic information
        ss->moveCount     = ++moveCount;
        ss->moveP4[BLACK] = board.cell(move).pattern4[BLACK];
        ss->moveP4[WHITE] = board.cell(move).pattern4[WHITE];

        // False forbidden move in Renju is considered as important move
        bool importantMove = ss->moveP4[self] >= J_FLEX2_2X || ss->moveP4[oppo] >= H_FLEX3
                             || (Rule == Rule::RENJU && ss->moveP4[BLACK] == FORBID);
        bool trivialMove = ss->moveP4[BLACK] == NONE && ss->moveP4[WHITE] == NONE;

        int  distOppo = Pos::distance(move, (ss - 1)->currentMove);
        int  distSelf = Pos::distance(move, (ss - 2)->currentMove);
        bool distract = distSelf > (Rule == RENJU ? 5 : 4) && distOppo > 4;

        // Step 12. Pruning at shallow depth
        // Do pruning only when we have non-losing moves, otherwise we may have a false mate.
        // Also skip pruning if there might be a child database record we want to read.
        if (!RootNode && !dbCheckChild && bestValue > VALUE_MATED_IN_MAX_PLY) {
            // Move count pruning: skip move if movecount is above threshold (~107 elo)
            if (moveCount >= futilityMoveCount(depth, improvement > 0))
                continue;

            // Skip trivial moves at lower depth (~2 elo at LTC)
            if (trivialMove && depth < TRIVIAL_PRUN_DEPTH)
                continue;

            // Policy based pruning (~10 elo)
            if (mp.hasPolicyScore() && mp.curMoveScore() < policyPruningScore<Rule>(depth))
                continue;

            // Prun distract defence move which is likely to delay a winning (~2 elo)
            if (oppo4 && depth < TRIVIAL_PRUN_DEPTH && ss->moveP4[oppo] < E_BLOCK4 && distract)
                continue;
        }

        // Step 13. Extensions
        Depth extension = 0;

        // Singular response extension for opponent B4 attack (~77 elo)
        if (oppo5)
            extension = OPPO5_EXT;

        // Singular extension: only one move fails high while other moves fails low on a search of
        // (alpha-s, beta-s), then this move is singular and should be extended. (~52 elo)
        else if (!RootNode && depth >= SE_DEPTH && move == ttMove
                 && !skipMove                                  // No recursive singular search
                 && std::abs(ttValue) < VALUE_MATE_IN_MAX_PLY  // ttmove value is not a mate
                 && (ttBound & BOUND_LOWER)                    // ttMove failed high last time
                 && ttDepth >= depth - SE_TTE_DEPTH            // ttEntry has enough depth to trust
        ) {
            bool  formerPv = !PvNode && ss->ttPv;
            Value singularBeta =
                std::max(ttValue - singularMargin<Rule>(depth, formerPv), -VALUE_MATE);

            // Backup current P4
            // Pattern4 moveP4Backup[SIDE_NB] = {ss->moveP4[BLACK], ss->moveP4[WHITE]};

            // Exclude ttMove from the reduced depth search to see the value of second best move
            ss->skipMove = move;
            value        = search<Rule, NonPV>(board,
                                        ss,
                                        singularBeta - 1,
                                        singularBeta,
                                        depth - singularReduction(depth, formerPv),
                                        cutNode);
            ss->skipMove = Pos::NONE;

            // Restore current P4
            // ss->moveP4[BLACK] = moveP4Backup[BLACK];
            // ss->moveP4[WHITE] = moveP4Backup[WHITE];

            // Extend if only the ttMove fails high, while other moves fails low.
            if (value < singularBeta) {
                // Extend two ply if current non-pv position is highly singular.
                if (!PvNode && value < singularBeta - doubleSEMargin(depth)
                    && ss->extraExtension < SE_EXTRA_MAX_DEPTH)
                    extension = 2.0f;
                else
                    extension = 1.0f;
            }
            // Multi-cut pruning: we also failed high on a reduced search without ttMove.
            else if (singularBeta >= beta)
                return beta;
            // Reduce if we are likely to fail high.
            else if (ttValue >= beta)
                extension = -SE_REDUCTION_FH;
        }

        // Extension for ttmove without singular extension (~12 elo)
        else if (move == ttMove) {
            // Extension for ttmove
            extension = PvNode ? TTM_EXT_PV : TTM_EXT_NONPV;

            // Additional extension for near B4 ttmove
            if (ss->moveP4[self] >= E_BLOCK4 && distSelf <= 6)
                extension += (distSelf <= 4 ? NEARB4_EXT_DIST4 : NEARB4_EXT_DIST6);
        }

        // Fail high reduction (~8 elo)
        if (likelyFailHigh) {
            if (ss->moveP4[self] >= E_BLOCK4) {
                // If we failed high for two continous E_BLOCK4 moves, extend rather than reduce
                if ((ss - 2)->moveP4[self] >= E_BLOCK4)
                    extension += 1.0f;
            }
            else
                extension -= 1.0f;
        }

        // Calculate new depth for this move
        Depth newDepth     = depth - 1.0f + extension;
        ss->currentMove    = move;
        ss->extraExtension = (ss - 1)->extraExtension + std::max(extension - 1.0f, 0.0f);

        // Step 14. Make the move
        board.move<Rule>(move);
        TT.prefetch(board.zobristKey());

        // Step 15. Late move reduction (LMR). Moves are searched with a reduced
        // depth and will be re-searched at full depth if fail high.
        if (depth >= 2 && moveCount > 1 + RootNode) {
            Depth r = reduction<Rule, PvNode>(searcher->reductions,
                                              depth,
                                              moveCount,
                                              improvement,
                                              beta - alpha,
                                              searchData->rootDelta);

            // Policy based reduction (~59 elo)
            if (mp.hasPolicyScore())
                r += policyReduction<Rule>(mp.curMoveScore()
                                           * (0.1f / Evaluation::PolicyBuffer::ScoreScale));

            // Dynamic reduction based on complexity (~2 elo)
            r += complexity * complexityReduction<Rule>(trivialMove, importantMove, distract);

            // Decrease reduction if position is or has been on the PV (~10 elo)
            if (ss->ttPv)
                r -= TTPV_NEG_REDUCTION;

            // Increase reduction for nodes that does not improve root alpha (~0 elo)
            if (!RootNode && (ss->ply & 1) && bestValue >= -searchData->rootAlpha)
                r += NO_ALPHA_IMPROVING_REDUCTION;

            // Increase reduction for cut nodes if is not killer moves (~5 elo)
            if (cutNode && !(!oppo4 && ss->isKiller(move) && ss->moveP4[self] < H_FLEX3))
                r += NOKILLER_CUTNODE_REDUCTION;

            // Increase reduction for useless defend move (~6 elo)
            if (oppo4 && ss->moveP4[oppo] < E_BLOCK4) {
                r += (distOppo > 4 ? OPPO_USELESS_DEFEND_REDUCTION : 0);
                r += (distSelf > 4 ? SELF_USELESS_DEFEND_REDUCTION : 0);
            }

            // Decrease reduction for continuous attack (~5 elo)
            if (!oppo4 && (ss - 2)->moveP4[self] >= H_FLEX3
                && (ss->moveP4[self] >= H_FLEX3 || distSelf <= 4 && ss->moveP4[self] >= J_FLEX2_2X))
                r -= CONTINUOUS_ATTACK_EXT;

            if constexpr (Rule == Rule::RENJU) {
                // Decrease reduction for false forbidden move in Renju (~6 elo)
                if (ss->moveP4[BLACK] == FORBID)
                    r -= FALSE_FORBID_LESS_REDUCTION;
            }

            // Update statScore of this node
            ss->statScore = statScore(searchData->mainHistory, self, move);

            // Decrease/increase reduction for moves with a good/bad history (~9 elo)
            r -= extensionFromStatScore(ss->statScore, depth);

            // Allow LMR to do deeper search in some circumstances
            // Clamp the LMR depth to newDepth (no depth less than one)
            Depth d = std::max(std::min(newDepth - r, newDepth + 1), 1.0f);

            value = -search<Rule, NonPV>(board, ss + 1, -(alpha + 1), -alpha, d, true);

            if (value > alpha && d < newDepth) {
                // Extra extension in lmr (~13 elo)
                Depth ext = lmrExtension<Rule>(newDepth, d, value, alpha, bestValue);
                // Do not allow more extension if extra extension is already high
                if (ss->extraExtension >= LMR_EXTRA_MAX_DEPTH)
                    ext = std::min(ext, 1.0f);
                else
                    ss->extraExtension += std::max(ext - 1.0f, 0.0f);
                newDepth += ext;

                if (d < newDepth)
                    value = -search<Rule, NonPV>(board,
                                                 ss + 1,
                                                 -(alpha + 1),
                                                 -alpha,
                                                 newDepth,
                                                 !cutNode);
            }
        }

        // Step 16. Full depth search when LMR is skipped or fails high
        else if (!PvNode || moveCount > 1) {
            // If expected reduction is high, we reduce search depth by 1 here
            value = -search<Rule, NonPV>(board, ss + 1, -(alpha + 1), -alpha, newDepth, !cutNode);
        }

        // For balance move mode, we also check if a move can trigger beta cut
        // (which is too good to be balanced), so we can safely discard this move.
        if (RootNode && options.balanceMode && moveCount > 1 && value > alpha)
            value = -search<Rule, NonPV>(board, ss + 1, -beta, -(beta - 1), newDepth, !cutNode);

        // For PV nodes only, do a full PV search on the first move or after a fail
        // high (in the latter case search only if value < beta), otherwise let the
        // parent node fail low with value <= alpha and try another move.
        if (PvNode
            && (moveCount == 1
                || (RootNode && options.balanceMode
                        ? balancedValue(value, options.balanceBias) > alpha
                        : value > alpha && (RootNode || value < beta)))) {
            (ss + 1)->pv[0]        = Pos::NONE;
            (ss + 1)->dbValueDepth = INT16_MIN;  // Clear database value depth of next move
            value = -search<Rule, PV>(board, ss + 1, -beta, -alpha, newDepth, false);
        }

        // Step 17. Undo move
        board.undo<Rule>();

        if (RootNode && thisThread->isMainThread())
            searcher->printer.printLeavingMove(*static_cast<MainSearchThread *>(thisThread),
                                               timectl,
                                               searchData->pvIdx,
                                               searchData->rootDepth,
                                               move);

        // Step 18. Check for a new best move
        // Finished searching the move. If a stop occurred, the return value of the search cannot
        // be trusted, so we return immediately without updating best move, PV and TT.
        if (thisThread->threads.isTerminating()) {
            if (thisThread->dbClient
                && !dbHit  // Write when no dbHit, we never overwrite any existing record with null
                && !Config::DatabaseReadonlyMode  // Never write in database readonly mode
                && ss->dbChildWritten  // Write anyway if we have children that have already written
            ) {
                Database::DBRecord newRecord {Database::LABEL_NULL, 0, 0};
                thisThread->dbClient->save(board, Rule, newRecord, Config::DatabaseOverwriteRule);
                (ss - 1)->dbChildWritten = true;
            }
            return VALUE_NONE;
        }
        // This move is blocked from database record
        if (value == VALUE_BLOCKED)
            continue;

        assert(value > -VALUE_INFINITE && value < VALUE_INFINITE);

        if (RootNode) {
            const bool balance2  = options.balanceMode == SearchOptions::BALANCE_TWO;
            Value      moveValue = value;
            RootMove  &rm =
                balance2
                     ? thisThread->rootMoves
                          [thisThread->balance2Moves[Balance2Move {board.getLastMove(), move}]]
                     : *std::find(thisThread->rootMoves.begin(), thisThread->rootMoves.end(), move);

            // If we are in balance move mode, map the original move value to its negetive
            // absolute value, which makes best move and PV selection based on how balanced
            // a move is rather than how good a move is. Note that move value recorded in
            // RootMove is kept unchanged for better eval output.
            if (options.balanceMode)
                value = balancedValue(value, options.balanceBias);

            // Check for PV move or new best move.
            rm.numNodes += thisThread->numNodes.load(std::memory_order_relaxed) - curNumNodes;
            rm.selDepth = thisThread->selDepth;
            if (moveCount == 1 || value > alpha) {
                rm.value = moveValue;
                rm.pv.resize(1 + balance2);

                assert((ss + 1)->pv);

                for (Pos *m = (ss + 1)->pv; *m != Pos::NONE; m++)
                    rm.pv.push_back(*m);

                assert(!balance2 || rm.pv.size() >= 2);

                // We record how often the best move has been changed in each iteration.
                // When the best move changes frequently, we allocate some more time.
                if (moveCount > 1)
                    searchData->bestMoveChanges++;
            }
            else {
                // All other moves but PV are set to the lowest value: this is not a problem as
                // the sort is stable and its rank in rootmoves is kept, just the PV is pushed up.
                rm.value = VALUE_NONE;
            }

            // We record how many moves is not mated at root, which can indicate singular root.
            if (value > VALUE_MATED_IN_MAX_PLY)
                nonMatedCount++;
        }

        // Update best value, best move and PV.
        if (value > bestValue) {
            bestValue = value;

            // Update database value depth of current move
            if (PvNode)
                ss->dbValueDepth = (ss + 1)->dbValueDepth + 1;

            if (value > alpha) {
                bestMove = move;  // Only update best move in pv or fail high node

                if (PvNode && !RootNode)  // Update pv even in fail-high case
                    ss->updatePv(move);

                if (value >= beta) {
                    break;  // Fail high
                }
                else {
                    alpha = value;  // Update alpha, make sure alpha < beta

                    if (RootNode)
                        searchData->rootAlpha = alpha;

                    // Reduce other moves if we have found at least one score improvement (~26 elo)
                    if (depth > 2 && depth < 12 && std::abs(value) < 2000)
                        depth -= ALPHA_IMPROVEMENT_REDUCTION;

                    // If we are in balance move mode, we also shrink beta as to narrow
                    // the search window to [-abs(v-bias)+bias, abs(v-bias)+bias].
                    if (RootNode && options.balanceMode) {
                        assert(alpha <= options.balanceBias);
                        beta = Value(2 * options.balanceBias) - alpha;
                        // Break if search window size reaches zero
                        if (alpha >= beta)
                            break;
                    }
                }
            }
        }

        if (RootNode && thisThread->isMainThread())
            searcher->printer.printMoveResult(*static_cast<MainSearchThread *>(thisThread),
                                              timectl,
                                              searchData->pvIdx,
                                              searchData->multiPv,
                                              searchData->rootDepth,
                                              move,
                                              value,
                                              bestMove == move);

        // Record good moves that improved alpha bound for move heuristic.
        histTracker.addSearchedMove(move, bestMove);
    }

    // Set singularRoot flag to true if we have only one singular move
    if (RootNode && nonMatedCount == 1 && (moveCount > 1 || thisThread->rootMoves.size() == 1))
        searchData->singularRoot = true;

    // Step 19. Check for mate
    // All legal moves have been searched and if there are no legal moves, it must be a mate
    // (in renju). If we are in a singular extension search then return a fail low score.
    if (!moveCount) {
        bestValue = skipMove ? alpha
                             : (board.p4Count(oppo, A_FIVE) ? mated_in(ss->ply + 2)
                                                            : mated_in(ss->ply + 4));

        // Do not return mate that longer than maxMoves option
        if (std::abs(bestValue) >= VALUE_MATE_IN_MAX_PLY) {
            if (board.nonPassMoveCount() + mate_step(bestValue, ss->ply) > options.maxMoves)
                bestValue = getDrawValue(board, options, ss->ply);
        }

        if (RootNode) {
            // All remaining losing root moves are marked with this value
            std::for_each(thisThread->rootMoves.begin() + searchData->pvIdx,
                          thisThread->rootMoves.end(),
                          [=](RootMove &rm) { rm.value = bestValue; });
        }
    }
    // If we have found a best move, update move heruistics
    else if (bestMove)
        histTracker.updateBestmoveStats(depth, bestMove, bestValue);

    // Step 20. Update database record
    Bound bound = bestValue >= beta ? BOUND_LOWER : PvNode && bestMove ? BOUND_EXACT : BOUND_UPPER;
    if (thisThread->dbClient
        && !Config::DatabaseReadonlyMode      // Never write in database readonly mode
        && !options.balanceMode               // Never write when we are doing balanced search
        && (!skipMove || ss->dbChildWritten)  // Never write when in singular extension
        && ss->numNullMoves == 0              // Never write when in null move search
        && !(RootNode && (searchData->pvIdx || options.blockMoves.size()))) {
        bool exact  = PvNode && bound == BOUND_EXACT;
        bool isWin  = bestValue > VALUE_MATE_IN_MAX_PLY && (bound & BOUND_LOWER);
        bool isLoss = bestValue < VALUE_MATED_IN_MAX_PLY && (bound & BOUND_UPPER);

        int writePly, writeDepth;
        if (isWin || isLoss) {
            if (ss->ply <= Config::DatabaseMateWritePly
                && mate_step(bestValue, ss->ply) >= Config::DatabaseMateWriteMinStep) {
                writePly   = Config::DatabaseMateWritePly;
                writeDepth = exact ? Config::DatabaseMateWriteMinDepthExact
                                   : Config::DatabaseMateWriteMinDepthNonExact;
            }
            else
                writePly = -1, writeDepth = MAX_DEPTH;
        }
        else {
            writePly   = exact ? Config::DatabasePVWritePly : Config::DatabaseNonPVWritePly;
            writeDepth = exact ? (std::abs(bestValue) <= Config::DatabaseWriteValueRange
                                      ? Config::DatabasePVWriteMinDepth
                                      : MAX_DEPTH)
                         : bound == BOUND_UPPER ? (bestValue <= Config::DatabaseWriteValueRange
                                                       ? Config::DatabaseNonPVWriteMinDepth
                                                       : MAX_DEPTH)
                                                : (-bestValue >= Config::DatabaseWriteValueRange
                                                       ? Config::DatabaseNonPVWriteMinDepth
                                                       : MAX_DEPTH);
        }

        if (RootNode
            || ss->dbChildWritten  // Write anyway if we have children that have already written
            || PvNode && ss->ply <= 1 + isLoss
                   && options.multiPV > 1   // Always add new record in multipv
            || ss->ply <= writePly - isWin  // Loss label are recorded one ply less
                   && (dbHit || depth >= writeDepth + isWin)
            || dbHit && (isWin || isLoss)  // Always try overwrite existing record with W/L record
            || PvNode && dbHit             // Try overwrite existing winrate with newer winrate
                   && ss->ply <= (exact ? Config::DatabaseExactOverwritePly
                                        : Config::DatabaseNonExactOverwritePly)) {
            // Assume we have already read record from the database
            Database::DBRecord newRecord;

            // Assign determined label if we have a sure win/loss
            // Do not inherit determined label from queried record,
            // only write determined label when we have searched it.
            newRecord.label = isWin    ? Database::LABEL_LOSE  // Loss for opponent
                              : isLoss ? Database::LABEL_WIN   // Win for opponent
                              : dbHit && !Database::isDeterminedLabel(dbRecord.label)
                                  ? dbRecord.label
                                  : Database::LABEL_NONE;
            newRecord.value = searchValueToStoredValue(bestValue, ss->ply);
            newRecord.setDepthBound((int)depth, bound);

            // Write if there is no db hit, or the new record satisfy the overwrite rule
            if (!dbHit
                || Database::checkOverwrite(dbRecord, newRecord, Config::DatabaseOverwriteRule)) {
                thisThread->dbClient->save(board,
                                           Rule,
                                           newRecord,
                                           dbHit ? Database::OverwriteRule::Always
                                                 : Config::DatabaseOverwriteRule);
                if (Config::DatabaseMandatoryParentWrite)
                    (ss - 1)->dbChildWritten = true;
            }
        }
    }

    // Adjust best value with Win/Loss record from database in PV node
    if (!RootNode && PvNode && dbHit) {
        if (dbLabelBound == BOUND_LOWER)
            bestValue = std::max(bestValue, dbValue);
        else if (dbLabelBound == BOUND_UPPER)
            bestValue = std::min(bestValue, dbValue);
        else if (int dbDepth = dbRecord.depth() + Config::DatabaseQueryResultDepthBoundBias;
                 (dbBound == BOUND_EXACT                          // Exact-cut
                  || dbBound == BOUND_LOWER && dbValue >= beta    // Beta-cut
                  || dbBound == BOUND_UPPER && dbValue <= alpha)  // Alpha-cut
                 && dbDepth > std::max((int)std::ceil(depth), ss->dbValueDepth)
                 && dbValue > bestValue - (dbDepth - (int)std::ceil(depth)) * 10)
            bestValue = dbValue, bound = dbBound, ss->dbValueDepth = dbDepth;
    }

    // Step 21. Save TT entry for this position
    // If no good move is found and the previous position was ttPv, then the previous
    // opponent move is probably good and the new position is added to the search tree.
    if (bestValue <= alpha)
        ss->ttPv = ss->ttPv || ((ss - 1)->ttPv && depth > 3);
    // Otherwise, a counter move has been found and if the position is the last leaf
    // in the search tree, remove the position from the search tree.
    else if (depth > 3)
        ss->ttPv = ss->ttPv && (ss + 1)->ttPv;

    // Don't save partial result in singular extension, multi pv at root or balance mode.
    if (!skipMove
        && !(RootNode && (searchData->pvIdx || options.balanceMode || options.blockMoves.size())))
        TT.store(posKey, bestValue, ss->staticEval, ss->ttPv, bound, bestMove, (int)depth, ss->ply);

    assert(bestValue > -VALUE_INFINITE && bestValue < VALUE_INFINITE);
    return bestValue;
}

/// The VCF search function only searches continuous VCF moves to avoid
/// search explosion. It returns the best evaluation in a VCF tree.
template <Rule Rule, NodeType NT>
Value vcfsearch(Board &board, SearchStack *ss, Value alpha, Value beta, Depth depth)
{
    constexpr bool PvNode = NT == PV || NT == Root;

    // Do some sanity check over input arguments
    assert(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);
    assert(PvNode || (alpha == beta - 1));
    assert(depth <= 0.0f);
    assert(0 <= ss->ply && ss->ply < MAX_PLY);

    // Step 1. Initialize node
    SearchThread *thisThread = board.thisThread();
    ABSearchData *searchData = thisThread->searchDataAs<ABSearchData>();
    thisThread->numNodes.fetch_add(1, std::memory_order_relaxed);

    Color self = board.sideToMove(), oppo = ~self;
    int   moveCount = 0;
    Value bestValue = -VALUE_INFINITE, value;
    Value oldAlpha  = alpha;  // Flag BOUND_EXACT when value above alpha in PVNode
    Pos   bestMove  = Pos::NONE;

    // Update selDepth (selDepth counts from 1, ply from 0)
    if (PvNode && thisThread->selDepth <= ss->ply)
        thisThread->selDepth = ss->ply + 1;

    // Step 2. Check for immediate draw and winning
    // Check if we reach the time limit
    if (thisThread->isMainThread())
        static_cast<MainSearchThread *>(thisThread)->checkExit();

    // Check if the board has been filled or we have reached the max game ply.
    if (board.movesLeft() == 0 || board.nonPassMoveCount() >= thisThread->options().maxMoves)
        return getDrawValue(board, thisThread->options(), ss->ply);

    // Check if we reached the max ply
    if (ss->ply >= MAX_PLY)
        return Evaluation::evaluate<Rule>(board, alpha, beta);

    // Check for immediate winning
    if ((value = quickWinCheck<Rule>(board, ss->ply, beta)) != VALUE_ZERO) {
        // Do not return mate that longer than maxMoves option
        if (board.nonPassMoveCount() + mate_step(value, ss->ply) > thisThread->options().maxMoves)
            value = getDrawValue(board, thisThread->options(), ss->ply);

        return value;
    }

    // Step 3. Mate distance pruning
    alpha = std::max(mated_in(ss->ply), alpha);
    beta  = std::min(mate_in(ss->ply + 1), beta);
    if (alpha >= beta)
        return alpha;

    // Step 4. Transposition table lookup
    HashKey posKey  = board.zobristKey();
    Value   ttValue = VALUE_NONE;
    Value   ttEval  = VALUE_NONE;
    bool    ttIsPv  = false;
    Bound   ttBound = BOUND_NONE;
    Pos     ttMove  = Pos::NONE;
    int     ttDepth = (int)DEPTH_LOWER_BOUND;
    bool    ttHit   = TT.probe(posKey, ttValue, ttEval, ttIsPv, ttBound, ttMove, ttDepth, ss->ply);

    // Check for an early TT cutoff (for all types of nodes)
    if (ttHit && ttDepth >= depth && (!PvNode || !thisThread->isMainThread())  // Show full PV
    ) {
        if (ttBound & BOUND_LOWER)
            alpha = std::max(alpha, ttValue);
        if (ttBound & BOUND_UPPER)
            beta = std::min(beta, ttValue);
        if (alpha >= beta)
            return ttValue;
    }

    // Step 5. Static position evaluation
    if (ttHit) {
        // Never assume anything about values stored in TT
        bestValue = ss->staticEval = ttEval;
        if (bestValue == VALUE_NONE)
            bestValue = ss->staticEval = Evaluation::evaluate<Rule>(board, alpha, beta);

        // Try to use ttValue as a better eval estimation
        if (ttValue != VALUE_NONE
            && (ttBound & (ttValue > ss->staticEval ? BOUND_LOWER : BOUND_UPPER)))
            bestValue = ttValue;
    }
    else {
        // In case of null move search use previous static eval with a different sign
        bestValue = ss->staticEval = (ss - 1)->currentMove == Pos::PASS
                                         ? -(ss - 1)->staticEval
                                         : Evaluation::evaluate<Rule>(board, alpha, beta);
    }

    // Stand pat. Return immediately if static value is at least beta
    if (bestValue >= beta) {
        // Save static evaluation into transposition table
        if (!ttHit)
            TT.store(posKey,
                     bestValue,
                     ss->staticEval,
                     false,
                     BOUND_LOWER,
                     Pos::NONE,
                     (int)DEPTH_NONE,
                     ss->ply);

        return bestValue;
    }
    // Keep improving alpha since we can stop anywhere in the move limited search.
    else if (PvNode && bestValue > alpha)
        alpha = bestValue;

    // Step 6. Delta pruning at non-PV node
    if (!PvNode && bestValue + qvcfDeltaMargin<Rule>(depth) < alpha) {
        // Save static evaluation into transposition table
        if (!ttHit)
            TT.store(posKey,
                     alpha,
                     ss->staticEval,
                     false,
                     BOUND_UPPER,
                     Pos::NONE,
                     (int)DEPTH_NONE,
                     ss->ply);

        return alpha;
    }

    // Step 7. Loop through the moves until no moves remain or a beta cutoff occurs.
    MovePicker mp(
        Rule,
        board,
        MovePicker::ExtraArgs<MovePicker::QVCF> {ttMove,
                                                 depth,
                                                 {(ss - 2)->moveP4[self], (ss - 4)->moveP4[self]}});

    while (Pos move = mp()) {
        assert(board.isLegal(move));

        ss->currentMove   = move;
        ss->moveCount     = ++moveCount;
        ss->moveP4[BLACK] = board.cell(move).pattern4[BLACK];
        ss->moveP4[WHITE] = board.cell(move).pattern4[WHITE];
        if (PvNode)
            (ss + 1)->pv[0] = Pos::NONE;

        // Step 8. Make and search the move
        board.move<Rule>(move);

        // Call defence-side vcf search
        value = -vcfdefend<Rule, NT>(board, ss + 1, -beta, -alpha, depth - 1);

        board.undo<Rule>();

        // Check if a stop occurred, we discard search result by returning none value
        if (thisThread->threads.isTerminating())
            return VALUE_NONE;

        // Step 9. Check for a new best move
        if (value > bestValue) {
            bestValue = value;

            if (value > alpha) {
                bestMove = move;

                if (PvNode)  // Update pv even in fail-high case
                    ss->updatePv(move);

                if (PvNode && value < beta)  // Update alpha
                    alpha = value;
                else  // Fail high
                    break;
            }
        }
    }

    // Step 10. Save TT entry for this position
    TT.store(posKey,
             bestValue,
             ss->staticEval,
             PvNode,
             bestValue >= beta                ? BOUND_LOWER
             : PvNode && bestValue > oldAlpha ? BOUND_EXACT
                                              : BOUND_UPPER,
             bestMove,
             (int)std::max(depth, DEPTH_QVCF),
             ss->ply);

    assert(bestValue > -VALUE_INFINITE && bestValue < VALUE_INFINITE);
    return bestValue;
}

/// The search function for defend node in VCF search.
template <Rule Rule, NodeType NT>
Value vcfdefend(Board &board, SearchStack *ss, Value alpha, Value beta, Depth depth)
{
    constexpr bool PvNode = NT == PV || NT == Root;

    // Do some sanity check over input arguments
    assert(-VALUE_INFINITE <= alpha && alpha < beta && beta <= VALUE_INFINITE);
    assert(PvNode || (alpha == beta - 1));
    assert(depth <= 0.0f);
    assert(0 <= ss->ply && ss->ply < MAX_PLY);

    // Step 1. Initialize node
    SearchThread *thisThread = board.thisThread();
    thisThread->numNodes.fetch_add(1, std::memory_order_relaxed);

    Color    self = board.sideToMove(), oppo = ~self;
    uint16_t oppo5 = board.p4Count(oppo, A_FIVE);  // opponent five
    Value    value;

    // Update selDepth (selDepth counts from 1, ply from 0)
    if (PvNode && thisThread->selDepth <= ss->ply)
        thisThread->selDepth = ss->ply + 1;

    // Step 2. Check for immediate evaluation, draw and winning
    // Return evaluation immediately if there is no vcf threat
    if (!oppo5)
        return Evaluation::evaluate<Rule>(board, alpha, beta);

    // Check if the board has been filled or we have reached the max game ply.
    if (board.movesLeft() == 0 || board.nonPassMoveCount() >= thisThread->options().maxMoves)
        return getDrawValue(board, thisThread->options(), ss->ply);

    // Check if we reached the max ply
    if (ss->ply >= MAX_PLY)
        return Evaluation::evaluate<Rule>(board, alpha, beta);

    // Step 3. Search the only defence move
    Pos move = board.stateInfo().lastPattern4(oppo, A_FIVE);
    assert(board.cell(move).pattern4[oppo] == A_FIVE);

    // For renju, if black's defence move is a forbidden point, black loses in two steps.
    if (Rule == Rule::RENJU && self == BLACK && board.checkForbiddenPoint(move)) {
        value = mated_in(ss->ply + 2);
    }
    else {
        ss->currentMove   = move;
        ss->moveCount     = 1;
        ss->moveP4[BLACK] = board.cell(move).pattern4[BLACK];
        ss->moveP4[WHITE] = board.cell(move).pattern4[WHITE];
        if (PvNode)
            (ss + 1)->pv[0] = Pos::NONE;

        board.move<Rule>(move);
        TT.prefetch(board.zobristKey());

        // Call attack-side vcf search
        // Note that we do not reduce depth for vcf defence move.
        value = -vcfsearch<Rule, NT>(board, ss + 1, -beta, -alpha, depth);

        board.undo<Rule>();

        // Update pv in PV node
        if (PvNode)
            ss->updatePv(move);
    }

    assert(value > -VALUE_INFINITE && value < VALUE_INFINITE
           || thisThread->threads.isTerminating());
    return value;
}

}  // namespace
