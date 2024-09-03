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

#include "../../core/iohelper.h"
#include "../../eval/eval.h"
#include "../../game/wincheck.h"
#include "../hashtable.h"
#include "../opening.h"
#include "../searchcommon.h"
#include "parameter.h"
#include "searcher.h"

#include <algorithm>

using namespace Search;
using namespace Search::MCTS;

namespace {

/// Compute the Cpuct exploration factor for the given parent node visits.
inline float cpuctExplorationFactor(uint32_t parentVisits)
{
    float cpuct = CpuctExploration;
    if (CpuctExplorationLog != 0)
        cpuct += CpuctExplorationLog * std::log(1.0f + parentVisits / CpuctExplorationBase);
    return cpuct * std::sqrt(parentVisits + CpuctParentVisitBias);
}

/// Compute the initial utility value for unexplored children, considering first play urgency.
template <bool Root>
inline float fpuValue(float parentAvgUtility, float parentRawUtility, float exploredPolicySum)
{
    const float fpuReductionMax = Root ? RootFpuReductionMax : FpuReductionMax;
    const float fpuLossRatio    = Root ? RootFpuLossProp : FpuLossProp;

    float blendWeight      = std::min(1.0f, std::pow(exploredPolicySum, FpuUtilityBlendPow));
    float parentUtilityFPU = blendWeight * parentAvgUtility + (1 - blendWeight) * parentRawUtility;
    float fpu              = parentUtilityFPU - fpuReductionMax * std::sqrt(exploredPolicySum);
    fpu -= (1 + fpu) * fpuLossRatio;
    return fpu;
}

/// Compute PUCT selection value with the given child statistics.
inline float puctSelectionValue(float    childUtility,
                                float    childPolicy,
                                uint32_t childVisits,
                                uint32_t childVirtualVisits,
                                float    cpuctExploration)
{
    float U = cpuctExploration * childPolicy / (1 + childVisits);
    float Q = childUtility;

    // Account for virtual losses
    if (childVirtualVisits > 0) {
        float    childUtilitySum = childUtility * childVisits;
        uint32_t childVisitsSum  = childVisits + childVirtualVisits;
        Q                        = (childUtilitySum - childVirtualVisits) / childVisitsSum;
    }

    return Q + U;
}

/// allocateOrFindNode: allocate a new node if it does not exist in the node table
/// @param nodeTable The node table to allocate or find the node
/// @param hash The hash key of the node
/// @param globalNodeAge The global node age to synchronize the node table
/// @return A pair of (the node pointer, whether the node is inserted by myself)
std::pair<Node *, bool>
allocateOrFindNode(NodeTable &nodeTable, HashKey hash, uint32_t globalNodeAge)
{
    // Try to find a transposition node with the board's zobrist hash
    Node *node         = nodeTable.findNode(hash);
    bool  didInsertion = false;

    // Allocate and insert a new child node if we do not find a transposition
    if (!node) {
        auto newNode                 = std::make_unique<Node>(hash, globalNodeAge);
        std::tie(node, didInsertion) = nodeTable.tryInsertNode(std::move(newNode));
    }

    return {node, didInsertion};
}

/// select: select the best child node according to the selection value
/// @param node The node to select child from, must be already expanded
/// @return A pair of (the non-null best child edge pointer, the child node pointer)
///   The child node pointer is nullptr if the edge is unexplored (has zero visit).
template <bool Root>
std::pair<Edge *, Node *> selectChild(Node &node, const Board &board)
{
    assert(!node.isLeaf());
    SearchThread *thisThread = board.thisThread();

    uint32_t parentVisits       = node.getVisits();
    float    cpuctExploration   = cpuctExplorationFactor(parentVisits);
    float    bestSelectionValue = -std::numeric_limits<float>::infinity();
    Edge    *bestEdge           = nullptr;
    Node    *bestNode           = nullptr;
    float    exploredPolicySum  = 0.0f;

    // Iterate through all expanded children to find the best selection value
    EdgeArray &edges = *node.getEdges();
    uint32_t   edgeIndex;
    for (edgeIndex = 0; edgeIndex < edges.numEdges; edgeIndex++) {
        Edge &childEdge = edges[edgeIndex];
        Pos   move      = childEdge.getMove();

        // Skip the edge if this move is not in the root move list
        if constexpr (Root) {
            auto rm = std::find(thisThread->rootMoves.begin(), thisThread->rootMoves.end(), move);
            if (rm == thisThread->rootMoves.end())
                continue;
        }

        // Get the child node of this edge
        Node *childNode = childEdge.getChild();
        // If this edge is not expanded, then the following edges must be unexpanded as well
        if (!childNode)
            break;

        // Accumulated explored policy sum
        float childPolicy = childEdge.getP();
        exploredPolicySum += childPolicy;

        // Compute selection value and update the best selection value
        uint32_t childVisits        = childEdge.getVisits();
        uint32_t childVirtualVisits = childNode->getVirtualVisits();
        float    childUtility       = -childNode->getQ();
        float    selectionValue     = puctSelectionValue(childUtility,
                                                  childPolicy,
                                                  childVisits,
                                                  childVirtualVisits,
                                                  cpuctExploration);
        if (selectionValue > bestSelectionValue) {
            bestSelectionValue = selectionValue;
            bestEdge           = &childEdge;
            bestNode           = childNode;
        }
    }

    // Compute selection value of the first unexplored child (which will have the highest
    // policy among the rest unexplored children)
    if (edgeIndex < edges.numEdges) {
        float fpuUtility = fpuValue<Root>(node.getQ(), node.getEvalUtility(), exploredPolicySum);

        Edge    &childEdge          = edges[edgeIndex];
        float    childPolicy        = childEdge.getP();
        uint32_t childVisits        = 0;  // Unexplored edge must has zero edge visit
        uint32_t childVirtualVisits = 0;  // Unexplored edge must has zero virtual visit
        float    selectionValue     = puctSelectionValue(fpuUtility,
                                                  childPolicy,
                                                  childVisits,
                                                  childVirtualVisits,
                                                  cpuctExploration);
        if (selectionValue > bestSelectionValue) {
            bestSelectionValue = selectionValue;
            bestEdge           = &childEdge;
            bestNode           = nullptr;  // No child node
        }
    }

    assert(bestEdge);
    return {bestEdge, bestNode};
}

/// evaluate: evaluate the value of this node and make the first visit
template <bool Root = false>
void evaluateNode(Node &node, const SearchOptions &options, const Board &board, int ply)
{
    SearchThread *thisThread = board.thisThread();

    if (!Root) {
        if (ply > thisThread->selDepth)
            thisThread->selDepth = ply;

        // Check if the board has been filled or we have reached the max game ply.
        if (board.movesLeft() == 0 || board.nonPassMoveCount() >= options.maxMoves) {
            Value value = getDrawValue(board, options, ply);
            node.setTerminal(value, ply);
            return;
        }

        // Check for immediate winning
        if (Value value = quickWinCheck(options.rule, board, ply); value != VALUE_ZERO) {
            // Do not return mate that longer than maxMoves option
            if (board.nonPassMoveCount() + mate_step(value, ply) > options.maxMoves)
                value = getDrawValue(board, options, ply);

            node.setTerminal(value, ply);
            return;
        }
    }

    // Evaluate value for new node that has not been visited
    Evaluation::ValueType v = Evaluation::computeEvaluatorValue(board);
    node.setNonTerminal(v.winLossRate(), v.draw());
}

/// expand: generate edges and evaluate the policy of this node
/// @return Whether this node has no valid move, which means this node is a terminal node.
template <bool Root = false>
bool expandNode(Node &node, const SearchOptions &options, const Board &board, int ply)
{
    if constexpr (Root) {
        MovePicker mp(options.rule, board, MovePicker::ExtraArgs<MovePicker::ROOT> {true});
        bool       noValidMove = node.createEdges(mp);
        assert(!node.isLeaf());
        assert(!noValidMove);
        return false;
    }
    else {
        MovePicker mp(options.rule,
                      board,
                      MovePicker::ExtraArgs<MovePicker::MAIN> {
                          Pos::NONE,
                          nullptr,
                          nullptr,
                      });

        bool noValidMove = node.createEdges(mp);
        if (noValidMove) {
            Value terminalValue =
                board.p4Count(~board.sideToMove(), A_FIVE) ? mated_in(ply + 2) : mated_in(ply + 4);

            // Do not return mate that longer than maxMoves option
            if (std::abs(terminalValue) >= VALUE_MATE_IN_MAX_PLY) {
                if (board.nonPassMoveCount() + mate_step(terminalValue, ply) > options.maxMoves)
                    terminalValue = getDrawValue(board, options, ply);
            }

            node.setTerminal(terminalValue, ply);
        }
        else {
            assert(!node.isLeaf());
        }
        return noValidMove;
    }
}

/// select and backpropagate: select the best child node and backpropagate the statistics
/// @param node The node to search, must been already allocated.
/// @param board The board state of this node. The board's hash must be equal to the node's.
/// @param ply The current search ply. Root node is zero.
/// @param visits The number of new visits for this playout.
/// @return The number of actual new visits added to this node.
template <bool Root = false>
uint32_t searchNode(Node &node, Board &board, int ply, uint32_t newVisits)
{
    assert(node.getHash() == board.zobristKey());

    SearchThread  *thisThread = board.thisThread();
    SearchOptions &options    = thisThread->options();
    MCTSSearcher  &searcher   = static_cast<MCTSSearcher &>(*thisThread->threads.searcher());

    // Discard visits in this node if it is unevaluated
    uint32_t parentVisits = node.getVisits();
    if (parentVisits == 0)
        return 0;

    if (Root)
        thisThread->selDepth = 0;

    // Cap new visits so that we dont do too much at one time
    newVisits = std::min(newVisits, uint32_t(parentVisits * MaxNewVisitsProp) + 1);

    // Return directly if this node is a terminal node and not at root
    if (!Root && node.isTerminal()) {
        node.incrementVisits(newVisits);
        return newVisits;
    }

    // Make sure the parent node is expanded before we select a child
    if (node.isLeaf()) {
        bool noValidMove = expandNode<Root>(node, options, board, ply);

        // If we found that there is no valid move, we mark this node as terminal
        // node the finish this visit.
        if (noValidMove) {
            node.incrementVisits(newVisits);
            return newVisits;
        }
    }

    bool     stopThisPlayout = false;
    uint32_t actualNewVisits = 0;
    while (!stopThisPlayout && newVisits > 0) {
        // Select the best edge to explore
        auto [childEdge, childNode] = selectChild<Root>(node, board);

        // Make the move to reach the child node
        Pos move = childEdge->getMove();
        board.move(options.rule, move);

        // Reaching a leaf node, expand it
        bool allocatedNode = false;
        if (!childNode) {
            HashKey hash = board.zobristKey();
            std::tie(childNode, allocatedNode) =
                allocateOrFindNode(*searcher.nodeTable, hash, searcher.globalNodeAge);

            // Remember this child node in the edge
            childEdge->setChild(childNode);
        }

        // Evaluate the new child node if we are the one who really allocated the node
        if (allocatedNode) {
            // Mark that we are now starting to visit this node
            node.beginVisit(1);
            evaluateNode(*childNode, options, board, ply + 1);

            // Increment child edge visit count
            childEdge->addVisits(1);
            node.updateStats();
            node.finishVisit(1, 1);
            actualNewVisits++;
            newVisits--;
        }
        else {
            // When transposition happens, we stop the playout if the child node has been
            // visited more times than the parent node. Only continue the playout if the
            // child node has been visited less times than the edge visits, or the absolute
            // child node visits is less than the given threshold.
            uint32_t childEdgeVisits = childEdge->getVisits();
            uint32_t childNodeVisits = childNode->getVisits();
            if (childEdgeVisits >= childNodeVisits
                || childNodeVisits < MinTranspositionSkipVisits) {
                node.beginVisit(newVisits);
                uint32_t actualChildNewVisits = searchNode(*childNode, board, ply + 1, newVisits);
                assert(actualChildNewVisits <= newVisits);

                if (actualChildNewVisits > 0) {
                    childEdge->addVisits(actualChildNewVisits);
                    node.updateStats();
                    actualNewVisits += actualChildNewVisits;
                }
                // Discard this playout if we can not make new visits to the best child,
                // since some other thread is evaluating it
                else
                    stopThisPlayout = true;

                node.finishVisit(newVisits, actualChildNewVisits);
                newVisits -= actualChildNewVisits;
            }
            else {
                // Increment edge visits without search the node
                childEdge->addVisits(1);
                node.updateStats();
                node.incrementVisits(1);
                actualNewVisits++;
                newVisits--;
            }
        }

        // Undo the move
        board.undo(options.rule);

        // Record root move's seldepth
        if constexpr (Root) {
            auto rmIt = std::find(thisThread->rootMoves.begin(), thisThread->rootMoves.end(), move);
            if (rmIt != thisThread->rootMoves.end()) {
                RootMove &rm = *rmIt;
                rm.selDepth  = std::max(rm.selDepth, thisThread->selDepth);
            }
        }
    }

    return actualNewVisits;
}

/// Select best move to play for the given node.
/// @param node The node to compute selection value. Must be expanded.
/// @param edgeIndices[out] The edge indices of selectable children.
/// @param selectionValues[out] The selection values of selectable children.
/// @param lcbValues[out] The lower confidence bound values of selectable children.
///     Only filled when we are using LCB for selection.
/// @param allowDirectPolicyMove If true and the node has no explored children,
///     allow to select from those unexplored children by policy prior directly.
/// @param forceUseLCB If non-zero, force to use LCB for selection. If the
///     value is positive, always use LCB, otherwise never use LCB.
/// @return The index of the best move (with highest selection value) to select.
///     Returns -1 if there is no selectable children.
int selectBestmoveOfChildNode(const Node            &node,
                              std::vector<uint32_t> &edgeIndices,
                              std::vector<float>    &selectionValues,
                              std::vector<float>    &lcbValues,
                              bool                   allowDirectPolicyMove,
                              int                    forceUseLCB)
{
    assert(!node.isLeaf());
    edgeIndices.clear();
    selectionValues.clear();
    lcbValues.clear();

    int   bestmoveIndex          = -1;
    float bestmoveSelectionValue = std::numeric_limits<float>::lowest();

    const EdgeArray &edges = *node.getEdges();
    for (uint32_t edgeIndex = 0; edgeIndex < edges.numEdges; edgeIndex++) {
        const Edge &childEdge = edges[edgeIndex];
        // Only select from expanded children in the first pass
        Node *childNode = childEdge.getChild();
        if (!childNode)
            continue;

        float childPolicy    = childEdge.getP();
        float selectionValue = 2.0f * childPolicy;

        assert(childNode->getVisits() > 0);
        uint32_t childVisits = childEdge.getVisits();
        // Skip zero visits children
        if (childVisits > 0) {
            // Discount the child visits by 1 and add small weight on raw policy
            float visitWeight = childVisits * float(childVisits - 1) / float(childVisits);
            selectionValue += visitWeight;
        }

        // Find the best edge with the highest selection value
        if (selectionValue > bestmoveSelectionValue) {
            bestmoveIndex          = edgeIndices.size();
            bestmoveSelectionValue = selectionValue;
        }
        edgeIndices.push_back(edgeIndex);
        selectionValues.push_back(selectionValue);
    }

    // Compute lower confidence bound values if needed
    if (forceUseLCB >= 0 && (forceUseLCB > 0 || UseLCBForBestmoveSelection)) {
        uint32_t bestLCBEdgeIndex = 0;
        float    bestLCBValue     = std::numeric_limits<float>::lowest();

        for (size_t i = 0; i < edgeIndices.size(); i++) {
            uint32_t    edgeIndex = edgeIndices[i];
            const Edge &childEdge = edges[edgeIndex];
            Node       *childNode = childEdge.getChild();
            assert(childNode);

            // TODO: add LCB computation here
            lcbValues.push_back(0.0f);

            if (selectionValues[i] > 0
                && selectionValues[i] >= MinVisitPropForLCB * bestmoveSelectionValue
                && lcbValues[i] > bestLCBValue) {
                bestLCBEdgeIndex = edgeIndex;
                bestLCBValue     = lcbValues[i];
            }
        }
    }

    // If we have no expanded children for selection, try select by raw policy if allowed
    if (edgeIndices.empty() && allowDirectPolicyMove) {
        for (uint32_t edgeIndex = 0; edgeIndex < edges.numEdges; edgeIndex++) {
            const Edge &childEdge   = edges[edgeIndex];
            float       childPolicy = childEdge.getP();
            if (childPolicy > bestmoveSelectionValue) {
                bestmoveIndex          = edgeIndices.size();
                bestmoveSelectionValue = childPolicy;
            }
            edgeIndices.push_back(edgeIndex);
            selectionValues.push_back(childPolicy);
        }
        assert(!edgeIndices.empty());
    }

    return bestmoveIndex;
}

/// Extract PV of the given node recursively.
/// @param node The node to extract PV.
/// @param pv[out] The extracted PV will be appended to this array.
/// @param maxDepth Only extract PV within this depth.
void extractPVOfChildNode(const Node &node, std::vector<Pos> &pv, int maxDepth = 100)
{
    const Node           *curNode = &node;
    std::vector<uint32_t> tempEdgeIndices;
    std::vector<float>    tempSelectionValues, tempLCBValues;
    for (int depth = 0; depth < maxDepth; depth++) {
        if (curNode->isLeaf())
            break;

        int bestmoveIndex = selectBestmoveOfChildNode(*curNode,
                                                      tempEdgeIndices,
                                                      tempSelectionValues,
                                                      tempLCBValues,
                                                      true,
                                                      0);
        if (bestmoveIndex < 0)
            break;

        uint32_t         bestEdgeIndex = tempEdgeIndices[bestmoveIndex];
        const EdgeArray &edges         = *curNode->getEdges();
        const Edge      &bestEdge      = edges[bestEdgeIndex];
        Pos              bestmove      = bestEdge.getMove();
        pv.push_back(bestmove);

        curNode = bestEdge.getChild();
        if (!curNode)
            break;
    }
}

}  // namespace

MCTSSearcher::MCTSSearcher()
{
    root          = nullptr;
    nodeTable     = nullptr;
    globalNodeAge = 0;
}

void MCTSSearcher::setMemoryLimit(size_t memorySizeKB)
{
    TT.resize(1);
}

size_t MCTSSearcher::getMemoryLimit() const
{
    return TT.hashSizeKB();
}

void MCTSSearcher::clear(ThreadPool &pool, bool clearAllMemory)
{
    root = nullptr;
    if (clearAllMemory) {
        nodeTable     = std::make_unique<NodeTable>(Config::NumNodeTableShardsPowerOfTwo);
        globalNodeAge = 0;
    }
}

void MCTSSearcher::searchMain(MainSearchThread &th)
{
    SearchOptions &opts  = th.options();
    Board         &board = *th.board;

    // Probe opening database and find if there is a prepared opening
    if (!opts.disableOpeningQuery
        && Opening::probeOpening(board, opts.rule, th.resultAction, th.bestMove)) {
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
            printer.printBestmoveWithoutSearch(pos, mated_in(0), 0, nullptr);
            return;
        }

        return;  // abnormal case: GUI might have a bug
    }
    // If we are winning, return directly
    else if (th.board->p4Count(th.board->sideToMove(), A_FIVE)) {
        assert(th.board->cell(th.rootMoves[0].pv[0]).pattern4[th.board->sideToMove()] == A_FIVE);
        th.rootMoves[0].value = mate_in(1);
        th.bestMove           = th.rootMoves[0].pv[0];
        return;
    }

    // Evaluator must be enabled for MCTS search
    if (!board.evaluator()) {
        FOR_EVERY_EMPTY_POS(th.board, pos)
        {
            th.bestMove     = pos;
            th.resultAction = ActionType::Move;
            ERRORL("Evaluator is not enabled, cannot use mcts search.");
            printer.printBestmoveWithoutSearch(pos, mated_in(0), 0, nullptr);
            return;
        }
    }

    // Init time management and transposition table
    timectl.init(opts.turnTime, opts.matchTime, opts.timeLeft, {board.ply(), board.movesLeft()});

    // Starts worker threads, then starts main thread
    printer.printSearchStarts(th, timectl);

    setupRootNode(th);       // Setup root node and other stuffs
    th.startOtherThreads();  // Starts non-main threads
    search(th);              // Starts main thread searching

    // Stop all threads if not already stopped and wait for all threads to stop
    th.threads.stopThinking();
    th.threads.waitForIdle(false);

    // Rank root moves and record best move
    updateRootMovesData(th);
    printer.printRootMoves(th, timectl, numSelectableRootMoves);

    // Do not record bestmove in pondering
    if (th.inPonder)
        return;
    th.bestMove = th.rootMoves[0].pv[0];

    // If swap check is needed, make swap decision according to the rule
    if (opts.swapable)
        th.resultAction = Opening::decideAction(*th.board, opts.rule, th.rootMoves[0].value);
    else if (opts.balanceMode == SearchOptions::BalanceMode::BALANCE_TWO)
        th.resultAction = ActionType::Move2;
    else
        th.resultAction = ActionType::Move;
}

void MCTSSearcher::search(SearchThread &th)
{
    SearchOptions &options = th.options();
    Board         &board   = *th.board;
    assert(!root->isLeaf());

    // Main search loop
    std::vector<Node *> selectedPath;
    while (!th.threads.isTerminating()) {
        uint32_t newNumPlayouts = MaxNumVisitsPerPlayout;

        // Cap new number of playouts to the maximum num nodes to visit
        if (options.maxNodes) {
            uint64_t nodesSearched = th.threads.nodesSearched();
            if (nodesSearched >= options.maxNodes)
                break;

            uint64_t maxNodesToVisit = options.maxNodes - nodesSearched;
            if (maxNodesToVisit < newNumPlayouts)
                newNumPlayouts = maxNodesToVisit;
        }

        uint32_t newNumNodes = searchNode<true>(*root, board, 0, newNumPlayouts);
        th.numNodes.fetch_add(newNumNodes, std::memory_order_relaxed);

        if (th.isMainThread()) {
            MainSearchThread &mainThread = static_cast<MainSearchThread &>(th);
            mainThread.checkExit(std::max(newNumNodes, 1u));

            bool printRootMoves = false;
            if (Config::NodesToPrintMCTSRootmoves > 0) {
                uint64_t currentNumNodes = th.threads.nodesSearched();
                uint64_t numElapsedNodes = currentNumNodes - lastOutputNodes;

                if (numElapsedNodes >= Config::NodesToPrintMCTSRootmoves) {
                    lastOutputNodes = currentNumNodes;
                    printRootMoves  = true;
                }
            }
            if (Config::TimeToPrintMCTSRootmoves > 0) {
                Time currentTime = now();
                Time elapsedTime = currentTime - lastOutputTime;
                if (elapsedTime >= Config::TimeToPrintMCTSRootmoves) {
                    lastOutputTime = currentTime;
                    printRootMoves = true;
                }
            }

            if (printRootMoves) {
                updateRootMovesData(mainThread);
                printer.printRootMoves(mainThread, timectl, numSelectableRootMoves);
            }

            if ((numSelectableRootMoves == 1 || th.rootMoves.size() == 1)
                && th.threads.nodesSearched() >= Config::NumNodesAfterSingularRoot)
                th.threads.stopThinking();
        }
    }
}

bool MCTSSearcher::checkTimeupCondition()
{
    if (timectl.elapsed() >= timectl.maximum())
        return true;
    if (timectl.checkStop(TimeControl::PlayoutParams {}))
        return true;
    return false;
}

void MCTSSearcher::setupRootNode(MainSearchThread &th)
{
    // Clear the searcher if we have not initialized yet
    if (!nodeTable)
        clear(th.threads, true);

    // Initialize search data
    lastOutputNodes        = 0;
    lastOutputTime         = now();
    numSelectableRootMoves = 0;

    // Get the current root position
    std::vector<Pos> rootPosition;
    for (int moveIndex = 0; moveIndex < th.board->ply(); moveIndex++) {
        Pos move = th.board->getHistoryMove(moveIndex);
        rootPosition.push_back(move);
    }

    // If the root position has not changed, we do not need to update the root node
    if (root && rootPosition == previousPosition)
        return;

    SearchOptions &opts = th.options();

    // Increment global node age
    globalNodeAge += 1;

    // Initialize the root node to expanded state
    std::tie(root, std::ignore) =
        allocateOrFindNode(*nodeTable, th.board->zobristKey(), globalNodeAge);
    if (root->getVisits() == 0)
        evaluateNode<true>(*root, opts, *th.board, 0);
    if (root->isLeaf())
        expandNode<true>(*root, opts, *th.board, 0);
    assert(root->getEdges()->numEdges > 0);

    // Update previous Position
    previousPosition = std::move(rootPosition);
}

void MCTSSearcher::updateRootMovesData(MainSearchThread &th)
{
    assert(root != nullptr);
    assert(!root->isLeaf());

    std::vector<uint32_t> edgeIndices;
    std::vector<float>    selectionValues, lcbValues;
    int                   bestChildIndex =
        selectBestmoveOfChildNode(*root, edgeIndices, selectionValues, lcbValues, true, 0);
    uint32_t maxNumRootMovesToPrint =
        std::max<uint32_t>(th.options().multiPV, Config::MaxNonPVRootmovesToPrint);
    numSelectableRootMoves = std::min<uint32_t>(edgeIndices.size(), maxNumRootMovesToPrint);

    for (RootMove &rm : th.rootMoves) {
        rm.selectionValue = std::numeric_limits<float>::lowest();
        rm.previousValue  = rm.value;
        rm.previousPv     = rm.pv;
        rm.pv.resize(1);
    }

    EdgeArray &edges = *root->getEdges();
    for (size_t i = 0; i < edgeIndices.size(); i++) {
        uint32_t edgeIndex = edgeIndices[i];
        Edge    &childEdge = edges[edgeIndex];
        Pos      move      = childEdge.getMove();

        // Skip edges that are not in the root move list
        auto rm = std::find(th.rootMoves.begin(), th.rootMoves.end(), move);
        if (rm == th.rootMoves.end())
            continue;

        // Record the root move's value for explored children
        float childPolicy = childEdge.getP();
        Node *childNode   = childEdge.getChild();
        if (childNode) {
            if (childNode->getVisits() > 0) {
                float childUtility = -childNode->getQ();

                rm->winRate  = childUtility * 0.5f + 0.5f;
                rm->drawRate = childNode ? childNode->getD() : 0.0f;
                rm->value    = Config::winRateToValue(rm->winRate);
            }
            rm->policyPrior = childPolicy;
            rm->numNodes    = childEdge.getVisits();
            extractPVOfChildNode(*childNode, rm->pv);
        }
        else {
            rm->policyPrior = childPolicy;
            rm->numNodes    = 0;
        }
        rm->selectionValue = selectionValues[i];
    }

    // Sort the root moves in descending order by selection value
    std::stable_sort(th.rootMoves.begin(),
                     th.rootMoves.end(),
                     [](const RootMove &m1, const RootMove &m2) {
                         return m1.selectionValue > m2.selectionValue;
                     });
}
