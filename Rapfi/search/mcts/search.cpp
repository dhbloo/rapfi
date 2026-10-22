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
#include "../../eval/evaluator.h"
#include "../../game/wincheck.h"
#include "../hashtable.h"
#include "../opening.h"
#include "../searchcommon.h"
#include "../searchconfig.h"
#include "parameter.h"
#include "searcher.h"

#include <algorithm>
#include <limits>

using namespace Search;
using namespace Search::MCTS;

namespace SimpleVCF {

struct SearchStack
{
    Pattern4 moveP4[2];
};

// forward declaration
template <Rule Rule>
Value vcfsearch(Board &board, SearchStack *ss, int ply, Value alpha, Value beta, Depth depth);
template <Rule Rule>
Value vcfdefend(Board &board, SearchStack *ss, int ply, Value alpha, Value beta, Depth depth);

Value vcf(Rule rule, Board &board, int ply)
{
    Value alpha = -VALUE_EVAL_MAX;
    Value beta  = VALUE_EVAL_MAX;
    Depth depth = 0;

    SearchStack  stack[MAX_MOVES + 4];
    SearchStack *ss = stack - ply + 4;
    for (int i : {1, 2, 3, 4}) {
        ss[ply - i].moveP4[BLACK] = NONE;
        ss[ply - i].moveP4[WHITE] = NONE;
    }

    if (board.p4Count(~board.sideToMove(), A_FIVE)) {
        switch (rule) {
        case FREESTYLE: return vcfdefend<FREESTYLE>(board, ss, ply, alpha, beta, depth);
        case STANDARD: return vcfdefend<STANDARD>(board, ss, ply, alpha, beta, depth);
        case RENJU: return vcfdefend<RENJU>(board, ss, ply, alpha, beta, depth);
        default: break;
        }
    }
    else {
        switch (rule) {
        case FREESTYLE: return vcfsearch<FREESTYLE>(board, ss, ply, alpha, beta, depth);
        case STANDARD: return vcfsearch<STANDARD>(board, ss, ply, alpha, beta, depth);
        case RENJU: return vcfsearch<RENJU>(board, ss, ply, alpha, beta, depth);
        default: break;
        }
    }

    return VALUE_ZERO;
}

template <Rule Rule>
Value vcfsearch(Board &board, SearchStack *ss, int ply, Value alpha, Value beta, Depth depth)
{
    Color                self = board.sideToMove();
    SearchThread        *thisThread = board.thisThread();
    const SearchOptions &options    = thisThread->options();
    Value                bestValue  = -VALUE_INFINITE;
    Value                oldAlpha   = alpha;  // Flag BOUND_EXACT when value above alpha in PVNode
    Pos                  bestMove   = Pos::NONE;
    if (ply > thisThread->selDepth)
        thisThread->selDepth = ply;

    // Check if the board has been filled or we have reached the max game ply.
    if (board.movesLeft() == 0 || board.nonPassMoveCount() >= options.maxMoves)
        return getDrawValue(board, options, board.ply());

    // Check for immediate winning
    if (Value value = quickWinCheck(options.rule, board, board.ply()); value != VALUE_ZERO) {
        // Do not return mate that longer than maxMoves option
        if (mate_step(value, 0) > options.maxMoves)
            value = getDrawValue(board, options, board.ply());

        return value;
    }

    // Mate distance pruning
    alpha = std::max(mated_in(board.ply()), alpha);
    beta  = std::min(mate_in(board.ply() + 1), beta);
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
    bool    ttHit   = TT.probe(posKey, ttValue, ttEval, ttIsPv, ttBound, ttMove, ttDepth, 0);

    // Check for an early TT cutoff (for all types of nodes)
    if (ttHit && ttDepth >= depth) {
        if (ttBound & BOUND_LOWER)
            alpha = std::max(alpha, ttValue);
        if (ttBound & BOUND_UPPER)
            beta = std::min(beta, ttValue);
        if (alpha >= beta)
            return ttValue;
    }

    // Stand pat for the situation that we do not find a mate
    bestValue = VALUE_ZERO;
    if (bestValue > alpha)
        alpha = bestValue;

    MovePicker mp(Rule,
                  board,
                  MovePicker::ExtraArgs<MovePicker::QVCF> {
                      Pos::NONE,
                      depth,
                      {ss[ply - 2].moveP4[self], ss[ply - 4].moveP4[self]}});

    while (Pos move = mp()) {
        assert(board.isLegal(move));
        ss[ply].moveP4[BLACK] = board.pattern4(move, BLACK);
        ss[ply].moveP4[WHITE] = board.pattern4(move, WHITE);

        // Make and search the move
        board.move<Rule, Board::MoveType::NO_EVAL>(move);

        // Call defence-side vcf search
        Value value = -vcfdefend<Rule>(board, ss, ply + 1, -beta, -alpha, depth - 1);

        board.undo<Rule, Board::MoveType::NO_EVAL>();

        // Check for a new best move
        if (value > bestValue) {
            bestValue = value;

            if (value > alpha) {
                bestMove = move;

                if (value < beta)  // Update alpha
                    alpha = value;
                else  // Fail high
                    break;
            }
        }
    }

    // Save TT entry for this position
    TT.store(posKey,
             bestValue,
             VALUE_NONE,
             true,
             bestValue >= beta      ? BOUND_LOWER
             : bestValue > oldAlpha ? BOUND_EXACT
                                    : BOUND_UPPER,
             bestMove,
             (int)std::max(depth, DEPTH_QVCF),
             0);

    assert(bestValue > -VALUE_INFINITE && bestValue < VALUE_INFINITE);
    return bestValue;
}

template <Rule Rule>
Value vcfdefend(Board &board, SearchStack *ss, int ply, Value alpha, Value beta, Depth depth)
{
    Color                self = board.sideToMove(), oppo = ~self;
    SearchThread        *thisThread = board.thisThread();
    const SearchOptions &options    = thisThread->options();
    if (ply > thisThread->selDepth)
        thisThread->selDepth = ply;

    // Return evaluation immediately if there is no vcf threat
    if (!board.p4Count(oppo, A_FIVE))
        return VALUE_ZERO;

    // Check if the board has been filled or we have reached the max game ply.
    if (board.movesLeft() == 0 || board.nonPassMoveCount() >= options.maxMoves)
        return getDrawValue(board, options, board.ply());

    // Search the only defence move
    Pos move = board.stateInfo().lastPattern4(oppo, A_FIVE);
    assert(board.pattern4(move, oppo) == A_FIVE);

    Value value;
    if (options.rule == Rule::RENJU && self == BLACK && board.checkForbiddenPoint(move)) {
        // For renju, if black's defence move is a forbidden point, black loses in two steps.
        value = mated_in(board.ply() + 2);
    }
    else {
        ss[ply].moveP4[BLACK] = board.pattern4(move, BLACK);
        ss[ply].moveP4[WHITE] = board.pattern4(move, WHITE);

        TT.prefetch(board.zobristKeyAfter(move));  // pre-make, overlaps the board update
        board.move<Rule, Board::MoveType::NO_EVAL>(move);

        // Call attack-side vcf search
        // Note that we do not reduce depth for vcf defence move.
        value = -vcfsearch<Rule>(board, ss, ply + 1, -beta, -alpha, depth);

        board.undo<Rule, Board::MoveType::NO_EVAL>();
    }

    assert(value > -VALUE_INFINITE && value < VALUE_INFINITE);
    return value;
}

}  // namespace SimpleVCF

namespace {

/// Compute the Cpuct exploration factor for the given parent node visits.
inline float cpuctExplorationFactor(uint32_t parentVisits)
{
    float cpuct = CpuctExploration;
    if (CpuctExplorationLog != 0)
        cpuct += CpuctExplorationLog * std::log(1.0f + parentVisits / CpuctExplorationBase);
    return cpuct * std::sqrt(parentVisits + 1e-2f);
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
                                float    childDraw,
                                float    parentDraw,
                                float    childPolicy,
                                uint32_t childVisits,
                                uint32_t childVirtualVisits,
                                float    cpuctExploration)
{
    float U = cpuctExploration * childPolicy / (1 + childVisits);
    float Q = childUtility;

    // Reduce utility value for drawish child nodes for PUCT selection
    // Encourage exploration for less drawish child nodes
    if (SearchCfg.drawUtilityPenalty != 0)
        Q -= SearchCfg.drawUtilityPenalty * childDraw * (1 - parentDraw);

    // Account for virtual losses
    if (childVirtualVisits > 0)
        Q = (Q * childVisits - childVirtualVisits) / (childVisits + childVirtualVisits);

    return Q + U;
}

/// allocateOrFindNode: allocate a new node if it does not exist in the node table
/// @param searcher The searcher owning the node table and the memory counter
/// @param hash The hash key of the node
/// @return A pair of (the node pointer, whether the node is inserted by myself)
std::pair<Node *, bool> allocateOrFindNode(MCTSSearcher &searcher, HashKey hash)
{
    // Try to find a transposition node with the board's zobrist hash
    Node *node         = searcher.nodeTable->findNode(hash);
    bool  didInsertion = false;

    // Allocate and insert a new child node if we do not find a transposition
    if (!node) {
        std::tie(node, didInsertion) =
            searcher.nodeTable->tryEmplaceNode(hash, searcher.graph.globalNodeAge);
        if (didInsertion)
            searcher.graphMemoryUsed.fetch_add(nodeFootprint(), std::memory_order_relaxed);
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

    uint32_t parentVisits     = node.getVisits();
    float    parentDraw       = node.getD();
    float    cpuctExploration = cpuctExplorationFactor(parentVisits);

    // Apply dynamic cpuct scaling based on parent utility variance if needed
    if (CpuctUtilityStdevScale > 0) {
        float parentUtilityVar = node.getQVar(CpuctUtilityVarPrior, CpuctUtilityVarPriorWeight);
        float parentUtilityStdevProp = std::sqrt(parentUtilityVar / CpuctUtilityVarPrior);
        float parentUtilityStdevFactor =
            1.0f + CpuctUtilityStdevScale * (parentUtilityStdevProp - 1.0f);
        cpuctExploration *= parentUtilityStdevFactor;
    }

    float bestSelectionValue = -std::numeric_limits<float>::infinity();
    Edge *bestEdge           = nullptr;
    Node *bestNode           = nullptr;
    float exploredPolicySum  = 0.0f;

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
        float    childDraw          = childNode->getD();
        float    selectionValue     = puctSelectionValue(childUtility,
                                                  childDraw,
                                                  parentDraw,
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
                                                  parentDraw,
                                                  parentDraw,
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

/// expand: generate edges and evaluate the policy of this node
/// @return Whether this node has no valid move, which means this node is a terminal node.
template <bool Root = false>
bool expandNode(Node &node, const SearchOptions &options, const Board &board, int ply)
{
    MCTSSearcher &searcher = static_cast<MCTSSearcher &>(*board.thisThread()->engine.searcher());

    if constexpr (Root) {
        MovePicker mp(options.rule,
                      board,
                      MovePicker::ExtraArgs<MovePicker::ROOT> {
                          true,
                          RootPolicyTemperature,
                      });
        bool       noValidMove = node.createEdges(mp, &searcher.graphMemoryUsed);
        assert(!node.isLeaf());
        assert(!noValidMove);
        return false;
    }
    else {
        MovePicker mp(options.rule,
                      board,
                      MovePicker::ExtraArgs<MovePicker::MAIN> {
                          Pos::NONE,
                          {},
                          true,
                          PolicyTemperature,
                      });

        bool noValidMove = node.createEdges(mp, &searcher.graphMemoryUsed);
        if (noValidMove) {
            Value terminalValue = board.p4Count(~board.sideToMove(), A_FIVE)
                                      ? mated_in(board.ply() + 2)
                                      : mated_in(board.ply() + 4);

            // Do not return mate that longer than maxMoves option
            if (std::abs(terminalValue) >= VALUE_MATE_IN_MAX_PLY) {
                if (mate_step(terminalValue, 0) > options.maxMoves)
                    terminalValue = getDrawValue(board, options, board.ply());
            }

            node.setTerminal(terminalValue);
        }
        else {
            assert(!node.isLeaf());
        }
        return noValidMove;
    }
}

/// evaluate: evaluate the value of this node and make the first visit
template <bool Root = false>
void evaluateNode(Node &node, const SearchOptions &options, Board &board, int ply)
{
    SearchThread *thisThread = board.thisThread();

    if (!Root) {
        if (ply > thisThread->selDepth)
            thisThread->selDepth = ply;

        // Check if the board has been filled or we have reached the max game ply.
        if (board.movesLeft() == 0 || board.nonPassMoveCount() >= options.maxMoves) {
            Value value = getDrawValue(board, options, board.ply());
            node.setTerminal(value);
            return;
        }

        // Check for immediate winning
        if (Value value = quickWinCheck(options.rule, board, board.ply()); value != VALUE_ZERO) {
            // Do not return mate that longer than maxMoves option
            if (mate_step(value, 0) > options.maxMoves)
                value = getDrawValue(board, options, board.ply());

            node.setTerminal(value);
            return;
        }

        // Search VCF
        if (Value value = SimpleVCF::vcf(options.rule, board, ply); value != VALUE_ZERO) {
            node.setTerminal(value);
            return;
        }
    }

    // Evaluate value for new node that has not been visited
    Evaluation::ValueType v = Evaluation::computeEvaluatorValue(board);
    node.setNonTerminal(v.winLossRate(), v.draw());

    // If ExpandWhenFirstEvaluate mode is enabled, we expand the node immediately
    if (SearchCfg.expandWhenFirstEvaluate)
        expandNode<Root>(node, options, board, ply);
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
    MCTSSearcher  &searcher   = static_cast<MCTSSearcher &>(*thisThread->engine.searcher());

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
            std::tie(childNode, allocatedNode) = allocateOrFindNode(searcher, hash);

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

/// Bestmove-selection pass 1: fill edgeIndices/selectionValues from all
/// expanded children of the node, accumulating their bound stats.
/// @return Tuple of (best index, best selection value, accumulated max bound).
std::tuple<int, float, ValueBound> computeSelectionValues(const Node            &node,
                                                          std::vector<uint32_t> &edgeIndices,
                                                          std::vector<float>    &selectionValues)
{
    int        bestmoveIndex          = -1;
    float      bestmoveSelectionValue = std::numeric_limits<float>::lowest();
    ValueBound maxBound {-VALUE_INFINITE};

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

        // Update bound stats
        maxBound |= childNode->getBound();
    }

    return {bestmoveIndex, bestmoveSelectionValue, maxBound};
}

/// Bestmove-selection pass 2 (UseLCBForBestmoveSelection): fill lcbValues and
/// give the best-LCB child a selection value bonus.
/// @return The updated best index.
int applyLCBSelection(const Node                  &node,
                      const std::vector<uint32_t> &edgeIndices,
                      std::vector<float>          &selectionValues,
                      std::vector<float>          &lcbValues,
                      int                          bestmoveIndex,
                      float                        bestmoveSelectionValue)
{
    const EdgeArray &edges = *node.getEdges();

    int   bestLCBIndex = -1;
    float bestLCBValue = std::numeric_limits<float>::lowest();

    std::vector<float> lcbRadius;

    // Compute LCB values for all selectable children and find highest LCB value
    for (size_t i = 0; i < edgeIndices.size(); i++) {
        uint32_t    edgeIndex = edgeIndices[i];
        const Edge &childEdge = edges[edgeIndex];
        Node       *childNode = childEdge.getChild();
        assert(childNode);

        float utilityMean = -childNode->getQ();

        // Only compute LCB for children with enough visits
        uint32_t childVisits = childNode->getVisits();
        if (childVisits < 2) {
            lcbValues.push_back(utilityMean - 1e5f);
            lcbRadius.push_back(1e4f);
        }
        else {
            float utilityVar = childNode->getQVar();
            float radius     = LCBStdevs * std::sqrt(utilityVar / childVisits);
            lcbValues.push_back(utilityMean - radius);
            lcbRadius.push_back(radius);

            if (selectionValues[i] > 0
                && selectionValues[i] >= LCBMinVisitProp * bestmoveSelectionValue
                && lcbValues[i] > bestLCBValue) {
                bestLCBIndex = i;
                bestLCBValue = lcbValues[i];
            }
        }
    }

    // Best LCB child gets a bonus on selection value
    if (bestLCBIndex >= 0) {
        float bestLCBSelectionValue = selectionValues[bestLCBIndex];
        for (size_t i = 0; i < edgeIndices.size(); i++) {
            if (i == bestLCBIndex)
                continue;

            // Compute how much the best LCB value is better than the current child
            float lcbBonus = bestLCBValue - lcbValues[i];
            if (lcbBonus <= 0)
                continue;

            // Compute how many times larger the radius can be before this LCB value is better
            float gain   = std::min(lcbBonus / lcbRadius[i] + 1.0f, 5.0f);
            float lbound = gain * gain * selectionValues[i];
            if (lbound > bestLCBSelectionValue)
                bestLCBSelectionValue = lbound;
        }
        selectionValues[bestLCBIndex] = bestLCBSelectionValue;
        bestmoveIndex                 = bestLCBIndex;
    }

    return bestmoveIndex;
}

/// Bestmove-selection pass 3 (a proven mate exists below): reweight children to
/// prefer the shortest proven mate.
/// @param maxBound The bound accumulated by computeSelectionValues.
/// @return The updated best index.
int overrideBestByMateBounds(const Node                  &node,
                             ValueBound                   maxBound,
                             const std::vector<uint32_t> &edgeIndices,
                             std::vector<float>          &selectionValues)
{
    const EdgeArray &edges = *node.getEdges();

    int   bestmoveIndex          = -1;
    float bestmoveSelectionValue = std::numeric_limits<float>::lowest();

    // Make best move the one with the maximum lower bound
    for (size_t i = 0; i < edgeIndices.size(); i++) {
        uint32_t    edgeIndex = edgeIndices[i];
        const Edge &childEdge = edges[edgeIndex];
        Node       *childNode = childEdge.getChild();
        assert(childNode);

        Value childLowerBound = static_cast<Value>(-childNode->getBound().upper);
        if (childLowerBound < VALUE_MATE_IN_MAX_PLY)  // Downweight non-proven mate moves
            selectionValues[i] *= 1e-9f;
        else if (childLowerBound < maxBound.lower)  // Downweight non-shorted mate moves
            selectionValues[i] *= 1e-3f * (1 + maxBound.lower - childLowerBound);

        // Find the best edge with the highest selection value
        if (selectionValues[i] > bestmoveSelectionValue) {
            bestmoveIndex          = i;
            bestmoveSelectionValue = selectionValues[i];
        }
    }

    return bestmoveIndex;
}

/// Bestmove-selection pass 4 (no expanded children): select from unexplored
/// children by their raw policy prior.
/// @return The updated best index.
int selectByRawPolicy(const Node            &node,
                      std::vector<uint32_t> &edgeIndices,
                      std::vector<float>    &selectionValues)
{
    const EdgeArray &edges = *node.getEdges();

    int   bestmoveIndex          = -1;
    float bestmoveSelectionValue = std::numeric_limits<float>::lowest();

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

    return bestmoveIndex;
}

/// Select best move to play for the given node.
/// @param node The node to compute selection value. Must be expanded.
/// @param edgeIndices[out] The edge indices of selectable children.
/// @param selectionValues[out] The selection values of selectable children.
/// @param lcbValues[out] The lower confidence bound values of selectable children.
///     Only filled when we are using LCB for selection.
/// @param allowDirectPolicyMove If true and the node has no explored children,
///     allow to select from those unexplored children by policy prior directly.
/// @return The index of the best move (with highest selection value) to select.
///     Returns -1 if there is no selectable children.
int selectBestmoveOfChildNode(const Node            &node,
                              std::vector<uint32_t> &edgeIndices,
                              std::vector<float>    &selectionValues,
                              std::vector<float>    &lcbValues,
                              bool                   allowDirectPolicyMove)
{
    assert(!node.isLeaf());
    edgeIndices.clear();
    selectionValues.clear();
    lcbValues.clear();

    auto [bestmoveIndex, bestmoveSelectionValue, maxBound] =
        computeSelectionValues(node, edgeIndices, selectionValues);

    // Compute lower confidence bound values if needed
    if (UseLCBForBestmoveSelection && !edgeIndices.empty())
        bestmoveIndex = applyLCBSelection(node,
                                          edgeIndices,
                                          selectionValues,
                                          lcbValues,
                                          bestmoveIndex,
                                          bestmoveSelectionValue);

    // Select best check mate move if possible
    if (maxBound.lower >= VALUE_MATE_IN_MAX_PLY)
        bestmoveIndex = overrideBestByMateBounds(node, maxBound, edgeIndices, selectionValues);

    // If we have no expanded children for selection, try select by raw policy if allowed
    if (edgeIndices.empty() && allowDirectPolicyMove)
        bestmoveIndex = selectByRawPolicy(node, edgeIndices, selectionValues);

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
                                                      true);
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

/// Process all node-table shards cooperatively: each calling thread repeatedly
/// grabs the next unprocessed shard index until none remain. Must be called
/// from a task running on every thread, sharing one counter across threads.
/// The shard's lock is held while f runs.
void processAllShards(NodeTable                                     &nodeTable,
                      std::atomic<size_t>                           &nextShardIndex,
                      const std::function<void(NodeTable::Shard &)> &f)
{
    for (;;) {
        size_t shardIdx = nextShardIndex.fetch_add(1, std::memory_order_relaxed);
        if (shardIdx >= nodeTable.getNumShards())
            return;

        NodeTable::Shard shard = nodeTable.getShardByShardIndex(shardIdx);
        std::unique_lock lock(shard.mutex);
        f(shard);
    }
}

/// Apply a custom function to the node and its children recursively.
/// @param node The node to traverse recursively.
/// @param f The function to apply to the node and its children.
/// @param prng The PRNG to use for permuting the children for multi-thread visit.
/// @param globalNodeAge The global node age to synchronize the visits.
void recursiveApply(Node &node, std::function<void(Node &)> *f, PRNG *prng, uint32_t globalNodeAge)
{
    std::atomic<uint32_t> &nodeAge = node.getAgeRef();
    // If the node's age has been updated, then the node's traversal is done
    if (nodeAge.load(std::memory_order_acquire) == globalNodeAge)
        return;

    if (!node.isLeaf()) {
        EdgeArray &edges = *node.getEdges();

        // Get the edge indices of the children to visit
        std::vector<uint32_t> edgeIndices;
        for (uint32_t edgeIndex = 0; edgeIndex < edges.numEdges; edgeIndex++) {
            Edge &childEdge = edges[edgeIndex];
            Node *childNode = childEdge.getChild();
            if (childNode)
                edgeIndices.push_back(edgeIndex);
        }

        // Shuffle the indices for better multi-thread visit
        if (prng)
            std::shuffle(edgeIndices.begin(), edgeIndices.end(), *prng);

        // Recursively apply the function to the children
        for (uint32_t edgeIndex : edgeIndices) {
            Edge &childEdge = edges[edgeIndex];
            Node *childNode = childEdge.getChild();
            recursiveApply(*childNode, f, prng, globalNodeAge);
        }
    }

    // If we are the one who first set the node age, we call the function
    uint32_t oldAge = nodeAge.exchange(globalNodeAge, std::memory_order_acq_rel);
    if (f && oldAge != globalNodeAge)
        (*f)(node);
}

}  // namespace

MCTSSearcher::MCTSSearcher() : Searcher(false)
{
    graph.root          = nullptr;
    graph.globalNodeAge = 0;
    nodeTable           = std::make_unique<NodeTable>(SearchCfg.numNodeTableShardsPowerOfTwo);
}

void MCTSSearcher::setMemoryLimit(size_t memorySizeKB)
{
    // The argument is the total MCTS search-memory budget: a small capped
    // slice goes to the (shared) VCF TT, the rest becomes the graph budget.
    size_t vcfTTMaxKB   = static_cast<size_t>(std::max(SearchCfg.mctsVCFTTMaxSizeKB, 1));
    size_t vcfTTDivisor = static_cast<size_t>(std::max(SearchCfg.mctsVCFTTBudgetDivisor, 1));

    memoryLimitKB = memorySizeKB;
    size_t vcfTTSizeKB;
    if (memorySizeKB == 0) {
        // No budget set: graph unbounded (keyed off memoryLimitKB == 0),
        // VCF TT at its default maximum.
        vcfTTSizeKB      = vcfTTMaxKB;
        graphMemoryLimit = 0;
    }
    else {
        vcfTTSizeKB    = std::clamp<size_t>(memorySizeKB / vcfTTDivisor, 1, vcfTTMaxKB);
        vcfTTSizeKB    = std::min(vcfTTSizeKB, memorySizeKB);
        size_t graphKB = memorySizeKB - vcfTTSizeKB;
        // Saturate instead of wrapping on absurd budgets (> 2^54 KiB)
        graphMemoryLimit = graphKB <= std::numeric_limits<size_t>::max() / 1024
                               ? graphKB * 1024
                               : std::numeric_limits<size_t>::max();
    }
    TT.resize(vcfTTSizeKB);
}

size_t MCTSSearcher::getMemoryLimit() const
{
    return memoryLimitKB;
}

void MCTSSearcher::clear(SearchEngine &pool, bool clearAllMemory)
{
    // Note: previousPosition is deliberately NOT reset here (preserved
    // semantics; see GraphState's lifecycle comment).
    graph.root = nullptr;
    if (!clearAllMemory)
        return;

    graph.globalNodeAge = 0;

    // Clear the node table using all threads, and wait for finish
    pool.main()->runTask([this](SearchThread &th) {
        std::atomic<size_t> numShardsProcessed = 0;
        th.engine.runOnAllThreads(
            [this, &numShardsProcessed](SearchThread &t) {
                processAllShards(*this->nodeTable, numShardsProcessed, [](NodeTable::Shard &shard) {
                    shard.table.clear();
                });
            },
            true);
    });
    pool.waitForIdle();
    graphMemoryUsed.store(0, std::memory_order_relaxed);

    // Reset node table num shards if needed
    if (nodeTable->getNumShards() != SearchCfg.numNodeTableShardsPowerOfTwo)
        nodeTable = std::make_unique<NodeTable>(SearchCfg.numNodeTableShardsPowerOfTwo);
}

const RootMove *MCTSSearcher::searchMain(SearchThread &th)
{
    SearchContext &ctx   = th.engine.ctx;
    Board         &board = *th.board;

    // Evaluator must be enabled for MCTS search
    if (!board.evaluator()) {
        FOR_EVERY_EMPTY_POS(th.board, pos)
        {
            ctx.bestMove     = pos;
            ctx.resultAction = ActionType::Move;
            ERRORL("Evaluator is not enabled, cannot use mcts search.");
            ctx.printer.printBestmoveWithoutSearch(th, pos, mated_in(0), 0, nullptr);
            return nullptr;
        }
    }

    // Initialize per-search scratch state
    scratch.numSelectableRootMoves = 0;
    scratch.lastOutputNodes        = 0;
    scratch.lastOutputTime         = now();

    // Starts worker threads, then starts main thread.
    // Root setup happens-before the workers read graph.root: runOnAllThreads
    // dispatches the tasks only after setupRootNode returns (on this thread),
    // and the task hand-off synchronizes through the thread mutex/condvar.
    ctx.printer.printSearchStarts(th, ctx.timectl);
    setupRootNode(th);  // Setup root node and other stuffs
    th.engine.runOnAllThreads([this](SearchThread &t) { search(t); }, true);

    // Rank root moves and record best move
    updateRootMovesData(th);
    ctx.printer.printRootMoves(th, ctx.timectl, scratch.numSelectableRootMoves);

    return &th.rootMoves[0];
}

void MCTSSearcher::search(SearchThread &th)
{
    SearchOptions &options = th.options();
    Board         &board   = *th.board;
    assert(!graph.root->isLeaf());

    // Main search loop
    std::vector<Node *> selectedPath;
    while (!th.engine.isTerminating()) {
        // Stop when the graph memory budget is exhausted
        if (memoryLimitKB
            && graphMemoryUsed.load(std::memory_order_relaxed) >= graphMemoryLimit)
            break;

        uint32_t newNumPlayouts = SearchCfg.maxNumVisitsPerPlayout;

        // Cap new number of playouts to the maximum num nodes to visit
        if (options.maxNodes) {
            uint64_t nodesSearched = th.engine.nodesSearched();
            if (nodesSearched >= options.maxNodes)
                break;

            uint64_t maxNodesToVisit = options.maxNodes - nodesSearched;
            if (maxNodesToVisit < newNumPlayouts)
                newNumPlayouts = maxNodesToVisit;
        }

        uint32_t newNumNodes = searchNode<true>(*graph.root, board, 0, newNumPlayouts);
        th.numNodes.fetch_add(newNumNodes, std::memory_order_relaxed);

        if (th.isMainThread()) {
            th.engine.ctx.checkExit(std::max(newNumNodes, 64u));

            bool printRootMoves = false;
            if (SearchCfg.nodesToPrintMCTSRootmoves > 0) {
                uint64_t currentNumNodes = th.engine.nodesSearched();
                uint64_t numElapsedNodes = currentNumNodes - scratch.lastOutputNodes;

                if (numElapsedNodes >= SearchCfg.nodesToPrintMCTSRootmoves) {
                    scratch.lastOutputNodes = currentNumNodes;
                    printRootMoves          = true;
                }
            }
            if (SearchCfg.timeToPrintMCTSRootmoves > 0) {
                Time currentTime = now();
                Time elapsedTime = currentTime - scratch.lastOutputTime;
                if (elapsedTime >= SearchCfg.timeToPrintMCTSRootmoves) {
                    scratch.lastOutputTime = currentTime;
                    printRootMoves         = true;
                }
            }

            if (printRootMoves) {
                updateRootMovesData(th);
                th.engine.ctx.printer.printRootMoves(th,
                                                      th.engine.ctx.timectl,
                                                      scratch.numSelectableRootMoves);
            }

            if (th.rootMoves.size() == 1
                && th.engine.nodesSearched() >= SearchCfg.numNodesAfterSingularRoot)
                th.engine.stopThinking();
        }
    }
}

bool MCTSSearcher::checkTimeupCondition(const TimeControl &timectl)
{
    if (timectl.elapsed() >= timectl.maximum())
        return true;
    if (timectl.checkStop())
        return true;
    return false;
}

void MCTSSearcher::setupRootNode(SearchThread &th)
{
    // Clear the searcher if we have not initialized yet
    if (!nodeTable)
        clear(th.engine, true);

    // Get the current root position
    std::vector<Pos> rootPosition;
    for (int moveIndex = 0; moveIndex < th.board->ply(); moveIndex++) {
        Pos move = th.board->getHistoryMove(moveIndex);
        rootPosition.push_back(move);
    }

    // Recycling must run whenever the graph memory budget is exceeded -
    // clear(false) keeps the table and the counter, so a new/backward/
    // unchanged position must still be able to reclaim the budget below.
    bool overBudget = memoryLimitKB
                      && graphMemoryUsed.load(std::memory_order_relaxed) >= graphMemoryLimit;

    // If the root position has not changed, we do not need to update the root node
    if (graph.root && rootPosition == graph.previousPosition && !overBudget)
        return;

    SearchOptions &opts = th.options();

    // Initialize the root node to expanded state
    std::tie(graph.root, std::ignore) = allocateOrFindNode(*this, th.board->zobristKey());
    if (graph.root->getVisits() == 0)
        evaluateNode<true>(*graph.root, opts, *th.board, 0);
    if (graph.root->isLeaf())
        expandNode<true>(*graph.root, opts, *th.board, 0);
    assert(graph.root->getEdges()->numEdges > 0);

    // Garbage collect old nodes (only when we go forward, and not with
    // singular root), or whenever the graph memory budget is exceeded.
    // Re-sample the budget here: root setup above may have allocated the
    // new root and its edges, crossing the limit after the first check.
    overBudget = overBudget
                 || (memoryLimitKB
                     && graphMemoryUsed.load(std::memory_order_relaxed) >= graphMemoryLimit);
    if ((rootPosition.size() >= graph.previousPosition.size() && th.rootMoves.size() > 1)
        || overBudget)
        recycleOldNodes(th);

    // Update previous Position
    graph.previousPosition = std::move(rootPosition);
}

void MCTSSearcher::recycleOldNodes(SearchThread &th)
{
    // Increment global node age
    graph.globalNodeAge += 1;

    std::atomic<uint32_t> numReachableNodes = 0;
    std::atomic<uint32_t> numRecycledNodes  = 0;

    std::function<void(Node &)> f = [&](Node &node) {
        numReachableNodes.fetch_add(1, std::memory_order_relaxed);
    };

    // Mark all reachable nodes from the root node
    th.engine.runOnAllThreads(
        [this, &f](SearchThread &t) {
            PRNG prng(Hash::LCHash(t.id));
            recursiveApply(*this->graph.root,
                           &f,
                           t.id ? &prng : nullptr,
                           this->graph.globalNodeAge);
        },
        true);

    // Remove all unreachable nodes
    std::atomic<size_t> numShardsProcessed = 0;

    th.engine.runOnAllThreads(
        [this, &numRecycledNodes, &numShardsProcessed](SearchThread &t) {
            processAllShards(
                *this->nodeTable,
                numShardsProcessed,
                [this, &numRecycledNodes](NodeTable::Shard &shard) {
                    for (auto it = shard.table.begin(); it != shard.table.end();) {
                        Node *node = std::addressof(const_cast<Node &>(*it));
                        if (node->getAgeRef().load(std::memory_order_relaxed)
                            != this->graph.globalNodeAge) {
                            size_t footprint = nodeFootprint();
                            if (const EdgeArray *edges = node->getEdges())
                                footprint += edgeArrayFootprint(edges->numEdges);
                            this->graphMemoryUsed.fetch_sub(footprint,
                                                            std::memory_order_relaxed);
                            it = shard.table.erase(it);
                            numRecycledNodes.fetch_add(1, std::memory_order_relaxed);
                        }
                        else
                            it++;
                    }
                });
        },
        true);

    MESSAGEL("Reachable nodes: " << numReachableNodes.load()
                                 << ", Recycled nodes: " << numRecycledNodes.load()
                                 << ", Root visit: " << graph.root->getVisits());
}

void MCTSSearcher::updateRootMovesData(SearchThread &th)
{
    assert(graph.root != nullptr);
    assert(!graph.root->isLeaf());

    std::vector<uint32_t> edgeIndices;
    std::vector<float>    selectionValues, lcbValues;
    selectBestmoveOfChildNode(*graph.root, edgeIndices, selectionValues, lcbValues, true);
    uint32_t maxNumRootMovesToPrint =
        std::max<uint32_t>(th.options().multiPV, SearchCfg.maxNonPVRootmovesToPrint);

    for (RootMove &rm : th.rootMoves) {
        rm.selectionValue = std::numeric_limits<float>::lowest();
        rm.previousValue  = rm.value;
        rm.previousPv     = rm.pv;
        rm.pv.resize(1);
    }

    EdgeArray &edges               = *graph.root->getEdges();
    scratch.numSelectableRootMoves = 0;
    for (size_t i = 0; i < edgeIndices.size(); i++) {
        uint32_t edgeIndex = edgeIndices[i];
        Edge    &childEdge = edges[edgeIndex];
        Pos      move      = childEdge.getMove();

        // Skip edges that are not in the root move list
        auto rm = std::find(th.rootMoves.begin(), th.rootMoves.end(), move);
        if (rm == th.rootMoves.end())
            continue;

        // Record the root move's value for explored children
        Node *childNode = childEdge.getChild();
        if (childNode) {
            ValueBound childBound = childNode->getBound();
            if (childNode->getVisits() > 0) {
                float childUtility = -childNode->getQ();

                rm->winRate  = childUtility * 0.5f + 0.5f;
                rm->drawRate = childNode->getD();
                if (Value lo = childBound.childLowerBound(); lo >= VALUE_MATE_IN_MAX_PLY)
                    rm->value = mate_in(std::max(mate_step(lo, 0) - th.board->ply(), 0));
                else if (Value up = childBound.childUpperBound(); up <= VALUE_MATED_IN_MAX_PLY)
                    rm->value = mated_in(std::max(mate_step(up, 0) - th.board->ply(), 0));
                else
                    rm->value = Evaluation::winRateToValue(rm->winRate);
                rm->utilityStdev = std::sqrt(childNode->getQVar());
                scratch.numSelectableRootMoves++;
            }
            rm->numNodes = childEdge.getVisits();
            extractPVOfChildNode(*childNode, rm->pv);
        }
        else {
            rm->numNodes = 0;
        }
        rm->policyPrior = childEdge.getP();
        rm->lcbValue =
            i < lcbValues.size() ? lcbValues[i] : std::numeric_limits<float>::quiet_NaN();
        rm->selectionValue = selectionValues[i];
    }

    // If we do not have any visited children, display all of them to show policy
    if (scratch.numSelectableRootMoves == 0)
        scratch.numSelectableRootMoves = th.rootMoves.size();
    scratch.numSelectableRootMoves =
        std::min(scratch.numSelectableRootMoves, maxNumRootMovesToPrint);

    // Sort the root moves in descending order by selection value
    std::stable_sort(th.rootMoves.begin(),
                     th.rootMoves.end(),
                     [](const RootMove &m1, const RootMove &m2) {
                         return m1.selectionValue > m2.selectionValue;
                     });
}
