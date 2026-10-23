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

#include "node.h"

#include "../../eval/scoretables.h"
#include "nodetable.h"

#include <type_traits>

namespace Search::MCTS {

namespace {

/// Raw storage type backing every EdgeArray allocation. The flexible-length
/// edge array is allocated as an array of this type, so all deallocation
/// paths must free it as an array of this same type.
using EdgeArrayAlloc = std::aligned_storage_t<sizeof(EdgeArray), alignof(EdgeArray)>;

}  // namespace

Node::Node(HashKey hash, uint32_t age)
    : hash(hash)
    , edges(nullptr)
    , n(0)
    , nVirtual(0)
    , q(1.0f)  // Default as a losing node for the parent's side of view
    , qSqr(1.0f)
    , d(0.0f)
    , age(age)
    , bound()
    , terminalValue(static_cast<Eval>(VALUE_NONE))
{}

Node::~Node()
{
    EdgeArray *edgeArray = edges.exchange(nullptr, std::memory_order_relaxed);
    if (edgeArray)
        delete[] reinterpret_cast<EdgeArrayAlloc *>(edgeArray);
}

void Node::writeTerminalStats(Value value)
{
    if (value >= VALUE_MATE_IN_MAX_PLY) {
        utility  = 1.0;
        drawRate = 0.0f;
    }
    else if (value <= VALUE_MATED_IN_MAX_PLY) {
        utility  = -1.0;
        drawRate = 0.0f;
    }
    else {
        utility  = Evaluation::valueToWinRate(value) * 2 - 1.0f;
        drawRate = 1.0f;
    }

    assert(utility >= -1.0f && utility <= 1.0f);
    assert(drawRate >= 0.0f && drawRate <= 1.0f);

    q.store(utility, std::memory_order_relaxed);
    qSqr.store(utility * utility, std::memory_order_relaxed);
    d.store(drawRate, std::memory_order_relaxed);
    bound.store(ValueBound {value}, std::memory_order_relaxed);
}

void Node::setTerminal(Value value)
{
    assert(value != VALUE_NONE);
    assert(-VALUE_INFINITE <= value && value <= VALUE_INFINITE);
    // This publication path is reserved for the single evaluating thread:
    // other threads discard their visits until the release store below.
    assert(getVisits() == 0);
    terminalValue.store(static_cast<Eval>(value), std::memory_order_relaxed);
    writeTerminalStats(value);
    n.store(1, std::memory_order_release);
}

bool Node::tryMarkTerminal(Value value)
{
    assert(value != VALUE_NONE);
    assert(-VALUE_INFINITE <= value && value <= VALUE_INFINITE);

    // Invariant: after publication, terminalValue moves away from VALUE_NONE
    // at most once in a node's lifetime. Claiming the transition with this
    // CAS makes the winner the only writer of the terminal statistics, and
    // the visit count is deliberately left untouched so visits accumulated
    // by concurrent playouts are preserved (a reader may transiently observe
    // a terminal node whose mean stats still hold the evaluated values; the
    // parent's next updateStats() recomputes from the terminal stats).
    Eval expected = static_cast<Eval>(VALUE_NONE);
    if (!terminalValue.compare_exchange_strong(expected,
                                               static_cast<Eval>(value),
                                               std::memory_order_relaxed))
        return false;

    writeTerminalStats(value);
    return true;
}

void Node::setNonTerminal(float utility, float drawRate)
{
    assert(utility >= -1.0f && utility <= 1.0f);
    assert(drawRate >= 0.0f && drawRate <= 1.0f);

    this->utility  = utility;
    this->drawRate = drawRate;
    terminalValue.store(static_cast<Eval>(VALUE_NONE), std::memory_order_relaxed);

    q.store(utility, std::memory_order_relaxed);
    qSqr.store(utility * utility, std::memory_order_relaxed);
    d.store(drawRate, std::memory_order_relaxed);
    n.store(1, std::memory_order_release);
}

bool Node::createEdges(MovePicker &movePicker, std::atomic<size_t> *allocCounter)
{
    Pos      moveList[MAX_MOVES];
    float    policyList[MAX_MOVES];
    uint32_t numEdges = 0;
    // Moves from the move picker should be sorted
    while (Pos move = movePicker()) {
        assert(movePicker.hasPolicyScore());
        assert(movePicker.hasNormalizedPolicy());
        moveList[numEdges]   = move;
        policyList[numEdges] = movePicker.curMoveNormalizePolicy();
        numEdges++;
    }

    // If no valid edges, then this node is a (mated) terminal node
    if (numEdges == 0)
        return true;

    size_t     numAllocs = (sizeof(EdgeArray) + numEdges * sizeof(Edge)) / sizeof(EdgeArrayAlloc);
    EdgeArray *tempEdges = reinterpret_cast<EdgeArray *>(new EdgeArrayAlloc[numAllocs]);

    // Copy the move and policy array to the allocated edge array
    tempEdges->numEdges = numEdges;
    for (uint32_t i = 0; i < numEdges; i++)
        new (&tempEdges->edges[i]) Edge(moveList[i], policyList[i]);

    EdgeArray *expected = nullptr;
    bool       suc = edges.compare_exchange_strong(expected, tempEdges, std::memory_order_release);
    // If we are not the one that sets the edge array, then we need to delete the temp edge array
    if (!suc)
        delete[] reinterpret_cast<EdgeArrayAlloc *>(tempEdges);
    else if (allocCounter)
        allocCounter->fetch_add(edgeArrayFootprint(numEdges), std::memory_order_relaxed);

    return false;
}

float Node::getQVar(float priorVar, float priorWeight) const
{
    uint32_t visits = n.load(std::memory_order_relaxed);
    if (visits < 2)
        return priorVar;

    float weight        = visits;
    float utilityAvg    = q.load(std::memory_order_relaxed);
    float utilitySqrAvg = qSqr.load(std::memory_order_relaxed);

    // sample_var = (utilitySqrAvg - utilityAvg * utilityAvg) * weight / (weight - 1.0f)
    // regula_var = (sample_var * (weight-1) + varPrior * priorWeight) / (weight-1 + priorWeight)
    float sampleVariance = std::max(utilitySqrAvg - utilityAvg * utilityAvg, 0.0f);
    float regularizedVariance =
        (sampleVariance * weight + priorVar * priorWeight) / (weight + priorWeight - 1.0f);
    return regularizedVariance;
}

void Node::updateStats()
{
    // Concurrency note: child visit counts and values are read with relaxed
    // atomics while other threads may be updating them. The aggregation is
    // statistically tolerant by design - a slightly stale or torn-across-
    // children snapshot only perturbs this node's running averages, which the
    // next updateStats() call recomputes from scratch.
    const EdgeArray *edgeArray = getEdges();
    // If this node is not expanded, then we do not need to update any stats
    if (!edgeArray)
        return;

    uint32_t   nSum     = 1;
    float      qSum     = utility;
    float      qSqrSum  = utility * utility;
    float      dSum     = drawRate;
    ValueBound maxBound = ValueBound {-VALUE_INFINITE};
    for (uint32_t i = 0; i < edgeArray->numEdges; i++) {
        const Edge &edge   = (*edgeArray)[i];
        uint32_t    childN = edge.getVisits();

        // No need to read from child node if it has zero edge visits
        if (childN == 0) {
            // All unvisited children all have uninitialized bound (-inf, inf)
            maxBound |= ValueBound {};
            break;
        }

        Node *childNode = edge.getChild();
        assert(childNode);  // child node should be guaranteed to be non-null

        float childQ    = childNode->q.load(std::memory_order_relaxed);
        float childQSqr = childNode->qSqr.load(std::memory_order_relaxed);
        float childD    = childNode->d.load(std::memory_order_relaxed);
        nSum += childN;
        qSum += childN * (-childQ);  // Flip side for child's utility
        qSqrSum += childN * childQSqr;
        dSum += childN * childD;
        maxBound |= childNode->bound.load(std::memory_order_relaxed);
    }

    float norm = 1.0f / nSum;
    q.store(qSum * norm, std::memory_order_relaxed);
    qSqr.store(qSqrSum * norm, std::memory_order_relaxed);
    d.store(dSum * norm, std::memory_order_relaxed);

    // Only update bound if this is not a terminal node
    if (!isTerminal())
        bound.store(maxBound, std::memory_order_relaxed);
}

}  // namespace Search::MCTS
