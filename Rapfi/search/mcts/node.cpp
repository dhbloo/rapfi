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

#include "../../config.h"
#include "nodetable.h"

namespace Search::MCTS {

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
{}

Node::~Node()
{
    EdgeArray *edgeArray = edges.exchange(nullptr, std::memory_order_relaxed);
    if (edgeArray)
        delete edgeArray;
}

void Node::setTerminal(Value value)
{
    assert(value != VALUE_NONE);
    assert(-VALUE_INFINITE <= value && value <= VALUE_INFINITE);
    terminalValue = value;
    if (value >= VALUE_MATE_IN_MAX_PLY) {
        utility  = 1.0;
        drawRate = 0.0f;
    }
    else if (value <= VALUE_MATED_IN_MAX_PLY) {
        utility  = -1.0;
        drawRate = 0.0f;
    }
    else {
        utility  = Config::valueToWinRate(value) * 2 - 1.0f;
        drawRate = 1.0f;
    }

    assert(utility >= -1.0f && utility <= 1.0f);
    assert(drawRate >= 0.0f && drawRate <= 1.0f);

    q.store(utility, std::memory_order_relaxed);
    qSqr.store(utility * utility, std::memory_order_relaxed);
    d.store(drawRate, std::memory_order_relaxed);
    bound.store(ValueBound {value}, std::memory_order_relaxed);
    n.store(1, std::memory_order_release);
}

void Node::setNonTerminal(float utility, float drawRate)
{
    assert(utility >= -1.0f && utility <= 1.0f);
    assert(drawRate >= 0.0f && drawRate <= 1.0f);

    this->utility       = utility;
    this->drawRate      = drawRate;
    this->terminalValue = VALUE_NONE;

    q.store(utility, std::memory_order_relaxed);
    qSqr.store(utility * utility, std::memory_order_relaxed);
    d.store(drawRate, std::memory_order_relaxed);
    n.store(1, std::memory_order_release);
}

bool Node::createEdges(MovePicker &movePicker)
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

    using AllocType      = std::aligned_storage_t<sizeof(EdgeArray), alignof(EdgeArray)>;
    size_t     numAllocs = (sizeof(EdgeArray) + numEdges * sizeof(Edge)) / sizeof(AllocType);
    EdgeArray *tempEdges = reinterpret_cast<EdgeArray *>(new AllocType[numAllocs]);

    // Copy the move and policy array to the allocated edge array
    tempEdges->numEdges = numEdges;
    for (uint32_t i = 0; i < numEdges; i++)
        new (&tempEdges->edges[i]) Edge(moveList[i], policyList[i]);

    EdgeArray *expected = nullptr;
    bool       suc = edges.compare_exchange_strong(expected, tempEdges, std::memory_order_release);
    // If we are not the one that sets the edge array, then we need to delete the temp edge array
    if (!suc)
        delete[] tempEdges;

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
