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

#pragma once

#include "../../core/pos.h"
#include "../../core/types.h"
#include "../movepick.h"

#include <atomic>
#include <cassert>
#include <cstring>
#include <memory>

namespace Search::MCTS {

class Node;
class NodeTable;

/// An edge in the MCTS graph.
/// It contains the move, policy and num visits of this edge.
class Edge
{
public:
    Edge(Pos move, float p) : move(move), edgeVisits(0), child(nullptr) { setP(p); }

    /// Get the move of this edge.
    Pos getMove() const { return move; }

    /// Converts a float32 policy in [0, 1] to a quantized 16bit policy.
    /// @see
    /// https://github.com/LeelaChessZero/lc0/blob/51f93b7c49720ee100d24aac54193a88ba98219a/src/mcts/node.cc#L130-L167
    void setP(float p)
    {
        assert(0.0f <= p && p <= 1.0f);
        constexpr int32_t roundings = (1 << 11) - (3 << 28);
        int32_t           bits;  // TODO: change to std::bit_cast in C++20
        std::memcpy(&bits, &p, sizeof(float));
        bits += roundings;
        policy = (bits < 0) ? 0 : static_cast<uint16_t>(bits >> 12);
    }

    /// Get the normalized policy of this edge.
    float getP() const
    {
        uint32_t bits = (static_cast<uint32_t>(policy) << 12) | (3 << 28);
        float    p;  // TODO: change to std::bit_cast in C++20
        std::memcpy(&p, &bits, sizeof(uint32_t));
        return p;
    }

    /// Get the number of edge visits of this edge.
    uint32_t getVisits() const { return edgeVisits.load(std::memory_order_acquire); }

    /// Increment the number of edge visits by delta atomically.
    /// @param delta The number of edge visits to change.
    /// @return The number of edge visits after change.
    uint32_t addVisits(uint32_t delta)
    {
        return edgeVisits.fetch_add(delta, std::memory_order_acq_rel);
    }

    /// Get the child node of this edge.
    /// @note when `getVisits() > 0`, the child node should be guaranteed to be non-null.
    Node *getChild() const { return child.load(std::memory_order_acquire); }

    /// Set the child node of this edge.
    void setChild(Node *node) { child.store(node, std::memory_order_release); }

private:
    /// Move of this edge in respect to the current side to move.
    Pos move;

    /// quantized and compressed 16bit policy value
    uint16_t policy;

    /// Number of finished visits of this edge.
    std::atomic<uint32_t> edgeVisits;

    /// Pointer to the child node.
    std::atomic<Node *> child;
};

/// Represents an allocated array of edges.
struct alignas(alignof(Edge)) EdgeArray
{
    uint32_t numEdges;
    Edge     edges[];

    /// Get the edge reference at the given index.
    Edge &operator[](uint32_t index)
    {
        assert(index < numEdges);
        return edges[index];
    }

    /// Get the const edge reference at the given index.
    const Edge &operator[](uint32_t index) const
    {
        assert(index < numEdges);
        return edges[index];
    }
};

static_assert(alignof(EdgeArray) == alignof(Edge), "Sanity check on EdgeArray's alignment");
static_assert(sizeof(Edge) % sizeof(EdgeArray) == 0, "Sanity check on Edge's size");

/// Represents the value bound of a node.
struct ValueBound
{
    Eval lower, upper;

    ValueBound() : lower(-VALUE_INFINITE), upper(VALUE_INFINITE) {};
    ValueBound(Value terminalValue) : lower(terminalValue), upper(terminalValue) {}

    /// Returns whether this bound is terminal.
    bool isTerminal() const { return lower == upper; }
    /// Returns the lower bound of this child node's bound.
    Value childLowerBound() const { return static_cast<Value>(-upper); }
    /// Returns the upper bound of this child node's bound.
    Value childUpperBound() const { return static_cast<Value>(-lower); }
    /// Add a child node's bound to this parent node's bound.
    ValueBound &operator|=(ValueBound childBound)
    {
        lower = std::max<Eval>(lower, -childBound.upper);
        upper = std::max<Eval>(upper, -childBound.lower);
        return *this;
    }
};

static_assert(std::atomic<ValueBound>::is_always_lock_free,
              "std::atomic<ValueBound> should be a lock free atomic variable");

/// A node in the MCTS graph.
/// It contains the edges, children, and statistics of this node.
class Node
{
public:
    /// Constructs a new unevaluated node with no children edges.
    /// @param hash The graph hash key of this node.
    /// @param age The initial age of this node.
    explicit Node(HashKey hash, uint32_t age);
    ~Node();

    // Disallow copy and move. We need node's address to have pointer stability.
    Node(const Node &rhs)            = delete;
    Node &operator=(const Node &rhs) = delete;
    Node(Node &&rhs)                 = delete;
    Node &operator=(Node &&rhs)      = delete;

    /// Set this node to be a terminal node and set num visits to 1.
    /// @param value The terminal value of this node.
    void setTerminal(Value value);

    /// Set this node to be a non-terminal node and set num visits to 1.
    /// @param utility The raw utility value of this node.
    /// @param drawRate The raw draw probability of this node.
    void setNonTerminal(float utility, float drawRate);

    /// Initializes the edges of this node from the given move picker.
    /// @param movePicker The move picker to generate the edges.
    /// @return Whether this node has no valid edges. If true,
    ///   this node is a terminal node that has been mated.
    bool createEdges(MovePicker &movePicker);

    /// Returns the graph hash key of this node.
    HashKey getHash() const { return hash; }

    /// Returns whether this node has no children.
    bool isLeaf() const { return edges.load(std::memory_order_relaxed) == nullptr; }

    /// Returns the edge array at the given index.
    /// If this node has no edges, returns nullptr.
    EdgeArray       *getEdges() { return edges.load(std::memory_order_relaxed); }
    const EdgeArray *getEdges() const { return edges.load(std::memory_order_relaxed); }

    /// Returns the average utility value of this node.
    float getQ() const { return q.load(std::memory_order_relaxed); }

    /// Returns the estimated sample variance of utility value.
    float getQVar(float priorVar = 1.0f, float priorWeight = 1.0f) const;

    /// Returns the average draw rate of this node.
    float getD() const { return d.load(std::memory_order_relaxed); }

    /// Returns the total visits of this node.
    uint32_t getVisits() const { return n.load(std::memory_order_acquire); }

    /// Returns the total virtual visits of this node.
    uint32_t getVirtualVisits() const { return nVirtual.load(std::memory_order_acquire); }

    /// Returns the reference to the age of this node.
    std::atomic<uint32_t> &getAgeRef() { return age; }

    /// Returns the evaluated utility value of this node.
    float getEvalUtility() const { return utility; }

    /// Returns the evaluated draw rate of this node.
    float getEvalDrawRate() const { return drawRate; }

    /// Returns the propogated value bound of this node.
    ValueBound getBound() const { return bound.load(std::memory_order_relaxed); }

    /// Returns if this node is terminal.
    bool isTerminal() const { return terminalValue != VALUE_NONE; }

    /// Update the average utility and the average draw rate from childrens.
    /// @param board The corresponding board of this node.
    /// @param nodeTable The node table for finding children nodes.
    void updateStats();

    /// Begin the visit of this node.
    void beginVisit(uint32_t newVisits)
    {
        nVirtual.fetch_add(newVisits, std::memory_order_acq_rel);
    }

    /// Finish the visit of this node by incrementing the total visits of this node.
    void finishVisit(uint32_t newVisits, uint32_t actualNewVisits)
    {
        if (actualNewVisits)
            n.fetch_add(actualNewVisits, std::memory_order_acq_rel);
        nVirtual.fetch_add(-newVisits, std::memory_order_release);
    }

    /// Directly increment the total visits of this node by delta.
    /// Usually used for visiting a terminal node.
    void incrementVisits(uint32_t delta) { n.fetch_add(delta, std::memory_order_acq_rel); }

private:
    /// The graph hash of this node.
    const HashKey hash;

    /// The edge array of this node, edges are sorted by normalized policy.
    std::atomic<EdgeArray *> edges;

    /// Total visits under this node's subgraph.
    /// For leaf node, this indicates if the node's value has been evaluated.
    /// For non-leaf node, this is the sum of all children's edge visits plus 1.
    std::atomic<uint32_t> n;

    /// Total started but not finished visits under this node's subgraph,
    /// mainly for computing virtual loss when multi-threading is used.
    std::atomic<uint32_t> nVirtual;

    /// Average utility (from current side to move) of this node, in [-1,1].
    std::atomic<float> q;

    /// Average squared utility of this node, for computing utility variance.
    std::atomic<float> qSqr;

    /// Average draw rate of this node in [0,1]. Not flipped when changing side.
    std::atomic<float> d;

    /// The age of this node, used to find and recycle unused nodes.
    std::atomic<uint32_t> age;

    /// The propogated terminal value bound of this node.
    std::atomic<ValueBound> bound;

    /// For non-terminal node, this stores the node's raw utility value in [-1,1].
    /// Higher values means better position from current side to move.
    float utility;

    /// For non-terminal node, this stores the node's raw draw probability in [0,1].
    /// (from current side to move).
    float drawRate;

    /// For terminal node, this stores the theoretical value (from current
    /// side to move), including the game ply to mate/mated. If this node is
    /// not a terminal node, this value is VALUE_NONE.
    Eval terminalValue;
};

}  // namespace Search::MCTS
