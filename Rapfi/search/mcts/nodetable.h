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

#include "../../core/types.h"
#include "node.h"

#include <memory>
#include <mutex>
#include <set>
#include <shared_mutex>

namespace Search::MCTS {

class NodeTable
{
public:
    using NodePtr = std::unique_ptr<Node>;
    struct NodeCompare
    {
        using is_transparent = void;
        bool operator()(HashKey lhs, const NodePtr &rhs) const { return lhs < rhs->getHash(); }
        bool operator()(const NodePtr &lhs, HashKey rhs) const { return lhs->getHash() < rhs; }
        bool operator()(const NodePtr &lhs, const NodePtr &rhs) const
        {
            return lhs->getHash() < rhs->getHash();
        }
    };
    using Table = std::set<NodePtr, NodeCompare>;

    struct Shard
    {
        size_t             index;
        Table             &table;
        std::shared_mutex &mutex;
    };

    NodeTable(size_t numShardsPowerOfTwo)
        : numShards(1 << numShardsPowerOfTwo)
        , mask(numShards - 1)
        , tables(std::make_unique<Table[]>(numShards))
        , mutexes(std::make_unique<std::shared_mutex[]>(numShards))
    {}

    /// Get the total number of shards of this node table.
    size_t getNumShards() const { return numShards; }

    /// Get the shard that contains the node with the given hash key.
    /// @note This function is thread-safe.
    Shard getShard(HashKey hash) const
    {
        size_t index = hash & mask;
        return Shard {index, tables[index], mutexes[index]};
    }

    /// Find the node with the given hash key.
    /// @return Pointer to the node if found, otherwise nullptr.
    /// @note This function uses reader lock to ensure thread-safety.
    Node *findNode(HashKey hash) const
    {
        Shard            shard = getShard(hash);
        std::shared_lock lock(shard.mutex);
        auto             it = shard.table.find(hash);
        return it != shard.table.end() ? it->get() : nullptr;
    }

    /// Try insert a new node into the table.
    /// @param nodePtr Pointer to the node to be inserted.
    /// @return A pair of (Pointer to the inserted node, Whether the node is
    ///   successfully inserted). If there is already a node inserted by other
    ///   threads, the pointer to that node is returned instead.
    std::pair<Node *, bool> tryInsertNode(std::unique_ptr<Node> nodePtr)
    {
        HashKey          hash  = nodePtr->getHash();
        Shard            shard = getShard(hash);
        std::unique_lock lock(shard.mutex);

        auto [it, inserted] = shard.table.emplace(std::move(nodePtr));
        return {it->get(), inserted};
    }

private:
    size_t                               numShards;
    size_t                               mask;
    std::unique_ptr<Table[]>             tables;
    std::unique_ptr<std::shared_mutex[]> mutexes;
};

}  // namespace Search::MCTS
