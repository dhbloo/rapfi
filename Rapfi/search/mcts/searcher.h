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
#include "../searchoutput.h"
#include "../searchthread.h"
#include "../timecontrol.h"
#include "node.h"
#include "nodetable.h"

#include <atomic>
#include <unordered_set>

namespace Search::MCTS {

class MCTSSearcher : public Searcher
{
public:
    /// Time controller
    TimeControl timectl;
    /// Printer for all search messages
    SearchPrinter printer;
    /// The node table for storing and finding all transposition nodes
    std::unique_ptr<NodeTable> nodeTable;
    /// The root node of the MCTS tree
    Node *root;
    /// The searched position of last root node
    std::vector<Pos> previousPosition;
    /// The global node age to synchronize the node table
    uint32_t globalNodeAge;
    /// The number of selectable root moves, set by updateRootMovesData().
    uint32_t numSelectableRootMoves;
    // The last number of nodes that we have printed search outputs
    uint64_t lastOutputNodes;
    // The last time that we have printed search outputs
    Time lastOutputTime;

    MCTSSearcher();
    ~MCTSSearcher() = default;

    std::unique_ptr<SearchData> makeSearchData(SearchThread &th) override { return nullptr; }

    /// Set the memory size limit of the search.
    void setMemoryLimit(size_t memorySizeKB) override;

    /// Get the current memory size limit of the search.
    size_t getMemoryLimit() const override;

    /// Clear the state of the searcher between two different games
    void clear(ThreadPool &pool, bool clearAllMemory) override;

    /// The thinking entry point. When program receives search command, main
    /// thread is started first and other threads are launched by main thread.
    void searchMain(MainSearchThread &th) override;

    /// The main best first search loop. It calls search() repeatedly until the stop
    /// condition is reached. Results are updated to thread bounded with the board.
    void search(SearchThread &th) override;

    /// Checks if current search reaches timeup condition.
    bool checkTimeupCondition() override;

private:
    /// Setup root node for the search
    void setupRootNode(MainSearchThread &th);

    /// Garbage collect all old nodes in the node table
    void recycleOldNodes(MainSearchThread &th);

    /// Rank the root moves and update PV, then print all root moves
    void updateRootMovesData(MainSearchThread &th);
};

}  // namespace Search::MCTS
