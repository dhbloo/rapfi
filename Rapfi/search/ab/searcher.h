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

#pragma once

#include "../../core/pos.h"
#include "../../core/types.h"
#include "../searcher.h"
#include "../searchoutput.h"
#include "../searchthread.h"
#include "../timecontrol.h"
#include "history.h"

#include <atomic>

namespace Search::AB {

struct ABSearchData : SearchData
{
    uint32_t         multiPv;          /// Current number of multi pv
    uint32_t         pvIdx;            /// Current searched pv index
    int              rootDepth;        /// Current searched depth
    Value            rootDelta;        /// Current window size of the root node
    Value            rootAlpha;        /// Current alpha value of the root node
    bool             singularRoot;     /// Is there only a single response at root?
    std::atomic<int> completedDepth;   /// Previously completed depth
    std::atomic<int> bestMoveChanges;  /// How many time best move has changed in this search

    MainHistory        mainHistory;         /// Heuristic history table
    CounterMoveHistory counterMoveHistory;  /// Counter move history table

    ~ABSearchData() = default;

    /// Clear all search states between two search.
    void clearData(SearchThread &th) override;
};

class ABSearcher : public Searcher
{
public:
    /// Time controller
    TimeControl timectl;
    /// Printer for all search messages
    SearchPrinter printer;

    // Time management information
    float previousTimeReduction;  // (keep for one game)
    Value previousBestValue;      // (keep for one game)

    /// Lookup tables used for reduction/purning, where index is depth or moveCount.
    std::array<Depth, MAX_MOVES + 1> reductions[RULE_NB];

    ~ABSearcher() = default;
    std::unique_ptr<SearchData> makeSearchData(SearchThread &th) override
    {
        return std::make_unique<ABSearchData>();
    }

    /// Set the memory size limit of the search.
    void setMemoryLimit(size_t memorySizeKB) override;

    /// Get the current memory size limit of the search.
    size_t getMemoryLimit() const override;

    /// Clear main thread states (and TT) between different games.
    void clear(ThreadPool &pool, bool clearAllMemory) override;

    /// The thinking entry point. When program receives search command, main
    /// thread is started first and other threads are launched by main thread.
    void searchMain(MainSearchThread &th) override;

    /// The main iterative deeping search loop. It calls search() repeatedly with increasing depth
    /// until the stop condition is reached. Results are updated to thread bounded with the board.
    void search(SearchThread &th) override;

    /// Checks if current search reaches timeup condition.
    bool checkTimeupCondition() override;

private:
    /// Choose the next search depth by checking completed depth of all other
    /// threads and selecting the next depth with least working threads to
    /// avoid repeated searching.
    int pickNextDepth(ThreadPool &threads, uint32_t thisId, int lastDepth) const;

    /// Pick thread with the best result according to eval and completed depth.
    SearchThread *pickBestThread(ThreadPool &threads) const;
};

}  // namespace Search::AB
