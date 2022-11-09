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
    bool             singularRoot;     /// Is there only a single response at root?
    std::atomic<int> completedDepth;   /// Previously completed depth
    std::atomic<int> bestMoveChanges;  /// How many time best move has changed in this search

    MainHistory        mainHistory;         /// Heuristic history table
    CounterMoveHistory counterMoveHistory;  /// Counter move history table

    ~ABSearchData() = default;
    void clearData() override;
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
    std::array<Depth, MAX_MOVES + 1> reductions;

    ~ABSearcher() = default;
    std::unique_ptr<SearchData> makeSearchData(SearchThread &th) override
    {
        return std::make_unique<ABSearchData>();
    }

    void clear(ThreadPool &pool, bool clearAllMemory) override;
    void searchMain(MainSearchThread &th) override;
    void search(SearchThread &th) override;
    bool checkTimeupCondition() override;

private:
    int           pickNextDepth(ThreadPool &threads, uint32_t thisId, int lastDepth) const;
    SearchThread *pickBestThread(ThreadPool &threads) const;
};

}  // namespace Search::AB
