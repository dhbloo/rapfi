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

#include "../core/platform.h"
#include "searchcommon.h"
#include "searchengine.h"
#include "searcher.h"

#include <atomic>
#include <functional>
#include <memory>
#include <unordered_map>

#ifdef MULTI_THREADING
    #include <condition_variable>
    #include <mutex>
    #include <thread>
#endif

class Board;  // forward declaration

namespace Database {
class DBClient;  // forward declaration
}

namespace Search {

/// SearchThread holds all per-thread search resources: a board clone, an
/// evaluator instance, a database client, the per-thread algorithm data and
/// the root move list, plus the task-dispatch machinery. The main search
/// thread is the one with id == 0 ("main" is a role, not a type).
class SearchThread
{
private:
    friend class SearchEngine;
    Numa::NumaNodeId numaId;
    // Atomic: waitForIdle() reads `running` and ~SearchThread writes `exit`
    // without holding the mutex. All wakeup-relevant writes stay under the
    // mutex and are paired with a notify, so the cv predicates cannot miss
    // a transition.
    std::atomic<bool> running, exit;

#ifdef MULTI_THREADING
    std::function<void(SearchThread &)> taskFunc;
    std::thread                         thread;
    std::mutex                          mutex;
    std::condition_variable             cv;

    void threadLoop();
#endif

public:
    /// Instantiate a new search thread.
    /// @param id ID of the new search thread, starting from 0 for main thread.
    /// @param bindGroup Whether to bind this thread to a NUMA group.
    explicit SearchThread(SearchEngine &searchEngine, uint32_t id);
    /// Start the thread loop. This should be called once after the thread is created.
    void init(bool bindGroup);
    /// Destroy this search thread. Search must be stopped before entering.
    ~SearchThread();
    /// Clear the thread state between two search.
    void clear();
    /// Setup the board instance in this thread, and update the evaluator.
    void setBoardAndEvaluator(const Board &board);
    /// Return if this thread is the main thread.
    bool isMainThread() const { return id == 0; }
    /// Launch a custom task in this thread.
    void runTask(std::function<void(SearchThread &)> task);
    /// Wait until threadLoop() enters idle state.
    void waitForIdle();

    /// Get the search data as a specific type.
    template <typename SearchDataType>
    SearchDataType *searchDataAs() const;

    /// Get the shared search options.
    SearchOptions &options() const;

public:
    /// The ID of this search thread.
    const uint32_t id;

    /// Reference to the search engine that this thread belongs to.
    SearchEngine &engine;

    /// Board instance of this thread
    std::unique_ptr<Board> board;

    /// NNUE evaluator instance
    std::unique_ptr<Evaluation::Evaluator> evaluator;

    /// Database client instance
    std::unique_ptr<Database::DBClient> dbClient;

    /// Custom search data created from searcher
    std::unique_ptr<SearchData> searchData;

    /// Root moves
    RootMoves rootMoves;

    /// Balance2 move -> root move index lookup table
    std::unordered_map<Balance2Move, size_t, Balance2Move::Hash> balance2Moves;

    // Common thread-related statistics
    // ----------------------------------------------------

    /// Nodes count searched by this thread
    std::atomic<uint64_t> numNodes;
    /// Maximum depth reached by this thread
    int selDepth;
};

template <typename SearchDataType>
inline SearchDataType *SearchThread::searchDataAs() const
{
    return static_cast<SearchDataType *>(searchData.get());
}

inline SearchOptions &SearchThread::options() const
{
    return engine.ctx.options;
}

inline uint64_t SearchEngine::nodesSearched() const
{
    uint64_t sum = 0;
    for (const auto &th : threads_)
        sum += th->numNodes.load(std::memory_order_relaxed);
    return sum;
}

}  // namespace Search
