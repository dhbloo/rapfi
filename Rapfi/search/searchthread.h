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

#include "../database/dbclient.h"
#include "../database/dbstorage.h"
#include "../eval/evaluator.h"
#include "searchcommon.h"
#include "searcher.h"
#include "timecontrol.h"

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

namespace Search {

class Searcher;    // forward declaration
class ThreadPool;  // forward declaration

/// SearchThread is the base class for all search algorithms. It contains per-thread
/// states, including a board clone, evaluator, root moves list, as well as various
/// heruistic states. All search algorithms must have a derived class from SearchThread
/// to implement their search routines and handle thread-related search states.
struct SearchThread
{
public:
    const uint32_t id;

private:
    friend struct MainSearchThread;
    friend class ThreadPool;
    bool running, exit;

    /// Launch a custom task in this thread.
    void runTask(std::function<void()> task);
    /// Wake up threadLoop() and start searching routine.
    void runSearch();
    /// Wake up threadLoop() and start clearing routine.
    void runClear();
    /// Wait until threadLoop() enters idle state.
    void waitForIdle();

#ifdef MULTI_THREADING
    std::function<void()>   taskFunc;
    std::thread             thread;
    std::mutex              mutex;
    std::condition_variable cv;

    void threadLoop();
#endif

public:
    /// Instantiate a new search thread.
    /// @param id ID of the new search thread, starting from 0 for main thread.
    /// @param bindGroup Whether to bind this thread to a NUMA group.
    explicit SearchThread(ThreadPool &threadPool, uint32_t id, bool bindGroup);
    /// Destory this search thread. Search must be stopped before entering.
    virtual ~SearchThread();
    /// Clear the thread state between two search.
    virtual void clear();
    /// Search entry point. After entering main thread search function, all threads are
    /// waked up by the main thread, then this function is called to start the search.
    virtual void search();
    /// Setup the board instance in this thread, and update the evaluator.
    virtual void setBoardAndEvaluator(const Board &board);
    /// Return if this thread is the main thread.
    bool isMainThread() const { return id == 0; }

    /// Get the search data as a specific type.
    template <typename SearchDataType>
    SearchDataType *searchDataAs() const;

    /// Get the shared search options.
    SearchOptions &options() const;

    /// Reference to the thread pool that this thread belongs to.
    ThreadPool &threads;

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
    std::atomic<uint64_t> numNodes;  /// Nodes count searched by this thread
    int                   selDepth;  /// Maximum depth reached by this thread
};

/// MainSearchThread class is the master thread in the Lazy SMP algorithm.
/// It also controls state needed in iterative deepening and time control.
struct MainSearchThread : public SearchThread
{
    MainSearchThread(ThreadPool &threadPool, bool bindGroup)
        : SearchThread::SearchThread(threadPool, 0, bindGroup)
    {}

    /// Clear the main thread state between two search.
    void clear() override;
    /// Main thread search entry point. After threadpool have finished preparation,
    /// this function is called. It is responsible for starting all other threads.
    void search() override;
    /// Check exit condition (time/nodes) and set ThreadPool's terminate flag.
    /// @return True if we should stop the search now.
    void checkExit();
    /// Mark pondering available for the last finished searching.
    void markPonderingAvailable();
    /// Start all non-main search threads. This function should be called in
    /// main thread to start searching routine in all other worker threads.
    void startOtherThreads();

    /// Current search options
    SearchOptions  searchOptions;
    SearchOptions &options() { return searchOptions; }

    /// Action type of the search result
    ActionType resultAction;
    /// Searched best move result
    Pos bestMove;
    /// Previous searched best move of last ply (used in selfplay)
    Pos previousPlyBestMove;
    /// Calls count before exit condition check
    uint32_t callsCnt;
    /// Should we start pondering after finishing this move?
    std::atomic_bool startPonderAfterThinking;
    /// Is in pondering mode?
    std::atomic_bool inPonder;
};

/// ThreadPool class is the container for all search threads. The first
/// thread in the threadpool is assumed to be the main thread. ThreadPool
/// also controls thread-related stuffs such as init, launching, stop and
/// collect various accumalated statistics.
class ThreadPool : public std::vector<std::unique_ptr<SearchThread>>
{
public:
    using EvaluatorMaker = std::unique_ptr<Evaluation::Evaluator>(int boardSize, Rule rule);

private:
    friend struct SearchThread;
    friend struct MainSearchThread;

    std::atomic_bool                     terminate;
    std::function<EvaluatorMaker>        evaluatorMaker;
    std::unique_ptr<Searcher>            searcherPtr;
    std::unique_ptr<Database::DBStorage> dbStoragePtr;

    template <typename T>
    T sum(std::atomic<T> SearchThread::*member, T init = T(0)) const
    {
        T sum = init;
        for (const auto &th : *this)
            sum += (th.get()->*member).load(std::memory_order_relaxed);
        return sum;
    }

public:
    /// Wait for all search threads to finish their current works.
    /// @param includingMainThread Whether to wait for main thread. Must be
    /// set to false when called inside the main thread to avoid deadlock.
    void waitForIdle(bool includingMainThread = true);
    /// Destroy all old threads and creates requested amount of threads.
    /// @note New threads will immediately go to sleep in threadLoop().
    void setNumThreads(size_t numThreads);
    /// Setup current searcher to a search algorithm.
    /// @param searcher The unique ptr to a search, must not be nullptr.
    void setupSearcher(std::unique_ptr<Searcher> searcher);
    /// @brief Setup a database storage instance to be used for searching.
    /// @param dbStorage The unique ptr to a dbStorage instance,
    ///     can be nullptr which means disable all usage of database.
    void setupDatabase(std::unique_ptr<Database::DBStorage> dbStorage);
    /// Setup evaluator maker for future evaluator creation.
    void setupEvaluator(std::function<EvaluatorMaker> evaluatorMaker);
    /// Start multi-threaded thinking for the given position.
    /// @param board The position to start searching.
    /// @param options Options of this search.
    /// @param inPonder If true, it is considered as pondering mode. No message will be shown.
    /// @note This is a non-blocking function. It returns immediately after starting all threads.
    void startThinking(const Board &board, const SearchOptions &options, bool inPonder = false);
    /// Notify all threads to stop thinking immediately.
    void stopThinking() { terminate.store(true, std::memory_order_relaxed); }
    /// Clear all threads state, searcher state and thread pool state for a new game.
    void clear(bool clearAllMemory);

    MainSearchThread    *main() const { return static_cast<MainSearchThread *>(front().get()); }
    Searcher            *searcher() const { return searcherPtr.get(); }
    Database::DBStorage *dbStorage() const { return dbStoragePtr.get(); }
    bool                 isTerminating() const { return terminate.load(std::memory_order_relaxed); }
    uint64_t             nodesSearched() const { return sum(&SearchThread::numNodes); }

    ThreadPool();
    ~ThreadPool();
};

template <typename SearchDataType>
inline SearchDataType *SearchThread::searchDataAs() const
{
    return static_cast<SearchDataType *>(searchData.get());
}

inline SearchOptions &SearchThread::options() const
{
    assert(threads.main());
    return threads.main()->searchOptions;
}

extern ThreadPool Threads;

}  // namespace Search
