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

#include "searchthread.h"

#include "../core/iohelper.h"
#include "../core/platform.h"
#include "../game/board.h"
#include "movepick.h"
#include "opening.h"
#include "searcher.h"

#include <algorithm>
#include <unordered_set>

namespace Search {

/// Global search thread pool
ThreadPool Threads;

SearchThread::SearchThread(ThreadPool &threadPool, uint32_t id, bool bindGroup)
    : id(id)
    , running(false)
    , exit(false)
#ifdef MULTI_THREADING
    , thread(&SearchThread::threadLoop, this)
#endif
    , threads(threadPool)
{
    // Create search data for this thread
    searchData = threads.searcher()->makeSearchData(*this);

    // Set thread affinity to a specific group if needed
    if (bindGroup)
        runTask([](SearchThread &th) {
            // If OS already scheduled us on a different group than 0 then don't overwrite
            // the choice, eventually we are one of many one-threaded processes running on
            // some Windows NUMA hardware, for instance in fishtest. To make it simple,
            // just check if running threads are below a threshold, in this case all this
            // NUMA machinery is not needed.
            WinProcGroup::bindThisThread(th.id);
        });

    waitForIdle();
}

SearchThread::~SearchThread()
{
    exit = true;

#ifdef MULTI_THREADING
    runTask(nullptr);
    thread.join();
#endif
}

void SearchThread::runTask(std::function<void(SearchThread &)> task)
{
#ifdef MULTI_THREADING
    {
        std::unique_lock<std::mutex> lock(mutex);
        cv.wait(lock, [&] { return !running; });
        taskFunc = std::move(task);
        running  = true;
    }
    cv.notify_one();
#else
    if (task)
        task(*this);
#endif
}

void SearchThread::waitForIdle()
{
#ifdef MULTI_THREADING
    if (!running)
        return;

    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [&] { return !running; });
#endif
}

#ifdef MULTI_THREADING
void SearchThread::threadLoop()
{
    while (true) {
        std::function<void(SearchThread &)> task;

        {
            std::unique_lock<std::mutex> lock(mutex);
            running = false;
            cv.notify_all();
            cv.wait(lock, [&] { return running; });

            if (exit)
                return;

            task     = std::move(taskFunc);
            taskFunc = nullptr;
        }

        if (task)
            task(*this);
    }
}
#endif

void SearchThread::clear()
{
    if (searchData)
        searchData->clearData(*this);
    rootMoves.clear();
    balance2Moves.clear();
    numNodes = 0;
    selDepth = 0;
}

void MainSearchThread::clear()
{
    SearchThread::clear();
    resultAction = ActionType::Move;
    bestMove = previousPlyBestMove = Pos::NONE;
    callsCnt                       = 0;
    startPonderAfterThinking       = false;
    inPonder                       = false;
}

void SearchThread::setBoardAndEvaluator(const Board &board)
{
    // Reset board instance in this thread to be null
    this->board.reset();

    // Setup evaluator in this thread
    if (!threads.evaluatorMaker)
        evaluator.reset();
    else {
        const int  boardSize = board.size();
        const Rule rule      = threads.main()->searchOptions.rule;

        // Clear loaded evaluator that does not match
        if (evaluator && (evaluator->boardSize != boardSize || evaluator->rule != rule))
            evaluator.reset();

        if (!evaluator)
            evaluator = threads.evaluatorMaker(boardSize, rule);
    }

    // Clone the board (this will also sync the evaluator to the board state)
    this->board = std::make_unique<Board>(board, this);
}

void MainSearchThread::checkExit(uint32_t elapsedCalls)
{
    // We only check exit condition after a number of calls.
    // This is to avoid expensive calculation in timeup condition checking.
    if (callsCnt > elapsedCalls) {
        callsCnt -= elapsedCalls;
        return;
    }

    // Resets callsCnt
    if (searchOptions.maxNodes)
        callsCnt = std::min(1024U, uint32_t(searchOptions.maxNodes / 1024U));
    else
        callsCnt = 1024;

    // Do not stop searching in pondering mode
    if (inPonder.load(std::memory_order_relaxed))
        return;

    // Check if we have reached node/time limits
    if (searchOptions.maxNodes && threads.nodesSearched() >= searchOptions.maxNodes
        || searchOptions.timeLimit && threads.searcher()->checkTimeupCondition()) {
        markPonderingAvailable();
        threads.stopThinking();
    }
}

void MainSearchThread::markPonderingAvailable()
{
    if (searchOptions.pondering && !inPonder.load(std::memory_order_relaxed))
        startPonderAfterThinking.store(true, std::memory_order_relaxed);
}

void MainSearchThread::startSearchingAndWait()
{
    Searcher *searcher = threads.searcher();

    // Starts searching in non-main threads
    for (size_t i = 1; i < threads.size(); i++)
        threads[i]->runTask([searcher](SearchThread &th) { searcher->search(th); });

    // Starts main thread searching
    searcher->search(*this);

    // Stop all threads if not already stopped
    threads.stopThinking();

    // Wait for all other threads to stop searching
    threads.waitForIdle(false);
}

void MainSearchThread::runCustomTaskAndWait(std::function<void(SearchThread &)> task)
{
    if (!task)
        return;

    // Run task in non-main threads
    for (size_t i = 1; i < threads.size(); i++)
        threads[i]->runTask(task);

    // Run task in main thread
    task(*this);

    // Wait for all other threads to finish
    threads.waitForIdle(false);
}

void ThreadPool::waitForIdle(bool includingMainThread)
{
    for (size_t i = (includingMainThread ? 0 : 1); i < size(); i++)
        (*this)[i]->waitForIdle();
}

void ThreadPool::setNumThreads(size_t numThreads)
{
    // Destroy all threads first
    while (!empty()) {
        back()->waitForIdle();
        pop_back();  // std::unique_ptr will automatically destroy thread
    }

    // Create requested amount of threads
    if (numThreads > 0) {
        bool bindGroup = numThreads > 8;

        // Make sure the first thread created is MainSearchThread
        push_back(std::make_unique<MainSearchThread>(*this, bindGroup));

#ifdef MULTI_THREADING
        while (size() < numThreads)
            push_back(std::make_unique<SearchThread>(*this, (uint32_t)size(), bindGroup));
#endif
    }
}

void ThreadPool::setupSearcher(std::unique_ptr<Searcher> newSearcher)
{
    size_t memLimitKB = 0;
    if (searcher())
        memLimitKB = searcher()->getMemoryLimit();

    assert(newSearcher);
    searcherPtr = std::move(newSearcher);

    if (memLimitKB)
        searcher()->setMemoryLimit(memLimitKB);

    // Re-instantiate all threads
    setNumThreads(size());
}

void ThreadPool::setupDatabase(std::unique_ptr<Database::DBStorage> dbStorage)
{
    if (!empty()) {
        waitForIdle();
        for (const auto &th : *this)
            th->dbClient.reset();
    }

    dbStoragePtr = std::move(dbStorage);
}

void ThreadPool::setupEvaluator(std::function<EvaluatorMaker> maker)
{
    evaluatorMaker = maker;

    if (empty())
        return;

    waitForIdle();
    for (const auto &th : *this) {
        th->board.reset();
        th->evaluator.reset();
    }
}

void ThreadPool::startThinking(const Board &board, const SearchOptions &options, bool inPonder)
{
    assert(size() > 0);
    assert(searcher());

    // Clear and init all threads state
    for (size_t i = 1; i < size(); i++) {
        (*this)[i]->runTask([this](SearchThread &th) {
            th.clear();

            // Create dbClient for each thread
            if (dbStorage() && !th.dbClient)
                th.dbClient = std::make_unique<Database::DBClient>(*dbStorage(),
                                                                   Database::RECORD_MASK_LVDB,
                                                                   Config::DatabaseCacheSize,
                                                                   Config::DatabaseRecordCacheSize);
        });
    }

    main()->clear();
    main()->inPonder      = inPonder;
    main()->searchOptions = options;
    terminate             = false;

    // Clone the input board to main thread and update evaluator
    main()->setBoardAndEvaluator(board);

    // Expand board candidate if needed
    Opening::expandCandidate(*main()->board);

    // Generate root moves for main search thread
    MovePicker movePicker(options.rule, *main()->board, MovePicker::ExtraArgs<MovePicker::ROOT> {});
    while (Pos m = movePicker()) {
        // Ignore blocked moves
        if (std::count(options.blockMoves.begin(), options.blockMoves.end(), m))
            continue;

        if (options.balanceMode == Search::SearchOptions::BALANCE_TWO) {
            // Use candidates before first move
            std::unordered_set<Pos> cands;
            FOR_EVERY_CAND_POS(main()->board, pos)
            {
                cands.insert(pos);
            }

            main()->board->move(options.rule, m);

            // Generate second move for balance2
            MovePicker movePicker2(options.rule,
                                   *main()->board,
                                   MovePicker::ExtraArgs<MovePicker::ROOT> {});
            while (Pos m2 = movePicker2()) {
                if (cands.find(m2) != cands.end()) {
                    Search::Balance2Move bm {m, m2};
                    main()->rootMoves.emplace_back(bm);
                    main()->balance2Moves[bm] = main()->rootMoves.size() - 1;
                }
            }

            main()->board->undo(options.rule);
        }
        else {
            main()->rootMoves.emplace_back(m);
        }
    }

    // Filter root moves with symmetry (not for balance two)
    if (options.balanceMode != Search::SearchOptions::BALANCE_TWO) {
        std::vector<Pos> rootMoveList;
        for (const auto &rm : main()->rootMoves) {
            rootMoveList.push_back(rm.pv[0]);
        }

        Opening::filterSymmetryMoves(*main()->board, rootMoveList);

        if (rootMoveList.size() < main()->rootMoves.size()) {
            // Remove root moves that are not in the filtered list
            auto pred = [&rootMoveList](const Search::RootMove &rm) -> bool {
                return std::find(rootMoveList.begin(), rootMoveList.end(), rm.pv[0])
                       == rootMoveList.end();
            };
            main()->rootMoves.erase(
                std::remove_if(main()->rootMoves.begin(), main()->rootMoves.end(), pred),
                main()->rootMoves.end());
        }
    }

    // Copy board and root moves from main thread to other threads
    for (size_t i = 1; i < size(); i++) {
        (*this)[i]->runTask([this](SearchThread &th) {
            th.setBoardAndEvaluator(*main()->board);
            th.rootMoves = main()->rootMoves;
            if (main()->searchOptions.balanceMode == Search::SearchOptions::BALANCE_TWO)
                th.balance2Moves = main()->balance2Moves;
        });
    }

    main()->runTask([searcher = searcher()](SearchThread &th) {
        searcher->searchMain(static_cast<MainSearchThread &>(th));
    });
}

void ThreadPool::clear(bool clearAllMemory)
{
    if (empty())
        setNumThreads(Config::DefaultThreadNum);

    if (searcher())
        searcher()->clear(*this, clearAllMemory);
}

ThreadPool::ThreadPool()
{
    // Set default searcher
    setupSearcher(Config::createSearcher());
}

ThreadPool::~ThreadPool()
{
#ifdef MULTI_THREADING
    // Stop if there are still some threads thinking
    stopThinking();
#endif
    // Explicitly free all threads
    setNumThreads(0);
}

}  // namespace Search
