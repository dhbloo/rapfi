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
#include "ab/searcher.h"
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
    , searching(false)
    , exit(false)
#ifdef MULTI_THREADING
    , thread(&SearchThread::threadLoop, this, bindGroup)
#endif
    , threads(threadPool)
{
    waitForIdle();
}

SearchThread::~SearchThread()
{
    assert(!searching);
    exit = true;

#ifdef MULTI_THREADING
    startSearching();
    thread.join();
#endif
}

void SearchThread::startSearching()
{
#ifdef MULTI_THREADING
    std::lock_guard<std::mutex> lock(mutex);
    searching = true;
    cv.notify_one();
#else
    search();
#endif
}

void SearchThread::waitForIdle()
{
#ifdef MULTI_THREADING
    if (!searching)
        return;

    std::unique_lock<std::mutex> lock(mutex);
    cv.wait(lock, [&] { return !searching; });
#endif
}

#ifdef MULTI_THREADING
void SearchThread::threadLoop(bool bindGroup)
{
    // If OS already scheduled us on a different group than 0 then don't overwrite
    // the choice, eventually we are one of many one-threaded processes running on
    // some Windows NUMA hardware, for instance in fishtest. To make it simple,
    // just check if running threads are below a threshold, in this case all this
    // NUMA machinery is not needed.
    if (bindGroup)
        WinProcGroup::bindThisThread(id);

    while (true) {
        std::unique_lock<std::mutex> lock(mutex);
        searching = false;
        cv.notify_all();
        cv.wait(lock, [&] { return searching; });

        if (exit)
            return;

        lock.unlock();
        search();
    }
}
#endif

void SearchThread::clear()
{
    if (searchData)
        searchData->clearData();
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

void SearchThread::search()
{
    assert(threads.searcher());
    threads.searcher()->search(*this);
}

void MainSearchThread::search()
{
    assert(threads.searcher());
    threads.searcher()->searchMain(*this);
}

void MainSearchThread::checkExit()
{
    // We only check exit condition after a number of calls.
    // This is to avoid expensive calculation in timeup condition checking.
    if (callsCnt-- > 0)
        return;

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

void MainSearchThread::startOtherThreads()
{
    for (size_t i = 1; i < threads.size(); i++)
        threads[i]->startSearching();
}

void ThreadPool::waitForIdle(bool includingMainThread)
{
    for (size_t i = (includingMainThread ? 0 : 1); i < size(); i++)
        (*this)[i]->waitForIdle();
}

void ThreadPool::setNumThreads(size_t numThreads)
{
    // Destroy all threads first
    if (!empty()) {
        waitForIdle();

        while (!empty())
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

        if (searcher()) {
            for (const auto &th : *this)
                th->searchData = searcher()->makeSearchData(*th);
        }
    }
}

void ThreadPool::setupSearcher(std::unique_ptr<Searcher> newSearcher)
{
    assert(newSearcher);
    searcherPtr = std::move(newSearcher);

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

void ThreadPool::updateEvaluator(const Board &board, bool enabled)
{
    if (empty())
        return;

    waitForIdle();

    // Reset board in all threads to null
    for (const auto &th : *this) {
        th->board.reset();
    }

    // Set evaluator in all threads
    if (!evaluatorMaker || !enabled) {
        for (const auto &th : *this)
            th->evaluator.reset();
    }
    else {
        // Clear all loaded evaluator that does not match
        if (main()->evaluator
            && (main()->evaluator->boardSize != board.size()
                || main()->evaluator->rule != main()->searchOptions.rule)) {
            for (const auto &th : *this)
                th->evaluator.reset();
        }

        for (const auto &th : *this) {
            if (!th->evaluator || th->evaluator->boardSize != board.size()
                || th->evaluator->rule != main()->searchOptions.rule) {
                th->evaluator = evaluatorMaker(board.size(), main()->searchOptions.rule);

                // Failed to create evaluator, remove all evaluator created
                if (!th->evaluator) {
                    for (const auto &t : *this)
                        t->evaluator.reset();
                    break;
                }
            }
        }
    }
}

void ThreadPool::startThinking(const Board &board, const SearchOptions &options, bool inPonder)
{
    assert(size() > 0);
    assert(searcher());
    waitForIdle();

    // Clear and init all threads state
    for (const auto &th : *this)
        th->clear();
    main()->inPonder      = inPonder;
    main()->searchOptions = options;
    terminate             = false;

    // Create dbClient for each thread
    if (dbStorage()) {
        for (size_t i = 0; i < size(); i++) {
            if (!(*this)[i]->dbClient)
                (*this)[i]->dbClient =
                    std::make_unique<Database::DBClient>(*dbStorage(),
                                                         Database::RECORD_MASK_LVDB,
                                                         Config::DatabaseCacheSize,
                                                         Config::DatabaseRecordCacheSize);
        }
    }

    // Create or update evaluator if needed
    updateEvaluator(board);

    // Clone input board to main thread
    main()->board = std::make_unique<Board>(board, main());

    // Expand board candidate if needed
    Opening::expandCandidate(*main()->board);

    // Generate root moves for main search thread
    MovePicker movePicker(options.rule, *main()->board, MovePicker::ExtraArgs<MovePicker::ROOT> {});
    while (Pos m = movePicker()) {
        if (!std::count(options.blockMoves.begin(), options.blockMoves.end(), m)) {
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
        auto &th      = (*this)[i];
        th->board     = std::make_unique<Board>(*main()->board, th.get());
        th->rootMoves = main()->rootMoves;
        if (options.balanceMode == Search::SearchOptions::BALANCE_TWO)
            th->balance2Moves = main()->balance2Moves;
    }

    main()->startSearching();
}

void ThreadPool::clear(bool clearAllMemory)
{
    waitForIdle();

    if (searcher())
        searcher()->clear(*this, clearAllMemory);

    for (const auto &th : *this)
        th->clear();

    terminate = false;
}

ThreadPool::ThreadPool()
{
    // Set default searcher to Alpha-beta searcher
    setupSearcher(std::make_unique<AB::ABSearcher>());
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
