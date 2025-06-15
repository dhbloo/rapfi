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
     , numaId(Numa::DefaultNumaNodeId)
     , running(false)
     , exit(false)
     , threads(threadPool)
 {
     // Nothing involving threads or synchronization happens here yet.
 }
 
 void SearchThread::init(bool bindGroup) {
     #ifdef MULTI_THREADING
         thread = std::thread(&SearchThread::threadLoop, this);
     #endif
     
         // Set thread affinity if requested
         if (bindGroup) {
             runTask([](SearchThread& th) {
                 th.numaId = Numa::bindThisThread(th.id);
             });
         }
     
         // Wait until the thread signals it's idle
         waitForIdle();
     
         // Allocate thread-local search data
         searchData = threads.searcher()->makeSearchData(*this);
     }
 
 SearchThread::~SearchThread() {
     #ifdef MULTI_THREADING
         {
             std::lock_guard<std::mutex> lock(mutex);
             exit = true;
             running = true;
             cv.notify_all();
         }
     
         if (thread.joinable())
             thread.join();
     #endif
     }
 
 void SearchThread::runTask(std::function<void(SearchThread &)> task)
 {
 #ifdef MULTI_THREADING
     if (std::this_thread::get_id() == thread.get_id()) {
         // We *are* the worker ⇒ enqueue "tail task" without waiting.
         std::lock_guard<std::mutex> lock(mutex);
         // at this point running is still true, we simply replace the functor
         taskFunc = std::move(task);
     }
     else {
         std::unique_lock<std::mutex> lock(mutex);
         cv.wait(lock, [&] { return !running; });
         taskFunc = std::move(task);
         running  = true;
         lock.unlock();
         cv.notify_one();
     }
 #else
     if (task)
         task(*this);
 #endif
 }
 
 void SearchThread::waitForIdle()
 {
 #ifdef MULTI_THREADING
     // Check deadlock if we are already in the worker thread
     assert(std::this_thread::get_id() != thread.get_id());
 
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
             if (!taskFunc) {
                 running = false;
                 cv.notify_all();
                 cv.wait(lock, [&] { return running || exit; });
             }
 
             if (exit)
                 return;
 
             std::swap(task, taskFunc);
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
 
     // Setup dbClient for each thread
     if (threads.dbStorage() && (!dbClient || &dbClient->getStorage() != threads.dbStorage())) {
         dbClient = std::make_unique<Database::DBClient>(*threads.dbStorage(),
                                                         Database::RECORD_MASK_LVDB,
                                                         Config::DatabaseCacheSize,
                                                         Config::DatabaseRecordCacheSize);
     }
 }
 
 void MainSearchThread::clear()
 {
     SearchThread::clear();
     resultAction             = ActionType::Move;
     bestMove                 = Pos::NONE;
     callsCnt                 = 0;
     startPonderAfterThinking = false;
     inPonder                 = false;
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
             evaluator = threads.evaluatorMaker(boardSize, rule, numaId);
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
 
 void MainSearchThread::runCustomTaskAndWait(std::function<void(SearchThread &)> task,
                                             bool                                includeSelf)
 {
     if (!task)
         return;
 
     // Run task in non-main threads
     for (size_t i = 1; i < threads.size(); i++)
         threads[i]->runTask(task);
 
     // Run task in main thread
     if (includeSelf)
         task(*this);
 
     // Wait for all other threads to finish
     threads.waitForIdle();
 }
 
 void ThreadPool::waitForIdle()
 {
     // Iterate all other threads and wait for them to finish
     for (auto &th : *this)
         if (th->thread.get_id() != std::this_thread::get_id())
             th->waitForIdle();
 }
 
 void ThreadPool::setNumThreads(size_t numThreads)
 {
     // Destroy all threads first (which will also wait for them to be idle)
     while (!empty())
         pop_back();  // std::unique_ptr will automatically destroy thread
 
     // Create requested amount of threads
     if (numThreads > 0) {
         bool bindGroup = numThreads > Numa::BindGroupThreshold;
 
         // Make sure the first thread created is MainSearchThread
         push_back(std::make_unique<MainSearchThread>(*this, bindGroup));
         (*this)[0]->init(bindGroup);
 
 #ifdef MULTI_THREADING
         while (size() < numThreads) {
             push_back(std::make_unique<SearchThread>(*this, (uint32_t)size(), bindGroup));
             (*this).back()->init(bindGroup);
         }
 #endif
     }
 }
 
 void ThreadPool::setupSearcher(std::unique_ptr<Searcher> newSearcher)
 {
     waitForIdle();
 
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
     if (!empty()) {
         waitForIdle();
         for (const auto &th : *this) {
             th->board.reset();
             th->evaluator.reset();
         }
     }
 
     evaluatorMaker = maker;
 }
 
 void ThreadPool::startThinking(const Board          &board,
                                const SearchOptions  &options,
                                bool                  inPonder,
                                std::function<void()> onStop)
 {
     assert(size() > 0);
     assert(searcher());
 
     // If we are already thinking, wait for it first
     waitForIdle();
     terminate = false;
 
     // Clean up main thread state and copy options
     main()->clear();
     main()->searchOptions = options;
     main()->inPonder      = inPonder;
 
     // Clone the input board to main thread and update evaluator
     main()->setBoardAndEvaluator(board);
 
     // Expand board candidate if needed
     Opening::expandCandidate(*main()->board);
 
     auto addMoveToRootMoves = [this](Pos m) {
         // Ignore blocked moves
         if (std::count(main()->searchOptions.blockMoves.begin(),
                        main()->searchOptions.blockMoves.end(),
                        m))
             return;
 
         if (main()->searchOptions.balanceMode == Search::SearchOptions::BALANCE_TWO) {
             // Use candidates before first move
             std::unordered_set<Pos> cands;
             FOR_EVERY_CAND_POS(main()->board, pos)
             {
                 cands.insert(pos);
             }
             main()->board->move(main()->searchOptions.rule, m);
             // Generate second move for balance2
             MovePicker movePicker2(main()->searchOptions.rule,
                                    *main()->board,
                                    MovePicker::ExtraArgs<MovePicker::ROOT> {});
             while (Pos m2 = movePicker2()) {
                 if (cands.find(m2) != cands.end()) {
                     Search::Balance2Move bm {m, m2};
                     main()->rootMoves.emplace_back(bm);
                     main()->balance2Moves[bm] = main()->rootMoves.size() - 1;
                 }
             }
             main()->board->undo(main()->searchOptions.rule);
         }
         else {
             main()->rootMoves.emplace_back(m);
         }
     };
 
     // Generate root moves for main search thread
     MovePicker movePicker(options.rule, *main()->board, MovePicker::ExtraArgs<MovePicker::ROOT> {});
     while (Pos m = movePicker()) {
         addMoveToRootMoves(m);
     }
 
     // If all legal moves are blocked, we select all candidate moves as root moves
     if (main()->rootMoves.empty() && options.blockMoves.size() > 0) {
         std::unordered_set<Pos> cands;
         FOR_EVERY_CAND_POS(main()->board, pos)
         {
             cands.insert(pos);
         }
 
         for (const auto &m : cands) {
             addMoveToRootMoves(m);
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
 
     // Launch a small task to clear threads state and copy state from main thread
     main()->runCustomTaskAndWait(
         [mainTh = main()](SearchThread &th) {
             th.clear();
             th.setBoardAndEvaluator(*mainTh->board);
             th.rootMoves     = mainTh->rootMoves;
             th.balance2Moves = mainTh->balance2Moves;
         },
         false);
 
     // Start the main search thread
     main()->runTask([this, searcher = searcher(), onStop = std::move(onStop)](SearchThread &th) {
         searcher->searchMain(static_cast<MainSearchThread &>(th));
         if (onStop)  // If onStop is set, queue a tail task to call it
             main()->runTask([onStop = std::move(onStop)](SearchThread &th) { onStop(); });
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
 