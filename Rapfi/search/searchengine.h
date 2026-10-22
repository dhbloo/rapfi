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
#include "../eval/evalconfig.h"
#include "searchcommon.h"
#include "searchoutput.h"
#include "timecontrol.h"

#include <atomic>
#include <functional>
#include <memory>
#include <vector>

class Board;  // forward declaration

namespace Database {
class DBStorage;           // forward declaration
enum class OverwriteRule;  // forward declaration (opaque)
}
namespace Evaluation {
class Evaluator;  // forward declaration
}
#ifdef POLICY_TRAINING
namespace Tuning {
class PolicyTraceSession;
}
#endif

namespace Search {

class Searcher;      // forward declaration
class SearchThread;  // forward declaration
class SearchEngine;  // forward declaration

/// Database access parameters captured once per search from DatabaseCfg, so the
/// search node loop reads stable values without reaching into the config hub.
/// (The overwrite biases are not part of the snapshot: DBClient reads them live
/// via checkOverwrite's default overload.)
struct DatabaseSearchParams
{
    bool readonlyMode;                   // DatabaseCfg.search.readonlyMode
    bool mandatoryParentWrite;           // DatabaseCfg.search.mandatoryParentWrite
    int  queryPly;                       // DatabaseCfg.search.queryPly
    int  queryPVIterPerPlyIncrement;     // DatabaseCfg.search.queryPVIterPerPlyIncrement
    int  queryNonPVIterPerPlyIncrement;  // DatabaseCfg.search.queryNonPVIterPerPlyIncrement
    int  queryResultDepthBoundBias;      // DatabaseCfg.search.queryResultDepthBoundBias
    int  mateWritePly;                   // DatabaseCfg.search.mateWritePly
    int  mateWriteMinStep;               // DatabaseCfg.search.mateWriteMinStep
    int  mateWriteMinDepthExact;         // DatabaseCfg.search.mateWriteMinDepthExact
    int  mateWriteMinDepthNonExact;      // DatabaseCfg.search.mateWriteMinDepthNonExact
    int  pvWritePly;                     // DatabaseCfg.search.pvWritePly
    int  pvWriteMinDepth;                // DatabaseCfg.search.pvWriteMinDepth
    int  nonPVWritePly;                  // DatabaseCfg.search.nonPVWritePly
    int  nonPVWriteMinDepth;             // DatabaseCfg.search.nonPVWriteMinDepth
    int  writeValueRange;                // DatabaseCfg.search.writeValueRange
    int  exactOverwritePly;              // DatabaseCfg.search.exactOverwritePly
    int  nonExactOverwritePly;           // DatabaseCfg.search.nonExactOverwritePly
    Database::OverwriteRule overwriteRule;  // DatabaseCfg.search.overwriteRule

    /// Capture the current DatabaseCfg.search knob values.
    static DatabaseSearchParams captureFromConfig();
};

/// SearchContext holds all state scoped to ONE search request: inputs copied at
/// startThinking, shared per-search state, and the result outputs. One instance
/// is owned by the search engine and reset per search. (Per-game state lives on
/// the Searcher implementations; per-thread state on SearchThread.)
struct SearchContext
{
    // --- Inputs (set by reset()) ---
    /// Current search options. Non-search operations (e.g. gomocup TRACE
    /// handlers) may assign this directly without a reset().
    SearchOptions options;
    /// Is in pondering mode? (read by the protocol thread while search runs)
    std::atomic_bool inPonder;

    // --- Shared per-search state ---
    /// Time controller of this search (init'd once per search)
    TimeControl timectl;
    /// Printer for all search messages (main search thread only)
    SearchPrinter printer;
    /// Calls count before exit condition check (main search thread only)
    uint32_t callsCnt;
    /// Should we start pondering after finishing this move?
    /// (set during search, read by the protocol thread)
    std::atomic_bool startPonderAfterThinking;
    /// Database access parameters captured once per search from Config
    DatabaseSearchParams dbParams;

    // --- Outputs ---
    /// Action type of the search result
    ActionType resultAction;
    /// Searched best move result
    Pos bestMove;

    /// Reset the context for a new search (the startThinking path).
    void reset(const SearchOptions &opts, bool ponder);
    /// Check exit condition (time/nodes) and set the engine's terminate flag.
    /// Main search thread only; throttled by callsCnt.
    void checkExit(uint32_t elapsedCalls = 1);
    /// Mark pondering available for the last finished searching.
    void markPonderingAvailable();

private:
    friend class SearchEngine;
    /// Back-reference to the owning search engine (set at engine construction).
    SearchEngine *engine = nullptr;
};

/// SearchEngine is the engine facade: it owns the search threads, the engine
/// services (searcher algorithm, database storage, evaluator factory) and the
/// per-search context, and orchestrates search sessions. The first thread is
/// the main search thread (id == 0) — "main" is a role, not a type: it drives
/// searchMain and owns time control duties, while workers only run the search
/// loop.
class SearchEngine
{
public:
    /// Type of the function that creates an evaluator instance.
    using EvaluatorMaker = Evaluation::EvaluatorMakerFunc;

private:
    friend class SearchThread;

    /// Run one search session on the main search thread: handle trivial cases
    /// needing no real search, start the session clock, run the searcher's
    /// main routine, and finalize the result.
    void runSearch(SearchThread &main);
    /// Handle searches that need no real search (opening book hit, empty root
    /// move list, immediate win). @return True if a result was produced.
    bool tryTrivialBestmove(SearchThread &main);
    /// Common result finalization: record best move and decide the result
    /// action (skipped entirely while pondering).
    void finalizeResult(SearchThread &main, const RootMove &finalMove);

    std::vector<std::unique_ptr<SearchThread>> threads_;
    std::atomic_bool                           terminate;
    EvaluatorMaker                             evaluatorMaker;
    std::unique_ptr<Searcher>                  searcherPtr;
    std::unique_ptr<Database::DBStorage>       dbStoragePtr;
#ifdef POLICY_TRAINING
    Tuning::PolicyTraceSession *policyTraceSessionPtr = nullptr;
#endif

public:
    /// The per-search context of this engine, reset at each startThinking.
    SearchContext ctx;

    /// Wait for (other) search threads to finish their current works.
    /// @note When called inside the main thread, it will only wait for other
    ///     threads to finish their current works, excluding the main thread itself.
    void waitForIdle();
    /// Destroy all old threads and creates requested amount of threads.
    /// @param numThreads The number of threads to create.
    /// @note New threads will immediately go to sleep in threadLoop().
    ///     This must never be called in the worker threads.
    void setNumThreads(size_t numThreads);
    /// Setup current searcher to a search algorithm.
    /// @param searcher The unique ptr to a search, must not be nullptr.
    void setupSearcher(std::unique_ptr<Searcher> searcher);
    /// @brief Setup a database storage instance to be used for searching.
    /// @param dbStorage The unique ptr to a dbStorage instance,
    ///     can be nullptr which means disable all usage of database.
    void setupDatabase(std::unique_ptr<Database::DBStorage> dbStorage);
    /// Setup evaluator maker for future evaluator creation.
    void setupEvaluator(EvaluatorMaker evaluatorMaker);
    /// Run a custom task on all threads and wait for them to finish.
    /// @param task The custom task to run in each thread.
    /// @param includeSelf If true, the task also runs in the calling thread
    ///     (passing the main thread); requires being called from the main
    ///     search thread's context. In single-thread builds this degenerates
    ///     to running the task synchronously iff includeSelf.
    void runOnAllThreads(std::function<void(SearchThread &)> task, bool includeSelf);
    /// Start multi-threaded thinking for the given position.
    /// @param board The position to start searching.
    /// @param options Options of this search.
    /// @param inPonder If true, it is considered as pondering mode. No message will be shown.
    /// @param onStop Function to be called (in main thread) when search is finished or interrupted.
    /// @note This is a non-blocking function in multi-threaded builds: it returns
    ///     right after starting the main thread. In single-thread builds it runs
    ///     the whole search synchronously before returning.
    void startThinking(const Board          &board,
                       const SearchOptions  &options,
                       bool                  inPonder = false,
                       std::function<void()> onStop   = nullptr);
    /// Notify all threads to stop thinking immediately.
    void stopThinking() { terminate.store(true, std::memory_order_relaxed); }
    /// Clear all threads state, searcher state and engine state for a new game.
    void clear(bool clearAllMemory);

    // Thread container accessors (composition over the private thread vector)
    auto   begin() { return threads_.begin(); }
    auto   end() { return threads_.end(); }
    auto   begin() const { return threads_.begin(); }
    auto   end() const { return threads_.end(); }
    size_t size() const { return threads_.size(); }
    bool   empty() const { return threads_.empty(); }
    std::unique_ptr<SearchThread>       &operator[](size_t i) { return threads_[i]; }
    const std::unique_ptr<SearchThread> &operator[](size_t i) const { return threads_[i]; }

    SearchThread        *main() const { return threads_.front().get(); }
    Searcher            *searcher() const { return searcherPtr.get(); }
    Database::DBStorage *dbStorage() const { return dbStoragePtr.get(); }
    bool                 isTerminating() const { return terminate.load(std::memory_order_relaxed); }
#ifdef POLICY_TRAINING
    void setPolicyTraceSession(Tuning::PolicyTraceSession *session)
    {
        waitForIdle();
        policyTraceSessionPtr = session;
    }
    Tuning::PolicyTraceSession *policyTraceSession() const { return policyTraceSessionPtr; }
    bool                        hasEvaluatorMaker() const { return bool(evaluatorMaker); }
#endif
    /// Sum of nodes searched by all threads. (Defined in searchthread.h, which
    /// completes the SearchThread type.)
    uint64_t nodesSearched() const;

    SearchEngine();
    ~SearchEngine();
};

extern SearchEngine Engine;

}  // namespace Search
