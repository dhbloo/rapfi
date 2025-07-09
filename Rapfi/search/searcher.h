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

#include <memory>

namespace Search {

class SearchThread;       // forward declaration
class MainSearchThread;  // forward declaration
class ThreadPool;         // forward declaration

struct SearchData
{
    virtual ~SearchData() = default;

    /// Clear the states of search data for a new search.
    virtual void clearData(SearchThread &th) = 0;
};

/// Searcher is the base class for implementation of all search algorithms.
class Searcher
{
public:
    virtual ~Searcher() = default;

    /// Creates a instance of search data for one search thread.
    virtual std::unique_ptr<SearchData> makeSearchData(SearchThread &th) = 0;

    /// Set the memory size limit of the search.
    /// @param memorySizeKB Maximum memory size in KiB. Should be greater than zero.
    virtual void setMemoryLimit(size_t memorySizeKB) = 0;
    /// Get the current memory size limit of the search.
    /// @return Maximum memory size in KiB.
    virtual size_t getMemoryLimit() const = 0;

    /// Clear all searcher states between different games.
    /// @param pool The thread pool that holds all the search threads.
    /// @param clearAllMemory Whether to clear all memory (TT, etc.)
    /// @note All threads in pool is guaranteed to be created by this searcher.
    virtual void clear(ThreadPool &pool, bool clearAllMemory) = 0;

    /// Main-thread search entry point. After threadpool have finished preparation,
    /// this function is called. It is responsible for starting all other threads.
    virtual void searchMain(MainSearchThread &th) = 0;
    /// Worker-thread search entry point. This function is called by each worker thread.
    virtual void search(SearchThread &th) = 0;

    /// Checks if a search reaches timeup condition.
    /// @return True if time is up, otherwise false.
    virtual bool checkTimeupCondition() = 0;
};

}  // namespace Search
