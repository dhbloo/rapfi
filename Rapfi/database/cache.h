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

#include <functional>
#include <list>
#include <optional>
#include <unordered_map>
#include <utility>

namespace Database {

/// A simple lru cache implementation with std::map and std::list.
template <typename KeyT, typename ValueT>
class LRUCacheTable
{
public:
    using KVType        = std::pair<KeyT, ValueT>;
    using ListIteratorT = typename std::list<KVType>::iterator;

    struct NullPopCollector
    {
        void operator()(KVType &&) {}
    };

    LRUCacheTable(size_t maxCap) : maxCapacity(maxCap) {}

    /// Put a new item into cache table.
    /// @return Return the reference to the inserted value.
    template <typename PopCollector = NullPopCollector>
    ValueT &put(KeyT key, const ValueT &value, PopCollector collector = {})
    {
        if (auto it = cacheItemMap.find(key); it != cacheItemMap.end()) {
            it->second->second = value;
            cacheItemList.splice(cacheItemList.begin(), cacheItemList, it->second);
        }
        else if (cacheItemList.size() >= maxCapacity) {
            cacheItemMap.erase(cacheItemList.back().first);
            collector(std::move(cacheItemList.back()));
            cacheItemList.splice(cacheItemList.begin(),
                                 cacheItemList,
                                 std::prev(cacheItemList.end()));
            cacheItemList.front() = std::make_pair(key, value);
            cacheItemMap[key]     = cacheItemList.begin();
        }
        else {
            cacheItemList.push_front(std::make_pair(key, value));
            cacheItemMap[key] = cacheItemList.begin();
        }

        return cacheItemList.front().second;
    }

    /// Try to get an item from cache table.
    /// @return Pointer to the cached value, or nullptr if not found.
    ValueT *get(KeyT key)
    {
        auto it = cacheItemMap.find(key);
        if (it != cacheItemMap.end()) {
            // Refresh item age
            ListIteratorT listIt = it->second;
            cacheItemList.splice(cacheItemList.begin(), cacheItemList, listIt);
            return &listIt->second;
        }

        return nullptr;
    }

    /// Try to remove an item from cache table.
    /// @return Whether the item is removed.
    bool remove(KeyT key)
    {
        auto it = cacheItemMap.find(key);
        if (it != cacheItemMap.end()) {
            cacheItemList.erase(it->second);
            cacheItemMap.erase(it);
            return true;
        }

        return false;
    }

    /// Clears all cached entries in the table.
    void clear()
    {
        cacheItemMap.clear();
        cacheItemList.clear();
    }

    /// Checks if a key exists in the table.
    bool exists(KeyT key) const { return cacheItemMap.find(key) != cacheItemMap.end(); }

    /// Return the number of items in the cache table.
    size_t size() const { return cacheItemMap.size(); }

    /// Return the total capacity of the cache table.
    size_t capacity() const { return maxCapacity; }

    /// Adjust the total capacity of the cache table.
    void setCapacity(size_t newCapacity)
    {
        maxCapacity = newCapacity;
        while (cacheItemMap.size() > maxCapacity) {
            cacheItemMap.erase(cacheItemList.back().first);
            cacheItemList.pop_back();
        }
    }

    /// Give a list view of all cached items.
    ListIteratorT begin() { return cacheItemList.begin(); }
    ListIteratorT end() { return cacheItemList.end(); }

private:
    std::list<KVType>                       cacheItemList;
    std::unordered_map<KeyT, ListIteratorT> cacheItemMap;
    size_t                                  maxCapacity;
};

}  // namespace Database
