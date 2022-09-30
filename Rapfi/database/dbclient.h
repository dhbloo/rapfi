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

#include "../config.h"
#include "../core/utils.h"
#include "cache.h"
#include "dbstorage.h"
#include "dbtypes.h"

#include <memory>
#include <vector>

class Board;  // forward declaration

namespace Database {

/// Creates a DBKey from the given board and rule. This removes symmetry redundancy
/// of the board and generates the key with lowest representation.
DBKey constructDBKey(const Board &board, Rule rule);

/// Transform a DBKey to its smallest equivalent key.
void toSmallestDBKey(DBKey &key);

/// The rule to use when a overwrite might occur to a position.
enum class OverwriteRule {
    Disabled,               /// No overwrite is allowed (only null record gets overwritten)
    Always,                 /// Always overwrite
    BetterLabel,            /// Overwrite if new record has better label (more determined result)
    BetterValue,            /// Overwrite if new record has more valuable value
    BetterDepthBound,       /// Better label or better depthbound
    BetterValueDepthBound,  /// Better label or more valuable value or better depthbound
};

/// Checks whether new record satisfy the overwrite condition
/// Both oldRecord and newRecord should have valid label, value, deoth, bound.
bool checkOverwrite(const DBRecord &oldRecord,
                    const DBRecord &newRecord,
                    OverwriteRule   owRule,
                    int             exactBias      = Config::DatabaseOverwriteExactBias,
                    int             depthBoundBias = Config::DatabaseOverwriteDepthBoundBias);

/// DBClient class is the interface for querying and saving results of positions.
///
/// It exposes a set of lookup functions that takes the Board object as input, which is meant
/// to be used in search or other scenarios that use a board as key to query or save positional
/// results. A conversion from the Board instance to a transposition and symmetry invarant db
/// key is performed in order to lookup the database storage.
/// A database instance can be created from various storage instance, such as an in-memory map
/// or an external (or even remote) kvstore data source. This provides the best scalability from
/// tiny in-mem database to huge database that might take hundreds of gigabytes.
/// Note that all query/save operations are not thread-safe, so it's best to let all threads
/// have their own instance of a database class.
class DBClient
{
public:
    /// Create a database client on top of a database storage instance.
    /// @param recordMask The part of record to be queried or updated.
    DBClient(DBStorage   &storage,
             DBRecordMask recordMask,
             size_t       dbCacheSize       = 0,
             size_t       dbRecordCacheSize = 0);
    /// This will can sync() before destroying current database instance.
    ~DBClient();

    /// Query db record of the current position.
    /// @return Whether current position exists in the database.
    bool query(const Board &board, Rule rule, DBRecord &record);

    /// Query all existing children of the current position.
    /// @return The list of all children, in forms of (Pos, Record).
    void queryChildren(const Board                           &board,
                       Rule                                   rule,
                       std::vector<std::pair<Pos, DBRecord>> &childRecords);

    /// Save a new DBRecord of current position into the database.
    /// @param owRule The overwrite rule to use when a overwrite might occur.
    ///     This is only applicable if ((mask & RECORD_MASK_LVDB) == RECORD_MASK_LVDB),
    ///     otherwise, the overwrite is always performed.
    /// @return Whether the save is actually performed.
    bool save(const Board &board, Rule rule, const DBRecord &record, OverwriteRule owRule);

    /// Delete a DBRecord of current position from the database.
    void del(const Board &board, Rule rule);

    /// The deletion type of a node.
    enum class DelType {
        NoDelete,           /// Do not delete any node of this branch
        NoDeleteRecursive,  /// Do not delete this node, but check its children
        DeleteRecursive,    /// Delete all nodes of this branch
    };
    /// Delete all children DBRecords of current position from the database.
    /// @param deleteFilter If not null, only delete children that satisfy the filter.
    void delChildren(const Board                       &board,
                     Rule                               rule,
                     std::function<DelType(DBRecord &)> deleteFilter = nullptr);

    /// Sync all recently added records to the database backend.
    /// @param clearCache If true, all old cache will be cleared so records will be refetched.
    void sync(bool clearCache = true);

private:
    DBStorage   &storage;
    DBRecordMask mask;

    /// For speeding up frequent database operations on same db entries, we use a LRUCache
    /// to hold all recent visited DBKey and DBRecord. When a DBRecord is updated, it is
    /// marked as dirty and will be pushed to the dbStorage at the next sync or it is
    /// poped out from cache list by other newer entries.
    struct EntryCache
    {
        DBKey    key;
        DBRecord record;
        bool     dirty;

        EntryCache() = default;
    };
    LRUCacheTable<HashKey, EntryCache> dbCache;

    /// For read-only operations on db entries, we use a fast lookup table to reduce
    /// redundent database query. This table is index only by hash key.
    struct DBRecordCache
    {
        /// For record with null hash key, the db record is considered invalid.
        static constexpr HashKey NullKey = HashKey(-1);

        using KVType = std::pair<HashKey, DBRecord>;
        DBRecordCache(size_t size) : table(size)
        {
            assert(isPowerOfTwo(size));
            clear();
        }
        KVType &operator[](HashKey key) { return table[(uint32_t)key & (table.size() - 1)]; }
        void    clear()
        {
            for (auto &[k, v] : table)
                k = NullKey;
        }

    private:
        std::vector<KVType> table;
    } dbRecordCache;
};

}  // namespace Database
