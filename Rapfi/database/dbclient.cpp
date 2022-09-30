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

#include "dbclient.h"

#include "../game/board.h"

#include <algorithm>
#include <functional>
#include <mutex>
#include <set>
#include <thread>

namespace {

using namespace Database;

/// Compare two label to see which one is more valuable
/// @return 1 if lhs label is more valuable, -1 if rhs label is more valuable.
inline int labelCompare(DBLabel lhs, DBLabel rhs)
{
    if (lhs != rhs) {
        if (isDeterminedLabel(lhs))
            return 1;
        if (isDeterminedLabel(rhs))
            return -1;
    }
    return 0;
}

/// Get the rank of the depth-bound of a record
inline int
depthBoundRank(const DBRecord &record, int overwriteExactBias, int overwriteDepthBoundBias)
{
    return record.depth()
           + (overwriteExactBias - std::min(overwriteDepthBoundBias, 0))
                 * (record.bound() == BOUND_EXACT);
};

/// Get the rank of a mate (determined) record
inline int mateRank(const DBRecord &record)
{
    return std::abs((int)record.value) * 2 + (record.bound() == BOUND_EXACT);
}

/// Check if new value is more valuable the the old value
/// @return true if newValue is more valuable; otherwise false.
bool checkExactValue(DBValue oldValue, DBValue newValue)
{
    DBValue absOldValue = std::abs(oldValue);
    DBValue absNewValue = std::abs(newValue);

    if (absOldValue < VALUE_MATE_IN_MAX_PLY)
        return true;  // For non-mate old value we always consider new value to be better
    else if (absNewValue < VALUE_MATE_IN_MAX_PLY)
        return false;  // Prefer to keep old mate score
    else {
        // Both value are mates:
        // 1. If both value have the same sign, we choose the one with larger absolute.
        // 2. If both value have different sign, we choose the newer one.
        return (oldValue < 0) != (newValue < 0) || absNewValue > absOldValue;
    }
};

/// Check if new record has a better value than old record.
/// @return True if new record is better, otherwise false.
bool checkNewRecordValue(const DBRecord &oldRecord, const DBRecord &newRecord)
{
    switch (newRecord.bound()) {
    default: return false;
    case BOUND_UPPER:
        return newRecord.value <= VALUE_MATED_IN_MAX_PLY
               && (oldRecord.bound() == BOUND_UPPER && newRecord.value < oldRecord.value
                   || oldRecord.bound() == BOUND_NONE);
    case BOUND_LOWER:
        return newRecord.value >= VALUE_MATE_IN_MAX_PLY
               && (oldRecord.bound() == BOUND_LOWER && newRecord.value > oldRecord.value
                   || oldRecord.bound() == BOUND_NONE);
    case BOUND_EXACT:
        return std::abs(newRecord.value) > VALUE_MATE_IN_MAX_PLY
               && (oldRecord.bound() != BOUND_EXACT
                   || checkExactValue(oldRecord.value, newRecord.value));
    }
};

/// Check if this key is referenced by any other key.
template <typename IsDeletedParentKeyPredicate>
bool isKeyReferenced(DBStorage &storage, const DBKey &key, IsDeletedParentKeyPredicate pred)
{
    DBKey parentKey;
    parentKey.rule           = key.rule;
    parentKey.boardWidth     = key.boardWidth;
    parentKey.boardHeight    = key.boardHeight;
    parentKey.sideToMove     = ~key.sideToMove;
    parentKey.numBlackStones = key.numBlackStones;
    parentKey.numWhiteStones = key.numWhiteStones;

    const StonePos *excludesBegin, *excludesEnd;
    if (parentKey.sideToMove == BLACK) {
        excludesBegin = key.blackStonesBegin();
        excludesEnd   = key.blackStonesEnd();
        if (parentKey.numBlackStones == 0)
            return false;
        parentKey.numBlackStones--;
    }
    else {
        excludesBegin = key.whiteStonesBegin();
        excludesEnd   = key.whiteStonesEnd();
        if (parentKey.numWhiteStones == 0)
            return false;
        parentKey.numWhiteStones--;
    }

    DBRecord record;
    for (const StonePos *excludePos = excludesBegin; excludePos < excludesEnd; excludePos++) {
        auto it = std::copy(key.blackStonesBegin(), excludePos, parentKey.stones);
        std::copy(excludePos + 1, key.whiteStonesEnd(), it);

        // Make sure we get the smallest parent key
        toSmallestDBKey(parentKey);

        if (pred(parentKey))
            continue;

        if (storage.get(parentKey, record, RECORD_MASK_NONE))
            return true;
    }

    return false;
}

/// Delete all children of a board position from the database storage.
/// @return The number of records deleted.
template <bool ParentDeleted, int MaxSplitPly = 4>
void recursiveDeleteChildren(DBStorage                                          &storage,
                             Board                                              &board,
                             Rule                                                rule,
                             const std::function<DBClient::DelType(DBRecord &)> &deleteFilter,
                             int                                                 threadId,
                             int                                                 ply)
{
    if (board.movesLeft() == 0)
        return;

    static std::set<DBKey> deletingKeys[MaxSplitPly];
    static std::set<DBKey> checkingKeys;
    static std::mutex      mutex[MaxSplitPly + 1];

    DBKey thisKey;
    if constexpr (ParentDeleted)
        thisKey = constructDBKey(board, rule);

    // Find all potential children of this board position
    std::vector<Pos> toDeletePos;
    toDeletePos.reserve(board.movesLeft());
    FOR_EVERY_EMPTY_POS(&board, pos) { toDeletePos.push_back(pos); }

    // Do permutation based on thread id
    if (threadId > 0) {
        for (size_t i = 0; i < toDeletePos.size(); i += (threadId + 1) / 2) {
            size_t swapIndex = (i * (threadId + 1)) % toDeletePos.size();
            std::swap(toDeletePos[i], toDeletePos[swapIndex]);
        }
    }

    // Delete all children
    DBKey    key;
    DBRecord record;
    for (auto pos : toDeletePos) {
        board.move(rule, pos);

        key = constructDBKey(board, rule);

        if (storage.get(key, record, deleteFilter ? RECORD_MASK_LVDB : RECORD_MASK_NONE)) {
            switch (deleteFilter ? deleteFilter(record) : DBClient::DelType::DeleteRecursive) {
            default: break;
            case Database::DBClient::DelType::NoDeleteRecursive: {
                bool recursiveCheck = true;
                if (ply == MaxSplitPly) {
                    std::lock_guard lock(mutex[ply]);
                    if (checkingKeys.find(key) != checkingKeys.end())
                        recursiveCheck = false;
                }

                if (recursiveCheck) {
                    std::pair<std::set<DBKey>::iterator, bool> insertedResult;
                    if (ply == MaxSplitPly) {
                        std::lock_guard lock(mutex[ply]);
                        insertedResult = checkingKeys.insert(key);
                    }

                    // Recursive check all children of this key
                    recursiveDeleteChildren<true>(storage,
                                                  board,
                                                  rule,
                                                  deleteFilter,
                                                  threadId,
                                                  ply + 1);

                    if (ply == MaxSplitPly && insertedResult.second) {
                        std::lock_guard lock(mutex[ply]);
                        checkingKeys.erase(insertedResult.first);
                    }
                }

            } break;
            case Database::DBClient::DelType::DeleteRecursive: {
                bool shouldDelete;
                if constexpr (ParentDeleted)
                    shouldDelete = !isKeyReferenced(storage, key, [&thisKey](const DBKey &key) {
                        return key == thisKey;
                    });
                else
                    shouldDelete =
                        !isKeyReferenced(storage, key, [](const DBKey &key) { return false; });

                if (shouldDelete) {
                    // Delete this key first so we can delete its children
                    storage.del(key);

                    std::pair<std::set<DBKey>::iterator, bool> insertedResult;
                    if (ply < MaxSplitPly) {
                        std::lock_guard lock(mutex[ply]);
                        insertedResult = deletingKeys[ply].insert(key);
                    }

                    // Find all children of this key and delete them
                    recursiveDeleteChildren<false>(storage,
                                                   board,
                                                   rule,
                                                   nullptr,
                                                   threadId,
                                                   ply + 1);

                    if (ply < MaxSplitPly && insertedResult.second) {
                        std::lock_guard lock(mutex[ply]);
                        deletingKeys[ply].erase(insertedResult.first);
                    }
                }
            } break;
            }
        }
        // Help other threads to delete current key if they are deleting it but not finished yet
        else if (ply < MaxSplitPly) {
            bool shouldDelete = false;
            {
                std::lock_guard lock(mutex[ply]);
                if (deletingKeys[ply].find(key) != deletingKeys[ply].end())
                    shouldDelete = true;
            }

            if (shouldDelete)
                // Find all children of this key and delete them
                recursiveDeleteChildren<false>(storage, board, rule, nullptr, threadId, ply + 1);
        }

        board.undo(rule);
    }
}

}  // namespace

namespace Database {

DBKey constructDBKey(const Board &board, Rule rule)
{
    DBKey    key[TRANS_NB];
    StonePos whiteStones[TRANS_NB][MAX_MOVES];
    for (int trans = IDENTITY; trans < TRANS_NB; trans++) {
        key[trans].rule           = rule;
        key[trans].boardWidth     = board.size();
        key[trans].boardHeight    = board.size();
        key[trans].sideToMove     = board.sideToMove();
        key[trans].numBlackStones = 0;
        key[trans].numWhiteStones = 0;
    }

    for (int ply = 0; ply < board.ply(); ply++) {
        Pos move = board.getHistoryMove(ply);
        if (move == Pos::PASS)
            continue;

        Color c = board.cell(move).piece;
        if (c == BLACK) {
            for (int trans = IDENTITY; trans < TRANS_NB; trans++) {
                Pos transformedPos = applyTransform(move, board.size(), (TransformType)trans);
                key[trans].stones[key[trans].numBlackStones++] = {transformedPos.x(),
                                                                  transformedPos.y()};
            }
        }
        else if (c == WHITE) {
            for (int trans = IDENTITY; trans < TRANS_NB; trans++) {
                Pos transformedPos = applyTransform(move, board.size(), (TransformType)trans);
                whiteStones[trans][key[trans].numWhiteStones++] = {transformedPos.x(),
                                                                   transformedPos.y()};
            }
        }
    }

    // Construct 8 symmetry database keys, and find the smallest one
    int smallestIndex = 0;

    for (int trans = IDENTITY; trans < TRANS_NB; trans++) {
        std::sort(key[trans].stones,
                  key[trans].stones + key[trans].numBlackStones,
                  std::less<StonePos>());
        std::copy(whiteStones[trans],
                  whiteStones[trans] + key[trans].numWhiteStones,
                  key[trans].stones + key[trans].numBlackStones);
        std::sort(key[trans].stones + key[trans].numBlackStones,
                  key[trans].stones + key[trans].numBlackStones + key[trans].numWhiteStones,
                  std::less<StonePos>());

        if (trans != smallestIndex && key[trans] < key[smallestIndex])
            smallestIndex = trans;
    }

    return key[smallestIndex];
}

void toSmallestDBKey(DBKey &key)
{
    DBKey  transformedKeys[TRANS_NB - 1];
    int    smallestIndex = 0;
    DBKey *smallestKey   = &key;

    // Construct 8 symmetry database keys, and find the smallest one
    for (int trans = IDENTITY + 1; trans < TRANS_NB; trans++) {
        DBKey &transKey         = transformedKeys[trans - 1];
        transKey.rule           = key.rule;
        transKey.boardWidth     = key.boardWidth;
        transKey.boardHeight    = key.boardHeight;
        transKey.sideToMove     = key.sideToMove;
        transKey.numBlackStones = key.numBlackStones;
        transKey.numWhiteStones = key.numWhiteStones;

        size_t numStones = 0;
        for (const StonePos *pos = key.blackStonesBegin(); pos < key.whiteStonesEnd(); pos++) {
            Pos originPos {pos->x, pos->y};
            Pos transformedPos = applyTransform(originPos, key.boardWidth, (TransformType)trans);
            transKey.stones[numStones++] = {transformedPos.x(), transformedPos.y()};
        }

        std::sort(transKey.stones,
                  transKey.stones + transKey.numBlackStones,
                  std::less<StonePos>());
        std::sort(transKey.stones + transKey.numBlackStones,
                  transKey.stones + transKey.numBlackStones + transKey.numWhiteStones,
                  std::less<StonePos>());

        if (transKey < *smallestKey)
            smallestIndex = trans, smallestKey = &transKey;
    }

    // Copy smallest index back to the key
    if (smallestIndex > 0)
        key = *smallestKey;
}

bool checkOverwrite(const DBRecord &oldRecord,
                    const DBRecord &newRecord,
                    OverwriteRule   owRule,
                    int             exactBias,
                    int             depthBoundBias)
{
    // Always overwrite null record (default constructed)
    if (oldRecord.isNull())
        return true;

    switch (owRule) {
    default:
    case OverwriteRule::Disabled: return false;
    case OverwriteRule::Always: return true;
    case OverwriteRule::BetterLabel: return labelCompare(newRecord.label, oldRecord.label) > 0;
    case OverwriteRule::BetterValue: {
        int labelResult = labelCompare(newRecord.label, oldRecord.label);
        return labelResult > 0
               || labelResult == 0 && isDeterminedLabel(oldRecord.label)
                      && mateRank(newRecord) > mateRank(oldRecord);
    }
    case OverwriteRule::BetterDepthBound: {
        int labelResult = labelCompare(newRecord.label, oldRecord.label);
        return labelResult > 0
               || labelResult == 0 && !isDeterminedLabel(oldRecord.label)
                      && depthBoundRank(newRecord, exactBias, depthBoundBias)
                             >= depthBoundRank(oldRecord, exactBias, depthBoundBias)
                                    + depthBoundBias;
    }
    case OverwriteRule::BetterValueDepthBound: {
        int labelResult = labelCompare(newRecord.label, oldRecord.label);
        return labelResult > 0
               || labelResult == 0
                      && (isDeterminedLabel(oldRecord.label)
                              ? mateRank(newRecord) > mateRank(oldRecord)
                              : depthBoundRank(newRecord, exactBias, depthBoundBias)
                                    >= depthBoundRank(oldRecord, exactBias, depthBoundBias)
                                           + depthBoundBias);
    }
    }
}

DBClient::DBClient(DBStorage   &storage,
                   DBRecordMask recordMask,
                   size_t       dbCacheSize,
                   size_t       dbRecordCacheSize)
    : storage(storage)
    , mask(recordMask)
    , dbCache(std::max<size_t>(dbCacheSize, 1))
    , dbRecordCache(std::max<size_t>(dbRecordCacheSize, 1))
{}

DBClient ::~DBClient()
{
    for (auto &[hashKey, entryCache] : dbCache) {
        if (entryCache.dirty)
            storage.set(entryCache.key, entryCache.record, mask);
    }
}

bool DBClient::query(const Board &board, Rule rule, DBRecord &record)
{
    HashKey hashKey = board.zobristKey();

    // Try find this record in dbRecordCache
    auto &[cachedHashKey, cachedRecord] = dbRecordCache[hashKey];
    if (cachedHashKey == hashKey)
        return record = cachedRecord, true;

    // Try find this database entry in dbCache
    auto entryCache = dbCache.get(hashKey);
    if (entryCache)
        return record = entryCache->record, true;

    // Read record from storage and save it in record cache
    DBKey dbKey = constructDBKey(board, rule);
    if (storage.get(dbKey, record, mask)) {
        // Save a new entry cache in dbCache
        dbCache.put(hashKey,
                    EntryCache {dbKey, record, RECORD_MASK_NONE},
                    [&](std::pair<HashKey, EntryCache> &&cache) {
                        auto &&entryCache = cache.second;
                        if (entryCache.dirty)
                            storage.set(entryCache.key, entryCache.record, mask);
                    });

        // Svae a new record cache in dbRecordCache
        dbRecordCache[hashKey] = std::make_pair(hashKey, record);

        return true;
    }

    return false;
}

void DBClient::queryChildren(const Board                           &board,
                             Rule                                   rule,
                             std::vector<std::pair<Pos, DBRecord>> &childRecords)
{
    DBRecord record;
    Board   &b = const_cast<Board &>(board);

    FOR_EVERY_EMPTY_POS(&b, pos)
    {
        if (rule == RENJU && board.sideToMove() == BLACK && board.checkForbiddenPoint(pos))
            continue;

        b.move(rule, pos);
        if (query(b, rule, record))
            childRecords.emplace_back(pos, record);
        b.undo(rule);
    }
}

bool DBClient::save(const Board &board, Rule rule, const DBRecord &record, OverwriteRule owRule)
{
    DBKey   dbKey;
    HashKey hashKey   = board.zobristKey();
    bool    overwrite = owRule == OverwriteRule::Always
                     || owRule != OverwriteRule::Disabled && !(mask & RECORD_MASK_LVDB);

    // Try find this database entry in dbCache
    if (auto entryCache = dbCache.get(hashKey); entryCache) {
        if (overwrite || checkOverwrite(entryCache->record, record, owRule)) {
            entryCache->record = record;
            entryCache->dirty  = true;

            dbRecordCache[hashKey] = std::make_pair(hashKey, record);
            return true;
        }
        else
            return false;
    }

    // Skip record fetch if we already know that we are going to overwrite it
    if (overwrite) {
        dbKey = constructDBKey(board, rule);
    }
    // Get record from dbRecordCache or dbStorage, as we need to check overwrite condition
    else {
        auto &[cachedHashKey, cachedRecord] = dbRecordCache[hashKey];
        if (cachedHashKey == hashKey) {
            overwrite = checkOverwrite(cachedRecord, record, owRule);
            if (overwrite)
                dbKey = constructDBKey(board, rule);
        }
        else {
            dbKey = constructDBKey(board, rule);
            DBRecord oldRecord;

            // Query record from dbStorage first
            if (storage.get(dbKey, oldRecord))
                overwrite = checkOverwrite(oldRecord, record, owRule);
            else  // Always overwrite if record does not exist
                overwrite = true;
        }
    }

    if (overwrite) {
        dbCache.put(hashKey,
                    EntryCache {dbKey, record, true},
                    [&](std::pair<HashKey, EntryCache> &&cache) {
                        auto &&entryCache = cache.second;
                        if (entryCache.dirty)
                            storage.set(entryCache.key, entryCache.record, mask);
                    });
        dbRecordCache[hashKey] = std::make_pair(hashKey, record);
        return true;
    }
    else
        return false;
}

void DBClient::del(const Board &board, Rule rule)
{
    HashKey hashKey = board.zobristKey();

    // First remove record cache from dbRecordCache if exists
    auto &[cachedHashKey, cachedRecord] = dbRecordCache[hashKey];
    if (cachedHashKey == hashKey)
        cachedHashKey = DBRecordCache::NullKey;

    // Try find this database entry in dbCache
    if (auto entryCache = dbCache.get(hashKey); entryCache) {
        storage.del(entryCache->key);
        dbCache.remove(hashKey);
        return;
    }
    else {
        DBKey dbKey = constructDBKey(board, rule);
        storage.del(dbKey);
    }
}

void DBClient::delChildren(const Board                       &board,
                           Rule                               rule,
                           std::function<DelType(DBRecord &)> deleteFilter)
{
    sync();  // clear all cached records first

    const size_t             numDeleteThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    threads.reserve(numDeleteThreads);

    for (int i = 0; i < numDeleteThreads; i++) {
        threads.emplace_back(
            [this, rule, &deleteFilter, id = i](std::unique_ptr<Board> board) {
                recursiveDeleteChildren<true>(storage, *board, rule, deleteFilter, id, 0);
            },
            std::make_unique<Board>(board, nullptr));
    }

    for (auto &th : threads)
        th.join();
}

void DBClient::sync(bool clearCache)
{
    for (auto &[hashKey, entryCache] : dbCache) {
        if (entryCache.dirty) {
            storage.set(entryCache.key, entryCache.record, mask);
            entryCache.dirty = false;
        }
    }

    if (clearCache) {
        // Clear all cache as records in dbStorage might be newer
        dbCache.clear();
        dbRecordCache.clear();
    }
}

}  // namespace Database
