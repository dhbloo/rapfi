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

#include "../core/pos.h"
#include "dbtypes.h"

#include <algorithm>
#include <cassert>
#include <cstring>
#include <ostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace Database {

/// StonePos represents a coordinate in DBKey, which can be sorted.
struct StonePos
{
    int8_t x;
    int8_t y;

    StonePos() = default;
    StonePos(int x, int y) : x(x), y(y) {}
    bool operator==(const StonePos &rhs) const { return x == rhs.x && y == rhs.y; }
    bool operator<(const StonePos &rhs) const
    {
        int lhs_rank = (int)x * FULL_BOARD_SIZE + (int)y;
        int rhs_rank = (int)rhs.x * FULL_BOARD_SIZE + (int)rhs.y;
        return lhs_rank < rhs_rank;
    }
};

/// The three-way comparator of two database key.
/// DBKey is sorted using an "ascending" lexicographical order, in the following:
///     1. rule
///     2. board width
///     3. board height
///     4. stone positions
///     5. side to move (black=0, white=1)
/// @return Negative if lhs < rhs; 0 if lhs > rhs; Positive if lhs > rhs.
template <typename DBKey1, typename DBKey2>
int databaseKeyCompare(const DBKey1 &lhs, const DBKey2 &rhs)
{
    if (int diff = (int)lhs.rule - (int)rhs.rule; diff != 0)
        return diff;
    if (int diff = (int)lhs.boardWidth - (int)rhs.boardWidth; diff != 0)
        return diff;
    if (int diff = (int)lhs.boardHeight - (int)rhs.boardHeight; diff != 0)
        return diff;

    size_t numBytesLhs       = 2 * ((size_t)lhs.numBlackStones + (size_t)lhs.numWhiteStones);
    size_t numBytesRhs       = 2 * ((size_t)rhs.numBlackStones + (size_t)rhs.numWhiteStones);
    size_t numBytesToCompare = std::min(numBytesLhs, numBytesRhs);
    if (numBytesLhs != numBytesRhs)
        return numBytesLhs - numBytesRhs;
    if (int diff = std::memcmp(lhs.stones, rhs.stones, numBytesToCompare); diff != 0)
        return diff;

    return (int)lhs.sideToMove - (int)rhs.sideToMove;
}

/// DBKey encodes the unique representation of one position, without redundancy of
/// any symmetry and transposition. This is done by only storing the position of stones
/// on board and the current side to move (which deals with possible pass moves).
struct DBKey
{
    Rule     rule;
    int8_t   boardWidth;
    int8_t   boardHeight;
    Color    sideToMove;
    uint16_t numBlackStones;
    uint16_t numWhiteStones;
    StonePos stones[MAX_MOVES];

    /// Uninitialized dbkey
    explicit DBKey() = default;
    /// Initialize a new database key with empty board.
    template <typename PosIterator>
    DBKey(Rule        rule,
          int         width,
          int         height,
          Color       sideToMove,
          PosIterator blackMovesBegin,
          PosIterator blackMovesEnd,
          PosIterator whiteMovesBegin,
          PosIterator whiteMovesEnd)
        : rule(rule)
        , boardWidth(width)
        , boardHeight(height)
        , sideToMove(sideToMove)
        , numBlackStones(blackMovesEnd - blackMovesBegin)
        , numWhiteStones(whiteMovesEnd - whiteMovesBegin)
    {
        assert(numBlackStones + numWhiteStones <= MAX_MOVES);
        size_t numStones = 0;

        for (PosIterator pos = blackMovesBegin; pos != blackMovesEnd; pos++)
            stones[numStones++] = {pos->x(), pos->y()};
        std::sort(stones, stones + numStones, std::less<StonePos>());

        for (PosIterator pos = whiteMovesBegin; pos != whiteMovesEnd; pos++)
            stones[numStones++] = {pos->x(), pos->y()};
        std::sort(stones + numBlackStones, stones + numStones, std::less<StonePos>());
    }
    DBKey(Rule                    rule,
          int                     width,
          int                     height,
          Color                   sideToMove,
          const std::vector<Pos> &blackMoves,
          const std::vector<Pos> &whiteMoves)
        : DBKey(rule,
                width,
                height,
                sideToMove,
                blackMoves.begin(),
                blackMoves.end(),
                whiteMoves.begin(),
                whiteMoves.end())
    {}
    /// Copy a database key
    DBKey(const DBKey &rhs)
        : rule(rhs.rule)
        , boardWidth(rhs.boardWidth)
        , boardHeight(rhs.boardHeight)
        , sideToMove(rhs.sideToMove)
        , numBlackStones(rhs.numBlackStones)
        , numWhiteStones(rhs.numWhiteStones)
    {
        assert(rhs.numStones() <= MAX_MOVES);
        std::copy(rhs.blackStonesBegin(), rhs.whiteStonesEnd(), stones);
    }

    int             numStones() const { return numBlackStones + numWhiteStones; }
    const StonePos *blackStonesBegin() const { return stones; }
    const StonePos *blackStonesEnd() const { return stones + numBlackStones; }
    const StonePos *whiteStonesBegin() const { return blackStonesEnd(); }
    const StonePos *whiteStonesEnd() const { return stones + numStones(); }

    /// Equal comparator of two database key.
    friend bool operator==(const DBKey &a, const DBKey &b) { return databaseKeyCompare(a, b) == 0; }
    /// Less comparator of two database key.
    friend bool operator<(const DBKey &a, const DBKey &b) { return databaseKeyCompare(a, b) < 0; }
    /// Print a database key.
    friend std::ostream &operator<<(std::ostream &os, const DBKey &key)
    {
        switch (key.rule) {
        case Rule::FREESTYLE: os << 'f'; break;
        case Rule::STANDARD: os << 's'; break;
        case Rule::RENJU: os << 'r'; break;
        default: os << '?'; break;
        }
        os << '-' << int(key.boardWidth) << '-' << int(key.boardHeight) << '-';
        for (const StonePos *s = key.blackStonesBegin(); s != key.blackStonesEnd(); s++)
            os << char('a' + s->x) << int(s->y + 1);
        os << '-';
        for (const StonePos *s = key.whiteStonesBegin(); s != key.whiteStonesEnd(); s++)
            os << char('a' + s->x) << int(s->y + 1);
        os << '-' << (key.sideToMove == BLACK ? 'b' : 'w');
        return os;
    }
};

/// DBStorage class defines the common interface to all database storages.
/// Subclass of DBStorage should implement storage functionaility (read/write/flush).
/// All operations on the db storage instance should be atmoic and thread-safe, so
/// multiple reads/write is allowed for different threads at the same time.
class DBStorage
{
public:
    /// Close the db storage. The db storage should flush all its unsaved written records.
    virtual ~DBStorage() = default;

    /// @brief Read a database record with the given key.
    /// @param key The key which uniquely identity one game position.
    /// @param record The record structure to save the queried result.
    /// @param mask The parts of record to be queried.
    /// @return True if read succeeded. False if read failed or there is no such key.
    virtual bool
    get(const DBKey &key, DBRecord &record, DBRecordMask mask = RECORD_MASK_ALL) noexcept = 0;

    /// Write a database record with the given key.
    /// @param key The key which uniquely identity one game position.
    /// @param record The record information to write with this position.
    /// @param mask The parts of record to be updated if it already exists.
    ///     For non existing record, this parameter is ignored.
    virtual void set(const DBKey &key, const DBRecord &record, DBRecordMask mask) noexcept = 0;

    /// Remove a database record with the given key.
    /// @param key The key which uniquely identity one game position.
    /// @note The key is ignored if it does not exist.
    virtual void del(const DBKey &key) noexcept = 0;

    /// Tell the db storage to push all its queued changes to the storage backend
    /// so the current instance can be closed immediately without further IO ops.
    /// This is useful when we want saving in advance without closing the storage.
    /// return Whether the flush operation succeeded.
    virtual bool flush() noexcept = 0;

    /// Return the number of entries in the database.
    virtual size_t size() noexcept = 0;

    /// Cursor type indicates the current position in the database storage,
    /// which can be used to scan the whole database.
    using Cursor = size_t;
    /// Iterate the entire database in an incremental way.
    /// @param cursor The start position in the database. Pass a zero cursor means
    ///     starting from the beginning of the database.
    /// @param count The number of entries to get in this call.
    /// @param out The container for receiving iterated entries. Previous elements
    ///     in the container will be kept. The actual elements retrieved can be get
    ///     by comparaing the size before and after the scan.
    /// @return Cursor that can be used for the next incremental scan, or the zero
    ///     cursor which means all entries in the database have been iterated.
    virtual Cursor
    scan(Cursor cursor, size_t count, std::vector<std::pair<DBKey, DBRecord>> &out) noexcept = 0;
};

/// The base exception class for a db storage error.
class DBStorageError : public ::std::runtime_error
{
public:
    DBStorageError(::std::string message) : ::std::runtime_error(message) {}
};

class DBStorageCorruptedRecordError : public DBStorageError
{
public:
    DBStorageCorruptedRecordError(::std::string storageName, ::std::string message)
        : DBStorageError(storageName + ": corrupted database record"
                         + (message.empty() ? std::string {} : " (" + message + ")"))
    {}
};

}  // namespace Database
