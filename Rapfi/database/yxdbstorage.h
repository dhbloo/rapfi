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

#include "dbstorage.h"

#include <filesystem>
#include <map>
#include <shared_mutex>

namespace Database {

/// CompactDBKey is a compact version of database key, which requires less memory
/// by allocaing the minimal needed memory to store stone positions on the fly.
struct CompactDBKey
{
    Rule      rule;
    int8_t    boardWidth;
    int8_t    boardHeight;
    Color     sideToMove;
    uint16_t  numBlackStones;
    uint16_t  numWhiteStones;
    StonePos *stones;

    explicit CompactDBKey(const DBKey &key);
    explicit CompactDBKey(Rule      rule,
                          int       width,
                          int       height,
                          Color     sideToMove,
                          uint16_t  numBlackStones,
                          uint16_t  numWhiteStones,
                          StonePos *stones)
        : rule(rule)
        , boardWidth(width)
        , boardHeight(height)
        , sideToMove(sideToMove)
        , numBlackStones(numBlackStones)
        , numWhiteStones(numWhiteStones)
        , stones(stones)
    {}
    ~CompactDBKey();
    explicit operator DBKey() const;

    const StonePos *blackStonesBegin() const { return stones; }
    const StonePos *blackStonesEnd() const { return stones + numBlackStones; }
    const StonePos *whiteStonesBegin() const { return stones + numBlackStones; }
    const StonePos *whiteStonesEnd() const { return stones + numBlackStones + numWhiteStones; }

    /// Equal comparator of two database key.
    friend bool operator==(const CompactDBKey &a, const CompactDBKey &b)
    {
        return databaseKeyCompare(a, b) == 0;
    }
};

struct CompactDBKeyCmp
{
    using is_transparent = void;

    /// Less comparator of two database key.
    bool operator()(const CompactDBKey &a, const CompactDBKey &b) const
    {
        return databaseKeyCompare(a, b) < 0;
    }

    /// Heterogeneous comparasion between compact database key and normal database key.
    bool operator()(const CompactDBKey &a, const DBKey &b) const
    {
        return databaseKeyCompare(a, b) < 0;
    }
    bool operator()(const DBKey &a, const CompactDBKey &b) const
    {
        return databaseKeyCompare(a, b) < 0;
    }
};

/// YXDBStorage implements DBStorage interface for the Yixin-Database file format.
class YXDBStorage : public DBStorage
{
public:
    /// Creates a yixin database storage instance by opening the database file.
    /// If the file does not exist, it will creates the file.
    /// @param filePath The path of the database.
    /// @param compressedSave Enables LZ4-compressed file saving.
    /// @param saveOnClose Whether to save when YXDB is destroyed.
    /// @param backupOnSave When saving the file, copy previous file with the
    ///     same name to a new file with '_bak' postfix.
    /// @note Throws DBStorageError if failed to open the file.
    YXDBStorage(std::filesystem::path filePath,
                bool                  compressedSave,
                bool                  saveOnClose,
                int                   numBackupsOnSave = 1,
                bool                  ignoreCorrupted  = false);
    /// Close the yixin database. All unsaved records will be flushed to file.
    virtual ~YXDBStorage();

    /// Returns the current file path.
    std::filesystem::path getOpenedFilePath() const { return filePath; }

    // -------------------------------------------------------------------
    // Implements the DBStorage interface
    bool   get(const DBKey &key, DBRecord &record, DBRecordMask mask) noexcept override;
    void   set(const DBKey &key, const DBRecord &record, DBRecordMask mask) noexcept override;
    void   del(const DBKey &key) noexcept override;
    bool   flush() noexcept override;
    size_t size() noexcept override;
    Cursor scan(Cursor                                   cursor,
                size_t                                   count,
                std::vector<std::pair<DBKey, DBRecord>> &out) noexcept override;
    // -------------------------------------------------------------------

private:
    std::filesystem::path                             filePath;
    std::map<CompactDBKey, DBRecord, CompactDBKeyCmp> recordsMap;
    std::shared_mutex                                 mutex;
    int                                               numBackupsOnSave;
    bool                                              compressedSave;
    bool                                              saveOnClose;
    bool                                              dirty;

    /// Loads all data from current stream into memory.
    void load(std::istream &is, bool ignoreCorrupted);
    /// Save the current in-memory data to the opened file.
    void save(std::ostream &os) noexcept;
};

}  // namespace Database
