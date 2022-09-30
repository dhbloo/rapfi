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

#include "yxdbstorage.h"

#include "../core/iohelper.h"
#include "dbtypes.h"

#include <fstream>
#include <mutex>
#include <string>

namespace Database {

CompactDBKey::CompactDBKey(const DBKey &key)
    : rule(key.rule)
    , boardWidth(key.boardWidth)
    , boardHeight(key.boardHeight)
    , sideToMove(key.sideToMove)
    , numBlackStones(key.numBlackStones)
    , numWhiteStones(key.numWhiteStones)
{
    stones = new StonePos[numBlackStones + numWhiteStones];
    std::copy(key.blackStonesBegin(), key.whiteStonesEnd(), stones);
}

CompactDBKey::~CompactDBKey()
{
    delete[] stones;
}

CompactDBKey::operator DBKey() const
{
    DBKey key;
    key.rule           = rule;
    key.boardWidth     = boardWidth;
    key.boardHeight    = boardHeight;
    key.sideToMove     = sideToMove;
    key.numBlackStones = numBlackStones;
    key.numWhiteStones = numWhiteStones;
    std::copy(stones, stones + numBlackStones + numWhiteStones, key.stones);
    return key;
}

YXDBStorage::YXDBStorage(std::filesystem::path filePath,
                         bool                  compressedSave,
                         bool                  saveOnClose,
                         int                   numBackupsOnSave,
                         bool                  ignoreCorrupted)
    : filePath(filePath)
    , numBackupsOnSave(numBackupsOnSave)
    , compressedSave(compressedSave)
    , saveOnClose(saveOnClose)
    , dirty(false)
{
    if (!std::filesystem::exists(filePath))
        return;

    std::ifstream file(filePath, std::ios::binary);
    if (!file.is_open() || !file)
        throw DBStorageError("Failed to open YXDB file at " + filePath.string());

    // Check LZ4 file magic to choose a compress type
    int magic;
    file.read(reinterpret_cast<char *>(&magic), sizeof(magic));
    file.seekg(0);

    {
        Compressor    compressor(static_cast<std::istream &>(file),
                              magic == 0x184D2204 ? Compressor::Type::LZ4_DEFAULT
                                                     : Compressor::Type::NO_COMPRESS);
        std::istream *istreamPtr = compressor.openInputStream();
        if (!istreamPtr || !*istreamPtr)
            throw DBStorageError("Failed to open LZ4 compressed YXDB file " + filePath.string());

        load(*istreamPtr, ignoreCorrupted);
    }
}

YXDBStorage::~YXDBStorage()
{
    if (saveOnClose)
        flush();
}

bool YXDBStorage::get(const DBKey &key, DBRecord &record, DBRecordMask mask) noexcept
{
    std::shared_lock<std::shared_mutex> readerLock(mutex);

    if (auto it = recordsMap.find(key); it != recordsMap.end()) {
        record.update(it->second, mask);
        return true;
    }
    else
        return false;
}

void YXDBStorage::set(const DBKey &key, const DBRecord &record, DBRecordMask mask) noexcept
{
    std::unique_lock<std::shared_mutex> writerLock(mutex);

    if (auto it = recordsMap.find(key); it != recordsMap.end())
        it->second.update(record, mask);
    else
        recordsMap.insert(std::make_pair(key, record));

    dirty = true;
}

void YXDBStorage::del(const DBKey &key) noexcept
{
    std::unique_lock<std::shared_mutex> writerLock(mutex);

    if (auto it = recordsMap.find(key); it != recordsMap.end()) {
        recordsMap.erase(it);
        dirty = true;
    }
}

bool YXDBStorage::flush() noexcept
{
    std::unique_lock<std::shared_mutex> lock(mutex);

    if (!dirty && std::filesystem::exists(filePath))
        return true;

    // Backup previous file first
    if (numBackupsOnSave && std::filesystem::exists(filePath)) {
        auto makeBackupFilePath = [this](int index) -> std::filesystem::path {
            std::filesystem::path backupPath = filePath;
            std::string           indexStr   = std::to_string(index);
            auto                  fn_noext   = backupPath.stem().concat("_bak").concat(indexStr);
            auto                  ext        = backupPath.extension();
            backupPath.replace_filename(fn_noext);
            backupPath.replace_extension(ext);
            return backupPath;
        };

        for (int i = numBackupsOnSave; i > 0; i--) {
            std::filesystem::path toClearPath = i == 1 ? filePath : makeBackupFilePath(i - 1);
            std::filesystem::path backupPath  = makeBackupFilePath(i);

            std::error_code ec;
            // Remove previous backup file
            std::filesystem::remove(backupPath, ec);
            // Move current file to be cleared to the previous backup file
            std::filesystem::rename(toClearPath, backupPath, ec);
        }
    }

    std::ofstream file(filePath, std::ios::binary | std::ios::trunc);
    if (file.is_open()) {
        Compressor    compressor(static_cast<std::ostream &>(file),
                              compressedSave ? Compressor::Type::LZ4_DEFAULT
                                                : Compressor::Type::NO_COMPRESS);
        std::ostream *ostreamPtr = compressor.openOutputStream();
        if (ostreamPtr && *ostreamPtr) {
            save(*ostreamPtr);
            dirty = false;
            return true;
        }
    }

    ERRORL("Failed to open YXDB file at " + filePath.string());
    return false;
}

size_t YXDBStorage::size() noexcept
{
    std::shared_lock<std::shared_mutex> readerLock(mutex);
    return recordsMap.size();
}

YXDBStorage::Cursor YXDBStorage::scan(Cursor                                   cursor,
                                      size_t                                   count,
                                      std::vector<std::pair<DBKey, DBRecord>> &out) noexcept
{
    std::shared_lock<std::shared_mutex> readerLock(mutex);

    // Find the starting iterator at the cursor
    auto it = recordsMap.begin();
    for (size_t i = 0; i < cursor; i++) {
        it++;

        if (it == recordsMap.end())
            break;
    }

    while (count > 0 && it != recordsMap.end()) {
        out.emplace_back(DBKey(it->first), it->second);
        it++;
        count--;
        cursor++;
    }

    return it == recordsMap.end() ? Cursor(0) : cursor;
}

void YXDBStorage::load(std::istream &is, bool ignoreCorrupted)
{
    uint32_t numRecords;
    is.read(reinterpret_cast<char *>(&numRecords), sizeof(numRecords));

    std::vector<int8_t> byteBuffer;
    byteBuffer.resize(1024);  // reserve initial space

    auto hint = recordsMap.begin();
    for (uint32_t recordIdx = 0; recordIdx < numRecords; recordIdx++) {
        // Read record key
        uint16_t numKeyBytes;
        is.read(reinterpret_cast<char *>(&numKeyBytes), sizeof(numKeyBytes));
        if (numKeyBytes == 0)
            continue;
        if (numKeyBytes > byteBuffer.size())
            byteBuffer.resize(numKeyBytes);
        is.read(reinterpret_cast<char *>(byteBuffer.data()), numKeyBytes);

        // Parse record key
        Rule rule = static_cast<Rule>(byteBuffer[0]);
        if (rule >= RULE_NB) {
            if (ignoreCorrupted)
                continue;
            throw DBStorageCorruptedRecordError(filePath.string(),
                                                "with invalid rule at index "
                                                    + std::to_string(recordIdx));
        }

        int boardXLen = byteBuffer[1], boardYLen = byteBuffer[2];
        if ((unsigned)boardXLen > ACTUAL_BOARD_SIZE || (unsigned)boardYLen > ACTUAL_BOARD_SIZE) {
            if (ignoreCorrupted)
                continue;
            throw DBStorageCorruptedRecordError(filePath.string(),
                                                "with invalid board size at index "
                                                    + std::to_string(recordIdx));
        }

        uint16_t numStones = (numKeyBytes - 3) / 2;
        if (numStones > boardXLen * boardYLen) {
            if (ignoreCorrupted)
                continue;
            throw DBStorageCorruptedRecordError(filePath.string(),
                                                "with invalid number of stones at index "
                                                    + std::to_string(recordIdx));
        }

        uint16_t numBlackStones = (numStones + 1) / 2;
        uint16_t numWhiteStones = numStones / 2;
        Color    sideToMove     = numBlackStones == numWhiteStones ? BLACK : WHITE;

        auto   stones   = std::make_unique<StonePos[]>(numStones);
        size_t stoneIdx = 0;
        for (uint16_t blackIdx = 0; blackIdx < numBlackStones; blackIdx++) {
            int8_t x = byteBuffer[3 + blackIdx * 2];
            int8_t y = byteBuffer[4 + blackIdx * 2];
            if (x == -1 && y == -1)  // the last pass move
                break;
            else if (x >= 0 && y >= 0 && x < boardXLen && y < boardYLen)
                stones[stoneIdx++] = {x, y};
            else if (ignoreCorrupted)
                goto next_record;
            else
                throw DBStorageCorruptedRecordError(filePath.string(),
                                                    "with invalid black pos at index "
                                                        + std::to_string(recordIdx));
        }
        for (uint16_t whiteIdx = 0; whiteIdx < numWhiteStones; whiteIdx++) {
            int8_t x = byteBuffer[3 + (numBlackStones + whiteIdx) * 2];
            int8_t y = byteBuffer[4 + (numBlackStones + whiteIdx) * 2];
            if (x == -1 && y == -1)  // the last pass move
                break;
            else if (x >= 0 && y >= 0 && x < boardXLen && y < boardYLen)
                stones[stoneIdx++] = {x, y};
            else if (ignoreCorrupted)
                goto next_record;
            else
                throw DBStorageCorruptedRecordError(filePath.string(),
                                                    "with invalid white pos at index "
                                                        + std::to_string(recordIdx));
        }

        // Read record message
        uint16_t numRecordBytes;
        is.read(reinterpret_cast<char *>(&numRecordBytes), sizeof(numRecordBytes));
        if (numRecordBytes > byteBuffer.size())
            byteBuffer.resize(numRecordBytes);
        is.read(reinterpret_cast<char *>(byteBuffer.data()), numRecordBytes);

        // Emplace db key and db record in map
        hint = recordsMap.emplace_hint(
            hint,
            std::piecewise_construct,
            std::forward_as_tuple(rule,
                                  boardXLen,
                                  boardYLen,
                                  sideToMove,
                                  numBlackStones,
                                  numWhiteStones,
                                  stones.release()),
            std::forward_as_tuple(DBRecord {
                numRecordBytes > 0 ? static_cast<DBLabel>(byteBuffer[0]) : DBLabel(0),
                numRecordBytes > 2 ? *reinterpret_cast<DBValue *>(&byteBuffer[1]) : DBValue(0),
                numRecordBytes > 4 ? *reinterpret_cast<DBDepthBound *>(&byteBuffer[3])
                                   : DBDepthBound(0),
                numRecordBytes > 5 ? std::string {reinterpret_cast<char *>(&byteBuffer[5]),
                                                  static_cast<size_t>(numRecordBytes - 5)}
                                   : std::string {}}));

    next_record:;
    }
}

void YXDBStorage::save(std::ostream &os) noexcept
{
    uint32_t numRecords = recordsMap.size();
    os.write(reinterpret_cast<char *>(&numRecords), sizeof(numRecords));

    std::vector<int8_t> byteBuffer;
    byteBuffer.reserve(1024);  // reserve initial space

    for (auto it = recordsMap.cbegin(); it != recordsMap.cend(); it++) {
        const CompactDBKey &key    = it->first;
        const DBRecord     &record = it->second;

        // Serialize record key
        byteBuffer.push_back(key.rule);
        byteBuffer.push_back(key.boardWidth);
        byteBuffer.push_back(key.boardHeight);
        Color normalSTM   = (key.numBlackStones + key.numWhiteStones) % 2 == 0 ? BLACK : WHITE;
        bool  addPassMove = normalSTM != key.sideToMove;
        for (auto posIt = key.blackStonesBegin(); posIt != key.blackStonesEnd(); posIt++) {
            byteBuffer.push_back(posIt->x);
            byteBuffer.push_back(posIt->y);
        }
        if (addPassMove && normalSTM == BLACK) {
            byteBuffer.push_back(-1);
            byteBuffer.push_back(-1);
        }
        for (auto posIt = key.whiteStonesBegin(); posIt != key.whiteStonesEnd(); posIt++) {
            byteBuffer.push_back(posIt->x);
            byteBuffer.push_back(posIt->y);
        }
        if (addPassMove && normalSTM == WHITE) {
            byteBuffer.push_back(-1);
            byteBuffer.push_back(-1);
        }

        // Write record key
        uint16_t numKeyBytes = byteBuffer.size();
        os.write(reinterpret_cast<char *>(&numKeyBytes), sizeof(numKeyBytes));
        os.write(reinterpret_cast<char *>(byteBuffer.data()), numKeyBytes);
        byteBuffer.clear();

        // Write record message
        if (record.isNull()) {
            uint16_t numRecordBytes = 0;
            os.write(reinterpret_cast<char *>(&numRecordBytes), sizeof(numRecordBytes));
        }
        else {
            uint16_t numRecordBytes = 5 + record.text.length();
            os.write(reinterpret_cast<char *>(&numRecordBytes), sizeof(numRecordBytes));
            os.write(reinterpret_cast<const char *>(&record.label), sizeof(DBLabel));
            os.write(reinterpret_cast<const char *>(&record.value), sizeof(DBValue));
            os.write(reinterpret_cast<const char *>(&record.depthbound), sizeof(DBDepthBound));
            os.write(record.text.c_str(), record.text.length());
        }
    }
}

}  // namespace Database
