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

#include "dbutils.h"

#include "../core/string.h"
#include "../game/board.h"
#include "../game/scopedmove.h"
#include "dbclient.h"
#include "dbconfig.h"
#include "parallelwalk.h"
#include "renlib.h"

#include <iomanip>
#include <sstream>
#include <vector>

namespace {

using namespace Database;

void copyDatabasePathToRoot(DBStorage &dbSrc, DBStorage &dbDst, Board &board, Rule rule)
{
    std::vector<Pos> path;
    path.reserve(board.ply());

    while (board.ply() > 0) {
        Pos move = board.getLastMove();
        path.push_back(move);
        board.undo(rule);
    }

    DBRecord record, tempRecord;
    DBKey    key;
    while (path.size()) {
        Pos pos = path.back();
        path.pop_back();

        board.move(rule, pos);
        key = constructDBKey(board, rule);

        if (dbSrc.get(key, record, RECORD_MASK_ALL)
            && !dbDst.get(key, tempRecord, RECORD_MASK_NONE))
            dbDst.set(key, record, RECORD_MASK_ALL);
    }
}

void copyDatabaseBranch(DBStorage &dbSrc,
                        DBStorage &dbDst,
                        Board     &board,
                        Rule       rule,
                        int        threadId,
                        int        ply)
{
    // Find all potential children of this board position, spread over subtrees
    std::vector<Pos> toCopyPos;
    toCopyPos.reserve(board.movesLeft());
    FOR_EVERY_EMPTY_POS(&board, pos)
    {
        toCopyPos.push_back(pos);
    }
    permuteForThread(toCopyPos, threadId);

    // Copy all children
    DBKey    key;
    DBRecord record, tempRecord;
    for (auto pos : toCopyPos) {
        board.move(rule, pos);

        key = constructDBKey(board, rule);

        if (dbSrc.get(key, record, RECORD_MASK_ALL)
            && !dbDst.get(key, tempRecord, RECORD_MASK_NONE)) {
            copyDatabaseBranch(dbSrc, dbDst, board, rule, threadId, ply + 1);
            dbDst.set(key, record, RECORD_MASK_ALL);
        }

        board.undo(rule);
    }
}

}  // namespace

namespace Database {

void databaseToCSVFile(::Database::DBStorage                               &dbStorage,
                       std::ostream                                        &csvStream,
                       std::function<bool(const DBKey &, const DBRecord &)> filter)
{
    constexpr size_t BatchSize = 2000;
    constexpr char   Sep       = ',';

    DBStorage::Cursor                       cursor;
    std::vector<std::pair<DBKey, DBRecord>> dbKeyRecords;
    dbKeyRecords.reserve(BatchSize);

    csvStream << "index" << Sep << "key" << Sep << "label" << Sep << "value" << Sep << "depth"
              << Sep << "bound" << Sep << "text" << '\n';

    size_t            count = 0;
    std::stringstream ss;
    do {
        dbKeyRecords.clear();
        cursor = dbStorage.scan(cursor, BatchSize, dbKeyRecords);

        for (const auto &[dbKey, dbRecord] : dbKeyRecords) {
            if (filter && !filter(dbKey, dbRecord))
                continue;

            csvStream << count << Sep << dbKey << Sep;

            if (dbRecord.isNull())
                csvStream << "(null)";
            else if (dbRecord.label == LABEL_NONE)
                csvStream << "(none)";
            else if (std::isprint((unsigned char)dbRecord.label)
                     && !std::isspace((unsigned char)dbRecord.label))
                csvStream << (char)dbRecord.label;
            else
                csvStream << '(' << (int)dbRecord.label << ')';

            csvStream << Sep << dbRecord.value << Sep << dbRecord.depth() << Sep;

            switch (dbRecord.bound()) {
            case BOUND_EXACT: csvStream << "exact"; break;
            case BOUND_LOWER: csvStream << "lower"; break;
            case BOUND_UPPER: csvStream << "upper"; break;
            default: csvStream << "none"; break;
            }

            std::stringstream ss;
            ss << std::quoted(dbRecord.text);
            std::string escapedText = ss.str();
            // Esacpe all '\n' with "\\n" in text
            // (\n is used to separate different board texts and may appear in comments)
            replaceAll(escapedText, "\n", "\\n");
            // Esacpe all '\b' with "\\b" in text
            // (\b is used to separate sub-sections in the text)
            replaceAll(escapedText, "\b", "\\b");
            // Remove all '\0' in text that may exist due to bug in early implementation
            replaceAll(escapedText, std::string_view {"\0", 1}, "");
            csvStream << Sep << escapedText << '\n';
            count++;
        }

    } while (cursor);
}

size_t mergeDatabase(DBStorage &dbDst, DBStorage &dbSrc, OverwriteRule owRule)
{
    constexpr size_t BatchSize = 2000;

    DBStorage::Cursor                       cursor;
    std::vector<std::pair<DBKey, DBRecord>> dbRecords;

    size_t writeCount = 0;
    do {
        dbRecords.clear();
        cursor = dbSrc.scan(cursor, BatchSize, dbRecords);

        for (auto &[dbKey, dbRecord] : dbRecords) {
            // Scoped per key so a failed lookup merges against an empty record
            // instead of the previous key's contents.
            DBRecord oldRecord {};
            if (!dbDst.get(dbKey, oldRecord, RECORD_MASK_ALL)
                || checkOverwrite(oldRecord,
                                  dbRecord,
                                  owRule,
                                  DatabaseCfg.search.overwriteExactBias,
                                  0)) {
                // Merge board texts.
                dbRecord.copyBoardTextFrom(oldRecord, false);

                // Merge comment if the newRecord does not have them.
                if (!oldRecord.comment().empty() && dbRecord.comment().empty())
                    dbRecord.setComment(oldRecord.comment());

                dbDst.set(dbKey, dbRecord, RECORD_MASK_ALL);
                writeCount++;
            }
            else {
                // Merge board texts.
                oldRecord.copyBoardTextFrom(dbRecord, false);

                // Merge comment if the oldRecord does not have them.
                if (oldRecord.comment().empty() && !dbRecord.comment().empty())
                    oldRecord.setComment(dbRecord.comment());

                dbDst.set(dbKey, oldRecord, RECORD_MASK_TEXT);
            }
        }

    } while (cursor);

    return writeCount;
}

size_t splitDatabase(DBStorage &dbSrc, DBStorage &dbDst, const Board &board, Rule rule)
{
    size_t sizeBeforeSplit = dbDst.size();

    parallelWalkOnBoardClones(board, [&dbSrc, &dbDst, rule](Board &board, int id) {
        copyDatabaseBranch(dbSrc, dbDst, board, rule, id, 0);
    });

    Board boardClone(board, nullptr);
    copyDatabasePathToRoot(dbSrc, dbDst, boardClone, rule);

    size_t sizeAfterSplit = dbDst.size();
    return sizeAfterSplit - sizeBeforeSplit;
}

size_t importLibToDatabase(DBStorage &dbDst, std::istream &libStream, Rule rule, int boardSize)
{
    size_t writeCount = 0;

    auto callback = [&](const Board       &board,
                        bool               hasTag,
                        const std::string *text,
                        const std::string *comment) {
        DBKey    key = constructDBKey(board, rule);
        DBRecord oldRecord, newRecord = {LABEL_NONE};
        bool     hasOldRecord = dbDst.get(key, oldRecord);

        if (!text && !comment) {
            // Write empty record if no record exists
            if (!hasOldRecord) {
                dbDst.set(key, newRecord, RECORD_MASK_LVDB);
                writeCount++;
            }

            return;
        }

        if (text && text->size() > 0) {
            const std::string &t = *text;
            if (t[0] == 'W') {
                newRecord.label = LABEL_WIN;
                if (t.size() > 1) {
                    int mateStepFromRoot = std::atoi(t.c_str() + 1);
                    if (mateStepFromRoot > board.ply()) {
                        newRecord.value = mated_in(mateStepFromRoot - board.ply());
                        newRecord.setDepthBound(0, BOUND_EXACT);
                    }
                }
            }
            else if (t[0] == 'L') {
                newRecord.label = LABEL_LOSE;
                if (t.size() > 1) {
                    int mateStepFromRoot = std::atoi(t.c_str() + 1);
                    if (mateStepFromRoot > board.ply()) {
                        newRecord.value = mate_in(mateStepFromRoot - board.ply());
                        newRecord.setDepthBound(0, BOUND_EXACT);
                    }
                }
            }
            else if (t[0] == DatabaseCfg.libfile.blackWinMark && board.sideToMove() == BLACK) {
                newRecord.label = LABEL_WIN;
            }
            else if (t[0] == DatabaseCfg.libfile.whiteWinMark && board.sideToMove() == WHITE) {
                newRecord.label = LABEL_WIN;
            }
            else if (t[0] == DatabaseCfg.libfile.blackLoseMark && board.sideToMove() == BLACK) {
                newRecord.label = LABEL_LOSE;
            }
            else if (t[0] == DatabaseCfg.libfile.whiteLoseMark && board.sideToMove() == WHITE) {
                newRecord.label = LABEL_LOSE;
            }
            else if ((t[0] == 'v' || t[0] == 'm') && t.length() > 1) {
                newRecord.label = LABEL_NONE;
                Value value =
                    std::clamp((Value)std::atoi(t.c_str() + 1), VALUE_EVAL_MIN, VALUE_EVAL_MAX);
                newRecord.value = DBValue(t[0] == 'v' ? -value : value);
                newRecord.setDepthBound(0, BOUND_EXACT);
            }
            else if (board.ply() > 0 && !DatabaseCfg.libfile.ignoreBoardText) {
                // Write parent record for board text, on the parent position.
                ScopedRuleUndo rewind(board, rule);
                DBClient       dbClient(dbDst, RECORD_MASK_TEXT);
                dbClient.setBoardText(
                    board,
                    rule,
                    rewind.lastMove(),
                    LegacyFileCPToUTF8(t, DatabaseCfg.legacyFileCodePage));
            }
        }

        if (comment && !DatabaseCfg.libfile.ignoreComment) {
            std::string newCmt =
                LegacyFileCPToUTF8(*comment, DatabaseCfg.legacyFileCodePage);
            replaceAll(newCmt, "\r\n", "\n");

            if (hasOldRecord) {
                std::string cmt;
                cmt = oldRecord.comment();
                if (!cmt.empty() && cmt.back() != '\b')
                    cmt.push_back('\b');
                cmt.append(newCmt);
                newRecord.setComment(cmt);
            }
            else
                newRecord.setComment(newCmt);
        }

        if (!hasOldRecord || checkOverwrite(oldRecord, newRecord, OverwriteRule::BetterValue)) {
            dbDst.set(key,
                      newRecord,
                      DBRecordMask((text ? RECORD_MASK_LVDB : RECORD_MASK_NONE)
                                   | (comment ? RECORD_MASK_TEXT : RECORD_MASK_NONE)));
            writeCount++;
        }
    };

    Renlib::traverseLib(libStream, boardSize, rule, callback);

    return writeCount;
}

size_t
exportDatabaseToLib(DBClient &dbClient, std::ostream &libStream, const Board &board, Rule rule)
{
    return Renlib::exportDatabase(libStream, dbClient, board, rule);
}

}  // namespace Database
