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

#include "dbcommand.h"

#include "../config.h"
#include "../core/iohelper.h"
#include "../core/utils.h"
#include "../database/dbclient.h"
#include "../database/dbconfig.h"
#include "../database/dbutils.h"
#include "../game/board.h"
#include "../search/searchcommon.h"

#include <cctype>
#include <fstream>
#include <functional>
#include <iomanip>
#include <sstream>
#include <string_view>

using namespace Database;

namespace Command {
std::filesystem::path readPathFromInput(std::istream &in)
{
    std::string path;
    in >> std::ws;
    std::getline(in, path);
    trimInplace(path);

    return pathFromConsoleString(path);
}

bool checkLastMoveIsNotPass(Board &board)
{
    if (board.passMoveCount() > 0 && board.getLastMove() == Pos::PASS) {
        ERRORL("Consecutive pass is not supported. "
               "There must be one non-pass move between two pass moves.");
        return true;
    }
    return false;
}

/// Parse a coord in form 'x,y' from in stream and check if the pos is legal.
/// Legal means the pos must be an empty cell or an non-consecutive pass move.
std::optional<Pos> parseLegalCoord(std::istream &is, Board &board)
{
    int  x = -2, y = -2;
    char comma;
    is >> x >> comma >> y;

    Pos pos = inputCoordConvert(x, y, board.size(), Config::GeneralCfg.ioCoordMode);
    if (board.isLegal(pos)) {
        if (checkLastMoveIsNotPass(board))
            return std::nullopt;
        return pos;
    }
    else {
        ERRORL("Coord is not valid or empty.");
        return std::nullopt;
    }
}

void saveDatabase(DBCommandContext &ctx)
{
    if (ctx.storage) {
        auto startTime = now();
        ctx.storage->flush();
        auto endTime = now();
        MESSAGEL("Saved database file using " << (endTime - startTime) << " ms.");
    }
}

void databaseToTxt(DBCommandContext &ctx, bool currentBoardSizeAndRule)
{
    auto txtPath = readPathFromInput(ctx.in);
    // Dispatched without CheckBoardOK: the currentBoardSizeAndRule filter needs
    // a board to define "current", so bail out quietly if none exists yet
    // (previously a null-board crash if YXDBTOTXT arrived before the first
    // board-allocating command).
    if (currentBoardSizeAndRule && !ctx.board)
        return;
    if (ctx.storage) {
        std::ofstream                                        txtout(txtPath);
        std::function<bool(const DBKey &, const DBRecord &)> filter = nullptr;
        if (currentBoardSizeAndRule)
            filter = [&](const DBKey &key, const DBRecord &record) -> bool {
                return key.boardHeight == ctx.board->size() && key.boardWidth == ctx.board->size()
                       && key.rule == ctx.options.rule.rule;
            };
        ::Database::databaseToCSVFile(*ctx.storage, txtout, filter);
        MESSAGEL("Wrote " << (currentBoardSizeAndRule ? "(current boardsize and rule)" : "(all)")
                          << " database to csv-format text file " << pathToConsoleString(txtPath));
    }
}

void libToDatabase(DBCommandContext &ctx)
{
    auto libPath = readPathFromInput(ctx.in);
    if (ctx.storage) {
        MESSAGEL("Importing from lib file " << pathToConsoleString(libPath)
                                            << ", this might take a while...");
        auto          startTime = now();
        std::ifstream libStream(libPath, std::ios::binary);
        if (!libStream) {
            ERRORL("Unable to open file " << pathToConsoleString(libPath) << " for reading.");
            return;
        }

        try {
            size_t writeCount = ::Database::importLibToDatabase(*ctx.storage,
                                                                libStream,
                                                                ctx.options.rule,
                                                                ctx.board ? ctx.board->size() : 15);
            auto   endTime    = now();
            MESSAGEL("Imported " << writeCount << " records from lib file using "
                                 << (endTime - startTime) << " ms.");
        }
        catch (const std::exception &e) {
            ERRORL("Failed to import lib file: " << e.what());
        }
    }
}

void databaseToLib(DBCommandContext &ctx)
{
    auto libPath = readPathFromInput(ctx.in);
    if (ctx.storage) {
        MESSAGEL("Exporting to lib file " << pathToConsoleString(libPath)
                                          << ", this might take a while...");
        auto          startTime = now();
        std::ofstream libStream(libPath, std::ios::binary);
        if (!libStream) {
            ERRORL("Unable to open file " << pathToConsoleString(libPath) << " for writing.");
            return;
        }

        try {
            DBClient dbClient(*ctx.storage, RECORD_MASK_ALL);
            size_t   nodeCount =
                ::Database::exportDatabaseToLib(dbClient, libStream, *ctx.board, ctx.options.rule);
            auto endTime = now();
            MESSAGEL("Exported " << nodeCount << " nodes to lib file using "
                                 << (endTime - startTime) << " ms.");
        }
        catch (const std::exception &e) {
            ERRORL("Failed to export database to lib file: " << e.what());
        }
    }
}

void getDatabasePosition(DBCommandContext &ctx)
{
    ctx.board->newGame(ctx.options.rule);

    // Read position sequence
    while (true) {
        std::string coordStr;
        if (!(ctx.in >> coordStr))
            break;
        upperInplace(coordStr);

        if (coordStr == "DONE")
            break;

        std::stringstream ss;
        ss << coordStr;
        auto pos = parseLegalCoord(ss, *ctx.board);

        if (pos.has_value())
            ctx.board->move(ctx.options.rule, *pos);
    }
}

void queryDatabaseAll(DBCommandContext &ctx, bool getPosition)
{
    using namespace ::Database;
    if (getPosition)
        getDatabasePosition(ctx);

    if (ctx.storage) {
        MESSAGEL("DATABASE REFRESH");

        DBClient dbClient(*ctx.storage, RECORD_MASK_ALL);

        // Get all child records and child ctx.board texts
        auto childRecords     = dbClient.queryChildren(*ctx.board, ctx.options.rule);
        auto boardPosAndTexts = dbClient.queryBoardTexts(*ctx.board, ctx.options.rule);

        // Iterate all empty positions
        auto childRecordIt      = childRecords.begin();
        auto boardPosAndTextsIt = boardPosAndTexts.begin();
        FOR_EVERY_EMPTY_POS(ctx.board, pos)
        {
            bool        printThisPos      = false;
            int         displayLabelValue = -1;
            std::string boardTextUTF8;
            DBRecord    record {};  // init as null record

            if (childRecordIt != childRecords.end() && childRecordIt->first == pos) {
                record                   = childRecordIt->second;
                std::string displayLabel = record.displayLabel();
                if (!displayLabel.empty()) {
                    // Encode the display label (maximum 4 chars) as an int32
                    if (displayLabel.length() > 4)
                        displayLabel = displayLabel.substr(0, 4);
                    displayLabelValue = 0;
                    for (char c : displayLabel)
                        displayLabelValue = (displayLabelValue << 8) | c;
                }
                childRecordIt++;
                printThisPos = true;
            }

            if (boardPosAndTextsIt != boardPosAndTexts.end() && boardPosAndTextsIt->first == pos) {
                boardTextUTF8 = std::move(boardPosAndTextsIt->second);
                boardPosAndTextsIt++;
                printThisPos = true;
            }

            // Print this position if it has DBRecord or ctx.board text
            if (printThisPos) {
                auto [cx, cy] = outputCoordConvert(pos, ctx.board->size(), Config::GeneralCfg.ioCoordMode);
                MESSAGEL("DATABASE " << cx << ' ' << cy << ' ' << displayLabelValue << ' '
                                     << record.value << ' ' << record.depth() << ' '
                                     << int(record.bound()) << ' ' << int(!record.comment().empty())
                                     << ' ' << UTF8ToConsoleCP(boardTextUTF8));
            }
        }

        MESSAGEL("DATABASE DONE");

        // Insert a new none record if no record is found at this position
        if (!Database::DatabaseCfg.search.readonlyMode) {
            DBRecord record;
            if (ctx.board->ply() > 0 && !dbClient.query(*ctx.board, ctx.options.rule, record))
                dbClient.save(*ctx.board, ctx.options.rule, DBRecord {LABEL_NONE}, OverwriteRule::Disabled);
        }
    }
}

void queryDatabaseOne(DBCommandContext &ctx, bool getPosition)
{
    using namespace ::Database;
    if (getPosition)
        getDatabasePosition(ctx);

    if (ctx.storage) {
        DBClient dbClient(*ctx.storage, RECORD_MASK_ALL);
        DBRecord record;
        if (dbClient.query(*ctx.board, ctx.options.rule, record))
            MESSAGEL("DATABASE ONE " << int(record.label) << ' ' << record.value << ' '
                                     << record.depth() << ' ' << int(record.bound()) << ' '
                                     << record.displayLabel());
        else
            MESSAGEL("DATABASE ONE 0 0 0 0");
    }
}

void queryDatabaseText(DBCommandContext &ctx, bool getPosition)
{
    using namespace ::Database;
    if (getPosition)
        getDatabasePosition(ctx);

    if (ctx.storage) {
        DBClient dbClient(*ctx.storage, RECORD_MASK_ALL);
        DBRecord record;
        if (dbClient.query(*ctx.board, ctx.options.rule, record) && !record.isNull())
            MESSAGEL("DATABASE TEXT " << std::quoted(UTF8ToConsoleCP(record.comment())));
        else
            MESSAGEL("DATABASE TEXT \"\"");
    }
}

void editDatabaseTVD(DBCommandContext &ctx)
{
    using namespace ::Database;
    int updateMask, newLabel, newValue, newDepth;
    ctx.in >> updateMask >> newLabel >> newValue >> newDepth;
    getDatabasePosition(ctx);

    if (ctx.storage && !Database::DatabaseCfg.search.readonlyMode) {
        switch (newLabel) {
        case 'W': newLabel = LABEL_WIN; break;
        case 'L': newLabel = LABEL_LOSE; break;
        case 'D': newLabel = LABEL_DRAW; break;
        case 'X': newLabel = LABEL_BLOCKMOVE; break;
        default: break;
        }

        DBClient dbClient(*ctx.storage, (DBRecordMask)updateMask);
        DBRecord record {DBLabel(newLabel), DBValue(newValue)};
        record.setDepthBound(newDepth, BOUND_EXACT);
        dbClient.save(*ctx.board, ctx.options.rule, record, OverwriteRule::Always);
        dbClient.sync();
        queryDatabaseOne(ctx, false);
    }
}

void editDatabaseText(DBCommandContext &ctx)
{
    using namespace ::Database;
    std::string newText;
    ctx.in >> std::ws >> std::quoted(newText);
    getDatabasePosition(ctx);

    if (ctx.storage && !Database::DatabaseCfg.search.readonlyMode) {
        DBClient dbClient(*ctx.storage, RECORD_MASK_TEXT);
        DBRecord record;
        if (!dbClient.query(*ctx.board, ctx.options.rule, record))
            record = DBRecord {LABEL_NONE};

        record.setComment(ConsoleCPToUTF8(newText));
        dbClient.save(*ctx.board, ctx.options.rule, record, OverwriteRule::Always);
    }
}

void editDatabaseBoardLabel(DBCommandContext &ctx)
{
    using namespace ::Database;
    std::string coordStr, newText;
    ctx.in >> coordStr;
    ctx.in.ignore(1);
    std::getline(ctx.in, newText);
    getDatabasePosition(ctx);

    std::stringstream ss;
    ss << coordStr;
    auto pos = parseLegalCoord(ss, *ctx.board);
    if (!pos.has_value())
        return;

    if (ctx.storage && !Database::DatabaseCfg.search.readonlyMode) {
        DBClient dbClient(*ctx.storage, RECORD_MASK_TEXT);
        dbClient.setBoardText(*ctx.board, ctx.options.rule, *pos, ConsoleCPToUTF8(newText));
    }
}

void deleteDatabaseAll(DBCommandContext &ctx, bool getPosition)
{
    using namespace ::Database;

    ctx.in >> std::ws;
    std::function<DBClient::DelType(DBRecord &)> deleteFilter = nullptr;
    if (auto firstChar = std::toupper(ctx.in.peek());
        firstChar == 'W' || firstChar == 'L' || firstChar == 'N') {
        std::string deleteType;
        ctx.in >> deleteType;
        upperInplace(deleteType);

        // Every delete type means "delete the records matching a label predicate,
        // keep the rest"; the RECURSIVE suffix only changes how non-matching records
        // are kept (NoDeleteRecursive also prunes the tree walk below them).
        bool               recursiveKeep = false;
        std::string        baseType      = deleteType;
        constexpr std::string_view RecursiveSuffix = "RECURSIVE";
        if (baseType.size() > RecursiveSuffix.size()
            && baseType.compare(baseType.size() - RecursiveSuffix.size(),
                                RecursiveSuffix.size(),
                                RecursiveSuffix)
                   == 0) {
            recursiveKeep = true;
            baseType.resize(baseType.size() - RecursiveSuffix.size());
        }

        std::function<bool(DBRecord &)> match = nullptr;
        if (baseType == "W")
            match = [](DBRecord &record) { return record.label == LABEL_WIN; };
        else if (baseType == "L")
            match = [](DBRecord &record) { return record.label == LABEL_LOSE; };
        else if (baseType == "WL")
            match = [](DBRecord &record) {
                return record.label == LABEL_WIN || record.label == LABEL_LOSE;
            };
        else if (baseType == "WLD")
            match = [](DBRecord &record) { return isDeterminedLabel(record.label); };
        else if (baseType == "NONWL")
            match = [](DBRecord &record) {
                return !(record.label == LABEL_WIN || record.label == LABEL_LOSE);
            };
        else if (baseType == "NONWLD")
            match = [](DBRecord &record) { return !isDeterminedLabel(record.label); };
        else if (baseType == "WLNOSTEP")
            match = [](DBRecord &record) {
                Value value = Value(-record.value);
                return record.label == LABEL_WIN && value <= VALUE_MATE_IN_MAX_PLY
                       || record.label == LABEL_LOSE && value >= VALUE_MATED_IN_MAX_PLY;
            };
        else if (baseType == "WLINSTEP") {
            int step;
            ctx.in >> step;
            match = [step](DBRecord &record) {
                Value value = Value(-record.value);
                return (record.label == LABEL_WIN && value > VALUE_MATE_IN_MAX_PLY
                        || record.label == LABEL_LOSE && value < VALUE_MATED_IN_MAX_PLY)
                       && mate_step(value, -1) <= step;
            };
        }
        else
            ERRORL("Unknown database delete type " << deleteType);

        if (match) {
            DBClient::DelType keep =
                recursiveKeep ? DBClient::DelType::NoDeleteRecursive : DBClient::DelType::NoDelete;
            deleteFilter = [match = std::move(match), keep](DBRecord &record) {
                return match(record) ? DBClient::DelType::DeleteRecursive : keep;
            };
        }
    }

    if (getPosition)
        getDatabasePosition(ctx);

    if (ctx.storage && !Database::DatabaseCfg.search.readonlyMode) {
        MESSAGEL("Deleting child records, this might take a while...");
        auto   startTime        = now();
        size_t sizeBeforeDelete = ctx.storage->size();
        {
            DBClient dbClient(*ctx.storage, RECORD_MASK_ALL);
            dbClient.delChildren(*ctx.board, ctx.options.rule, deleteFilter);
        }
        size_t sizeAfterDelete = ctx.storage->size();
        auto   endTime         = now();
        MESSAGEL("Done deleting " << (sizeBeforeDelete - sizeAfterDelete) << " records using "
                                  << (endTime - startTime) << " ms.");

        queryDatabaseAll(ctx, false);
    }
}

void deleteDatabaseOne(DBCommandContext &ctx, bool getPosition)
{
    using namespace ::Database;
    if (getPosition)
        getDatabasePosition(ctx);

    if (ctx.storage && !Database::DatabaseCfg.search.readonlyMode) {
        DBClient dbClient(*ctx.storage, RECORD_MASK_ALL);
        dbClient.del(*ctx.board, ctx.options.rule);
    }
}

void splitDatabase(DBCommandContext &ctx)
{
    auto databasePath = readPathFromInput(ctx.in);
    if (ctx.storage) {
        std::string dbPathUTF8 = databasePath.u8string();
        if (auto dbToSplit = Database::createDBStorage(Database::DatabaseCfg, dbPathUTF8)) {
            auto   startTime  = now();
            size_t writeCount = ::Database::splitDatabase(*ctx.storage,
                                                          *dbToSplit,
                                                          *ctx.board,
                                                          ctx.options.rule);
            auto   endTime    = now();
            MESSAGEL("Write " << writeCount << " records into the split database using "
                              << (endTime - startTime) << " ms.");
        }
    }
}

void mergeDatabase(DBCommandContext &ctx)
{
    auto databasePath = readPathFromInput(ctx.in);
    if (ctx.storage) {
        std::string dbPathUTF8 = databasePath.u8string();
        if (auto dbToMerge = Database::createDBStorage(Database::DatabaseCfg, dbPathUTF8)) {
            size_t writeCount = ::Database::mergeDatabase(*ctx.storage,
                                              *dbToMerge,
                                              Database::DatabaseCfg.search.overwriteRule);
            MESSAGEL("Merged " << writeCount << " out of " << dbToMerge->size()
                               << " records into the database.");
        }
    }
}

}  // namespace Command
