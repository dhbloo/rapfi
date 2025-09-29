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

#include "../core/iohelper.h"
#include "../core/utils.h"
#include "../database/dbclient.h"
#include "../database/dbstorage.h"
#include "../database/dbutils.h"
#include "../database/yxdbstorage.h"
#include "../game/board.h"
#include "argutils.h"
#include "command.h"

#define CXXOPTS_NO_REGEX
#include <cxxopts.hpp>
#include <fstream>
#include <sstream>

using namespace Database;

namespace {

enum class DatabaseType { YixinDB };

auto makeDBCreationOptions(std::string headline)
{
    cxxopts::Options options(headline);
    options.add_options()                                       //
        ("url", "Database URL", cxxopts::value<std::string>())  //
        ("type",
         "Database type",
         cxxopts::value<std::string>()->default_value("yixindb"))  //
        ("yixindb-compressed-save",
         "YixinDB - compressed save",
         cxxopts::value<bool>()->default_value("true"))  //
        ("yixindb-save-on-close",
         "YixinDB - saved on close",
         cxxopts::value<bool>()->default_value("true"))  //
        ("yixindb-backup-on-save",
         "YixinDB - backup before saving",
         cxxopts::value<bool>()->default_value("true"))  //
        ("yixindb-ignore-corrupted",
         "YixinDB - ignore corrupted data",
         cxxopts::value<bool>()->default_value("false"));

    return options;
}

DatabaseType parseDatabaseType(std::string dbTypeStr)
{
    if (dbTypeStr == "yixindb")
        return DatabaseType::YixinDB;
    else
        throw std::invalid_argument("unknown database type " + dbTypeStr);
}

std::unique_ptr<DBStorage> createDBStorage(const cxxopts::ParseResult &args)
{
    std::string databaseURL  = args["url"].as<std::string>();
    std::string databaseType = args["type"].as<std::string>();

    if (databaseType == "yixindb") {
        auto yxdbStorage =
            std::make_unique<YXDBStorage>(pathFromConsoleString(databaseURL),
                                          args["yixindb-compressed-save"].as<bool>(),
                                          args["yixindb-save-on-close"].as<bool>(),
                                          args["yixindb-backup-on-save"].as<bool>(),
                                          args["yixindb-ignore-corrupted"].as<bool>());
        if (yxdbStorage->size() > 0)
            MESSAGEL("Yixindb loaded " << yxdbStorage->size() << " entries from " << databaseURL);
        return yxdbStorage;
    }
    else
        throw std::invalid_argument("unknown database type " + databaseType);
}

std::unique_ptr<DBStorage> createDBStorageFromCmdline(std::string cmdline)
{
    // Split arguments from cmdline
    std::vector<char *> arguments;
    std::istringstream  iss(cmdline);
    std::string         token;
    arguments.push_back(0);
    while (iss >> token) {
        char *arg = new char[token.size() + 1];
        std::copy(token.begin(), token.end(), arg);
        arg[token.size()] = '\0';
        arguments.push_back(arg);
    }

    auto                       options   = makeDBCreationOptions("database create");
    std::unique_ptr<DBStorage> dbStorage = nullptr;
    try {
        auto args = options.parse(arguments.size(), &arguments[0]);
        dbStorage = createDBStorage(args);
    }
    catch (const std::exception &e) {
        ERRORL("Failed to create database: " << e.what());
    }

    for (size_t i = 1; i < arguments.size(); i++)
        delete[] arguments[i];

    return dbStorage;
}

DBKey parseDBKey(std::string keyStr)
{
    auto keyParts = split(keyStr, "-", true);
    if (keyParts.size() != 6)
        throw std::invalid_argument("key must have 6 parts");

    Rule rule;
    if (keyParts[0] == "f")
        rule = FREESTYLE;
    else if (keyParts[0] == "s")
        rule = STANDARD;
    else if (keyParts[0] == "r")
        rule = RENJU;
    else
        throw std::invalid_argument("invalid rule " + std::string(keyParts[0]));

    int boardWidth  = std::atoi(keyParts[1].data());
    int boardHeight = std::atoi(keyParts[2].data());

    std::vector<Pos> blackStones =
        Command::parsePositionString(keyParts[3], boardWidth, boardHeight);
    std::vector<Pos> whiteStones =
        Command::parsePositionString(keyParts[4], boardWidth, boardHeight);

    Color sideToMove;
    if (keyParts[5] == "b")
        sideToMove = BLACK;
    else if (keyParts[5] == "w")
        sideToMove = WHITE;
    else
        throw std::invalid_argument("invalid sideToMove " + std::string(keyParts[5]));

    return {rule, boardWidth, boardHeight, sideToMove, blackStones, whiteStones};
}

DBRecord parseDBRecord(std::istream &is, const DBRecord &defaultRecordValue)
{
    DBRecord                 record = defaultRecordValue;
    std::vector<std::string> recordParts;
    std::string              s;
    while (is.peek() != '\n' && is >> std::quoted(s)) {
        recordParts.push_back(s);
        while (is.peek() == ' ' || is.peek() == '\t')
            is.ignore(1);
    }

    if (recordParts.size() % 2 != 0)
        throw std::invalid_argument("incomplete key-value pair of key " + recordParts.back());

    for (size_t i = 0; i < recordParts.size(); i += 2) {
        auto &recordPartKey   = recordParts[i];
        auto &recordPartValue = recordParts[i + 1];

        if (recordPartKey == "label") {
            if (recordPartValue.length() == 1)
                record.label = static_cast<DBLabel>(recordPartValue[0]);
            else if (recordPartValue.length() > 1 && recordPartValue[0] == '(') {
                size_t rightPar = recordPartValue.find_first_of(')', 1);
                if (rightPar != std::string::npos) {
                    recordPartValue = recordPartValue.substr(1, rightPar - 1);
                    if (recordPartValue == "null")
                        record.label = DBLabel(0);
                    else
                        record.label = DBLabel(std::atoi(recordPartValue.c_str()));
                }
                else
                    throw std::invalid_argument("unclosed label");
            }
            else
                throw std::invalid_argument("invalid label");
        }
        else if (recordPartKey == "value")
            record.value = std::atoi(recordPartValue.data());
        else if (recordPartKey == "depth")
            record.setDepthBound(std::atoi(recordPartValue.data()), record.bound());
        else if (recordPartKey == "bound") {
            Bound bound;
            if (recordPartValue == "lower")
                bound = BOUND_LOWER;
            else if (recordPartValue == "upper")
                bound = BOUND_UPPER;
            else if (recordPartValue == "exact")
                bound = BOUND_EXACT;
            else if (recordPartValue == "none")
                bound = BOUND_NONE;
            else
                throw std::invalid_argument("invalid bound");

            record.setDepthBound(record.depth(), bound);
        }
        else if (recordPartKey == "text")
            record.text = ConsoleCPToUTF8(recordPartValue);
        else
            throw std::invalid_argument("unknown record name " + recordPartKey);
    }

    return record;
}

void printDBQuery(std::ostream &os, DBStorage &dbStorage, const DBKey &dbKey)
{
    DBRecord dbRecord;
    if (dbStorage.get(dbKey, dbRecord)) {
        os << "label ";
        if (dbRecord.isNull())
            os << "(null)";
        else if (dbRecord.label == LABEL_NONE)
            os << "(none)";
        else if (std::isprint(dbRecord.label) && !std::isspace(dbRecord.label))
            os << (char)dbRecord.label;
        else
            os << '(' << (int)dbRecord.label << ')';
        os << " value " << dbRecord.value;
        os << " depth " << dbRecord.depth();
        os << " bound ";
        switch (dbRecord.bound()) {
        case BOUND_EXACT: os << "exact"; break;
        case BOUND_LOWER: os << "lower"; break;
        case BOUND_UPPER: os << "upper"; break;
        default: os << "none"; break;
        }
        os << " text " << std::quoted(UTF8ToConsoleCP(dbRecord.text)) << std::endl;
    }
    else
        os << "(null)" << std::endl;
}

}  // namespace

void Command::database(int argc, char *argv[])
{
    std::unique_ptr<DBStorage> dbStorage;
    std::istringstream         commandStream;

    auto options = makeDBCreationOptions("rapfi database");
    options.add_options()  //
        ("commands",
         "Database commands (seperate multiple commands with ';')",
         cxxopts::value<std::string>()->default_value(""))  //
        ("h,help", "Print database usage");

    try {
        auto args = options.parse(argc, argv);

        if (args.count("help")) {
            std::cout << options.help() << std::endl;
            std::exit(EXIT_SUCCESS);
        }

        {  // Load database command sequences
            std::string commands = args["commands"].as<std::string>();
            std::replace(commands.begin(), commands.end(), ';', '\n');
            if (!commands.empty()) {
                commands.push_back('\n');
                commandStream = std::istringstream(std::move(commands));
            }
        }

        dbStorage = createDBStorage(args);
    }
    catch (const std::exception &e) {
        ERRORL("database command: " << e.what());
        std::exit(EXIT_FAILURE);
    }

    std::istream &is =
        commandStream.peek() == std::istringstream::traits_type::eof() ? std::cin : commandStream;
    std::unique_ptr<Board> board = std::make_unique<Board>(15);
    Rule                   rule  = FREESTYLE;

    for (;;) {
        std::string cmd;
        is >> cmd;

        // Stop the command loop when reaching EOF
        if (is.eof())
            break;
        else if (!is)
            is.clear();

        upperInplace(cmd);

        if (cmd == "QUIT")
            break;
        else if (cmd == "PUTBOARD") {
            std::string ruleStr;
            std::string posStr;
            int         boardWidth, boardHeight;
            is >> ruleStr >> boardWidth >> boardHeight >> posStr;

            try {
                if (boardWidth < 5 || boardWidth > MAX_BOARD_SIZE)
                    throw std::invalid_argument("board width out of range [5,22]");
                if (boardHeight < 5 || boardHeight > MAX_BOARD_SIZE)
                    throw std::invalid_argument("board height out of range [5,22]");
                if (boardWidth != boardHeight)
                    throw std::invalid_argument(
                        "currently board width and board height must be the same");

                rule                      = parseRule(ruleStr);
                std::vector<Pos> position = parsePositionString(posStr, boardWidth, boardHeight);

                if (board->size() != boardWidth)
                    board = std::make_unique<Board>(boardWidth);
                board->newGame(rule);
                for (Pos pos : position)
                    board->move(rule, pos);
                std::cout << "OK" << std::endl;
            }
            catch (const std::exception &e) {
                ERRORL("Invalid input: " << e.what());
            }
        }
        else if (cmd == "CURRENTBOARD") {
            std::cout << rule << ' ' << board->size() << ' ' << board->size() << ' ';
            if (board->nonPassMoveCount() == 0)
                std::cout << "(empty)";
            else
                std::cout << board->positionString();
            std::cout << std::endl;
        }
        else if (cmd == "CURRENTKEY") {
            DBKey key = ::Database::constructDBKey(*board, rule);
            std::cout << key << std::endl;
        }
        else if (cmd == "BOARDGET") {
            DBKey dbKey = constructDBKey(*board, rule);
            printDBQuery(std::cout, *dbStorage, dbKey);
        }
        else if (cmd == "BOARDSET") {
            DBKey dbKey = constructDBKey(*board, rule);
            try {
                DBRecord defaultRecord;
                if (!dbStorage->get(dbKey, defaultRecord))
                    defaultRecord = {};
                DBRecord dbRecord = parseDBRecord(is, defaultRecord);
                dbStorage->set(dbKey, dbRecord, RECORD_MASK_ALL);
                std::cout << "OK" << std::endl;
            }
            catch (const std::exception &e) {
                ERRORL("Invalid record: " << e.what());
            }
        }
        else if (cmd == "BOARDDEL") {
            DBKey dbKey = constructDBKey(*board, rule);
            dbStorage->del(dbKey);
            std::cout << "OK" << std::endl;
        }
        else if (cmd == "KEYGET") {
            try {
                std::string key;
                is >> key;
                DBKey dbKey = parseDBKey(key);
                printDBQuery(std::cout, *dbStorage, dbKey);
            }
            catch (const std::exception &e) {
                ERRORL("Invalid key: " << e.what());
            }
        }
        else if (cmd == "KEYSET") {
            try {
                std::string key;
                is >> key;
                DBKey dbKey = parseDBKey(key);
                try {
                    DBRecord defaultRecord;
                    if (!dbStorage->get(dbKey, defaultRecord))
                        defaultRecord = {};
                    DBRecord dbRecord = parseDBRecord(is, defaultRecord);
                    dbStorage->set(dbKey, dbRecord, RECORD_MASK_ALL);
                    std::cout << "OK" << std::endl;
                }
                catch (const std::exception &e) {
                    ERRORL("Invalid record: " << e.what());
                }
            }
            catch (const std::exception &e) {
                ERRORL("Invalid key: " << e.what());
            }
        }
        else if (cmd == "KEYDEL") {
            try {
                std::string key;
                is >> key;
                DBKey dbKey = parseDBKey(key);
                dbStorage->del(dbKey);
                std::cout << "OK" << std::endl;
            }
            catch (const std::exception &e) {
                ERRORL("Invalid key: " << e.what());
            }
        }
        else if (cmd == "DBSIZE") {
            std::cout << dbStorage->size() << std::endl;
        }
        else if (cmd == "DBFLUSH") {
            dbStorage->flush();
            std::cout << "OK" << std::endl;
        }
        else if (cmd == "DBTOCSV") {
            std::string csvPath;
            std::getline(is, csvPath);
            trimInplace(csvPath);

            if (csvPath.empty())
                databaseToCSVFile(*dbStorage, std::cout);
            else {
                std::ofstream csvStream(csvPath);
                if (csvStream.is_open() && csvStream) {
                    databaseToCSVFile(*dbStorage, csvStream);
                    std::cout << "OK" << std::endl;
                }
                else
                    ERRORL("Failed to open csv file " << csvPath);
            }
        }
        else if (cmd == "DBDELETEALL") {
            MESSAGEL("Deleting all child records, this might take a while...");
            auto   startTime        = now();
            size_t sizeBeforeDelete = dbStorage->size();
            {
                DBClient dbClient(*dbStorage, RECORD_MASK_ALL);
                dbClient.delChildren(*board, rule);
            }
            size_t sizeAfterDelete = dbStorage->size();
            auto   endTime         = now();
            MESSAGEL("Done deleting " << (sizeBeforeDelete - sizeAfterDelete) << " records using "
                                      << (endTime - startTime) << " ms.");
        }
        else if (cmd == "DBMERGE") {
            std::string cmdline;
            std::getline(is, cmdline);
            if (auto dbToMerge = createDBStorageFromCmdline(cmdline)) {
                size_t writeCount =
                    mergeDatabase(*dbStorage, *dbToMerge, Config::DatabaseOverwriteRule);
                MESSAGEL("Merged " << writeCount << " out of " << dbToMerge->size()
                                   << " records into the database.");
            }
        }
        else if (cmd == "DBSPLIT") {
            std::string cmdline;
            std::getline(is, cmdline);
            if (auto dbToSplit = createDBStorageFromCmdline(cmdline)) {
                MESSAGEL("Spliting branch " << board->positionString()
                                            << ", this might take a while...");
                auto   startTime  = now();
                size_t writeCount = splitDatabase(*dbStorage, *dbToSplit, *board, rule);
                auto   endTime    = now();
                MESSAGEL("Write " << writeCount << " records into the spilted database using "
                                  << (endTime - startTime) << " ms.");
            }
        }
        else if (cmd == "LIBTODB") {
            std::string libPath;
            std::getline(is, libPath);
            trimInplace(libPath);

            std::ifstream libStream(libPath, std::ios::binary);
            if (libStream.is_open() && libStream) {
                size_t writeCount = importLibToDatabase(*dbStorage, libStream, rule, board->size());
                MESSAGEL("Imported " << writeCount << " records from lib file " << libPath);
            }
            else
                ERRORL("Failed to open lib file " << libPath);
        }
        else if (cmd == "DBTOLIB") {
            std::string libPath;
            std::getline(is, libPath);
            trimInplace(libPath);

            std::ofstream libStream(libPath, std::ios::binary);
            if (libStream.is_open() && libStream) {
                MESSAGEL("Exporting to lib file " << libPath << ", this might take a while...");
                auto     startTime = now();
                DBClient dbClient(*dbStorage, RECORD_MASK_ALL);
                size_t   nodeCount = exportDatabaseToLib(dbClient, libStream, *board, rule);
                auto     endTime   = now();
                MESSAGEL("Exported " << nodeCount << " nodes to lib file " << libPath << " using "
                                     << (endTime - startTime) << " ms.");
            }
            else
                ERRORL("Failed to open lib file " << libPath);
        }
    }
}
