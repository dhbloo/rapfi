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

#include "../core/iohelper.h"
#include "../game/board.h"

#include <iomanip>
#include <sstream>
#include <thread>
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
        if (move == Pos::PASS)
            board.undoPassMove();
        else
            board.undo(rule);
    }

    DBRecord record, tempRecord;
    DBKey    key;
    while (path.size()) {
        Pos pos = path.back();
        path.pop_back();

        if (board.getLastMove() == Pos::PASS)
            board.doPassMove();
        else
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
    // Find all potential children of this board position
    std::vector<Pos> toCopyPos;
    toCopyPos.reserve(board.movesLeft());
    FOR_EVERY_EMPTY_POS(&board, pos) { toCopyPos.push_back(pos); }

    // Do permutation based on thread id
    if (threadId > 0) {
        for (size_t i = 0; i < toCopyPos.size(); i += (threadId + 1) / 2) {
            size_t swapIndex = (i * (threadId + 1)) % toCopyPos.size();
            std::swap(toCopyPos[i], toCopyPos[swapIndex]);
        }
    }

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

namespace Renlib {

// Renlib Tree Traversal code adopted from original implementation by mc_14, Nov, 2016.
class RenlibReader
{
public:
    /// The callback type for traversing a node.
    using CallbackFunc = void(const Board       &board,
                              bool               hasTag,
                              const std::string *text,
                              const std::string *comment);

    /// Open a lib file for reading.
    RenlibReader(std::istream &libStream);

    /// Traverse a lib file and call the callback function for each node.
    /// @return The number of nodes read.
    size_t traverse(int boardSize, Rule rule, std::function<CallbackFunc> callback);

private:
    static constexpr int BYTE_QUEUE_SIZE = 3;

    // about file header
    static constexpr int HEADER_SIZE              = 20;
    static constexpr int MAJOR_FILE_VERSION_INDEX = 8;
    static constexpr int MINOR_FILE_VERSION_INDEX = 9;

    enum NodeFlag {
        MASK_TEXT    = 0x01,
        MASK_NOMOVE  = 0x02,
        MASK_START   = 0x04,
        MASK_COMMENT = 0x08,
        MASK_TAG     = 0x10,
        MASK_NOCHILD = 0x40,
        MASK_SIBLING = 0x80,
    };

    /// LibNode represents a node in lib file, which contains current move, flag and text/comment.
    struct LibNode
    {
        Pos         move;
        NodeFlag    flag;
        std::string text;
        std::string comment;

        bool hasText() const { return flag & MASK_TEXT; }
        bool hasComment() const { return flag & MASK_COMMENT; }
        bool hasTag() const { return flag & MASK_TAG; }
        bool hasChild() const { return !(flag & MASK_NOCHILD); }
        bool hasSibling() const { return flag & MASK_SIBLING; }
    };

    std::istream &in;
    struct ByteElement
    {
        uint8_t ch;
        bool    ok;
    } byteQueue[BYTE_QUEUE_SIZE];

    bool        hasNextNode() { return byteQueue[0].ok && byteQueue[1].ok; }
    void        fetchOneByte();
    uint8_t     popByte();
    std::string readFileHead();
    LibNode     readNode();
    size_t      processNode(Board                       &board,
                            Rule                         rule,
                            const LibNode               *node,
                            std::function<CallbackFunc> &callback);
};

}  // namespace Renlib

namespace Database {

void databaseToCSVFile(::Database::DBStorage &dbStorage, std::ostream &csvStream)
{
    constexpr size_t BatchSize = 2000;
    constexpr char   Sep       = ',';

    DBStorage::Cursor                       cursor {0};
    std::vector<std::pair<DBKey, DBRecord>> dbKeyRecords;
    dbKeyRecords.reserve(BatchSize);

    csvStream << "index" << Sep << "key" << Sep << "label" << Sep << "value" << Sep << "depth"
              << Sep << "bound" << Sep << "text" << '\n';

    size_t count = 0;
    do {
        dbKeyRecords.clear();
        cursor = dbStorage.scan(cursor, BatchSize, dbKeyRecords);

        for (const auto &[dbKey, dbRecord] : dbKeyRecords) {
            csvStream << count << Sep << dbKey << Sep;

            if (dbRecord.isNull())
                csvStream << "(null)";
            else if (dbRecord.label == LABEL_NONE)
                csvStream << "(none)";
            else if (std::isprint(dbRecord.label) && !std::isspace(dbRecord.label))
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

            csvStream << Sep << std::quoted(dbRecord.text) << '\n';
            count++;
        }

    } while (cursor);
}

size_t mergeDatabase(DBStorage &dbDst, DBStorage &dbSrc, OverwriteRule owRule)
{
    constexpr size_t BatchSize = 2000;

    DBStorage::Cursor                       cursor {0};
    std::vector<std::pair<DBKey, DBRecord>> dbRecords;

    size_t   writeCount = 0;
    DBRecord oldRecord;
    do {
        dbRecords.clear();
        cursor = dbSrc.scan(cursor, BatchSize, dbRecords);

        for (const auto &[dbKey, dbRecord] : dbRecords) {
            if (!dbDst.get(dbKey, oldRecord, RECORD_MASK_ALL)
                || checkOverwrite(oldRecord,
                                  dbRecord,
                                  owRule,
                                  Config::DatabaseOverwriteExactBias,
                                  0)) {
                dbDst.set(dbKey, dbRecord, RECORD_MASK_ALL);
                writeCount++;
            }
        }

    } while (cursor);

    return writeCount;
}

size_t splitDatabase(DBStorage &dbSrc, DBStorage &dbDst, const Board &board, Rule rule)
{
    const size_t             numDeleteThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    threads.reserve(numDeleteThreads);

    size_t sizeBeforeSplit = dbDst.size();

    for (int i = 0; i < numDeleteThreads; i++) {
        threads.emplace_back(
            [&, rule, id = i](std::unique_ptr<Board> board) {
                copyDatabaseBranch(dbSrc, dbDst, *board, rule, id, 0);
            },
            std::make_unique<Board>(board, nullptr));
    }

    copyDatabasePathToRoot(dbSrc, dbDst, const_cast<Board &>(board), rule);

    for (auto &th : threads)
        th.join();

    size_t sizeAfterSplit = dbDst.size();
    return sizeAfterSplit - sizeBeforeSplit;
}

size_t importLibToDatabase(DBStorage &dbDst, std::istream &libStream, Rule rule, int boardSize)
{
    size_t writeCount = 0;

    try {
        Renlib::RenlibReader libReader(libStream);

        auto callback = [&](const Board       &board,
                            bool               hasTag,
                            const std::string *text,
                            const std::string *comment) {
            DBKey    key = constructDBKey(board, rule);
            DBRecord oldRecord, newRecord = {LABEL_NONE};

            if (!text && !comment) {
                // Write empty record if no record exists
                if (!dbDst.get(key, oldRecord)) {
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
                else if (t[0] == Config::DatabaseLibBlackWinMark && board.sideToMove() == BLACK) {
                    newRecord.label = LABEL_WIN;
                }
                else if (t[0] == Config::DatabaseLibWhiteWinMark && board.sideToMove() == WHITE) {
                    newRecord.label = LABEL_WIN;
                }
                else if (t[0] == Config::DatabaseLibBlackLoseMark && board.sideToMove() == BLACK) {
                    newRecord.label = LABEL_LOSE;
                }
                else if (t[0] == Config::DatabaseLibWhiteLoseMark && board.sideToMove() == WHITE) {
                    newRecord.label = LABEL_LOSE;
                }
                else if ((t[0] == 'v' || t[0] == 'm') && t.length() > 1) {
                    newRecord.label = LABEL_NONE;
                    Value value =
                        std::clamp((Value)std::atoi(t.c_str() + 1), VALUE_EVAL_MIN, VALUE_EVAL_MAX);
                    newRecord.value = DBValue(t[0] == 'v' ? -value : value);
                    newRecord.setDepthBound(0, BOUND_EXACT);
                }
                else if (t.length() == 1)
                    newRecord.label = (DBLabel)std::toupper(t[0]);
                else
                    newRecord.text = t;
            }

            if (comment && !Config::DatabaseLibIgnoreComment) {
                if (newRecord.text.empty())
                    newRecord.text = *comment;
                else {
                    newRecord.text.push_back('\b');
                    newRecord.text.append(*comment);
                }
            }

            if (!dbDst.get(key, oldRecord)
                || checkOverwrite(oldRecord, newRecord, OverwriteRule::BetterValue)) {
                dbDst.set(key,
                          newRecord,
                          DBRecordMask((text ? RECORD_MASK_LVDB : RECORD_MASK_NONE)
                                       | (comment ? RECORD_MASK_TEXT : RECORD_MASK_NONE)));
                writeCount++;
            }
        };

        libReader.traverse(boardSize, rule, callback);
    }
    catch (const std::exception &e) {
        ERRORL("Error when reading lib file: " << e.what());
    }

    return writeCount;
}

}  // namespace Database

namespace Renlib {

RenlibReader::RenlibReader(std::istream &libStream) : in(libStream)
{
    for (int i = 0; i < BYTE_QUEUE_SIZE; i++)
        fetchOneByte();  // init buffer
}

size_t RenlibReader::traverse(int boardSize, Rule rule, std::function<CallbackFunc> callback)
{
    if (boardSize > 15)
        throw std::runtime_error("currently only boardsize <= 15 is supported");

    // 1) Read header
    readFileHead();

    // 2) Read root node
    size_t nodeCount = 0;
    if (hasNextNode()) {
        LibNode rootNode = readNode();
        nodeCount++;

        // remove "ROOT" move for old Renlib format
        if (rootNode.move == Pos::NONE)
            rootNode = readNode();

        auto board = std::make_unique<Board>(boardSize);
        board->newGame(rule);

        nodeCount += processNode(*board, rule, &rootNode, callback);
    }
    else
        throw std::runtime_error("no root node in lib");

    return nodeCount;
}

void RenlibReader::fetchOneByte()
{
    // move 1 byte ahead
    for (int i = 0; i < BYTE_QUEUE_SIZE - 1; i++)
        byteQueue[i] = byteQueue[i + 1];

    // insert new
    if (!in.eof()) {
        char tmpc;
        in.get(tmpc);
        byteQueue[BYTE_QUEUE_SIZE - 1].ch = uint8_t(tmpc);
        byteQueue[BYTE_QUEUE_SIZE - 1].ok = true;
    }
    else {
        byteQueue[BYTE_QUEUE_SIZE - 1].ch = uint8_t(0xff);
        byteQueue[BYTE_QUEUE_SIZE - 1].ok = false;
    }
}

uint8_t RenlibReader::popByte()
{
    auto byte = byteQueue[0];
    fetchOneByte();
    if (!byte.ok)
        throw std::runtime_error("Poping invalid byte!");
    return byte.ch;
}

std::string RenlibReader::readFileHead()
{
    char header[HEADER_SIZE + 1];
    for (int i = 0; i < 20; i++) {
        uint8_t byte = popByte();
        header[i] = byte = 0xff ? '_' : (char)byte;
    }
    header[HEADER_SIZE]              = 0;
    header[MAJOR_FILE_VERSION_INDEX] = header[MAJOR_FILE_VERSION_INDEX] + '0';
    header[MINOR_FILE_VERSION_INDEX] = header[MINOR_FILE_VERSION_INDEX] + '0';

    return header;
}

RenlibReader::LibNode RenlibReader::readNode()
{
    uint8_t move = popByte();
    uint8_t flag = popByte();
    LibNode node {
        move ? Pos {(move & 0x0f) - 1, (move & 0xf0) >> 4} : Pos::NONE,
        NodeFlag(flag),
    };

    if (node.hasText())
        popByte(), popByte();  // Skip 0x00, 0x01

    if (node.hasComment()) {
        std::stringstream ss;
        while (true) {  // look ahead two byte
            uint8_t byte1 = popByte();
            uint8_t byte2 = popByte();
            if (byte1) {
                ss << (char)byte1;
                if (byte2)
                    ss << (char)byte2;
            }
            if (!byte1 || !byte2)
                break;
        }
        node.comment = ss.str();
    }

    if (node.hasText()) {
        std::stringstream ss;
        while (true) {  // look ahead two byte
            uint8_t byte1 = popByte();
            uint8_t byte2 = popByte();
            if (byte1) {
                ss << (char)byte1;
                if (byte2)
                    ss << (char)byte2;
            }
            if (!byte1 || !byte2)
                break;
        }
        node.text = ss.str();
    }

    return node;
}

size_t RenlibReader::processNode(Board                       &board,
                                 Rule                         rule,
                                 const LibNode               *node,
                                 std::function<CallbackFunc> &callback)
{
    size_t  nodeCount = 0;
    LibNode siblingNode;

    do {
        bool ignoreChildren = false;
        if (node->move == Pos::NONE) {
            if (board.passMoveCount() >= MAX_PASS_MOVES - 1)
                throw std::runtime_error("too many pass move");
            board.doPassMove();
        }
        else if (board.isInBoard(node->move)) {
            if (board.isEmpty(node->move)) {
                board.move(rule, node->move);

                // Call callback function for this node
                if (callback)
                    callback(board,
                             node->hasTag(),
                             node->hasText() ? &node->text : nullptr,
                             node->hasComment() ? &node->comment : nullptr);
            }
            else {
                // Ignore this invalid branch
                ignoreChildren = true;
            }
        }
        else
            throw std::runtime_error("invalid move in lib");

        // Recursive process all child nodes
        if (node->hasChild()) {
            if (hasNextNode()) {
                LibNode childNode = readNode();
                nodeCount++;

                if (ignoreChildren) {
                    std::function<CallbackFunc> emptyCallback;
                    nodeCount += processNode(board, rule, &childNode, emptyCallback);
                }
                else
                    nodeCount += processNode(board, rule, &childNode, callback);
            }
            else
                throw std::runtime_error("no left child node in lib");
        }

        // Undo the move
        if (!ignoreChildren) {
            if (board.getLastMove() == Pos::PASS)
                board.undoPassMove();
            else
                board.undo(rule);
        }

        // Process next sibling node
        if (node->hasSibling()) {
            if (hasNextNode()) {
                siblingNode = readNode();
                nodeCount++;

                node = &siblingNode;
            }
            else
                throw std::runtime_error("no right sibling node in lib");
        }
        else
            node = nullptr;  // stop the loop

    } while (node);

    return nodeCount;
}

}  // namespace Renlib
