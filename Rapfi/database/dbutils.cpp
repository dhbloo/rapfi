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
#include "dbclient.h"

#include <iomanip>
#include <map>
#include <sstream>
#include <vector>
#ifdef MULTI_THREADING
    #include <thread>
#endif

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
    // Find all potential children of this board position
    std::vector<Pos> toCopyPos;
    toCopyPos.reserve(board.movesLeft());
    FOR_EVERY_EMPTY_POS(&board, pos)
    {
        toCopyPos.push_back(pos);
    }

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

// Renlib Tree Writing code following the same pattern as RenlibReader.
class RenlibWriter
{
public:
    /// Open a lib file for writing.
    RenlibWriter(std::ostream &libStream);

    /// Export database records to a lib file by traversing the game tree.
    /// @param dbClient The database client to query records from.
    /// @param board The board instance to use for traversal.
    /// @param rule Game rule to use.
    /// @return The number of nodes written.
    size_t exportDatabase(Database::DBClient &dbClient, const Board &board, Rule rule);

private:
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

    std::ostream &out;

    // Current node state for writing
    Pos currentMove;
    NodeFlag currentFlags;
    std::string currentText;
    std::string currentComment;

    void writeFileHeader();
    void writeByte(uint8_t byte);
    void writeNode(int boardSize);
    size_t exportSubTree(Database::DBClient &dbClient, Board &board, Rule rule);
};

}  // namespace Renlib

namespace Database {

void databaseToCSVFile(::Database::DBStorage                               &dbStorage,
                       std::ostream                                        &csvStream,
                       std::function<bool(const DBKey &, const DBRecord &)> filter)
{
    constexpr size_t BatchSize = 2000;
    constexpr char   Sep       = ',';

    DBStorage::Cursor                       cursor {0};
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

            std::stringstream ss;
            ss << std::quoted(dbRecord.text);
            std::string escapedText = ss.str();
            // Esacpe all '\n' with "\\n" in text
            // (\n is used to seperate different board texts and may appear in comments)
            replaceAll(escapedText, "\n", "\\n");
            // Esacpe all '\b' with "\\b" in text
            // (\b is used to seperate sub-sections in the text)
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

    DBStorage::Cursor                       cursor {0};
    std::vector<std::pair<DBKey, DBRecord>> dbRecords;

    size_t   writeCount = 0;
    DBRecord oldRecord;
    do {
        dbRecords.clear();
        cursor = dbSrc.scan(cursor, BatchSize, dbRecords);

        for (auto &[dbKey, dbRecord] : dbRecords) {
            if (!dbDst.get(dbKey, oldRecord, RECORD_MASK_ALL)
                || checkOverwrite(oldRecord,
                                  dbRecord,
                                  owRule,
                                  Config::DatabaseOverwriteExactBias,
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

#if defined(MULTI_THREADING)
    const size_t             numDeleteThreads = std::thread::hardware_concurrency();
    std::vector<std::thread> threads;
    threads.reserve(numDeleteThreads);

    for (int i = 0; i < numDeleteThreads; i++) {
        threads.emplace_back(
            [&, rule, id = i](std::unique_ptr<Board> board) {
                copyDatabaseBranch(dbSrc, dbDst, *board, rule, id, 0);
            },
            std::make_unique<Board>(board, nullptr));
    }

    for (auto &th : threads)
        th.join();
#else
    copyDatabaseBranch(dbSrc, dbDst, const_cast<Board &>(board), rule, 0, 0);
#endif
    copyDatabasePathToRoot(dbSrc, dbDst, const_cast<Board &>(board), rule);

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
                else if (board.ply() > 0 && !Config::DatabaseLibIgnoreBoardText) {
                    // Write parent record for board text
                    Pos    lastMove = board.getLastMove();
                    Board &b        = const_cast<Board &>(board);
                    b.undo(rule);
                    {
                        DBClient dbClient(dbDst, RECORD_MASK_TEXT);
                        dbClient.setBoardText(b, rule, lastMove, LegacyFileCPToUTF8(t));
                    }
                    b.move(rule, lastMove);
                }
            }

            if (comment && !Config::DatabaseLibIgnoreComment) {
                std::string newCmt = LegacyFileCPToUTF8(*comment);
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

        libReader.traverse(boardSize, rule, callback);
    }
    catch (const std::exception &e) {
        ERRORL("Error when reading lib file: " << e.what());
    }

    return writeCount;
}

size_t exportDatabaseToLib(DBClient &dbClient, std::ostream &libStream, const Board &board, Rule rule)
{
    try {
        Renlib::RenlibWriter libWriter(libStream);
        return libWriter.exportDatabase(dbClient, board, rule);
    }
    catch (const std::exception &e) {
        throw std::runtime_error("Failed to export database to lib: " + std::string(e.what()));
    }
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
        if (rootNode.move == Pos::PASS)
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
        move ? Pos {(move & 0x0f) - 1, (move & 0xf0) >> 4} : Pos::PASS,
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
        if (node->move == Pos::PASS) {
            if (board.passMoveCount() >= board.cellCount())
                throw std::runtime_error("too many pass move");
            board.move(rule, Pos::PASS);
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
        if (!ignoreChildren)
            board.undo(rule);

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


RenlibWriter::RenlibWriter(std::ostream &libStream) : out(libStream) {}

size_t RenlibWriter::exportDatabase(Database::DBClient &dbClient, const Board &board, Rule rule)
{
    if (board.size() > 15)
        throw std::runtime_error("currently only boardsize <= 15 is supported");

    // 1) Write header
    writeFileHeader();

    // 2) Create a copy of the board for modifications during traversal
    Board boardCopy(board, nullptr);

    // 3) Set up root node (but don't write it, just start with children)
    currentMove = Pos::NONE;  // No root node to write
    currentFlags = static_cast<NodeFlag>(0);
    currentText = "";
    currentComment = "";
    for (int i = 0; i < board.ply(); i++) {
        currentMove = boardCopy.getHistoryMove(i);
        if (i + 1 < board.ply())
            writeNode(board.size());
    }

    DBRecord record;
    if (dbClient.query(boardCopy, rule, record) && !record.isNull()) {
        std::string label = record.displayLabel();
        currentText.append(upperInplace(label));
        currentComment = record.comment();
    }

    // Export all children from root
    return exportSubTree(dbClient, boardCopy, rule);
}

void RenlibWriter::writeFileHeader()
{
    // Write standard Renlib header (20 bytes)
    // Based on the document: 0xff + "renlib" + version info + padding
    const uint8_t header[HEADER_SIZE] = {
        0xff, 0x52, 0x65, 0x6E, 0x4C, 0x69, 0x62, 0xff,  // 0xff + "RenLib "
        3,    // major version (3.0+)
        0,    // minor version
        0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff  // padding
    };

    for (int i = 0; i < HEADER_SIZE; i++)
        writeByte(header[i]);
}

void RenlibWriter::writeByte(uint8_t byte)
{
    out.put(static_cast<char>(byte));
}

void RenlibWriter::writeNode(int boardSize)
{
    // Write move byte
    if (currentMove == Pos::PASS || currentMove == Pos::NONE) {
        writeByte(0);
    } else {
        // X-coordinate starts from 0, Y-coordinate starts from 1
        int x = outputCoordXConvert(currentMove, boardSize);
        int y = outputCoordYConvert(currentMove, boardSize);
        uint8_t moveByte = (x << 4) | ((y + 1) & 0x0f);
        writeByte(moveByte);
    }

    // Set text and comment flag if text exists
    if (!currentText.empty())
        currentFlags = static_cast<NodeFlag>(currentFlags | MASK_TEXT);
    if (!currentComment.empty())
        currentFlags = static_cast<NodeFlag>(currentFlags | MASK_COMMENT);

    // Write flag byte
    writeByte(static_cast<uint8_t>(currentFlags));

    // Write text placeholder if text exists
    if (currentFlags & MASK_TEXT) {
        writeByte(0x00);
        writeByte(0x01);
    }

    // Write comment if exists (before text according to the document)
    if (currentFlags & MASK_COMMENT) {
        writeByte(0x08);                        // Comment marker
        for (unsigned char ch : currentComment) // Write comment bytes
            writeByte(ch);
        // Add padding based on total length (including 0x08 byte)
        size_t totalLength = 1 + currentComment.length(); // Include 0x08 byte
        if (totalLength % 2 == 1) {
            writeByte(0x00);                    // Odd total length: append one 0x00 byte
        } else {
            writeByte(0x00);                    // Even total length: append two 0x00 bytes
            writeByte(0x00);
        }
    }

    // Write text if exists (after comment according to the document)
    if (currentFlags & MASK_TEXT) {
        for (unsigned char ch : currentText)  // Write text bytes
            writeByte(ch);
        // Add padding based on text length
        if (currentText.length() % 2 == 1) {
            writeByte(0x00);                  // Odd length: append one 0x00 byte
        } else {
            writeByte(0x00);                  // Even length: append two 0x00 bytes
            writeByte(0x00);
        }
    }
}

size_t RenlibWriter::exportSubTree(Database::DBClient &dbClient, Board &board, Rule rule)
{
    // Query all children of current position
    auto children = dbClient.queryChildren(board, rule);
    if (children.empty())
        currentFlags = static_cast<NodeFlag>(currentFlags | MASK_NOCHILD);

    // Write current node first (pre-order traversal)
    // The current node info should be set by the caller
    size_t nodeCount = 0;
    if (currentMove != Pos::NONE || !currentText.empty() || !currentComment.empty()) {
        writeNode(board.size());
        nodeCount++;
    }

    // If no children, return
    if (children.empty())
        return nodeCount;

    // Query board texts for current position
    auto boardTexts = dbClient.queryBoardTexts(board, rule);
    std::map<Pos, std::string> textMap;
    for (const auto &[pos, text] : boardTexts) {
        if (!text.empty()) {
            textMap[pos] = text;
        }
    }

    for (size_t i = 0; i < children.size(); i++) {
        const auto &[pos, record] = children[i];

        // Set up current node info for next recursion
        currentMove = pos;
        currentFlags = (i < children.size() - 1) ? MASK_SIBLING : static_cast<NodeFlag>(0);
        currentText = "";
        currentComment = "";

        // Set text from board texts
        if (textMap.find(pos) != textMap.end())
            currentText = textMap[pos];

        // Make the move
        board.move(rule, pos);

        // Read the display label of this record
        if (!record.isNull()) {
            std::string label = record.displayLabel();
            currentText.append(upperInplace(label));
            currentComment = record.comment();
        }

        // Recursively export this subtree (which will write the node first)
        nodeCount += exportSubTree(dbClient, board, rule);

        // Undo the move
        board.undo(rule);
    }

    return nodeCount;
}

}  // namespace Renlib
