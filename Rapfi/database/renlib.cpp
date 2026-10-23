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

#include "renlib.h"

#include "../core/string.h"
#include "../game/board.h"
#include "dbclient.h"

#include <map>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <utility>

namespace {

using Renlib::MAX_RENLIB_BOARD_SIZE;

/// Size of the fixed lib file header in bytes.
constexpr int HEADER_SIZE = 20;

/// Flag bits of a lib node's flag byte.
enum NodeFlag {
    MASK_TEXT    = 0x01,
    MASK_NOMOVE  = 0x02,
    MASK_START   = 0x04,
    MASK_COMMENT = 0x08,
    MASK_TAG     = 0x10,
    MASK_NOCHILD = 0x40,
    MASK_SIBLING = 0x80,
};

// Renlib Tree Traversal code adopted from original implementation by mc_14, Nov, 2016.
class RenlibReader
{
public:
    /// Open a lib file for reading.
    RenlibReader(std::istream &libStream);

    /// Traverse a lib file and call the callback function for each node.
    /// @return The number of nodes read.
    size_t traverse(int boardSize, Rule rule, std::function<Renlib::TraverseCallback> callback);

private:
    static constexpr int BYTE_QUEUE_SIZE = 3;

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

    bool    hasNextNode() { return byteQueue[0].ok && byteQueue[1].ok; }
    void    fetchOneByte();
    uint8_t popByte();
    void    skipFileHeader();
    LibNode readNode();
    size_t  processNode(Board                                   &board,
                        Rule                                     rule,
                        const LibNode                           *node,
                        std::function<Renlib::TraverseCallback> &callback,
                        int                                      depth);
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
    std::ostream &out;

    // Current node state for writing
    Pos         currentMove;
    NodeFlag    currentFlags;
    std::string currentText;
    std::string currentComment;

    void   writeFileHeader();
    void   writeByte(uint8_t byte);
    void   writeNode(int boardSize);
    size_t exportSubTree(Database::DBClient &dbClient, Board &board, Rule rule);
};

RenlibReader::RenlibReader(std::istream &libStream) : in(libStream)
{
    for (int i = 0; i < BYTE_QUEUE_SIZE; i++)
        fetchOneByte();  // init buffer
}

size_t RenlibReader::traverse(int                                     boardSize,
                              Rule                                    rule,
                              std::function<Renlib::TraverseCallback> callback)
{
    if (boardSize > MAX_RENLIB_BOARD_SIZE)
        throw std::runtime_error("currently only boardsize <= 15 is supported");

    // 1) Read header
    skipFileHeader();

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

        nodeCount += processNode(*board, rule, &rootNode, callback, 0);
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

    // insert new; a byte is only valid if the read actually succeeded (checking
    // eof() alone would treat a failed stream as an endless source of stale bytes)
    char tmpc;
    if (in.get(tmpc)) {
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

void RenlibReader::skipFileHeader()
{
    for (int i = 0; i < HEADER_SIZE; i++)
        popByte();
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
        // Comments are stored with a leading 0x08 marker byte; it is part of
        // the encoding, not of the comment itself.
        if (!node.comment.empty() && node.comment.front() == '\b')
            node.comment.erase(0, 1);
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

size_t RenlibReader::processNode(Board                                   &board,
                                 Rule                                     rule,
                                 const LibNode                           *node,
                                 std::function<Renlib::TraverseCallback> &callback,
                                 int                                      depth)
{
    // A well-formed tree is bounded by the board capacity (stones plus passes);
    // ignored branches add a little extra. Anything deeper is malformed input
    // and must not be allowed to exhaust the native stack.
    if (depth > 2 * board.cellCount() + 64)
        throw std::runtime_error("lib tree too deep");

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
                    std::function<Renlib::TraverseCallback> emptyCallback;
                    nodeCount += processNode(board, rule, &childNode, emptyCallback, depth + 1);
                }
                else
                    nodeCount += processNode(board, rule, &childNode, callback, depth + 1);
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
    if (board.size() > MAX_RENLIB_BOARD_SIZE)
        throw std::runtime_error("currently only boardsize <= 15 is supported");

    // 1) Write header
    writeFileHeader();

    // 2) Create a copy of the board for modifications during traversal
    Board boardCopy(board, nullptr);

    // 3) Set up root node (but don't write it, just start with children)
    currentMove    = Pos::NONE;  // No root node to write
    currentFlags   = static_cast<NodeFlag>(0);
    currentText    = "";
    currentComment = "";
    for (int i = 0; i < board.ply(); i++) {
        currentMove = boardCopy.getHistoryMove(i);
        if (i + 1 < board.ply())
            writeNode(board.size());
    }

    Database::DBRecord record;
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
        0xff, 0x52, 0x65, 0x6E, 0x4C, 0x69, 0x62, 0xff,             // 0xff + "RenLib "
        3,                                                          // major version (3.0+)
        0,                                                          // minor version
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
    }
    else {
        // Inverse of the reader's decoding: 1-based x in the low nibble, 0-based y
        // in the high nibble, in raw board coordinates (no display conversion).
        uint8_t moveByte = ((currentMove.y() & 0x0f) << 4) | ((currentMove.x() + 1) & 0x0f);
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
        writeByte(0x08);                         // Comment marker
        for (unsigned char ch : currentComment)  // Write comment bytes
            writeByte(ch);
        // Add padding based on total length (including 0x08 byte)
        size_t totalLength = 1 + currentComment.length();  // Include 0x08 byte
        if (totalLength % 2 == 1) {
            writeByte(0x00);  // Odd total length: append one 0x00 byte
        }
        else {
            writeByte(0x00);  // Even total length: append two 0x00 bytes
            writeByte(0x00);
        }
    }

    // Write text if exists (after comment according to the document)
    if (currentFlags & MASK_TEXT) {
        for (unsigned char ch : currentText)  // Write text bytes
            writeByte(ch);
        // Add padding based on text length
        if (currentText.length() % 2 == 1) {
            writeByte(0x00);  // Odd length: append one 0x00 byte
        }
        else {
            writeByte(0x00);  // Even length: append two 0x00 bytes
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
    auto                       boardTexts = dbClient.queryBoardTexts(board, rule);
    std::map<Pos, std::string> textMap;
    for (const auto &[pos, text] : boardTexts) {
        if (!text.empty()) {
            textMap[pos] = text;
        }
    }

    for (size_t i = 0; i < children.size(); i++) {
        const auto &[pos, record] = children[i];

        // Set up current node info for next recursion
        currentMove    = pos;
        currentFlags   = (i < children.size() - 1) ? MASK_SIBLING : static_cast<NodeFlag>(0);
        currentText    = "";
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

}  // namespace

namespace Renlib {

size_t traverseLib(std::istream                   &libStream,
                   int                             boardSize,
                   Rule                            rule,
                   std::function<TraverseCallback> callback)
{
    RenlibReader libReader(libStream);
    return libReader.traverse(boardSize, rule, std::move(callback));
}

size_t exportDatabase(std::ostream       &libStream,
                      Database::DBClient &dbClient,
                      const Board        &board,
                      Rule                rule)
{
    RenlibWriter libWriter(libStream);
    return libWriter.exportDatabase(dbClient, board, rule);
}

}  // namespace Renlib
