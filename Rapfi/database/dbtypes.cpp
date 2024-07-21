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

#include "dbtypes.h"

#include "../config.h"
#include "../core/utils.h"

#include <map>

namespace {

constexpr char coordToHex(int coord)
{
    return coord < 10 ? '0' + coord : 'A' + coord - 10;
}
constexpr int hexToCoord(char hex)
{
    return hex <= '9' ? hex - '0' : hex - 'A' + 10;
}

constexpr char BoardTextHeader[] = "@BTXT@";
constexpr int  HeaderLength      = sizeof(BoardTextHeader) - 1;

}  // namespace

namespace Database {

std::string DBRecord::comment() const
{
    size_t start = 0;
    // Skip special board text segment at beginning
    if (hasBoardText()) {
        start = text.find('\b');
        if (start == std::string::npos)
            return {};  // empty comment
        start++;        // skip '\b'
    }

    std::string cmt = text.substr(start);
    // Replace all '\b' in the comment with '\n' for display
    return replaceAll(cmt, "\b", "\n");
}

void DBRecord::setComment(std::string comment)
{
    // Delete all comments in text excluding special board text segment
    if (hasBoardText()) {
        size_t start = text.find('\b');
        if (start != std::string::npos)
            text.erase(start, text.size() - start);
    }
    else
        text.clear();

    // Replace CRLF to LF in comment text for consistency
    trimInplace(replaceAll(comment, "\r\n", "\n"));
    if (comment.empty())
        return;

    if (!text.empty())
        text.push_back('\b');
    // Use null-terminated string here to remove extra '\0'
    text.append(comment.c_str());
}

bool DBRecord::hasBoardText() const
{
    return text.rfind(BoardTextHeader, 0) == 0;  // startswith
}

std::string DBRecord::boardText(Pos canonicalPos)
{
    if (!hasBoardText())
        return {};  // No board text is recorded

    size_t           boardTextSegmentSize = std::min(text.find('\b'), text.size());
    std::string_view boardTextSegment(text.data() + HeaderLength,
                                      boardTextSegmentSize - HeaderLength);

    auto boardTextPairs = split(boardTextSegment, "\n");
    for (std::string_view boardTextPair : boardTextPairs) {
        if (boardTextPair.size() > 2 && hexToCoord(boardTextPair[0]) == canonicalPos.x()
            && hexToCoord(boardTextPair[1]) == canonicalPos.y())
            return std::string(boardTextPair.substr(2));
    }

    return {};  // Did not find a board text
}

std::vector<std::pair<Pos, std::string_view>> DBRecord::getAllBoardTexts() const
{
    if (!hasBoardText())
        return {};  // No board text is recorded

    // A list of (Canonical Pos, Board Text) pairs
    std::vector<std::pair<Pos, std::string_view>> boardTexts;

    size_t           boardTextSegmentSize = std::min(text.find('\b'), text.size());
    std::string_view boardTextSegment(text.data() + HeaderLength,
                                      boardTextSegmentSize - HeaderLength);

    auto boardTextPairs = split(boardTextSegment, "\n");
    for (std::string_view boardTextPair : boardTextPairs) {
        if (boardTextPair.size() > 2) {
            Pos canonicalPos {hexToCoord(boardTextPair[0]), hexToCoord(boardTextPair[1])};
            boardTexts.emplace_back(canonicalPos, boardTextPair.substr(2));
        }
    }

    return boardTexts;
}

void DBRecord::setBoardText(Pos canonicalPos, std::string boardText)
{
    // Make sure no newlines or '\b' characters or whitespaces are in the text
    boardText.erase(std::remove_if(boardText.begin(),
                                   boardText.end(),
                                   [](unsigned char c) {
                                       return c == '\n' || c == '\r' || c == '\b'
                                              || std::isspace(c);
                                   }),
                    boardText.cend());

    std::string boardTextSegment = BoardTextHeader;
    size_t      boardTextCount   = 0;

    // Remove previous board text if it is recorded
    if (hasBoardText()) {
        size_t           boardTextSegmentSize = std::min(text.find('\b'), text.size());
        std::string_view oldBoardTextSegment(text.data() + HeaderLength,
                                             boardTextSegmentSize - HeaderLength);

        auto boardTextPairs = split(oldBoardTextSegment, "\n");
        for (std::string_view boardTextPair : boardTextPairs) {
            if (boardTextPair.size() > 2
                && (hexToCoord(boardTextPair[0]) != canonicalPos.x()
                    || hexToCoord(boardTextPair[1]) != canonicalPos.y())) {
                // Remove extra '\0' due to bug in early implementation
                if (boardTextPair.back() == '\0')
                    boardTextPair.remove_suffix(1);
                boardTextSegment.append(boardTextPair);
                boardTextSegment.push_back('\n');
                boardTextCount++;
            }
        }

        if (boardTextSegmentSize < text.size())
            boardTextSegmentSize++;           // Erase '\b'
        text.erase(0, boardTextSegmentSize);  // Remove board text segment
    }

    if (!boardText.empty()) {
        boardTextSegment.push_back(coordToHex(canonicalPos.x()));
        boardTextSegment.push_back(coordToHex(canonicalPos.y()));
        // Use null-terminated string here to remove extra '\0'
        boardTextSegment.append(boardText.c_str());
        boardTextCount++;
    }

    // Do not insert board text segment if no board text
    if (boardTextCount == 0)
        return;

    // Insert board text segment to text entry of this dbRecord
    if (text.empty())
        text = std::move(boardTextSegment);
    else {
        boardTextSegment.push_back('\b');
        text.insert(0, boardTextSegment);
    }
}

void DBRecord::clearAllBoardText()
{
    if (text.rfind(BoardTextHeader, 0) != 0)
        return;

    size_t boardTextSegmentSize = std::min(text.find('\b'), text.size());
    if (boardTextSegmentSize < text.size())
        boardTextSegmentSize++;           // Erase '\b'
    text.erase(0, boardTextSegmentSize);  // Remove board text segment
}

void DBRecord::copyBoardTextFrom(const DBRecord &rhs, bool overwrite)
{
    // If rhs has no board text, skip
    if (!rhs.hasBoardText())
        return;

    // If this has no board text, copy directly from rhs
    if (!hasBoardText()) {
        size_t boardTextSegmentSize = std::min(rhs.text.find('\b'), rhs.text.size());
        if (text.empty()) {
            text = std::string_view {text.data(), boardTextSegmentSize};
        }
        else {
            text.insert(0, rhs.text, 0, boardTextSegmentSize);
            text.insert(boardTextSegmentSize, "\b");
        }
    }

    // Merge board texts position by position
    std::map<Pos, std::string> boardTextToMerge;

    // Get all positions in rhs that have board text
    {
        size_t           boardTextSegmentSize = std::min(rhs.text.find('\b'), rhs.text.size());
        std::string_view boardTextSegment(rhs.text.data() + HeaderLength,
                                          boardTextSegmentSize - HeaderLength);
        for (std::string_view boardTextPair : split(boardTextSegment, "\n")) {
            if (boardTextPair.size() > 2) {
                Pos p {hexToCoord(boardTextPair[0]), hexToCoord(boardTextPair[1])};

                // Remove extra '\0' due to bug in early implementation
                if (boardTextPair.back() == '\0')
                    boardTextPair.remove_suffix(1);

                boardTextToMerge.emplace(p, boardTextPair);
            }
        }
    }

    std::string boardTextSegment = BoardTextHeader;
    size_t      boardTextCount   = 0;
    // Get all positions in this and remove either one according to overwrite flag
    {
        size_t           boardTextSegmentSize = std::min(text.find('\b'), text.size());
        std::string_view oldBoardTextSegment(text.data() + HeaderLength,
                                             boardTextSegmentSize - HeaderLength);

        for (std::string_view boardTextPair : split(oldBoardTextSegment, "\n")) {
            if (boardTextPair.size() <= 2)
                continue;

            Pos p {hexToCoord(boardTextPair[0]), hexToCoord(boardTextPair[1])};

            // Remove extra '\0' due to bug in early implementation
            if (boardTextPair.back() == '\0')
                boardTextPair.remove_suffix(1);

            if (auto it = boardTextToMerge.find(p); it != boardTextToMerge.end()) {
                if (overwrite)
                    boardTextSegment.append(it->second.c_str());
                else
                    boardTextSegment.append(boardTextPair);
                boardTextToMerge.erase(it);
            }
            else {
                boardTextSegment.append(boardTextPair);
            }
            boardTextSegment.push_back('\n');
            boardTextCount++;
        }

        if (boardTextSegmentSize < text.size())
            boardTextSegmentSize++;           // Erase '\b'
        text.erase(0, boardTextSegmentSize);  // Remove board text segment
    }

    // Add remaining board texts in rhs
    for (const auto &[p, boardText] : boardTextToMerge) {
        boardTextSegment.append(boardText.c_str());
        boardTextSegment.push_back('\n');
        boardTextCount++;
    }

    // Do not insert board text segment if no board text
    if (boardTextCount == 0)
        return;

    // Insert board text segment to text entry of this dbRecord
    if (text.empty())
        text = std::move(boardTextSegment);
    else {
        boardTextSegment.push_back('\b');
        text.insert(0, boardTextSegment);
    }
}

std::string DBRecord::displayLabel() const
{
    std::string displayLabel;

    if (label > 0) {
        displayLabel.push_back(label);

        if (label == LABEL_WIN || label == LABEL_LOSE) {
            Value mateValue = Value(-value);
            if (label == LABEL_WIN && mateValue > VALUE_MATE_IN_MAX_PLY
                || label == LABEL_LOSE && mateValue < VALUE_MATED_IN_MAX_PLY)
                displayLabel += std::to_string(mate_step(mateValue, -1));
            else
                displayLabel.push_back('*');
        }
    }
    else if (label == LABEL_NONE && bound() == BOUND_EXACT) {
        float winRate      = Config::valueToWinRate(Value(-value));
        int   winRateLabel = std::clamp(int(winRate * 100), 0, 99);

        displayLabel = std::to_string(winRateLabel);
        displayLabel.push_back('%');
    }

    return displayLabel;
}

}  // namespace Database
