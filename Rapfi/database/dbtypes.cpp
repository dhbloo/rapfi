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

namespace {

constexpr char coordToHex(int coord)
{
    return coord < 10 ? '0' + coord : 'A' + coord - 10;
}
constexpr int hexToCoord(char hex)
{
    return hex <= '9' ? hex - '0' : hex - 'A' + 10;
}

}  // namespace

namespace Database {

std::string DBRecord::comment() const
{
    size_t start = 0;
    // Skip special board text segment at beginning
    if (text.rfind("@BTXT@", 0) == 0) {
        start = text.find('\b');
        if (start == std::string::npos)
            return {};  // empty comment
        start++;        // skip '\b'
    }

    std::string cmt = text.substr(start);
    return replaceAll(cmt, "\b", "\n");
}

void DBRecord::setComment(const std::string &comment)
{
    // Delete all comments in text excluding special board text segment
    if (text.rfind("@BTXT@", 0) == 0) {
        size_t start = text.find('\b');
        if (start != std::string::npos)
            text.erase(start, text.size() - start);
    }
    else
        text.clear();

    if (!text.empty())
        text.push_back('\b');
    text.append(comment);
}

std::string DBRecord::boardText(Pos canonicalPos)
{
    if (text.rfind("@BTXT@", 0) != 0)
        return {};  // No board text is recorded

    size_t boardTextSegmentSize = text.find('\b');
    if (boardTextSegmentSize == std::string::npos)
        boardTextSegmentSize = text.size();
    std::string_view boardTextSegment(text.data() + 6, boardTextSegmentSize - 6);

    auto boardTextPairs = split(boardTextSegment, "\n");
    for (std::string_view boardTextPair : boardTextPairs) {
        if (boardTextPair.size() < 2)
            continue;

        if (hexToCoord(boardTextPair[0]) == canonicalPos.x()
            && hexToCoord(boardTextPair[1]) == canonicalPos.y())
            return std::string(boardTextPair.substr(2));
    }

    return {};  // Did not find a board text
}

void DBRecord::setBoardText(Pos canonicalPos, std::string boardText)
{
    // Make sure no newlines or '\b' characters or whitespaces are in the text
    boardText.erase(
        std::remove_if(boardText.begin(),
                       boardText.end(),
                       [](unsigned char c) { return c == '\n' || c == '\b' || std::isspace(c); }),
        boardText.cend());

    std::string boardTextSegment = "@BTXT@";
    size_t      boardTextCount   = 0;

    // Remove previous board text if it is recorded
    if (text.rfind("@BTXT@", 0) == 0) {
        size_t boardTextSegmentSize = text.find('\b');
        if (boardTextSegmentSize == std::string::npos)
            boardTextSegmentSize = text.size();
        std::string_view oldBoardTextSegment(text.data() + 6, boardTextSegmentSize - 6);

        auto boardTextPairs = split(oldBoardTextSegment, "\n");
        for (std::string_view boardTextPair : boardTextPairs) {
            if (boardTextPair.size() > 2
                && (hexToCoord(boardTextPair[0]) != canonicalPos.x()
                    || hexToCoord(boardTextPair[1]) != canonicalPos.y())) {
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
        boardTextSegment.append(boardText);
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
