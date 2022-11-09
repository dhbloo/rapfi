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

#include "../core/types.h"

#include <string>

namespace Database {

/// DBLabel represents a one-byte tag that attached to a node in the game DAG.
enum DBLabel : int8_t {
    LABEL_NULL = 0,   /// Null record, default constructed, only stores key in database
    LABEL_NONE = -1,  /// Undetermined result

    LABEL_RESULT_MARKS_BEGIN = 32,

    LABEL_FORCEMOVE = '!',  /// the forced move (will be used as root move if exist)
    LABEL_WIN       = 'w',  /// a winning position
    LABEL_LOSE      = 'l',  /// a losing position
    LABEL_DRAW      = 'd',  /// a draw position
    LABEL_BLOCKMOVE = 'x',  /// a blocked position (will not be considered in search)

    LABEL_RESULT_MARKS_END = 127,
};

/// Return true if the label has a determined result.
constexpr bool isDeterminedLabel(DBLabel label)
{
    return label == LABEL_WIN || label == LABEL_LOSE || label == LABEL_DRAW;
}

inline bool operator==(DBLabel lhs, DBLabel rhs)
{
    return std::tolower(lhs) == std::tolower(rhs);
}
inline bool operator!=(DBLabel lhs, DBLabel rhs)
{
    return !(lhs == rhs);
}

/// DBDepthBound represents two bytes that contain a bound and a depth of one search result.
typedef int16_t DBDepthBound;

/// DBValue represents the (16bit) value from a search result, with the same meaning as Value.
/// A Valid value shoud be in range [-30000, 30000]. Value with absolute numeric that larger
/// than VALUE_MATE_IN_MAX_PLY is consided as a mate value, and steps to mate can be get with
/// mate_step() function.
typedef int16_t DBValue;

/// DBRecordMask specify what parts of a record are selected.
enum DBRecordMask {
    RECORD_MASK_NONE       = 0x0,
    RECORD_MASK_LABEL      = 0x1,
    RECORD_MASK_VALUE      = 0x2,
    RECORD_MASK_DEPTHBOUND = 0x4,
    RECORD_MASK_TEXT       = 0x8,

    RECORD_MASK_LVDB = RECORD_MASK_LABEL | RECORD_MASK_VALUE | RECORD_MASK_DEPTHBOUND,
    RECORD_MASK_ALL  = RECORD_MASK_LVDB | RECORD_MASK_TEXT
};

/// DBRecord contains all the information for one position.
struct DBRecord
{
    DBLabel      label;       // label ('l' or 'w' or '\0', 1 byte)
    DBValue      value;       // value (int16, 2 bytes, optional)
    DBDepthBound depthbound;  // depth & bound (int16, 2 bytes, optional)
    std::string  text;  // utf-8 text message (string ending with '\0', (n3 - 5) bytes, optional)

    /// Return the depth component of a depth bound.
    int depth() const { return int(depthbound) >> 2; }
    /// Return the bound component of a depth bound.
    Bound bound() const { return Bound(int(depthbound) & 0b11); }
    /// Set a new depth and bound for this record.
    DBDepthBound setDepthBound(int depth, Bound bound)
    {
        return depthbound = int16_t((depth << 2) | bound);
    }
    /// Checks if this record is a null record
    bool isNull() const { return label == LABEL_NULL; }
    /// Update label, value, depth, bound of this record
    void update(const DBRecord &rhs, DBRecordMask mask)
    {
        if (mask & RECORD_MASK_LABEL)
            label = rhs.label;
        if (mask & RECORD_MASK_VALUE)
            value = rhs.value;
        if (mask & RECORD_MASK_DEPTHBOUND)
            depthbound = rhs.depthbound;
        if (mask & RECORD_MASK_TEXT) {
            text = rhs.text;
            // Make sure text is not saved as null label
            if (label == LABEL_NULL && !text.empty())
                label = LABEL_NONE;
        }
    }
};

}  // namespace Database
