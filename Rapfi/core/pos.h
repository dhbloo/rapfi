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

#include <cstdint>
#include <cstdlib>
#include <functional>

// -------------------------------------------------
// Board size & limits

/// Full board size. We use uint64_t bitboard and each cell takes 2 bits.
constexpr int FULL_BOARD_SIZE = 32;
/// Full board array size. This also counts all boundary cells.
constexpr int FULL_BOARD_CELL_COUNT = FULL_BOARD_SIZE * FULL_BOARD_SIZE;
/// Reserved boundary for branchless query in Board class implementation.
constexpr int BOARD_BOUNDARY = 5;
/// The actual maximum board size we can use.
constexpr int MAX_BOARD_SIZE = FULL_BOARD_SIZE - 2 * BOARD_BOUNDARY;
/// The maximum possible moves on an empty board (including a PASS).
constexpr int MAX_MOVES = MAX_BOARD_SIZE * MAX_BOARD_SIZE + 1;

// -------------------------------------------------

/// Pos represents a move coordinate on board
struct Pos
{
public:
    int16_t _pos;

    Pos() = default;
    constexpr Pos(int x, int y) : _pos(((y + BOARD_BOUNDARY) << 5) | (x + BOARD_BOUNDARY)) {}
    constexpr explicit Pos(int16_t _pos) : _pos(_pos) {}
    constexpr int x() const { return (_pos & 31) - BOARD_BOUNDARY; }
    constexpr int y() const { return (_pos >> 5) - BOARD_BOUNDARY; }
    constexpr     operator int() const { return _pos; }
    inline bool   valid() const { return _pos >= PASS._pos && _pos < FULL_BOARD_END._pos; }
    inline int    moveIndex() const { return y() * MAX_BOARD_SIZE + x(); }
    inline bool   isInBoard(int boardWidth, int boardHeight) const;

    static int distance(Pos p1, Pos p2);
    static int lineDistance(Pos p1, Pos p2);

    static const Pos NONE;
    static const Pos PASS;
    static const Pos FULL_BOARD_START;
    static const Pos FULL_BOARD_END;
};

inline constexpr Pos Pos::NONE {0};
inline constexpr Pos Pos::PASS {-1};
inline constexpr Pos Pos::FULL_BOARD_START {0};
inline constexpr Pos Pos::FULL_BOARD_END {FULL_BOARD_CELL_COUNT};

/// Direction represents one of the eight line directions on the board
enum Direction : int16_t {
    UP    = -FULL_BOARD_SIZE,
    LEFT  = -1,
    DOWN  = -UP,
    RIGHT = -LEFT,

    UP_LEFT    = UP + LEFT,
    UP_RIGHT   = UP + RIGHT,
    DOWN_LEFT  = DOWN + LEFT,
    DOWN_RIGHT = DOWN + RIGHT
};

constexpr Direction DIRECTION[] = {RIGHT, DOWN, UP_RIGHT, DOWN_RIGHT};

// -------------------------------------------------
// Pos/Direction related operations

constexpr Pos &operator++(Pos &p)
{
    return p = Pos(p + 1);
}
constexpr Pos &operator--(Pos &p)
{
    return p = Pos(p - 1);
}
constexpr Pos operator++(Pos &p, int)
{
    Pos tmp = p;
    ++p;
    return tmp;
}
constexpr Pos operator--(Pos &p, int)
{
    Pos tmp = p;
    --p;
    return tmp;
}

// Additional operators to add a Direction to a Pos

constexpr Pos operator+(Pos p, Direction i)
{
    return Pos(int(p) + i);
}
constexpr Pos operator-(Pos p, Direction i)
{
    return Pos(int(p) - i);
}
inline Pos &operator+=(Pos &p, Direction i)
{
    return p = Pos(int(p) + i);
}
inline Pos &operator-=(Pos &p, Direction i)
{
    return p = Pos(int(p) - i);
}

/// Check whether the pos is inside a board with width and height.
inline bool Pos::isInBoard(int boardWidth, int boardHeight) const
{
    int X = x(), Y = y();
    return X >= 0 && X < boardWidth && Y >= 0 && Y < boardHeight;
}

/// Return the Chebyshev Distance between two pos.
/// @note If any of the pos is a pass or none, it will return -1.
inline int Pos::distance(Pos p1, Pos p2)
{
    if (p1 <= Pos::NONE || p2 <= Pos::NONE)
        return -1;

    int xDist = std::abs(p1.x() - p2.x());
    int yDist = std::abs(p1.y() - p2.y());
    return xDist < yDist ? yDist : xDist;
}

/// Return the line distance between the two pos.
/// If two pos lies in a line, return the Chebyshev Distance of the two pos.
/// Otherwise, it will return FULL_BOARD_SIZE.
/// @note If any of the pos is a pass or none, it will return -1.
inline int Pos::lineDistance(Pos p1, Pos p2)
{
    if (p1 <= Pos::NONE || p2 <= Pos::NONE)
        return -1;

    int xDelta = p1.x() - p2.x();
    int yDelta = p1.y() - p2.y();

    if (xDelta == 0)
        return std::abs(yDelta);
    else if (yDelta == 0 || xDelta - yDelta == 0 || xDelta + yDelta == 0)
        return std::abs(xDelta);
    else
        return FULL_BOARD_SIZE;
}

template <>
struct std::hash<Pos>
{
    std::size_t operator()(Pos const &p) const noexcept { return p; }
};

constexpr Direction operator+(Direction d1, Direction d2)
{
    return Direction(int(d1) + int(d2));
}
constexpr Direction operator*(int i, Direction d)
{
    return Direction(i * int(d));
}
constexpr Direction operator*(Direction d, int i)
{
    return Direction(int(d) * i);
}

// -------------------------------------------------

/// TransformType represents one of the eight transforms.
enum TransformType {
    IDENTITY,    // (x, y) -> (x, y)
    ROTATE_90,   // (x, y) -> (y, s - x)
    ROTATE_180,  // (x, y) -> (s - x, s - y)
    ROTATE_270,  // (x, y) -> (s - y, x)
    FLIP_X,      // (x, y) -> (x, s - y)
    FLIP_Y,      // (x, y) -> (s - x, y)
    FLIP_XY,     // (x, y) -> (y, x)
    FLIP_YX,     // (x, y) -> (s - y, s - x)
    TRANS_NB
};

/// Check if a transform type is applicable to rectangle.
constexpr bool isRectangleTransform(TransformType t)
{
    return t == IDENTITY || t == ROTATE_180 || t == FLIP_X || t == FLIP_Y;
}

/// Apply a type of transform to the pos with a square board size.
constexpr Pos applyTransform(Pos pos, int boardsize, TransformType trans)
{
    int x = pos.x(), y = pos.y();
    int s = boardsize - 1;

    switch (trans) {
    case ROTATE_90:  // (x, y) -> (y, s - x)
        return {y, s - x};
    case ROTATE_180:  // (x, y) -> (s - x, s - y)
        return {s - x, s - y};
    case ROTATE_270:  // (x, y) -> (s - y, x)
        return {s - y, x};
    case FLIP_X:  // (x, y) -> (x, s - y)
        return {x, s - y};
    case FLIP_Y:  // (x, y) -> (s - x, y)
        return {s - x, y};
    case FLIP_XY:  // (x, y) -> (y, x)
        return {y, x};
    case FLIP_YX:  // (x, y) -> (s - y, s - x)
        return {s - y, s - x};
    default: return {x, y};
    }
}

/// Apply a type of transform to the pos with a rectangle board size.
constexpr Pos applyTransform(Pos pos, int sizeX, int sizeY, TransformType trans)
{
    if (sizeX == sizeY)
        return applyTransform(pos, sizeX, trans);

    int x = pos.x(), y = pos.y();
    int sx = sizeX - 1, sy = sizeY - 1;

    switch (trans) {
    case ROTATE_180:  // (x, y) -> (sx - x, sy - y)
        return {sx - x, sy - y};
    case FLIP_X:  // (x, y) -> (x, sy - y)
        return {x, sy - y};
    case FLIP_Y:  // (x, y) -> (sx - x, y)
        return {sx - x, y};
    default: return {x, y};
    }
}

// -------------------------------------------------
// Direction ranges

const Direction RANGE_LINE2[] = {
    UP_LEFT * 2,
    UP * 2,
    UP_RIGHT * 2,
    UP_LEFT,
    UP,
    UP_RIGHT,
    LEFT * 2,
    LEFT,
    RIGHT * 2,
    RIGHT,
    DOWN_LEFT,
    DOWN,
    DOWN_RIGHT,
    DOWN_LEFT * 2,
    DOWN * 2,
    DOWN_RIGHT * 2,
};

const Direction RANGE_SQUARE2[] = {
    UP_LEFT * 2,
    UP_LEFT + UP,
    UP * 2,
    UP_RIGHT + UP,
    UP_RIGHT * 2,
    UP_LEFT + LEFT,
    UP_LEFT,
    UP,
    UP_RIGHT,
    UP_RIGHT + RIGHT,
    LEFT * 2,
    LEFT,
    RIGHT,
    RIGHT * 2,
    DOWN_LEFT + LEFT,
    DOWN_LEFT,
    DOWN,
    DOWN_RIGHT,
    DOWN_RIGHT + RIGHT,
    DOWN_LEFT * 2,
    DOWN_LEFT + DOWN,
    DOWN * 2,
    DOWN_RIGHT + DOWN,
    DOWN_RIGHT * 2,
};

const Direction RANGE_SQUARE2_LINE3[] = {
    UP_LEFT * 3,
    UP * 3,
    UP_RIGHT * 3,
    UP_LEFT * 2,
    UP_LEFT + UP,
    UP * 2,
    UP_RIGHT + UP,
    UP_RIGHT * 2,
    UP_LEFT + LEFT,
    UP_LEFT,
    UP,
    UP_RIGHT,
    UP_RIGHT + RIGHT,
    LEFT * 3,
    LEFT * 2,
    LEFT,
    RIGHT,
    RIGHT * 2,
    RIGHT * 3,
    DOWN_LEFT + LEFT,
    DOWN_LEFT,
    DOWN,
    DOWN_RIGHT,
    DOWN_RIGHT + RIGHT,
    DOWN_LEFT * 2,
    DOWN_LEFT + DOWN,
    DOWN * 2,
    DOWN_RIGHT + DOWN,
    DOWN_RIGHT * 2,
    DOWN_LEFT * 3,
    DOWN * 3,
    DOWN_RIGHT * 3,
};

const Direction RANGE_SQUARE3[] = {
    UP_LEFT * 3,
    UP_LEFT * 2 + UP,
    UP_LEFT + UP * 2,
    UP * 3,
    UP_RIGHT + UP * 2,
    UP_RIGHT * 2 + UP,
    UP_RIGHT * 3,
    UP_LEFT * 2 + LEFT,
    UP_LEFT * 2,
    UP_LEFT + UP,
    UP * 2,
    UP_RIGHT + UP,
    UP_RIGHT * 2,
    UP_RIGHT * 2 + RIGHT,
    UP_LEFT + LEFT * 2,
    UP_LEFT + LEFT,
    UP_LEFT,
    UP,
    UP_RIGHT,
    UP_RIGHT + RIGHT,
    UP_RIGHT + RIGHT * 2,
    LEFT * 3,
    LEFT * 2,
    LEFT,
    RIGHT,
    RIGHT * 2,
    RIGHT * 3,
    DOWN_LEFT + LEFT * 2,
    DOWN_LEFT + LEFT,
    DOWN_LEFT,
    DOWN,
    DOWN_RIGHT,
    DOWN_RIGHT + RIGHT,
    DOWN_RIGHT + RIGHT * 2,
    DOWN_LEFT * 2 + LEFT,
    DOWN_LEFT * 2,
    DOWN_LEFT + DOWN,
    DOWN * 2,
    DOWN_RIGHT + DOWN,
    DOWN_RIGHT * 2,
    DOWN_RIGHT * 2 + RIGHT,
    DOWN_LEFT * 3,
    DOWN_LEFT * 2 + DOWN,
    DOWN_LEFT + DOWN * 2,
    DOWN * 3,
    DOWN_RIGHT + DOWN * 2,
    DOWN_RIGHT * 2 + DOWN,
    DOWN_RIGHT * 3,
};

const Direction RANGE_LINE4[] = {
    UP_LEFT * 4,
    UP * 4,
    UP_RIGHT * 4,
    UP_LEFT * 3,
    UP * 3,
    UP_RIGHT * 3,
    UP_LEFT * 2,
    UP * 2,
    UP_RIGHT * 2,
    UP_LEFT,
    UP,
    UP_RIGHT,
    LEFT * 4,
    LEFT * 3,
    LEFT * 2,
    LEFT,
    RIGHT,
    RIGHT * 2,
    RIGHT * 3,
    RIGHT * 4,
    DOWN_LEFT,
    DOWN,
    DOWN_RIGHT,
    DOWN_LEFT * 2,
    DOWN * 2,
    DOWN_RIGHT * 2,
    DOWN_LEFT * 3,
    DOWN * 3,
    DOWN_RIGHT * 3,
    DOWN_LEFT * 4,
    DOWN * 4,
    DOWN_RIGHT * 4,
};

const Direction RANGE_SQUARE2_LINE4[] = {
    UP_LEFT * 4,
    UP * 4,
    UP_RIGHT * 4,
    UP_LEFT * 3,
    UP * 3,
    UP_RIGHT * 3,
    UP_LEFT * 2,
    UP_LEFT + UP,
    UP * 2,
    UP_RIGHT + UP,
    UP_RIGHT * 2,
    UP_LEFT + LEFT,
    UP_LEFT,
    UP,
    UP_RIGHT,
    UP_RIGHT + RIGHT,
    LEFT * 4,
    LEFT * 3,
    LEFT * 2,
    LEFT,
    RIGHT,
    RIGHT * 2,
    RIGHT * 3,
    RIGHT * 4,
    DOWN_LEFT + LEFT,
    DOWN_LEFT,
    DOWN,
    DOWN_RIGHT,
    DOWN_RIGHT + RIGHT,
    DOWN_LEFT * 2,
    DOWN_LEFT + DOWN,
    DOWN * 2,
    DOWN_RIGHT + DOWN,
    DOWN_RIGHT * 2,
    DOWN_LEFT * 3,
    DOWN * 3,
    DOWN_RIGHT * 3,
    DOWN_LEFT * 4,
    DOWN * 4,
    DOWN_RIGHT * 4,
};

const Direction RANGE_SQUARE3_LINE4[] = {
    UP_LEFT * 4,
    UP * 4,
    UP_RIGHT * 4,

    UP_LEFT * 3,
    UP_LEFT * 2 + UP,
    UP_LEFT + UP * 2,
    UP * 3,
    UP_RIGHT + UP * 2,
    UP_RIGHT * 2 + UP,
    UP_RIGHT * 3,

    UP_LEFT * 2 + LEFT,
    UP_LEFT * 2,
    UP_LEFT + UP,
    UP * 2,
    UP_RIGHT + UP,
    UP_RIGHT * 2,
    UP_RIGHT * 2 + RIGHT,

    UP_LEFT + LEFT * 2,
    UP_LEFT + LEFT,
    UP_LEFT,
    UP,
    UP_RIGHT,
    UP_RIGHT + RIGHT,
    UP_RIGHT + RIGHT * 2,

    LEFT * 4,
    LEFT * 3,
    LEFT * 2,
    LEFT,
    RIGHT,
    RIGHT * 2,
    RIGHT * 3,
    RIGHT * 4,

    DOWN_LEFT + LEFT * 2,
    DOWN_LEFT + LEFT,
    DOWN_LEFT,
    DOWN,
    DOWN_RIGHT,
    DOWN_RIGHT + RIGHT,
    DOWN_RIGHT + RIGHT * 2,

    DOWN_LEFT * 2 + LEFT,
    DOWN_LEFT * 2,
    DOWN_LEFT + DOWN,
    DOWN * 2,
    DOWN_RIGHT + DOWN,
    DOWN_RIGHT * 2,
    DOWN_RIGHT * 2 + RIGHT,

    DOWN_LEFT * 3,
    DOWN_LEFT * 2 + DOWN,
    DOWN_LEFT + DOWN * 2,
    DOWN * 3,
    DOWN_RIGHT + DOWN * 2,
    DOWN_RIGHT * 2 + DOWN,
    DOWN_RIGHT * 3,

    DOWN_LEFT * 4,
    DOWN * 4,
    DOWN_RIGHT * 4,
};

const Direction RANGE_SQUARE4[] = {
    UP_LEFT * 4,
    UP_LEFT * 3 + UP,
    UP_LEFT * 2 + UP * 2,
    UP_LEFT + UP * 3,
    UP * 4,
    UP_RIGHT + UP * 3,
    UP_RIGHT * 2 + UP * 2,
    UP_RIGHT * 3 + UP,
    UP_RIGHT * 4,

    UP_LEFT * 3 + LEFT,
    UP_LEFT * 3,
    UP_LEFT * 2 + UP,
    UP_LEFT + UP * 2,
    UP * 3,
    UP_RIGHT + UP * 2,
    UP_RIGHT * 2 + UP,
    UP_RIGHT * 3,
    UP_RIGHT * 3 + RIGHT,

    UP_LEFT * 2 + LEFT * 2,
    UP_LEFT * 2 + LEFT,
    UP_LEFT * 2,
    UP_LEFT + UP,
    UP * 2,
    UP_RIGHT + UP,
    UP_RIGHT * 2,
    UP_RIGHT * 2 + RIGHT,
    UP_RIGHT * 2 + RIGHT * 2,

    UP_LEFT + LEFT * 3,
    UP_LEFT + LEFT * 2,
    UP_LEFT + LEFT,
    UP_LEFT,
    UP,
    UP_RIGHT,
    UP_RIGHT + RIGHT,
    UP_RIGHT + RIGHT * 2,
    UP_RIGHT + RIGHT * 3,

    LEFT * 4,
    LEFT * 3,
    LEFT * 2,
    LEFT,
    RIGHT,
    RIGHT * 2,
    RIGHT * 3,
    RIGHT * 4,

    DOWN_LEFT + LEFT * 3,
    DOWN_LEFT + LEFT * 2,
    DOWN_LEFT + LEFT,
    DOWN_LEFT,
    DOWN,
    DOWN_RIGHT,
    DOWN_RIGHT + RIGHT,
    DOWN_RIGHT + RIGHT * 2,
    DOWN_RIGHT + RIGHT * 3,

    DOWN_LEFT * 2 + LEFT * 2,
    DOWN_LEFT * 2 + LEFT,
    DOWN_LEFT * 2,
    DOWN_LEFT + DOWN,
    DOWN * 2,
    DOWN_RIGHT + DOWN,
    DOWN_RIGHT * 2,
    DOWN_RIGHT * 2 + RIGHT,
    DOWN_RIGHT * 2 + RIGHT * 2,

    DOWN_LEFT * 3 + LEFT,
    DOWN_LEFT * 3,
    DOWN_LEFT * 2 + DOWN,
    DOWN_LEFT + DOWN * 2,
    DOWN * 3,
    DOWN_RIGHT + DOWN * 2,
    DOWN_RIGHT * 2 + DOWN,
    DOWN_RIGHT * 3,
    DOWN_RIGHT * 3 + RIGHT,

    DOWN_LEFT * 4,
    DOWN_LEFT * 3 + DOWN,
    DOWN_LEFT * 2 + DOWN * 2,
    DOWN_LEFT + DOWN * 3,
    DOWN * 4,
    DOWN_RIGHT + DOWN * 3,
    DOWN_RIGHT * 2 + DOWN * 2,
    DOWN_RIGHT * 3 + DOWN,
    DOWN_RIGHT * 4,
};
