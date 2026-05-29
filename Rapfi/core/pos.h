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

#include <array>
#include <cstdint>
#include <cstdlib>
#include <functional>

// -------------------------------------------------
// Board address-space layout
//
// Positions are packed into a single 16-bit integer with a power-of-two row stride, and a
// few rows/columns of WALL boundary on every side. This lets neighbour lookups skip explicit
// edge checks: walking off the board lands on WALL cells that the pattern code naturally
// classifies as DEAD.

constexpr int FULL_BOARD_SIZE       = 32;  ///< Row stride (power of two), including boundary.
constexpr int FULL_BOARD_CELL_COUNT = FULL_BOARD_SIZE * FULL_BOARD_SIZE;
constexpr int BOARD_BOUNDARY        = 5;  ///< Padding cells on every side, filled with WALL.
constexpr int MAX_BOARD_SIZE        = FULL_BOARD_SIZE - 2 * BOARD_BOUNDARY;
constexpr int MAX_MOVES             = MAX_BOARD_SIZE * MAX_BOARD_SIZE + 1;  ///< +1 for PASS.

// -------------------------------------------------
// Pos

/// Packed (x, y) coordinate in the board address space `[NONE, FULL_BOARD_END)`.
///
/// The packing `((y + BOARD_BOUNDARY) << 5) | (x + BOARD_BOUNDARY)` puts the playable cells
/// inside a 32-column strip, lets a `Direction` offset be added with plain integer arithmetic,
/// and keeps the WALL boundary inside addressable range so border checks are unnecessary in
/// the inner pattern-lookup loops.
struct Pos
{
    Pos() = default;
    constexpr Pos(int x, int y) : _pos(((y + BOARD_BOUNDARY) << 5) | (x + BOARD_BOUNDARY)) {}
    constexpr explicit Pos(int16_t pos) : _pos(pos) {}

    constexpr int x() const { return (_pos & 31) - BOARD_BOUNDARY; }
    constexpr int y() const { return (_pos >> 5) - BOARD_BOUNDARY; }

    /// Implicit conversion to the packed integer index (used for array indexing and ordering).
    constexpr operator int() const { return _pos; }

    /// True iff this is an addressable index: PASS, NONE, a wall cell, or a playable cell.
    /// Use `isInBoard(w, h)` for the stricter "inside the playable region" test.
    inline bool valid() const { return _pos >= PASS._pos && _pos < FULL_BOARD_END._pos; }

    /// Row-major index into a `MAX_BOARD_SIZE * MAX_BOARD_SIZE` array (boundary excluded).
    inline int moveIndex() const { return y() * MAX_BOARD_SIZE + x(); }

    inline bool isInBoard(int boardWidth, int boardHeight) const;

    /// Chebyshev (king-move) distance. Returns -1 if either operand is PASS or NONE.
    static int distance(Pos p1, Pos p2);

    /// Chebyshev distance if both operands share a horizontal / vertical / diagonal line;
    /// otherwise `FULL_BOARD_SIZE`. Returns -1 if either operand is PASS or NONE.
    static int lineDistance(Pos p1, Pos p2);

    static const Pos NONE;              ///< Index 0: first boundary cell, also used as sentinel.
    static const Pos PASS;              ///< Sentinel for a pass / null move (`_pos == -1`).
    static const Pos FULL_BOARD_START;  ///< First index for full-address-space iteration.
    static const Pos FULL_BOARD_END;    ///< One past the last addressable cell.

private:
    int16_t _pos;  ///< Packed coordinate; read via operator int().
};

inline constexpr Pos Pos::NONE {0};
inline constexpr Pos Pos::PASS {-1};
inline constexpr Pos Pos::FULL_BOARD_START {0};
inline constexpr Pos Pos::FULL_BOARD_END {FULL_BOARD_CELL_COUNT};

// -------------------------------------------------
// Direction

/// Offset added to a `Pos` to reach a neighbour. Precomputed for the packed row stride so
/// `pos + dir` is a single integer add.
enum Direction : int16_t {
    UP    = -FULL_BOARD_SIZE,
    LEFT  = -1,
    DOWN  = -UP,
    RIGHT = -LEFT,

    UP_LEFT    = UP + LEFT,
    UP_RIGHT   = UP + RIGHT,
    DOWN_LEFT  = DOWN + LEFT,
    DOWN_RIGHT = DOWN + RIGHT,
};

/// The four canonical line directions iterated in pattern lookups: horizontal, vertical, and
/// the two diagonals. Each line pattern walks outward from a cell in one of these directions
/// and its inverse.
constexpr Direction DIRECTION[] = {RIGHT, DOWN, UP_RIGHT, DOWN_RIGHT};

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
// Pos / Direction arithmetic

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

inline bool Pos::isInBoard(int boardWidth, int boardHeight) const
{
    int X = x(), Y = y();
    return X >= 0 && X < boardWidth && Y >= 0 && Y < boardHeight;
}

inline int Pos::distance(Pos p1, Pos p2)
{
    if (p1 <= Pos::NONE || p2 <= Pos::NONE)
        return -1;

    int xDist = std::abs(p1.x() - p2.x());
    int yDist = std::abs(p1.y() - p2.y());
    return xDist < yDist ? yDist : xDist;
}

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

// -------------------------------------------------
// Board symmetries

/// The eight symmetries of a square board (D4 dihedral group).
enum TransformType {
    IDENTITY,    ///< (x, y) -> (x, y)
    ROTATE_90,   ///< (x, y) -> (y, s - x)
    ROTATE_180,  ///< (x, y) -> (s - x, s - y)
    ROTATE_270,  ///< (x, y) -> (s - y, x)
    FLIP_X,      ///< (x, y) -> (x, s - y) - reflect across the horizontal mid-line.
    FLIP_Y,      ///< (x, y) -> (s - x, y) - reflect across the vertical mid-line.
    FLIP_XY,     ///< (x, y) -> (y, x)     - reflect across the main diagonal.
    FLIP_YX,     ///< (x, y) -> (s - y, s - x) - reflect across the anti-diagonal.
    TRANS_NB
};

/// Whether the transform is meaningful on a non-square board (no axis-swap).
constexpr bool isRectangleTransform(TransformType t)
{
    return t == IDENTITY || t == ROTATE_180 || t == FLIP_X || t == FLIP_Y;
}

/// Apply a transform on a square board of side `boardsize`.
constexpr Pos applyTransform(Pos pos, int boardsize, TransformType trans)
{
    int x = pos.x(), y = pos.y();
    int s = boardsize - 1;

    switch (trans) {
    case ROTATE_90: return {y, s - x};
    case ROTATE_180: return {s - x, s - y};
    case ROTATE_270: return {s - y, x};
    case FLIP_X: return {x, s - y};
    case FLIP_Y: return {s - x, y};
    case FLIP_XY: return {y, x};
    case FLIP_YX: return {s - y, s - x};
    default: return {x, y};
    }
}

/// Apply a transform on a rectangular board. Axis-swap transforms degrade to IDENTITY when
/// `sizeX != sizeY`.
constexpr Pos applyTransform(Pos pos, int sizeX, int sizeY, TransformType trans)
{
    if (sizeX == sizeY)
        return applyTransform(pos, sizeX, trans);

    int x = pos.x(), y = pos.y();
    int sx = sizeX - 1, sy = sizeY - 1;

    switch (trans) {
    case ROTATE_180: return {sx - x, sy - y};
    case FLIP_X: return {x, sy - y};
    case FLIP_Y: return {sx - x, y};
    default: return {x, y};
    }
}

// -------------------------------------------------
// Protocol coordinate orientation

/// How protocol (x, y) integer pairs are remapped to/from internal board coordinates, so the
/// engine can match different GUIs' coordinate orientations. The active mode is held by the
/// configuration layer (`Config::IOCoordMode`) and passed explicitly to the conversion
/// helpers in iohelper.h.
enum class CoordConvertionMode {
    NONE,     ///< No remapping.
    X_FLIPY,  ///< Keep x, flip y.
    FLIPY_X,  ///< Swap and flip: (x, y) <-> (flipped y, x).
};

// -------------------------------------------------
// Candidate ranges
//
// A "candidate range" is the set of cells (expressed as Direction offsets from a played
// stone) that the engine considers as move candidates around that stone. A range is
// parameterised by a filled square of side `2*SquareDist + 1` and an extension along the
// eight directions out to `LineDist` cells.

/// Set of cells considered as move candidates around played stones. Larger ranges produce
/// more candidates but slower move generation; `FULL_BOARD` disables the optimization.
enum class CandidateRange {
    SQUARE2,        ///< 5x5 square.
    SQUARE2_LINE3,  ///< 5x5 square plus 8-direction line out to 3 cells.
    SQUARE3,        ///< 7x7 square.
    SQUARE3_LINE4,  ///< 7x7 square plus 8-direction line out to 4 cells.
    SQUARE4,        ///< 9x9 square.
    FULL_BOARD,     ///< Every empty cell is a candidate.
    CAND_RANGE_NB,
};

namespace detail {

/// Predicate: does offset (dx, dy) belong to the (SquareDist, LineDist) candidate range?
/// The centre cell (0, 0) is always excluded.
constexpr bool inCandidateRange(int dx, int dy, int squareDist, int lineDist)
{
    int ax = dx < 0 ? -dx : dx;
    int ay = dy < 0 ? -dy : dy;
    if (ax == 0 && ay == 0)
        return false;
    if (ax > lineDist || ay > lineDist)
        return false;
    bool inSquare = ax <= squareDist && ay <= squareDist;
    bool onLine   = dx == 0 || dy == 0 || dx == dy || dx == -dy;
    return inSquare || onLine;
}

template <int SquareDist, int LineDist>
constexpr int candidateRangeCellCount()
{
    int n = 0;
    for (int dy = -LineDist; dy <= LineDist; ++dy)
        for (int dx = -LineDist; dx <= LineDist; ++dx)
            if (inCandidateRange(dx, dy, SquareDist, LineDist))
                ++n;
    return n;
}

/// Build the Direction array for a (SquareDist, LineDist) candidate range, ordered
/// row-major (y ascending, then x ascending) with the centre cell skipped.
template <int SquareDist, int LineDist>
constexpr auto buildCandidateRange()
{
    std::array<Direction, candidateRangeCellCount<SquareDist, LineDist>()> arr {};
    int                                                                    i = 0;
    for (int dy = -LineDist; dy <= LineDist; ++dy)
        for (int dx = -LineDist; dx <= LineDist; ++dx)
            if (inCandidateRange(dx, dy, SquareDist, LineDist))
                arr[i++] = Direction(dy * FULL_BOARD_SIZE + dx);
    return arr;
}

}  // namespace detail

inline constexpr auto RANGE_SQUARE2       = detail::buildCandidateRange<2, 2>();
inline constexpr auto RANGE_SQUARE2_LINE3 = detail::buildCandidateRange<2, 3>();
inline constexpr auto RANGE_SQUARE3       = detail::buildCandidateRange<3, 3>();
inline constexpr auto RANGE_SQUARE3_LINE4 = detail::buildCandidateRange<3, 4>();
inline constexpr auto RANGE_SQUARE4       = detail::buildCandidateRange<4, 4>();

/// Internal range used only by VCF move generation (search/movepick.cpp), not a board-
/// selectable CandidateRange. Kept here next to the other ranges since it shares the builder.
inline constexpr auto RANGE_SQUARE2_LINE4 = detail::buildCandidateRange<2, 4>();

/// Resolved data for a board-selectable candidate range.
struct CandidateRangeInfo
{
    const Direction *offsets;      ///< Neighbour offset table; nullptr for FULL_BOARD.
    size_t           offsetCount;  ///< Length of `offsets`; 0 for FULL_BOARD.
    int              expandDist;   ///< Candidate-area expansion distance; 0 for FULL_BOARD.
};

/// Map a board-selectable CandidateRange to its precomputed offset table and expansion
/// distance, keeping the enum and the RANGE_* arrays in agreement in one place. FULL_BOARD
/// has no table (every empty cell is a candidate) and yields {nullptr, 0, 0}.
constexpr CandidateRangeInfo candidateRangeInfo(CandidateRange range)
{
    switch (range) {
    case CandidateRange::SQUARE2: return {RANGE_SQUARE2.data(), RANGE_SQUARE2.size(), 2};
    case CandidateRange::SQUARE2_LINE3:
        return {RANGE_SQUARE2_LINE3.data(), RANGE_SQUARE2_LINE3.size(), 3};
    case CandidateRange::SQUARE3: return {RANGE_SQUARE3.data(), RANGE_SQUARE3.size(), 3};
    case CandidateRange::SQUARE3_LINE4:
        return {RANGE_SQUARE3_LINE4.data(), RANGE_SQUARE3_LINE4.size(), 3};
    case CandidateRange::SQUARE4: return {RANGE_SQUARE4.data(), RANGE_SQUARE4.size(), 4};
    default: return {nullptr, 0, 0};  // FULL_BOARD
    }
}
