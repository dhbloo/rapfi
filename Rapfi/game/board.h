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

#include "../config.h"
#include "../core/hash.h"
#include "../core/platform.h"
#include "../core/pos.h"
#include "../core/types.h"
#include "bitboard.h"
#include "candarea.h"
#include "pattern.h"

#include <array>
#include <cassert>
#include <memory>

namespace Search {
class SearchThread;
}
namespace Evaluation {
class Evaluator;
}

/// Iterate `pos` over every on-board (non-wall) cell, in ascending Pos order.
#define FOR_EVERY_POSITION(board, pos) \
    for (Bitboard::Cursor _posCur((board)->onBoard()); Pos pos = _posCur.next();)

/// Iterate `pos` over every empty cell, in ascending Pos order.
#define FOR_EVERY_EMPTY_POS(board, pos) \
    for (Bitboard::Cursor _emptyCur((board)->emptyCells()); Pos pos = _emptyCur.next();)

/// Iterate `pos` over every cell of the candidate-area rectangle `area` (row-major). Walks each
/// row as a contiguous `Pos` run, since incrementing a `Pos` steps one cell along the row.
#define FOR_EVERY_CANDAREA_POS(pos, area)                                                     \
    for (int8_t _y = (area).y0, _y1 = (area).y1, _x0 = (area).x0, _x1 = (area).x1; _y <= _y1; \
         _y++)                                                                                \
        for (Pos pos {_x0, _y}, _rowEnd {_x1, _y}; pos <= _rowEnd; pos++)

/// Iterate `pos` over every empty candidate cell, in ascending Pos order. candidateIterSet()
/// already excludes occupied cells and cells outside the candidate bounding box, so the body needs
/// no per-cell filtering. The set is a snapshot, so a probing move/undo in the body is safe.
#define FOR_EVERY_CAND_POS(board, pos)                                                       \
    for (Bitboard _candSet = (board)->candidateIterSet(), *_candOnce = &_candSet; _candOnce; \
         _candOnce = nullptr)                                                                \
        for (Bitboard::Cursor _candCur(_candSet); Pos pos = _candCur.next();)

/// Iterate `pos` over every empty candidate cell, in ascending Pos order, WITHOUT the
/// candidate-area box clip that FOR_EVERY_CAND_POS applies (i.e. the full candidate set intersected
/// with empty). Like FOR_EVERY_CAND_POS the set is a snapshot, so a probing move/undo in the body
/// is safe.
#define FOR_EVERY_EMPTY_CAND_POS(board, pos)                                        \
    for (Bitboard _ecSet = (board)->emptyCandidates(), *_ecOnce = &_ecSet; _ecOnce; \
         _ecOnce = nullptr)                                                         \
        for (Bitboard::Cursor _ecCur(_ecSet); Pos pos = _ecCur.next();)

/// Per-ply snapshot of the incremental board state. One StateInfo exists per move played, so a
/// move only has to record what changed and undo() can restore the previous ply cheaply.
struct StateInfo
{
    Bitboard candidates;  ///< Candidate-move set at this ply (snapshotted per ply).
    CandArea candArea;    ///< Bounding box of candidate cells at this ply.
    Pos      lastMove;    ///< The move that produced this ply (Pos::PASS for a pass).
    /// Move that first created a flex four for this side (set only on a 0 -> nonzero transition).
    Pos lastFlex4AttackMove[SIDE_NB];
    /// Most recent empty cell reaching C_BLOCK4_FLEX3 / B_FLEX4 / A_FIVE, per side.
    Pos lastPattern4Move[SIDE_NB][3];
    /// Count of empty cells holding each Pattern4, per side.
    uint16_t p4Count[SIDE_NB][PATTERN4_NB];
    Value    valueBlack;  ///< Incremental classical evaluation from black's view.

    /// The most recent empty cell that reached the given Pattern4.
    /// @note p4 must be one of [C_BLOCK4_FLEX3, B_FLEX4, A_FIVE].
    Pos lastPattern4(Color side, Pattern4 p4) const
    {
        assert(p4 >= C_BLOCK4_FLEX3 && p4 <= A_FIVE);
        return lastPattern4Move[side][p4 - C_BLOCK4_FLEX3];
    }
};

/// All per-cell state the engine maintains: the stone (or EMPTY/WALL) and the incrementally-updated
/// patterns, scores and evaluation for both sides. Candidacy is tracked separately, in the per-ply
/// candidate bitboard rather than on the cell.
struct Cell
{
    Color     piece;              ///< Stone color, or EMPTY / WALL.
    Pattern4  pattern4[SIDE_NB];  ///< Aggregate four-direction pattern, per side.
    Score     score[SIDE_NB];     ///< Move-ordering score, per side.
    Value     valueBlack;    ///< This cell's contribution to the classical eval, from black's view.
    Pattern2x pattern2x[4];  ///< Line pattern of both colors, one per direction.

    /// The line-level pattern of this cell in direction `dir`, from `c`'s perspective.
    Pattern pattern(Color c, int dir) const
    {
        assert(c == BLACK || c == WHITE);
        return c == BLACK ? pattern2x[dir].patBlack : pattern2x[dir].patWhite;
    }

    /// The combined four-direction pattern code at this cell for one side (indexes the score
    /// and eval tables).
    template <Color C>
    PatternCode pcode() const
    {
        static_assert(C == BLACK || C == WHITE);
        if constexpr (C == BLACK)
            return PatternConfig::PCODE[pattern2x[0].patBlack][pattern2x[1].patBlack]
                                       [pattern2x[2].patBlack][pattern2x[3].patBlack];
        else
            return PatternConfig::PCODE[pattern2x[0].patWhite][pattern2x[1].patWhite]
                                       [pattern2x[2].patWhite][pattern2x[3].patWhite];
    }

    /// Recompute both sides' pattern4 and move-ordering score from fresh pattern codes. Each
    /// side's score blends its own attacking score with the opponent's defensive score.
    template <Rule R>
    void updatePattern4AndScore(PatternCode pcodeBlack, PatternCode pcodeWhite)
    {
        Pattern4Score p4ScoreBlack = Config::getP4Score(R, BLACK, pcodeBlack);
        Pattern4Score p4ScoreWhite = Config::getP4Score(R, WHITE, pcodeWhite);
        pattern4[BLACK]            = (Pattern4)p4ScoreBlack;
        pattern4[WHITE]            = (Pattern4)p4ScoreWhite;
        score[BLACK]               = p4ScoreBlack.scoreSelf() + p4ScoreWhite.scoreOppo();
        score[WHITE]               = p4ScoreWhite.scoreSelf() + p4ScoreBlack.scoreOppo();
    }
};

/// The central board-position representation. Beyond the cell grid it keeps the per-direction
/// bitkeys and a per-ply StateInfo/UpdateCache history, so move() and undo() update the position
/// incrementally rather than rescanning the board. The copy constructor is explicit (and the
/// implicit copy deleted) to avoid accidental expensive copies.
class Board
{
public:
    /// How much state move()/undo() should maintain. Cheaper modes skip work the caller does not
    /// need, and must be paired (the same MT in the matching undo()).
    enum class MoveType {
        NORMAL,        ///< Update cell, pattern, score, classical eval, and external evaluator.
        NO_EVALUATOR,  ///< As NORMAL but skip the external evaluator.
        NO_EVAL,       ///< Update cell, pattern and score only (no eval at all).
        NO_EVAL_MULTI  ///< As NO_EVAL, but do not flip the side to move.
    };

    /// Allocate a board of the given size and candidate range. The position is uninitialized
    /// until newGame() is called.
    /// @param boardSize Size of the board, in range [1, MAX_BOARD_SIZE].
    explicit Board(int boardSize, CandidateRange candRange = Config::DefaultCandidateRange);
    /// Clone a board from another and bind a search thread to the clone.
    /// @param other Board object to clone from.
    /// @param thread Search thread to bind (nullptr for no binding).
    explicit Board(const Board &other, Search::SearchThread *thread);
    Board(const Board &) = delete;

    // ------------------------------------------------------------------------
    // board modifier (and dynamic dispatch version)

    /// Initialize the board to an empty board state of rule R.
    /// @tparam R Rule to initialize the board.
    template <Rule R>
    void newGame();

    /// Play a move and incrementally update the board state.
    /// @param pos Pos to put the next stone. Pos::PASS is allowed.
    /// @tparam R Game rule to use.
    /// @tparam MT How much state to maintain (see MoveType).
    /// @note Recursive pass moves are allowed, but the total number of passes must stay below
    ///     cellCount(). As long as consecutive pass moves are not allowed, this holds.
    template <Rule R, MoveType MT = MoveType::NORMAL>
    void move(Pos pos);

    /// Undo move and rollback the board state.
    /// @tparam R Game rule to use.
    /// @tparam MT Type of this move. Must match the move type used in move().
    template <Rule R, MoveType MT = MoveType::NORMAL>
    void undo();

    /// A dynamic dispatch version of newGame().
    void newGame(Rule rule);
    /// A dynamic dispatch version of move().
    void move(Rule rule, Pos pos);
    /// A dynamic dispatch version of undo().
    void undo(Rule rule);

    // ------------------------------------------------------------------------
    // special helper function

    /// Flip the side to move without recording a ply in the state info. Intended only for local
    /// board-checking routines and must be used in pairs. For an actual pass, use move(Pos::PASS).
    void flipSide() { currentSide = ~currentSide; }

    // ------------------------------------------------------------------------
    // pos-specific info queries

    /// Get cell of piece at pos on board.
    inline const Cell &cell(Pos pos) const
    {
        assert(pos >= 0 && pos < FULL_BOARD_CELL_COUNT);
        return cells[pos];
    }

    /// Get color of piece at pos on board.
    inline Color get(Pos pos) const
    {
        assert(pos >= 0 && pos < FULL_BOARD_CELL_COUNT);
        return cells[pos].piece;
    }

    /// Check if the pos is in the region of current board size.
    bool isInBoard(Pos pos) const { return pos.isInBoard(boardSize, boardSize); }

    /// Check if the pos is on an empty cell.
    /// @pos The pos to query, which is assumed to meet 'pos.valid() == true'.
    bool isEmpty(Pos pos) const { return get(pos) == EMPTY; }

    /// Check if the pos is legal (on an empty cell or is a pass move). The PASS check comes first
    /// so a pass never reaches isEmpty()/get(), which would index out of the cell grid for
    /// Pos::PASS.
    bool isLegal(Pos pos) const { return pos.valid() && (pos == Pos::PASS || isEmpty(pos)); }

    /// Whether `pos` is currently a move candidate (some played stone's range covers it). This is
    /// the raw in-range test (no box / empty filtering), as used by neighbor move generation.
    bool isCandidate(Pos pos) const { return stateInfo().candidates.test(pos); }

    /// Whether `pos` is an empty candidate cell: in the raw candidate set and currently empty (no
    /// box clip). Off-board positions read as false, so a neighbor probe needs no range check.
    bool isEmptyCandidate(Pos pos) const
    {
        return emptyBB.test(pos) && stateInfo().candidates.test(pos);
    }

    /// Bitboard of all on-board (non-wall) cells; constant for the board's lifetime.
    const Bitboard &onBoard() const { return onBoardBB; }

    /// Bitboard of all currently-empty cells.
    const Bitboard &emptyCells() const { return emptyBB; }

    /// The set of candidate cells to iterate this ply: in-range AND empty AND inside the candidate
    /// bounding box. Returned by value as a snapshot, so iteration is unaffected by a probing
    /// move/undo in the loop body. Used by FOR_EVERY_CAND_POS.
    Bitboard candidateIterSet() const
    {
        Bitboard bb;
        bb.buildCandSet(stateInfo().candidates, emptyBB, stateInfo().candArea);
        return bb;
    }

    /// The candidate cells to iterate ignoring the bounding box: in-range AND empty (`candidates &
    /// emptyCells`). Like candidateIterSet() but without the candArea clip, so it includes every
    /// empty candidate the board tracks. Returned by value as a snapshot. Used by
    /// FOR_EVERY_EMPTY_CAND_POS.
    Bitboard emptyCandidates() const
    {
        Bitboard bb;
        bb.setIntersect(stateInfo().candidates, emptyBB);
        return bb;
    }

    /// Check if a pos on board is forbidden point in Renju rule.
    /// @param pos Pos to check forbidden. Should be an empty cell.
    /// @return Whether this pos is a real forbidden point.
    bool checkForbiddenPoint(Pos pos) const;

    /// Query the bitkey at the given pos and direction.
    template <Rule R>
    uint64_t getKeyAt(Pos pos, int dir) const;

    // ------------------------------------------------------------------------
    // general board info queries

    int                    size() const { return boardSize; }
    int                    cellCount() const { return boardCellCount; }
    Pos                    centerPos() const { return {boardSize / 2, boardSize / 2}; }
    Pos                    startPos() const { return {0, 0}; }
    Pos                    endPos() const { return {boardSize - 1, boardSize - 1}; }
    Search::SearchThread  *thisThread() const { return thisThread_; }
    Evaluation::Evaluator *evaluator() const { return evaluator_; }

    // ------------------------------------------------------------------------
    // current board state queries

    int   ply() const { return moveCount; };
    int   nonPassMoveCount() const { return moveCount - passMoveCount(); }
    int   passMoveCount() const { return passCount[BLACK] + passCount[WHITE]; }
    int   passMoveCountOfSide(Color side) const { return passCount[side]; }
    int   movesLeft() const { return boardCellCount - nonPassMoveCount(); };
    Color sideToMove() const { return currentSide; }

    /// Fetch the current board hash key.
    HashKey zobristKey() const { return currentZobristKey ^ Hash::zobristSide[currentSide]; }

    /// Compute the board hash key after a move, without actually making the move.
    HashKey zobristKeyAfter(Pos pos) const
    {
        return currentZobristKey ^ Hash::zobristSide[~currentSide]
               ^ (pos != Pos::PASS ? Hash::zobrist[currentSide][pos] : HashKey {});
    }

    /// Get the current pattern4 accumulate counter for one side.
    uint16_t p4Count(Color side, Pattern4 p4) const { return stateInfo().p4Count[side][p4]; }

    // ------------------------------------------------------------------------
    // history board state queries

    /// Get history move pos according to move index.
    /// @param moveIndex Index of the move, in range [0, ply()).
    inline Pos getHistoryMove(int moveIndex) const
    {
        assert(moveIndex >= 0 && moveIndex < moveCount);
        return stateInfos[moveIndex + 1].lastMove;
    }

    /// Get recent history move pos according to reverse index.
    /// @param reverseIndex Reverse index starting from 0 (0 means the last move).
    inline Pos getRecentMove(int reverseIndex) const
    {
        assert(reverseIndex >= 0);
        int index = moveCount - reverseIndex;
        return index <= 0 ? Pos::NONE : stateInfos[index].lastMove;
    }

    /// Get state info of ply according to reverse index.
    /// @param reverseIndex Reverse index starting from 0 (0 means current state).
    inline const StateInfo &stateInfo(int reverseIndex = 0) const
    {
        assert(0 <= reverseIndex && reverseIndex <= moveCount);
        return stateInfos[moveCount - reverseIndex];
    }

    /// Get the last move played, same as getRecentMove(0).
    Pos getLastMove() const { return getRecentMove(0); }

    /// Get the last actual move (excluding null move) of one side.
    /// @param side Color of side to find move.
    /// @return Pos::NONE if there is no such move, otherwise returns the actual move.
    Pos getLastActualMoveOfSide(Color side) const;

    // ------------------------------------------------------------------------
    // miscellaneous

    /// Expand candidate area around the given pos.
    /// @param pos Pos on board to be expanded.
    /// @param fillDist Distance of filled square that will be expanded.
    /// @param lineDist Distance of line that will be expanded.
    void expandCandArea(Pos pos, int fillDist, int lineDist);

    /// Construct a position string from current board state.
    /// Position string example: `h8h7i7g9`.
    std::string positionString() const;

    /// Trace the current board state (used for debugging).
    std::string trace() const;

private:
    /// Saved per-cell state of one updated cell, so undo() can restore it without recomputation.
    struct SingleCellUpdateCache
    {
        Pattern4 pattern4[SIDE_NB];
        Score    score[SIDE_NB];
        Value    valueBlack;
    };
    /// Per-move scratch for the cells a single move touches. A move updates at most 2*L cells per
    /// direction over 4 directions; with the largest half-line length L=5 that is 40 cells.
    using UpdateCache = std::array<SingleCellUpdateCache, 40>;

    /// The board cell grid. Sized to the padded coordinate space (boundary walls included) so
    /// neighbour accesses never need an in-range check.
    Cell cells[FULL_BOARD_CELL_COUNT];

    // Per-direction bitkeys: 2 bits per cell, indexed so one line maps to a contiguous run of
    // bits. Each array is indexed by the line's fixed coordinate; the bracket shows bit order.
    uint64_t bitKey0[FULL_BOARD_SIZE];          // horizontal   [RIGHT(MSB) - LEFT(LSB)]
    uint64_t bitKey1[FULL_BOARD_SIZE];          // vertical     [DOWN(MSB) - UP(LSB)]
    uint64_t bitKey2[FULL_BOARD_SIZE * 2 - 1];  // main diag    [UP_RIGHT(MSB) - DOWN_LEFT(LSB)]
    uint64_t bitKey3[FULL_BOARD_SIZE * 2 - 1];  // anti diag    [DOWN_RIGHT(MSB) - UP_LEFT(LSB)]

    int     boardSize;           ///< Side length of the board.
    int     boardCellCount;      ///< Number of playable cells (boardSize^2).
    int     moveCount;           ///< Number of moves played (stones + passes).
    int     passCount[SIDE_NB];  ///< Number of passes by each side.
    Color   currentSide;         ///< The side to move.
    HashKey currentZobristKey;   ///< Zobrist key of the position (side-to-move excluded).
    std::unique_ptr<StateInfo[]>   stateInfos;   ///< Per-ply state history, indexed by ply.
    std::unique_ptr<UpdateCache[]> updateCache;  ///< Per-ply saved cell state for undo, by ply.
    const Direction *candidateRange;  ///< Offset table defining a move's candidate neighborhood.
    uint32_t         candidateRangeSize;  ///< Number of offsets in `candidateRange`.
    uint32_t         candAreaExpandDist;  ///< Candidate-area expansion distance per move.
    /// `candidateRange` precomputed as per-row dx-masks, so move() marks a stone's candidate
    /// neighborhood with a few word-wide ORs instead of a per-offset scatter.
    Bitboard::Stencil candStencil;
    Bitboard onBoardBB;  ///< All non-wall cells; built once in newGame (static for the game).
    Bitboard emptyBB;    ///< All empty cells; one bit flipped per move()/undo().
    Evaluation::Evaluator *evaluator_;   ///< External evaluator (may be null).
    Search::SearchThread  *thisThread_;  ///< Owning search thread (may be null).

    void setBitKey(Pos pos, Color c);
    void flipBitKey(Pos pos, Color c);
};

/// OR color `c`'s bit into all four directional bitkeys at `pos` (used to seed empty cells).
inline void Board::setBitKey(Pos pos, Color c)
{
    assert(c == BLACK || c == WHITE);
    int x = pos.x() + BOARD_BOUNDARY;
    int y = pos.y() + BOARD_BOUNDARY;

    const uint64_t mask = 0x1 + c;
    bitKey0[y] |= mask << (2 * x);
    bitKey1[x] |= mask << (2 * y);
    bitKey2[x + y] |= mask << (2 * x);
    bitKey3[FULL_BOARD_SIZE - 1 - x + y] |= mask << (2 * x);
}

/// Toggle color `c`'s bit in all four directional bitkeys at `pos` (placing or removing a stone).
inline void Board::flipBitKey(Pos pos, Color c)
{
    assert(c == BLACK || c == WHITE);
    int x = pos.x() + BOARD_BOUNDARY;
    int y = pos.y() + BOARD_BOUNDARY;

    assert(x >= 0 && x < FULL_BOARD_SIZE);
    assert(y >= 0 && y < FULL_BOARD_SIZE);
    assert(x + y >= 0 && x + y < 2 * FULL_BOARD_SIZE - 1);
    assert(FULL_BOARD_SIZE - 1 - x + y >= 0
           && FULL_BOARD_SIZE - 1 - x + y < 2 * FULL_BOARD_SIZE - 1);

    const uint64_t mask = 0x1 + c;
    bitKey0[y] ^= mask << (2 * x);
    bitKey1[x] ^= mask << (2 * y);
    bitKey2[x + y] ^= mask << (2 * x);
    bitKey3[FULL_BOARD_SIZE - 1 - x + y] ^= mask << (2 * x);
}

/// Extract the line bitkey centered on `pos` in direction `dir`, rotated so the center cell sits
/// at the table-expected position for pattern lookup.
template <Rule R>
inline uint64_t Board::getKeyAt(Pos pos, int dir) const
{
    assert(dir >= 0 && dir < 4);

    constexpr int L = PatternConfig::HalfLineLen<R>;
    int           x = pos.x() + BOARD_BOUNDARY;
    int           y = pos.y() + BOARD_BOUNDARY;

    switch (dir) {
    default:
    case 0: return rotr(bitKey0[y], 2 * (x - L));
    case 1: return rotr(bitKey1[x], 2 * (y - L));
    case 2: return rotr(bitKey2[x + y], 2 * (x - L));
    case 3: return rotr(bitKey3[FULL_BOARD_SIZE - 1 - x + y], 2 * (x - L));
    }
}

inline void Board::newGame(Rule rule)
{
    switch (rule) {
    case FREESTYLE: return newGame<FREESTYLE>();
    case STANDARD: return newGame<STANDARD>();
    case RENJU: return newGame<RENJU>();
    default: assert(false && "invalid rule");
    }
}

inline void Board::move(Rule rule, Pos pos)
{
    switch (rule) {
    case FREESTYLE: return move<FREESTYLE>(pos);
    case STANDARD: return move<STANDARD>(pos);
    case RENJU: return move<RENJU>(pos);
    default: assert(false && "invalid rule");
    }
}

inline void Board::undo(Rule rule)
{
    switch (rule) {
    case FREESTYLE: return undo<FREESTYLE>();
    case STANDARD: return undo<STANDARD>();
    case RENJU: return undo<RENJU>();
    default: assert(false && "invalid rule");
    }
}
