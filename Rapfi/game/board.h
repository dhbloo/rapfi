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
#include "pattern.h"

#include <array>
#include <cassert>

namespace Search {
class SearchThread;
}
namespace Evaluation {
class Evaluator;
}

#define FOR_EVERY_POSITION(board, pos)                                   \
    for (Pos pos = (board)->startPos(); pos <= (board)->endPos(); pos++) \
        if ((board)->get(pos) != WALL)

#define FOR_EVERY_EMPTY_POS(board, pos)                                  \
    for (Pos pos = (board)->startPos(); pos <= (board)->endPos(); pos++) \
        if ((board)->isEmpty(pos))

#define FOR_EVERY_CANDAREA_POS(board, pos, candArea) \
    for (int8_t _y = (candArea).y0,                  \
                y1 = (candArea).y1,                  \
                x0 = (candArea).x0,                  \
                x1 = (candArea).x1,                  \
                _x = x0;                             \
         _y <= y1;                                   \
         _y++, _x = x0)                              \
        for (Pos pos {_x, _y}; _x <= x1; _x++, pos++)

#define FOR_EVERY_CAND_POS(board, pos)                                \
    FOR_EVERY_CANDAREA_POS(board, pos, (board)->stateInfo().candArea) \
    if ((board)->isEmpty(pos) && (board)->cell(pos).isCandidate())

/// CandArea struct represents a rectangle area on board which can be considered
/// as move candidate.
struct CandArea
{
    int8_t x0, y0, x1, y1;

    CandArea() : x0(INT8_MAX), y0(INT8_MAX), x1(INT8_MIN), y1(INT8_MIN) {}
    CandArea(int8_t x0, int8_t y0, int8_t x1, int8_t y1) : x0(x0), y0(y0), x1(x1), y1(y1) {}

    /// Expand candidate area at pos with a range of square with length 'dist'.
    void expand(Pos pos, int boardSize, int dist)
    {
        int x = pos.x(), y = pos.y();

        x0 = std::min((int)x0, std::max(x - dist, 0));
        y0 = std::min((int)y0, std::max(y - dist, 0));
        x1 = std::max((int)x1, std::min(x + dist, boardSize - 1));
        y1 = std::max((int)y1, std::min(y + dist, boardSize - 1));
    }
};

/// StateInfo struct records all incremental board information used in one ply.
struct StateInfo
{
    CandArea candArea;
    Pos      lastMove;
    Pos      lastFlex4AttackMove[SIDE_NB];
    Pos      lastPattern4Move[SIDE_NB][3];
    uint16_t p4Count[SIDE_NB][PATTERN4_NB];
    Value    valueBlack;

    /// Query the last emerged pattern4 pos.
    /// @note p4 must be one of [C_BLOCK4_FLEX3, B_FLEX4, A_FIVE].
    Pos lastPattern4(Color side, Pattern4 p4) const
    {
        assert(p4 >= C_BLOCK4_FLEX3 && p4 <= A_FIVE);
        return lastPattern4Move[side][p4 - C_BLOCK4_FLEX3];
    }
};

/// Cell struct contains all information for a move cell on board, including current
/// stone piece, candidate, pattern, pattern4 and move score.
struct Cell
{
    Color     piece;
    uint8_t   cand;
    Pattern4  pattern4[SIDE_NB];
    Score     score[SIDE_NB];
    Value     valueBlack;
    Pattern2x pattern2x[4];

    /// Check if this cell is a move candidate, which can be used in move generation.
    bool isCandidate() const { return cand > 0; }

    /// Get the line level pattern of this cell.
    Pattern pattern(Color c, int dir) const
    {
        assert(c == BLACK || c == WHITE);
        return c == BLACK ? pattern2x[dir].patBlack : pattern2x[dir].patWhite;
    }

    /// Get the complex pattern code at this cell for one side.
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

    /// Update the pattern4 and score with the new pattern code for both sides.
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

/// Board class is the main class used to represent a board position state.
/// It also records the whole board history info and bitboard state, to speed up
/// move() and undo() update methods. Copy constructor is explicit to avoid
/// unintended expensive copy operation.
class Board
{
public:
    /// MoveType represents the update mode of move/undo.
    enum class MoveType { NORMAL, NO_EVALUATOR, NO_EVAL, NO_EVAL_MULTI };

    /// Creates a board with board size and condidate range.
    /// @param boardSize Size of the board, in range [1, MAX_BOARD_SIZE].
    explicit Board(int boardSize, CandidateRange candRange = Config::DefaultCandidateRange);
    /// Clone a board object from other board and bind a search thread to it.
    /// @param other Board object to clone from.
    /// @param thread Search thread to be binded (nullptr for not binding).
    explicit Board(const Board &other, Search::SearchThread *thread);
    Board(const Board &) = delete;
    ~Board();

    // ------------------------------------------------------------------------
    // board modifier (and dynamic dispatch version)

    /// Initialize the board to an empty board state of rule R.
    /// @tparam R Rule to initialize the board.
    template <Rule R>
    void newGame();

    /// Make move and incremental update the board state.
    /// @param pos Pos to put the next stone. A Pass move is allowed.
    /// @tparam R Game rule to use.
    /// @tparam MT Type of this move. There are four types of move:
    ///     1. NORMAL: Updates cell, pattern, score, eval and external evaluator.
    ///     2. NO_EVALUATOR: Updates cell, pattern, score, eval.
    ///     3. NO_EVAL: Updates cell, pattern, score.
    ///     4. NO_EVAL_MULTI: Updates cell, pattern, score. Side to move is not swapped.
    /// @note Recursive pass move is allowed, but the total number of null moves
    ///     must be not greater than MAX_PASS_MOVES. As long as consecutive pass
    ///     moves are not allowed, this condition should be met.
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

    /// Flip current side to move without recording in state info.
    /// This is only served for some special board checking proecess. It must be
    /// used in pair locally. If a pass is desired, use move(Pos::PASS) instead.
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

    /// Check if the pos is legal (on an empty cell or is a pass move).
    bool isLegal(Pos pos) const { return pos.valid() && (isEmpty(pos) || pos == Pos::PASS); }

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
    struct SingleCellUpdateCache
    {
        Pattern4 pattern4[SIDE_NB];
        Score    score[SIDE_NB];
        Value    valueBlack;
    };
    using UpdateCache = std::array<SingleCellUpdateCache, 40>;

    /// The cells array of the board. It is designed to be larger than the actual
    /// board size, thus relaxing the need to check if an index is in range.
    Cell cells[FULL_BOARD_CELL_COUNT];

    // Bitkeys of 4 directions, used as key to index pattern update.
    uint64_t bitKey0[FULL_BOARD_SIZE];          // [RIGHT(MSB) - LEFT(LSB)]
    uint64_t bitKey1[FULL_BOARD_SIZE];          // [DOWN(MSB) - UP(LSB)]
    uint64_t bitKey2[FULL_BOARD_SIZE * 2 - 1];  // [UP_RIGHT(MSB) - DOWN_LEFT(LSB)]
    uint64_t bitKey3[FULL_BOARD_SIZE * 2 - 1];  // [DOWN_RIGHT(MSB) - UP_LEFT(LSB)]

    int                    boardSize;           /// Size of the board
    int                    boardCellCount;      /// Number of cells of the board
    int                    moveCount;           /// Number of moves played (=numStones+numPasses)
    int                    passCount[SIDE_NB];  /// Number of passes for both sides
    Color                  currentSide;         /// The current side to move
    HashKey                currentZobristKey;   /// The current zobrist key
    StateInfo             *stateInfos;          /// StateInfo array pointer
    UpdateCache           *updateCache;         /// UpdateCache array pointer
    const Direction       *candidateRange;      /// Candidate array pointer
    uint32_t               candidateRangeSize;  /// Size of candidate array
    uint32_t               candAreaExpandDist;  /// Expand distance of candidate area
    Evaluation::Evaluator *evaluator_;          /// External evaluator pointer
    Search::SearchThread  *thisThread_;         /// External search thread pointer

    void setBitKey(Pos pos, Color c);
    void flipBitKey(Pos pos, Color c);
};

/// Set bitkey of 4 directions at pos to color.
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

/// Flip the color of bitkey of 4 directions at pos.
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
    assert(rule < RULE_NB);
    void (Board::*F[])() = {&Board::newGame<FREESTYLE>,
                            &Board::newGame<STANDARD>,
                            &Board::newGame<RENJU>};
    (this->*F[rule])();
}

inline void Board::move(Rule rule, Pos pos)
{
    assert(rule < RULE_NB);
    void (Board::*F[])(
        Pos) = {&Board::move<FREESTYLE>, &Board::move<STANDARD>, &Board::move<RENJU>};
    (this->*F[rule])(pos);
}

inline void Board::undo(Rule rule)
{
    assert(rule < RULE_NB);
    void (Board::*F[])() = {&Board::undo<FREESTYLE>, &Board::undo<STANDARD>, &Board::undo<RENJU>};
    (this->*F[rule])();
}
