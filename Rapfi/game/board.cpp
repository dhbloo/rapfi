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

#include "board.h"

#include "../core/iohelper.h"
#include "../core/pos.h"
#include "../core/utils.h"
#include "../eval/evaluator.h"
#include "../search/searchthread.h"

#include <algorithm>
#include <cstring>  // for std::memset
#include <iomanip>
#include <sstream>
#include <tuple>

namespace {

/// Checks whether current p4Count in stateInfo matches that on board (used in debug).
bool checkP4(const Board *board)
{
    int p4[SIDE_NB][PATTERN4_NB] = {0};
    FOR_EVERY_EMPTY_POS(board, pos)
    {
        p4[BLACK][board->cell(pos).pattern4[BLACK]]++;
        p4[WHITE][board->cell(pos).pattern4[WHITE]]++;
    }
    for (Color c : {BLACK, WHITE})
        for (Pattern4 i = FORBID; i < PATTERN4_NB; i = Pattern4(i + 1)) {
            if (p4[c][i] != board->stateInfo().p4Count[c][i])
                return false;
        }
    return true;
}

}  // namespace

Board::Board(int boardSize, CandidateRange candRange)
    : boardSize(boardSize)
    , boardCellCount(boardSize * boardSize)
    , moveCount(0)
    , passCount {0, 0}
    , currentSide(BLACK)
    , currentZobristKey(0)
    , candidateRange(nullptr)
    , candidateRangeSize(0)
    , evaluator_(nullptr)
    , thisThread_(nullptr)
{
    assert(0 < boardSize && boardSize <= MAX_BOARD_SIZE);
    stateInfos  = new StateInfo[1 + boardCellCount * 2] {};
    updateCache = new UpdateCache[1 + boardCellCount * 2];

    // Set candidate range of the board
    switch (candRange) {
    case CandidateRange::SQUARE2:
        candidateRange     = RANGE_SQUARE2;
        candidateRangeSize = arraySize(RANGE_SQUARE2);
        candAreaExpandDist = 2;
        break;
    case CandidateRange::SQUARE2_LINE3:
        candidateRange     = RANGE_SQUARE2_LINE3;
        candidateRangeSize = arraySize(RANGE_SQUARE2_LINE3);
        candAreaExpandDist = 3;
        break;
    case CandidateRange::SQUARE3:
        candidateRange     = RANGE_SQUARE3;
        candidateRangeSize = arraySize(RANGE_SQUARE3);
        candAreaExpandDist = 3;
        break;
    case CandidateRange::SQUARE3_LINE4:
        candidateRange     = RANGE_SQUARE3_LINE4;
        candidateRangeSize = arraySize(RANGE_SQUARE3_LINE4);
        candAreaExpandDist = 3;
        break;
    case CandidateRange::SQUARE4:
        candidateRange     = RANGE_SQUARE4;
        candidateRangeSize = arraySize(RANGE_SQUARE4);
        candAreaExpandDist = 4;
        break;
    default:  // Full board condidate
        break;
    }
}

Board::Board(const Board &other, Search::SearchThread *thread)
    : boardSize(other.boardSize)
    , boardCellCount(other.boardCellCount)
    , moveCount(other.moveCount)
    , passCount {other.passCount[0], other.passCount[1]}
    , currentSide(other.currentSide)
    , currentZobristKey(other.currentZobristKey)
    , candidateRange(other.candidateRange)
    , candidateRangeSize(other.candidateRangeSize)
    , candAreaExpandDist(other.candAreaExpandDist)
    , evaluator_(thread ? thread->evaluator.get() : nullptr)
    , thisThread_(thread)
{
    std::copy_n(other.cells, FULL_BOARD_CELL_COUNT, cells);
    std::copy_n(other.bitKey0, arraySize(bitKey0), bitKey0);
    std::copy_n(other.bitKey1, arraySize(bitKey1), bitKey1);
    std::copy_n(other.bitKey2, arraySize(bitKey2), bitKey2);
    std::copy_n(other.bitKey3, arraySize(bitKey3), bitKey3);

    stateInfos  = new StateInfo[1 + boardCellCount * 2] {};
    updateCache = new UpdateCache[1 + boardCellCount * 2];
    // Only copy stateinfo in [0, moveCount]
    std::copy_n(other.stateInfos, 1 + moveCount, stateInfos);
    std::copy_n(other.updateCache, 1 + moveCount, updateCache);

    // Sync evaluator state with board state
    if (evaluator_)
        evaluator_->syncWithBoard(*this);
}

Board::~Board()
{
    delete[] stateInfos;
    delete[] updateCache;
}

template <Rule R>
void Board::newGame()
{
    // Zero out cells and bitkeys
    std::fill_n(cells, FULL_BOARD_CELL_COUNT, Cell {});
    std::fill_n(bitKey0, arraySize(bitKey0), 0);
    std::fill_n(bitKey1, arraySize(bitKey1), 0);
    std::fill_n(bitKey2, arraySize(bitKey2), 0);
    std::fill_n(bitKey3, arraySize(bitKey3), 0);

    // Init board state to empty
    moveCount         = 0;
    passCount[BLACK]  = 0;
    passCount[WHITE]  = 0;
    currentSide       = BLACK;
    currentZobristKey = Hash::zobrist[BLACK][FULL_BOARD_CELL_COUNT - 1];
    for (Pos i = Pos::FULL_BOARD_START; i < Pos::FULL_BOARD_END; i++) {
        cells[i].piece = i.isInBoard(boardSize, boardSize) ? EMPTY : WALL;

        if (cells[i].piece == EMPTY) {
            setBitKey(i, BLACK);
            setBitKey(i, WHITE);
        }
    }

    // Init state info of the first ply with rule R
    StateInfo &st = stateInfos[moveCount];
    std::memset(&st, 0, sizeof(StateInfo));

    Value valueBlack = VALUE_ZERO;
    FOR_EVERY_POSITION(this, pos)
    {
        Cell &c = cells[pos];

        for (int dir = 0; dir < 4; dir++) {
            c.pattern2x[dir] = PatternConfig::lookupPattern<R>(getKeyAt<R>(pos, dir));

            assert(c.pattern2x[dir].patBlack <= F1);
            assert(c.pattern2x[dir].patWhite <= F1);
        }

        PatternCode pcode[SIDE_NB] = {c.pcode<BLACK>(), c.pcode<WHITE>()};
        c.updatePattern4AndScore<R>(pcode[BLACK], pcode[WHITE]);
        st.p4Count[BLACK][c.pattern4[BLACK]]++;
        st.p4Count[WHITE][c.pattern4[WHITE]]++;
        valueBlack += c.valueBlack = Config::getValueBlack(R, pcode[BLACK], pcode[WHITE]);
    }
    st.valueBlack = valueBlack;
    st.candArea   = CandArea();

    // For full board candidate range, we manually set all empty cells to candidates.
    if (candidateRangeSize == 0)
        expandCandArea(centerPos(), size() / 2, 0);

    assert(checkP4(this));

    // Reset evaluator state to empty board
    if (evaluator_)
        evaluator_->initEmptyBoard();
}

template void Board::newGame<FREESTYLE>();
template void Board::newGame<STANDARD>();
template void Board::newGame<RENJU>();

template <Rule R, Board::MoveType MT>
void Board::move(Pos pos)
{
    // handle the case when the pos is a PASS move
    if (UNLIKELY(pos == Pos::PASS)) {
        assert(passMoveCount() < cellCount());

        StateInfo &st = stateInfos[++moveCount];
        st            = stateInfos[moveCount - 1];
        st.lastMove   = Pos::PASS;

        passCount[currentSide]++;
        currentSide = ~currentSide;

        // after move evaluator update
        if (MT == MoveType::NORMAL && evaluator_)
            evaluator_->afterPass(*this);
        return;
    }

    assert(pos.valid());
    assert(isEmpty(pos));

    // before move evaluator update
    if (MT == MoveType::NORMAL && evaluator_)
        evaluator_->beforeMove(*this, pos);

    UpdateCache &pc = updateCache[moveCount];
    StateInfo   &st = stateInfos[++moveCount];
    st              = stateInfos[moveCount - 1];
    st.lastMove     = pos;
    st.candArea.expand(pos, boardSize, candAreaExpandDist);

    cells[pos].piece = currentSide;
    currentZobristKey ^= Hash::zobrist[currentSide][pos];
    flipBitKey(pos, currentSide);

    Value deltaValueBlack            = VALUE_ZERO;
    int   f4CountBeforeMove[SIDE_NB] = {p4Count(BLACK, B_FLEX4), p4Count(WHITE, B_FLEX4)};
    int   updateCacheIdx             = 0;

    constexpr int L         = PatternConfig::HalfLineLen<R>;
    int           x         = pos.x() + BOARD_BOUNDARY;
    int           y         = pos.y() + BOARD_BOUNDARY;
    uint64_t      bitKey[4] = {
        rotr(bitKey0[y], 2 * (x - 2 * L)),
        rotr(bitKey1[x], 2 * (y - 2 * L)),
        rotr(bitKey2[x + y], 2 * (x - 2 * L)),
        rotr(bitKey3[FULL_BOARD_SIZE - 1 - x + y], 2 * (x - 2 * L)),
    };

    for (int i = -L; i <= L; i += 1 + (i == -1)) {
        for (int dir = 0; dir < 4; dir++) {
            Pos   posi = pos + DIRECTION[dir] * i;
            Cell &c    = cells[posi];
            if (c.piece != EMPTY)
                continue;

            if constexpr (MT == MoveType::NORMAL || MT == MoveType::NO_EVALUATOR) {
                deltaValueBlack -= c.valueBlack;
            }

            c.pattern2x[dir] = PatternConfig::lookupPattern<R>(bitKey[dir]);

            pc[updateCacheIdx].pattern4[BLACK] = c.pattern4[BLACK];
            pc[updateCacheIdx].pattern4[WHITE] = c.pattern4[WHITE];
            pc[updateCacheIdx].score[BLACK]    = c.score[BLACK];
            pc[updateCacheIdx].score[WHITE]    = c.score[WHITE];
            if constexpr (MT == MoveType::NORMAL || MT == MoveType::NO_EVALUATOR) {
                pc[updateCacheIdx].valueBlack = c.valueBlack;
            }
            updateCacheIdx++;

            PatternCode pcode[SIDE_NB] = {c.pcode<BLACK>(), c.pcode<WHITE>()};

            if constexpr (MT == MoveType::NORMAL || MT == MoveType::NO_EVALUATOR) {
                deltaValueBlack += c.valueBlack =
                    Config::getValueBlack(R, pcode[BLACK], pcode[WHITE]);
            }

            st.p4Count[BLACK][c.pattern4[BLACK]]--;
            st.p4Count[WHITE][c.pattern4[WHITE]]--;
            c.updatePattern4AndScore<R>(pcode[BLACK], pcode[WHITE]);
            st.p4Count[BLACK][c.pattern4[BLACK]]++;
            st.p4Count[WHITE][c.pattern4[WHITE]]++;

            if (c.pattern4[BLACK] >= C_BLOCK4_FLEX3)
                st.lastPattern4Move[BLACK][c.pattern4[BLACK] - C_BLOCK4_FLEX3] = posi;
            if (c.pattern4[WHITE] >= C_BLOCK4_FLEX3)
                st.lastPattern4Move[WHITE][c.pattern4[WHITE] - C_BLOCK4_FLEX3] = posi;
        }

        const int shamt = 2 + 2 * (i == -1);
        bitKey[0] >>= shamt;
        bitKey[1] >>= shamt;
        bitKey[2] >>= shamt;
        bitKey[3] >>= shamt;
    }

    const Cell &c = cell(pos);
    if (MT == MoveType::NORMAL || MT == MoveType::NO_EVALUATOR) {
        st.valueBlack += deltaValueBlack - c.valueBlack;
    }
    st.p4Count[BLACK][c.pattern4[BLACK]]--;
    st.p4Count[WHITE][c.pattern4[WHITE]]--;

    if (MT != MoveType::NO_EVAL_MULTI)
        currentSide = ~currentSide;

    assert(checkP4(this));
    assert(updateCacheIdx <= std::tuple_size_v<UpdateCache>);

    for (size_t i = 0; i < candidateRangeSize; i++)
        cells[pos + candidateRange[i]].cand++;

    for (Color c : {BLACK, WHITE}) {
        if (!f4CountBeforeMove[c] && p4Count(c, B_FLEX4))
            st.lastFlex4AttackMove[c] = pos;
    }

    // after move evaluator update
    if (MT == MoveType::NORMAL && evaluator_)
        evaluator_->afterMove(*this, pos);
}

template void Board::move<FREESTYLE, Board::MoveType::NORMAL>(Pos pos);
template void Board::move<FREESTYLE, Board::MoveType::NO_EVAL>(Pos pos);
template void Board::move<STANDARD, Board::MoveType::NORMAL>(Pos pos);
template void Board::move<STANDARD, Board::MoveType::NO_EVAL>(Pos pos);
template void Board::move<RENJU, Board::MoveType::NORMAL>(Pos pos);
template void Board::move<RENJU, Board::MoveType::NO_EVAL>(Pos pos);

template <Rule R, Board::MoveType MT>
void Board::undo()
{
    assert(moveCount > 0);
    Pos lastPos = getLastMove();

    // handle the case when the last move is a PASS
    if (UNLIKELY(lastPos == Pos::PASS)) {
        currentSide = ~currentSide;
        assert(passCount[currentSide] > 0);
        passCount[currentSide]--;
        moveCount--;

        // after undo evaluator update
        if (MT == MoveType::NORMAL && evaluator_)
            evaluator_->afterUndoPass(*this);
        return;
    }

    // before undo evaluator update
    if (MT == MoveType::NORMAL && evaluator_)
        evaluator_->beforeUndo(*this, lastPos);

    if (MT != MoveType::NO_EVAL_MULTI)
        currentSide = ~currentSide;
    assert(get(lastPos) == currentSide);

    flipBitKey(lastPos, currentSide);
    currentZobristKey ^= Hash::zobrist[currentSide][lastPos];
    cells[lastPos].piece = EMPTY;

    moveCount--;
    const UpdateCache &pc             = updateCache[moveCount];
    int                updateCacheIdx = 0;

    constexpr int L         = PatternConfig::HalfLineLen<R>;
    int           x         = lastPos.x() + BOARD_BOUNDARY;
    int           y         = lastPos.y() + BOARD_BOUNDARY;
    uint64_t      bitKey[4] = {
        rotr(bitKey0[y], 2 * (x - 2 * L)),
        rotr(bitKey1[x], 2 * (y - 2 * L)),
        rotr(bitKey2[x + y], 2 * (x - 2 * L)),
        rotr(bitKey3[FULL_BOARD_SIZE - 1 - x + y], 2 * (x - 2 * L)),
    };

    for (int i = -L; i <= L; i += 1 + (i == -1)) {
        for (int dir = 0; dir < 4; dir++) {
            Pos   posi = lastPos + DIRECTION[dir] * i;
            Cell &c    = cells[posi];
            if (c.piece != EMPTY)
                continue;

            c.pattern2x[dir]  = PatternConfig::lookupPattern<R>(bitKey[dir]);
            c.pattern4[BLACK] = pc[updateCacheIdx].pattern4[BLACK];
            c.pattern4[WHITE] = pc[updateCacheIdx].pattern4[WHITE];
            c.score[BLACK]    = pc[updateCacheIdx].score[BLACK];
            c.score[WHITE]    = pc[updateCacheIdx].score[WHITE];
            if constexpr (MT == MoveType::NORMAL || MT == MoveType::NO_EVALUATOR) {
                c.valueBlack = pc[updateCacheIdx].valueBlack;
            }
            updateCacheIdx++;
        }

        const int shamt = 2 + 2 * (i == -1);
        bitKey[0] >>= shamt;
        bitKey[1] >>= shamt;
        bitKey[2] >>= shamt;
        bitKey[3] >>= shamt;
    }

    assert(checkP4(this));
    assert(updateCacheIdx <= std::tuple_size_v<UpdateCache>);

    for (size_t i = 0; i < candidateRangeSize; i++)
        cells[lastPos + candidateRange[i]].cand--;

    // after undo evaluator update
    if (MT == MoveType::NORMAL && evaluator_)
        evaluator_->afterUndo(*this, lastPos);
}

template void Board::undo<FREESTYLE, Board::MoveType::NORMAL>();
template void Board::undo<FREESTYLE, Board::MoveType::NO_EVAL>();
template void Board::undo<STANDARD, Board::MoveType::NORMAL>();
template void Board::undo<STANDARD, Board::MoveType::NO_EVAL>();
template void Board::undo<RENJU, Board::MoveType::NORMAL>();
template void Board::undo<RENJU, Board::MoveType::NO_EVAL>();

bool Board::checkForbiddenPoint(Pos pos) const
{
    const Cell &fpCell = cell(pos);
    if (fpCell.pattern4[BLACK] != FORBID)
        return false;

    // This pos is a possible black forbidden point.
    // Still need to check possible false forbidden point.

    int winByFour = 0;
    for (int dir = 0; dir < 4; dir++) {
        // If this forbidden point is a Overline, it must be a true forbidden point.
        if (fpCell.pattern2x[dir].patBlack == OL)
            return true;
        // Otherwise if it has at least two Four(B4/F4), it must be a true forbidden point.
        else if (fpCell.pattern2x[dir].patBlack == B4 || fpCell.pattern2x[dir].patBlack == F4) {
            if (++winByFour >= 2)
                return true;
        }
    }

    // Check false forbidden point by putting stones recursively.
    // Cast out const qualifier of this as we guarantee not to modify its state.
    Board &board    = const_cast<Board &>(*this);
    Color  prevSide = board.currentSide;

    board.currentSide = BLACK;
    board.move<Rule::RENJU, MoveType::NO_EVAL_MULTI>(pos);

    constexpr int MaxFindDist = 4;
    int           winByThree  = 0;

    for (int dir = 0; dir < 4; dir++) {
        // Only look line that is possible to become a FLEX4 or FIVE
        Pattern p = fpCell.pattern2x[dir].patBlack;

        // double three forbidden type
        if (p != F3 && p != F3S)  // p must be one of F3, F3S
            continue;

        Pos posi = pos;
        for (int i = 0; i < MaxFindDist; i++) {
            posi -= DIRECTION[dir];

            if (const Cell &c = cell(posi); c.piece == EMPTY) {
                if (c.pattern4[BLACK] == B_FLEX4 || c.pattern(BLACK, dir) == F5
                    || c.pattern4[BLACK] == FORBID && c.pattern(BLACK, dir) == F4
                           && !checkForbiddenPoint(posi)) {
                    winByThree++;
                    goto next_direction;
                }
                break;
            }
            else if (c.piece != BLACK)
                break;
        }
        posi = pos;
        for (int i = 0; i < MaxFindDist; i++) {
            posi += DIRECTION[dir];

            if (const Cell &c = cell(posi); c.piece == EMPTY) {
                if (c.pattern4[BLACK] == B_FLEX4 || c.pattern(BLACK, dir) == F5
                    || c.pattern4[BLACK] == FORBID && c.pattern(BLACK, dir) == F4
                           && !checkForbiddenPoint(posi)) {
                    winByThree++;
                    goto next_direction;
                }
                break;
            }
            else if (c.piece != BLACK)
                break;
        }

    next_direction:
        if (winByThree >= 2)
            break;
    }

    board.undo<Rule::RENJU, MoveType::NO_EVAL_MULTI>();
    board.currentSide = prevSide;

    return winByThree >= 2;
}

Pos Board::getLastActualMoveOfSide(Color side) const
{
    assert(side == BLACK || side == WHITE);

    for (int reverseIdx = 0; reverseIdx < moveCount; reverseIdx++) {
        Pos move = getRecentMove(reverseIdx);
        if (move == Pos::PASS)
            continue;
        if (get(move) == side)
            return move;
    }

    return Pos::NONE;
}

void Board::expandCandArea(Pos pos, int fillDist, int lineDist)
{
    CandArea &area = stateInfos[moveCount].candArea;
    int       x = pos.x(), y = pos.y();

    auto candCondition = [&](Pos p) {
        return p >= 0 && p < FULL_BOARD_CELL_COUNT && isEmpty(p) && !cell(p).isCandidate();
    };

    area.expand(pos, boardSize, std::max(fillDist, lineDist));

    for (int i = std::max(3, fillDist + 1); i <= lineDist; i++) {
        for (int dir = 0; dir < 4; dir++) {
            Pos posi = pos + DIRECTION[dir] * i;
            if (candCondition(posi))
                cells[posi].cand++;
        }
    }
    for (int xi = -fillDist; xi <= fillDist; xi++) {
        for (int yi = -fillDist; yi <= fillDist; yi++) {
            Pos posi {x + xi, y + yi};
            if (candCondition(posi))
                cells[posi].cand++;
        }
    }
}

std::string Board::positionString() const
{
    std::stringstream ss;
    for (int i = 0; i < ply(); i++) {
        Pos pos = getHistoryMove(i);
        if (pos == Pos::PASS)
            ss << "--";
        else
            ss << char('a' + pos.x()) << (1 + pos.y());
    }
    return ss.str();
}

std::string Board::trace() const
{
    std::stringstream ss;
    const StateInfo  &st = stateInfo();

    ss << "Hash: " << std::hex << zobristKey() << std::dec << '\n';
    ss << "Ply: " << ply() << "\n";
    ss << "NonPassCount: " << nonPassMoveCount() << "\n";
    ss << "PassCount[Black]: " << passMoveCountOfSide(BLACK)
       << "  PassCount[White]: " << passMoveCountOfSide(WHITE) << "\n";
    ss << "SideToMove: " << sideToMove() << "\n";
    ss << "LastPos: " << getLastMove() << '\n';
    ss << "Eval[Black]: " << st.valueBlack << '\n';
    ss << "LastP4[Black][A]: " << st.lastPattern4(BLACK, A_FIVE)
       << "  LastP4[White][A]: " << st.lastPattern4(WHITE, A_FIVE) << '\n';
    ss << "LastP4[Black][B]: " << st.lastPattern4(BLACK, B_FLEX4)
       << "  LastP4[White][B]: " << st.lastPattern4(WHITE, B_FLEX4) << '\n';
    ss << "LastP4[Black][C]: " << st.lastPattern4(BLACK, C_BLOCK4_FLEX3)
       << "  LastP4[White][C]: " << st.lastPattern4(WHITE, C_BLOCK4_FLEX3) << '\n';
    ss << "LastF4[Black]: " << st.lastFlex4AttackMove[BLACK]
       << "  LastF4[White]: " << st.lastFlex4AttackMove[WHITE] << '\n';

    auto printBoard = [&](auto &&posTextFunc, int textWidth = 1) {
        FOR_EVERY_POSITION(this, pos)
        {
            int x = pos.x(), y = pos.y();
            if (x != 0 || y != 0)
                ss << ' ';
            if (x == 0 && y != 0)
                ss << '\n';
            posTextFunc(pos);
            if (x == size() - 1)
                ss << ' ' << y + 1;
        }
        ss << '\n';
        for (int x = 0; x < size(); x++)
            ss << std::setw(textWidth) << char(x + 65) << " ";
        ss << '\n';
    };

    auto printPiece = [&](Pos pos) {
        switch (get(pos)) {
        case BLACK: ss << 'X'; break;
        case WHITE: ss << 'O'; break;
        case EMPTY: ss << (cell(pos).isCandidate() ? '*' : '.'); break;
        default: ss << ' '; break;
        }
    };

    ss << "----------------Board----------------\n";
    printBoard(printPiece);

    ss << "----------Pattern4----Black----------\n";
    printBoard([&](Pos pos) {
        if (isEmpty(pos))
            ss << cell(pos).pattern4[BLACK];
        else
            ss << '.';
    });

    ss << "----------Pattern4----White----------\n";
    printBoard([&](Pos pos) {
        if (isEmpty(pos))
            ss << cell(pos).pattern4[WHITE];
        else
            ss << '.';
    });

    ss << "----------Score-------Black----------\n";
    printBoard(
        [&](Pos pos) {
            if (isEmpty(pos))
                ss << std::setw(3) << cell(pos).score[BLACK];
            else {
                ss << '[';
                printPiece(pos);
                ss << ']';
            }
        },
        3);

    ss << "----------Score-------White----------\n";
    printBoard(
        [&](Pos pos) {
            if (isEmpty(pos))
                ss << std::setw(3) << cell(pos).score[WHITE];
            else {
                ss << '[';
                printPiece(pos);
                ss << ']';
            }
        },
        3);

    if (evaluator_) {
        Evaluation::PolicyBuffer policyBuf(boardSize);
        policyBuf.setComputeFlagForAllEmptyCell(*this);

        // Calcualate policy for the current side
        evaluator_->evaluatePolicy(*this, policyBuf);

        ss << "----------Policy------Self-----------\n";
        printBoard(
            [&](Pos pos) {
                if (isEmpty(pos))
                    ss << std::setw(4) << policyBuf.score(pos);
                else {
                    ss << " [";
                    printPiece(pos);
                    ss << "]";
                }
            },
            4);

        // Calcualate for the opponent side
        const_cast<Board *>(this)->flipSide();
        evaluator_->evaluatePolicy(*this, policyBuf);
        const_cast<Board *>(this)->flipSide();

        ss << "----------Policy------Oppo-----------\n";
        printBoard(
            [&](Pos pos) {
                if (isEmpty(pos))
                    ss << std::setw(4) << policyBuf.score(pos);
                else {
                    ss << " [";
                    printPiece(pos);
                    ss << "]";
                }
            },
            4);
    }

    return ss.str();
}
