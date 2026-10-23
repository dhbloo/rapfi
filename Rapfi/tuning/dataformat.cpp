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

#include "dataformat.h"

#include <algorithm>
#include <cstring>
#include <istream>
#include <ostream>

namespace {

using namespace Tuning;

// ------------------------------------------------------------------------
// Wire record layouts (kept private: all encoding/decoding goes through the
// functions below; the layouts are shared with the Python Trainer and
// c-gomoku-cli, see the header notice)

/// Head of one .bin entry record, followed by uint16_t position[ply].
struct BinEntryHead
{
    uint16_t result : 2;     // game outcome: 0=loss, 1=draw, 2=win (side to move pov)
    uint16_t ply : 9;        // current number of stones on board
    uint16_t boardsize : 5;  // board size in [5-22]
    uint16_t rule : 3;       // gomocup rule number: 0=freestyle, 1=standard, 4=renju
    uint16_t move : 13;      // move output by the engine
};
static_assert(sizeof(BinEntryHead) == 4);

/// Head of one .binpack game record, followed by uint16_t position[initPly]
/// and PackedMove moveSequence[moveCount].
struct PackedEntryHead
{
    uint32_t boardSize : 5;   // board size in [5-22]
    uint32_t rule : 3;        // gomocup rule number: 0=freestyle, 1=standard, 4=renju
    uint32_t result : 4;      // game outcome: 0=loss, 1=draw, 2=win (first player pov)
    uint32_t totalPly : 10;   // total number of stones on board after game ended
    uint32_t initPly : 10;    // initial number of stones on board when game started
    uint32_t gameTag : 14;    // game tag of this game, reserved for future use
    uint32_t moveCount : 18;  // the count of move sequence
};
static_assert(sizeof(PackedEntryHead) == 8);

/// One (multi-pv) move of a .binpack game record.
struct PackedMove
{
    uint16_t isFirst : 1;   // is this move the first in multipv?
    uint16_t isLast : 1;    // is this move the last in multipv?
    uint16_t isNoEval : 1;  // does this move contain no eval info?
    uint16_t isPass : 1;    // is this a pass (yields the turn, places no stone)?
    uint16_t reserved : 2;  // reserved for future use
    uint16_t move : 10;     // move output from engine
    int16_t  eval;          // eval output from engine
};
static_assert(sizeof(PackedMove) == 4);

/// Whether the stream is cleanly positioned at end-of-stream.
bool atRecordBoundaryEOF(std::istream &is)
{
    return is.eof() || is.peek() == std::istream::traits_type::eof();
}

[[noreturn]] void throwBadMove(Pos pos, int boardsize, const char *what)
{
    throw DatasetError(std::string("wrong ") + what + " in dataset ([" + std::to_string(pos.x())
                       + "," + std::to_string(pos.y()) + "] in boardsize "
                       + std::to_string(boardsize) + ")");
}

}  // namespace

namespace Tuning {

uint16_t encodeU16Move(Pos move)
{
    if (move == Pos::NONE || move == Pos::PASS)
        return UINT16_MAX;  // should not happen, but we just set it to uint16_t(-1)
    else
        return (move.x() << 5) | move.y();
}

Pos decodeU16Move(uint16_t move)
{
    int x = (move >> 5) & 0x1f;
    int y = move & 0x1f;
    // Wire fields cover [0,31] but only [0, MAX_BOARD_SIZE) is a coordinate
    // (UINT16_MAX marks an absent move). Map anything else to Pos::NONE so
    // readers reject it through their isInBoard checks and throw the typed
    // DatasetError instead of tripping the Pos constructor's debug asserts.
    if (x >= MAX_BOARD_SIZE || y >= MAX_BOARD_SIZE)
        return Pos::NONE;
    return Pos {x, y};
}

// ------------------------------------------------------------------------

bool readBinEntry(std::istream &is, DataEntry *entry, BinDecodeScratch &scratch)
{
    if (atRecordBoundaryEOF(is))
        return false;

    BinEntryHead ehead;
    is.read(reinterpret_cast<char *>(&ehead), sizeof(BinEntryHead));
    if (!is)
        throw DatasetError("truncated entry head in dataset");

    // Check legality of the entry head
    if (ehead.boardsize < 5 || ehead.boardsize > MAX_BOARD_SIZE)
        throw DatasetError("wrong boardsize in dataset");
    Rule rule = decodeWireRule(ehead.rule);
    if (rule >= RULE_NB)
        throw DatasetError("wrong rule in dataset");
    if (ehead.result != 0 && ehead.result != 1 && ehead.result != 2)
        throw DatasetError("wrong result in dataset");
    if (ehead.ply > ehead.boardsize * ehead.boardsize)
        throw DatasetError("wrong ply in dataset");

    if (!entry) {
        // Just skip the position move sequence. A short skip only sets eofbit,
        // so the skipped length must be checked explicitly.
        is.ignore(ehead.ply * sizeof(uint16_t));
        if (is.gcount() != std::streamsize(ehead.ply * sizeof(uint16_t)))
            throw DatasetError("truncated entry in dataset");
        return true;
    }

    clearPayload(entry->payload);  // reset payload of the previous entry, if reused
    entry->position.clear();
    entry->position.reserve(ehead.ply);

    uint16_t position[MAX_MOVES];
    is.read(reinterpret_cast<char *>(position), ehead.ply * sizeof(uint16_t));
    if (!is)
        throw DatasetError("truncated entry in dataset");

    // Occupancy mask over the padded coordinate space for duplicate detection
    std::memset(scratch.seen, 0, sizeof(scratch.seen));
    auto testAndSet = [&scratch](Pos pos) {
        uint64_t &word = scratch.seen[uint16_t(pos) >> 6];
        uint64_t  bit  = uint64_t(1) << (uint16_t(pos) & 63);
        bool      dup  = word & bit;
        word |= bit;
        return dup;
    };

    for (uint16_t ply = 0; ply < ehead.ply; ply++) {
        Pos pos = decodeU16Move(position[ply]);
        if (!pos.isInBoard(ehead.boardsize, ehead.boardsize))
            throwBadMove(pos, ehead.boardsize, "move sequence");
        if (testAndSet(pos))
            throwBadMove(pos, ehead.boardsize, "move sequence (duplicate)");
        entry->position.push_back(pos);
    }

    Pos bestMove = decodeU16Move(ehead.move);
    if (!bestMove.isInBoard(ehead.boardsize, ehead.boardsize) || testAndSet(bestMove))
        throwBadMove(bestMove, ehead.boardsize, "best move");

    entry->move      = bestMove;
    entry->eval      = VALUE_NONE;  // represent as no eval
    entry->boardsize = ehead.boardsize;
    entry->rule      = rule;
    entry->result    = Result(ehead.result);
    return true;
}

void writeBinEntry(std::ostream &os, const DataEntry &entry)
{
    // The .bin wire has no pass flag; a pass anywhere in the entry is unrepresentable.
    if (entry.move == Pos::PASS
        || std::find(entry.position.begin(), entry.position.end(), Pos::PASS)
               != entry.position.end())
        throw DatasetError("simple binary format cannot store a pass move");

    BinEntryHead ehead;
    uint16_t     position[MAX_MOVES];

    ehead.result    = entry.result;
    ehead.ply       = (uint16_t)entry.position.size();
    ehead.boardsize = entry.boardsize;
    ehead.rule      = encodeWireRule(entry.rule);
    ehead.move      = encodeU16Move(entry.move);
    for (size_t i = 0; i < ehead.ply; i++)
        position[i] = encodeU16Move(entry.position[i]);

    os.write(reinterpret_cast<char *>(&ehead), sizeof(BinEntryHead));
    os.write(reinterpret_cast<char *>(position), sizeof(uint16_t) * ehead.ply);
}

// ------------------------------------------------------------------------

bool readPackedGame(std::istream        &is,
                    GameEntry           &game,
                    PackedDecodeScratch &scratch,
                    size_t               maxRecordBytes,
                    bool                 retainExtraPVs)
{
    if (atRecordBoundaryEOF(is))
        return false;

    PackedEntryHead ehead;
    is.read(reinterpret_cast<char *>(&ehead), sizeof(PackedEntryHead));
    if (!is)
        throw DatasetError("truncated game head in dataset");

    // Check legality of the entry head
    if (ehead.boardSize < 5 || ehead.boardSize > 22)
        throw DatasetError("wrong boardsize in dataset");
    Rule rule = decodeWireRule(ehead.rule);
    if (rule >= RULE_NB)
        throw DatasetError("wrong rule in dataset");
    if (ehead.result != 0 && ehead.result != 1 && ehead.result != 2)
        throw DatasetError("wrong result in dataset");
    if (ehead.totalPly > ehead.boardSize * ehead.boardSize)
        throw DatasetError("wrong ply in dataset");
    if (ehead.initPly > ehead.totalPly || ehead.initPly >= MAX_MOVES)
        throw DatasetError("wrong initial ply in dataset");

    // Reserve credits before any variable-size record allocation. The factor
    // of four covers geometric capacity, an old/new buffer pair during
    // reallocation, and allocator slack while the raw and decoded forms coexist.
    const uint64_t perWireMoveEnvelope = 4ULL
                                         * (sizeof(PackedMove) + sizeof(GameEntry::MoveData)
                                            + (retainExtraPVs ? sizeof(PVMove) : 0));
    constexpr uint64_t FixedRecordEnvelope = sizeof(GameEntry) + 4ULL * MAX_MOVES * sizeof(Pos);
    uint64_t recordEnvelope = FixedRecordEnvelope + uint64_t(ehead.moveCount) * perWireMoveEnvelope;
    if (recordEnvelope > maxRecordBytes)
        throw DatasetError("packed game allocation envelope exceeds the configured record limit ("
                           + std::to_string(recordEnvelope) + " bytes required)");

    game.boardsize = ehead.boardSize;
    game.rule      = rule;
    game.initPosition.clear();
    game.initPosition.reserve(ehead.initPly);
    game.moveSequence.clear();

    // The wire stores the result from the first mover's pov; GameEntry wants white pov.
    Color startSide = ehead.initPly % 2 == 0 ? BLACK : WHITE;
    game.result     = startSide == WHITE ? Result(ehead.result) : flipResult(Result(ehead.result));

    // Read the initial position move sequence
    uint16_t position[MAX_MOVES];
    is.read(reinterpret_cast<char *>(position), ehead.initPly * sizeof(uint16_t));
    if (!is)
        throw DatasetError("truncated game record in dataset");

    for (uint32_t ply = 0; ply < ehead.initPly; ply++) {
        Pos pos = decodeU16Move(position[ply]);
        if (!pos.isInBoard(game.boardsize, game.boardsize))
            throwBadMove(pos, game.boardsize, "move sequence");
        game.initPosition.push_back(pos);
    }

    // Bulk-read the whole move sequence, then decode from memory (one stream
    // read per game instead of one per move)
    scratch.buffer.resize(ehead.moveCount * sizeof(PackedMove));
    is.read(scratch.buffer.data(), scratch.buffer.size());
    if (ehead.moveCount > 0 && !is)
        throw DatasetError("truncated game record in dataset");

    for (uint32_t i = 0; i < ehead.moveCount; i++) {
        PackedMove moveData;
        std::memcpy(&moveData, scratch.buffer.data() + i * sizeof(PackedMove), sizeof(PackedMove));

        Pos pos;
        if (moveData.isPass)
            pos = Pos::PASS;
        else {
            pos = decodeU16Move(moveData.move);
            if (!pos.isInBoard(game.boardsize, game.boardsize))
                throwBadMove(pos, game.boardsize, "move sequence");
        }
        Eval eval = moveData.isNoEval ? (Eval)VALUE_NONE : (Eval)moveData.eval;

        if (moveData.isFirst)
            game.moveSequence.push_back({pos, eval, {}});
        else {
            // An extra multi-pv move of the current ply
            if (game.moveSequence.empty())
                throw DatasetError("multipv move without a first move in dataset");
            if (retainExtraPVs) {
                MovePayload &payload = game.moveSequence.back().payload;
                if (auto *pvs = std::get_if<ExtraPVArray>(&payload))
                    pvs->push_back({pos, eval});
                else
                    payload.emplace<ExtraPVArray>().push_back({pos, eval});
            }
        }
    }

    return true;
}

void writePackedGame(std::ostream &os, const GameEntry &game)
{
    // The wire has no pass flag for initial-position moves (only move records
    // carry isPass), so a pass in the opening is unrepresentable. Reject loudly
    // instead of encoding it as a bogus on-board stone.
    for (Pos p : game.initPosition)
        if (p == Pos::PASS)
            throw DatasetError(
                "packed binary format cannot store a pass move in the initial position");

    PackedEntryHead ehead;
    uint16_t        position[MAX_MOVES];

    // totalPly counts stones on board after the game ended (passes place none)
    uint32_t totalPly = (uint32_t)game.initPosition.size();
    for (const auto &m : game.moveSequence)
        totalPly += m.move != Pos::PASS;

    // The wire stores the result from the first mover's pov; GameEntry holds white pov.
    Color startSide = game.initPosition.size() % 2 == 0 ? BLACK : WHITE;

    ehead.boardSize = game.boardsize;
    ehead.rule      = encodeWireRule(game.rule);
    ehead.result    = startSide == WHITE ? game.result : flipResult(game.result);
    ehead.totalPly  = totalPly;
    ehead.initPly   = (uint32_t)game.initPosition.size();
    ehead.gameTag   = 0;
    // Move count is summed for all pv lists (the main move plus any extra PVs)
    uint32_t moveCount = 0;
    for (const auto &m : game.moveSequence) {
        const ExtraPVArray *pvs = extraPVs(m.payload);
        moveCount += 1 + (pvs ? (uint32_t)pvs->size() : 0);
    }
    ehead.moveCount = moveCount;

    // Write entry header, then the initial position
    os.write(reinterpret_cast<char *>(&ehead), sizeof(PackedEntryHead));
    for (size_t i = 0; i < ehead.initPly; i++)
        position[i] = encodeU16Move(game.initPosition[i]);
    os.write(reinterpret_cast<char *>(position), sizeof(uint16_t) * ehead.initPly);

    PackedMove moveData;
    moveData.reserved = 0;
    for (const auto &m : game.moveSequence) {
        const ExtraPVArray *pvs         = extraPVs(m.payload);
        size_t              numExtraPVs = pvs ? pvs->size() : 0;

        // Write main pv
        moveData.isFirst  = true;
        moveData.isLast   = numExtraPVs == 0;
        moveData.isNoEval = m.eval == VALUE_NONE;
        moveData.isPass   = m.move == Pos::PASS;
        moveData.move     = encodeU16Move(m.move);
        moveData.eval     = moveData.isNoEval ? 0 : m.eval;
        os.write(reinterpret_cast<char *>(&moveData), sizeof(PackedMove));

        // Write extra pvs if any
        moveData.isFirst = false;
        for (size_t i = 0; i < numExtraPVs; i++) {
            const PVMove &pv  = (*pvs)[i];
            moveData.isLast   = i == numExtraPVs - 1;
            moveData.isNoEval = pv.eval == VALUE_NONE;
            moveData.isPass   = pv.move == Pos::PASS;
            moveData.move     = encodeU16Move(pv.move);
            moveData.eval     = moveData.isNoEval ? 0 : pv.eval;
            os.write(reinterpret_cast<char *>(&moveData), sizeof(PackedMove));
        }
    }
}

}  // namespace Tuning
