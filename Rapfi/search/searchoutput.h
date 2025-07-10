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

#include "../core/pos.h"
#include "../core/types.h"

namespace Search {

class SearchThread;
class MainSearchThread;
class TimeControl;

/// SearchPrinter controls all message outputs during searching. It should
/// only be called from main search thread to avoid possible IO racing.
struct SearchPrinter
{
    /// Print when search starts.
    void printSearchStarts(MainSearchThread &th, const TimeControl &tc);
    /// Print when entering the subtree of one move.
    void printEnteringMove(MainSearchThread  &th,
                           const TimeControl &tc,
                           int                pvIdx,
                           int                rootDepth,
                           Pos                move);
    /// Print when leaving the subtree of one move.
    void printLeavingMove(MainSearchThread  &th,
                          const TimeControl &tc,
                          int                pvIdx,
                          int                rootDepth,
                          Pos                move);
    /// Print when getting the result of one move.
    void printMoveResult(MainSearchThread  &th,
                         const TimeControl &tc,
                         int                pvIdx,
                         int                numPv,
                         int                rootDepth,
                         Pos                move,
                         Value              moveValue,
                         bool               isNewBest);
    /// Print when the eval is outside of search window (happens in aspiration search)
    void printOutOfWindowResult(MainSearchThread  &th,
                                const TimeControl &tc,
                                int                rootDepth,
                                int                pvIdx,
                                int                numPv,
                                Value              alpha,
                                Value              beta);
    /// Print when one Pv search completes.
    void printPvCompletes(MainSearchThread  &th,
                          const TimeControl &tc,
                          int                rootDepth,
                          int                pvIdx,
                          int                numPv);
    /// Print when one iterative depth completes.
    void printDepthCompletes(MainSearchThread &th, const TimeControl &tc, int rootDepth);
    /// Print root moves after some visits completed. (MCTS)
    void printRootMoves(MainSearchThread &th, const TimeControl &tc, size_t numRootMovesToDisplay);
    /// Print when search finishes.
    void printSearchEnds(MainSearchThread  &th,
                         const TimeControl &tc,
                         int                rootDepth,
                         SearchThread      &bestThread);
    /// Print when search is not needed to choose a bestmove.
    /// @param bestMove The best move to print.
    /// @param moveValue The theoretical value of this best move.
    /// @param rootDepth The theoretical depth of thie best move.
    /// @param pv A optional Pv to print.
    void printBestmoveWithoutSearch(MainSearchThread &th,
                                    Pos               bestMove,
                                    Value             moveValue,
                                    int               rootDepth,
                                    std::vector<Pos> *pv);

private:
    /// Checks should we output realtime messages.
    bool showRealtime(MainSearchThread &th, const TimeControl &tc, int rootDepth);
    /// Checks should we output realtime messages in move picking loop.
    bool showRealtimeInLoop(MainSearchThread &th, const TimeControl &tc, int rootDepth);
    /// @brief Checks should we output info.
    bool showInfo(MainSearchThread &th);

    static constexpr int REALTIME_MIN_DEPTH   = 8;
    static constexpr int REALTIME_MIN_ELAPSED = 200;
};

}  // namespace Search
