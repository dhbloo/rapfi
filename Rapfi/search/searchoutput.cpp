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

#include "searchoutput.h"

#include "../config.h"
#include "../core/iohelper.h"
#include "../game/board.h"
#include "searchthread.h"
#include "timecontrol.h"

#define REALTIME(type, pos, size)                                                  \
    MESSAGEL("REALTIME " << (type) << ' ' << outputCoordXConvert(pos, size) << ',' \
                         << outputCoordYConvert(pos, size))

#define INFO(type, ...) sync_cout() << "INFO " << type << ' ' << __VA_ARGS__ << std::endl

namespace Search {

void SearchPrinter::printSearchStarts(MainSearchThread &th, const TimeControl &tc)
{
    if (Config::MessageMode == MsgMode::NORMAL) {
        if (th.options().timeLimit && !th.inPonder) {
            MESSAGEL("OptiTime " << timeText(tc.optimum()) << " | MaxTime "
                                 << timeText(tc.maximum()));
        }
    }
}

void SearchPrinter::printEnteringMove(MainSearchThread  &th,
                                      const TimeControl &tc,
                                      int                pvIdx,
                                      int                rootDepth,
                                      Pos                move)
{
    if (showRealtimeInLoop(th, tc, rootDepth) && !Config::AspirationWindow && pvIdx == 0
        && !th.inPonder.load(std::memory_order_relaxed))
        REALTIME("POS", move, th.board->size());
}

void SearchPrinter::printLeavingMove(MainSearchThread  &th,
                                     const TimeControl &tc,
                                     int                pvIdx,
                                     int                rootDepth,
                                     Pos                move)
{
    if (showRealtimeInLoop(th, tc, rootDepth) && !Config::AspirationWindow && pvIdx == 0
        && !th.inPonder.load(std::memory_order_relaxed))
        REALTIME("DONE", move, th.board->size());
}

void SearchPrinter::printMoveResult(MainSearchThread  &th,
                                    const TimeControl &tc,
                                    int                pvIdx,
                                    int                numPv,
                                    int                rootDepth,
                                    Pos                move,
                                    Value              moveValue,
                                    bool               isNewBest)
{
    if (showRealtimeInLoop(th, tc, rootDepth) && pvIdx == 0
        && !th.inPonder.load(std::memory_order_relaxed)) {
        if (moveValue <= VALUE_MATED_IN_MAX_PLY)
            REALTIME("LOST", move, th.board->size());
        else if (isNewBest)
            REALTIME("BEST", move, th.board->size());
    }
}

void SearchPrinter::printOutOfWindowResult(MainSearchThread  &th,
                                           const TimeControl &tc,
                                           int                rootDepth,
                                           int                pvIdx,
                                           int                numPv,
                                           Value              alpha,
                                           Value              beta)
{
    return;
}

void SearchPrinter::printPvCompletes(MainSearchThread  &th,
                                     const TimeControl &tc,
                                     int                rootDepth,
                                     int                pvIdx,
                                     int                numPv)
{
    // Do not print search messages in ponder mode
    if (th.inPonder.load(std::memory_order_relaxed))
        return;

    uint64_t nodes = th.threads.nodesSearched();
    uint64_t speed = nodes * 1000 / std::max(tc.elapsed(), (Time)1);
    if (!th.threads.isTerminating()) {
        RootMove &curMove = th.rootMoves[pvIdx];

        if (showInfo(th)) {
            INFO("PV", pvIdx);
            INFO("NUMPV", numPv);
            INFO("DEPTH", rootDepth);
            INFO("SELDEPTH", curMove.selDepth);
            INFO("NODES", curMove.numNodes);
            INFO("TOTALNODES", nodes);
            INFO("TOTALTIME", tc.elapsed());
            INFO("SPEED", speed);
            INFO("EVAL", curMove.value);
            INFO("WINRATE", Config::valueToWinRate(curMove.value));
            INFO("BESTLINE", MovesText {curMove.pv, true, true, th.board->size()});
            INFO("PV", "DONE");
        }

        if (numPv > 1 && Config::MessageMode == MsgMode::NORMAL)
            MESSAGEL("(" << pvIdx + 1 << ") " << curMove.value << " | " << rootDepth << "-"
                         << curMove.selDepth << " | " << MovesText {curMove.pv});
        else if (Config::MessageMode == MsgMode::UCILIKE) {
            if (numPv > 1)
                MESSAGEL("depth " << rootDepth << "-" << curMove.selDepth << " multipv "
                                  << pvIdx + 1 << " ev " << curMove.value << " n "
                                  << nodesText(nodes) << " n/ms " << (speed / 1000) << " tm "
                                  << tc.elapsed() << " pv " << MovesText {curMove.pv});
            else
                MESSAGEL("depth " << rootDepth << "-" << curMove.selDepth << " ev " << curMove.value
                                  << " n " << nodesText(nodes) << " n/ms " << (speed / 1000)
                                  << " tm " << tc.elapsed() << " pv " << MovesText {curMove.pv});
        }
    }

    if (showRealtime(th, tc, rootDepth)) {
        MESSAGEL("REALTIME REFRESH");
        REALTIME("BEST", th.rootMoves[0].pv[0], th.board->size());
    }
}

void SearchPrinter::printDepthCompletes(MainSearchThread &th, const TimeControl &tc, int rootDepth)
{
    if (Config::MessageMode == MsgMode::NORMAL) {
        bool showPonder = th.inPonder.load(std::memory_order_relaxed);

        MESSAGEL((showPonder ? "[Pondering] " : "")
                 << "Depth " << rootDepth << "-" << th.rootMoves[0].selDepth << " | Eval "
                 << th.rootMoves[0].value << " | Time " << timeText(tc.elapsed()) << " | "
                 << MovesText {th.rootMoves[0].pv});
    }
}

void SearchPrinter::printRootMoves(MainSearchThread  &th,
                                   const TimeControl &tc,
                                   size_t             numRootMovesToDisplay)
{
    // Do not print search messages in ponder mode
    if (th.inPonder.load(std::memory_order_relaxed))
        return;

    uint64_t nodes = th.threads.nodesSearched();
    uint64_t speed = nodes * 1000 / std::max(tc.elapsed(), (Time)1);

    numRootMovesToDisplay = std::min(numRootMovesToDisplay, th.rootMoves.size());
    for (size_t pvIdx = 0; pvIdx < numRootMovesToDisplay; pvIdx++) {
        RootMove &curMove = th.rootMoves[pvIdx];

        if (showInfo(th)) {
            INFO("PV", pvIdx);
            INFO("NUMPV", numRootMovesToDisplay);
            INFO("SELDEPTH", curMove.selDepth);
            INFO("NODES", curMove.numNodes);
            INFO("TOTALNODES", nodes);
            INFO("TOTALTIME", tc.elapsed());
            INFO("SPEED", speed);
            INFO("EVAL", curMove.value);
            INFO("WINRATE", curMove.winRate);
            INFO("DRAWRATE", curMove.drawRate);
            INFO("PRIOR", curMove.policyPrior);
            INFO("STDEV", curMove.utilityStdev);
            INFO("LCBVALUE", curMove.lcbValue);
            INFO("BESTLINE", MovesText {curMove.pv, true, true, th.board->size()});
            INFO("PV", "DONE");
        }

        std::ios oldState(nullptr);
        oldState.copyfmt(std::cout);
        std::cout << std::fixed << std::setprecision(2);

        if (Config::MessageMode == MsgMode::NORMAL) {
            MESSAGEL("(" << pvIdx + 1 << ") " << curMove.value << " (W " << (curMove.winRate * 100)
                         << ", D " << (curMove.drawRate * 100) << ", S " << curMove.utilityStdev
                         << ") | V " << nodesText(curMove.numNodes) << " | SD " << curMove.selDepth
                         << " | " << MovesText {curMove.pv});
        }
        else if (Config::MessageMode == MsgMode::UCILIKE) {
            MESSAGEL("multipv " << pvIdx + 1 << " ev " << curMove.value << " w "
                                << (curMove.winRate * 100) << " d " << (curMove.drawRate * 100)
                                << " stdev " << curMove.utilityStdev << " v "
                                << nodesText(curMove.numNodes) << " seldepth " << curMove.selDepth
                                << " n " << nodesText(nodes) << " n/ms " << (speed / 1000) << " tm "
                                << tc.elapsed() << " prior " << curMove.policyPrior << " pv "
                                << MovesText {curMove.pv});
        }

        std::cout.copyfmt(oldState);
    }

    if (Config::MessageMode == MsgMode::NORMAL) {
        MESSAGEL("Speed " << speedText(speed) << " | Visit " << nodesText(nodes) << " | Time "
                          << timeText(tc.elapsed()));
    }
}

void SearchPrinter::printSearchEnds(MainSearchThread  &th,
                                    const TimeControl &tc,
                                    int                rootDepth,
                                    SearchThread      &bestThread)
{
    if (Config::MessageMode == MsgMode::NORMAL || Config::MessageMode == MsgMode::BRIEF) {
        uint64_t nodes      = th.threads.nodesSearched();
        uint64_t speed      = nodes * 1000 / std::max(tc.elapsed(), (Time)1);
        bool     showPonder = th.inPonder.load(std::memory_order_relaxed);

        MESSAGEL((showPonder ? "[Pondering] " : "")
                 << "Speed " << speedText(speed) << " | Depth " << rootDepth << "-"
                 << bestThread.rootMoves[0].selDepth << " | Eval " << bestThread.rootMoves[0].value
                 << " | Node " << nodesText(nodes) << " | Time " << timeText(tc.elapsed()));

        // Outputs full PV if not shown before
        if (Config::MessageMode == MsgMode::BRIEF || &bestThread != &th) {
            // Select a longer PV for final output
            if (bestThread.rootMoves[0].pv.size() <= 2
                && bestThread.rootMoves[0].previousPv.size() > 2)
                MESSAGEL("Bestline " << MovesText {bestThread.rootMoves[0].previousPv});
            else
                MESSAGEL("Bestline " << MovesText {bestThread.rootMoves[0].pv});
        }
    }
}

void SearchPrinter::printBestmoveWithoutSearch(MainSearchThread &th,
                                               Pos               bestMove,
                                               Value             moveValue,
                                               int               rootDepth,
                                               std::vector<Pos> *pv)
{
    if (showInfo(th)) {
        INFO("PV", 0);
        INFO("NUMPV", 1);
        INFO("DEPTH", rootDepth);
        INFO("SELDEPTH", 0);
        INFO("NODES", 0);
        INFO("TOTALNODES", 0);
        INFO("TOTALTIME", 0);
        INFO("SPEED", 0);
        INFO("EVAL", moveValue);
        INFO("WINRATE", Config::valueToWinRate(moveValue));
        INFO("BESTLINE", MovesText {*pv, true, true, th.board->size()});
        INFO("PV", "DONE");
    }

    // A message for compatibility with UCI
    if (Config::MessageMode == MsgMode::UCILIKE) {
        if (pv)
            MESSAGEL("depth " << rootDepth << "-" << 0 << " ev " << moveValue
                              << " n 0 n/ms 0 tm 0 pv " << MovesText {*pv});
        else
            MESSAGEL("depth " << rootDepth << "-" << 0 << " ev " << moveValue
                              << " n 0 n/ms 0 tm 0 pv (NONE)");
    }
}

bool SearchPrinter::showRealtime(MainSearchThread &th, const TimeControl &tc, int rootDepth)
{
    return (th.options().infoMode & SearchOptions::INFO_REALTIME)
           && rootDepth >= REALTIME_MIN_DEPTH;
}

bool SearchPrinter::showRealtimeInLoop(MainSearchThread &th, const TimeControl &tc, int rootDepth)
{
    return showRealtime(th, tc, rootDepth) && tc.elapsed() >= REALTIME_MIN_ELAPSED
           && th.options().balanceMode != SearchOptions::BALANCE_TWO;
}

bool SearchPrinter::showInfo(MainSearchThread &th)
{
    return th.options().infoMode & SearchOptions::INFO_DETAIL;
}

}  // namespace Search
