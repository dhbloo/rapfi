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

#include "../game/board.h"
#include "../tuning/datawriter.h"
#include "evaluator.h"

#include <fstream>

namespace Evaluation {

class EvalLogger
{
public:
    EvalLogger(std::string logpath, int numMaxSamples, int sampleInterval = 0)
        : writer(logpath, 50000, nullptr, false)
        , numMaxSamples(numMaxSamples)
        , sampleInterval(sampleInterval)
        , sampleCount(0)
        , skipCount(0)
    {}
    ~EvalLogger() = default;

    bool checkSample()
    {
        if (sampleCount >= numMaxSamples)
            return false;
        if (skipCount-- <= 0) {
            skipCount = sampleInterval;
            return true;
        }
        return false;
    }

    void logValueAndPolicy(const Board &board, const ValueType &value, const PolicyBuffer &policy)
    {
        if (!checkSample())
            return;

        // Do not log positions with pass moves
        if (board.passMoveCount() > 0)
            return;

        std::vector<Pos> positions;
        for (int ply = 0; ply < board.ply(); ply++) {
            Pos move = board.getHistoryMove(ply);
            positions.push_back(move);
        }

        Tuning::DataEntry entry {
            positions,
            uint8_t(board.size()),
            FREESTYLE,
            Tuning::RESULT_UNKNOWN,
            Tuning::DataEntry::POLICY_ARRAY_FLOAT,
            Pos::NONE,
            Eval(value.value()),
            new float[board.cellCount() + 1],
        };

        PolicyBuffer policyCopy = policy;
        policyCopy.applySoftmax();

        size_t policyIndex = 0;
        FOR_EVERY_POSITION(&board, pos)
        {
            entry.policyF32[policyIndex++] = policyCopy[pos];
        }
        entry.policyF32[board.cellCount()] = 0.0f;  // PASS policy

        writer.writeEntryWithSoftValueTarget(entry, value.win(), value.loss(), value.draw());
        sampleCount++;
    }

private:
    Tuning::NumpyDataWriter writer;
    int                     numMaxSamples;
    int                     sampleInterval;
    int                     sampleCount;
    int                     skipCount;
};

}  // namespace Evaluation
