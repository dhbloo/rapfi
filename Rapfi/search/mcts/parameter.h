/*
 *  Rapfi, a Gomoku/Renju playing engine supporting piskvork protocol.
 *  Copyright (C) 2024  Rapfi developers
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

namespace Search::MCTS {

constexpr float CpuctExploration     = 1.0f;
constexpr float CpuctExplorationLog  = 0.4f;
constexpr float CpuctExplorationBase = 500;
constexpr float CpuctParentVisitBias = 0.1f;

constexpr float FpuReductionMax     = 0.1f;
constexpr float FpuLossProp         = 0.0f;
constexpr float RootFpuReductionMax = 0.05f;
constexpr float RootFpuLossProp     = 0.0f;
constexpr float FpuUtilityBlendPow  = 2.0f;

constexpr uint32_t MinTranspositionSkipVisits = 10;

constexpr bool  UseLCBForBestmoveSelection = false;
constexpr float LCBStdevs = 5;
constexpr float MinVisitPropForLCB = 0.2f;

}  // namespace Search::MCTS
