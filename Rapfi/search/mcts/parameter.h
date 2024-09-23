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

#include <cstdint>

namespace Search::MCTS {

constexpr float MaxNewVisitsProp = 0.275f;

constexpr float CpuctExploration     = 0.40f;
constexpr float CpuctExplorationLog  = 0.75f;
constexpr float CpuctExplorationBase = 336;
constexpr float CpuctParentVisitBias = 0.1f;

constexpr float CpuctUtilityStdevScale     = 0.035f;
constexpr float CpuctUtilityVarPrior       = 0.18f;
constexpr float CpuctUtilityVarPriorWeight = 2.14f;

constexpr float FpuReductionMax     = 0.075f;
constexpr float FpuLossProp         = 0.001f;
constexpr float RootFpuReductionMax = 0.075f;
constexpr float RootFpuLossProp     = 0.0036f;
constexpr float FpuUtilityBlendPow  = 1.73f;

constexpr uint32_t MinTranspositionSkipVisits = 12;

constexpr bool  UseLCBForBestmoveSelection = true;
constexpr float LCBStdevs                  = 5.0f;
constexpr float LCBMinVisitProp            = 0.12f;

constexpr float PolicyTemperature = 1.0f;

}  // namespace Search::MCTS
