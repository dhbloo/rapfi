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

constexpr float MaxNewVisitsProp = 0.36f;

constexpr float CpuctExploration     = 0.35f;
constexpr float CpuctExplorationLog  = 1.02f;
constexpr float CpuctExplorationBase = 328;

constexpr float CpuctUtilityStdevScale     = 0.05f;
constexpr float CpuctUtilityVarPrior       = 0.15f;
constexpr float CpuctUtilityVarPriorWeight = 1.80f;

constexpr float FpuReductionMax     = 0.055f;
constexpr float FpuLossProp         = 0.0008f;
constexpr float RootFpuReductionMax = 0.07f;
constexpr float RootFpuLossProp     = 0.003f;
constexpr float FpuUtilityBlendPow  = 0.75f;

constexpr uint32_t MinTranspositionSkipVisits = 11;

constexpr bool  UseLCBForBestmoveSelection = true;
constexpr float LCBStdevs                  = 6.28f;
constexpr float LCBMinVisitProp            = 0.1f;

constexpr float PolicyTemperature     = 0.90f;
constexpr float RootPolicyTemperature = 1.05f;

}  // namespace Search::MCTS
