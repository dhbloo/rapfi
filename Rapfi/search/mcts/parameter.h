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

#include "../../config.h"
#include "../../tuning/tunemap.h"

#include <cstdint>

namespace Search::MCTS {

inline float MaxNewVisitsProp = 0.36f;
TUNE(MaxNewVisitsProp);

inline float CpuctExploration     = 0.39f;
inline float CpuctExplorationLog  = 0.98f;
inline float CpuctExplorationBase = 340;
TUNE(CpuctExploration);
TUNE(CpuctExplorationLog);
TUNE(CpuctExplorationBase);

inline float CpuctUtilityStdevScale     = 0.043f;
inline float CpuctUtilityVarPrior       = 0.16f;
inline float CpuctUtilityVarPriorWeight = 1.87f;
TUNE(CpuctUtilityStdevScale);
TUNE(CpuctUtilityVarPrior);
TUNE(CpuctUtilityVarPriorWeight);

inline float FpuReductionMax     = 0.06f;
inline float FpuLossProp         = 0.0008f;
inline float RootFpuReductionMax = 0.073f;
inline float RootFpuLossProp     = 0.0036f;
inline float FpuUtilityBlendPow  = 0.84f;
TUNE(FpuReductionMax);
TUNE(FpuLossProp);
TUNE(RootFpuReductionMax);
TUNE(RootFpuLossProp);
TUNE(FpuUtilityBlendPow);

inline uint32_t MinTranspositionSkipVisits = 11;

inline bool  UseLCBForBestmoveSelection = true;
inline float LCBStdevs                  = 6.28f;
inline float LCBMinVisitProp            = 0.1f;

inline float PolicyTemperature     = 0.91f;
inline float RootPolicyTemperature = 1.05f;
TUNE(RootPolicyTemperature, 0.7f, 1.1f);
TUNE(PolicyTemperature, 0.7f, 1.3f);

inline float ChildDrawPow  = 1.0f;
inline float ParentDrawPow = 1.0f;
TUNE(ChildDrawPow);
TUNE(ParentDrawPow);
TUNE(Config::DrawUtilityPenalty, 0.0f, 1.0f);

}  // namespace Search::MCTS
