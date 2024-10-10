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

inline float MaxNewVisitsProp = 0.275f;
TUNE(MaxNewVisitsProp);

inline float CpuctExploration     = 0.40f;
inline float CpuctExplorationLog  = 0.75f;
inline float CpuctExplorationBase = 336;
inline float CpuctParentVisitBias = 0.1f;
TUNE(CpuctExploration);
TUNE(CpuctExplorationLog);
TUNE(CpuctExplorationBase);
TUNE(CpuctParentVisitBias);

inline float CpuctUtilityStdevScale     = 0.035f;
inline float CpuctUtilityVarPrior       = 0.18f;
inline float CpuctUtilityVarPriorWeight = 2.14f;
TUNE(CpuctUtilityStdevScale, 0.0f, 0.25f);
TUNE(CpuctUtilityVarPrior);
TUNE(CpuctUtilityVarPriorWeight);

inline float FpuReductionMax     = 0.075f;
inline float FpuLossProp         = 0.001f;
inline float RootFpuReductionMax = 0.075f;
inline float RootFpuLossProp     = 0.0036f;
inline float FpuUtilityBlendPow  = 1.73f;
TUNE(FpuReductionMax);
TUNE(FpuLossProp, 0.0f, 0.01f);
TUNE(RootFpuReductionMax);
TUNE(RootFpuLossProp, 0.0f, 0.01f);
TUNE(FpuUtilityBlendPow);

inline uint32_t MinTranspositionSkipVisits = 12;
TUNE(MinTranspositionSkipVisits);

inline bool  UseLCBForBestmoveSelection = true;
inline float LCBStdevs                  = 5.0f;
inline float LCBMinVisitProp            = 0.12f;

TUNE(LCBStdevs);
TUNE(LCBMinVisitProp);

inline float RootPolicyTemperature = 1.0f;
inline float PolicyTemperature     = 1.0f;
TUNE(RootPolicyTemperature, 0.5f, 1.5f);
TUNE(PolicyTemperature, 0.5f, 1.5f);

TUNE(Config::MaxNumVisitsPerPlayout, 1, 256);

}  // namespace Search::MCTS
