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

#include "tuner.h"

namespace Tuning::detail {

struct MoveScoreLossSettings
{
    Float gamma;
    Float logitFactor;
};

MoveScoreLossSettings makeMoveScoreLossSettings(const TuningConfig &config);

Float computeMoveScoreLoss(const PreparedCorpus         &corpus,
                           size_t                        sample,
                           const std::vector<TuneParam> &params,
                           const MoveScoreLossSettings  &settings);

void computeMoveScoreGradient(const PreparedCorpus         &corpus,
                              size_t                        sample,
                              std::vector<TuneGradient>    &grads,
                              const std::vector<TuneParam> &params,
                              const MoveScoreLossSettings  &settings,
                              Float                         sampleWeight = Float(1));

}  // namespace Tuning::detail
