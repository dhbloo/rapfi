/*
 *  Rapfi, a Gomoku/Renju playing engine supporting piskvork protocol.
 *  Copyright (C) 2022 Rapfi developers
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 */

#pragma once

#include "tuner.h"

namespace Tuning::detail {

MoveScoreScaleReport
makeMoveScoreReference(const TuningConfig                            &config,
                       const std::vector<MoveScoreReferencePosition> &referencePositions);

}  // namespace Tuning::detail
