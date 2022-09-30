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

#include <vector>

namespace Tuning {

template <typename T>
class AdamOptimizer
{
public:
    AdamOptimizer(std::size_t numParams,
                  T      lr          = T(0.001),
                  T      weightDecay = T(0.0),
                  T      beta1       = T(0.9),
                  T      beta2       = T(0.999),
                  T      epsilon     = T(1e-8));

    void step(std::vector<T> &params, const std::vector<T> &gradients);

    T currentLR() const { return lr; }

private:
    std::vector<T> m, v;
    T              lr, weightDecay;
    T              beta1, beta2;
    T              epsilon;
    T              t;
};

}  // namespace Tuning