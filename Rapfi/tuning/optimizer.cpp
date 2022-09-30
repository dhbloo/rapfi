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

#include "optimizer.h"

#include <cassert>
#include <cmath>

namespace Tuning {

template <typename T>
AdamOptimizer<T>::AdamOptimizer(std::size_t numParams,
                                T           lr,
                                T           weightDecay,
                                T           beta1,
                                T           beta2,
                                T           epsilon)
    : lr(lr)
    , weightDecay(weightDecay)
    , beta1(beta1)
    , beta2(beta2)
    , epsilon(epsilon)
    , t(0)
{
    m.resize(numParams);
    v.resize(numParams);
}

template <typename T>
void AdamOptimizer<T>::step(std::vector<T> &params, const std::vector<T> &gradients)
{
    const size_t numParams = params.size();
    assert(numParams == m.size());
    assert(numParams == gradients.size());

    t += T(1.0);

    for (size_t i = 0; i < numParams; i++) {
        m[i] = beta1 * m[i] + (T(1.0) - beta1) * gradients[i];
        v[i] = beta2 * v[i] + (T(1.0) - beta2) * gradients[i] * gradients[i];

        T m_corr = m[i] / (T(1.0) - std::pow(beta1, t));
        T v_corr = v[i] / (T(1.0) - std::pow(beta2, t));

        params[i] -= lr * (m_corr / (std::sqrt(v_corr) + epsilon) + weightDecay * params[i]);
    }
}

}  // namespace Tuning

template class Tuning::AdamOptimizer<float>;
template class Tuning::AdamOptimizer<double>;
