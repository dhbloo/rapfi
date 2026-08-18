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

#include <algorithm>
#include <cassert>
#include <cmath>
#include <limits>
#include <numeric>
#include <stdexcept>

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
    , stepCount(0)
{
    m.resize(numParams);
    v.resize(numParams);
    nextParams.resize(numParams);
    activeStepCounts.assign(numParams, 0);
}

template <typename T>
void AdamOptimizer<T>::stepMasked(std::vector<T>             &params,
                                  const std::vector<T>       &gradients,
                                  const std::vector<T>       &learningRates,
                                  const std::vector<uint8_t> &activeMask)
{
    const size_t numParams = params.size();
    assert(numParams == m.size());
    assert(numParams == gradients.size());
    assert(numParams == learningRates.size());
    assert(numParams == activeMask.size());

    for (size_t i = 0; i < numParams; i++) {
        if (!activeMask[i]) {
            nextParams[i] = params[i];
            continue;
        }
        const T parameterLR = learningRates[i];
        if (!std::isfinite(parameterLR) || parameterLR <= 0)
            throw std::invalid_argument("Adam received an invalid active parameter learning rate");

        if (activeStepCounts[i] == std::numeric_limits<size_t>::max())
            throw std::overflow_error("Adam active step count overflow");
        activeStepCounts[i]++;
        m[i] = beta1 * m[i] + (T(1.0) - beta1) * gradients[i];
        v[i] = beta2 * v[i] + (T(1.0) - beta2) * gradients[i] * gradients[i];

        const double biasCorrection1 = 1.0 - std::pow(double(beta1), double(activeStepCounts[i]));
        const double biasCorrection2 = 1.0 - std::pow(double(beta2), double(activeStepCounts[i]));
        const T      mCorr           = T(double(m[i]) / biasCorrection1);
        const T      vCorr           = T(double(v[i]) / biasCorrection2);
        const T      nextParam =
            params[i]
            - parameterLR * (mCorr / (std::sqrt(vCorr) + epsilon) + weightDecay * params[i]);
        if (!std::isfinite(m[i]) || !std::isfinite(v[i]) || !std::isfinite(nextParam))
            throw std::runtime_error("Adam produced a non-finite optimizer state");
        nextParams[i] = nextParam;
    }
    params.swap(nextParams);
}

template <typename T>
void AdamOptimizer<T>::step(std::vector<T> &params, const std::vector<T> &gradients)
{
    stepImpl(params, gradients, nullptr);
}

template <typename T>
void AdamOptimizer<T>::step(std::vector<T>       &params,
                            const std::vector<T> &gradients,
                            const std::vector<T> &learningRates)
{
    stepImpl(params, gradients, &learningRates);
}

template <typename T>
void AdamOptimizer<T>::stepImpl(std::vector<T>       &params,
                                const std::vector<T> &gradients,
                                const std::vector<T> *learningRates)
{
    const size_t numParams = params.size();
    assert(numParams == m.size());
    assert(numParams == gradients.size());
    assert(!learningRates || numParams == learningRates->size());
    if (learningRates)
        for (T parameterLR : *learningRates)
            if (!std::isfinite(parameterLR) || parameterLR <= 0)
                throw std::invalid_argument("Adam received an invalid parameter learning rate");

    stepCount++;

    // Bias correction is constant for every parameter in this step. Computing
    // these powers inside the parameter loop is particularly expensive for the
    // large policy table, and some compilers do not hoist std::pow themselves.
    const double biasCorrection1 = 1.0 - std::pow(double(beta1), double(stepCount));
    const double biasCorrection2 = 1.0 - std::pow(double(beta2), double(stepCount));

    for (size_t i = 0; i < numParams; i++) {
        m[i] = beta1 * m[i] + (T(1.0) - beta1) * gradients[i];
        v[i] = beta2 * v[i] + (T(1.0) - beta2) * gradients[i] * gradients[i];

        T m_corr = T(double(m[i]) / biasCorrection1);
        T v_corr = T(double(v[i]) / biasCorrection2);

        const T parameterLR = learningRates ? (*learningRates)[i] : lr;
        T       nextParam =
            params[i]
            - parameterLR * (m_corr / (std::sqrt(v_corr) + epsilon) + weightDecay * params[i]);
        if (!std::isfinite(m[i]) || !std::isfinite(v[i]) || !std::isfinite(nextParam))
            throw std::runtime_error("Adam produced a non-finite optimizer state");
        nextParams[i] = nextParam;
    }

    // Publish the complete update only after every candidate parameter passed
    // validation. A failed step cannot expose a partially updated vector.
    params.swap(nextParams);
}

template <typename T>
AdamOptimizer<T>
AdamOptimizer<T>::coalesced(const std::vector<std::vector<std::size_t>> &groups) const
{
    if (groups.empty())
        throw std::invalid_argument("Adam state coalescing requires a nonempty partition");

    std::vector<uint8_t> seen(m.size(), uint8_t(0));
    AdamOptimizer<T>     result(groups.size(), lr, weightDecay, beta1, beta2, epsilon);
    result.stepCount          = stepCount;
    const bool usesActiveAges = std::any_of(activeStepCounts.begin(),
                                            activeStepCounts.end(),
                                            [](size_t age) { return age != 0; });
    for (size_t groupIndex = 0; groupIndex < groups.size(); groupIndex++) {
        const std::vector<size_t> &group = groups[groupIndex];
        if (group.empty())
            throw std::invalid_argument("Adam state coalescing contains an empty group");
        for (size_t source : group) {
            if (source >= m.size())
                throw std::out_of_range("Adam state coalescing source index is out of range");
            if (seen[source]++)
                throw std::invalid_argument("Adam state coalescing partition overlaps");
        }

        if (group.size() == 1) {
            const size_t source                 = group.front();
            result.m[groupIndex]                = m[source];
            result.v[groupIndex]                = v[source];
            result.activeStepCounts[groupIndex] = activeStepCounts[source];
            continue;
        }

        const size_t mergedAge =
            usesActiveAges ? std::accumulate(group.begin(),
                                             group.end(),
                                             size_t(0),
                                             [&](size_t age, size_t source) {
                                                 return std::max(age, activeStepCounts[source]);
                                             })
                           : stepCount;
        for (size_t source : group) {
            const size_t sourceAge = usesActiveAges ? activeStepCounts[source] : stepCount;
            if (sourceAge != mergedAge)
                throw std::invalid_argument(
                    "Adam state coalescing requires equal ages within each merged group");
        }
        double correctedFirst      = 0;
        double correctedRootSecond = 0;
        for (size_t source : group) {
            const size_t age = usesActiveAges ? activeStepCounts[source] : stepCount;
            if (age == 0) {
                if (m[source] != T(0) || v[source] != T(0))
                    throw std::runtime_error("Adam inactive state contains nonzero moments");
                continue;
            }
            const double correction1 = 1.0 - std::pow(double(beta1), double(age));
            const double correction2 = 1.0 - std::pow(double(beta2), double(age));
            const double first       = double(m[source]) / correction1;
            const double second      = double(v[source]) / correction2;
            if (!std::isfinite(first) || !std::isfinite(second) || second < 0)
                throw std::runtime_error("Adam state coalescing received invalid moments");
            correctedFirst += first;
            correctedRootSecond += std::sqrt(second);
        }

        const double mergedCorrection1 =
            mergedAge == 0 ? 0.0 : 1.0 - std::pow(double(beta1), double(mergedAge));
        const double mergedCorrection2 =
            mergedAge == 0 ? 0.0 : 1.0 - std::pow(double(beta2), double(mergedAge));
        const double mergedFirst  = correctedFirst * mergedCorrection1;
        const double mergedSecond = correctedRootSecond * correctedRootSecond * mergedCorrection2;
        if (!std::isfinite(mergedFirst) || !std::isfinite(mergedSecond))
            throw std::runtime_error("Adam state coalescing produced invalid moments");
        result.m[groupIndex]                = T(mergedFirst);
        result.v[groupIndex]                = T(mergedSecond);
        result.activeStepCounts[groupIndex] = usesActiveAges ? mergedAge : 0;
    }
    if (std::find(seen.begin(), seen.end(), uint8_t(0)) != seen.end())
        throw std::invalid_argument("Adam state coalescing partition is incomplete");
    return result;
}

template <typename T>
void AdamOptimizer<T>::swap(AdamOptimizer &other) noexcept
{
    using std::swap;
    m.swap(other.m);
    v.swap(other.v);
    nextParams.swap(other.nextParams);
    activeStepCounts.swap(other.activeStepCounts);
    swap(lr, other.lr);
    swap(weightDecay, other.weightDecay);
    swap(beta1, other.beta1);
    swap(beta2, other.beta2);
    swap(epsilon, other.epsilon);
    swap(stepCount, other.stepCount);
}

}  // namespace Tuning

template class Tuning::AdamOptimizer<float>;
template class Tuning::AdamOptimizer<double>;
