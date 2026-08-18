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

#include "tunerobjective.h"

#include "../core/math.h"
#include "../eval/evaluator.h"
#include "tunerdetail.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace Tuning::detail {

namespace {

    Float policyScore(const PolicyCandidate        &candidate,
                      const std::vector<TuneParam> &params,
                      const MoveScoreLossSettings &)
    {
        return params[candidate.indices[0]] + params[candidate.indices[1]];
    }

    void accumulatePolicyGradient(const PolicyCandidate     &candidate,
                                  std::vector<TuneGradient> &grads,
                                  Float                      gradient)
    {
        grads[candidate.indices[0]] += gradient;
        grads[candidate.indices[1]] += gradient;
    }

}  // namespace

MoveScoreLossSettings makeMoveScoreLossSettings(const Tuning::TuningConfig &config)
{
    Float logitFactor = config.trainingSemantics == TrainingSemantics::Bootstrap
                            ? Float(1)
                            : Float(config.moveScoreScale)
                                  / Float(Evaluation::PolicyBuffer::ScoreScale);
    return {Float(config.moveScoreLossGamma), logitFactor};
}

Float computeMoveScoreLoss(const PreparedCorpus         &corpus,
                           size_t                        sample,
                           const std::vector<TuneParam> &params,
                           const MoveScoreLossSettings  &settings)
{
    uint32_t begin = corpus.policyOffsets()[sample];
    uint32_t end   = corpus.policyOffsets()[sample + 1];
    uint16_t best  = corpus.bestCandidates()[sample];
    if (begin == end || best == PreparedCorpus::NoPolicyTarget)
        return 0;

    const auto &candidates  = corpus.policyCandidates();
    uint32_t    targetBegin = corpus.policyTargetOffsets()[sample];
    uint32_t    targetEnd   = corpus.policyTargetOffsets()[sample + 1];
    auto        score       = [&](const PolicyCandidate &candidate) {
        return policyScore(candidate, params, settings);
    };
    Float maxScore = std::numeric_limits<Float>::lowest();
    for (uint32_t i = begin; i < end; i++)
        maxScore = std::max(maxScore, score(candidates[i]) * settings.logitFactor);

    Float sumExp = 0;
    for (uint32_t i = begin; i < end; i++)
        sumExp += std::exp(score(candidates[i]) * settings.logitFactor - maxScore);

    if (settings.gamma == 0) {
        if (targetBegin == targetEnd)
            return maxScore + std::log(sumExp)
                   - score(candidates[begin + best]) * settings.logitFactor;

        Float targetMass  = 0;
        Float targetScore = 0;
        for (uint32_t i = targetBegin; i < targetEnd; i++) {
            const PolicyTargetTerm &target = corpus.policyTargets()[i];
            targetMass += target.weight;
            targetScore +=
                target.weight * score(candidates[begin + target.candidate]) * settings.logitFactor;
        }
        return maxScore + std::log(sumExp) - targetScore / targetMass;
    }

    Float scoreClass  = score(candidates[begin + best]) * settings.logitFactor - maxScore;
    Float xClass      = std::exp(scoreClass) / sumExp;
    Float focalWeight = std::pow(Float(1) - xClass, settings.gamma);
    return focalWeight * (-scoreClass + std::log(sumExp));
}

void computeMoveScoreGradient(const PreparedCorpus         &corpus,
                              size_t                        sample,
                              std::vector<TuneGradient>    &grads,
                              const std::vector<TuneParam> &params,
                              const MoveScoreLossSettings  &settings,
                              Float                         sampleWeight)
{
    uint32_t begin = corpus.policyOffsets()[sample];
    uint32_t end   = corpus.policyOffsets()[sample + 1];
    uint16_t best  = corpus.bestCandidates()[sample];
    if (begin == end || best == PreparedCorpus::NoPolicyTarget)
        return;

    const auto &candidates  = corpus.policyCandidates();
    uint32_t    targetBegin = corpus.policyTargetOffsets()[sample];
    uint32_t    targetEnd   = corpus.policyTargetOffsets()[sample + 1];
    auto        score       = [&](const PolicyCandidate &candidate) {
        return policyScore(candidate, params, settings);
    };
    auto addGradient = [&](const PolicyCandidate &candidate, Float gradient) {
        accumulatePolicyGradient(candidate, grads, gradient * sampleWeight);
    };
    Float maxScore = std::numeric_limits<Float>::lowest();
    for (uint32_t i = begin; i < end; i++)
        maxScore = std::max(maxScore, score(candidates[i]) * settings.logitFactor);

    Float sumExp = 0;
    for (uint32_t i = begin; i < end; i++)
        sumExp += std::exp(score(candidates[i]) * settings.logitFactor - maxScore);

    Float invSumExp = Float(1) / sumExp;
    if (settings.gamma == 0) {
        Float gradientScale = CoeffScale * settings.logitFactor;
        Float targetMass    = 0;
        for (uint32_t i = targetBegin; i < targetEnd; i++)
            targetMass += corpus.policyTargets()[i].weight;

        if (targetBegin == targetEnd) {
            for (uint32_t i = begin; i < end; i++) {
                Float probability =
                    std::exp(score(candidates[i]) * settings.logitFactor - maxScore) * invSumExp;
                Float dCEdScore = i == begin + best ? probability - 1 : probability;
                addGradient(candidates[i], gradientScale * dCEdScore);
            }
        }
        else {
            for (uint32_t i = begin; i < end; i++) {
                Float probability =
                    std::exp(score(candidates[i]) * settings.logitFactor - maxScore) * invSumExp;
                addGradient(candidates[i], gradientScale * probability);
            }
            Float invTargetMass = Float(1) / targetMass;
            for (uint32_t i = targetBegin; i < targetEnd; i++) {
                const PolicyTargetTerm &term   = corpus.policyTargets()[i];
                const PolicyCandidate  &target = candidates[begin + term.candidate];
                Float                   weight = term.weight * invTargetMass;
                addGradient(target, -gradientScale * weight);
            }
        }
        return;
    }

    Float bestExp = std::exp(score(candidates[begin + best]) * settings.logitFactor - maxScore);
    Float Pt      = bestExp * invSumExp;
    Float logPt =
        score(candidates[begin + best]) * settings.logitFactor - maxScore - std::log(sumExp);
    Float PtClamped        = std::clamp(Pt, Float(1e-6), Float(1 - 1e-6));
    Float PtLogPtDivPtSub1 = PtClamped / (PtClamped - 1) * logPt;
    Float dFLdCE = std::pow(1 - Pt, settings.gamma) * (settings.gamma * PtLogPtDivPtSub1 + 1);

    for (uint32_t i = begin; i < end; i++) {
        Float probability =
            std::exp(score(candidates[i]) * settings.logitFactor - maxScore) * invSumExp;
        Float dCEdScore = i == begin + best ? Pt - 1 : probability;
        Float dFLdScore = dFLdCE * dCEdScore;
        Float gradient  = CoeffScale * settings.logitFactor * dFLdScore;
        addGradient(candidates[i], gradient);
    }
}

}  // namespace Tuning::detail
