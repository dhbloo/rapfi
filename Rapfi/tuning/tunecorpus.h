/*
 *  Rapfi, a Gomoku/Renju playing engine supporting piskvork protocol.
 *  Copyright (C) 2022  Rapfi developers
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <limits>
#include <vector>

namespace Tuning {

using ParameterId = uint16_t;

struct TuneCoeff
{
    int16_t     coeff;
    ParameterId index;
};
static_assert(sizeof(TuneCoeff) == 4, "TuneCoeff must remain compact");

struct PolicyCandidate
{
    ParameterId indices[2];
};
static_assert(sizeof(PolicyCandidate) == 4, "PolicyCandidate must remain compact");

struct PolicyTargetTerm
{
    uint16_t candidate;  ///< Sample-local policy candidate index.
    uint16_t weight;     ///< Unnormalized quantized target mass.
};
static_assert(sizeof(PolicyTargetTerm) == 4, "PolicyTargetTerm must remain compact");

/// Compact CSR/SoA storage for prepared tuning samples.
class PreparedCorpus
{
public:
    static constexpr uint16_t NoPolicyTarget = std::numeric_limits<uint16_t>::max();

    PreparedCorpus();

    size_t size() const { return results_.size(); }
    bool   empty() const { return results_.empty(); }

    void clear();
    void reserveSamples(size_t count);
    void append(uint8_t                              boardSize,
                uint8_t                              resultTimesTwo,
                int16_t                              staticEval,
                const std::vector<TuneCoeff>        &evalTerms,
                const std::vector<PolicyCandidate>  &policyCandidates,
                const std::vector<PolicyTargetTerm> &policyTargets,
                uint16_t                             bestCandidate);
    void append(PreparedCorpus &&other);
    void appendRange(const PreparedCorpus &other, size_t begin, size_t count);
    size_t
    appendRangePeakCapacityBytes(const PreparedCorpus &other, size_t begin, size_t count) const;
    void reserveAppendRange(const PreparedCorpus &other, size_t begin, size_t count);

    static PreparedCorpus fromSections(std::vector<uint8_t>          boardSizes,
                                       std::vector<uint8_t>          results,
                                       std::vector<int16_t>          staticEvals,
                                       std::vector<uint16_t>         bestCandidates,
                                       std::vector<uint32_t>         evalOffsets,
                                       std::vector<TuneCoeff>        evalTerms,
                                       std::vector<uint32_t>         policyOffsets,
                                       std::vector<PolicyCandidate>  policyCandidates,
                                       std::vector<uint32_t>         policyTargetOffsets,
                                       std::vector<PolicyTargetTerm> policyTargets);

    const std::vector<uint8_t>         &boardSizes() const { return boardSizes_; }
    const std::vector<uint8_t>         &results() const { return results_; }
    const std::vector<int16_t>         &staticEvals() const { return staticEvals_; }
    const std::vector<uint16_t>        &bestCandidates() const { return bestCandidates_; }
    const std::vector<uint32_t>        &evalOffsets() const { return evalOffsets_; }
    const std::vector<TuneCoeff>       &evalTerms() const { return evalTerms_; }
    const std::vector<uint32_t>        &policyOffsets() const { return policyOffsets_; }
    const std::vector<PolicyCandidate> &policyCandidates() const { return policyCandidates_; }
    const std::vector<uint32_t>        &policyTargetOffsets() const { return policyTargetOffsets_; }
    const std::vector<PolicyTargetTerm> &policyTargets() const { return policyTargets_; }

    size_t capacityBytes() const;
    size_t storageBytes() const;

private:
    std::vector<uint8_t>          boardSizes_;
    std::vector<uint8_t>          results_;
    std::vector<int16_t>          staticEvals_;
    std::vector<uint16_t>         bestCandidates_;
    std::vector<uint32_t>         evalOffsets_;
    std::vector<TuneCoeff>        evalTerms_;
    std::vector<uint32_t>         policyOffsets_;
    std::vector<PolicyCandidate>  policyCandidates_;
    std::vector<uint32_t>         policyTargetOffsets_;
    std::vector<PolicyTargetTerm> policyTargets_;
};

}  // namespace Tuning
