/*
 *  Rapfi, a Gomoku/Renju playing engine supporting piskvork protocol.
 *  Copyright (C) 2022  Rapfi developers
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 */

#include "tunecorpus.h"

#include <algorithm>
#include <iterator>
#include <stdexcept>
#include <string>

namespace Tuning {
namespace {

    uint32_t checkedOffset(size_t current, size_t added, const char *section)
    {
        constexpr size_t MaxOffset = std::numeric_limits<uint32_t>::max();
        if (current > MaxOffset || added > MaxOffset - current)
            throw std::length_error(std::string("prepared corpus ") + section
                                    + " exceeds 32-bit shard offsets");
        return static_cast<uint32_t>(current + added);
    }

    template <typename T>
    size_t vectorCapacityBytes(const std::vector<T> &values)
    {
        return values.capacity() * sizeof(T);
    }

    size_t growthCapacity(size_t capacity, size_t required)
    {
        if (required <= capacity)
            return capacity;
        size_t geometric = capacity;
        if (geometric < std::numeric_limits<size_t>::max() - geometric / 2 - 1)
            geometric += geometric / 2 + 1;
        else
            geometric = std::numeric_limits<size_t>::max();
        return std::max(required, geometric);
    }

    size_t checkedSizeSum(size_t current, size_t added, const char *section)
    {
        if (added > std::numeric_limits<size_t>::max() - current)
            throw std::length_error(std::string("prepared corpus ") + section
                                    + " size overflows size_t");
        return current + added;
    }

}  // namespace

PreparedCorpus::PreparedCorpus() : evalOffsets_ {0}, policyOffsets_ {0}, policyTargetOffsets_ {0} {}

void PreparedCorpus::clear()
{
    boardSizes_.clear();
    results_.clear();
    staticEvals_.clear();
    bestCandidates_.clear();
    evalTerms_.clear();
    policyCandidates_.clear();
    policyTargets_.clear();
    evalOffsets_.assign(1, 0);
    policyOffsets_.assign(1, 0);
    policyTargetOffsets_.assign(1, 0);
}

void PreparedCorpus::reserveSamples(size_t count)
{
    boardSizes_.reserve(count);
    results_.reserve(count);
    staticEvals_.reserve(count);
    bestCandidates_.reserve(count);
    evalOffsets_.reserve(count + 1);
    policyOffsets_.reserve(count + 1);
    policyTargetOffsets_.reserve(count + 1);
}

void PreparedCorpus::append(uint8_t                              boardSize,
                            uint8_t                              resultTimesTwo,
                            int16_t                              staticEval,
                            const std::vector<TuneCoeff>        &evalTerms,
                            const std::vector<PolicyCandidate>  &policyCandidates,
                            const std::vector<PolicyTargetTerm> &policyTargets,
                            uint16_t                             bestCandidate)
{
    if (boardSize == 0)
        throw std::invalid_argument("prepared sample board size is out of range");
    if (resultTimesTwo > 2)
        throw std::invalid_argument("prepared sample result is out of range");
    if (bestCandidate != NoPolicyTarget && bestCandidate >= policyCandidates.size())
        throw std::invalid_argument("prepared sample policy target is out of range");
    if (!policyTargets.empty() && bestCandidate == NoPolicyTarget)
        throw std::invalid_argument("prepared sample soft policy target has no best candidate");
    uint16_t previousCandidate = 0;
    bool     firstTarget       = true;
    for (const PolicyTargetTerm &target : policyTargets) {
        if (target.candidate >= policyCandidates.size())
            throw std::invalid_argument("prepared sample soft policy target is out of range");
        if (target.weight == 0)
            throw std::invalid_argument("prepared sample soft policy target has zero weight");
        if (!firstTarget && target.candidate <= previousCandidate)
            throw std::invalid_argument("prepared sample soft policy targets are not ordered");
        previousCandidate = target.candidate;
        firstTarget       = false;
    }

    uint32_t evalEnd = checkedOffset(evalTerms_.size(), evalTerms.size(), "value terms");
    uint32_t policyEnd =
        checkedOffset(policyCandidates_.size(), policyCandidates.size(), "policy candidates");
    uint32_t policyTargetEnd =
        checkedOffset(policyTargets_.size(), policyTargets.size(), "policy targets");

    evalTerms_.insert(evalTerms_.end(), evalTerms.begin(), evalTerms.end());
    policyCandidates_.insert(policyCandidates_.end(),
                             policyCandidates.begin(),
                             policyCandidates.end());
    policyTargets_.insert(policyTargets_.end(), policyTargets.begin(), policyTargets.end());
    boardSizes_.push_back(boardSize);
    results_.push_back(resultTimesTwo);
    staticEvals_.push_back(staticEval);
    bestCandidates_.push_back(bestCandidate);
    evalOffsets_.push_back(evalEnd);
    policyOffsets_.push_back(policyEnd);
    policyTargetOffsets_.push_back(policyTargetEnd);
}

void PreparedCorpus::append(PreparedCorpus &&other)
{
    if (other.empty())
        return;

    checkedOffset(evalTerms_.size(), other.evalTerms_.size(), "value terms");
    checkedOffset(policyCandidates_.size(), other.policyCandidates_.size(), "policy candidates");
    checkedOffset(policyTargets_.size(), other.policyTargets_.size(), "policy targets");

    // Take over the first complete fragment without copying. Later inserts
    // deliberately rely on vector's geometric growth instead of reserving the
    // exact new total for every fragment, which would repeatedly copy the
    // complete accumulated corpus.
    if (empty()) {
        *this = std::move(other);
        return;
    }

    uint32_t evalBase         = static_cast<uint32_t>(evalTerms_.size());
    uint32_t policyBase       = static_cast<uint32_t>(policyCandidates_.size());
    uint32_t policyTargetBase = static_cast<uint32_t>(policyTargets_.size());

    evalTerms_.insert(evalTerms_.end(),
                      std::make_move_iterator(other.evalTerms_.begin()),
                      std::make_move_iterator(other.evalTerms_.end()));
    policyCandidates_.insert(policyCandidates_.end(),
                             std::make_move_iterator(other.policyCandidates_.begin()),
                             std::make_move_iterator(other.policyCandidates_.end()));
    policyTargets_.insert(policyTargets_.end(),
                          std::make_move_iterator(other.policyTargets_.begin()),
                          std::make_move_iterator(other.policyTargets_.end()));
    boardSizes_.insert(boardSizes_.end(), other.boardSizes_.begin(), other.boardSizes_.end());
    results_.insert(results_.end(), other.results_.begin(), other.results_.end());
    staticEvals_.insert(staticEvals_.end(), other.staticEvals_.begin(), other.staticEvals_.end());
    bestCandidates_.insert(bestCandidates_.end(),
                           other.bestCandidates_.begin(),
                           other.bestCandidates_.end());

    for (size_t i = 1; i < other.evalOffsets_.size(); i++)
        evalOffsets_.push_back(evalBase + other.evalOffsets_[i]);
    for (size_t i = 1; i < other.policyOffsets_.size(); i++)
        policyOffsets_.push_back(policyBase + other.policyOffsets_[i]);
    for (size_t i = 1; i < other.policyTargetOffsets_.size(); i++)
        policyTargetOffsets_.push_back(policyTargetBase + other.policyTargetOffsets_[i]);
}

void PreparedCorpus::appendRange(const PreparedCorpus &other, size_t begin, size_t count)
{
    if (begin > other.size() || count > other.size() - begin)
        throw std::out_of_range("prepared corpus sample range is out of bounds");
    if (count == 0)
        return;

    uint32_t sourceEvalBegin         = other.evalOffsets_[begin];
    uint32_t sourceEvalEnd           = other.evalOffsets_[begin + count];
    uint32_t sourcePolicyBegin       = other.policyOffsets_[begin];
    uint32_t sourcePolicyEnd         = other.policyOffsets_[begin + count];
    uint32_t sourcePolicyTargetBegin = other.policyTargetOffsets_[begin];
    uint32_t sourcePolicyTargetEnd   = other.policyTargetOffsets_[begin + count];
    checkedOffset(evalTerms_.size(), sourceEvalEnd - sourceEvalBegin, "value terms");
    checkedOffset(policyCandidates_.size(),
                  sourcePolicyEnd - sourcePolicyBegin,
                  "policy candidates");
    checkedOffset(policyTargets_.size(),
                  sourcePolicyTargetEnd - sourcePolicyTargetBegin,
                  "policy targets");
    uint32_t evalBase         = static_cast<uint32_t>(evalTerms_.size());
    uint32_t policyBase       = static_cast<uint32_t>(policyCandidates_.size());
    uint32_t policyTargetBase = static_cast<uint32_t>(policyTargets_.size());

    boardSizes_.insert(boardSizes_.end(),
                       other.boardSizes_.begin() + begin,
                       other.boardSizes_.begin() + begin + count);
    results_.insert(results_.end(),
                    other.results_.begin() + begin,
                    other.results_.begin() + begin + count);
    staticEvals_.insert(staticEvals_.end(),
                        other.staticEvals_.begin() + begin,
                        other.staticEvals_.begin() + begin + count);
    bestCandidates_.insert(bestCandidates_.end(),
                           other.bestCandidates_.begin() + begin,
                           other.bestCandidates_.begin() + begin + count);
    evalTerms_.insert(evalTerms_.end(),
                      other.evalTerms_.begin() + sourceEvalBegin,
                      other.evalTerms_.begin() + sourceEvalEnd);
    policyCandidates_.insert(policyCandidates_.end(),
                             other.policyCandidates_.begin() + sourcePolicyBegin,
                             other.policyCandidates_.begin() + sourcePolicyEnd);
    policyTargets_.insert(policyTargets_.end(),
                          other.policyTargets_.begin() + sourcePolicyTargetBegin,
                          other.policyTargets_.begin() + sourcePolicyTargetEnd);

    for (size_t i = 1; i <= count; i++)
        evalOffsets_.push_back(evalBase + other.evalOffsets_[begin + i] - sourceEvalBegin);
    for (size_t i = 1; i <= count; i++)
        policyOffsets_.push_back(policyBase + other.policyOffsets_[begin + i] - sourcePolicyBegin);
    for (size_t i = 1; i <= count; i++)
        policyTargetOffsets_.push_back(policyTargetBase + other.policyTargetOffsets_[begin + i]
                                       - sourcePolicyTargetBegin);
}

size_t PreparedCorpus::appendRangePeakCapacityBytes(const PreparedCorpus &other,
                                                    size_t                begin,
                                                    size_t                count) const
{
    if (begin > other.size() || count > other.size() - begin)
        throw std::out_of_range("prepared corpus sample range is out of bounds");
    uint32_t sourceEvalBegin         = other.evalOffsets_[begin];
    uint32_t sourceEvalEnd           = other.evalOffsets_[begin + count];
    uint32_t sourcePolicyBegin       = other.policyOffsets_[begin];
    uint32_t sourcePolicyEnd         = other.policyOffsets_[begin + count];
    uint32_t sourcePolicyTargetBegin = other.policyTargetOffsets_[begin];
    uint32_t sourcePolicyTargetEnd   = other.policyTargetOffsets_[begin + count];

    size_t total = capacityBytes();
    size_t peak  = total;
    auto   plan  = [&](size_t capacity, size_t required, size_t elementSize) {
        size_t nextCapacity = growthCapacity(capacity, required);
        if (nextCapacity == capacity)
            return;
        if (nextCapacity > std::numeric_limits<size_t>::max() / elementSize
            || total > std::numeric_limits<size_t>::max() - nextCapacity * elementSize)
            throw std::length_error("prepared corpus capacity byte count overflows size_t");
        peak = std::max(peak, total + nextCapacity * elementSize);
        total -= capacity * elementSize;
        total += nextCapacity * elementSize;
    };

    size_t samplesRequired = checkedSizeSum(size(), count, "sample");
    size_t offsetsRequired = checkedSizeSum(samplesRequired, 1, "offset");
    plan(boardSizes_.capacity(), samplesRequired, sizeof(boardSizes_[0]));
    plan(results_.capacity(), samplesRequired, sizeof(results_[0]));
    plan(staticEvals_.capacity(), samplesRequired, sizeof(staticEvals_[0]));
    plan(bestCandidates_.capacity(), samplesRequired, sizeof(bestCandidates_[0]));
    plan(evalOffsets_.capacity(), offsetsRequired, sizeof(evalOffsets_[0]));
    plan(evalTerms_.capacity(),
         checkedSizeSum(evalTerms_.size(), sourceEvalEnd - sourceEvalBegin, "value term"),
         sizeof(evalTerms_[0]));
    plan(policyOffsets_.capacity(), offsetsRequired, sizeof(policyOffsets_[0]));
    plan(policyCandidates_.capacity(),
         checkedSizeSum(policyCandidates_.size(),
                        sourcePolicyEnd - sourcePolicyBegin,
                        "policy candidate"),
         sizeof(policyCandidates_[0]));
    plan(policyTargetOffsets_.capacity(), offsetsRequired, sizeof(policyTargetOffsets_[0]));
    plan(policyTargets_.capacity(),
         checkedSizeSum(policyTargets_.size(),
                        sourcePolicyTargetEnd - sourcePolicyTargetBegin,
                        "policy target"),
         sizeof(policyTargets_[0]));
    return peak;
}

void PreparedCorpus::reserveAppendRange(const PreparedCorpus &other, size_t begin, size_t count)
{
    if (begin > other.size() || count > other.size() - begin)
        throw std::out_of_range("prepared corpus sample range is out of bounds");
    uint32_t sourceEvalBegin         = other.evalOffsets_[begin];
    uint32_t sourceEvalEnd           = other.evalOffsets_[begin + count];
    uint32_t sourcePolicyBegin       = other.policyOffsets_[begin];
    uint32_t sourcePolicyEnd         = other.policyOffsets_[begin + count];
    uint32_t sourcePolicyTargetBegin = other.policyTargetOffsets_[begin];
    uint32_t sourcePolicyTargetEnd   = other.policyTargetOffsets_[begin + count];
    size_t   samplesRequired         = checkedSizeSum(size(), count, "sample");
    size_t   offsetsRequired         = checkedSizeSum(samplesRequired, 1, "offset");

    boardSizes_.reserve(growthCapacity(boardSizes_.capacity(), samplesRequired));
    results_.reserve(growthCapacity(results_.capacity(), samplesRequired));
    staticEvals_.reserve(growthCapacity(staticEvals_.capacity(), samplesRequired));
    bestCandidates_.reserve(growthCapacity(bestCandidates_.capacity(), samplesRequired));
    evalOffsets_.reserve(growthCapacity(evalOffsets_.capacity(), offsetsRequired));
    evalTerms_.reserve(growthCapacity(
        evalTerms_.capacity(),
        checkedSizeSum(evalTerms_.size(), sourceEvalEnd - sourceEvalBegin, "value term")));
    policyOffsets_.reserve(growthCapacity(policyOffsets_.capacity(), offsetsRequired));
    policyCandidates_.reserve(growthCapacity(policyCandidates_.capacity(),
                                             checkedSizeSum(policyCandidates_.size(),
                                                            sourcePolicyEnd - sourcePolicyBegin,
                                                            "policy candidate")));
    policyTargetOffsets_.reserve(growthCapacity(policyTargetOffsets_.capacity(), offsetsRequired));
    policyTargets_.reserve(
        growthCapacity(policyTargets_.capacity(),
                       checkedSizeSum(policyTargets_.size(),
                                      sourcePolicyTargetEnd - sourcePolicyTargetBegin,
                                      "policy target")));
}

PreparedCorpus PreparedCorpus::fromSections(std::vector<uint8_t>          boardSizes,
                                            std::vector<uint8_t>          results,
                                            std::vector<int16_t>          staticEvals,
                                            std::vector<uint16_t>         bestCandidates,
                                            std::vector<uint32_t>         evalOffsets,
                                            std::vector<TuneCoeff>        evalTerms,
                                            std::vector<uint32_t>         policyOffsets,
                                            std::vector<PolicyCandidate>  policyCandidates,
                                            std::vector<uint32_t>         policyTargetOffsets,
                                            std::vector<PolicyTargetTerm> policyTargets)
{
    size_t sampleCount = results.size();
    if (boardSizes.size() != sampleCount || staticEvals.size() != sampleCount
        || bestCandidates.size() != sampleCount || evalOffsets.size() != sampleCount + 1
        || policyOffsets.size() != sampleCount + 1 || policyTargetOffsets.size() != sampleCount + 1)
        throw std::invalid_argument("prepared corpus section counts are inconsistent");
    if (evalOffsets.empty() || evalOffsets.front() != 0 || evalOffsets.back() != evalTerms.size()
        || policyOffsets.empty() || policyOffsets.front() != 0
        || policyOffsets.back() != policyCandidates.size() || policyTargetOffsets.empty()
        || policyTargetOffsets.front() != 0 || policyTargetOffsets.back() != policyTargets.size())
        throw std::invalid_argument("prepared corpus terminal offsets are invalid");

    for (size_t i = 0; i < sampleCount; i++) {
        if (boardSizes[i] == 0)
            throw std::invalid_argument("prepared corpus board size is out of range");
        if (results[i] > 2)
            throw std::invalid_argument("prepared corpus result is out of range");
        if (evalOffsets[i] > evalOffsets[i + 1] || policyOffsets[i] > policyOffsets[i + 1]
            || policyTargetOffsets[i] > policyTargetOffsets[i + 1])
            throw std::invalid_argument("prepared corpus offsets are not monotonic");
        uint32_t policyCount = policyOffsets[i + 1] - policyOffsets[i];
        if (bestCandidates[i] != NoPolicyTarget && bestCandidates[i] >= policyCount)
            throw std::invalid_argument("prepared corpus policy target is out of range");
        if (policyTargetOffsets[i] != policyTargetOffsets[i + 1]
            && bestCandidates[i] == NoPolicyTarget)
            throw std::invalid_argument("prepared corpus soft policy target has no best candidate");
        uint16_t previousCandidate = 0;
        bool     firstTarget       = true;
        for (uint32_t targetIndex = policyTargetOffsets[i];
             targetIndex < policyTargetOffsets[i + 1];
             targetIndex++) {
            const PolicyTargetTerm &target = policyTargets[targetIndex];
            if (target.candidate >= policyCount)
                throw std::invalid_argument("prepared corpus soft policy target is out of range");
            if (target.weight == 0)
                throw std::invalid_argument("prepared corpus soft policy target has zero weight");
            if (!firstTarget && target.candidate <= previousCandidate)
                throw std::invalid_argument("prepared corpus soft policy targets are not ordered");
            previousCandidate = target.candidate;
            firstTarget       = false;
        }
    }

    PreparedCorpus corpus;
    corpus.boardSizes_          = std::move(boardSizes);
    corpus.results_             = std::move(results);
    corpus.staticEvals_         = std::move(staticEvals);
    corpus.bestCandidates_      = std::move(bestCandidates);
    corpus.evalOffsets_         = std::move(evalOffsets);
    corpus.evalTerms_           = std::move(evalTerms);
    corpus.policyOffsets_       = std::move(policyOffsets);
    corpus.policyCandidates_    = std::move(policyCandidates);
    corpus.policyTargetOffsets_ = std::move(policyTargetOffsets);
    corpus.policyTargets_       = std::move(policyTargets);
    return corpus;
}

size_t PreparedCorpus::capacityBytes() const
{
    return vectorCapacityBytes(boardSizes_) + vectorCapacityBytes(results_)
           + vectorCapacityBytes(staticEvals_) + vectorCapacityBytes(bestCandidates_)
           + vectorCapacityBytes(evalOffsets_) + vectorCapacityBytes(evalTerms_)
           + vectorCapacityBytes(policyOffsets_) + vectorCapacityBytes(policyCandidates_)
           + vectorCapacityBytes(policyTargetOffsets_) + vectorCapacityBytes(policyTargets_);
}

size_t PreparedCorpus::storageBytes() const
{
    return boardSizes_.size() * sizeof(boardSizes_[0]) + results_.size() * sizeof(results_[0])
           + staticEvals_.size() * sizeof(staticEvals_[0])
           + bestCandidates_.size() * sizeof(bestCandidates_[0])
           + evalOffsets_.size() * sizeof(evalOffsets_[0])
           + evalTerms_.size() * sizeof(evalTerms_[0])
           + policyOffsets_.size() * sizeof(policyOffsets_[0])
           + policyCandidates_.size() * sizeof(policyCandidates_[0])
           + policyTargetOffsets_.size() * sizeof(policyTargetOffsets_[0])
           + policyTargets_.size() * sizeof(policyTargets_[0]);
}

}  // namespace Tuning
