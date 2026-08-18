/*
 *  Rapfi, a Gomoku/Renju playing engine supporting piskvork protocol.
 *  Copyright (C) 2022  Rapfi developers
 *
 *  This program is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 */

#include "policytrainer.h"

#include "../config.h"
#include "../eval/classicalpolicy.h"
#include "../eval/scoretables.h"
#include "../game/board.h"
#include "optimizer.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <limits>
#include <stdexcept>

namespace Tuning {
namespace {

    constexpr size_t   PatternCount     = Evaluation::PolicyStoredPatternCount;
    constexpr size_t   PatternPairs     = PatternCount * PatternCount;
    constexpr size_t   ContextCount     = Evaluation::PolicyStoredContextCount;
    constexpr size_t   ParameterCount   = ContextCount * PatternPairs;
    constexpr uint16_t MissingParameter = std::numeric_limits<uint16_t>::max();

    struct Candidate
    {
        uint16_t    parameter;
        Color       self;
        PatternCode pcodeBlack;
        PatternCode pcodeWhite;
        int32_t     positionScore;
        int32_t     runtimeOffset;
    };

    struct Event
    {
        PolicyTraceSplit       split;
        Rule                   rule      = FREESTYLE;
        int                    table     = FREESTYLE;
        uint8_t                boardSize = 0;
        std::vector<Candidate> candidates;
        uint16_t               cutoff = MissingParameter;
        std::vector<uint16_t>  earlierSearched;
    };

    struct Corpus
    {
        std::vector<Event> events;
    };

    bool searchedNonCutoff(PolicyTraceDisposition disposition)
    {
        return disposition == PolicyTraceDisposition::SearchedFailLow
               || disposition == PolicyTraceDisposition::SearchedAlphaImprovement;
    }

    uint16_t parameterIndex(size_t tableSlot, uint8_t context, Pattern4 self, Pattern4 opponent)
    {
        if (context >= ContextCount)
            return MissingParameter;
        if (self == FORBID || opponent == FORBID)
            return MissingParameter;
        const size_t pair = Evaluation::policyPatternStorageIndex(self) * PatternCount
                            + Evaluation::policyPatternStorageIndex(opponent);
        const size_t index = tableSlot * ParameterCount + size_t(context) * PatternPairs + pair;
        if (index >= size_t(MissingParameter))
            throw std::logic_error("policy parameter index exceeds compact surface");
        return static_cast<uint16_t>(index);
    }

    int
    currentBasePositionScore(Rule rule, Color self, PatternCode pcodeBlack, PatternCode pcodeWhite)
    {
        return Evaluation::classicalPolicyBaseScore(rule, self, pcodeBlack, pcodeWhite);
    }

    Corpus readCorpus(const std::vector<std::filesystem::path> &paths,
                      const std::vector<int>                   &destinationTables)
    {
        if (destinationTables.empty())
            throw std::invalid_argument("policy training requires a destination table");
        for (size_t i = 0; i < destinationTables.size(); i++) {
            if (destinationTables[i] < 0 || destinationTables[i] >= RULE_NB + 1)
                throw std::invalid_argument("policy destination table is invalid");
            if (std::find(destinationTables.begin(),
                          destinationTables.begin() + i,
                          destinationTables[i])
                != destinationTables.begin() + i)
                throw std::invalid_argument("policy destination tables contain a duplicate");
        }
        Corpus corpus;
        for (const std::filesystem::path &path : paths) {
            readPolicyTrace(path, [&](const PolicyTraceEvent &source) {
                const int  table = Evaluation::tableIndex(source.rule, source.sideToMove);
                const auto selected =
                    std::find(destinationTables.begin(), destinationTables.end(), table);
                if (selected == destinationTables.end())
                    return;
                const size_t tableSlot = size_t(selected - destinationTables.begin());
                Board        board(source.boardSize);
                board.newGame(source.rule);
                for (Pos move : source.history) {
                    if (move != Pos::PASS && !board.isLegal(move))
                        throw std::runtime_error("policy trace history contains an illegal move");
                    board.move(source.rule, move);
                }
                if (board.zobristKey() != source.positionKey
                    || board.sideToMove() != source.sideToMove)
                    throw std::runtime_error("policy trace position does not match its history");
                const uint8_t policyContext =
                    static_cast<uint8_t>(Evaluation::classifyPolicyContext(board));
                std::vector<const PolicyTraceCandidate *> eligible;
                for (const PolicyTraceCandidate &candidate : source.candidates)
                    if (candidate.policyOrdinal != std::numeric_limits<uint16_t>::max())
                        eligible.push_back(&candidate);
                std::sort(eligible.begin(), eligible.end(), [](const auto *lhs, const auto *rhs) {
                    return lhs->policyOrdinal < rhs->policyOrdinal;
                });
                if (eligible.size() < 2)
                    return;
                if (eligible.size() >= MissingParameter)
                    throw std::runtime_error("policy trace has too many eligible candidates");

                Event event;
                event.split     = source.split;
                event.rule      = source.rule;
                event.table     = table;
                event.boardSize = source.boardSize;
                event.candidates.reserve(eligible.size());
                for (size_t ordinal = 0; ordinal < eligible.size(); ordinal++) {
                    const PolicyTraceCandidate &candidate = *eligible[ordinal];
                    if (!board.isEmpty(candidate.move))
                        throw std::runtime_error("policy trace candidate is not empty");
                    const int64_t runtimeOffset =
                        int64_t(candidate.mainHistory) + int64_t(candidate.counterMove);
                    if (runtimeOffset < std::numeric_limits<int32_t>::min()
                        || runtimeOffset > std::numeric_limits<int32_t>::max())
                        throw std::runtime_error("policy trace runtime offset overflows int32");
                    const auto [pcodeBlack, pcodeWhite] = board.pcodePair(candidate.move);
                    event.candidates.push_back(
                        {parameterIndex(tableSlot,
                                        policyContext,
                                        board.pattern4(candidate.move, board.sideToMove()),
                                        board.pattern4(candidate.move, ~board.sideToMove())),
                         board.sideToMove(),
                         pcodeBlack,
                         pcodeWhite,
                         currentBasePositionScore(source.rule,
                                                  board.sideToMove(),
                                                  pcodeBlack,
                                                  pcodeWhite),
                         static_cast<int32_t>(runtimeOffset)});
                    if (candidate.disposition == PolicyTraceDisposition::SearchedCutoff)
                        event.cutoff = static_cast<uint16_t>(ordinal);
                    else if (searchedNonCutoff(candidate.disposition)
                             && event.cutoff == MissingParameter)
                        event.earlierSearched.push_back(static_cast<uint16_t>(ordinal));
                }

                if (source.nodeType != PolicyTraceNodeType::NonPv
                    || int64_t(source.beta) != int64_t(source.alpha) + 1
                    || event.cutoff == MissingParameter)
                    event.earlierSearched.clear();
                corpus.events.push_back(std::move(event));
            });
        }
        return corpus;
    }

    struct DifferentiableScore
    {
        double value;
        double derivative;
    };

    DifferentiableScore candidatePositionScore(const Candidate           &candidate,
                                               const std::vector<double> &parameters,
                                               int                        residualLimit)
    {
        const double residual = candidate.parameter == MissingParameter
                                    ? 0
                                    : std::clamp(parameters[candidate.parameter],
                                                 -double(residualLimit),
                                                 double(residualLimit));
        const double raw      = double(candidate.positionScore) + residual;
        const double minimum  = double(std::numeric_limits<Score>::min());
        const double maximum  = double(std::numeric_limits<Score>::max());
        return {std::clamp(raw, minimum, maximum),
                candidate.parameter != MissingParameter && raw > minimum && raw < maximum ? 1.0
                                                                                          : 0.0};
    }

    DifferentiableScore candidateRuntimeScore(const Candidate           &candidate,
                                              const std::vector<double> &parameters,
                                              int                        residualLimit)
    {
        const DifferentiableScore position =
            candidatePositionScore(candidate, parameters, residualLimit);
        const double raw     = position.value + candidate.runtimeOffset;
        const double minimum = double(std::numeric_limits<Score>::min());
        const double maximum = double(std::numeric_limits<Score>::max());
        return {std::clamp(raw, minimum, maximum),
                raw > minimum && raw < maximum ? position.derivative : 0.0};
    }

    bool hasTrainablePair(const Event &event)
    {
        if (event.cutoff == MissingParameter || event.cutoff >= event.candidates.size())
            return false;
        if (event.candidates[event.cutoff].parameter != MissingParameter)
            return true;
        return std::any_of(event.earlierSearched.begin(),
                           event.earlierSearched.end(),
                           [&](uint16_t earlier) {
                               return earlier < event.candidates.size()
                                      && event.candidates[earlier].parameter != MissingParameter;
                           });
    }

    void accumulateGradient(const Event               &event,
                            std::vector<double>       &gradient,
                            const std::vector<double> &parameters,
                            double                     scoreScale,
                            int                        residualLimit,
                            double                     eventWeight)
    {
        const Candidate &cutoff = event.candidates[event.cutoff];
        for (uint16_t earlier : event.earlierSearched) {
            const Candidate          &negative = event.candidates[earlier];
            const DifferentiableScore cutoffScore =
                candidateRuntimeScore(cutoff, parameters, residualLimit);
            const DifferentiableScore negativeScore =
                candidateRuntimeScore(negative, parameters, residualLimit);
            const double margin = (cutoffScore.value - negativeScore.value) / scoreScale;
            const double negativeProbability = margin >= 0
                                                   ? std::exp(-margin) / (1.0 + std::exp(-margin))
                                                   : 1.0 / (1.0 + std::exp(margin));
            const double derivative          = -eventWeight * negativeProbability / scoreScale
                                      / double(event.earlierSearched.size());
            if (cutoffScore.derivative != 0)
                gradient[cutoff.parameter] += derivative * cutoffScore.derivative;
            if (negativeScore.derivative != 0)
                gradient[negative.parameter] -= derivative * negativeScore.derivative;
        }
    }

    std::vector<double> initialParameters(const std::vector<int> &destinationTables)
    {
        std::vector<double> parameters(ParameterCount * destinationTables.size());
        for (size_t tableSlot = 0; tableSlot < destinationTables.size(); tableSlot++)
            for (size_t context = 0; context < ContextCount; context++)
                for (size_t self = 0; self < PatternCount; self++)
                    for (size_t opponent = 0; opponent < PatternCount; opponent++) {
                        const size_t index = tableSlot * ParameterCount
                                             + (context * PatternCount + self) * PatternCount
                                             + opponent;
                        parameters[index] = Evaluation::POLICY_CROSS
                            [destinationTables[tableSlot]][context]
                            [Evaluation::policyPatternFromStorageIndex(self)]
                            [Evaluation::policyPatternFromStorageIndex(opponent)];
                    }
        return parameters;
    }

    std::vector<Score> roundedParameters(const std::vector<double> &parameters, int limit)
    {
        std::vector<Score> rounded(parameters.size());
        for (size_t i = 0; i < parameters.size(); i++) {
            const double clipped = std::clamp(parameters[i], -double(limit), double(limit));
            rounded[i]           = static_cast<Score>(clipped >= 0 ? std::floor(clipped + 0.5)
                                                         : std::ceil(clipped - 0.5));
        }
        return rounded;
    }

    void installParameters(const std::vector<Score> &parameters,
                           const std::vector<int>   &destinationTables)
    {
        for (size_t tableSlot = 0; tableSlot < destinationTables.size(); tableSlot++)
            for (size_t context = 0; context < ContextCount; context++)
                for (size_t self = 0; self < PatternCount; self++)
                    for (size_t opponent = 0; opponent < PatternCount; opponent++) {
                        const size_t index = tableSlot * ParameterCount
                                             + (context * PatternCount + self) * PatternCount
                                             + opponent;
                        Evaluation::POLICY_CROSS[destinationTables[tableSlot]][context]
                                                [Evaluation::policyPatternFromStorageIndex(self)]
                                                [Evaluation::policyPatternFromStorageIndex(
                                                    opponent)] = parameters[index];
                    }
        Evaluation::activatePolicyCross();
    }

}  // namespace

struct PolicyTraceTrainer::Impl
{
    Impl(const std::vector<std::filesystem::path> &tracePaths,
         const PolicyTrainingConfig               &trainingConfig)
        : config(trainingConfig)
        , corpus(readCorpus(tracePaths, config.destinationTables))
        , parameters(initialParameters(config.destinationTables))
        , initial(parameters)
        , optimizer(parameters.size(), config.learningRate, 0.0, 0.9, 0.999, 1e-8)
    {
        std::array<size_t, MAX_BOARD_SIZE + 1> boardEventCounts {};
        std::array<size_t, RULE_NB + 1>        tableEventCounts {};
        size_t                                 trainableEvents = 0;
        for (const Event &event : corpus.events) {
            if (event.split != PolicyTraceSplit::Train || event.earlierSearched.empty()
                || !hasTrainablePair(event))
                continue;
            if (event.boardSize == 0 || event.boardSize > MAX_BOARD_SIZE)
                throw std::logic_error("policy event has an invalid board size");
            boardEventCounts[event.boardSize]++;
            tableEventCounts[event.table]++;
            trainableEvents++;
        }
        if (trainableEvents == 0)
            throw std::runtime_error("policy corpus has no trainable natural-cutoff events");
        tableEventWeights.fill(0.0);
        for (int table : config.destinationTables) {
            if (tableEventCounts[table] == 0)
                throw std::runtime_error(
                    "policy corpus has no trainable natural-cutoff events for a selected table");
            tableEventWeights[table] =
                double(trainableEvents)
                / double(config.destinationTables.size() * tableEventCounts[table]);
        }

        boardEventWeights.fill(1.0);
        if (config.equalBoardWeighting) {
            const size_t activeBoards = std::count_if(boardEventCounts.begin(),
                                                      boardEventCounts.end(),
                                                      [](size_t count) { return count != 0; });
            for (size_t boardSize = 1; boardSize < boardEventCounts.size(); boardSize++) {
                const size_t count = boardEventCounts[boardSize];
                if (count != 0)
                    boardEventWeights[boardSize] =
                        double(trainableEvents) / double(activeBoards * count);
            }
        }
    }

    void step()
    {
        std::vector<double> gradient(parameters.size());
        double              mass = 0;
        for (const Event &event : corpus.events) {
            if (event.split != PolicyTraceSplit::Train || event.earlierSearched.empty()
                || !hasTrainablePair(event))
                continue;
            const double eventWeight =
                boardEventWeights[event.boardSize] * tableEventWeights[event.table];
            accumulateGradient(event,
                               gradient,
                               parameters,
                               config.scoreScale,
                               config.residualLimit,
                               eventWeight);
            mass += eventWeight;
        }
        if (mass <= 0)
            throw std::logic_error("policy corpus has non-positive optimization mass");
        for (size_t i = 0; i < gradient.size(); i++)
            gradient[i] = gradient[i] / mass
                          + config.anchor * (parameters[i] - initial[i]) / parameters.size();
        optimizer.step(parameters, gradient);
        for (double &parameter : parameters)
            parameter =
                std::clamp(parameter, -double(config.residualLimit), double(config.residualLimit));
    }

    void refreshBaseScores()
    {
        for (Event &event : corpus.events)
            for (Candidate &candidate : event.candidates)
                candidate.positionScore = currentBasePositionScore(event.rule,
                                                                   candidate.self,
                                                                   candidate.pcodeBlack,
                                                                   candidate.pcodeWhite);
    }

    PolicyTrainingResult saveModel(const std::filesystem::path &outputModelPath)
    {
        std::filesystem::path temporaryPath = outputModelPath;
        temporaryPath += ".tmp";
        if (std::filesystem::exists(outputModelPath) || std::filesystem::exists(temporaryPath))
            throw std::runtime_error("policy model output already exists");

        PolicyTrainingResult     result;
        const std::vector<Score> rounded = roundedParameters(parameters, config.residualLimit);
        result.parameterCount            = parameters.size();
        result.minimum   = rounded.empty() ? 0 : *std::min_element(rounded.begin(), rounded.end());
        result.maximum   = rounded.empty() ? 0 : *std::max_element(rounded.begin(), rounded.end());
        double squareSum = 0;
        for (Score value : rounded)
            squareSum += double(value) * double(value);
        result.rms = std::sqrt(squareSum / double(rounded.size()));

        bool published = false;
        try {
            installParameters(rounded, config.destinationTables);
            std::ofstream output(temporaryPath, std::ios::binary);
            if (!output)
                throw std::runtime_error("unable to open temporary policy model output");
            Config::exportModel(output);
            output.flush();
            if (!output)
                throw std::runtime_error("failed to write policy model output");
            output.close();

            std::filesystem::rename(temporaryPath, outputModelPath);
            published = true;
        }
        catch (...) {
            if (!published) {
                std::error_code ignored;
                std::filesystem::remove(temporaryPath, ignored);
            }
            throw;
        }
        return result;
    }
    PolicyTrainingConfig                   config;
    Corpus                                 corpus;
    std::vector<double>                    parameters;
    std::vector<double>                    initial;
    std::array<double, MAX_BOARD_SIZE + 1> boardEventWeights;
    std::array<double, RULE_NB + 1>        tableEventWeights;
    AdamOptimizer<double>                  optimizer;
};

PolicyTraceTrainer::PolicyTraceTrainer(const std::vector<std::filesystem::path> &tracePaths,
                                       const PolicyTrainingConfig               &config)
{
    if (tracePaths.empty() || config.epochs == 0 || !std::isfinite(config.learningRate)
        || config.learningRate <= 0 || !std::isfinite(config.anchor) || config.anchor < 0
        || !std::isfinite(config.scoreScale) || config.scoreScale <= 0 || config.residualLimit <= 0
        || config.residualLimit > std::numeric_limits<Score>::max())
        throw std::invalid_argument("invalid policy training configuration");
    impl_ = std::make_unique<Impl>(tracePaths, config);
}

PolicyTraceTrainer::~PolicyTraceTrainer()                                         = default;
PolicyTraceTrainer::PolicyTraceTrainer(PolicyTraceTrainer &&) noexcept            = default;
PolicyTraceTrainer &PolicyTraceTrainer::operator=(PolicyTraceTrainer &&) noexcept = default;

void PolicyTraceTrainer::step()
{
    impl_->step();
}

void PolicyTraceTrainer::refreshBaseScores()
{
    impl_->refreshBaseScores();
}

PolicyTrainingResult PolicyTraceTrainer::saveModel(const std::filesystem::path &outputModelPath)
{
    return impl_->saveModel(outputModelPath);
}

}  // namespace Tuning
