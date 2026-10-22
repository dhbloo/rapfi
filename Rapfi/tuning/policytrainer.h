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

#include "policytrace.h"

#include <cstddef>
#include <filesystem>
#include <memory>
#include <vector>

namespace Tuning {

struct PolicyTrainingConfig
{
    size_t           epochs              = 0;
    double           learningRate        = 0;
    double           anchor              = 0;
    double           scoreScale          = 0;
    int              residualLimit       = 0;
    bool             equalBoardWeighting = false;
    std::vector<int> destinationTables {FREESTYLE};
};

struct PolicyTrainingResult
{
    size_t parameterCount = 0;
    Score  minimum        = 0;
    Score  maximum        = 0;
    double rms            = 0;
};

class PolicyTraceTrainer
{
public:
    PolicyTraceTrainer(const std::vector<std::filesystem::path> &tracePaths,
                       const PolicyTrainingConfig               &config);
    ~PolicyTraceTrainer();

    PolicyTraceTrainer(PolicyTraceTrainer &&) noexcept;
    PolicyTraceTrainer &operator=(PolicyTraceTrainer &&) noexcept;

    PolicyTraceTrainer(const PolicyTraceTrainer &)            = delete;
    PolicyTraceTrainer &operator=(const PolicyTraceTrainer &) = delete;

    void                 step();
    void                 refreshBaseScores();
    PolicyTrainingResult saveModel(const std::filesystem::path &outputModelPath);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace Tuning
