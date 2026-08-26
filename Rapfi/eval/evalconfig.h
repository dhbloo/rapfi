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

#include "../core/platform.h"
#include "../core/types.h"

#include <filesystem>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <vector>

namespace Evaluation {

class Evaluator;  // forward declaration (eval/evaluator.h)

/// EvaluatorConfig groups the classical/evaluator switching margins and draw
/// adjustments read from "[model.evaluator]". One mutable global instance
/// (EvalCfg) holds the live configuration; config.cpp's readEvaluator fills it
/// (behind the type+weights gate - a margins-only table has no effect), and
/// the gomocup INFO handlers assign the draw fields directly.
struct EvaluatorConfig
{
    float marginWinLossScale[RULE_NB]    = {1.22f, 0.86f, 1.24f};
    float marginWinLossExponent[RULE_NB] = {3.04f, 3.44f, 2.56f};
    float marginScale[RULE_NB]           = {351.0f, 379.2f, 389.4f};
    float drawBlackWinRate               = 0.5f;
    float drawRatio                      = 1.0f;
};

/// The global live evaluator configuration.
extern EvaluatorConfig EvalCfg;

/// Transient parsed form of the "[model.evaluator]" type+weights keys. Built
/// by config.cpp's readEvaluator, consumed by makeEvaluatorMaker, then
/// discarded. Fields are std::optional because the maker's control flow
/// branches on cpptoml key PRESENCE: a present-but-empty weight_file = ""
/// resolves and attempts the load (falling through to the next entry on
/// failure), while an absent key raises the "must specify weight_file..."
/// error - the capture mirrors cpptoml::option engagement 1:1.
struct EvaluatorWeightsConfig
{
    std::string type;       ///< "mix9svq" | "mix10" | "ort"
    std::string ortDevice;  ///< [model.evaluator] ort_device ("ort" only)
    struct WeightFiles
    {
        std::optional<std::string> file, fileBlack, fileWhite;  ///< u8 paths, unresolved
    };
    std::vector<WeightFiles> weights;
};

/// Resolves a (relative) weight path to a full path; injected so eval/ stays
/// free of command/ dependencies (config.cpp passes Command::getModelFullPath).
using PathResolver = std::function<std::filesystem::path(const std::filesystem::path &)>;

/// Signature of an evaluator maker: called per (board size, rule, NUMA node)
/// to construct an Evaluator, or nullptr if none is compatible.
using EvaluatorMakerFunc =
    std::function<std::unique_ptr<Evaluator>(int boardSize, Rule rule, Numa::NumaNodeId numaId)>;

/// Build an evaluator maker from a parsed weights config. Path resolution and
/// the weight-entry iteration happen lazily inside the returned maker, per
/// invocation, exactly as the old config.cpp lambda did.
/// @throws std::runtime_error for an unsupported cfg.type (readEvaluator calls
///     this inside loadConfig's try, preserving the parse-failure path).
EvaluatorMakerFunc makeEvaluatorMaker(EvaluatorWeightsConfig cfg, PathResolver resolvePath);

}  // namespace Evaluation
