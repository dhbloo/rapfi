/*
 *  Rapfi, a Gomoku/Renju playing engine supporting piskvork protocol.
 *  Copyright (C) 2024  Rapfi developers
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

#include "evaluator.h"

#include <filesystem>

namespace Evaluation::onnx {

/// A warpper around onnx environment and session instance.
class OnnxModel;

/// Accumulator is used to track current board state and is responsible for converting
/// current board state to onnx model input and collecting onnx model output.
class OnnxAccumulator
{
public:
    virtual ~OnnxAccumulator() = default;
    /// Init accumulator state. This will only be called once at model startup.
    /// This function can throw any exception if init failed for this model.
    virtual void init(OnnxModel &model) = 0;
    /// Set accumulator state to empty board.
    virtual void clear(OnnxModel &model) = 0;
    /// Make a move on the board at coord (x, y) with piece color.
    virtual void move(OnnxModel &model, Color color, int x, int y) = 0;
    /// Undo a move on the board at coord (x, y) which was placed with piece color.
    virtual void undo(OnnxModel &model, Color color, int x, int y) = 0;
    /// Calculate value of current network state.
    virtual ValueType evaluateValue(OnnxModel &model) = 0;
    /// Calculate policy value of current network state.
    virtual void evaluatePolicy(OnnxModel &model, PolicyBuffer &policyBuffer) = 0;
};

class OnnxEvaluator : public Evaluator
{
public:
    OnnxEvaluator(int                   boardSize,
                  Rule                  rule,
                  std::filesystem::path onnxModelPath,
                  std::string           device);
    ~OnnxEvaluator();

    void initEmptyBoard();
    void beforeMove(const Board &board, Pos pos);
    void afterUndo(const Board &board, Pos pos);

    ValueType evaluateValue(const Board &board, AccLevel level);
    void      evaluatePolicy(const Board &board, PolicyBuffer &policyBuffer, AccLevel level);

private:
    OnnxModel /* non-owning ptr */  *model;
    std::unique_ptr<OnnxAccumulator> accumulator[2];
};

}  // namespace Evaluation::onnx
