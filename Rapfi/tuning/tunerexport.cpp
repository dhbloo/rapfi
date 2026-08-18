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

#include "tunerexport.h"

#include "../config.h"
#include "../core/iohelper.h"
#include "../eval/eval.h"
#include "../eval/evaluator.h"
#include "../game/board.h"
#include "tunerdetail.h"

#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace {

using Tuning::Float;
using Tuning::MoveScoreScaleReport;
using Tuning::TuneParam;
using Tuning::detail::encodeIntegerForTruncatingExport;

template <typename T>
double sortedQuantile(const std::vector<T> &values, double probability)
{
    if (values.empty())
        throw std::invalid_argument("quantile requires at least one observation");
    double position = probability * double(values.size() - 1);
    size_t lower    = static_cast<size_t>(position);
    size_t upper    = std::min(lower + 1, values.size() - 1);
    double fraction = position - double(lower);
    return double(values[lower]) * (1 - fraction) + double(values[upper]) * fraction;
}

struct DispersionMeasurement
{
    std::array<double, RULE_NB + 1> dispersion   = {};
    std::array<size_t, RULE_NB + 1> lists        = {};
    std::array<size_t, RULE_NB + 1> observations = {};
};

bool isTunedMoveScoreTable(const Tuning::TuningConfig &config, int table)
{
    return table == FREESTYLE && config.tuneRule[FREESTYLE]
           || table == STANDARD && config.tuneRule[STANDARD]
           || (table == RENJU || table == RENJU + WHITE) && config.tuneRule[RENJU];
}

bool isQuietMove(const Board &board, Pos pos)
{
    Color    self     = board.sideToMove();
    Pattern4 selfP4   = board.pattern4(pos, self);
    Pattern4 opponent = board.pattern4(pos, ~self);
    auto     quiet    = [](Pattern4 pattern) { return pattern != FORBID && pattern < E_BLOCK4; };
    return quiet(selfP4) && quiet(opponent);
}

DispersionMeasurement measureMoveScoreDispersion(
    const std::vector<Tuning::MoveScoreReferencePosition> &referencePositions,
    const Tuning::TuningConfig                            &config)
{
    std::array<std::vector<double>, RULE_NB + 1> centeredScores;
    std::vector<Score>                           moveScores;
    DispersionMeasurement                        measurement;
    for (const Tuning::MoveScoreReferencePosition &reference : referencePositions) {
        if (!config.tuneRule[reference.rule])
            continue;
        Board board(reference.boardSize);
        board.newGame(reference.rule);
        for (Pos move : reference.moves) {
            board.move(reference.rule, move);

            moveScores.clear();
            FOR_EVERY_EMPTY_CAND_POS(&board, pos)
            {
                if (isQuietMove(board, pos))
                    moveScores.push_back(board.score(reference.rule, pos, board.sideToMove()));
            }
            if (moveScores.size() < 2)
                continue;

            std::sort(moveScores.begin(), moveScores.end());
            double median = sortedQuantile(moveScores, 0.5);
            int    table  = Evaluation::tableIndex(reference.rule, board.sideToMove());
            for (Score score : moveScores)
                centeredScores[table].push_back(double(score) - median);
            measurement.lists[table]++;
        }
    }

    for (int table = 0; table < RULE_NB + 1; table++) {
        if (!isTunedMoveScoreTable(config, table))
            continue;
        if (centeredScores[table].empty())
            throw std::runtime_error(
                "move-score scale reference has no quiet-band observations for table "
                + std::to_string(table));
        std::sort(centeredScores[table].begin(), centeredScores[table].end());
        measurement.observations[table] = centeredScores[table].size();
        measurement.dispersion[table]   = sortedQuantile(centeredScores[table], 0.75)
                                        - sortedQuantile(centeredScores[table], 0.25);
        if (!std::isfinite(measurement.dispersion[table]) || measurement.dispersion[table] <= 0)
            throw std::runtime_error(
                "move-score quiet-band reference dispersion is not positive for table "
                + std::to_string(table));
    }
    return measurement;
}

int projectionContextForComponent(int table, size_t side)
{
    if (side == 0 || table < RENJU)
        return table;
    return table == RENJU ? RENJU + WHITE : RENJU;
}

Score projectedMoveScore(Score                               score,
                         int                                 table,
                         size_t                              side,
                         const Tuning::MoveScoreScaleReport &report,
                         const Tuning::TuningConfig         &config)
{
    if (!report.projectionEnabled || !isTunedMoveScoreTable(config, table))
        return score;
    int       context   = projectionContextForComponent(table, side);
    double    center    = report.projectionCenters[table][side];
    double    projected = center + report.projectionFactors[context] * (double(score) - center);
    long long rounded   = std::llround(projected);
    return static_cast<Score>(std::clamp<long long>(rounded,
                                                    static_cast<long long>(config.moveScoreMin),
                                                    static_cast<long long>(config.moveScoreMax)));
}

}  // namespace

namespace Tuning::detail {

MoveScoreScaleReport
makeMoveScoreReference(const TuningConfig                            &config,
                       const std::vector<MoveScoreReferencePosition> &referencePositions)
{
    MoveScoreScaleReport report;
    if (!config.projectMoveScoreScale || !config.tuneMoveScore)
        return report;

    DispersionMeasurement loaded = measureMoveScoreDispersion(referencePositions, config);
    report.projectionEnabled     = true;
    report.referenceLists        = loaded.lists;
    report.referenceObservations = loaded.observations;
    report.loadedDispersion      = loaded.dispersion;
    for (int table = 0; table < RULE_NB + 1; table++) {
        if (!isTunedMoveScoreTable(config, table))
            continue;
        report.projectedTables[table]     = true;
        report.referenceDispersion[table] = config.moveScoreReferenceDispersion > 0
                                                ? config.moveScoreReferenceDispersion
                                                : loaded.dispersion[table];
        if (!std::isfinite(report.referenceDispersion[table])
            || report.referenceDispersion[table] <= 0)
            throw std::invalid_argument("move-score reference dispersion must be positive");
        MESSAGEL("Move-score table "
                 << table << " scale reference = " << report.referenceDispersion[table]
                 << " (loaded " << report.loadedDispersion[table] << ", "
                 << report.referenceObservations[table] << " quiet observations, version "
                 << "quiet-centered-iqr-v1).");
    }
    return report;
}

}  // namespace Tuning::detail

namespace Tuning {

/// saveParams() saves tuneParams back to their associated config value
void Tuner::saveParams() const
{
    for (const ParamsSyncRecord &record : syncRecords) {
        assert(tuneParams.size() >= record.baseIndex + record.numElems * record.paramPerElem);

        for (size_t i = 0; i < record.numElems; i++)
            for (size_t j = 0; j < record.paramPerElem; j++)
                record.setter(record[i],
                              j,
                              tuneParams[record.baseIndex + i * record.paramPerElem + j]);
    }
    for (const TiedMoveScoreSyncRecord &record : tiedMoveScoreSyncRecords)
        for (size_t i = 0; i < record.parameterIndices.size(); i++)
            for (size_t side = 0; side < 2; side++)
                record.scores[i][side] =
                    decodeMoveScoreParam(tuneParams[record.parameterIndices[i][side]]);
    MESSAGEL(tuneParams.size() << " parameters saved.");
    projectMoveScoreScale();
}

void Tuner::synchronizeProjectedMoveScores()
{
    if (!config.tuneMoveScore || !config.projectMoveScoreScale)
        throw std::logic_error(
            "move-score projection boundary requires active projected move-score tuning");

    saveParams();
    if (tiedMoveScoreSyncRecords.empty())
        throw std::logic_error("move-score projection requires a coalesced tied layout");
    for (const TiedMoveScoreSyncRecord &record : tiedMoveScoreSyncRecords)
        for (size_t i = 0; i < record.parameterIndices.size(); i++)
            for (size_t side = 0; side < 2; side++) {
                ParameterId index   = record.parameterIndices[i][side];
                TuneParam   encoded = encodeIntegerForTruncatingExport(record.scores[i][side],
                                                                     config.moveScoreScale,
                                                                     config.moveScoreBias);
                if (decodeMoveScoreParam(encoded) != record.scores[i][side])
                    throw std::runtime_error(
                        "projected move-score boundary is not representable by tuner parameters");
                tuneParams[index] = encoded;
            }
    MESSAGEL("Projected move-score parameters synchronized at curriculum boundary.");
}

/// decodeMoveScoreParam() is the release trainer's shared policy quantization.
Score Tuner::decodeMoveScoreParam(TuneParam param) const
{
    Float score = Float(param) * config.moveScoreScale + config.moveScoreBias;
    return static_cast<Score>(
        std::clamp(score, Float(config.moveScoreMin), Float(config.moveScoreMax)));
}

MoveScoreScaleReport Tuner::projectMoveScoreScale() const
{
    MoveScoreScaleReport report = moveScoreReference;
    if (!config.projectMoveScoreScale || !config.tuneMoveScore)
        return report;

    DispersionMeasurement unprojected =
        measureMoveScoreDispersion(moveScoreReferencePositions, config);
    report.unprojectedDispersion = unprojected.dispersion;

    for (int table = 0; table < RULE_NB + 1; table++) {
        if (!report.projectedTables[table])
            continue;
        report.projectionFactors[table] =
            report.referenceDispersion[table] / report.unprojectedDispersion[table];
        if (!std::isfinite(report.projectionFactors[table]) || report.projectionFactors[table] <= 0)
            throw std::runtime_error("move-score scale projection factor is not positive");
    }

    std::vector<Score> tableValues;
    tableValues.reserve(PCODE_NB);
    for (int table = 0; table < RULE_NB + 1; table++) {
        if (!isTunedMoveScoreTable(config, table))
            continue;
        for (size_t side = 0; side < 2; side++) {
            tableValues.clear();
            for (const MoveScorePair &score : Evaluation::P4SCORES[table])
                tableValues.push_back(score[side]);
            std::sort(tableValues.begin(), tableValues.end());
            report.projectionCenters[table][side] = sortedQuantile(tableValues, 0.5);
        }
    }

    for (int table = 0; table < RULE_NB + 1; table++) {
        if (!isTunedMoveScoreTable(config, table))
            continue;
        for (MoveScorePair &score : Evaluation::P4SCORES[table])
            for (size_t side = 0; side < 2; side++)
                score[side] = projectedMoveScore(score[side], table, side, report, config);
    }

    DispersionMeasurement exported =
        measureMoveScoreDispersion(moveScoreReferencePositions, config);
    report.exportedDispersion = exported.dispersion;
    for (int table = 0; table < RULE_NB + 1; table++) {
        if (!report.projectedTables[table])
            continue;
        double factor = report.projectionFactors[table];
        double tolerance = std::max(1.0, report.referenceDispersion[table] * 0.02);
        if (std::abs(report.exportedDispersion[table] - report.referenceDispersion[table])
            > tolerance)
            throw std::runtime_error(
                "move-score scale projection missed its reference dispersion for table "
                + std::to_string(table));
        MESSAGEL("Move-score table "
                 << table << " scale projection: D_ref = " << report.referenceDispersion[table]
                 << ", D_raw = " << report.unprojectedDispersion[table] << ", factor = " << factor
                 << ", D_export = " << report.exportedDispersion[table] << ".");
        if (factor < 0.9 || factor > 1.1)
            MESSAGEL("Warning: move-score table "
                     << table << " projection factor differs from 1.0 by more than 10%.");
    }
    return report;
}

}  // namespace Tuning
