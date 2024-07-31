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

#include "../core/pos.h"
#include "../core/types.h"
#include "../game/board.h"

#include <array>
#include <flat.hpp/flat_map.hpp>
#include <functional>
#include <unordered_map>
#include <vector>

namespace Tuning {

typedef double Float;
typedef Float  TuneParam;  // the numeric type used for parameter in tuning

/// LossType represents a type of loss function to use.
enum class LossType { L1, L2, BCE };

/// TuneCoeff struct is the coefficient for a tunable value.
/// It represents an atomic term in the linear evaluation formula,
/// for a TuneElement E: eval = ... + E.coeff * Param(index) + ...
template <typename CoeffType = int16_t, typename IndexType = uint16_t>
struct TuneCoeff
{
    CoeffType coeff;
    IndexType index;
};

/// CoeffVec struct is a vector of coefficients C=(C0, C1, ..., Cn).
/// @tparam Storage The underlying storage type of the coefficient vector.
template <typename Storage>
struct CoeffVec : Storage
{
    Float linearValue(const std::vector<TuneParam> &params) const;
};
/// DynamicCoeffVec is CoeffVec with dynamic storage.
using DynamicCoeffVec = CoeffVec<std::vector<TuneCoeff<>>>;
/// FixedCoeffVec is CoeffVec with fixed storage of Size.
template <size_t Size>
using FixedCoeffVec = CoeffVec<std::array<TuneCoeff<>, Size>>;

/// TuneEntry struct contains all information needed to tune a position.
struct TuneEntry
{
    static constexpr size_t CoeffScale = 8;

    Float           result;  // win rate [0.0~1.0], 0=loss, 0.5=draw, 1=win
    Value           staticEval;
    Pos             bestMove;
    DynamicCoeffVec evalCoeffs;
    flat_hpp::flat_map<Pos, FixedCoeffVec<2>> scoreCoeffs;

    TuneEntry(const struct DataEntry          &dataEntry,
              const class Tuner               &tuner,
              std::unordered_map<int, Board> &&boardObjectCache = {});

    Float computeLinearEval(const std::vector<TuneParam> &params) const
    {
        return evalCoeffs.linearValue(params) * (Float(1) / CoeffScale);
    }
    template <bool UseTunedEval = true>
    Float computeEvalLoss(const std::vector<TuneParam> &params, Float K, LossType loss) const;
    Float computeMoveScoreLoss(const std::vector<TuneParam> &params, Float gamma) const;
    void  computeEvalGradient(std::vector<Float>           &grads,
                              const std::vector<TuneParam> &params,
                              Float                         K,
                              LossType                      loss) const;
    void  computeMoveScoreGradient(std::vector<Float>           &grads,
                                   const std::vector<TuneParam> &params,
                                   Float                         gamma) const;
};

/// ParamGetter gets a TuneParam from a raw parameter given by an address and an offset.
template <typename AddrType = void *>
using ParamGetter = std::function<TuneParam(AddrType, size_t)>;
/// ParamSetter sets a raw parameter given by an address and an offset to a TuneParam.
template <typename AddrType = void *>
using ParamSetter = std::function<void(AddrType, size_t, TuneParam)>;

/// ParamsRecord struct is a helper to sync tune parameters with the config.
/// It also maps a range of addresses that contains the tunable parameters
/// to its base index in the tuneParams array.
struct ParamsSyncRecord
{
    size_t   baseIndex;
    size_t   numElems;
    uint32_t elemSize;
    uint32_t paramPerElem;
    void    *address;

    ParamGetter<> getter;
    ParamSetter<> setter;

    void       *operator[](size_t i) const { return static_cast<char *>(address) + elemSize * i; }
    friend bool operator<(void *addr, const ParamsSyncRecord &record)
    {
        return addr < record.address;
    }
};

/// TuningConfig struct records all configuration settings used in Tuner
struct TuningConfig
{
    // --------------------------------------------
    // General training settings

    size_t   batchSize           = 8192;
    size_t   maxTuneEntries      = UINT32_MAX;
    double   learningRate        = 0.01;
    double   weightDecay         = 0.0;
    double   moveScoreLossGamma  = 0.0;
    double   moveScoreScale      = 24.0;
    double   moveScoreBias       = 24.0;
    Score    moveScoreMin        = -999;
    Score    moveScoreMax        = 999;
    LossType lossType            = LossType::BCE;
    bool     shuffleTuneEntries  = false;
    bool     tuneEval            = true;
    bool     tuneMoveScore       = false;
    bool     randomMoveScoreInit = false;

    // --------------------------------------------
    // Data entry filter settings

    bool     tuneRule[RULE_NB] = {};
    uint8_t  boardSizeMin      = 5;
    uint8_t  boardSizeMax      = MAX_BOARD_SIZE;
    uint16_t minPly            = 1;
    uint16_t minPlyBeforeFull  = 50;

    // --------------------------------------------
    // Scaling Factor searching settings

    bool   usePreviousScalingFactor = false;
    int    nIterations              = 10;
    int    nStepsPerIteration       = 10;
    double scalingFactorMin         = 100;
    double scalingFactorMax         = 400;
    size_t recomputeInterval        = 0;
};

/// TuningStatistic struct records all current statistic in tuning process.
/// This can be used to produce a training record for reporting.
struct TuningStatistic
{
    size_t currentEpoch;
    double valueLoss, policyLoss;
    double valueValLoss, policyValLoss;
    double elapsedSeconds;
    double scalingFactor;
};

/// Tuner runs the whole tuning process for the given dataset and tuning config.
/// Some actions in tuning process will be performed in parallel.
class Tuner
{
public:
    Tuner(class Dataset &trainDataset, class Dataset *valDataset, TuningConfig config = {});
    Tuner(const Tuner &) = delete;

    void run(size_t epochs, std::function<void(TuningStatistic)> callback = nullptr);
    void saveParams() const;

private:
    friend struct TuneEntry;

    const TuningConfig            config;
    std::vector<TuneEntry>        trainTuneEntries, valTuneEntries;
    std::vector<TuneParam>        tuneParams;
    std::vector<ParamsSyncRecord> syncRecords;

    void  initParams();
    void  initTuneEntries(std::vector<TuneEntry> &tuneEntries, class Dataset &dataset);
    Float searchOptimalInvScalingFactor() const;
    template <bool UseTunedEval = true>
    Float computeEvaluationLoss(Float K, bool validation) const;
    Float computeMoveScoreLoss(bool validation) const;
    void  computeGradientBatch(std::vector<Float> &grads, Float K, size_t batchIdx);

    void   addParams(void         *address,
                     size_t        numElems,
                     uint32_t      elemSize,
                     uint32_t      paramPerElem,
                     ParamGetter<> getter,
                     ParamSetter<> setter);
    size_t paramIndex(void *address, size_t offset = 0) const;

    /* helper functions to add typed params to synced tune params */

    template <typename T, size_t ParamPerElem = 1>
    void addSingleParam(T &param, ParamGetter<const T &> getter, ParamSetter<T &> setter);
    template <typename T, size_t Length, size_t ParamPerElem = 1>
    void
    addArrayParams(T (&paramArray)[Length], ParamGetter<const T &> getter, ParamSetter<T &> setter);
};

}  // namespace Tuning

template <typename T, size_t ParamPerElem>
inline void
Tuning::Tuner::addSingleParam(T &param, ParamGetter<const T &> getter, ParamSetter<T &> setter)
{
    addParams(
        &param,  // std::addressof() might be better
        1,
        sizeof(T),
        ParamPerElem,
        [getter = std::move(getter)](void *addr, size_t offset) -> TuneParam {
            return getter(*static_cast<const T *>(addr), offset);
        },
        [setter = std::move(setter)](void *addr, size_t offset, TuneParam param) -> void {
            setter(*static_cast<T *>(addr), offset, param);
        });
}

template <typename T, size_t Length, size_t ParamPerElem>
inline void Tuning::Tuner::addArrayParams(T                      (&paramArray)[Length],
                                          ParamGetter<const T &> getter,
                                          ParamSetter<T &>       setter)
{
    addParams(
        paramArray,
        Length,
        sizeof(T),
        ParamPerElem,
        [getter = std::move(getter)](void *addr, size_t offset) -> TuneParam {
            return getter(*static_cast<const T *>(addr), offset);
        },
        [setter = std::move(setter)](void *addr, size_t offset, TuneParam param) -> void {
            setter(*static_cast<T *>(addr), offset, param);
        });
}
