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

#include "tuner.h"

#include "../config.h"
#include "../core/iohelper.h"
#include "../eval/eval.h"
#include "BS_thread_pool.hpp"
#include "dataset.h"
#include "optimizer.h"

#include <algorithm>
#include <cmath>
#include <future>
#include <iomanip>
#include <iterator>
#include <mutex>
#include <numeric>
#include <random>

namespace {

using Tuning::Float;
using Tuning::LossType;

inline bool checkEqual(Float a, Float b)
{
    return std::abs(a - b) <= Float(1.0);
}

/// sigmoid(x) = 1/(1+exp(-x))
inline Float sigmoid(Float x)
{
    return Float(1) / (Float(1) + std::exp(-x));
}

inline Float scoreToWinrate(Float score, Float invScalingFactor)
{
    return sigmoid(score * invScalingFactor);
}

inline Float scoreToWinrateGrad(Float winrate, Float invScalingFactor)
{
    return winrate * (Float(1) - winrate) * invScalingFactor;
}

inline Float lossFunction(LossType lt, Float pred, Float target)
{
    switch (lt) {
    case LossType::L1: return std::abs(target - pred);
    case LossType::L2: {
        Float dist = target - pred;
        return dist * dist;
    }
    case LossType::BCE: {
        constexpr Float BiasEps = 1e-8;
        Float           loss    = -(target * std::log(pred) + (1 - target) * std::log(1 - pred));
        Float           bias =
            target * std::log(target + BiasEps) + (1 - target) * std::log(1 + BiasEps - target);
        return (loss + bias) / 2;
    }
    default: return Float(0);
    }
}

inline Float lossFunctionGrad(LossType lt, Float pred, Float target)
{
    Float diff = pred - target;
    switch (lt) {
    case LossType::L1: return (Float(0) < diff) - (diff < Float(0));
    case LossType::L2: return Float(2) * diff;
    case LossType::BCE: return diff / (pred * (Float(1) - pred));
    default: return Float(0);
    }
}

/// Collect coefficient from eval info
template <typename Collector>
void collectEvalCoeffs(Rule r, const Evaluation::EvalInfo &evalInfo, Collector collect)
{
    Color self = evalInfo.self, oppo = ~self;

    // Collect pattern code coefficient
    for (size_t pcode = 0; pcode < PCODE_NB; pcode++) {
        int coeff[2][SIDE_NB] = {{evalInfo.plyBack[0].pcodeCount[BLACK][pcode],
                                  evalInfo.plyBack[0].pcodeCount[WHITE][pcode]},
                                 {evalInfo.plyBack[1].pcodeCount[BLACK][pcode],
                                  evalInfo.plyBack[1].pcodeCount[WHITE][pcode]}};

        // Coefficient is scaled at 2x
        if (r == RENJU) {
            collect(coeff[0][self] + coeff[1][self], 2, &Config::EVALS[r + self][pcode]);
            collect(-coeff[0][oppo] - coeff[1][oppo], 2, &Config::EVALS[r + oppo][pcode]);
        }
        else {
            collect(coeff[0][self] - coeff[0][oppo] + coeff[1][self] - coeff[1][oppo],
                    2,
                    &Config::EVALS[r][pcode]);
        }
    }

    // Collect threat eval coefficient
    collect(1, 1, &Config::EVALS_THREAT[Config::tableIndex(r, self)][evalInfo.threatMask]);
}

/// Collect coefficient from board pattern codes
template <typename Collector>
void collectMoveScoreCoeffs(Rule r, const Board &board, Collector collect)
{
    Color self = board.sideToMove(), oppo = ~self;

    // Collect scores of all candidate moves
    FOR_EVERY_EMPTY_POS(&board, pos)
    {
        const Cell &c = board.cell(pos);
        if (c.isCandidate()) {
            collect(pos,
                    1,
                    1,
                    &Config::P4SCORES[Config::tableIndex(r, self)][c.pcode<BLACK>()],
                    &Config::P4SCORES[Config::tableIndex(r, oppo)][c.pcode<WHITE>()]);
        }
    }
}

static BS::thread_pool ThreadPool;

template <typename InputIt, typename T, typename BinaryOp, typename UnaryOp>
T parallel_transform_reduce(InputIt first, InputIt last, T init, BinaryOp reduce, UnaryOp transform)
{
    auto length = std::distance(first, last);
    if (length == 0)
        return init;

    constexpr size_t MaxBlockSize = 16384;
    constexpr size_t MinBlockSize = 4096;
    size_t           numThreads   = ThreadPool.get_thread_count();
    size_t blockSize = std::clamp<size_t>(length / numThreads, MinBlockSize, MaxBlockSize);
    size_t numBlocks = (length + blockSize - 1) / blockSize;
    std::vector<std::future<T>> futures;

    // parallel transform phase
    for (size_t i = 0; i < numBlocks; ++i) {
        auto blockBegin = first + i * blockSize;
        auto blockEnd   = std::min(blockBegin + blockSize, last);

        futures.emplace_back(ThreadPool.submit_task([blockBegin, blockEnd, transform]() {
            T blockResult = T {};
            for (auto it = blockBegin; it != blockEnd; ++it)
                blockResult += transform(*it);
            return blockResult;
        }));
    }

    // parallel reduce phase
    size_t futureBegin = 0;
    size_t futureEnd   = futures.size();
    size_t chunkSize   = std::clamp<size_t>(futureEnd / numThreads, MinBlockSize, MaxBlockSize);
    while (futureEnd - futureBegin >= chunkSize) {
        size_t newNumFeatures = 0;

        for (size_t i = futureBegin; i + (chunkSize - 1) < futureEnd; i += chunkSize) {
            futures.emplace_back(ThreadPool.submit_task([&futures, reduce, chunkSize, i]() {
                T chunkResult = T {};
                for (size_t j = 0; j < chunkSize; ++j)
                    chunkResult = reduce(chunkResult, futures[i + j].get());
                return chunkResult;
            }));
            newNumFeatures++;
            futureBegin += chunkSize;
        }

        futureEnd += newNumFeatures;
        chunkSize =
            std::clamp<size_t>((futureEnd - futureBegin) / numThreads, MinBlockSize, MaxBlockSize);
    }

    // sequential reduce phase
    T result = init;
    for (size_t i = futureBegin; i < futureEnd; ++i)
        result = reduce(result, futures[i].get());

    return result;
}

}  // namespace

namespace Tuning {

/// CoeffVec::linearValue() computes linear value from coeffs and tuner params.
template <typename Storage>
Float CoeffVec<Storage>::linearValue(const std::vector<TuneParam> &params) const
{
    Float value = Float(0);
    for (const auto c : *this)
        value += c.coeff * params[c.index];
    return value;
}

/// TuneEntry::TuneEntry() init tune entry from raw data entry.
TuneEntry::TuneEntry(const DataEntry                 &dataEntry,
                     const class Tuner               &tuner,
                     std::unordered_map<int, Board> &&boardObjectCache)
{
    // Get board object from cache to save redundent memory allocation
    if (boardObjectCache.find(dataEntry.boardsize) == boardObjectCache.end()) {
        boardObjectCache.emplace(std::piecewise_construct,
                                 std::forward_as_tuple(dataEntry.boardsize),
                                 std::forward_as_tuple(dataEntry.boardsize));
    }

    // Init board from the given position
    Board &board = boardObjectCache.find(dataEntry.boardsize)->second;
    board.newGame(dataEntry.rule);
    for (Pos pos : dataEntry.position) {
        board.move(dataEntry.rule, pos);
    }

    result     = dataEntry.result == RESULT_WIN    ? Float(1)
                 : dataEntry.result == RESULT_DRAW ? Float(0.5)
                                                   : Float(0);
    staticEval = Evaluation::evaluate(board, dataEntry.rule);

    // Calculate all coefficients with certain rule and evalInfo
    if (tuner.config.tuneEval) {
        // Init all infomation used in evaluation
        Evaluation::EvalInfo evalInfo(board, dataEntry.rule);

        collectEvalCoeffs(dataEntry.rule,
                          evalInfo,
                          [this, &tuner](int coeff, int coeffScale, void *addr) {
                              // Check overflow in coefficient
                              assert(INT16_MIN <= coeff && coeff <= INT16_MAX);

                              // Skip zero coefficient
                              if (coeff == 0)
                                  return;

                              evalCoeffs.push_back({(int16_t)(coeff * CoeffScale / coeffScale),
                                                    (uint16_t)tuner.paramIndex(addr)});
                          });
    }

    // Calculate all coefficients with best move (if best move is not none)
    if (tuner.config.tuneMoveScore
        && dataEntry.move != Pos {dataEntry.boardsize, dataEntry.boardsize}
        && board.cell(dataEntry.move).isCandidate()) {
        bestMove = dataEntry.move;
        collectMoveScoreCoeffs(
            dataEntry.rule,
            board,
            [this,
             &tuner](Pos pos, auto coeffSelf, auto coeffOppo, void *addrSelf, void *addrOppo) {
                // Check overflow in coefficient
                assert(INT16_MIN <= coeffSelf && coeffSelf <= INT16_MAX);
                assert(INT16_MIN <= coeffOppo && coeffOppo <= INT16_MAX);

                scoreCoeffs[pos][0] = {(int16_t)(coeffSelf * CoeffScale),
                                       (uint16_t)tuner.paramIndex(addrSelf, 0)};
                scoreCoeffs[pos][1] = {(int16_t)(coeffOppo * CoeffScale),
                                       (uint16_t)tuner.paramIndex(addrOppo, 1)};
            });
        assert(scoreCoeffs.find(bestMove) != scoreCoeffs.end());
    }
    else {
        bestMove = Pos::NONE;
    }

    // Check if linear eval is correct
    assert(checkEqual(Float(staticEval), computeLinearEval(tuner.tuneParams)));
}

/// TuneEntry::computeEvalLoss() computes eval loss with the given K.
template <bool UseTunedEval>
Float TuneEntry::computeEvalLoss(const std::vector<TuneParam> &params, Float K, LossType loss) const
{
    if (evalCoeffs.empty())
        return Float(0);

    Float eval    = UseTunedEval ? computeLinearEval(params) : Float(staticEval);
    Float winrate = scoreToWinrate(eval, K);
    return lossFunction(loss, winrate, result);
}

/// TuneEntry::computeMoveScoreLoss() computes loss of move scores with best move.
Float TuneEntry::computeMoveScoreLoss(const std::vector<TuneParam> &params, Float gamma) const
{
    if (scoreCoeffs.empty())
        return Float(0);

    // Cross Entropy Loss:
    // loss(x, class) = -x[class] + log(Sum_j(exp(x[j])))
    //                = -(x[class] - maxScore) + log(Sum_j(exp(x[j] - maxScore)))
    // Focal Cross-Entropy Loss:
    // loss(x, class) = pow(1 - x[class], gamma) * (-x[class] + log(Sum_j(exp(x[j]))))
    //                = pow(1 - x[class], gamma) *
    //                     (-(x[class] - maxScore) + log(Sum_j(exp(x[j] - maxScore))))

    Float scores[FULL_BOARD_CELL_COUNT];
    Float maxScore = std::numeric_limits<Float>::lowest();
    Float sum      = Float(0);
    std::fill(std::begin(scores), std::end(scores), std::numeric_limits<Float>::lowest());

    for (const auto &[pos, coeffs] : scoreCoeffs) {
        Float score = coeffs.linearValue(params) * Float(1.0 / CoeffScale);
        scores[pos] = score;
        maxScore    = std::max(score, maxScore);
    }
    for (size_t pos = 0; pos < arraySize(scores); pos++) {
        if (scores[pos] != std::numeric_limits<Float>::lowest())
            sum += std::exp(scores[pos] - maxScore);
    }

    assert(scores[bestMove] != std::numeric_limits<Float>::lowest());
    Float score_class = scores[bestMove] - maxScore;
    Float x_class     = std::exp(score_class) / sum;
    Float focalWeight = std::pow(Float(1) - x_class, gamma);
    return focalWeight * (-score_class + std::log(sum));
}

/// TuneEntry::computeEvalGradient() computes gradient of params for this tune entry
/// based on loss function of eval. The gradient is not normalized.
void TuneEntry::computeEvalGradient(std::vector<Float>           &grads,
                                    const std::vector<TuneParam> &params,
                                    Float                         K,
                                    LossType                      loss) const
{
    if (evalCoeffs.empty())
        return;

    // For linear evaluation: Eval = ... + coeffs[i] * params[i] + ...
    // Its gradient to l2 loss: d_l2/d_params[j] = -2K/N * Sum_i^N(coeffs[j] * E_i),
    // where N is the number of tune entries.
    //   E_i = W_i * (1 - W_i) * (Result_i - W_i)
    //   W_i = sigmoid(K * E_i)

    Float winrate        = scoreToWinrate(computeLinearEval(params), K);
    Float dWinrate_dEval = scoreToWinrateGrad(winrate, K);
    Float dL_dWinrate    = lossFunctionGrad(loss, winrate, result);
    Float dL_dEval       = dL_dWinrate * dWinrate_dEval;

    for (const auto &c : evalCoeffs)
        grads[c.index] += c.coeff * dL_dEval;
}

/// TuneEntry::computeMoveScoreGradient() computes gradient of params for this tune
/// entry based on loss function of move score. The gradient is not normalized.
void TuneEntry::computeMoveScoreGradient(std::vector<Float>           &grads,
                                         const std::vector<TuneParam> &params,
                                         Float                         gamma) const
{
    if (scoreCoeffs.empty())
        return;

    // For linear move score: score[pos] = Sum_i(coeffs[i] * params[i])
    // d_ce/d_score[i] = exp(X_i - m) / Sum_j^N(exp(X_j - m)) - (i == class),
    // d_score[i]/d_params[j] = coeffs[j],
    // where N is the number of move scores.
    Float scores[FULL_BOARD_CELL_COUNT];
    Float x[FULL_BOARD_CELL_COUNT];
    Float maxScore = std::numeric_limits<Float>::lowest();
    Float sumExp   = Float(0);
    std::fill(std::begin(scores), std::end(scores), std::numeric_limits<Float>::lowest());

    for (const auto &[pos, coeffs] : scoreCoeffs) {
        Float score = coeffs.linearValue(params) / CoeffScale;
        scores[pos] = score;
        maxScore    = std::max(score, maxScore);
    }
    for (size_t pos = 0; pos < arraySize(scores); pos++) {
        if (scores[pos] != std::numeric_limits<Float>::lowest()) {
            x[pos] = std::exp(scores[pos] - maxScore);
            sumExp += x[pos];
        }
        else {
            x[pos] = Float(0);
        }
    }

    // Gradient of Focal Loss for multi-class classification.
    Float invSumExp        = Float(1) / sumExp;
    Float Pt               = x[bestMove] * invSumExp;
    Float logPt            = scores[bestMove] - maxScore - std::log(sumExp);
    Float PtClamped        = std::clamp(Pt, Float(1e-6), Float(1 - 1e-6));
    Float PtLogPtDivPtSub1 = PtClamped / (PtClamped - 1) * logPt;
    assert(!std::isnan(PtLogPtDivPtSub1));
    Float dFLdCE = std::pow(1 - Pt, gamma) * (gamma * PtLogPtDivPtSub1 + 1);
    for (const auto &[pos, coeffs] : scoreCoeffs) {
        Float dCEdScore = pos == bestMove ? Pt - 1 : x[pos] * invSumExp;
        Float dFLdScore = dFLdCE * dCEdScore;

        for (const auto c : coeffs)
            grads[c.index] += c.coeff * dFLdScore;
    }
}

Tuner::Tuner(Dataset &trainDataset, Dataset *valDataset, TuningConfig config) : config(config)
{
    MESSAGEL("Start initializing parameters...");
    initParams();

    MESSAGEL("Start initializing tune entries from training dataset...");
    initTuneEntries(trainTuneEntries, trainDataset);

    if (valDataset) {
        MESSAGEL("Start initializing tune entries from validation dataset...");
        initTuneEntries(valTuneEntries, *valDataset);
    }
}

/// run() runs the tuner for specified epochs. After each epoch completed, callback will be called.
void Tuner::run(size_t epochs, std::function<void(TuningStatistic)> callback)
{
    Time initTime = now();

    // Set Float output precision
    std::cout << std::setprecision(std::min(std::numeric_limits<Float>::digits10, 7)) << std::fixed;

    // Search a new K or use previous K
    Float K = Float(1.0) / Config::ScalingFactor;
    if (config.usePreviousScalingFactor) {
        MESSAGEL("Use previous inv scaling factor = " << K);
    }
    else {
        MESSAGEL("Start seaching for optimal inv scaling factor...");
        K = searchOptimalInvScalingFactor();
    }

    // Note: the last non-full batch of tune entries will be dropped
    size_t numBatches = trainTuneEntries.size() / config.batchSize;

    // Init gradient array and optimizer
    std::vector<Float>   gradients(tuneParams.size());
    AdamOptimizer<Float> optim(tuneParams.size(),
                               Float(config.learningRate),
                               Float(config.weightDecay));

    MESSAGEL("Start training for " << epochs << " epochs, lr = " << optim.currentLR()
                                   << ", batch size = " << config.batchSize
                                   << ", number of batches = " << numBatches << ".");

    for (size_t epoch = 0; epoch <= epochs; epoch++) {
        Time startTime = now();

        for (size_t batch = 0; epoch > 0 && batch < numBatches; batch++) {
            // Zero out all gradients
            std::fill(gradients.begin(), gradients.end(), Float(0));

            // Compute gradient of all parameters using current K
            computeGradientBatch(gradients, K, batch);

            // Update parameters with gradient using optimizer
            optim.step(tuneParams, gradients);
        }

        // Print out current epoch and loss
        Float valueLoss     = computeEvaluationLoss(K, false);
        Float policyLoss    = computeMoveScoreLoss(false);
        Float valueValLoss  = computeEvaluationLoss(K, true);
        Float policyValLoss = computeMoveScoreLoss(true);
        Time  elapsed       = now() - startTime;
        if (!valTuneEntries.empty())
            MESSAGEL("Epoch " << epoch << " | Value " << valueLoss << " | Policy " << policyLoss
                              << " | ValueVal " << valueValLoss << " | PolicyVal " << policyValLoss
                              << " | Time(ms) " << elapsed);
        else
            MESSAGEL("Epoch " << epoch << " | Value " << valueLoss << " | Policy " << policyLoss
                              << " | Time(ms) " << elapsed);

        // Call callback after each epoch completed
        if (callback) {
            TuningStatistic stat;
            stat.currentEpoch   = epoch;
            stat.valueLoss      = valueLoss;
            stat.policyLoss     = policyLoss;
            stat.valueValLoss   = valueValLoss;
            stat.policyValLoss  = policyValLoss;
            stat.elapsedSeconds = double(elapsed) / 1000.0;
            stat.scalingFactor  = 1.0 / double(K);
            callback(stat);
        }

        // Recompute K for tuned evaluation
        if (!config.usePreviousScalingFactor && epoch > 0 && config.recomputeInterval
            && epoch % config.recomputeInterval == 0) {
            K = searchOptimalInvScalingFactor();
        }
    }

    Time totalElapsed = now() - initTime;
    MESSAGEL("Training completed in " << (totalElapsed / 1000) << " seconds.");
}

/// initParams() inits tuneParams according to their value in config. It also
/// associates TuneParam index with its config address. Parameters loaded from
/// config will be automatically saved back when Tuner is destroyed.
void Tuner::initParams()
{
    std::vector<int>                       ruleSetIdx;
    PRNG                                   prng;
    std::uniform_real_distribution<double> rand;

    if (config.tuneRule[FREESTYLE])
        ruleSetIdx.push_back(FREESTYLE);
    if (config.tuneRule[STANDARD])
        ruleSetIdx.push_back(STANDARD);
    if (config.tuneRule[RENJU]) {
        ruleSetIdx.push_back(RENJU + BLACK);
        ruleSetIdx.push_back(RENJU + WHITE);
    }

    for (int r : ruleSetIdx) {
        if (config.tuneEval) {
            addArrayParams<Eval>(
                Config::EVALS[r],
                [](const Eval &ev, size_t) { return TuneParam(ev); },
                [](Eval &ev, size_t, TuneParam param) { ev = Eval(param); });

            addArrayParams<Eval>(
                Config::EVALS_THREAT[r],
                [](const Eval &ev, size_t) { return TuneParam(ev); },
                [](Eval &ev, size_t, TuneParam param) { ev = Eval(param); });
        }

        if (config.tuneMoveScore) {
            addArrayParams<Pattern4Score, arraySize(Config::P4SCORES[0]), 2>(
                Config::P4SCORES[r],
                [invScale   = 1.0 / config.moveScoreScale,
                 bias       = config.moveScoreBias,
                 randomInit = config.randomMoveScoreInit,
                 &rand,
                 &prng](const Pattern4Score &p4score, size_t offset) {
                    Float score = (Score)p4score[offset];
                    return TuneParam(randomInit ? rand(prng) : (score - bias) * invScale);
                },
                [scoreMin = (Float)config.moveScoreMin,
                 scoreMax = (Float)config.moveScoreMax,
                 scale    = config.moveScoreScale,
                 bias =
                     config.moveScoreBias](Pattern4Score &p4score, size_t offset, TuneParam param) {
                    Float score     = param * scale + bias;
                    p4score[offset] = (Score)std::clamp(score, scoreMin, scoreMax);
                });
        }
    }

    MESSAGEL(tuneParams.size() << " parameters initialized.");
}

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

    MESSAGEL(tuneParams.size() << " parameters saved.");
}

/// initTuneEntries() inits tuneEntries from dataEntry read from datasets.
/// DataEntry that does not satisfy a certain condition will be skipped.
void Tuner::initTuneEntries(std::vector<TuneEntry> &tuneEntries, class Dataset &dataset)
{
    tuneEntries.clear();

    std::vector<std::future<void>> tuneEntryJobs;
    std::vector<DataEntry>         dataEntries;
    std::mutex                     tuneEntriesMutex;

    // Read dataset and convert to TuneEntry vectors parallelly
    size_t totalEntriesRead = 0;
    while (totalEntriesRead < config.maxTuneEntries) {
        // Read raw data entries from dataset
        dataEntries.reserve(config.batchSize);
        for (size_t i = 0; i < config.batchSize; i++) {
            DataEntry dataEntry;
            if (!dataset.next(&dataEntry))
                break;
            dataEntries.push_back(std::move(dataEntry));
        }

        // Stop reading loop when there is no more data entry
        if (dataEntries.empty())
            break;
        else
            totalEntriesRead += dataEntries.size();

        // Transform batched raw data entry to batched tune entry in parallel
        ThreadPool.detach_task(
            [this, &tuneEntries, &tuneEntriesMutex, data = std::move(dataEntries)]() {
                std::unordered_map<int, Board> boardObjectCache;
                std::vector<TuneEntry>         entries;
                entries.reserve(data.size());

                // Init tune entry from raw data entry
                for (size_t i = 0; i < data.size(); i++) {
                    // Filter out data entry that does not satisfy condition
                    if (!config.tuneRule[data[i].rule] || data[i].boardsize < config.boardSizeMin
                        || data[i].boardsize > config.boardSizeMax
                        || data[i].position.size() < config.minPly
                        || data[i].position.size() + config.minPlyBeforeFull
                               > int(data[i].boardsize) * int(data[i].boardsize))
                        continue;

                    entries.emplace_back(data[i], *this, std::move(boardObjectCache));
                }

                {
                    std::lock_guard<std::mutex> lock(tuneEntriesMutex);
                    tuneEntries.insert(tuneEntries.end(),
                                       std::make_move_iterator(entries.begin()),
                                       std::make_move_iterator(entries.end()));
                }
            });

        dataEntries.clear();
    }

    MESSAGEL("Read " << totalEntriesRead << " tune entries from dataset, initializing...");

    // Collect tune Entries from all jobs
    while (!ThreadPool.wait_for(std::chrono::seconds(10)))
        MESSAGEL(tuneEntries.size() << " tune entries are initialized...");

    tuneEntries.shrink_to_fit();
    MESSAGEL(tuneEntries.size() << " tune entries initialized.");

    // Shuffle tune entries if needed
    if (config.shuffleTuneEntries) {
        MESSAGEL("Shuffling tune entries...");

        PRNG prng;
        std::shuffle(tuneEntries.begin(), tuneEntries.end(), prng);
    }
}

/// searchOptimalInvScalingFactor() searches the optimal K in formula:
/// sigma = 1 + (1 / exp(-Eval * K)). by brute-forcely steps through the whole
/// scaling factor space for many iterations and finds the point that minimize
/// error of the current static evaluation and win rate in target tune entries.
Float Tuner::searchOptimalInvScalingFactor() const
{
    assert(config.nStepsPerIteration);

    Float startK = 1.0 / Float(config.scalingFactorMin);
    Float endK   = 1.0 / Float(config.scalingFactorMax);
    Float stepK  = (endK - startK) / Float(config.nStepsPerIteration);
    Float bestK  = 0;

    for (int iter = 1; iter <= config.nIterations; iter++) {
        Float k        = startK;
        Float bestLoss = std::numeric_limits<Float>::max();

        for (int i = 0; i < config.nStepsPerIteration; i++) {
            Float loss = computeEvaluationLoss<false>(k, false);

            if (loss < bestLoss) {
                bestLoss = loss;
                bestK    = k;
            }

            k += stepK;
        }

        MESSAGEL("Iteration " << iter << " | K " << bestK << " | Loss " << bestLoss);

        startK = bestK - stepK;
        endK   = bestK + stepK;
        stepK  = stepK * 2 / Float(config.nStepsPerIteration);
    }

    MESSAGEL("Optimal inv scaling factor K = " << bestK << " after " << config.nIterations
                                               << " iteration.");

    return bestK;
}

/// computeEvaluationLoss() computes loss between the current tuned/static
/// evaluation and target win rate in all tune entries using the given K.
template <bool UseTunedEval>
Float Tuner::computeEvaluationLoss(Float K, bool validation) const
{
    const std::vector<TuneEntry> &entries = validation ? valTuneEntries : trainTuneEntries;

    if (entries.empty())
        return Float(0.0);

    return parallel_transform_reduce(
               entries.begin(),
               entries.end(),
               Float(0.0),
               std::plus<Float>(),
               [this, K](const TuneEntry &e) {
                   return e.computeEvalLoss<UseTunedEval>(tuneParams, K, config.lossType);
               })
           / Float(entries.size());
}

/// computeMoveScoreLoss() computes loss of current move scores between
/// the target best move in all tune entries.
Float Tuner::computeMoveScoreLoss(bool validation) const
{
    const std::vector<TuneEntry> &entries = validation ? valTuneEntries : trainTuneEntries;

    if (entries.empty())
        return Float(0.0);

    return parallel_transform_reduce(entries.begin(),
                                     entries.end(),
                                     Float(0.0),
                                     std::plus<Float>(),
                                     [this](const TuneEntry &e) {
                                         return e.computeMoveScoreLoss(tuneParams,
                                                                       config.moveScoreLossGamma);
                                     })
           / Float(entries.size());
}

/// computeGradients() computes gradients of all parameters used in one tune
/// entries batch and accumulates them into gradients vector. These gradients
/// then will be used to tune the parameters with a gradient descent optimizer.
void Tuner::computeGradientBatch(std::vector<Float> &grads, Float K, size_t batchIdx)
{
    assert(grads.size() == tuneParams.size());
    typedef std::vector<TuneEntry>::const_iterator TuneEntryIterator;

    // Calculate size of each job and number of total jobs in parallel
    constexpr size_t numPartitions = 64;
    size_t           jobSize       = config.batchSize / numPartitions;
    size_t           numJobs       = config.batchSize / jobSize;
    size_t           undivided     = config.batchSize - numJobs * jobSize;

    std::vector<std::future<std::vector<Float>>> gradJobs;
    TuneEntryIterator batchBegin = trainTuneEntries.cbegin() + batchIdx * config.batchSize;

    for (size_t jobIdx = 0; jobIdx < numJobs + bool(undivided); jobIdx++) {
        // Get range of tune entries for this job
        TuneEntryIterator jobBegin = batchBegin + jobIdx * jobSize;
        // Deals with the rest tune entries
        TuneEntryIterator jobEnd = jobBegin + (jobIdx == numJobs ? undivided : jobSize);

        // Accumulate local gradient asynchronously
        auto job = ThreadPool.submit_task([this, K, jobBegin, jobEnd] {
            std::vector<Float> localGrads(tuneParams.size(), Float(0.0));

            for (TuneEntryIterator e = jobBegin; e != jobEnd; e++) {
                e->computeEvalGradient(localGrads, tuneParams, K, config.lossType);
                e->computeMoveScoreGradient(localGrads, tuneParams, config.moveScoreLossGamma);
            }

            // Scale gradient according to batch size
            Float S = 1 / Float(config.batchSize);
            for (Float &gradient : localGrads)
                gradient *= S;

            return localGrads;
        });
        gradJobs.push_back(std::move(job));
    }

    // Accumulate gradients from all jobs
    for (auto &job : gradJobs) {
        std::vector<Float> localGrads = job.get();
        assert(grads.size() == localGrads.size());

        for (size_t i = 0; i < grads.size(); i++)
            grads[i] += localGrads[i];
    }
}

/// addParams() adds a continous range of params in config to tuneParams
void Tuner::addParams(void         *address,
                      size_t        numElems,
                      uint32_t      elemSize,
                      uint32_t      paramPerElem,
                      ParamGetter<> getter,
                      ParamSetter<> setter)
{
    assert(paramPerElem > 0);
    size_t baseIndex = tuneParams.size();

    // Insert new record to correct position in sorted syncRecords
    auto record =
        *syncRecords.insert(std::upper_bound(syncRecords.begin(), syncRecords.end(), address),
                            ParamsSyncRecord {baseIndex,
                                              numElems,
                                              elemSize,
                                              paramPerElem,
                                              address,
                                              std::move(getter),
                                              std::move(setter)});

    // Init parameters from getter and add them to tuneParams
    size_t numParams = numElems * paramPerElem;
    tuneParams.reserve(baseIndex + numParams);
    for (size_t i = 0; i < numParams; i++) {
        TuneParam param = record.getter(record[i / paramPerElem], i % paramPerElem);
        assert(!std::isnan(param));
        tuneParams.emplace_back(param);
    }
}

/// paramIndex() finds tuneParams index according to address of its config value
size_t Tuner::paramIndex(void *addr, size_t offset) const
{
    assert(!syncRecords.empty());

    // Find sync record using binary search
    const ParamsSyncRecord &record =
        *(std::upper_bound(syncRecords.begin(), syncRecords.end(), addr) - 1);
    assert(record[0] <= addr && addr < record[record.numElems]);

    // Compare distance in bytes and convert it to TuneParam index
    size_t bytesDist = static_cast<char *>(addr) - static_cast<char *>(record.address);
    size_t elemIndex = bytesDist / record.elemSize;
    assert(elemIndex < record.numElems);
    assert(record.baseIndex + elemIndex < tuneParams.size());
    assert(offset < record.paramPerElem);
    return record.baseIndex + elemIndex * record.paramPerElem + offset;
}

}  // namespace Tuning
