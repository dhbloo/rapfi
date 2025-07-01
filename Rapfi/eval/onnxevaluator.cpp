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

#include "onnxevaluator.h"

#include "../core/utils.h"
#include "../game/board.h"
#include "weightloader.h"

#include <onnxruntime_cxx_api.h>

namespace Evaluation::onnx {

/// A simple classification on onnx execution provider and device id.
enum OnnxDevice {
    DEFAULT_DEV = 0,
    CPU_ST      = 1,
    CPU_MT      = 2,
    CUDA_0      = 3,
    CUDA_MAX    = CUDA_0 + 127,
    TRT_0       = CUDA_MAX + 1,
    TRT_MAX     = TRT_0 + 127,
};

/// Parse a onnx device enum from a device string.
OnnxDevice parseDeviceString(std::string device)
{
    if (device.empty())
        return DEFAULT_DEV;

    upperInplace(device);
    if (device == "CPU" || device == "CPU-SINGLETHREAD")
        return CPU_ST;
    else if (device == "CPU-MULTITHREAD")
        return CPU_MT;
    else if (device == "CUDA")
        return CUDA_0;
    else if (device.rfind("CUDA:", 0) == 0) {
        int cudaDeviceId = std::stoi(device.substr(5));
        if (CUDA_0 + cudaDeviceId <= CUDA_MAX)
            return OnnxDevice(CUDA_0 + cudaDeviceId);
        else
            throw std::runtime_error("out of range cuda device " + device);
    }
    else if (device == "TRT")
        return TRT_0;
    else if (device.rfind("TRT:", 0) == 0) {
        int trtDeviceId = std::stoi(device.substr(4));
        if (TRT_0 + trtDeviceId <= TRT_MAX)
            return OnnxDevice(TRT_0 + trtDeviceId);
        else
            throw std::runtime_error("out of range trt device " + device);
    }
    else
        throw std::runtime_error("unknown device " + device);
}

/// Get the device string of a onnx device enum.
std::string deviceString(OnnxDevice device)
{
    if (device == CPU_ST)
        return "cpu-singlethread";
    else if (device == CPU_MT)
        return "cpu-multithread";
    else if (CUDA_0 <= device && device <= CUDA_MAX)
        return "cuda:" + std::to_string(device - CUDA_0);
    else if (TRT_0 <= device && device <= TRT_MAX)
        return "trt:" + std::to_string(device - TRT_0);
    else
        return "unknown";
}

/// The version of the onnx model input/output specification.
enum OnnxModelIOVersion : uint16_t {
    VERSION_START  = 1,
    RAPFI_MODEL_V1 = 1,
    VERSION_END,
};

/// Get the default device of this machine.
OnnxDevice getDefaultDevice()
{
    return CPU_ST;
}

/// Arguments for creating a onnx model instance.
struct OnnxModelArguments
{
    OnnxDevice device;

    bool operator==(const OnnxModelArguments &other) const { return device == other.device; }
};

/// Warpper interface for an onnx session instance.
class OnnxModel
{
public:
    OnnxModel(std::istream &istream, OnnxModelArguments args)
    {
        setupSessionOptions(args.device);

        // Read model data from stream to memory
        std::vector<char> modelData {
            std::istreambuf_iterator<char> {istream},
            std::istreambuf_iterator<char> {},
        };

        session = Ort::Session(getGlobalEnvInstance(),
                               modelData.data(),
                               modelData.size(),
                               sessionOptions);

        auto    metainfo         = session.GetModelMetadata();
        int64_t modelVersionMask = metainfo.GetVersion();
        int64_t modelVersion     = (modelVersionMask >> 48) & 0xFFFF;
        modelRuleMask            = (modelVersionMask >> 32) & 0xFFFF;
        modelBoardSizeMask       = modelVersionMask & 0xFFFFFFFF;

        if (modelVersion >= VERSION_START && modelVersion < VERSION_END)
            modelIOVersion = static_cast<OnnxModelIOVersion>(modelVersion);
        else
            throw std::runtime_error("unknown onnx model IO version "
                                     + std::to_string(modelVersion));

        allocator = Ort::AllocatorWithDefaultOptions {};
    }

    bool supportRule(Rule rule) const
    {
        switch (rule) {
        case FREESTYLE: return modelRuleMask & 0x1;
        case STANDARD: return modelRuleMask & 0x2;
        case RENJU: return modelRuleMask & 0x4;
        default: return false;
        }
    }

    bool supportBoardSize(int boardSize) const
    {
        return modelBoardSizeMask & (1 << (boardSize - 1));
    }

    OnnxModelIOVersion getIOVersion() const { return modelIOVersion; }

    std::vector<std::string> getInputNames() const
    {
        std::vector<std::string> inputNames;
        size_t                   numInputs = session.GetInputCount();
        for (size_t i = 0; i < numInputs; i++) {
            auto inputName = session.GetInputNameAllocated(i, allocator);
            inputNames.emplace_back(inputName.get());
        }
        return inputNames;
    }

    std::vector<std::string> getOutputNames() const
    {
        std::vector<std::string> outputNames;
        size_t                   numOutputs = session.GetOutputCount();
        for (size_t i = 0; i < numOutputs; i++) {
            auto outputName = session.GetOutputNameAllocated(i, allocator);
            outputNames.emplace_back(outputName.get());
        }
        return outputNames;
    }

    void run(const std::vector<const char *> &inputNames,
             const std::vector<Ort::Value>   &inputValues,
             const std::vector<const char *> &outputNames,
             std::vector<Ort::Value>         &outputValues)
    {
        session.Run(runOptions,
                    inputNames.data(),
                    inputValues.data(),
                    inputNames.size(),
                    outputNames.data(),
                    outputValues.data(),
                    outputNames.size());
    }

private:
    void setupSessionOptions(OnnxDevice device)
    {
        if (device == CPU_ST) {
            sessionOptions.SetIntraOpNumThreads(1);
            sessionOptions.SetInterOpNumThreads(1);
            return;
        }
        else if (device == CPU_MT) {
            sessionOptions.SetIntraOpNumThreads(0);
            sessionOptions.SetInterOpNumThreads(0);
            sessionOptions.AddConfigEntry("session.intra_op.allow_spinning", "0");
            sessionOptions.AddConfigEntry("session.inter_op.allow_spinning", "0");
            return;
        }
        else if (CUDA_0 <= device && device <= CUDA_MAX) {
#ifdef USE_ORT_GPU_EP
            OrtCUDAProviderOptions cudaOptions;
            cudaOptions.device_id                 = device - CUDA_0;
            cudaOptions.arena_extend_strategy     = 0;
            cudaOptions.cudnn_conv_algo_search    = OrtCudnnConvAlgoSearchExhaustive;
            cudaOptions.do_copy_in_default_stream = 1;
            sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);
#else
            throw std::runtime_error("Onnx cuda device is not supported in this build");
#endif
        }
        else if (TRT_0 <= device && device <= TRT_MAX) {
            throw std::runtime_error("Onnx tensorrt device is not supported in this build");
        }
        else {
            throw std::runtime_error("unsupported Onnx device " + std::to_string(device));
        }
    }

    static Ort::Env &getGlobalEnvInstance()
    {
        struct OrtEnvWarpper : Ort::Env
        {
            enum {
#ifdef NDEBUG
                DefaultLoggingLevel = ORT_LOGGING_LEVEL_WARNING
#else
                DefaultLoggingLevel = ORT_LOGGING_LEVEL_WARNING
#endif
            };

            OrtEnvWarpper()
                : Ort::Env(static_cast<OrtLoggingLevel>(DefaultLoggingLevel),
                           "",
                           ortLoggingFunction,
                           nullptr)
            {
                MESSAGEL("Initialized onnx runtime " << Ort::GetVersionString());
            }

            static void ortLoggingFunction(void           *param,
                                           OrtLoggingLevel severity,
                                           const char     *category,
                                           const char     *logid,
                                           const char     *code_location,
                                           const char     *message)
            {
                const char *SeverityName[] = {"verbose", "info", "warning", "error", "fatal"};
                MESSAGEL("Ort(" << category << ") " << SeverityName[severity] << " at ["
                                << code_location << "]: " << message);
            }
        };
        static OrtEnvWarpper env;
        return env;
    }

    Ort::Session                     session {nullptr};
    Ort::SessionOptions              sessionOptions;
    Ort::AllocatorWithDefaultOptions allocator;
    Ort::RunOptions                  runOptions;
    int32_t                          modelBoardSizeMask;
    int16_t                          modelRuleMask;
    OnnxModelIOVersion               modelIOVersion;
};

class OnnxRapfiModelV1 : public OnnxAccumulator
{
public:
    OnnxRapfiModelV1(int boardSize, Color side)
        : boardWidth {boardSize}
        , boardHeight {boardSize}
        , sideToMove {side}
        , outputDirty {true}
    {
        if (boardSize > MAX_BOARD_SIZE)
            throw std::runtime_error("board size " + std::to_string(boardSize) + " is too large");

        boardInput.resize(BatchSize * 2 * boardHeight * boardWidth);
        globalInput.resize(BatchSize * 1);
        valueOutput.resize(BatchSize * 3);
        policyOutput.resize(BatchSize * boardHeight * boardWidth);
    }

    void init(OnnxModel &model) override
    {
        auto memoryInfo = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

        inputTensors.clear();
        inputNames.clear();
        for (const auto &name : model.getInputNames()) {
            if (name == "board_input") {
                const int64_t boardInputShape[] = {BatchSize,
                                                   2,
                                                   (int64_t)boardHeight,
                                                   (int64_t)boardWidth};

                inputTensors.push_back(
                    Ort::Value::CreateTensor<int8_t>(memoryInfo,
                                                     boardInput.data(),
                                                     boardInput.size(),
                                                     boardInputShape,
                                                     arraySize(boardInputShape)));
                inputNames.push_back("board_input");
            }
            else if (name == "global_input") {
                const int64_t stmInputShape[] = {BatchSize, 1};

                inputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo,
                                                                       globalInput.data(),
                                                                       globalInput.size(),
                                                                       stmInputShape,
                                                                       arraySize(stmInputShape)));
                inputNames.push_back("global_input");
            }
            else {
                throw std::runtime_error("unknown input name in onnx model (rapfi_model_v1): "
                                         + name);
            }
        }

        outputTensors.clear();
        outputNames.clear();
        std::vector<std::string> modelOutputNames = model.getOutputNames();
        if (contains(modelOutputNames, "value")) {
            const int64_t valueOutputShape[] = {BatchSize, 3};

            outputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo,
                                                                    valueOutput.data(),
                                                                    valueOutput.size(),
                                                                    valueOutputShape,
                                                                    arraySize(valueOutputShape)));
            outputNames.push_back("value");
        }
        else {
            throw std::runtime_error("no 'value' output in onnx model (rapfi_model_v1)");
        }
        if (contains(modelOutputNames, "policy")) {
            const int64_t policyOutputShape[] = {BatchSize,
                                                 (int64_t)boardHeight,
                                                 (int64_t)boardWidth};

            outputTensors.push_back(Ort::Value::CreateTensor<float>(memoryInfo,
                                                                    policyOutput.data(),
                                                                    policyOutput.size(),
                                                                    policyOutputShape,
                                                                    arraySize(policyOutputShape)));
            outputNames.push_back("policy");
        }
        else {
            throw std::runtime_error("no 'policy' output in onnx model (rapfi_model_v1)");
        }
    }

    void clear(OnnxModel &model) override
    {
        std::fill(boardInput.begin(), boardInput.end(), 0);
        globalInput[0] = sideToMove == BLACK ? -1.0f : 1.0f;
        outputDirty    = true;
    }

    void move(OnnxModel &model, Color color, int x, int y) override
    {
        const size_t channelStride = boardHeight * boardWidth;
        const size_t posOffset     = y * boardWidth + x;

        boardInput[0 * channelStride + posOffset] = color == sideToMove;   // self plane
        boardInput[1 * channelStride + posOffset] = color == ~sideToMove;  // oppo plane

        outputDirty = true;
    }

    void undo(OnnxModel &model, Color color, int x, int y) override
    {
        const size_t channelStride = boardHeight * boardWidth;
        const size_t posOffset     = y * boardWidth + x;

        boardInput[0 * channelStride + posOffset] = 0;  // self plane
        boardInput[1 * channelStride + posOffset] = 0;  // oppo plane

        outputDirty = true;
    }

    ValueType evaluateValue(OnnxModel &model) override
    {
        if (outputDirty)
            runInference(model);

        return ValueType(valueOutput[0], valueOutput[1], valueOutput[2], true);
    }

    void evaluatePolicy(OnnxModel &model, PolicyBuffer &policyBuffer) override
    {
        if (outputDirty)
            runInference(model);

        // Copy policy output to policy buffer
        for (int i = 0; i < boardHeight * boardWidth; i++)
            policyBuffer(i) = policyOutput[i];
    }

private:
    static constexpr int64_t     BatchSize     = 1;
    static constexpr const char *InputNames[]  = {"board_input", "global_input"};
    static constexpr const char *OutputNames[] = {"value", "policy"};

    void runInference(OnnxModel &model)
    {
        model.run(inputNames, inputTensors, outputNames, outputTensors);
        outputDirty = false;
    }

    const int           boardWidth, boardHeight;
    const Color         sideToMove;
    bool                outputDirty;
    std::vector<int8_t> boardInput;
    std::vector<float>  globalInput;
    std::vector<float>  valueOutput;
    std::vector<float>  policyOutput;

    std::vector<Ort::Value>   inputTensors;
    std::vector<const char *> inputNames;
    std::vector<Ort::Value>   outputTensors;
    std::vector<const char *> outputNames;
};

struct OnnxModelLoader : WeightLoader<OnnxModel, OnnxModelArguments>
{
    OnnxModelLoader(int boardSize, Rule rule, std::filesystem::path onnxModelPath)
        : boardSize(boardSize)
        , rule(rule)
        , onnxModelPath(onnxModelPath)
    {}

    LargePagePtr<OnnxModel> load(std::istream &is, OnnxModelArguments args) override
    {
        try {
            auto ptr = make_unique_large_page<OnnxModel>(is, args);
            if (!ptr->supportRule(rule))
                return nullptr;
            if (!ptr->supportBoardSize(boardSize))
                return nullptr;

            MESSAGEL("Initialized onnx model " << pathToConsoleString(onnxModelPath)
                                               << " on device: " << deviceString(args.device));
            return ptr;
        }
        catch (const std::exception &e) {
            ERRORL("Failed to create onnx model: " << e.what());
            return nullptr;
        }
    }

private:
    int                   boardSize;
    Rule                  rule;
    std::filesystem::path onnxModelPath;
};

static WeightRegistry<OnnxModelLoader> OnnxModelRegistry;

std::unique_ptr<OnnxAccumulator>
makeOnnxAccumulator(OnnxModel &model, int boardSize, Rule rule, Color side)
{
    std::unique_ptr<OnnxAccumulator> ptr;
    switch (model.getIOVersion()) {
    case RAPFI_MODEL_V1: ptr = std::make_unique<OnnxRapfiModelV1>(boardSize, side); break;
    default:
        throw std::runtime_error("unsupported onnx model IO version "
                                 + std::to_string(model.getIOVersion()));
    }

    ptr->init(model);
    return ptr;
}

}  // namespace Evaluation::onnx

namespace Evaluation::onnx {

OnnxEvaluator::OnnxEvaluator(int                   boardSize,
                             Rule                  rule,
                             std::filesystem::path onnxModelPath,
                             std::string           device)
    : Evaluator(boardSize, rule)
{
    if (!std::filesystem::exists(onnxModelPath))
        throw std::runtime_error("Onnx model file not found: "
                                 + pathToConsoleString(onnxModelPath));

    OnnxModelArguments args;
    args.device = parseDeviceString(device);
    if (args.device == DEFAULT_DEV)
        args.device = getDefaultDevice();

    OnnxModelLoader loader {boardSize, rule, onnxModelPath};
    model =
        OnnxModelRegistry.loadWeightFromFile(loader, onnxModelPath, Numa::DefaultNumaNodeId, args);
    if (!model)
        throw std::runtime_error("Failed to load onnx model from "
                                 + pathToConsoleString(onnxModelPath));

    accumulator[BLACK] = makeOnnxAccumulator(*model, boardSize, rule, BLACK);
    accumulator[WHITE] = makeOnnxAccumulator(*model, boardSize, rule, WHITE);
}

OnnxEvaluator::~OnnxEvaluator()
{
    OnnxModelRegistry.unloadWeight(model);
}

void OnnxEvaluator::initEmptyBoard()
{
    accumulator[BLACK]->clear(*model);
    accumulator[WHITE]->clear(*model);
}

void OnnxEvaluator::beforeMove(const Board &board, Pos pos)
{
    for (Color c : {BLACK, WHITE}) {
        accumulator[c]->move(*model, board.sideToMove(), pos.x(), pos.y());
    }
}

void OnnxEvaluator::afterUndo(const Board &board, Pos pos)
{
    for (Color c : {BLACK, WHITE}) {
        accumulator[c]->undo(*model, board.sideToMove(), pos.x(), pos.y());
    }
}

ValueType OnnxEvaluator::evaluateValue(const Board &board, AccLevel level)
{
    Color self = board.sideToMove();
    return accumulator[self]->evaluateValue(*model);
}

void OnnxEvaluator::evaluatePolicy(const Board &board, PolicyBuffer &policyBuffer, AccLevel level)
{
    Color self = board.sideToMove();
    accumulator[self]->evaluatePolicy(*model, policyBuffer);
}

}  // namespace Evaluation::onnx
