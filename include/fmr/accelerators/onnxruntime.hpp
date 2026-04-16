#pragma once

#include <any>
#include <string>
#include <memory>
#include <utility>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <exception>
#include <filesystem>
#include <unordered_map>

#include <onnxruntime_c_api.h>
#include <onnxruntime_cxx_api.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <spdlog/common.h>
#include <spdlog/logger.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <fmr/accelerators/accelerator.hpp>
#include <fmr/core/image.hpp>

namespace fmr {

class onnxruntime : public accelerator {
public:
    explicit onnxruntime(std::shared_ptr<Ort::Env> env = nullptr, std::shared_ptr<spdlog::logger> logger = nullptr);
    virtual ~onnxruntime() = default;
    bool load_model(const std::string& modelPath, const std::any& backendConfig = {}) override;
    bool predict_raw(const std::vector<tensor>& inputs, std::vector<tensor>& outputs, const std::any& runOptions = {}) override;
    virtual std::vector<tensor> input_details() const override;
    virtual std::vector<tensor> output_details() const override;
    void print_metadata() const override;

    void set_tensor_memory_info(Ort::MemoryInfo newInfo);
    Ort::ConstMemoryInfo tensor_memory_info() const;
    const std::vector<Ort::Value> &last_output_tensors() const;

    static size_t get_type_size(tensor_data_type type);

protected:
    std::vector<const char *> get_output_names(const std::vector<tensor> &outputs);
    ONNXTensorElementDataType map_to_ort_type(tensor_data_type type);
    tensor_data_type map_to_fmr_type(ONNXTensorElementDataType type);

private:
    std::shared_ptr<Ort::Env> m_env;
    std::unique_ptr<Ort::Session> m_session;
    Ort::MemoryInfo m_tensor_memory_info;
    OrtAllocator *m_cpu_allocator = nullptr;
    std::shared_ptr<Ort::RunOptions> m_run_options;
    std::string m_model_path;

    std::vector<tensor> m_inputs;
    std::vector<tensor> m_outputs;
    std::vector<Ort::Value> m_last_output_tensors;
};

// Definitions

inline onnxruntime::onnxruntime(std::shared_ptr<Ort::Env> env, std::shared_ptr<spdlog::logger> logger)
    : m_env(env)
    , m_tensor_memory_info(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault))
{
    if (!logger) {
        logger = spdlog::default_logger()->clone("fmr.accelators.ort");
        logger->set_level(spdlog::level::debug);
    }
    set_logger(logger);

    if (!m_env)
        m_env = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ONNX_Inference");

    // Allocator just for the strings and metadata
    m_cpu_allocator = Ort::AllocatorWithDefaultOptions();
}

inline bool onnxruntime::load_model(const std::string& modelPath, const std::any& backendConfig)
{
    if (!std::filesystem::exists(modelPath)
        || modelPath.find_last_of(".onnx") == std::string::npos) {

        logger()->critical("Model does not exist as \"{}\" or has invalid extension.", modelPath);
        return false;
    }

    try {
        std::shared_ptr<Ort::SessionOptions> session_options = std::make_shared<Ort::SessionOptions>();

        if (backendConfig.has_value() && backendConfig.type() == typeid(std::shared_ptr<Ort::SessionOptions>)) {
            session_options = std::any_cast<std::shared_ptr<Ort::SessionOptions>>(backendConfig);
        } else {
            session_options->SetInterOpNumThreads(1);
            session_options->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);
        }

        m_model_path = modelPath;
        #ifdef WIN32
            std::wstring model_path(modelPath.begin(), modelPath.end());
        #else
            std::string model_path = modelPath;;
        #endif

        m_session = std::make_unique<Ort::Session>(*m_env, model_path.c_str(), *session_options);

        // Populate the inputs/outputs
        const auto input_names = m_session->GetInputNames();
        for (size_t i = 0; i < m_session->GetInputCount(); ++i) {
            auto info = m_session->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo();

            tensor t;
            t.name = input_names.at(i);
            t.shape = info.GetShape();
            t.type = map_to_fmr_type(info.GetElementType());
            m_inputs.emplace_back(t);
        }

        const auto output_names = m_session->GetOutputNames();
        for (size_t i = 0; i < m_session->GetOutputCount(); ++i) {
            auto info = m_session->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo();

            tensor t;
            t.name = output_names.at(i);
            t.shape = info.GetShape();
            t.type = map_to_fmr_type(info.GetElementType());
            m_outputs.emplace_back(t);
        }

        // Metadata
        std::unordered_map<std::string, std::string> model_metadata;
        const auto metadata = m_session->GetModelMetadata();
        const auto keys = metadata.GetCustomMetadataMapKeysAllocated(m_cpu_allocator);
        for (const auto &key : keys) {
            model_metadata[key.get()] = metadata.LookupCustomMetadataMapAllocated(key.get(), m_cpu_allocator).get();
        }
        set_model_metadata(model_metadata);
    } catch (const std::exception &e) {
        logger()->critical("Failed to load model, reason {}, path {}", e.what(), modelPath);
        return false;
    }

    return true;
}

inline bool onnxruntime::predict_raw(const std::vector<tensor>& inputs, std::vector<tensor>& outputs, const std::any& runOptions)
{
    try {
        std::shared_ptr<Ort::RunOptions> run_options = std::make_shared<Ort::RunOptions>();

        if (runOptions.has_value() && runOptions.type() == typeid(std::shared_ptr<Ort::RunOptions>)) {
            run_options = std::any_cast<std::shared_ptr<Ort::RunOptions>>(runOptions);
        }

        std::vector<Ort::Value> input_tensors;
        std::vector<const char *> input_names;
        for (const auto& tensor : inputs) {
            input_tensors.push_back(
                Ort::Value::CreateTensor(
                    m_tensor_memory_info,
                    tensor.data,
                    tensor.size_bytes,
                    tensor.shape.data(),
                    tensor.shape.size(),
                    map_to_ort_type(tensor.type)
                )
            );

            input_names.push_back(tensor.name.c_str());
        }

        std::vector<const char *> output_names = get_output_names(m_outputs);
        m_last_output_tensors = m_session->Run(*run_options, input_names.data(), input_tensors.data(), input_tensors.size(), output_names.data(), output_names.size());
        // Map output tensors
        outputs.clear();
        for (size_t i = 0; i < m_last_output_tensors.size(); ++i) {
            auto &ort_tensor = m_last_output_tensors.at(i);
            const auto &info = ort_tensor.GetTensorTypeAndShapeInfo();

            tensor t;
            t.name = output_names.at(i);
            t.shape = info.GetShape();
            t.type = map_to_fmr_type(info.GetElementType());
            t.data = ort_tensor.GetTensorMutableRawData();
            t.device = static_cast<int>(ort_tensor.GetTensorMemoryInfo().GetDeviceType());
            t.size_bytes = vec_product(t.shape) * get_type_size(t.type);

            outputs.emplace_back(t);
        }

    } catch (const std::exception &e) {
        logger()->critical("Failed inference, {}", e.what());
        return false;
    }

    return true;
}

inline std::vector<tensor> onnxruntime::input_details() const
{
    return m_inputs;
}

inline std::vector<tensor> onnxruntime::output_details() const
{
    return m_outputs;
}

inline void onnxruntime::print_metadata() const
{
    const Ort::ModelMetadata &metadata = m_session->GetModelMetadata();
    logger()->debug("Model metadata:");
    logger()->debug("  File: {}", m_model_path);
    logger()->debug("  Graph Name: {}", metadata.GetGraphNameAllocated(m_cpu_allocator).get());
    logger()->debug("  Graph Description: {}", metadata.GetGraphDescriptionAllocated(m_cpu_allocator).get());
    logger()->debug("  Description: {}", metadata.GetDescriptionAllocated(m_cpu_allocator).get());
    logger()->debug("  Domain: {}", metadata.GetDomainAllocated(m_cpu_allocator).get());
    logger()->debug("  Producer: {}", metadata.GetProducerNameAllocated(m_cpu_allocator).get());
    logger()->debug("  Version: {}", metadata.GetVersion());

    logger()->debug("  Custom Metadata:");
    const auto metadata_metadata = model_metadata();
    for (const auto &[key, val] : metadata_metadata) {
        logger()->debug("    {}: {}", key, val);
    }

    logger()->debug("  Inputs:");
    for (size_t i = 0; i < m_inputs.size(); ++i) {
        logger()->debug("    Name: {}", m_inputs.at(i).name);
        // logger()->debug("    Type: {}", m_inputs.at(i).type);
        logger()->debug("    Shape: {}", m_inputs.at(i).shape);
    }

    logger()->debug("  Outputs:");
    for (size_t i = 0; i < m_outputs.size(); ++i) {
        logger()->debug("    Name: {}", m_outputs.at(i).name);

        // My bad man, but this is the easiest way add a \n at the end.
        if (i == m_outputs.size() - 1)
            logger()->debug("    Shape: {}\n", m_outputs.at(i).shape);
        else
            logger()->debug("    Shape: {}", m_outputs.at(i).shape);
    }
}

inline void onnxruntime::set_tensor_memory_info(Ort::MemoryInfo newInfo)
{
    m_tensor_memory_info = std::move(newInfo);
}

inline Ort::ConstMemoryInfo onnxruntime::tensor_memory_info() const
{
    return m_tensor_memory_info.GetConst();
}

inline const std::vector<Ort::Value> &onnxruntime::last_output_tensors() const
{
    return m_last_output_tensors;
}

inline size_t onnxruntime::get_type_size(tensor_data_type type)
{
    switch (type) {
        case tensor_data_type::FLOAT32: return sizeof(float);
        case tensor_data_type::FLOAT16: return sizeof(Ort::Float16_t);
        case tensor_data_type::INT8: return sizeof(int8_t);
        case tensor_data_type::UINT8: return sizeof(uint8_t);
        case tensor_data_type::INT32: return sizeof(int32_t);
        case tensor_data_type::INT64: return sizeof(int64_t);
        default: return 0;
    }

    return 0;
}

inline std::vector<const char *> onnxruntime::get_output_names(const std::vector<tensor> &outputs)
{
    std::vector<const char *> names;
    for (const auto &t : outputs) {
        names.push_back(t.name.c_str());
    }

    return names;
}

inline ONNXTensorElementDataType onnxruntime::map_to_ort_type(tensor_data_type type)
{
    switch (type) {
        case tensor_data_type::FLOAT32: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT;
        case tensor_data_type::FLOAT16: return ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16;
        case tensor_data_type::INT8:    return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8;
        case tensor_data_type::UINT8:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8;
        case tensor_data_type::INT32:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32;
        case tensor_data_type::INT64:   return ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64;
        default: return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
    }

    return ONNX_TENSOR_ELEMENT_DATA_TYPE_UNDEFINED;
}

inline tensor_data_type onnxruntime::map_to_fmr_type(ONNXTensorElementDataType type)
{
    switch (type) {
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT:   return tensor_data_type::FLOAT32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_FLOAT16: return tensor_data_type::FLOAT16;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT8:    return tensor_data_type::INT8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_UINT8:   return tensor_data_type::UINT8;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT32:   return tensor_data_type::INT32;
        case ONNX_TENSOR_ELEMENT_DATA_TYPE_INT64:   return tensor_data_type::INT64;
        default: return tensor_data_type::UNKNOWN;
    }

    return tensor_data_type::UNKNOWN;
}

} // fmr