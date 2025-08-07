#include <filesystem>
#include <thread>

#include <fmt/format.h>
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>

#include <fmr/accelarators/onnxruntime.h>

namespace fmr {

inline std::shared_ptr<spdlog::logger> logger = spdlog::default_logger()->clone("fmr.accelators.ort");

onnxruntime::onnxruntime(
    const PredictorConfig *config,
    const std::shared_ptr<Ort::Env>& env,
    const std::shared_ptr<CustomOrtAllocator>& allocator,
    const std::shared_ptr<Ort::MemoryInfo>& memoryInfo)
{
    if (!config || !config->model.has_value())
        throw std::runtime_error("Invalid config, please provide a valid config with model info!");

    if (!m_env)
        m_env = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ONNX_Inferece");

    if (!m_memory_info)
        m_memory_info = std::make_shared<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

    std::string model_path = config->model->path.value_or("");
    if (!std::filesystem::exists(model_path)
        || model_path.find_last_of(".onnx") == std::string::npos)
        throw std::runtime_error("Model does not exist or has invalid extension. Please download a model first and try again!");


    try {
        Ort::SessionOptions sessionOptions;
        // TODO: This has to be decided by the user
        int intraop_threads = std::min(4, static_cast<int>(std::thread::hardware_concurrency()));
        sessionOptions.SetIntraOpNumThreads(intraop_threads);
        sessionOptions.SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_ALL);

        m_available_eps = Ort::GetAvailableProviders();

#if 1
        if (std::find(m_available_eps.begin(), m_available_eps.end(), "CUDAExecutionProvider")
            != m_available_eps.end()) {

            OrtCUDAProviderOptions cudaOptions;
            sessionOptions.AppendExecutionProvider_CUDA(cudaOptions);

            m_selected_eps.emplace_back("CUDAExecutionProvider");
        } else {
            logger->warn(fmt::format("Inference device {} not available", "CUDAExecutionProvider"));
        }
#endif

#if 1
        if (std::find(m_available_eps.begin(), m_available_eps.end(), "OpenVINOExecutionProvider")
            != m_available_eps.end()) {

            std::unordered_map<std::string, std::string> options;
            options["device_type"] = "AUTO:GPU,CPU";
            options["precision"] = "ACCURACY";
            options["num_of_threads"] = "4";
            sessionOptions.AppendExecutionProvider_OpenVINO_V2(options);

            m_selected_eps.emplace_back("OpenVINOExecutionProvider");
        }
        else {
            logger->warn(fmt::format("Inference device {} not available", "OpenVINOExecutionProvider"));
        }
#endif

        m_selected_eps.emplace_back("CPUExecutionProvider");

#ifdef _WIN32
        std::wstring modelPath(model_path.begin(), model_path.end());
#else
        std::string modelPath(model_path);
#endif
        m_session = Ort::Session(*m_env, modelPath.c_str(), sessionOptions);

        if (!m_allocator)
            m_allocator = std::make_shared<CustomOrtAllocator>(Ort::Allocator(m_session, *m_memory_info));

        // input nodes.
        int input_nodes_ = m_session.GetInputCount();
        std::vector<const char *> input_names_;
        std::vector<std::vector<int64_t>> input_shapes_;
        for (int i = 0; i < input_nodes_; ++i) {
            Ort::AllocatedStringPtr name = m_session.GetInputNameAllocated(i, m_allocator->get());
            m_input_names_alloc.emplace_back(std::move(name));
            input_names_.emplace_back(m_input_names_alloc.back().get());

            Ort::TypeInfo input_type_info = m_session.GetInputTypeInfo(i);
            input_shapes_.emplace_back(input_type_info.GetTensorTypeAndShapeInfo().GetShape());
        }

        set_input_nodes(input_nodes_);
        set_input_names(std::move(input_names_));
        set_input_shapes(std::move(input_shapes_));

        // output nodes
        int output_nodes_ = m_session.GetOutputCount();
        std::vector<const char *> output_names_;
        std::vector<std::vector<int64_t>> output_shapes_;
        for (int i = 0; i < output_nodes_; ++i) {
            Ort::AllocatedStringPtr name = m_session.GetOutputNameAllocated(i, m_allocator->get());
            m_output_names_alloc.emplace_back(std::move(name));
            output_names_.emplace_back(m_output_names_alloc.back().get());

            Ort::TypeInfo output_type_info = m_session.GetOutputTypeInfo(i);
            output_shapes_.emplace_back(output_type_info.GetTensorTypeAndShapeInfo().GetShape());
        }

        set_output_nodes(output_nodes_);
        set_output_names(std::move(output_names_));
        set_output_shapes(std::move(output_shapes_));

        // metadata
        std::unordered_map<std::string, std::string> model_metadata_;
        const auto &metadata = m_session.GetModelMetadata();
        const auto &keys = metadata.GetCustomMetadataMapKeysAllocated(m_allocator->get());
        for (const Ort::AllocatedStringPtr &key : keys)
            model_metadata_[key.get()] = metadata.LookupCustomMetadataMapAllocated(key.get(), m_allocator->get()).get();

        set_model_metadata(std::move(model_metadata_));
    } catch (const Ort::Exception& e) {
        throw std::runtime_error(fmt::format("ONNXRuntime error {}: {}", static_cast<int>(e.GetOrtErrorCode()), e.what()));
    } catch (const std::exception& e) {
        throw std::runtime_error(fmt::format("Standard exception caught: {}", e.what()));
    }
}

void onnxruntime::predict_raw(std::vector<std::vector<float>> &data,
                                                 std::vector<std::vector<int64_t>> customInputShapes)
{
    if (customInputShapes.empty())
        customInputShapes = input_shapes();

    if (data.size() != customInputShapes.size())
        throw std::runtime_error(fmt::format("Input shapes mismatch. data size {} != input_shape size {}", data.size(), customInputShapes.size()));

    std::vector<Ort::Value> input_tensors;
    for(size_t i = 0; i < customInputShapes.size(); ++i) {
        Ort::Value tensor = Ort::Value::CreateTensor<float>(
            *m_memory_info,
            data.at(i).data(),
            data.at(i).size(),
            customInputShapes.at(i).data(),
            customInputShapes.at(i).size()
        );

        input_tensors.emplace_back(std::move(tensor));
    }

    m_lastOutputTensors = m_session.Run(
        Ort::RunOptions{nullptr},
        input_names().data(),
        input_tensors.data(),
        input_nodes(),
        output_names().data(),
        output_nodes()
    );
}

const float *onnxruntime::tensor_data(int index)
{
    if (index < 0 || index >= m_lastOutputTensors.size()) {
        logger->critical(fmt::format("Inavlid tensor index provided: {}"), index);
        return nullptr;
    }

    return m_lastOutputTensors.at(index).GetTensorData<float>();
}

const std::vector<int64_t> onnxruntime::tensor_shape(int index)
{
    if (index < 0 || index >= m_lastOutputTensors.size()) {
        logger->critical(fmt::format("Inavlid tensor index provided: {}"), index);
        return {};
    }

    return m_lastOutputTensors.at(index).GetTensorTypeAndShapeInfo().GetShape();
}

void onnxruntime::print_metadata() const
{
    const Ort::ModelMetadata &metadata = m_session.GetModelMetadata();
    logger->debug("Model metadata:");
    logger->debug(fmt::format("\tFile: {}" ,m_config->model->path.value_or("").c_str()));
    logger->debug(fmt::format("\tGraph Name: {}", metadata.GetGraphNameAllocated(m_allocator->get()).get()));

    logger->debug("\tCustom Metadata:");
    for (const auto &[key, val] : model_metadata()) {
        logger->debug(fmt::format("\t {}: ", key, val));
    }

    logger->debug("\tInputs:");
    const std::vector<const char *> &input_names_ = input_names();
    const std::vector<std::vector<int64_t>> &input_shapes_ = input_shapes();
    for (size_t i = 0; i < input_nodes(); ++i) {
        logger->debug(fmt::format("\t  Name: {}", input_names_[i]));
        logger->debug(fmt::format("\t  Type: {}", input_shapes_[i]));
    }

    logger->debug("\tOutputs:");
    const std::vector<const char *> &output_names_ = output_names();
    const std::vector<std::vector<int64_t>> &output_shapes_ = output_shapes();
    for (size_t i = 0; i < output_nodes(); ++i) {
        logger->debug(fmt::format("\t  Name: {}", output_names_[i]));
        logger->debug(fmt::format("\t  Shape: {}", output_shapes_[i]));
    }
}

OrtAllocator* onnxruntime::allocator() const
{
    return m_allocator->get();
}

std::shared_ptr<CustomOrtAllocator> onnxruntime::custom_allocator() const
{
    return m_allocator;
}

std::shared_ptr<Ort::MemoryInfo> onnxruntime::memory_info() const
{
    return m_memory_info;
}

}
