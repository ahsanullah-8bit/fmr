#pragma once

#include <filesystem>
#include <memory>

#include <onnxruntime_cxx_api.h>
#include <fmt/format.h>
#include <fmt/ranges.h>
#include <spdlog/spdlog.h>
#include <spdlog/sinks/stdout_color_sinks.h>

#include <fmr/accelarators/accelerator.hpp>
#include <fmr/config/predictorconfig.hpp>
#include <fmr/wrappers/customortallocator.hpp>

namespace fmr {

    class onnxruntime : public accelerator {
    public:
        explicit onnxruntime(
            const predictor_config &config,
            std::shared_ptr<Ort::Env> env = nullptr,
            std::shared_ptr<Ort::SessionOptions> sessionOptions = nullptr,
            std::shared_ptr<custom_ort_alloc> allocator = nullptr,
            std::shared_ptr<Ort::MemoryInfo> memoryInfo = nullptr
            );

        void predict_raw(std::vector<std::vector<float>> &data,
                                 std::vector<std::vector<int64_t>> customInputShapes = {}) override;
        const float *tensor_data(int index) override;
        const std::vector<int64_t> tensor_shape(int index) override;
        void set_run_options(std::shared_ptr<Ort::RunOptions> runOptions);
        void set_logger(std::shared_ptr<spdlog::logger> logger) override;

        void print_metadata() const override;
        OrtAllocator* allocator() const;
        std::shared_ptr<custom_ort_alloc> custom_allocator() const;
        std::shared_ptr<Ort::MemoryInfo> memory_info() const;

    private:
        const predictor_config &m_config;
        std::shared_ptr<Ort::Env> m_env;
        Ort::Session m_session { nullptr };
        std::shared_ptr<custom_ort_alloc> m_allocator;
        std::shared_ptr<Ort::MemoryInfo> m_memory_info;
        std::shared_ptr<Ort::RunOptions> m_run_options;

        // Vectors to hold allocated input and output node names
        std::vector<Ort::AllocatedStringPtr> m_input_names_alloc;
        std::vector<Ort::AllocatedStringPtr> m_output_names_alloc;
        std::vector<std::string> m_available_eps;
        std::vector<std::string> m_selected_eps;

        std::vector<Ort::Value> m_lastOutputTensors;

        std::shared_ptr<spdlog::logger> m_logger;
    };

    // Definitions

    inline onnxruntime::onnxruntime(const predictor_config &config,
                                    std::shared_ptr<Ort::Env> env,
                                    std::shared_ptr<Ort::SessionOptions> sessionOptions,
                                    std::shared_ptr<custom_ort_alloc> allocator,
                                    std::shared_ptr<Ort::MemoryInfo> memoryInfo)
        : m_config(config)
        , m_env(env)
        , m_allocator(allocator)
        , m_memory_info(memoryInfo)
        , m_logger(spdlog::default_logger()->clone("fmr.accelators.ort"))
    {
        m_logger->set_level(spdlog::level::debug);

        if (!config.model_path.has_value())
            throw std::runtime_error("Invalid config, please provide a valid config with model info!");

        if (!m_env)
            m_env = std::make_shared<Ort::Env>(ORT_LOGGING_LEVEL_WARNING, "ONNX_Inference");

        if (!m_memory_info)
            m_memory_info = std::make_shared<Ort::MemoryInfo>(Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault));

        std::string model_path = config.model_path.value_or("");
        if (!std::filesystem::exists(model_path)
            || model_path.find_last_of(".onnx") == std::string::npos)
            throw std::runtime_error("Model does not exist or has invalid extension. Please download a model first and try again!");


        try {
            if (!sessionOptions) {
                sessionOptions = std::make_shared<Ort::SessionOptions>();
                // TODO: This has to be decided by the user
                int intraop_threads = std::min(4, static_cast<int>(std::thread::hardware_concurrency()));
                sessionOptions->SetIntraOpNumThreads(intraop_threads);
                sessionOptions->SetGraphOptimizationLevel(GraphOptimizationLevel::ORT_ENABLE_BASIC);
            }

            m_available_eps = Ort::GetAvailableProviders();

            m_selected_eps.emplace_back("CPUExecutionProvider");

#ifdef _WIN32
            std::wstring model_path_(model_path.begin(), model_path.end());
#else
            std::string model_path_(model_path);
#endif
            m_session = Ort::Session(*m_env, model_path_.c_str(), *sessionOptions);

            if (!m_allocator)
                m_allocator = std::make_shared<custom_ort_alloc>(Ort::Allocator(m_session, *m_memory_info));

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

            // Output nodes
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

            // Metadata
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

    inline void onnxruntime::predict_raw(std::vector<std::vector<float>> &data,
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

        if (!m_run_options)
            m_run_options = std::make_shared<Ort::RunOptions>(nullptr);

        m_lastOutputTensors = m_session.Run(
            *m_run_options,
            input_names().data(),
            input_tensors.data(),
            input_nodes(),
            output_names().data(),
            output_nodes()
            );
    }

    inline const float *onnxruntime::tensor_data(int index)
    {
        if (index < 0 || index >= m_lastOutputTensors.size()) {
            m_logger->critical("Inavlid tensor index provided: {}", index);
            return nullptr;
        }

        return m_lastOutputTensors.at(index).GetTensorData<float>();
    }

    inline const std::vector<int64_t> onnxruntime::tensor_shape(int index)
    {
        if (index < 0 || index >= m_lastOutputTensors.size()) {
            m_logger->critical("Inavlid tensor index provided: {}", index);
            return {};
        }

        return m_lastOutputTensors.at(index).GetTensorTypeAndShapeInfo().GetShape();
    }

    inline void onnxruntime::set_run_options(std::shared_ptr<Ort::RunOptions> runOptions)
    {
        m_run_options = runOptions;
    }

    inline void onnxruntime::set_logger(std::shared_ptr<spdlog::logger> logger)
    {
        m_logger = logger;
    }

    inline void onnxruntime::print_metadata() const
    {
        const Ort::ModelMetadata &metadata = m_session.GetModelMetadata();
        m_logger->debug("Model metadata:");
        m_logger->debug("  File: {}", m_config.model_path.value_or("").c_str());
        m_logger->debug("  Graph Name: {}", metadata.GetGraphNameAllocated(m_allocator->get()).get());

        m_logger->debug("  Custom Metadata:");
        const auto metadata_metadata = model_metadata();
        for (const auto &[key, val] : metadata_metadata) {
            m_logger->debug("    {}: {}", key, val);
        }

        m_logger->debug("  Inputs:");
        const std::vector<const char *> &input_names_ = input_names();
        const std::vector<std::vector<int64_t>> &input_shapes_ = input_shapes();
        for (size_t i = 0; i < input_nodes(); ++i) {
            m_logger->debug("    Name: {}", input_names_[i]);
            m_logger->debug("    Type: {}", input_shapes_[i]);
        }

        m_logger->debug("  Outputs:");
        const std::vector<const char *> &output_names_ = output_names();
        const std::vector<std::vector<int64_t>> &output_shapes_ = output_shapes();
        for (size_t i = 0; i < output_nodes(); ++i) {
            m_logger->debug("    Name: {}", output_names_[i]);
            m_logger->debug("    Shape: {}", output_shapes_[i]);
        }
    }

    inline OrtAllocator* onnxruntime::allocator() const
    {
        return m_allocator->get();
    }

    inline std::shared_ptr<custom_ort_alloc> onnxruntime::custom_allocator() const
    {
        return m_allocator;
    }

    inline std::shared_ptr<Ort::MemoryInfo> onnxruntime::memory_info() const
    {
        return m_memory_info;
    }
}
