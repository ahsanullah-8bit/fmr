#pragma once

#include <memory>

#include <onnxruntime_cxx_api.h>

#include <fmr/accelarators/accelerator.h>
#include <fmr/config/predictorconfig.h>
#include <fmr/wrappers/customortallocator.h>

namespace fmr {

    class FMR_EXPORT onnxruntime : accelerator {
    public:
        explicit onnxruntime(
            const PredictorConfig *config,
            const std::shared_ptr<Ort::Env>& env = nullptr,
            const std::shared_ptr<CustomOrtAllocator>& allocator = nullptr,
            const std::shared_ptr<Ort::MemoryInfo> &memoryInfo = nullptr
		);

        void predict_raw(std::vector<std::vector<float>> &data,
                                 std::vector<std::vector<int64_t>> customInputShapes = {}) override;
        const float *tensor_data(int index) override;
        const std::vector<int64_t> tensor_shape(int index) override;

        void print_metadata() const override;
        OrtAllocator* allocator() const;
        std::shared_ptr<CustomOrtAllocator> custom_allocator() const;
        std::shared_ptr<Ort::MemoryInfo> memory_info() const;

    private:
        PredictorConfig *m_config;
        std::shared_ptr<Ort::Env> m_env;
        Ort::Session m_session { nullptr };
        std::shared_ptr<CustomOrtAllocator> m_allocator;
        std::shared_ptr<Ort::MemoryInfo> m_memory_info;

        // Vectors to hold allocated input and output node names
        std::vector<Ort::AllocatedStringPtr> m_input_names_alloc;
        std::vector<Ort::AllocatedStringPtr> m_output_names_alloc;
        std::vector<std::string> m_available_eps;
        std::vector<std::string> m_selected_eps;

        std::vector<Ort::Value> m_lastOutputTensors;
    };

}
