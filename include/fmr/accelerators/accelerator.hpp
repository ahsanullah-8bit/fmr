#pragma once

#include <any>
#include <cassert>
#include <memory>
#include <string>
#include <unordered_map>

#include <spdlog/logger.h>
#include <fmt/os.h>

namespace fmr {

enum class tensor_data_type {
    FLOAT32,
    FLOAT16,
    INT8,
    UINT8,
    INT32,
    INT64,
    UNKNOWN
};

struct tensor {
    std::string name;
    std::vector<int64_t> shape;
    tensor_data_type type;
    void* data;
    int device;
    size_t size_bytes;
};

class accelerator {
public:
    virtual ~accelerator() = default;
    virtual bool load_model(const std::string& modelPath, const std::any& backendConfig = {}) = 0;
    virtual bool predict_raw(const std::vector<tensor>& inputs, std::vector<tensor>& outputs, const std::any& runOptions = {}) = 0;
    virtual std::vector<tensor> input_details() const = 0;
    virtual std::vector<tensor> output_details() const = 0;
    virtual void print_metadata() const;

    std::shared_ptr<spdlog::logger> logger() const;
    void set_logger(std::shared_ptr<spdlog::logger> logger);
    const std::unordered_map<std::string, std::string>& model_metadata() const;

protected:
    void set_model_metadata(const std::unordered_map<std::string, std::string> &newModelMetadata);

private:
    std::shared_ptr<spdlog::logger> m_logger;
    std::unordered_map<std::string, std::string> m_model_metadata;
};

// Definitions

inline void accelerator::print_metadata() const
{}

inline std::shared_ptr<spdlog::logger> accelerator::logger() const
{
    return m_logger;
}

inline void accelerator::set_logger(std::shared_ptr<spdlog::logger> logger)
{
    m_logger = logger;
}

inline const std::unordered_map<std::string, std::string> &accelerator::model_metadata() const
{
    return m_model_metadata;
}

inline void accelerator::set_model_metadata(const std::unordered_map<std::string, std::string> &newModelMetadata)
{
    m_model_metadata = newModelMetadata;
}

}
