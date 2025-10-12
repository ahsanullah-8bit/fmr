#pragma once

#include <cassert>
#include <string>
#include <unordered_map>

#include <fmt/os.h>

namespace fmr {

class accelerator {
public:
    virtual void predict_raw(std::vector<std::vector<float>> &data,
                                     std::vector<std::vector<int64_t> > customInputShapes = {}) = 0;
    virtual const float *tensor_data(int index) = 0;
    virtual const std::vector<int64_t> tensor_shape(int index) = 0;
    virtual void print_metadata() const;

    const std::unordered_map<std::string, std::string>& model_metadata() const;
    const std::vector<std::vector<int64_t>> &input_shapes() const;
    const std::vector<std::vector<int64_t>> &output_shapes() const;
    size_t input_nodes() const;
    size_t output_nodes() const;
    const std::vector<const char *> &input_names() const;
    const std::vector<const char *> &output_names() const;


protected:
    void set_input_shapes(const std::vector<std::vector<int64_t> > &newInput_shapes);
    void set_output_shapes(const std::vector<std::vector<int64_t> > &newOutput_shapes);
    void set_input_nodes(size_t newInput_nodes);
    void set_output_nodes(size_t newOutput_nodes);
    void set_model_metadata(const std::unordered_map<std::string, std::string> &newModel_metadata);
    void set_input_names(const std::vector<const char *> &newInput_names);
    void set_output_names(const std::vector<const char *> &newOutput_names);

private:
    std::vector<std::vector<int64_t>> m_input_shapes;
    std::vector<std::vector<int64_t>> m_output_shapes;
    size_t m_input_nodes = 1, m_output_nodes = 1;
    std::unordered_map<std::string, std::string> m_model_metadata;
    std::vector<const char *> m_input_names;
    std::vector<const char *> m_output_names;
};

// Definitions

inline void accelerator::print_metadata() const
{}

inline const std::unordered_map<std::string, std::string> &accelerator::model_metadata() const
{
    return m_model_metadata;
}

inline const std::vector<std::vector<int64_t>> &accelerator::input_shapes() const
{
    return m_input_shapes;
}

inline const std::vector<std::vector<int64_t>> &accelerator::output_shapes() const
{
    return m_output_shapes;
}

inline size_t accelerator::input_nodes() const
{
    return m_input_nodes;
}

inline size_t accelerator::output_nodes() const
{
    return m_output_nodes;
}

inline const std::vector<const char *> &accelerator::input_names() const
{
    return m_input_names;
}

inline const std::vector<const char *> &accelerator::output_names() const
{
    return m_output_names;
}

inline void accelerator::set_input_shapes(const std::vector<std::vector<int64_t> > &newInput_shapes)
{
    m_input_shapes = newInput_shapes;
}

inline void accelerator::set_output_shapes(const std::vector<std::vector<int64_t> > &newOutput_shapes)
{
    m_output_shapes = newOutput_shapes;
}

inline void accelerator::set_input_nodes(size_t newInput_nodes)
{
    m_input_nodes = newInput_nodes;
}

inline void accelerator::set_output_nodes(size_t newOutput_nodes)
{
    m_output_nodes = newOutput_nodes;
}

inline void accelerator::set_model_metadata(const std::unordered_map<std::string, std::string> &newModel_metadata)
{
    m_model_metadata = newModel_metadata;
}

inline void accelerator::set_input_names(const std::vector<const char *> &newInput_names)
{
    m_input_names = newInput_names;
}

inline void accelerator::set_output_names(const std::vector<const char *> &newOutput_names)
{
    m_output_names = newOutput_names;
}

}
