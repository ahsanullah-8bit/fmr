#pragma once

#include <string>
#include <optional>
#include <vector>
#include <map>

struct ModelConfig {
    std::optional<std::string> path;
    std::optional<std::string> labelmap_path;
    std::optional<int> width;
    std::optional<int> height;
    std::optional<std::map<int,std::string>> labelmap;
};

struct PredictorConfig {
    std::optional<ModelConfig> model;
    std::optional<int> batch_size;
    std::optional<std::vector<int>> kpt_shape; // for pose model
};
