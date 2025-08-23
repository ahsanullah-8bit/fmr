#pragma once

#include <string>
#include <optional>
#include <vector>
#include <map>

namespace fmr {

struct model_config {
    std::optional<std::string> path;
    std::optional<std::string> labelmap_path;
    std::optional<int> width;
    std::optional<int> height;
    std::optional<std::map<int,std::string>> labelmap;
};

struct predictor_config {
    std::optional<model_config> model;
    std::optional<int> batch_size;
    std::optional<std::vector<int>> kpt_shape; // for pose model
};

}
