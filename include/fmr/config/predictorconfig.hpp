#pragma once

#include <string>
#include <optional>

namespace fmr {

struct predictor_config {
    std::optional<std::string> model_path;
};

}
