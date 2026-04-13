#pragma once

#include <array>
#include <string>
#include <vector>
#include <optional>

namespace fmr {

struct paddleocr_config {
    enum image_mode {
        RGB, BGR
    };

    enum normalization_order {
        hwc
    };

    enum unclip_mode {
        Box,
        Polygon
    };

    enum ctc_mode {
        GreedySearch,
        BeamSearch,
        PrefixBeamSearch
    };

    std::optional<image_mode> img_mode;       // det
    std::optional<normalization_order> norm_order;     // det
    std::optional<std::array<float, 3>> mean; // det, cls, rec
    std::optional<std::array<float, 3>> std;  // det, cls, rec
    std::optional<float> scale;               // det, cls, rec
    std::optional<int> channels;

    std::optional<float> thresh;              // det
    std::optional<float> box_thresh;          // det
    std::optional<int> min_box_size;          // det
    std::optional<int> max_candidates;        // det
    std::optional<float> unclip_ratio;        // det
    std::optional<unclip_mode> unclip_mode;   // det
    std::optional<ctc_mode> ctc_mode;         // rec
    std::optional<int> limit_side_len;        // det

    std::optional<int> batch;                 // det, cls, rec
    std::optional<int> stride;                // det, cls, rec
    std::optional<std::array<int, 2>> imgsz;  // det, cls, rec
    std::optional<std::vector<std::string>> labels; // cls
    std::optional<std::vector<char>> character_dict;// rec
};

}