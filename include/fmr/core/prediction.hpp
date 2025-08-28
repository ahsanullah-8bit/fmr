#pragma once

#include <string>
#include <vector>

#include <opencv2/core/core.hpp>

namespace fmr {

    struct prediction {
        float conf;
        std::string label;
        int label_id = -1;
        int tracker_id = -1;
        cv::Rect box;
        cv::RotatedRect obb;
        cv::Mat mask;  // single-channel (8UC1) mask in full resolution
        // contains x, y, and conf
        std::vector<cv::Point3f> points;
    };

    using predictions_t = std::vector<prediction>;
}
