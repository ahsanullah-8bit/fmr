#pragma once

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

namespace fmr {
    struct keypoint {
        cv::Point2f point;
        float conf;
    };

    struct prediction {
        float conf;
        std::string label;
        int label_id = -1;
        int tracker_id = -1;
        cv::Rect box;
        cv::RotatedRect obb;
        cv::Mat mask;  // single-channel (8UC1) mask in full resolution
        // contains cv::Point2f, and conf
        std::vector<keypoint> points;
    };
    using predictions_t = std::vector<prediction>;
}
