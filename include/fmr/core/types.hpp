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
        std::vector<keypoint> points; // contains cv::Point2f, and conf
    };
    using predictions_t = std::vector<prediction>;

    struct box_t {
        std::vector<cv::Point2f> points;
        float score;
    };
    using boxes_t = std::vector<box_t>;

    using quad_t = std::array<cv::Point2f, 4>;
    struct quad_score {
        quad_t quad;
        float score;
    };
    using quad_scores_t = std::vector<quad_score>;

    namespace paddle::ocr {
        using det = box_t;
        using dets_t = std::vector<det>;
        
        typedef struct {
            int label_id;
            std::string label;
            float score;
        } cls;
        using clss_t = std::vector<cls>;

        typedef struct {
            std::string text;
            float score;
        } rec;
        using recs_t = std::vector<rec>;

        struct result {
            det det;
            cls cls;
            rec rec;
        };

        using results_t = std::vector<result>;
    }
}
