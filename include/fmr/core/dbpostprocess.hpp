#include <vector>

#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <clipper2/clipper.h>

#include <fmr/core/types.hpp>

namespace fmr {

struct contour2boxes_params {
    cv::Mat binary_mask;
    cv::Mat conf_map;
    cv::Size orig_size;
    float box_thresh;
    float unclip_ratio;
    int min_box_size;
    int max_candidates;
};

// Differential Binarization

// Finds the 4 corner points of a contour
inline std::pair<std::vector<cv::Point2f>, float> get_minimal_box(const std::vector<cv::Point2f> &contour) {
    cv::RotatedRect m_rect = cv::minAreaRect(contour);
    std::vector<cv::Point2f> points(4);
    m_rect.points(points.data());

    // Sort points: Top-Left, Top-Right, Bottom-Right, Bottom-Left
    std::sort(points.begin(), points.end(), [](const cv::Point2f &a, const cv::Point2f &b) {
        return a.x < b.x;
    });

    std::vector<cv::Point2f> reorder(4);
    int index_1, index_2, index_3, index_4;
    if (points[1].y > points[0].y) {
        index_1 = 0; index_4 = 1;
    } else {
        index_1 = 1; index_4 = 0;
    }

    if (points[3].y > points[2].y) {
        index_2 = 2; index_3 = 3;
    } else {
        index_2 = 3; index_3 = 2;
    }

    reorder[0] = points[index_1];
    reorder[1] = points[index_2];
    reorder[2] = points[index_3];
    reorder[3] = points[index_4];

    return {reorder, std::min(m_rect.size.width, m_rect.size.height)};
}

// Expands the shrunk text box back to original size
inline std::vector<cv::Point2f> unclip_polygon(const std::vector<cv::Point2f> &polygon, float unclip_ratio = 1.5f) {
    using namespace Clipper2Lib;

    // Use a scale factor for floating point precision in Clipper2's integer-based logic
    constexpr double CLIPPER_SCALE = 1000.0;

    double area = cv::contourArea(polygon);
    double length = cv::arcLength(polygon, true);
    
    // Skip degenerate polygons
    if (length <= 0) return {};

    // DBNet expansion distance formula
    double distance = (area * unclip_ratio) / length;

    // 1. Convert OpenCV points to Clipper2 Path64 with scaling
    Path64 path;
    for (const auto &pt : polygon) {
        path.push_back(Point64(pt.x * CLIPPER_SCALE, pt.y * CLIPPER_SCALE));
    }

    // 2. Perform Offsetting (Expansion)
    // JoinType::Round prevents "spikes" on sharp corners
    // EndType::Polygon ensures a closed shape
    Paths64 input_paths = {path};
    Paths64 solution = InflatePaths(input_paths, distance * CLIPPER_SCALE, JoinType::Round, EndType::Polygon);

    std::vector<cv::Point2f> unclipped;
    if (solution.empty() || solution[0].empty()) return unclipped;

    // 3. Convert back to cv::Point2f and descale
    for (const auto &pt : solution[0]) {
        unclipped.push_back(cv::Point2f(
            static_cast<float>(pt.x / CLIPPER_SCALE), 
            static_cast<float>(pt.y / CLIPPER_SCALE)
        ));
    }

    return unclipped;
}

// Averages the confidence values within the box
inline float calculate_box_score(const cv::Mat &confMap, const std::vector<cv::Point2f> &box) {
    int width = confMap.cols;
    int height = confMap.rows;

    std::vector<int> x_coords, y_coords;
    for (const auto &pt : box) {
        x_coords.push_back(std::clamp(static_cast<int>(pt.x), 0, width - 1));
        y_coords.push_back(std::clamp(static_cast<int>(pt.y), 0, height - 1));
    }

    auto [xmin, xmax] = std::minmax_element(x_coords.begin(), x_coords.end());
    auto [ymin, ymax] = std::minmax_element(y_coords.begin(), y_coords.end());

    int roi_w = *xmax - *xmin + 1;
    int roi_h = *ymax - *ymin + 1;

    cv::Mat mask = cv::Mat::zeros(roi_h, roi_w, CV_8UC1);
    std::vector<cv::Point> roi_box;
    for (const auto &pt : box) {
        roi_box.push_back(cv::Point(static_cast<int>(pt.x) - *xmin, static_cast<int>(pt.y) - *ymin));
    }

    std::vector<std::vector<cv::Point>> contours = {roi_box};
    cv::fillPoly(mask, contours, cv::Scalar(1));

    return static_cast<float>(cv::mean(confMap(cv::Rect(*xmin, *ymin, roi_w, roi_h)), mask)[0]);
}

// Extracts boxes from the binary mask
inline std::vector<box_t> contours_to_boxes(const contour2boxes_params &p) {
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(p.binary_mask, contours, cv::RETR_LIST, cv::CHAIN_APPROX_SIMPLE);

    const float width_scale = static_cast<float>(p.orig_size.width) / p.binary_mask.cols;
    const float height_scale = static_cast<float>(p.orig_size.height) / p.binary_mask.rows;


    std::vector<box_t> boxes;
    size_t n = std::min(contours.size(), static_cast<size_t>(p.max_candidates));

    for (size_t i = 0; i < n; ++i) {
        if (contours[i].size() < 4) 
            continue;

        std::vector<cv::Point2f> contour_f;
        for (const auto &pt : contours[i]) contour_f.push_back(cv::Point2f(pt.x, pt.y));

        auto [min_box, min_side] = get_minimal_box(contour_f);
        if (min_side < p.min_box_size) 
            continue;

        // CRITICAL FIX: Use confMap (the float probability map) for scoring
        float score = calculate_box_score(p.conf_map, min_box);
        if (score < p.box_thresh) 
            continue;

        auto unclipped = unclip_polygon(min_box, p.unclip_ratio);
        if (unclipped.empty()) 
            continue;

        auto [final_box, final_side] = get_minimal_box(unclipped);
        if (final_side < p.min_box_size + 2) 
            continue;

        // Scaling back to original image coordinates
        for (auto &pt : final_box) {
            pt.x = std::clamp(pt.x * width_scale, 0.0f, static_cast<float>(p.orig_size.width - 1));
            pt.y = std::clamp(pt.y * height_scale, 0.0f, static_cast<float>(p.orig_size.height - 1));
        }

        boxes.push_back({final_box, score});
    }

    return boxes;
}

}