#pragma once

#include <fstream>
#include <numeric>
#include <type_traits>
#include <random>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <fmt/format.h>

#include <fmr/core/types.hpp>

namespace fmr {

template <typename T>
inline typename std::enable_if<std::is_arithmetic<T>::value, T>::type
clamp(const T &value, const T &low, const T &high)
{
    T validLow = low < high ? low : high;
    T validHigh = low < high ? high : low;

    if (value < validLow)
        return validLow;
    if (value > validHigh)
        return validHigh;
    return value;
}

inline std::unordered_map<int, std::string> read_labels(const std::string &path)
{
    std::unordered_map<int, std::string> labels;
    std::ifstream stream(path, std::ios_base::in);

    std::string label;
    for (int i = 0; std::getline(stream, label); ++i) {
        label.erase(label.find_last_not_of(" \t\r\n") + 1);
        labels[i] = label;
    }

    return labels;
}

inline size_t vec_product(const std::vector<int64_t> &vector)
{
    return std::accumulate(vector.begin(), vector.end(), 1ull, std::multiplies<int64_t>());
}

inline void letter_box(const cv::Mat& image, cv::Mat& outImage,
                       const cv::Size& newShape,
                       const cv::Scalar& color = cv::Scalar(114, 114, 114),
                       const bool scale = true)
{
    float ratio = std::min(static_cast<float>(newShape.height) / image.rows,
                           static_cast<float>(newShape.width) / image.cols);

    if (!scale)
        ratio = std::min(ratio, 1.0f);

    const cv::Size size_unpdd(std::round(image.cols * ratio), std::round(image.rows * ratio));

    const int pad_hori = newShape.width - size_unpdd.width;
    const int pad_vert = newShape.height - size_unpdd.height;
    const int pad_top = pad_vert / 2;
    const int pad_bottom = pad_vert - pad_top;
    const int pad_left = pad_hori / 2;
    const int pad_right = pad_hori - pad_left;

    cv::resize(image, outImage, size_unpdd);
    cv::copyMakeBorder(outImage, outImage, pad_top, pad_bottom, pad_left, pad_right, cv::BORDER_CONSTANT, color);
}

inline void norm_and_permute(cv::Mat &img, float *&buffer, float scale = 1.0f / 255.0f)
{
    // normalize 0-1
    img.convertTo(img, CV_32FC3, scale);

    const int channels = img.channels();
    const int image_area = img.cols * img.rows;

    // split and permute at once
    std::vector<cv::Mat> out_channels(channels);
    for (int c = 0; c < channels; ++c)
        out_channels[c] = cv::Mat(img.rows, img.cols, CV_32FC1, buffer + c * image_area);

    cv::split(img, out_channels);
}

inline void norm_and_permute(std::vector<cv::Mat> &batch, std::vector<float> &buffer, float scale = 1.0f / 255.0f)
{
    if (batch.empty())
        return;

    // expect 3 channels and buffer size exactly equal to (batch.size * img.area)
    const cv::Mat &first = batch.at(0);
    if (first.channels() != 3
        || buffer.size() != batch.size() * 3 * first.cols * first.rows) {
        // TODO: Add a warning.
        return;
    }

    for (int b = 0; b < batch.size(); ++b) {
        cv::Mat &img = batch[b];
        float *offset_ptr = buffer.data() + b * (img.channels() * img.cols * img.rows);
        norm_and_permute(img, offset_ptr, scale);
    }
}

inline cv::Mat sigmoid(const cv::Mat& src) {
    cv::Mat dst;
    cv::exp(-src, dst);
    dst += 1.0;
    cv::divide(1.0, dst, dst);
    return dst;
}

inline cv::Rect scale_coords(const cv::Size &resizedImageShape,
                             const cv::Rect &coords,
                             const cv::Size &originalImageShape,
                             bool clip,
                             float gain,
                             int padX,
                             int padY)
{
    cv::Rect result;
    result.x = static_cast<int>(std::round((coords.x - padX) / gain));
    result.y = static_cast<int>(std::round((coords.y - padY) / gain));
    result.width = static_cast<int>(std::round(coords.width / gain));
    result.height = static_cast<int>(std::round(coords.height / gain));

    if (clip) {
        result.x = clamp(result.x, 0, originalImageShape.width);
        result.y = clamp(result.y, 0, originalImageShape.height);
        result.width = clamp(result.width, 0, originalImageShape.width - result.x);
        result.height = clamp(result.height, 0, originalImageShape.height - result.y);
    }

    return result;
}

inline cv::Rect scale_coords(const cv::Size &resizedImageShape,
                             const cv::Rect &box,
                             const cv::Size &originalImageShape,
                             bool clip = true)
{
    const float gain = std::min(static_cast<float>(resizedImageShape.height) / static_cast<float>(originalImageShape.height),
                                static_cast<float>(resizedImageShape.width) / static_cast<float>(originalImageShape.width));

    const int pad_x = static_cast<int>(std::round((resizedImageShape.width - originalImageShape.width * gain) / 2.0f));
    const int pad_y = static_cast<int>(std::round((resizedImageShape.height - originalImageShape.height * gain) / 2.0f));

    return scale_coords(resizedImageShape, box, originalImageShape, clip, gain, pad_x, pad_y);
}

inline cv::RotatedRect scale_coords(const cv::Size& resizedImageShape,
                                    const cv::RotatedRect& coords,
                                    const cv::Size& originalImageShape,
                                    bool clip = true)
{
    const float gain = std::min(static_cast<float>(resizedImageShape.height) / static_cast<float>(originalImageShape.height),
                                static_cast<float>(resizedImageShape.width) / static_cast<float>(originalImageShape.width));

    const float pad_x = (resizedImageShape.width - originalImageShape.width * gain) / 2.0f;
    const float pad_y = (resizedImageShape.height - originalImageShape.height * gain) / 2.0f;

    const float cx = (coords.center.x - pad_x) / gain;
    const float cy = (coords.center.y - pad_y) / gain;
    const float w = coords.size.width / gain;
    const float h = coords.size.height / gain;

    cv::RotatedRect result(cv::Point2f(cx, cy), cv::Size2f(w, h), coords.angle);

    if (clip) {
        // clip corners instead of rect (since rotated)
        std::vector<cv::Point2f> pts(4);
        result.points(pts.data());
        for (auto& p : pts) {
            p.x = clamp(p.x, 0.0f, static_cast<float>(originalImageShape.width));
            p.y = clamp(p.y, 0.0f, static_cast<float>(originalImageShape.height));
        }
        result = cv::minAreaRect(pts); // rebuild clipped rect
    }

    return result;
}

inline cv::Point2f scale_coords(const cv::Size &resizedImageShape,
                                const cv::Point2f &point,
                                const cv::Size &originalImageShape,
                                bool clip = true)
{
    const float gain = std::min(static_cast<float>(resizedImageShape.height) / static_cast<float>(originalImageShape.height),
                                static_cast<float>(resizedImageShape.width) / static_cast<float>(originalImageShape.width));

    const int pad_x = static_cast<int>(std::round((resizedImageShape.width - originalImageShape.width * gain) / 2.0f));
    const int pad_y = static_cast<int>(std::round((resizedImageShape.height - originalImageShape.height * gain) / 2.0f));

    cv::Point2f result;
    result.x = std::round((point.x - pad_x) / gain);
    result.y = std::round((point.y - pad_y) / gain);

    if (clip) {
        result.x = clamp(result.x, 0.0f, (float)originalImageShape.width);
        result.y = clamp(result.y, 0.0f, (float)originalImageShape.height);
    }

    return result;
}

inline std::vector<int> nms_bboxes(const std::vector<cv::Rect>& boxes,
                                   const std::vector<float>& scores,
                                   const float scoreThreshold,
                                   const float iouThreshold)
{

    std::vector<int> result_indices;
    // indices.clear();

    const size_t num_boxes = boxes.size();
    if (num_boxes < 1)
        return {};

    // filter and sort based on scores
    std::vector<int> sorted_indices;
    sorted_indices.reserve(num_boxes);
    for (size_t i = 0; i < num_boxes; ++i) {
        if (scores[i] >= scoreThreshold) {
            sorted_indices.emplace_back(static_cast<int>(i));
        }
    }

    if (sorted_indices.empty())
        return {};

    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [&scores](int idx1, int idx2) {
                  return scores[idx1] > scores[idx2];
              });

    // precompute box areas
    std::vector<float> areas(num_boxes, 0.0f);
    for (size_t i = 0; i < num_boxes; ++i) {
        areas[i] = boxes[i].width * boxes[i].height;
    }

    // suppression mask to mark suppressed boxes.
    std::vector<bool> suppressed(num_boxes, false);

    // suppress sorted boxes with high IoU
    for (size_t i = 0; i < sorted_indices.size(); ++i) {
        const int current_idx = sorted_indices[i];
        if (suppressed[current_idx]) {
            continue;
        }

        // select the current box as a valid detection
        result_indices.push_back(current_idx);

        const cv::Rect& current_box = boxes[current_idx];
        const float x1_max = current_box.x;
        const float y1_max = current_box.y;
        const float x2_max = current_box.x + current_box.width;
        const float y2_max = current_box.y + current_box.height;
        const float area_current = areas[current_idx];

        // compare IoU of the current box with the rest
        for (size_t j = i + 1; j < sorted_indices.size(); ++j) {
            int compare_idx = sorted_indices[j];
            if (suppressed[compare_idx]) {
                continue;
            }

            const cv::Rect& compare_box = boxes[compare_idx];
            const float x1 = std::max(x1_max, static_cast<float>(compare_box.x));
            const float y1 = std::max(y1_max, static_cast<float>(compare_box.y));
            const float x2 = std::min(x2_max, static_cast<float>(compare_box.x + compare_box.width));
            const float y2 = std::min(y2_max, static_cast<float>(compare_box.y + compare_box.height));

            const float inter_width = x2 - x1;
            const float inter_height = y2 - y1;

            if (inter_width <= 0 || inter_height <= 0) {
                continue;
            }

            const float intersection = inter_width * inter_height;
            const float unionArea = area_current + areas[compare_idx] - intersection;
            const float iou = (unionArea > 0.0f) ? (intersection / unionArea) : 0.0f;

            if (iou > iouThreshold) {
                suppressed[compare_idx] = true;
            }
        }
    }

    return result_indices;
}

inline float obb_iou(const cv::RotatedRect& r1, const cv::RotatedRect& r2) {
    std::vector<cv::Point2f> inter_pts;
    const int res = cv::rotatedRectangleIntersection(r1, r2, inter_pts);
    if (res == cv::INTERSECT_NONE)
        return 0.0f;

    float inter_area = 0.0f;
    if (!inter_pts.empty())
        inter_area = cv::contourArea(inter_pts);

    float union_area = r1.size.area() + r2.size.area() - inter_area;
    return inter_area / (union_area + 1e-7f);
}

inline std::vector<int> nms_obbs(const std::vector<cv::RotatedRect>& boxes,
                                 const std::vector<float>& scores,
                                 const float scoreThreshold,
                                 const float iouThreshold) {
    const size_t num_boxes = boxes.size();
    if (num_boxes < 1)
        return {};
    
    // filter based on scores
    std::vector<int> sorted_indices;
    sorted_indices.reserve(num_boxes);
    for (size_t i = 0; i < num_boxes; ++i) {
        if (scores[i] >= scoreThreshold) {
            sorted_indices.emplace_back(static_cast<int>(i));
        }
    }
    
    if (sorted_indices.empty())
        return {};
    
    std::iota(sorted_indices.begin(), sorted_indices.end(), 0);
    // sort based on scores
    std::sort(sorted_indices.begin(), sorted_indices.end(),
              [&scores](int idx1, int idx2) {
                  return scores[idx1] > scores[idx2];
              });

    std::vector<int> results;
    for (int i : sorted_indices) {
        bool keep = true;
        for (int j : results) {
            if (obb_iou(boxes[i], boxes[j]) > iouThreshold) {
                keep = false;
                break;
            }
        }
        if (keep) {
            results.push_back(i);
        }
    }

    return results;
}

inline std::vector<cv::Scalar> generate_colors(const std::unordered_map<int, std::string> &classNames,
                                               int seed = 42)
{
    static std::unordered_map<size_t, std::vector<cv::Scalar>> colorCache;

    size_t hashKey = 0;
    for (const auto& [_, name] : classNames) {
        hashKey ^= std::hash<std::string>{}(name) + 0x9e3779b9 + (hashKey << 6) + (hashKey >> 2);
    }

    auto it = colorCache.find(hashKey);
    if (it != colorCache.end()) {
        return it->second;
    }

    std::vector<cv::Scalar> colors;
    colors.reserve(classNames.size());

    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> uni(0, 255);

    for (size_t i = 0; i < classNames.size(); ++i) {
        colors.emplace_back(cv::Scalar(uni(rng), uni(rng), uni(rng)));
    }

    // Cache the generated colors for future use
    colorCache.emplace(hashKey, colors);

    return colorCache[hashKey];
}

inline std::vector<cv::Scalar> generate_colors(size_t size)
{
    static std::mt19937 rng(size);
    std::uniform_real_distribution<float> hue_dist(0.0f, 180.0f);

    std::vector<cv::Scalar> colors;
    colors.reserve(size);
    for (size_t i = 0; i < size; ++i) {
        float h = hue_dist(rng);
        float s = 200 + (rng() % 56);  // 200–255
        float v = 200 + (rng() % 56);  // 200–255

        cv::Mat hsv(1,1,CV_8UC3, cv::Scalar(h, s, v));
        cv::Mat bgr;
        cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);

        cv::Vec3b bgr_pixel = bgr.at<cv::Vec3b>(0,0);
        colors.emplace_back(bgr_pixel[0], bgr_pixel[1], bgr_pixel[2]);
    }

    return colors;
}

inline void draw_bboxes(cv::Mat &image,
                        const std::vector<prediction> &predictions,
                        const std::unordered_map<int, std::string> &labels,
                        const std::vector<cv::Scalar> &colors,
                        float maskAlpha = 0.3f)
{
    const int font_face = cv::FONT_HERSHEY_SIMPLEX;
    const double font_scale = std::min(image.rows, image.cols) * 0.0008;
    const int thickness = std::max(1, static_cast<int>(std::min(image.rows, image.cols) * 0.002));

    for (const auto& prediction : predictions) {
        if (prediction.label_id < 0 || prediction.label_id >= labels.size())
            continue;

        const cv::Scalar& color = colors.empty() ? cv::Scalar(0, 0, 255) : colors[prediction.label_id % colors.size()];
        std::string label;

        if (labels.empty()) {
            label = fmt::format("{}%",
                                static_cast<int>(prediction.conf * 100));
        } else if (prediction.tracker_id == -1) {
            label = fmt::format("{} - {}%",
                                labels.at(prediction.label_id),
                                static_cast<int>(prediction.conf * 100));
        } else {
            label = fmt::format("{} - {} - {}%",
                                labels.at(prediction.label_id),
                                prediction.tracker_id,
                                static_cast<int>(prediction.conf * 100));
        }

        cv::rectangle(image, prediction.box, color, 2,  cv::LINE_AA);

        int baseline = 0;
        const cv::Size text_size = cv::getTextSize(label, font_face, font_scale, thickness, &baseline);

        const int label_y = std::max(prediction.box.y, text_size.height + 5);
        const cv::Point label_tl(prediction.box.x, label_y - text_size.height - 5);
        const cv::Point label_br(prediction.box.x + text_size.width + 5, label_y + baseline - 5);

        // label background
        cv::rectangle(image, label_tl, label_br, color, cv::FILLED);
        cv::putText(image, label, cv::Point(prediction.box.x + 2, label_y - 2),
                    font_face, font_scale, cv::Scalar(255, 255, 255),
                    thickness, cv::LINE_AA);
    }
}

inline void draw_obbs(cv::Mat &image,
                      const std::vector<prediction> &predictions,
                      const std::unordered_map<int, std::string> &labels,
                      const std::vector<cv::Scalar> &colors,
                      float maskAlpha = 0.3f)
{
    const int font_face = cv::FONT_HERSHEY_SIMPLEX;
    const double font_scale = std::min(image.rows, image.cols) * 0.0008;
    const int thickness = std::max(1, static_cast<int>(std::min(image.rows, image.cols) * 0.002));

    for (const auto& prediction : predictions) {
        if (prediction.label_id < 0 || prediction.label_id >= labels.size())
            continue;

        const cv::Scalar& color = colors.empty() ? cv::Scalar(0, 0, 255) : colors[prediction.label_id % colors.size()];
        std::string label;
        if (labels.empty()) {
            label = fmt::format("{}%",
                                static_cast<int>(prediction.conf * 100));
        } else if (prediction.tracker_id == -1) {
            label = fmt::format("{} - {}%",
                                labels.at(prediction.label_id),
                                static_cast<int>(prediction.conf * 100));
        } else {
            label = fmt::format("{} - {} - {}%",
                                labels.at(prediction.label_id),
                                prediction.tracker_id,
                                static_cast<int>(prediction.conf * 100));
        }

        cv::Point2f vertices[4];
        prediction.obb.points(vertices);

        for (int i = 0; i < 4; i++)
            cv::line(image, vertices[i], vertices[(i+1) % 4], color, 2, cv::LINE_AA);

        int baseline = 0;
        const cv::Size text_size = cv::getTextSize(label, font_face, font_scale, thickness, &baseline);

        const cv::Rect brect = prediction.obb.boundingRect();
        const int x = brect.x;
        const int y = std::max(brect.y, text_size.height + 5);

        cv::rectangle(image,
                      cv::Point(x, y - text_size.height - 5),
                      cv::Point(x + text_size.width + 5, y + baseline - 5),
                      color, cv::FILLED);
        cv::putText(image, label,
                    cv::Point(x + 2, y - 2),
                    font_face, font_scale, cv::Scalar(255, 255, 255),
                    thickness, cv::LINE_AA);
    }
}

inline void draw_keypoints(cv::Mat &image,
                           const std::vector<prediction> &predictions,
                           const std::vector<std::pair<int, int>> &poseSkeleton,
                           const std::unordered_map<int, std::string> &labels,
                           const std::vector<cv::Scalar> &colors,
                           bool drawBox = false,
                           float maskAlpha = 0.3f)
{
    if (predictions.empty())
        return;

    if (drawBox)
        draw_bboxes(image, predictions, labels, colors, maskAlpha);

    const float scale_factor = std::min(image.rows, image.cols) / 1280.0f;  // reference 1280px size
    const int line_thickness = std::max(2, static_cast<int>(3 * scale_factor));
    const int kpt_radius = std::max(3, static_cast<int>(5 * scale_factor));

    for (const auto& prediction : predictions) {
        if (prediction.label_id < 0 || prediction.label_id >= labels.size())
            continue;

        const size_t num_kpts = prediction.points.size();
        const std::vector<keypoint> &kpts = prediction.points;

        // draw keypoints
        for (size_t i = 0; i < num_kpts; ++i)
            cv::circle(image, kpts[i].point, kpt_radius, colors[i % colors.size()], -1, cv::LINE_AA);

        // draw skeleton connections
        for (size_t j = 0; j < poseSkeleton.size(); ++j) {
            const auto [src, dst] = poseSkeleton[j];
            if (src < num_kpts && dst < num_kpts) {
                cv::line(image, kpts[src].point, kpts[dst].point,
                         colors[src % colors.size()],
                         line_thickness, cv::LINE_AA);
            }
        }
    }
}

inline void draw_segmentations(cv::Mat &image,
                               const std::vector<prediction> &predictions,
                               const std::unordered_map<int, std::string> &labels,
                               const std::vector<cv::Scalar> &colors,
                               bool drawBox = false,
                               float maskAlpha = 0.3f)
{
    if (predictions.empty())
        return;

    if (drawBox)
        draw_bboxes(image, predictions, labels, colors, maskAlpha);

    for (const auto& prediction : predictions) {
        if (prediction.label_id < 0
            || prediction.label_id >= labels.size()
            || prediction.mask.empty())
            continue;

        const cv::Scalar& color = colors.empty() ? cv::Scalar(0, 0, 255) : colors[prediction.label_id % colors.size()];
        constexpr float dark_factor = 0.7;
        const cv::Scalar& darker_color{color[0] * dark_factor,
                                       color[1] * dark_factor,
                                       color[2] * dark_factor };

        cv::Mat colored_mask(image.size(), image.type(), cv::Scalar(0, 0, 0));
        colored_mask.setTo(darker_color, prediction.mask);

        cv::addWeighted(image, 1.0, colored_mask, maskAlpha, 0, image);
    }
}

}
