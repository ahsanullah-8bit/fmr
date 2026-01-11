#pragma once

#include <string>
#include <unordered_set>
#include <unordered_map>
#include <algorithm>
#include <fstream>

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

namespace fmr {

inline bool is_image(const std::string& filename)
{
    static const std::unordered_set<std::string> image_exts = {
        ".bmp", ".dib", ".jpg", ".jpeg", ".jpe", ".jp2", ".png",
        ".tif", ".tiff", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".webp"
    };

    // Extract lowercase extension from filename
    size_t dot_pos = filename.find_last_of('.');
    std::string ext = filename.substr(dot_pos);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (image_exts.count(ext))
        return true;

    return false;
}

inline bool is_video(const std::string& filename)
{
    static const std::unordered_set<std::string> video_exts = {
        ".avi", ".mp4", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv", ".flv"
    };

    // Extract lowercase extension from filename
    size_t dot_pos = filename.find_last_of('.');
    std::string ext = filename.substr(dot_pos);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (video_exts.count(ext))
        return true;

    return false;
}

inline std::unordered_map<int, std::string> read_names(const std::string &filepath)
{
    std::ifstream in_stream;
    in_stream.open(filepath, std::ios_base::in);

    if (!in_stream.is_open())
        return {};

    std::unordered_map<int, std::string> result;
    std::string name;
    for (size_t i = 0; !in_stream.eof(); ++i) {
        std::getline(in_stream, name);
        result[i] = name;
    }

    return result;
}

inline void draw_metrics(const std::vector<std::string> &metrics, cv::Mat &img)
{
    int baseline = 0;
    const int font_face = cv::FONT_HERSHEY_SIMPLEX;
    const double font_scale = std::min(img.cols, img.rows) * 0.0008;
    const int font_thickness = std::max(1, (int)(std::min(img.cols, img.rows) * 0.002));

    int y = 20; // start some pixels down from top
    for (const auto &metric : metrics) {
        cv::Size text_size = cv::getTextSize(metric, font_face, font_scale, font_thickness, &baseline);

        // Background box (a little padding)
        cv::rectangle(img,
                      cv::Point(0, y - text_size.height - baseline),
                      cv::Point(text_size.width + 5, y + baseline),
                      cv::Scalar(255, 255, 255), cv::FILLED);

        // Put the text
        cv::putText(img, metric, cv::Point(2, y),
                    font_face, font_scale,
                    cv::Scalar(0, 0, 0), font_thickness, cv::LINE_AA);

        // Pove y down for next line
        y += text_size.height + baseline + 5; // add spacing
    }
}

}
