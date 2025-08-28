#pragma once

struct box_info {
    cv::Rect box;
    cv::Rect nms_box;
    float conf;
    int class_id;
};

struct obb_info {
    cv::RotatedRect box;
    cv::RotatedRect nms_box;
    float conf;
    int class_id;
};
