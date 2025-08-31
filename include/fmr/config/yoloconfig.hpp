#pragma once

#include <string>
#include <optional>
#include <array>
#include <unordered_map>

namespace fmr {
    struct yolo_config {
        typedef enum {
            Uknown,
            Detect,
            OBB,
            Pose,
            Segment,
            Classify
        } task_t;

        std::optional<std::string> name;
        std::optional<std::string> version;
        std::optional<task_t> task;
        std::optional<int> stride;
        std::optional<int> batch;
        std::optional<int> channels;
        std::optional<std::array<int, 2>> imgsz;
        std::optional<std::unordered_map<int, std::string>> names;
        std::optional<float> confidence;
        std::optional<float> iou_threshold;
        std::optional<std::array<int, 2>> kpt_shape;
        std::optional<std::vector<std::pair<int, int>>> kpt_skeleton;

        static task_t taskForString(const std::string &task)
        {
            if (task == "detect")
                return Detect;

            if (task == "segment")
                return Segment;

            if (task == "pose")
                return Pose;

            if (task == "classify")
                return Classify;

            if (task == "obb")
                return OBB;

            return Uknown;
        }

    };
}
