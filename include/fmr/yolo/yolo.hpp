#pragma once

#include <vector>

#include <spdlog/spdlog.h>
#include <opencv2/core/mat.hpp>
#include <yaml-cpp/yaml.h>

#include <fmr/accelarators/accelerator.hpp>
#include <fmr/config/yoloconfig.hpp>
#include <fmr/core/prediction.hpp>
#include <fmr/core/image.hpp>

namespace fmr {

class yolo {
public:
    explicit yolo(std::unique_ptr<accelerator> &inferSession, std::shared_ptr<yolo_config> config = nullptr);
    virtual ~yolo();
    virtual std::vector<predictions_t> predict(const std::vector<cv::Mat> &batch);
    virtual void draw(std::vector<cv::Mat> &batch, const std::vector<predictions_t>& predictionsList, float maskAlpha = 0.3f) const;
    virtual void draw(cv::Mat &img, const predictions_t& predictions, float maskAlpha = 0.3f) const;

    virtual bool has_dyn_batch();
    virtual bool has_dyn_shape();
    std::shared_ptr<yolo_config> config() const;
    const std::vector<cv::Scalar> &colors() const;
    void set_colors(const std::vector<cv::Scalar> &newColors);

protected:
    std::unique_ptr<accelerator> &infer_session();
    virtual std::vector<predictions_t> postprocess_detections(const std::vector<cv::Mat> &batch, int batch_indx, int sel_batch_size, cv::Size res_size);
    virtual std::vector<predictions_t> postprocess_obb_detections(const std::vector<cv::Mat> &batch, int batch_indx, int sel_batch_size, cv::Size res_size);
    virtual std::vector<predictions_t> postprocess_keypoints(const std::vector<cv::Mat> &batch, int batch_indx, int sel_batch_size, cv::Size res_size);
    virtual std::vector<predictions_t> postprocess_segmentations(const std::vector<cv::Mat> &batch, int batch_indx, int sel_batch_size, cv::Size res_size);
    virtual std::vector<predictions_t> postprocess_classifications(const std::vector<cv::Mat> &batch, int batch_indx, int sel_batch_size, cv::Size res_size);

private:
    std::unique_ptr<accelerator> &m_infer_session;
    std::shared_ptr<yolo_config> m_config;
    std::vector<cv::Scalar> m_colors;

    std::shared_ptr<spdlog::logger> m_logger = spdlog::default_logger()->clone("fmr.yolo");
};

// definition

inline yolo::yolo(std::unique_ptr<accelerator> &inferSession, std::shared_ptr<yolo_config> config)
    : m_infer_session(inferSession)
    , m_config(config)
{
    if (!config)
        m_config = std::make_shared<yolo_config>();

    // pull out yolo specific metadata
    const std::unordered_map<std::string, std::string> &metadata = m_infer_session->model_metadata();

    if ((!m_config->task || m_config->task == yolo_config::Uknown) && metadata.find("task") != metadata.end()) {
        m_config->task = yolo_config::taskForString(metadata.at("task"));
    } else {
        throw std::runtime_error("Please specify a valid yolo task in the config");
    }

    // common
    if (!m_config->stride && metadata.find("stride") !=  metadata.end())
        m_config->stride = std::stoi(metadata.at("stride"));

    // TODO: make sure all models satisfy this
    if (!m_config->batch && metadata.find("batch") !=  metadata.end())
        m_config->batch = std::stoi(metadata.at("batch"));

    if (!m_config->imgsz && metadata.find("imgsz") != metadata.end()) {
        YAML::Node node = YAML::Load(metadata.at("imgsz"));
        if (node)
            m_config->imgsz = node.as<std::array<int, 2>>();
    }

    YAML::Node node;
    if (!m_config->names
        && metadata.find("names") != metadata.end()
        && (node = YAML::Load(metadata.at("names")))) {

        m_config->names = node.as<std::unordered_map<int, std::string>>();
    }
    else
        throw std::runtime_error("No labels/names/classes were found. Please provide labels for this model!");

    // specific
    if (m_config->task == yolo_config::Detect) {
        // detect
        m_colors = generate_colors(m_config->names.value());

    } else if (m_config->task == yolo_config::Pose) {
        // pose
        if (!m_config->kpt_shape && metadata.find("kpt_shape") != metadata.end()) {
            YAML::Node kpt_shape = YAML::Load(metadata.at("kpt_shape"));
            if (kpt_shape)
                m_config->kpt_shape = std::array<int, 2>();
        }

        if (!m_config->kpt_skeleton && metadata.find("kpt_skeleton") != metadata.end()) {
            YAML::Node kpt_skeleton = YAML::Load(metadata.at("kpt_skeleton"));
            if (kpt_skeleton)
                m_config->kpt_skeleton = kpt_skeleton.as<std::vector<std::pair<int, int>>>();
        } else {
            m_config->kpt_skeleton = {
                // face connections
                {0,1}, {0,2}, {1,3}, {2,4},
                // head-to-shoulder connections
                {3,5}, {4,6},
                // arms
                {5,7}, {7,9}, {6,8}, {8,10},
                // body
                {5,6}, {5,11}, {6,12}, {11,12},
                // legs
                {11,13}, {13,15}, {12,14}, {14,16}
            };
        }

        // colors
        m_colors = {
            cv::Scalar(0,128,255),    // 0
            cv::Scalar(51,153,255),   // 1
            cv::Scalar(102,178,255),  // 2
            cv::Scalar(0,230,230),    // 3
            cv::Scalar(255,153,255),  // 4
            cv::Scalar(255,204,153),  // 5
            cv::Scalar(255,102,255),  // 6
            cv::Scalar(255,51,255),   // 7
            cv::Scalar(255,178,102),  // 8
            cv::Scalar(255,153,51),   // 9
            cv::Scalar(153,153,255),  // 10
            cv::Scalar(102,102,255),  // 11
            cv::Scalar(51,51,255),    // 12
            cv::Scalar(153,255,153),  // 13
            cv::Scalar(102,255,102),  // 14
            cv::Scalar(51,255,51),    // 15
            cv::Scalar(0,255,0),      // 16
            cv::Scalar(255,0,0),      // 17
            cv::Scalar(0,0,255),      // 18
            cv::Scalar(255,255,255)   // 19
        };
    }

}

inline yolo::~yolo()
{}

inline std::vector<predictions_t> yolo::predict(const std::vector<cv::Mat> &batch)
{
    if (batch.empty())
        return {};

    const auto &input_shapes = m_infer_session->input_shapes();
    // TODO: Support for models with more than 1 input shapes
    if(input_shapes.size() != 1) {
        m_logger->warn(fmt::format("Only one input tensor was expected, got {}. Skipping prediction!", input_shapes.size()));
        return {};
    }

    std::vector<int64_t> input_shape(input_shapes.at(0)); // BCHW
    std::vector<predictions_t> predictions_list;
    predictions_list.reserve(batch.size());

    for(size_t b = 0; b < batch.size(); ++b) {
        const size_t sel_end = std::max(batch.size(),  b + m_config->batch.value_or(1));
        const size_t sel_size = sel_end - b;

        int max_h = input_shape[2];
        int max_w = input_shape[3];

        if (has_dyn_shape()) {
            int model_stride = m_config->stride.value_or(32);

            if (max_h == -1) {
                if (m_config->imgsz) {
                    max_h = m_config->imgsz->at(0);
                } else {
                    for (size_t s = 0; s < sel_size; ++s)
                        max_h = std::max(max_h, batch[b + s].rows);

                    if (max_h % model_stride != 0)
                        max_h = ((max_h / model_stride) + 1) * model_stride;
                }

                input_shape[2] = max_h;
            }

            if (max_w == -1) {
                if (m_config->imgsz) {
                    max_h = m_config->imgsz->at(0);
                } else {
                    for (size_t s = 0; s < sel_size; ++s)
                        max_w = std::max(max_w, batch[b + s].cols);

                    if (max_w % model_stride != 0)
                        max_w = ((max_w / model_stride) + 1) * model_stride;
                }

                input_shape[3] = max_w;
            }

            input_shape[0] = sel_size;
        }

        std::vector<cv::Mat> sel_batch;
        for (size_t s = 0; s < sel_size; ++s) {
            cv::Mat resized_img;
            letter_box(batch[b + s], resized_img, cv::Size(max_w, max_h));
            sel_batch.emplace_back(resized_img);
        }

        std::vector<std::vector<float>> inputs(1,  std::vector<float>(vec_product(input_shape), 0.0f));
        norm_and_permute(sel_batch, inputs[0]);

        m_infer_session->predict_raw(inputs, { input_shape });

        cv::Size resized_size(input_shape[3], input_shape[2]);
        std::vector<predictions_t> predictions;
        switch (m_config->task.value_or(yolo_config::Uknown)) {
        case yolo_config::Detect: {
            predictions = postprocess_detections(batch, b, sel_size, resized_size);
            break;
        }
        case yolo_config::OBB: {
            predictions = postprocess_obb_detections(batch, b, sel_size, resized_size);
        }
        default:
            break;
        }

        predictions_list.insert(predictions_list.end(), predictions.begin(), predictions.end());
    }

    return predictions_list;
}

inline void yolo::draw(std::vector<cv::Mat> &batch, const std::vector<predictions_t> &predictionsList, float maskAlpha) const
{
    for (size_t i = 0; i < batch.size(); ++i) {
        draw(batch[i], predictionsList.at(i), maskAlpha);
    }
}

inline void yolo::draw(cv::Mat &img, const predictions_t &predictions, float maskAlpha) const
{
    yolo_config::task_t task = m_config->task.value_or(yolo_config::Uknown);
    switch (task) {
    case yolo_config::Detect:
        draw_bbox(img, predictions, m_config->names.value(), m_colors);
        break;
    default:
        break;
    }
}

inline bool yolo::has_dyn_batch()
{
    const auto input_shapes = m_infer_session->input_shapes();
    if (input_shapes.size() == 1                // is exactly one
        && input_shapes.at(0).size() == 4       // has size 4
        && input_shapes.at(0).at(0) == -1) {    // has 0 index equal -1
        return true;
    }

    return false;
}

inline bool yolo::has_dyn_shape()
{
    const auto input_shapes = m_infer_session->input_shapes();
    if (input_shapes.size() == 1                    // is exactly one
        && input_shapes.at(0).size() == 4           // has size 4
        && (input_shapes.at(0).at(2) == -1          // has 2 index equal -1 (height)
            || input_shapes.at(0).at(3) == -1)) {   // has 3 index equal -1 (width)
        return true;
    }

    return false;
}

inline std::shared_ptr<yolo_config> yolo::config() const
{
    return m_config;
}

inline const std::vector<cv::Scalar> &yolo::colors() const
{
    return m_colors;
}

inline void yolo::set_colors(const std::vector<cv::Scalar> &newColors)
{
    size_t names_size = m_config->names.value().size();
    if (names_size != newColors.size()) {
        m_logger->warn(fmt::format("Colors and names set mismatch, defaults will be used: {} != {}", names_size, newColors.size()));
        return;
    }

    m_colors = newColors;
}

inline std::unique_ptr<accelerator> &yolo::infer_session()
{
    return m_infer_session;
}

inline std::vector<predictions_t> yolo::postprocess_detections(const std::vector<cv::Mat> &batch, int batch_indx, int sel_batch_size, cv::Size res_size)
{
    const std::vector<int64_t> shape0 = m_infer_session->tensor_shape(0);
    const float* output0_data = m_infer_session->tensor_data(0);

    const int out_batch_size = shape0.at(0);
    const int out_num_features = shape0.at(1);
    const int out_num_detections = shape0.at(2);
    const int out_num_classes = out_num_features - 4; // 4 is cx, cy, w, h

    std::vector<predictions_t> predictions_list;
    predictions_list.reserve(sel_batch_size);
    for (size_t p = 0; p < sel_batch_size; ++p) {
        std::vector<box_info> boxes;
        boxes.reserve(out_num_detections);

        /*
         * We got 8400 detections, of each one of 84 features, of each batch.
         * D (detections) being the fastest-varying, then F (features) and then B (batch)
         * The structure of each batch is laid out something like this for [B, F, D]
            Batch 0
                Feature 0
                    Detections 0, 1, ... D-1
                Feature 1
                    Detections 0, 1, ... D-1
                ...
            Batch 1
                Feature 0
                    Detections 0, 1, ... D-1
                Feature 1
                    Detections 0, 1, ... D-1
                ...
            ...
        */
        const float *batch_offsetptr = output0_data + p * (out_num_features * out_num_detections); // jumps p * 84 * 8400 for batch p.
        for (size_t i = 0; i < out_num_detections; ++i) {
            // since its [B, F, D], not [B, D, F]. we hover over each feature's detections
            float cx = batch_offsetptr[0 * out_num_detections + i];
            float cy = batch_offsetptr[1 * out_num_detections + i];
            float w = batch_offsetptr[2 * out_num_detections + i];
            float h = batch_offsetptr[3 * out_num_detections + i];

            int class_id = -1;
            float max_score = 0.0f;
            for (int c = 0; c < out_num_classes; ++c) {
                const float score = batch_offsetptr[(4 + c) * out_num_detections + i];
                if (max_score < score) {
                    max_score = score;
                    class_id = c;
                }
            }

            if (max_score < m_config->confidence.value_or(0.4f))
                continue;

            const cv::Size orig_size(batch[batch_indx + p].cols, batch[batch_indx + p].rows);
            // const cv::Size res_size(input_shape[3], input_shape[2]);
            const cv::Rect coords(cx - w / 2.0f, cy - h / 2.0f, w, h); // from (cx, cy, w, h) to (x, y, w, h)
            cv::Rect scaled_box = scale_coords(res_size, coords, orig_size, true);

            // round coordinates for integer pixel positions
            scaled_box.x = std::round(scaled_box.x);
            scaled_box.y = std::round(scaled_box.y);
            scaled_box.width = std::round(scaled_box.width);
            scaled_box.height = std::round(scaled_box.height);

            // adjust NMS box coordinates to prevent overlap between classes
            cv::Rect nms_box = scaled_box;
            // arbitrary offset to differentiate classes
            nms_box.x += class_id * 7880;
            nms_box.y += class_id * 7880;

            boxes.emplace_back(box_info{scaled_box, nms_box, max_score, class_id});
        }

        // apply Non-Maximum Suppression (NMS) to eliminate redundant detections
        const std::vector<int> indices = nms_bboxes(boxes, m_config->confidence.value_or(0.4f), m_config->iou_threshold.value_or(0.4f));

        const auto &labels = m_config->names.value();
        predictions_t results;
        results.reserve(indices.size());
        for (const int idx : indices) {
            prediction prediction;
            prediction.box = boxes[idx].box;
            prediction.conf = boxes[idx].conf;
            prediction.label_id = boxes[idx].class_id;
            prediction.label = labels.at(prediction.label_id);

            results.emplace_back(prediction);
        }

        predictions_list.emplace_back(results);
    }

    return predictions_list;
}

inline std::vector<predictions_t> yolo::postprocess_obb_detections(const std::vector<cv::Mat> &batch, int batch_indx, int sel_batch_size, cv::Size res_size)
{
    // expected shape [1, num_features, num_detections]
    const std::vector<int64_t> shape0 = m_infer_session->tensor_shape(0);
    const float* output0_data = m_infer_session->tensor_data(0);

    const int out_batch_size = shape0.at(0);
    const int out_num_features = shape0.at(1);
    const int out_num_detections = shape0.at(2);
    const int out_num_classes = out_num_features - 5; // 5 is cx, cy, w, h, angle

    std::vector<predictions_t> predictions_list;
    predictions_list.reserve(sel_batch_size);
    for (size_t p = 0; p < sel_batch_size; ++p) {
        std::vector<obb_info> boxes;
        boxes.reserve(out_num_detections);

        /*
         * We got 8400 detections, of each one of 84 features, of each batch.
         * D (detections) being the fastest-varying, then F (features) and then B (batch)
         * The structure of each batch is laid out something like this for [B, F, D]
            Batch 0
                Feature 0
                    Detections 0, 1, ... D-1
                Feature 1
                    Detections 0, 1, ... D-1
                ...
            Batch 1
                Feature 0
                    Detections 0, 1, ... D-1
                Feature 1
                    Detections 0, 1, ... D-1
                ...
            ...
        */
        const float *batch_offsetptr = output0_data + p * (out_num_features * out_num_detections); // jumps p * 84 * 8400 for batch p.
        for (size_t i = 0; i < out_num_detections; ++i) {
            // since its [B, F, D], not [B, D, F]. we hover over each feature's detections
            float cx = batch_offsetptr[0 * out_num_detections + i];
            float cy = batch_offsetptr[1 * out_num_detections + i];
            float w = batch_offsetptr[2 * out_num_detections + i];
            float h = batch_offsetptr[3 * out_num_detections + i];

            int class_id = -1;
            float max_score = 0.0f;
            for (int c = 0; c < out_num_classes; ++c) {
                const float score = batch_offsetptr[(4 + c) * out_num_detections + i];
                if (max_score < score) {
                    max_score = score;
                    class_id = c;
                }
            }

            const float angle = batch_offsetptr[(4 + out_num_classes) * out_num_detections];

            if (max_score < m_config->confidence.value_or(0.4f))
                continue;

            const cv::Size orig_size(batch[batch_indx + p].cols, batch[batch_indx + p].rows);
            // const cv::Size res_size(input_shape[3], input_shape[2]);
            const cv::RotatedRect coords(cv::Point2f(cx, cy), cv::Size2f(w, h), angle);
            cv::RotatedRect scaled_obb = scale_coords(res_size, coords, orig_size, true);

            // round coordinates for integer pixel positions
            scaled_obb.center.x = std::round(scaled_obb.center.x);
            scaled_obb.center.y = std::round(scaled_obb.center.y);
            scaled_obb.size.width = std::round(scaled_obb.size.width);
            scaled_obb.size.height = std::round(scaled_obb.size.height);

            // adjust NMS box coordinates to prevent overlap between classes
            cv::RotatedRect nms_obb = scaled_obb;
            // arbitrary offset to differentiate classes
            nms_obb.center.x += class_id * 7880;
            nms_obb.center.y += class_id * 7880;

            boxes.emplace_back(obb_info{scaled_obb, nms_obb, max_score, class_id});
        }

        // apply Non-Maximum Suppression (NMS) to eliminate redundant detections
        const std::vector<int> indices = nms_obbs(boxes, m_config->confidence.value_or(0.4f), m_config->iou_threshold.value_or(0.4f));

        const auto &labels = m_config->names.value();
        predictions_t results;
        results.reserve(indices.size());
        for (const int idx : indices) {
            prediction prediction;
            prediction.obb = boxes[idx].box;
            prediction.conf = boxes[idx].conf;
            prediction.label_id = boxes[idx].class_id;
            prediction.label = labels.at(prediction.label_id);

            results.emplace_back(prediction);
        }

        predictions_list.emplace_back(results);
    }


    return predictions_list;
}

inline std::vector<predictions_t> yolo::postprocess_keypoints(const std::vector<cv::Mat> &batch, int batch_indx, int sel_batch_size, cv::Size res_size)
{
    return {};
}

inline std::vector<predictions_t> yolo::postprocess_segmentations(const std::vector<cv::Mat> &batch, int batch_indx, int sel_batch_size, cv::Size res_size)
{
    return {};
}

inline std::vector<predictions_t> yolo::postprocess_classifications(const std::vector<cv::Mat> &batch, int batch_indx, int sel_batch_size, cv::Size res_size)
{
    return {};
}

}
