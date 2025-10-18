#pragma once

#include <vector>

#include <spdlog/spdlog.h>
#include <opencv2/core/mat.hpp>
#include <yaml-cpp/yaml.h>

#include <fmr/core/types.hpp>
#include <fmr/core/image.hpp>
#include <fmr/accelarators/accelerator.hpp>
#include <fmr/config/yoloconfig.hpp>

namespace fmr {

class yolo {
public:
    explicit yolo(std::unique_ptr<accelerator> &inferSession, std::shared_ptr<yolo_config> config = nullptr, std::vector<cv::Scalar> colors = {});
    virtual ~yolo();
    virtual std::vector<predictions_t> predict(const std::vector<cv::Mat> &batch);
    virtual void draw(std::vector<cv::Mat> &batch, const std::vector<predictions_t>& predictionsList, float maskAlpha = 0.3f) const;
    virtual void draw(cv::Mat &img, const predictions_t& predictions, float maskAlpha = 0.3f) const;

    virtual bool has_dyn_batch();
    virtual bool has_dyn_shape();
    std::shared_ptr<yolo_config> config() const;
    const std::vector<cv::Scalar> &colors() const;
    void set_colors(const std::vector<cv::Scalar> &newColors);
    void set_logger(std::shared_ptr<spdlog::logger> logger);

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

    std::shared_ptr<spdlog::logger> m_logger;
};

// Definition

inline yolo::yolo(std::unique_ptr<accelerator> &inferSession, std::shared_ptr<yolo_config> config, std::vector<cv::Scalar> colors)
    : m_infer_session(inferSession)
    , m_config(config)
    , m_logger(spdlog::default_logger()->clone("fmr.yolo"))
{
    m_logger->set_level(spdlog::level::debug);

    if (!config)
        m_config = std::make_shared<yolo_config>();

    // Pull out yolo specific metadata
    const std::unordered_map<std::string, std::string> &metadata = m_infer_session->model_metadata();

    if ((!m_config->task || m_config->task == yolo_config::Uknown) && metadata.find("task") != metadata.end()) {
        m_config->task = yolo_config::taskForString(metadata.at("task"));
    } else {
        throw std::runtime_error("Please specify a valid yolo task in the config");
    }

    // Common
    if (!m_config->stride && metadata.find("stride") !=  metadata.end())
        m_config->stride = std::stoi(metadata.at("stride"));

    if (!m_config->batch) {
        // User didn't provide batch
        if (metadata.find("batch") !=  metadata.end()){
            // Assign what model metadata has
            m_config->batch = std::stoi(metadata.at("batch"));
        } else {
            // Fallback to 1 (dynamic) or input shape (fixed)
            m_config->batch = has_dyn_batch() ? 1 : inferSession->input_shapes().at(0).at(0);
        }
    } else {
        // Fixed shape? compare with input shape. if mismatch, enfore
        if (!has_dyn_batch()
            && inferSession->input_shapes().at(0).at(0) != m_config->batch.value())
            m_config->batch = inferSession->input_shapes().at(0).at(0);
    }

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

    // Specific
    std::mt19937 rng(std::random_device{}());
    if (m_config->task == yolo_config::Pose) {
        // pose
        if (!m_config->kpt_shape && metadata.find("kpt_shape") != metadata.end()) {
            YAML::Node kpt_shape = YAML::Load(metadata.at("kpt_shape"));
            if (kpt_shape)
                m_config->kpt_shape = kpt_shape.as<std::array<int, 2>>();
        }

        if (!m_config->kpt_skeleton) {
            if (metadata.find("kpt_skeleton") != metadata.end()) {
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
        }

        // colors
        int total_colors = 17; // default of yolo11-pose
        if (m_colors.empty() && m_config->kpt_shape && !m_config->kpt_shape->empty()) {
            // Generate colors based on the amount of keypoints vs number of classes
            total_colors = std::max(m_config->kpt_shape->at(0), static_cast<int>(m_config->names->size()));
        } else {
            m_logger->warn("Couldn't determine kpt_shape and generate colors, using default 17 colors");
        }

        m_colors = generate_colors(total_colors, rng);
    }

    if (m_colors.empty())
        m_colors = generate_colors(m_config->names.value().size(), rng);
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
        m_logger->warn("Only one input tensor was expected, got {}. Skipping prediction!", input_shapes.size());
        return {};
    }

    const int batch_size = m_config->batch.value();
    std::vector<int64_t> input_shape = input_shapes.at(0); // BCHW
    std::vector<predictions_t> predictions_list;
    predictions_list.reserve(batch.size());

    const auto task = m_config->task.value_or(yolo_config::Uknown);
    for(size_t b = 0; b < batch.size();) {
        const size_t sel_end = batch_size < 0                                   // batch is set to -1
                                   ? batch.size()                               // use the whole batch
                                   : std::min(batch.size(), b + batch_size);    // else, the specific size
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
                    max_w = m_config->imgsz->at(0);
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

        const cv::Size new_shape(max_w, max_h);
        std::vector<cv::Mat> sel_batch;
        for (size_t s = 0; s < sel_size; ++s) {
            const cv::Mat img = batch[b + s];
            cv::Mat resized_img;

            if (task == yolo_config::Classify) {
                cv::resize(img, resized_img, new_shape);
            } else {
                letter_box(img, resized_img, new_shape);
            }

            cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB);
            normalize(resized_img);

            sel_batch.emplace_back(resized_img);
        }

        std::vector<std::vector<float>> inputs(1,  std::vector<float>(vec_product(input_shape), 0.0f));
        permute(sel_batch, inputs[0]);

        m_infer_session->predict_raw(inputs, { input_shape });

        cv::Size resized_size(input_shape[3], input_shape[2]);
        std::vector<predictions_t> predictions;

        switch (m_config->task.value_or(yolo_config::Uknown)) {
        case yolo_config::Detect:
            predictions = postprocess_detections(batch, b, sel_size, resized_size);
            break;
        case yolo_config::OBB:
            predictions = postprocess_obb_detections(batch, b, sel_size, resized_size);
            break;
        case yolo_config::Pose:
            predictions = postprocess_keypoints(batch, b, sel_size, resized_size);
            break;
        case yolo_config::Segment:
            predictions = postprocess_segmentations(batch, b, sel_size, resized_size);
            break;
        case yolo_config::Classify:
            predictions = postprocess_classifications(batch, b, sel_size, resized_size);
            break;
        default:
            break;
        }

        predictions_list.insert(predictions_list.end(), predictions.begin(), predictions.end());
        b = sel_end;
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
        draw_bboxes(img, predictions, m_config->names.value(), m_colors);
        break;
    case yolo_config::OBB:
        draw_obbs(img, predictions, m_config->names.value(), m_colors, maskAlpha);
        break;
    case yolo_config::Pose:
        draw_keypoints(img, predictions, m_config->kpt_skeleton.value(), m_config->names.value(), m_colors);
        break;
    case yolo_config::Segment:
        draw_segmentations(img, predictions, m_config->names.value(), m_colors, true);
        break;
    case yolo_config::Classify:
        draw_classifications(img, predictions, m_config->names.value(), m_colors);
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
        m_logger->warn("Colors and names set mismatch, defaults will be used: {} != {}", names_size, newColors.size());
        return;
    }

    m_colors = newColors;
}

inline void yolo::set_logger(std::shared_ptr<spdlog::logger> logger)
{
    m_logger = logger;
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
        const cv::Size orig_img_size(batch[batch_indx + p].cols, batch[batch_indx + p].rows);

        std::vector<cv::Rect> boxes;
        std::vector<float> scores;
        std::vector<int> class_ids;
        std::vector<cv::Rect> nms_boxes;

        /* i.e.
         * We got 8400 detections, per 84 features, per image in a batch.
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
            // Since its [B, F, D], not [B, D, F]. we hover over each feature's detections.
            // We find the score first, to filter out low confidence detections, even though
            // it comes after the box values
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

            const float cx = batch_offsetptr[0 * out_num_detections + i];
            const float cy = batch_offsetptr[1 * out_num_detections + i];
            const float w = batch_offsetptr[2 * out_num_detections + i];
            const float h = batch_offsetptr[3 * out_num_detections + i];

            const cv::Rect box(cx - w / 2.0f, cy - h / 2.0f, w, h); // from (cx, cy, w, h) to (x, y, w, h)
            // Adjust NMS box coordinates to prevent overlap between classes
            cv::Rect nms_box = box;
            nms_box.x += class_id * 7880;
            nms_box.y += class_id * 7880;

            boxes.emplace_back(box);
            scores.emplace_back(max_score);
            class_ids.emplace_back(class_id);
            nms_boxes.emplace_back(nms_box);
        }

        // Apply Non-Maximum Suppression (NMS) to eliminate redundant detections
        const std::vector<int> indices = nms_bboxes(nms_boxes, scores, m_config->confidence.value_or(0.4f), m_config->iou_threshold.value_or(0.4f));

        const auto &labels = m_config->names.value();
        predictions_t results;
        results.reserve(indices.size());
        for (const int idx : indices) {
            prediction prediction;
            prediction.box = scale_coords(res_size, boxes[idx], orig_img_size);
            prediction.conf = scores[idx];
            prediction.label_id = class_ids[idx];
            prediction.label = labels.at(prediction.label_id);

            results.emplace_back(prediction);
        }

        predictions_list.emplace_back(results);
    }

    return predictions_list;
}

inline std::vector<predictions_t> yolo::postprocess_obb_detections(const std::vector<cv::Mat> &batch, int batch_indx, int sel_batch_size, cv::Size res_size)
{
    // Expected shape [1, num_features, num_detections]
    const std::vector<int64_t> shape0 = m_infer_session->tensor_shape(0);
    const float* output0_data = m_infer_session->tensor_data(0);

    const int out_batch_size = shape0.at(0);
    const int out_num_features = shape0.at(1);
    const int out_num_detections = shape0.at(2);
    const int out_num_classes = out_num_features - 5; // 5 is cx, cy, w, h, angle

    std::vector<predictions_t> predictions_list;
    predictions_list.reserve(sel_batch_size);
    for (size_t p = 0; p < sel_batch_size; ++p) {
        std::vector<cv::RotatedRect> obbs;
        std::vector<cv::RotatedRect> nms_obbs_;
        std::vector<float> scores;
        std::vector<int> class_ids;

        /*
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
        const float *batch_offsetptr = output0_data + p * (out_num_features * out_num_detections);
        for (size_t i = 0; i < out_num_detections; ++i) {
            // Since its [B, F, D], not [B, D, F]. we hover over each feature's detections
            // expected layout: cx, cy, w, h, scores, angle
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

            const float cx = batch_offsetptr[0 * out_num_detections + i];
            const float cy = batch_offsetptr[1 * out_num_detections + i];
            const float w = batch_offsetptr[2 * out_num_detections + i];
            const float h = batch_offsetptr[3 * out_num_detections + i];
            const float angle_rad = batch_offsetptr[(4 + out_num_classes) * out_num_detections + i];
            const float angle_deg = angle_rad * 180.0f / CV_PI;

            const cv::Size orig_size(batch[batch_indx + p].cols, batch[batch_indx + p].rows);
            const cv::RotatedRect coords(cv::Point2f(cx, cy), cv::Size2f(w, h), angle_deg);
            const cv::RotatedRect scaled_obb = scale_coords(res_size, coords, orig_size, true);

            // Adjust NMS box coordinates to prevent overlap between classes
            cv::RotatedRect nms_obb = scaled_obb;
            // Arbitrary offset to differentiate classes
            nms_obb.center.x += class_id * 7880;
            nms_obb.center.y += class_id * 7880;

            obbs.emplace_back(scaled_obb);
            scores.emplace_back(max_score);
            class_ids.emplace_back(class_id);
            nms_obbs_.emplace_back(nms_obb);
        }

        // Apply Non-Maximum Suppression (NMS) to eliminate redundant detections
        const std::vector<int> indices = nms_obbs(nms_obbs_, scores, m_config->confidence.value_or(0.4f), m_config->iou_threshold.value_or(0.4f));

        const auto &labels = m_config->names.value();
        predictions_t results;
        results.reserve(indices.size());
        for (const int idx : indices) {
            prediction prediction;
            prediction.obb = obbs[idx];
            prediction.conf = scores[idx];
            prediction.label_id = class_ids[idx];
            prediction.label = labels.at(prediction.label_id);

            results.emplace_back(prediction);
        }

        predictions_list.emplace_back(results);
    }


    return predictions_list;
}

inline std::vector<predictions_t> yolo::postprocess_keypoints(const std::vector<cv::Mat> &batch, int batch_indx, int sel_batch_size, cv::Size res_size)
{
    const std::vector<int64_t> shape0 = m_infer_session->tensor_shape(0);
    const float* output0_data = m_infer_session->tensor_data(0);

    // [N, 4 + num_classes + kpts * 3, num_preds]
    // [N, cxcywh + classes_scores + kpts * xyv, num_preds] (expected).
    const size_t out_batch_size = shape0.at(0);
    const size_t out_num_features = shape0.at(1);
    const size_t out_num_detections = shape0.at(2);
    const int num_keypoints = m_config->kpt_shape->at(0);
    const int num_kp_features = m_config->kpt_shape->at(1);
    const int out_num_classes = out_num_features - 4 - num_keypoints * num_kp_features;

    std::vector<predictions_t> results_list;
    results_list.reserve(sel_batch_size);
    for (size_t b = 0; b < sel_batch_size; ++b) {
        std::vector<cv::Rect> boxes;
        std::vector<cv::Rect> nms_boxes;
        std::vector<std::vector<keypoint>> keypoints_list;
        std::vector<float> scores;
        std::vector<int> class_ids;

        /*
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
        const float *batch_offsetptr = output0_data + b * (out_num_features * out_num_detections); // Jumps b * features * predictions for batch b.
        for (size_t i = 0; i < out_num_detections; ++i) {
            // Since its [B, F, D], not [B, D, F]. We hover over each feature's detections
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

            const float cx = batch_offsetptr[0 * out_num_detections + i];
            const float cy = batch_offsetptr[1 * out_num_detections + i];
            const float w = batch_offsetptr[2 * out_num_detections + i];
            const float h = batch_offsetptr[3 * out_num_detections + i];

            const cv::Size orig_size(batch[batch_indx + b].cols, batch[batch_indx + b].rows);
            const cv::Rect coords(cx - w / 2.0f, cy - h / 2.0f, w, h);
            const cv::Rect scaled_box = scale_coords(res_size, coords, orig_size, true);

            cv::Rect nms_box = scaled_box;
            nms_box.x += class_id * 7880; // arbitrary offset to differentiate classes
            nms_box.y += class_id * 7880;

            std::vector<keypoint> keypoints;
            for (int k = 0; k < num_keypoints; ++k) {
                const int kp_offset = 4 + out_num_classes + k * num_kp_features;
                const float x = batch_offsetptr[(0 + kp_offset) * out_num_detections + i];
                const float y = batch_offsetptr[(1 + kp_offset) * out_num_detections + i];
                const float conf = batch_offsetptr[(2 + kp_offset) * out_num_detections + i];

                const cv::Point2f kpt = scale_coords(res_size, cv::Point2f(x, y), orig_size);
                keypoints.emplace_back(keypoint{ kpt, conf });
            }

            boxes.emplace_back(scaled_box);
            nms_boxes.emplace_back(nms_box);
            keypoints_list.emplace_back(keypoints);
            scores.emplace_back(max_score);
            class_ids.emplace_back(class_id);
        }

        // Apply Non-Maximum Suppression (NMS) to eliminate redundant detections
        const std::vector<int> indices = nms_bboxes(nms_boxes, scores, m_config->confidence.value_or(0.4f), m_config->iou_threshold.value_or(0.4f));

        const auto &labels = m_config->names.value();
        predictions_t results;
        results.reserve(indices.size());
        for (const int idx : indices) {
            prediction prediction;
            prediction.box = boxes[idx];
            prediction.points = keypoints_list[idx];
            prediction.conf = scores[idx];
            prediction.label_id = class_ids[idx];
            prediction.label = labels.at(prediction.label_id);

            results.emplace_back(prediction);
        }

        results_list.emplace_back(results);
    }

    return results_list;
}

inline std::vector<predictions_t> yolo::postprocess_segmentations(const std::vector<cv::Mat> &batch, int batch_indx, int sel_batch_size, cv::Size res_size)
{
    const std::vector<int64_t> shape0 = m_infer_session->tensor_shape(0); // [B, F, D]    -> [1, 116, num_detections] - e.g 80 class + 4 bbox parms + 32 seg masks = 116
    const std::vector<int64_t> shape1 = m_infer_session->tensor_shape(1); // [B, M, H, W] -> [1, 32, maskH, maskW]
    const float* output0_data = m_infer_session->tensor_data(0);
    const float* output1_data = m_infer_session->tensor_data(1);

    const int out_batch_size = shape0.at(0);
    const int out_num_features = shape0.at(1);
    const int out_num_detections = shape0.at(2);

    const int out_num_masks = shape1.at(1);
    const int out_mask_h = shape1.at(2);
    const int out_mask_w = shape1.at(3);
    const int out_mask_size = out_mask_h * out_mask_w;

    const int out_num_classes = out_num_features - 4 - out_num_masks; // 4 is cx, cy, w, h + 32 masks

    std::vector<predictions_t> predictions_list;
    predictions_list.reserve(sel_batch_size);
    for (size_t p = 0; p < sel_batch_size; ++p) {

        /*
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
        const float *batch_offsetptr0 = output0_data + p * (out_num_features * out_num_detections);
        const float *batch_offsetptr1 = output1_data + p * (out_num_masks * out_mask_size);

        cv::Mat prototype_flat(out_num_masks, out_mask_size, CV_32F);
        for(int m = 0; m < out_num_masks; ++m) {
            const float *mask_ptr = batch_offsetptr1 + m * out_mask_size;
            cv::Mat row = prototype_flat.row(m); // 1 x (H*W)
            std::memcpy(row.ptr<float>(), mask_ptr, sizeof(float) * out_mask_size);
        }

        std::vector<cv::Rect> boxes;
        std::vector<cv::Rect> nms_boxes;
        std::vector<float> scores;
        std::vector<int> class_ids;
        std::vector<std::vector<float>> mask_coefs_list;

        for (size_t i = 0; i < out_num_detections; ++i) {
            // Since its [B, F, D], not [B, D, F]. we hover over each feature's detections.
            // We find the score first, to filter out low confidence detections, even though
            // it comes after the box values
            int class_id = -1;
            float max_score = 0.0f;
            for (int c = 0; c < out_num_classes; ++c) {
                const float score = batch_offsetptr0[(4 + c) * out_num_detections + i];
                if (max_score < score) {
                    max_score = score;
                    class_id = c;
                }
            }

            if (max_score < m_config->confidence.value_or(0.4f))
                continue;

            const float cx = batch_offsetptr0[0 * out_num_detections + i];
            const float cy = batch_offsetptr0[1 * out_num_detections + i];
            const float w = batch_offsetptr0[2 * out_num_detections + i];
            const float h = batch_offsetptr0[3 * out_num_detections + i];

            std::vector<float> mask_coefs(out_num_masks);
            for(int m = 0; m < out_num_masks; ++m) {
                mask_coefs[m] = batch_offsetptr0[(4 + out_num_classes + m) * out_num_detections + i];
            }

            const cv::Rect box(cx - w / 2.0f, cy - h / 2.0f, w, h); // from (cx, cy, w, h) to (x, y, w, h)
            // Adjust NMS box coordinates to prevent overlap between classes
            cv::Rect nms_box = box;
            nms_box.x += class_id * 7880;
            nms_box.y += class_id * 7880;

            boxes.emplace_back(box);
            nms_boxes.emplace_back(nms_box);
            scores.emplace_back(max_score);
            class_ids.emplace_back(class_id);
            mask_coefs_list.emplace_back(mask_coefs);
        }

        // Apply Non-Maximum Suppression (NMS) to eliminate redundant detections
        const std::vector<int> indices = nms_bboxes(nms_boxes, scores, m_config->confidence.value_or(0.4f), m_config->iou_threshold.value_or(0.4f));

        const cv::Size orig_img_size(batch[batch_indx + p].cols, batch[batch_indx + p].rows);
        const float gain = std::min(static_cast<float>(res_size.height) / orig_img_size.height,
                                    static_cast<float>(res_size.width) / orig_img_size.width);
        const int scaled_w = static_cast<int>(orig_img_size.width * gain);
        const int scaled_h = static_cast<int>(orig_img_size.height * gain);
        const float pad_w = (res_size.width - scaled_w) / 2.0f;
        const float pad_h = (res_size.height - scaled_h) / 2.0f;
        const float mask_scale_x = static_cast<float>(out_mask_w) / res_size.width;
        const float mask_scale_y = static_cast<float>(out_mask_h) / res_size.height;
        int x1 = static_cast<int>(std::round((pad_w - 0.1f) * mask_scale_x));
        int y1 = static_cast<int>(std::round((pad_h - 0.1f) * mask_scale_y));
        int x2 = static_cast<int>(std::round((res_size.width - pad_w + 0.1f) * mask_scale_x));
        int y2 = static_cast<int>(std::round((res_size.height - pad_h + 0.1f) * mask_scale_y));
        x1 = std::max(0, std::min(x1, out_mask_w - 1));
        y1 = std::max(0, std::min(y1, out_mask_h - 1));
        x2 = std::max(x1, std::min(x2, out_mask_w));
        y2 = std::max(y1, std::min(y2, out_mask_h));

        if (x2 <= x1 || y2 <= y1) {
            // Skip all detections, no valid crop possible
            return predictions_list;
        }

        const cv::Rect crop_rect(x1, y1, x2 - x1, y2 - y1);
        cv::Mat mask_flat;         // 1 x (H*W) float
        cv::Mat final_mask_2d;     // H x W float
        cv::Mat resized_mask;      // orig_img_size float/uchar as needed
        cv::Mat binary_mask;
        cv::Mat mask_canvas(orig_img_size, CV_8U); // final mask canvas per-prediction (reused)

        const auto &labels = m_config->names.value();
        predictions_t results;
        results.reserve(indices.size());
        for (const int idx : indices) {
            prediction prediction;
            prediction.box = scale_coords(res_size, boxes[idx], orig_img_size, true, gain, pad_w, pad_h);
            prediction.conf = scores[idx];
            prediction.label_id = class_ids[idx];
            prediction.label = labels.at(prediction.label_id);

            // Process mask
            cv::Mat coeffs(1, out_num_masks, CV_32F); // coeffs: 1 x M
            const std::vector<float>& mask_coefs = mask_coefs_list[idx];

            std::memcpy(coeffs.ptr<float>(), mask_coefs.data(), sizeof(float) * out_num_masks);
            mask_flat = coeffs * prototype_flat; // -> (1 x N)
            final_mask_2d = mask_flat.reshape(1, out_mask_h); // to 2D: (H x W)

            if (final_mask_2d.isContinuous())
                final_mask_2d = final_mask_2d.clone();

            final_mask_2d = sigmoid(final_mask_2d);

            const cv::Mat cropped_mask = final_mask_2d(crop_rect);
            cv::resize(cropped_mask, resized_mask, orig_img_size);

            cv::threshold(resized_mask, binary_mask, 0.5, 255.0, cv::THRESH_BINARY);
            if (binary_mask.type() != CV_8U)
                binary_mask.convertTo(binary_mask, CV_8U); // values 0 or 255

            mask_canvas.setTo(0);
            cv::Rect roi = prediction.box & cv::Rect(0, 0, binary_mask.cols, binary_mask.rows);
            if (roi.area() > 0)
                binary_mask(roi).copyTo(mask_canvas(roi));

            prediction.mask = mask_canvas.clone();
            results.emplace_back(prediction);
        }

        predictions_list.emplace_back(results);
    }

    return predictions_list;
}

inline std::vector<predictions_t> yolo::postprocess_classifications(const std::vector<cv::Mat> &batch, int batch_indx, int sel_batch_size, cv::Size res_size)
{
    const std::vector<int64_t> shape0 = m_infer_session->tensor_shape(0); // expected [B, 1000]
    const float* output0_data = m_infer_session->tensor_data(0);

    const int out_batch_size = shape0.at(0);
    const int out_num_logits = shape0.at(1);

    const float max_probability = m_config->confidence.value_or(0.0010f); // equvalent of 0.10, if 0.0010 * 100

    std::vector<predictions_t> predictions_list;
    predictions_list.reserve(sel_batch_size);
    for (size_t p = 0; p < sel_batch_size; ++p) {
        const float *batch_offsetptr = output0_data + p * out_num_logits;

        // Max logit for numerical stability
        float max_logit = *std::max_element(batch_offsetptr, batch_offsetptr + out_num_logits);

        // Softmax probabilities
        float sum_exp = 0.0f;
        std::vector<float> exps(out_num_logits);
        for (int i = 0; i < out_num_logits; ++i) {
            exps[i] = std::exp(batch_offsetptr[i] - max_logit);
            sum_exp += exps[i];
        }

        const auto &labels = m_config->names.value();
        predictions_t predictions;
        for (int i = 0; i < out_num_logits; ++i) {
            const float probability = exps[i] / sum_exp;
            if (probability < max_probability)
                continue;

            prediction prediction;
            prediction.conf = probability;
            prediction.label_id = i;
            prediction.label = labels.at(prediction.label_id);
            predictions.emplace_back(prediction);
        }

        std::sort(predictions.begin(), predictions.end(),
                  [](const prediction& a, const prediction& b) {
                      return a.conf > b.conf;
                  });

        predictions_list.emplace_back(predictions);
    }

    return predictions_list;
}

}
