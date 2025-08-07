#include <spdlog/spdlog.h>

#include <fmr/yolo/yolo.h>
#include <fmr/core/image.h>

namespace fmr {

inline std::shared_ptr<spdlog::logger> logger = spdlog::default_logger()->clone("fmr.yolo");

yolo::yolo(std::unique_ptr<accelerator> &inferSession, std::shared_ptr<yolo_config> config)
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

    if (!m_config->imgsz && metadata.find("imgsz") != metadata.end())
        m_config->imgsz = std::array<int, 2>();

    if (!m_config->names && metadata.find("names") != metadata.end())
        m_config->names = std::unordered_map<int, std::string>();
    else
        throw std::runtime_error("No labels/names/classes were found. Please provide labels for this model!");

    // specific
    if (m_config->task == yolo_config::Pose) {
        if (!m_config->kpt_shape && metadata.find("kpt_shape") != metadata.end())
            m_config->kpt_shape = std::array<int, 2>();
    }

    // colors
    m_colors = generate_colors(m_config->names.value());
}

yolo::~yolo()
{}

std::vector<predictions_t> yolo::predict(const std::vector<cv::Mat> &batch)
{
    if (batch.empty())
        return {};

    static const auto &input_shapes = m_infer_session->input_shapes();
    // TODO: Support for models with more than 1 input shapes
    if(input_shapes.size() != 1) {
        logger->warn(fmt::format("Only one input tensor was expected, got {}. Skipping prediction!", input_shapes.size()));
        return {};
    }

    std::vector<int64_t> input_shape(input_shapes.at(0)); // BCHW
    std::vector<predictions_t> predictions_list;
    predictions_list.reserve(batch.size());

    for(size_t b = 0; b < batch.size(); ++b) {
        const size_t sel_end = std::max(batch.size(),  b + m_config->batch.value_or(1));
        const size_t sel_size = sel_end - b;

        int max_w, max_h;
        for (size_t s = 0; s < sel_size; ++s) {
            max_w = std::max(max_w, batch[b + s].cols);
            max_h = std::max(max_h, batch[b + s].rows);
        }

        int model_stride = m_config->stride.value_or(32);
        if (max_h % model_stride != 0)
            max_h = ((max_h / model_stride) + 1) * model_stride;
        if (max_w % model_stride != 0)
            max_w = ((max_w / model_stride) + 1) * model_stride;

        input_shape[0] = sel_size;
        input_shape[2] = max_h;
        input_shape[3] = max_w;

        std::vector<cv::Mat> sel_batch;
        for (size_t s = 0; s < sel_size; ++s) {
            cv::Mat resized_img;
            letter_box(batch[b + s], resized_img, cv::Size(max_w, max_h));
            sel_batch.emplace_back(resized_img);
        }

        std::vector<std::vector<float>> inputs(1,  std::vector<float>(vec_product(input_shape), 0.0f));
        norm_and_permute(sel_batch, inputs[0]);

        m_infer_session->predict_raw(inputs, { input_shape });

        const std::vector<int64_t> shape0 = m_infer_session->tensor_shape(0);
        const float* output0_data = m_infer_session->tensor_data(0);

        const int out_batch_size = shape0.at(0);
        const int out_num_features = shape0.at(1);
        const int out_num_detections = shape0.at(2);
        const int out_num_classes = out_num_features - 4; // 4 is cx, cy, w, h

        if (out_batch_size != sel_size) {
            throw std::runtime_error(fmt::format("Batch mismatch, input {} != {} output", sel_size, out_batch_size));
        }

        predictions_list.reserve(out_batch_size);
        for (size_t p = 0; p < out_batch_size; ++p) {
            std::vector<cv::Rect> boxes;
            std::vector<float> confs;
            std::vector<int> class_ids;
            std::vector<cv::Rect> nms_boxes;
            boxes.reserve(out_num_detections);
            confs.reserve(out_num_detections);
            class_ids.reserve(out_num_detections);
            nms_boxes.reserve(out_num_detections);

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

                const cv::Size orig_size(batch[b + p].cols, batch[b + p].rows);
                const cv::Size res_size(input_shape[3], input_shape[2]);
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

                boxes.emplace_back(scaled_box);
                confs.emplace_back(max_score);
                class_ids.emplace_back(class_id);
                nms_boxes.emplace_back(nms_box);
            }

            // apply Non-Maximum Suppression (NMS) to eliminate redundant detections
            const std::vector<int> indices = nms_bboxes(nms_boxes, confs, m_config->confidence.value_or(0.4f), m_config->iou_threshold.value_or(0.4f));

            static const auto &labels = m_config->names.value();
            predictions_t results;
            results.reserve(indices.size());
            for (const int idx : indices) {
                prediction prediction;
                prediction.box = boxes[idx];
                prediction.conf = confs[idx];
                prediction.label_id = class_ids[idx];
                prediction.label = labels.at(prediction.label_id);

                results.emplace_back(prediction);
            }

            predictions_list.emplace_back(results);
        }

    }

    return predictions_list;
}

void yolo::draw(std::vector<std::vector<cv::Mat>> &batch, const std::vector<predictions_t> &predictionsList, float maskAlpha) const
{
    // TODO
}

void yolo::draw(cv::Mat &img, const predictions_t &predictions, float maskAlpha) const
{
    // TODO
    yolo_config::task_t task = m_config->task.value_or(yolo_config::Uknown);
    switch (task) {
    case yolo_config::Detect:
        draw_bbox(img, predictions, m_config->names.value(), m_colors);
        break;
    default:
        break;
    }
}

bool yolo::has_dyn_batch()
{
    // TODO
    return false;
}

bool yolo::has_dyn_shape()
{
    // TODO
    return false;
}

std::shared_ptr<yolo_config> yolo::config() const
{
    return m_config;
}

const std::vector<cv::Scalar> &yolo::colors() const
{
    return m_colors;
}

void yolo::set_colors(const std::vector<cv::Scalar> &newColors)
{
    size_t names_size = m_config->names.value().size();
    if (names_size != newColors.size()) {
        logger->warn(fmt::format("Colors and names set mismatch, defaults will be used: {} != {}", names_size, newColors.size()));
        return;
    }

    m_colors = newColors;
}

yolo::yolo(std::unique_ptr<accelerator> &inferSession)
    : m_infer_session(inferSession)
{}

std::unique_ptr<accelerator> &yolo::infer_session()
{
    return m_infer_session;
}

}
