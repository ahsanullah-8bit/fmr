#pragma once

#include <cstdint>
#include <array>
#include <vector>
#include <algorithm>

#include <clipper2/clipper.h>
#include <clipper2/clipper.offset.h>
#include <opencv2/highgui.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include <fmr/accelerators/accelerator.hpp>
#include <fmr/config/paddleocrconfig.hpp>
#include <fmr/core/types.hpp>
#include <fmr/core/image.hpp>
#include <fmr/core/dbpostprocess.hpp>

namespace fmr::paddle::ocr {

class detector {
public:
    explicit detector(accelerator *inferSession, const paddleocr_config &config);
    virtual ~detector();
    virtual std::vector<dets_t> predict(const std::vector<cv::Mat> &batch);

    virtual bool has_dyn_batch();
    virtual bool has_dyn_shape();

protected:
    virtual std::vector<dets_t> postprocess_detections(const std::vector<cv::Mat> &batch, int batch_indx, int sel_batch_size, cv::Size res_size, const std::vector<tensor> &outputs);
    std::vector<box_t> extract_text_boxes(const cv::Mat &probMap, const cv::Size &origSize);

private:
    accelerator *m_infer_session = nullptr;
    paddleocr_config m_config;
    std::vector<cv::Scalar> m_colors;

    std::shared_ptr<spdlog::logger> m_logger;
};

inline detector::detector(accelerator *inferSession,
                                    const paddleocr_config &config)
    : m_infer_session(inferSession)
    , m_config(config)
    , m_logger(spdlog::default_logger()->clone("fmr.paddleocr.det"))
{
    // Some defaults
    // Image mode
    if (!m_config.img_mode)
        m_config.img_mode = paddleocr_config::BGR;

    // Normalization order
    if (!m_config.norm_order)
        m_config.norm_order = paddleocr_config::hwc;

    // Mean
    if (!m_config.mean)
        m_config.mean = {0.485f, 0.456f, 0.406f};

    // Std
    if (!m_config.std)
        m_config.std = {0.229f, 0.224f, 0.225f};

    // Scale
    if (!m_config.scale)
        m_config.scale = 1.0f / 255;

    // Threshold
    if (!m_config.thresh)
        m_config.thresh = 0.3f;

    // Box threshold
    if (!m_config.box_thresh)
        m_config.box_thresh = 0.3f;

    // Minimum Box Size
    if (!m_config.min_box_size)
        m_config.min_box_size = 3; // pixels

    // Maximum No. Of Candidates
    if (!m_config.max_candidates)
        m_config.max_candidates = 1000;

    // Unclip ratio
    if (!m_config.unclip_ratio)
        m_config.unclip_ratio = 1.5f;

    // Unclip mode
    if (!m_config.unclip_mode)
        m_config.unclip_mode = paddleocr_config::Box;

    // CTC decoding mode
    if (!m_config.ctc_mode)
        m_config.ctc_mode = paddleocr_config::GreedySearch;

    // Batch
    auto inputs = m_infer_session->input_details();
    auto model_batch = static_cast<int>(inputs.at(0).shape.at(0));
    if (!m_config.batch) {
        // User didn't provide batch
        // Fallback to 1 (dynamic) or input shape (fixed)
        m_config.batch = has_dyn_batch() ? 1 : model_batch;
    } else {
        // Fixed shape? compare with input shape. if mismatch, enforce
        if (!has_dyn_batch()
            && model_batch != m_config.batch.value())
            m_config.batch = model_batch;
    }

    // Stride
    if (!m_config.stride) {
        m_config.stride = 32;
    }
}

inline detector::~detector()
{}

inline std::vector<dets_t> detector::predict(const std::vector<cv::Mat> &batch)
{
    if (batch.empty())
        return {};

    const int batch_size = m_config.batch.value();
    std::vector<dets_t> predictions_list;
    predictions_list.reserve(batch_size);

    for (size_t b = 0; b < batch.size();) {
        const size_t sel_end = batch_size < 0                                   // batch is set to -1
                                   ? batch.size()                               // use the whole batch
                                   : std::min(batch.size(), b + batch_size);    // else, the specific size
        const size_t sel_size = sel_end - b;

        auto inputs = m_infer_session->input_details();
        if(inputs.size() > 1) {
            m_logger->warn("This PaddleOCR detector implementation expects and uses only 1 input tensor, got {}.", inputs.size());
        }
        auto &input = inputs.at(0);
        int max_w = input.shape.at(3);
        int max_h = input.shape.at(2);

        if (has_dyn_shape()) {
            int model_stride = m_config.stride.value_or(32);

            if (max_h == -1) {
                if (m_config.imgsz && m_config.imgsz->at(0) > 0) {
                    max_h = m_config.imgsz->at(0);
                } else {
                    for (size_t s = 0; s < sel_size; ++s)
                        max_h = std::max(max_h, batch[b + s].rows);

                    if (max_h % model_stride != 0)
                        max_h = ((max_h / model_stride) + 1) * model_stride;
                }

                input.shape[2] = max_h;
            }

            if (max_w == -1) {
                if (m_config.imgsz && m_config.imgsz->at(1) > 0) {
                    max_w = m_config.imgsz->at(1);
                } else {
                    for (size_t s = 0; s < sel_size; ++s)
                        max_w = std::max(max_w, batch[b + s].cols);

                    if (max_w % model_stride != 0)
                        max_w = ((max_w / model_stride) + 1) * model_stride;
                }

                input.shape[3] = max_w;
            }

        }
        
        input.shape[0] = sel_size;

        const cv::Size resized_size(max_w, max_h);
        std::vector<cv::Mat> sel_batch;
        for (size_t s = 0; s < sel_size; ++s) {
            const cv::Mat img = batch[b + s];
            cv::Mat resized_img;

            cv::resize(img, resized_img, resized_size);
            cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB);
            normalize_imagenet(resized_img, m_config.mean.value(), m_config.std.value(), m_config.scale.value());

            sel_batch.emplace_back(resized_img);
        }

        std::vector<float> input_data(vec_product(input.shape), 0.0f);
        permute(sel_batch, input_data);

        input.data = reinterpret_cast<void*>(input_data.data());
        input.size_bytes = input_data.size() * sizeof(float);

        std::vector<tensor> outputs;
        m_infer_session->predict_raw(inputs, outputs);

        std::vector<dets_t> predictions = postprocess_detections(batch, b, sel_size, resized_size, outputs);
        predictions_list.insert(predictions_list.end(), predictions.begin(), predictions.end());
        b = sel_end;
    }

    return predictions_list;
}

inline bool detector::has_dyn_batch()
{
    const auto inputs = m_infer_session->input_details();
    return inputs.size() == 1                       // is exactly one
        && inputs.at(0).shape.size() == 4      // has size 4
        && inputs.at(0).shape.at(0) == -1; // has 0 index equal -1
}

inline bool detector::has_dyn_shape()
{
    const auto inputs = m_infer_session->input_details();
    return inputs.size() == 1                            // is exactly one
        && inputs.at(0).shape.size() == 4            // has size 4
        && (inputs.at(0).shape.at(2) == -1      // has 2 index equal -1 (height)
            || inputs.at(0).shape.at(3) == -1); // has 3 index equal -1 (width)
}

inline std::vector<dets_t> detector::postprocess_detections(const std::vector<cv::Mat> &batch,
                                                            int batch_indx,
                                                            int sel_batch_size,
                                                            cv::Size res_size,
                                                            const std::vector<tensor> &outputs)
{
    const std::vector<int64_t> shape0 = outputs.at(0).shape;
    const float* output0_data = reinterpret_cast<float*>(outputs.at(0).data);

    // Expect [N, 1, H, W]
    const size_t out_batch_size = shape0.at(0);
    const size_t output_channels = shape0.at(1);
    const size_t output_h = shape0.at(2);
    const size_t output_w = shape0.at(3);

    std::vector<dets_t> predictions_list(sel_batch_size);
    for (size_t s = 0; s < sel_batch_size; ++s) {

        const size_t output_area = output_h * output_w;
        const float *batch_offsetptr = output0_data + s * (output_channels * output_area);
        const cv::Mat prob_map(output_h, output_w, CV_32FC1, const_cast<float*>(batch_offsetptr));

        // cv::namedWindow("Heatmap Debugging", cv::WINDOW_NORMAL);
        // cv::imshow("Heatmap Debugging", visualize_prob_heatmap(prob_map));
        // cv::waitKey(0);

        const auto img = batch.at(batch_indx + s);
        predictions_list[s] = extract_text_boxes(prob_map, cv::Size(img.cols, img.rows));
    }

    return predictions_list;
}

inline std::vector<box_t> detector::extract_text_boxes(const cv::Mat &probMap, const cv::Size &origSize) 
{
    cv::Mat binary;
    cv::threshold(probMap, binary, m_config.thresh.value(), 1.0, cv::THRESH_BINARY);

    cv::Mat binary_uint8;
    binary.convertTo(binary_uint8, CV_8UC1, 255.0);

    return contours_to_boxes({binary_uint8, probMap, origSize, m_config.box_thresh.value(), m_config.unclip_ratio.value(), m_config.min_box_size.value(), m_config.max_candidates.value()});
}


} // fmr::paddle::ocr