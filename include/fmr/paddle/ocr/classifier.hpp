

#include <algorithm>
#include <iterator>
#include <vector>

#include <opencv2/core/mat.hpp>
#include <opencv2/core/types.hpp>

#include <fmr/accelerators/accelerator.hpp>
#include <fmr/config/paddleocrconfig.hpp>
#include <fmr/core/image.hpp>
#include <fmr/core/types.hpp>

namespace fmr::paddle::ocr {

class classifier {
public:
    explicit classifier(accelerator *inferSession, const paddleocr_config &config);
    virtual ~classifier();
    virtual clss_t predict(const std::vector<cv::Mat> &batch);    
    bool has_dyn_batch();
    bool has_dyn_shape();

private:
    accelerator *m_infer_session = nullptr;
    paddleocr_config m_config;

    std::shared_ptr<spdlog::logger> m_logger;
}; // classifier

inline classifier::classifier(accelerator *inferSession, const paddleocr_config &config) 
    : m_infer_session(inferSession)
    , m_config(config)
{
    // Some defaults
    // Mean
    if (!m_config.mean)
        m_config.mean = {0.485f, 0.456f, 0.406f};

    // Std
    if (!m_config.std)
        m_config.std = {0.229f, 0.224f, 0.225f};

    // Scale
    if (!m_config.scale)
        m_config.scale = 1.0f / 255;

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

    // TODO: Image input size

    // Labels
    if (!m_config.labels) {
        m_config.labels = {"0_degree", "180_degree" };
    }
}

inline classifier::~classifier()
{}

inline clss_t classifier::predict(const std::vector<cv::Mat> &batch) {
    if (batch.empty())
        return {};

    const int batch_size = m_config.batch.value();
    clss_t predictions_list(batch.size(), {});

    for (size_t b = 0; b < batch.size();) {
        const size_t sel_end = batch_size < 0                                   // batch is set to -1
                                ? batch.size()                               // use the whole batch
                                : std::min(batch.size(), b + batch_size);    // else, the specific size
        const size_t sel_size = sel_end - b;

        auto inputs = m_infer_session->input_details();
        if(inputs.size() > 1) {
            m_logger->warn("This PaddleOCR classifier implementation expects and uses only 1 input tensor, got {}.", inputs.size());
        }

        auto &input = inputs.at(0);

        // BCHW
        int max_h = input.shape.at(2);
        int max_w = input.shape.at(3);

        if (has_dyn_shape()) {
            int model_stride = m_config.stride.value_or(32);

            if (max_h == -1) {
                if (m_config.imgsz) {
                    max_h = m_config.imgsz->at(0);
                } else {
                    for (size_t s = 0; s < sel_size; ++s)
                        max_h = std::max(max_h, batch.at(b + s).rows);

                    if (max_h % model_stride != 0)
                        max_h = ((max_h / model_stride) + 1) * model_stride;
                }

                input.shape[2] = max_h;
            }

            if (max_w == -1) {
                if (m_config.imgsz) {
                    max_w = m_config.imgsz->at(0);
                } else {
                    for (size_t s = 0; s < sel_size; ++s)
                        max_w = std::max(max_w, batch.at(b + s).cols);

                    if (max_w % model_stride != 0)
                        max_w = ((max_w / model_stride) + 1) * model_stride;
                }

                input.shape[3] = max_w;
            }

        }

        input.shape[0] = sel_size;

        const cv::Size new_shape(max_w, max_h);
        std::vector<cv::Mat> sel_batch;
        for (size_t s = 0; s < sel_size; ++s) {
            const cv::Mat img = batch.at(b + s);
            cv::Mat resized_img;

            cv::resize(img, resized_img, new_shape);
            normalize_imagenet(resized_img, m_config.mean.value(), m_config.std.value(), m_config.scale.value());

            const int pad_right = max_w - resized_img.cols;
            const int pad_bottom = max_h - resized_img.rows;
            if (pad_right > 0 || pad_bottom > 0) {
                cv::copyMakeBorder(resized_img, resized_img,
                                0, pad_bottom, 0, pad_right,
                                cv::BORDER_CONSTANT, cv::Scalar(0, 0, 0));
            }

            sel_batch.emplace_back(resized_img);
        }

        std::vector<float> input_data(vec_product(input.shape), 0.0f);
        permute(sel_batch, input_data);

        input.data = reinterpret_cast<void*>(input_data.data());
        input.size_bytes = input_data.size() * sizeof(float);

        std::vector<tensor> outputs;
        m_infer_session->predict_raw(inputs, outputs);

        // expect ['Reshape_233_o0__d0', 2]
        cv::Size resized_size(input.shape.at(3), input.shape.at(2));
        const std::vector<int64_t> shape0 = outputs.at(0).shape;
        const float* output0_data = reinterpret_cast<float*>(outputs.at(0).data);

        for (size_t s = 0; s < sel_size; ++s) {
            const float *first = &output0_data[s * shape0.at(1)];
            const float *last = &output0_data[(s + 1) * shape0.at(1)];
            const float *max_element = std::max_element(first, last);

            predictions_list[b + s].label_id = static_cast<int>(std::distance(first, max_element));
            predictions_list[b + s].label = m_config.labels->at(predictions_list.at(b + s).label_id);
            predictions_list[b + s].score = *max_element;
        }

        b = sel_end;
    }

    return predictions_list;
}

inline bool classifier::has_dyn_batch() {
    const auto inputs = m_infer_session->input_details();
    return inputs.size() == 1                       // is exactly one
        && inputs.at(0).shape.size() == 4      // has size 4
        && inputs.at(0).shape.at(0) == -1; // has 0 index equal -1
}

inline bool classifier::has_dyn_shape() {
    const auto inputs = m_infer_session->input_details();
    return inputs.size() == 1                            // is exactly one
        && inputs.at(0).shape.size() == 4           // has size 4
        && (inputs.at(0).shape.at(2) == -1      // has 2 index equal -1 (height)
            || inputs.at(0).shape.at(3) == -1); // has 3 index equal -1 (width)
}

} // fmr::paddle::ocr