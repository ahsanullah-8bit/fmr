
#include <opencv2/core.hpp>
#include <opencv2/core/base.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <vector>
#include <algorithm>
#include <cstddef>
#include <iterator>

#include <opencv2/core/types.hpp>

#include <fmr/accelerators/accelerator.hpp>
#include <fmr/config/paddleocrconfig.hpp>
#include <fmr/core/types.hpp>
#include <fmr/core/image.hpp>

namespace fmr::paddle::ocr {

class recognizer {
public:

    explicit recognizer(accelerator *inferSession, const paddleocr_config &config);
    virtual ~recognizer();
    virtual recs_t predict(const std::vector<cv::Mat> &batch);

    bool has_dyn_batch();
    bool has_dyn_shape();

protected:
    virtual std::vector<size_t> argsort(const std::vector<float> &container);

private:
    accelerator *m_infer_session = nullptr;
    paddleocr_config m_config;
    std::vector<cv::Scalar> m_colors;

    std::shared_ptr<spdlog::logger> m_logger;
};

inline recognizer::recognizer(accelerator *inferSession, const paddleocr_config &config)
    : m_infer_session(inferSession)
    , m_config(config)
{
    // Some defaults
    // Mean
    if (!m_config.mean)
        m_config.mean = {0.5f, 0.5f, 0.5f};

    // Std
    if (!m_config.std)
        m_config.std = {0.5f, 0.5f, 0.5f};

    // Scale
    if (!m_config.scale)
        m_config.scale = 1.0f / 255.0f;

    // Batch
    if (!m_config.batch) {
        // User didn't provide batch
        // Fallback to 1 (dynamic) or input shape (fixed)
        m_config.batch = has_dyn_batch() ? 1 : static_cast<int>(inferSession->input_shapes().at(0).at(0));
    } else {
        // Fixed shape? compare with input shape. if mismatch, enforce
        if (!has_dyn_batch()
            && static_cast<int>(inferSession->input_shapes().at(0).at(0)) != m_config.batch.value())
            m_config.batch = static_cast<int>(inferSession->input_shapes().at(0).at(0));
    }

    // Stride
    if (!m_config.stride) {
        m_config.stride = 32;
    }

    // if (!m_config.imgsz)
    //     m_config.imgsz = {48, 320};

    if (!m_config.ctc_mode)
        m_config.ctc_mode = paddleocr_config::GreedySearch;

    if (!m_config.character_dict) {
        // This code assumingly defaults to the character list for en_PP-OCRv4_mobile_rec_infer model
        // If yours is different, set this property during configuration.
        m_config.character_dict = {
            '0','1','2','3','4','5','6','7','8','9',
            ':',';','<','=','>','?','@',
            'A','B','C','D','E','F','G','H','I','J','K','L','M',
            'N','O','P','Q','R','S','T','U','V','W','X','Y','Z',
            '[','\\',']','^','_','`',
            'a','b','c','d','e','f','g','h','i','j','k','l','m',
            'n','o','p','q','r','s','t','u','v','w','x','y','z',
            '{','|','}','~',
            '!','"','#','$','%','&','\'','(',')','*','+',
            ',','-','.','/',' ',' '
        };

        /*
            NOTE: There's an additional space at the end, even though there is a space character already.
            The output of the model named en_PP-OCRv4_mobile_rec_infer (which I tested this with) was [-1, -1, 97]
            stating 97 characters output. The 0 index is skipped as part of the CTC Decoding process and
            we should have 96 visible characters in the dictionary. Now, the inference.yml provided with 
            the model only has 95 characters, even their ppocr/utils/en_dict.txt file has the same amount.
            The code they wrote for an example (deploy/cpp_infer) was adding
            a space character for any index > character_list_.size(). I'm just doing the same and if you
            find anything, let me know.
        */
    }
}

inline recognizer::~recognizer()
{}

inline recs_t recognizer::predict(const std::vector<cv::Mat> &batch)
{
    if (batch.empty())
        return {};

    std::vector<float> batch_ratios;
    for (const auto &img : batch) {
        batch_ratios.emplace_back(static_cast<float>(img.cols) / img.rows);
    }
    
    const std::vector<size_t> sorted_ratio_indices = argsort(batch_ratios);
    
    const int batch_size = m_config.batch.value();
    const auto input_shape = m_infer_session->input_shapes().at(0); // BCHW ['DynamicDimension.0', 3, 48, 'DynamicDimension.1']
    recs_t predictions_list(batch.size(), {});
    
    for (size_t b = 0; b < batch.size();) {
        const size_t sel_end = batch_size < 0                                   // batch is set to -1
                        ? batch.size()                                          // use the whole batch
                        : std::min(batch.size(), b + batch_size);   // else, the specific size
        const size_t sel_size = sel_end - b;

        auto custom_input_shape = input_shape;
        int max_h = custom_input_shape.at(2);
        int max_w = custom_input_shape.at(3);

        if (has_dyn_shape()) {
            int model_stride = m_config.stride.value_or(32);

            if (max_h == -1) {
                // TODO: This should never be true, AT ALL.
                // if (m_config.imgsz) {
                //     max_h = m_config.imgsz->at(0);
                // } else {
                //     for (size_t s = 0; s < sel_size; ++s)
                //         max_h = std::max(max_h, batch[b + s].rows);

                //     if (max_h % model_stride != 0)
                //         max_h = ((max_h / model_stride) + 1) * model_stride;
                // }

                // custom_input_shape[2] = max_h;
            }

            if (max_w == -1) {
                if (m_config.imgsz) {
                    max_w = m_config.imgsz->at(0);
                } else {
                    for (size_t s = 0; s < sel_size; ++s) {
                        const cv::Mat img = batch.at(sorted_ratio_indices.at(b + s));
                        float ratio = static_cast<float>(img.cols) / img.rows;
                        int target_w = static_cast<int>(std::round(max_h * ratio));
                        max_w = std::max(max_w, target_w);
                    }

                }

                if (max_w % model_stride != 0)
                    max_w = ((max_w / model_stride) + 1) * model_stride;

                custom_input_shape[3] = max_w;
            }

        }

        custom_input_shape[0] = sel_size;

        std::vector<cv::Mat> sel_batch;
        for (size_t s = 0; s < sel_size; ++s) {
            const cv::Mat img = batch.at(sorted_ratio_indices.at(b + s));
            float ratio = static_cast<float>(img.cols) / img.rows;
            int target_w = static_cast<int>(std::round(max_h * ratio));

            cv::Mat resized_img;
            cv::resize(img, resized_img, cv::Size(target_w, max_h));
            cv::copyMakeBorder(resized_img, resized_img, 0, 0, 0, max_w - target_w, cv::BORDER_CONSTANT);

            // cv::imshow("Rec Padded", resized_img);
            // cv::waitKey();

            cv::cvtColor(resized_img, resized_img, cv::COLOR_BGR2RGB);
            normalize_imagenet(resized_img, m_config.mean.value(), m_config.std.value(), m_config.scale.value());

            sel_batch.emplace_back(resized_img);
        }

        std::vector<std::vector<float>> inputs(1,  std::vector<float>(vec_product(custom_input_shape), 0.0f));
        permute(sel_batch, inputs[0]);

        m_infer_session->predict_raw(inputs, { custom_input_shape });

        // expect ['DynamicDimension.0', 'Reshape_524_o0__d2', 18385] [-1, T, C] (Batch, SequenceLength, NumClasses)
        const std::vector<int64_t> shape0 = m_infer_session->tensor_shape(0);
        const float* output0_data = m_infer_session->tensor_data(0);

        const size_t out_batch_size = shape0.at(0);
        const size_t out_seq_len = shape0.at(1);
        const size_t out_num_classes = shape0.at(2); // (number of characters in the dictionary + 1 for the blank character).

        // Connectionist Temporal Classification Decoding
        const auto &labels = m_config.character_dict.value();
        for (size_t s = 0; s < sel_size; ++s) {
            std::string text;
            int last_index = -1;
            float score = 0.0f;
            int count = 0;

            for (int n = 0; n < out_seq_len; ++n) {
                const float *first = output0_data + (s * out_seq_len + n) * out_num_classes;
                const float *last =  output0_data + (s * out_seq_len + n + 1) * out_num_classes;
                const float *max_element_it = std::max_element(first, last);
                const int argmax_pos = std::distance(first, max_element_it);

                if (argmax_pos > 0                      // skip blank
                    && argmax_pos - 1 < labels.size() 
                    && argmax_pos != last_index) {      // collapse repeats
                        text += labels.at(argmax_pos - 1);
                        score += *max_element_it;
                        ++count;
                }

                last_index = argmax_pos;
            }

            if (count == 0)
                continue;

            score /= count;

            predictions_list[sorted_ratio_indices.at(b + s)].text = std::move(text);
            predictions_list[sorted_ratio_indices.at(b + s)].score = score;
        }

        b = sel_end;
    }

    return predictions_list;
}

inline bool recognizer::has_dyn_batch()
{
    const auto input_shapes = m_infer_session->input_shapes();
    if (input_shapes.size() == 1                        // is exactly one
        && input_shapes.at(0).size() == 4           // has size 4
        && input_shapes.at(0).at(0) == -1) {   // has 0 index equal -1
        return true;
    }

    return false;
}

inline bool recognizer::has_dyn_shape()
{
    const auto input_shapes = m_infer_session->input_shapes();
    if (input_shapes.size() == 1                            // is exactly one
        && input_shapes.at(0).size() == 4               // has size 4
        && (input_shapes.at(0).at(2) == -1         // has 2 index equal -1 (height)
            || input_shapes.at(0).at(3) == -1)) {  // has 3 index equal -1 (width)
        return true;
    }

    return false;
}

inline std::vector<size_t> recognizer::argsort(const std::vector<float> &container)
{
    size_t n = container.size();
    std::vector<size_t> indexes(n, 0);
    for (size_t i = 0; i < n; ++i)
        indexes[i] = i;

    std::sort(indexes.begin(), indexes.end(),
              [&container](size_t pos1, size_t pos2) noexcept {
                  return container[pos1] < container[pos2];
              });

    return indexes;
}

} // fmr::paddle::ocr