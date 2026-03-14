#pragma once

#include <fmr/core/types.hpp>
#include <fmr/core/image.hpp>
#include <fmr/accelerators/accelerator.hpp>
#include <fmr/config/paddleocrconfig.hpp>
#include <fmr/paddle/ocr/classifier.hpp>
#include <fmr/paddle/ocr/detector.hpp>
#include <fmr/paddle/ocr/recognizer.hpp>
#include <opencv2/highgui.hpp>

namespace fmr::paddle::ocr {

class pipeline {
public:
    explicit pipeline(accelerator *inferSessionDet, accelerator *inferSessionCls, accelerator *inferSessionRec, const paddleocr_config &paddleocrConfig);
    explicit pipeline(const detector &det, const classifier &cls, const recognizer &rec);
    virtual ~pipeline();
    std::vector<results_t> predict(const std::vector<cv::Mat> &batch);
    virtual void draw(cv::Mat img, const results_t &results) const;
    virtual void draw(std::vector<cv::Mat> &batch, const std::vector<results_t> &resultsList) const;

private:
    detector m_det;
    classifier m_cls;
    recognizer m_rec;
}; // pipeline

inline pipeline::pipeline(accelerator *inferSessionDet, accelerator *inferSessionCls, accelerator *inferSessionRec, const paddleocr_config &paddleocrConfig)
    : m_det(inferSessionDet, paddleocrConfig)
    , m_cls(inferSessionCls, paddleocrConfig)
    , m_rec(inferSessionRec, paddleocrConfig)
{}

inline pipeline::pipeline(const detector &det, const classifier &cls, const recognizer &rec)
    : m_det(det)
    , m_cls(cls)
    , m_rec(rec)
{}

inline pipeline::~pipeline()
{}

inline std::vector<results_t> pipeline::predict(const std::vector<cv::Mat> &batch)
{
    constexpr float line_tolerance = 10.0f;
    std::vector<results_t> predictions_list(batch.size());

    auto boxes_list = m_det.predict(batch);
    for (int b = 0; b < batch.size(); ++b) {
        auto boxes = boxes_list.at(b);
        if (boxes.empty())
            continue;
        
        std::sort(boxes.begin(), boxes.end(), 
            [&line_tolerance](const det& a, const det& b) {
                if (std::abs(a.points.at(0).y - b.points.at(0).y) < line_tolerance)
                    return a.points.at(0).x < b.points.at(0).x; // same line → left to right
                
                return a.points.at(0).y < b.points.at(0).y; // otherwise top to bottom
            }
        );
    
        cv::Mat img = batch.at(b);
        std::vector<cv::Mat> cropped_batch;
        cropped_batch.reserve(boxes.size());
        for (const auto &box : boxes) {
            cv::Mat crop_img;
            perspective_crop(img, crop_img, box.points, 0.2f);
            cropped_batch.emplace_back(crop_img);
        }
        
        auto cls_results = m_cls.predict(cropped_batch);
        
        // rotate the imgs
        for (int i = 0; i < cropped_batch.size(); ++i) {
            if (cls_results.at(i).label % 2 == 1 
            && cls_results.at(i).score > 0.4f) {
                cv::rotate(cropped_batch.at(i), cropped_batch.at(i), cv::ROTATE_180);
            }
        }

        auto rec_results = m_rec.predict(cropped_batch);

        results_t results;
        for (int s = 0; s < boxes.size(); ++s) {
            result res;
            res.det = boxes.at(s);
            res.cls = cls_results.at(s);
            res.rec = rec_results.at(s);
            results.emplace_back(res);
        }

        predictions_list[b] = results;
    }

    return predictions_list;
}

inline void pipeline::draw(cv::Mat img, const results_t &results) const
{
    draw_ocr(img, results, {});
}

inline void pipeline::draw(std::vector<cv::Mat> &batch, const std::vector<results_t> &resultsList) const
{
    for (size_t i = 0; i < batch.size(); ++i) {
        draw(batch.at(i), resultsList.at(i));
    }
}

} // fmr::paddle::ocr