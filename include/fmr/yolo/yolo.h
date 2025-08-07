#pragma once

#include <vector>

#include <opencv2/core/mat.hpp>

#include <fmr/accelarators/accelerator.h>
#include <fmr/config/yoloconfig.h>
#include <fmr/core/prediction.h>

namespace fmr {

    class yolo {
	public:
        explicit yolo(std::unique_ptr<accelerator> &inferSession, std::shared_ptr<yolo_config> config = nullptr);
        virtual ~yolo();
        virtual std::vector<predictions_t> predict(const std::vector<cv::Mat> &batch);
        virtual void draw(std::vector<std::vector<cv::Mat>> &batch, const std::vector<predictions_t>& predictionsList, float maskAlpha = 0.3f) const;
        virtual void draw(cv::Mat &img, const predictions_t& predictions, float maskAlpha = 0.3f) const;

        virtual bool has_dyn_batch();
        virtual bool has_dyn_shape();
        std::shared_ptr<yolo_config> config() const;
        const std::vector<cv::Scalar> &colors() const;
        void set_colors(const std::vector<cv::Scalar> &newColors);

    protected:
        explicit yolo(std::unique_ptr<accelerator> &inferSession);
        std::unique_ptr<accelerator> &infer_session();

    private:
        std::unique_ptr<accelerator> &m_infer_session;
        std::shared_ptr<yolo_config> m_config;
        std::vector<cv::Scalar> m_colors;
    };
}
