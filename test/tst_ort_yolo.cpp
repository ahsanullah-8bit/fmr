#include <filesystem>
#include <gtest/gtest.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/imgcodecs.hpp>

#include <fmr/accelarators/accelerator.hpp>
#include <fmr/accelarators/onnxruntime.hpp>
#include <fmr/yolo/yolo.hpp>

class TestOrtYolo : public ::testing::Test {
protected:
    static void SetUpTestSuite() {}

    static void TearDownTestSuite() {}

    void SetUp() override {}

    void TearDown() override {}
};

// yolo11n detect model inference on ORT CPU EP
TEST_F(TestOrtYolo, OrtCPUYolo11n) {
    fmr::predictor_config config;
    config.batch_size = 1;
    config.model = fmr::model_config();
    config.model->path= "assets/models/yolo11n.onnx";

    std::unique_ptr<fmr::accelerator> ort = std::make_unique<fmr::onnxruntime>(config);
    fmr::yolo yolo(ort);

    cv::Mat img = cv::imread("assets/images/boats.jpg");
    std::vector<cv::Mat> batch = { img };

    std::vector<fmr::predictions_t> results = yolo.predict(batch);
    yolo.draw(batch, results);

    std::filesystem::create_directories("results");
    cv::imwrite("results/yolo11n-cpu.jpg", img);
}

// yolo11x detect model inference on ORT CPU EP
TEST_F(TestOrtYolo, OrtCPUYolo11x) {
    fmr::predictor_config config;
    config.batch_size = 1;
    config.model = fmr::model_config();
    config.model->path= "assets/models/yolo11x.onnx";

    std::unique_ptr<fmr::accelerator> ort = std::make_unique<fmr::onnxruntime>(config);
    fmr::yolo yolo(ort);

    cv::Mat img = cv::imread("assets/images/boats.jpg");
    std::vector<cv::Mat> batch = { img };

    std::vector<fmr::predictions_t> results = yolo.predict(batch);
    yolo.draw(batch, results);

    std::filesystem::create_directories("results");
    cv::imwrite("results/yolo11x-cpu.jpg", img);
}
