#include <filesystem>
#include <gtest/gtest.h>

#include <opencv2/core/mat.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgcodecs.hpp>

#include <fmr/core/image.hpp>

class TestImage : public ::testing::Test {
protected:
    static void SetUpTestSuite() {}

    static void TearDownTestSuite() {}

    void SetUp() override {}

    void TearDown() override {}
};

TEST_F(TestImage, DrawOrientedBoundingBoxes)
{
    cv::Mat img = cv::imread("assets/images/boats.jpg");

    // x: 762, y: 338, w: 160, h: 46, angle: 1.4341294

    fmr::prediction prediction;
    prediction.obb = cv::RotatedRect { cv::Point2f(762, 338), cv::Size2f(160, 46), 1.4341294};
    prediction.conf = 0.45;
    prediction.label = "boat";
    prediction.label_id = 1;
    std::unordered_map<int, std::string> labels = { {1, "boat"}, {}, {} };

    fmr::draw_obbs(img, { prediction }, labels, {});

    cv::imshow("TestDrawObb", img);
    cv::waitKey();
}

