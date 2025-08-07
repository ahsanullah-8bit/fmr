#include <gtest/gtest.h>

#include <fmr/accelarators/accelerator.h>
#include <fmr/accelarators/onnxruntime.h>
#include <fmr/yolo/yolo.h>

class TestOrtYolo : public ::testing::Test {
protected:
    static void SetUpTestSuite() {}

    static void TearDownTestSuite() {}

    void SetUp() override {}

    void TearDown() override {}
};

// yolo11n detect model inference on ORT CPU EP
TEST_F(TestOrtYolo, OrtCPUYolo11) {

}
