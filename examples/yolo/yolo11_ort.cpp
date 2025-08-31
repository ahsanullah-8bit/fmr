#include <string>
#include <filesystem>

#include <spdlog/spdlog.h>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <fmr/core/framespersecond.hpp>
#include <fmr/accelarators/accelerator.hpp>
#include <fmr/accelarators/onnxruntime.hpp>
#include <fmr/yolo/yolo.hpp>
#include <fmr/config/yoloconfig.hpp>

void draw_metrics(const std::vector<std::string> &metrics, cv::Mat &img);

int main(int argc, char* argv[])
{
    // Paths to the model, labels, input video, and output video
    std::string modelPath;
    std::string labelsPath;
    std::string videoPath;
    std::shared_ptr<spdlog::logger> logger = spdlog::default_logger()->clone("fmr.yolo");

    // Usage: video_inference.exe <model_path> <labels_file_path> <video_input_source> <video_output_source>
    // if (argc < 4) {
    //     logger->error("Usage: {} <model_path> <labels_file_path> <video_input_source>\n", argv[0]);
    //     return 1;
    // }

    modelPath = argc > 1 ? argv[1] : "models/yolo11n-320.onnx";
    labelsPath = argc > 2 ? argv[2] : "";
    videoPath = argc > 3 ? argv[3] : "C:/Users/MadGuy/Videos/J Utah - Driving Downtown - New York City 4K - USA.mp4";

    if (!std::filesystem::exists(videoPath)) {
        logger->error("Selected video file doesn't exist at {}.", videoPath);
        return -1;
    }

    fmr::predictor_config ort_config;
    ort_config.model = fmr::model_config();
    ort_config.model->path = modelPath;
    std::unique_ptr<fmr::accelerator> ort = std::make_unique<fmr::onnxruntime>(ort_config);

    fmr::yolo yolo(ort);

    fmr::frames_per_second fps;
    fps.start();

    cv::VideoCapture cap(videoPath);
    if (!cap.isOpened()) {
        logger->error("Could not open the video file.");
        return -1;
    }

    cv::Mat frame;
    std::string win_name = "YOLO11";
    cv::namedWindow(win_name, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
    while (cap.read(frame)) {
        std::vector<fmr::predictions_t> predictions = yolo.predict({ frame });
        yolo.draw(frame, predictions[0]);

        std::string fps_text = fmt::format("FPS: {}", std::to_string((int)fps.fps()));
        draw_metrics({ fps_text }, frame);

        cv::imshow(win_name, frame);
        int key = cv::pollKey();
        if (key == 'q' || key == 27) // 'q' or ESC
            break;

        // check if window was closed
        if (cv::getWindowProperty(win_name, cv::WND_PROP_VISIBLE) < 1)
            break;

        fps.update();
    }

    cap.release();
    cv::destroyAllWindows();

    return 0;
}

void draw_metrics(const std::vector<std::string> &metrics, cv::Mat &img) {
    int baseline = 0;
    const int font_face = cv::FONT_HERSHEY_SIMPLEX;
    const double font_scale = std::min(img.cols, img.rows) * 0.0008;
    const int font_thickness = std::max(1, (int)(std::min(img.cols, img.rows) * 0.002));

    int y = 20; // start some pixels down from top
    for (const auto &metric : metrics) {
        cv::Size text_size = cv::getTextSize(metric, font_face, font_scale, font_thickness, &baseline);

        // background box (a little padding)
        cv::rectangle(img,
                      cv::Point(0, y - text_size.height - baseline),
                      cv::Point(text_size.width + 5, y + baseline),
                      cv::Scalar(255, 255, 255), cv::FILLED);

        // put the text
        cv::putText(img, metric, cv::Point(2, y),
                    font_face, font_scale,
                    cv::Scalar(0, 0, 0), font_thickness, cv::LINE_AA);

        // move y down for next line
        y += text_size.height + baseline + 5; // add spacing
    }
}
