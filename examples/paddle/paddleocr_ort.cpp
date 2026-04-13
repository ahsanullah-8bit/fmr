#include <memory>
#include <vector>

#include <opencv2/core/types.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgcodecs/imgcodecs.hpp>
#include <spdlog/common.h>
#include <spdlog/logger.h>
#include <spdlog/spdlog.h>
#include <argparse/argparse.hpp>

#include <fmr/config/predictorconfig.hpp>
#include <fmr/config/paddleocrconfig.hpp>
#include <fmr/accelerators/accelerator.hpp>
#include <fmr/accelerators/onnxruntime.hpp>
#include <fmr/paddle/ocr/pipeline.hpp>
#include <fmr/core/types.hpp>
#include <fmr/core/image.hpp>
#include <fmr/core/framespersecond.hpp>
#include <common/helpers.hpp>

void print_ocr(const std::vector<fmr::paddle::ocr::results_t> results, std::shared_ptr<spdlog::logger> logger);

int main(int argc, char* argv[]) {
    std::shared_ptr<spdlog::logger> logger = spdlog::default_logger()->clone("fmr.examples.yolo11_ort");
    logger->set_level(spdlog::level::debug);

    argparse::ArgumentParser parser("paddleocr_ort");
    parser.add_argument("--model_det")
        .help("Path to your PaddleOCR text detection model")
        .required()
        .default_value(std::string("assets/models/PP-OCRv5_mobile_det_infer_onnx/inference.onnx"));

    parser.add_argument("--model_cls")
        .help("Path to your PaddleOCR text classification model")
        .required()
        .default_value(std::string("assets/models/PP-LCNet_x1_0_textline_ori_infer_onnx/inference.onnx"));

    parser.add_argument("--model_rec")
        .help("Path to your PaddleOCR text recognition model")
        .required()
        .default_value(std::string("assets/models/en_PP-OCRv4_mobile_rec_infer_onnx/inference.onnx"));

    parser.add_argument("--source")
        .help("Path to the video/image file")
        .required()
        .default_value(std::string("assets/images/lp2.jpg"));

    parser.add_argument("--source2")
        .help("Path to the 2nd video/image file. For batched inference")
        .default_value(std::string());

    try {
        parser.parse_args(argc, argv);
    } catch (const std::exception& err) {
        logger->critical(err.what());
        std::exit(1);
    }

    // Validate source files
    std::string source1 = parser.get<std::string>("--source");
    if (!std::filesystem::exists(source1)) {
        logger->critical("Selected source file doesn't exist. Please provide a valid source file!");
        std::exit(1);
    }

    std::string source2 = parser.get<std::string>("--source2");
    if (!source2.empty()
        && !std::filesystem::exists(source2)) {
        logger->critical("Selected 2nd source file doesn't exist. Please provide a valid source file!");
    }

    // det
    fmr::predictor_config ort_det_config;
    ort_det_config.model_path = parser.get<std::string>("--model_det");
    
    std::unique_ptr<fmr::accelerator> ort_det = std::make_unique<fmr::onnxruntime>(ort_det_config);
    ort_det->print_metadata();
    
    fmr::paddleocr_config det_config;
    fmr::paddle::ocr::detector det(ort_det.get(), det_config);
    
    // cls
    fmr::predictor_config ort_cls_config;
    ort_cls_config.model_path = parser.get<std::string>("--model_cls");
    
    std::unique_ptr<fmr::accelerator> ort_cls = std::make_unique<fmr::onnxruntime>(ort_cls_config);
    ort_cls->print_metadata();
    
    fmr::paddleocr_config cls_config;
    fmr::paddle::ocr::classifier cls(ort_cls.get(), cls_config);
    
    // rec
    fmr::predictor_config ort_rec_config;
    ort_rec_config.model_path = parser.get<std::string>("--model_rec");
    // TODO: Load characters
    
    std::unique_ptr<fmr::accelerator> ort_rec = std::make_unique<fmr::onnxruntime>(ort_rec_config);
    ort_rec->print_metadata();
    
    fmr::paddleocr_config rec_config;
    fmr::paddle::ocr::recognizer rec(ort_rec.get(), rec_config);

    // final pipeline setup
    fmr::paddle::ocr::pipeline paddleocr(det, cls, rec);

    // OpenCV display setup
    std::string win_name = "PaddleOCR-1st", win2_name = "PaddleOCR-2nd";
    if (fmr::is_image(source1)) {
        // Is an image
        std::vector<cv::Mat> batch = { cv::imread(source1) };

        bool is_valid_img2 = !source2.empty() && fmr::is_image(source2);
        if (is_valid_img2)
            batch.emplace_back(cv::imread(source2));

        // The given image must be in the model's recognition range, meaning it shouldn't
        // be too big either. This is a just a lazy way to acheive that at the moment.
        if (batch.at(0).total() > 409'600) // 640x640
            fmr::letter_box(batch.at(0), batch.at(0), cv::Size(640, 640));
        if (batch.size() > 1 && batch.at(1).total() > 409'600) // 640x640
            fmr::letter_box(batch.at(1), batch.at(1), cv::Size(640, 640));

        const auto preds = paddleocr.predict(batch);
        paddleocr.draw(batch, preds);

        cv::namedWindow(win_name, cv::WINDOW_NORMAL);
        cv::resizeWindow(win_name, cv::Size(800, 600));
        cv::imshow(win_name, batch.at(0));

        print_ocr(preds, logger);

        if (is_valid_img2) {
            cv::namedWindow(win2_name, cv::WINDOW_NORMAL);
            cv::resizeWindow(win2_name, cv::Size(800, 600));
            cv::imshow(win2_name, batch.at(1));
        }

        cv::waitKey();
    } else if (fmr::is_video(source1)) {
        // Is a video
        fmr::frames_per_second fps;
        // fps.start();

        std::unique_ptr<cv::VideoCapture> cap = std::make_unique<cv::VideoCapture>(source1);
        if (cap && !cap->isOpened()) {
            logger->error("Could not open the video file.");
            std::exit(1);
        }

        std::unique_ptr<cv::VideoCapture> cap2;
        if (!source2.empty() && fmr::is_video(source2)) {
            cap2 = std::make_unique<cv::VideoCapture>(source2);
            if (!cap2->isOpened()) {
                logger->error("Could not open the 2nd video file.");
                cap2 = nullptr;
            } else {
                cv::namedWindow(win2_name, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
            }
        }

        cv::Mat frame, frame2;
        cv::namedWindow(win_name, cv::WINDOW_NORMAL | cv::WINDOW_KEEPRATIO);
        while (cap || cap2) {
            std::vector<cv::Mat> batch;

            if (cap) {
                if (!cap->read(frame)) {
                    // loop again
                    cap->set(cv::CAP_PROP_POS_FRAMES, 0);
                    cap->read(frame);
                }

                if (!frame.empty())
                    batch.emplace_back(frame);
            }

            if (cap2) {
                if (!cap2->read(frame2)) {
                    cap2->set(cv::CAP_PROP_POS_FRAMES, 0);
                    cap2->read(frame2);
                }

                if (!frame2.empty())
                    batch.emplace_back(frame2);
            }

            // Inference
            const auto predictions = paddleocr.predict(batch);
            paddleocr.draw(batch, predictions);

            if (cap) {
                std::string fps_text = fmt::format("FPS: {}", std::to_string((int)fps.fps()));
                fmr::draw_metrics({ fps_text }, batch.at(0));
                cv::imshow(win_name, batch.at(0));
            }

            if (cap2) {
                cv::Mat img = cap ? batch.at(1) : batch.at(0);
                cv::imshow(win2_name, img);
            }

            int key = cv::pollKey();
            if (key == 'q' || key == 27) // 'q' or ESC
                break;

            // Check if window was closed
            if (cap && cv::getWindowProperty(win_name, cv::WND_PROP_VISIBLE) < 1) {
                cap = nullptr;
            }

            // Check if window 2 was closed
            if (cap2 && cv::getWindowProperty(win2_name, cv::WND_PROP_VISIBLE) < 1) {
                cap2 = nullptr;
            }

            fps.update();
        }

        cv::destroyAllWindows();
    }

    return 0;
}

void print_ocr(const std::vector<fmr::paddle::ocr::results_t> results, std::shared_ptr<spdlog::logger> logger) 
{
    int img_idx = 0;

    for (const auto &predictions : results) {
        logger->debug("Image {}:", img_idx++);

        for (const auto &p : predictions) {

            // Format detection box points
            std::string pts;
            for (const auto &pt : p.det.points) {
                pts += fmt::format("({:.1f},{:.1f}) ", pt.x, pt.y);
            }

            logger->debug("det: score={:.2f} pts=[{}]", p.det.score, pts);
            logger->debug("cls: score={:.2f} label={}", p.cls.score, p.cls.label);
            logger->debug("rec: score={:.2f} text=\"{}\"\n", p.rec.score, p.rec.text);
        }
    }
}