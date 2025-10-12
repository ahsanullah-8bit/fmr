#include <string>
#include <filesystem>
#include <unordered_set>

#include <spdlog/spdlog.h>
#include <argparse/argparse.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>

#include <fmr/core/framespersecond.hpp>
#include <fmr/accelarators/accelerator.hpp>
#include <fmr/accelarators/onnxruntime.hpp>
#include <fmr/yolo/yolo.hpp>
#include <fmr/config/yoloconfig.hpp>

bool is_image(const std::string& filename);
bool is_video(const std::string& filename);
std::unordered_map<int, std::string> read_names(const std::string &filepath);
void draw_metrics(const std::vector<std::string> &metrics, cv::Mat &img);

int main(int argc, char* argv[])
{
    std::shared_ptr<spdlog::logger> logger = spdlog::default_logger()->clone("fmr.examples.yolo11_ort");
    argparse::ArgumentParser parser("yolo11_ort");
    parser.add_argument("--model")
        .help("Path to your model")
        .required()
        .default_value(std::string("assets/models/yolo11n.onnx"));
    parser.add_argument("--source")
        .help("Path to the video/image file")
        .required()
        .default_value(std::string("assets/images/bus.jpg"));
    parser.add_argument("--source2")
        .help("Path to the 2nd video/image file. For batched inference")
        .default_value(std::string());
    parser.add_argument("--task")
        .help("Type of task (detect, obb, pose, segment, classify). It's a must, if the model don't have metadata")
        .choices("detect", "obb", "pose", "segment", "classify");
    parser.add_argument("--labels")
        .help("Path to the labels/names file");

    try {
        parser.parse_args(argc, argv);
    }
    catch (const std::exception& err) {
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

    // Model path is already validated by the fmr::onnxruntime
    fmr::predictor_config ort_config;
    ort_config.model_path = parser.get<std::string>("--model");

    std::shared_ptr<fmr::yolo_config> yolo_config = std::make_shared<fmr::yolo_config>();
    if (auto task = parser.present("--task"))
        yolo_config->task = fmr::yolo_config::taskForString(task.value());

    if (parser.present("--labels")) {
        if(std::filesystem::exists(parser.get<std::string>("--labels"))) {
            yolo_config->names = read_names(parser.get<std::string>("--labels"));
        } else {
            logger->warn("Selected labels path doesn't exist. Ignoring!");
        }
    }

    std::unique_ptr<fmr::accelerator> ort = std::make_unique<fmr::onnxruntime>(ort_config);
    ort->print_metadata();

    fmr::yolo yolo(ort, yolo_config);

    std::string win_name = "YOLO11-1st", win2_name = "YOLO11-2nd";
    if (is_image(source1)) {
        // Is an image
        std::vector<cv::Mat> batch = { cv::imread(source1) };

        bool is_valid_img2 = !source2.empty() && is_image(source2);
        if (is_valid_img2)
            batch.emplace_back(cv::imread(source2));

        const auto preds = yolo.predict(batch);
        yolo.draw(batch, preds);

        cv::imshow(win_name, batch.at(0));

        if (is_valid_img2)
            cv::imshow(win2_name, batch.at(1));

        cv::waitKey();
    } else if (is_video(source1)) {
        // Is a video
        fmr::frames_per_second fps;
        // fps.start();

        std::unique_ptr<cv::VideoCapture> cap = std::make_unique<cv::VideoCapture>(source1);
        if (cap && !cap->isOpened()) {
            logger->error("Could not open the video file.");
            std::exit(1);
        }

        std::unique_ptr<cv::VideoCapture> cap2;
        if (!source2.empty() && is_video(source2)) {
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
            std::vector<fmr::predictions_t> predictions = yolo.predict(batch);
            yolo.draw(batch, predictions);

            if (cap) {
                std::string fps_text = fmt::format("FPS: {}", std::to_string((int)fps.fps()));
                draw_metrics({ fps_text }, batch.at(0));
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

bool is_image(const std::string& filename)
{
    static const std::unordered_set<std::string> image_exts = {
        ".bmp", ".dib", ".jpg", ".jpeg", ".jpe", ".jp2", ".png",
        ".tif", ".tiff", ".pbm", ".pgm", ".ppm", ".sr", ".ras", ".webp"
    };

    // Extract lowercase extension from filename
    size_t dot_pos = filename.find_last_of('.');
    std::string ext = filename.substr(dot_pos);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (image_exts.count(ext))
        return true;

    return false;
}

bool is_video(const std::string& filename)
{
    static const std::unordered_set<std::string> video_exts = {
        ".avi", ".mp4", ".mov", ".mkv", ".mpg", ".mpeg", ".wmv", ".flv"
    };

    // Extract lowercase extension from filename
    size_t dot_pos = filename.find_last_of('.');
    std::string ext = filename.substr(dot_pos);
    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);

    if (video_exts.count(ext))
        return true;

    return false;
}

std::unordered_map<int, std::string> read_names(const std::string &filepath)
{
    std::ifstream in_stream;
    in_stream.open(filepath, std::ios_base::in);

    if (!in_stream.is_open())
        return {};

    std::unordered_map<int, std::string> result;
    std::string name;
    for (size_t i = 0; !in_stream.eof(); ++i) {
        std::getline(in_stream, name);
        result[i] = name;
    }

    return result;
}

void draw_metrics(const std::vector<std::string> &metrics, cv::Mat &img)
{
    int baseline = 0;
    const int font_face = cv::FONT_HERSHEY_SIMPLEX;
    const double font_scale = std::min(img.cols, img.rows) * 0.0008;
    const int font_thickness = std::max(1, (int)(std::min(img.cols, img.rows) * 0.002));

    int y = 20; // start some pixels down from top
    for (const auto &metric : metrics) {
        cv::Size text_size = cv::getTextSize(metric, font_face, font_scale, font_thickness, &baseline);

        // Background box (a little padding)
        cv::rectangle(img,
                      cv::Point(0, y - text_size.height - baseline),
                      cv::Point(text_size.width + 5, y + baseline),
                      cv::Scalar(255, 255, 255), cv::FILLED);

        // Put the text
        cv::putText(img, metric, cv::Point(2, y),
                    font_face, font_scale,
                    cv::Scalar(0, 0, 0), font_thickness, cv::LINE_AA);

        // Pove y down for next line
        y += text_size.height + baseline + 5; // add spacing
    }
}
