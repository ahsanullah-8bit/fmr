# fmr

**fmr** is a utility library focused on providing utility classes and functions, for realtime production level usage of Computer Vision models, in C++. 

> [!NOTE]
> The library is called Framer. But since I found a framework or something with a same name. I had to rename it to fmr, short for Framer.

## Contents
* [Quick Start](#quick-start)
* [Models Supported](#models-supported)
* [Detailed Usage](#detailed-usage)
    * [Classes and Structs](#classes-and-structs)
    * [Functions](#functions)
    * [Examples](#examples)
* [CMake Integration](#cmake-integration)
    * [Requirements](#requirements)
    * [Integrate as a Submodule](#integrate-as-a-submodule)
    * [Build and Integrate](#build-and-integrate)
    * [Integrate through `FetchContent`](#integrate-through-fetchContent)
* [License](#license)
* [Acknowledgments](#acknowledgments)

## Quick Start
fmr file structure looks like this
```bash
fmr
├───accelarators
│       accelerator.hpp
│       onnxruntime.hpp
│
├───config
│       predictorconfig.hpp
│       yoloconfig.hpp
│
├───core
│       framespersecond.hpp
│       image.hpp
│       types.hpp
│
├───wrappers
│       customortallocator.hpp
│
└───yolo
        yolo.hpp
```

so, you may include which ever header you like. 

Accelarators include ONNXRuntime at the moment. To use an accelarator, you need a `fmr::predictor_config` instance, to set up relevent parameters needed. i.e.

```c++
#include <fmr/accelarators/accelerator.hpp>
#include <fmr/accelarators/onnxruntime.hpp>

fmr::predictor_config config;
config.model = fmr::model_config();
config.model->path = "path/to/your/model";

std::unique_ptr<fmr::accelerator> ort = std::make_unique<fmr::onnxruntime>(config);
```

and then, to run inference using a model, you need a `fmr::yolo` instance feeded with a `fmr::yolo_config` instance, for configuration. i.e.

```c++
// load an image
cv::Mat img = cv::imread("path/to/your/model");

// setup parameters
fmr::yolo_config yolo_config;
yolo_config.task = fmr::yolo_config::Detect;
yolo_config.names = ...; // an unordered_map<int,string>

// setup yolo, predict and draw
fmr::yolo yolo(ort);
const auto predictions = yolo.predict({ img });
// and you can optionally draw the predictions
yolo.draw(img, predications[0]);
```

Note that, most of the `fmr::yolo_config`'s members are optional. Because `ultralytics` often embeds the metadata into the model and you don't 
need to provide everything. For a detailed guide, see [Classes](#classes).

## Models Supported

These are some of the models tested with fmr
- YOLO11 (detect, obb, segment, pose, classify) of all sizes.
- YOLOv10n
- YOLOv9(c,t)
- YOLOv8 (detect, obb, segment, pose, classify) of all sizes.
- YOLOv5nu
- YOLOv3-tinyu

> [!NOTE]
> Models modified or customized may work, if they expect similar pre-and-post processing techniques to that of the above.

## Detailed Usage

### Classes and Structs

Config classes
- [model_config](#model_config)
- [predictor_config](#predictor_config)
- [yolo_config](#yolo_config)

Accelarator classes
- [accelarator](#accelarator)
- [onnxruntime](#onnxruntime)

Model classes
- [yolo](#yolo)

Wrapper classes
- [custom_ort_alloc](#custom_ort_alloc)

Util classes
- [prediction]()
- [keypoint]()
- [frames_per_second]()

> [!IMPORTANT]
> fmr always prioritizes your configuration over any metadata embedded in the model. For instance, if you explicitly set a task type, that setting takes precedence over the model’s own metadata. 
> This gives you full control and full responsibility.

#### [model_config](#classes-and-structs)

```c++
// fmr/config/predictorconfig.hpp

struct model_config {   // considered for deprecation
    std::optional<std::string> path; // path to your model.
};
```

#### [predictor_config](#classes-and-structs)

Just has a single property / option
```c++
// fmr/config/predictorconfig.hpp

struct predictor_config {
    std::optional<model_config> model;
};
```

#### [yolo_config](#classes-and-structs)

All of these are optional. But some has default values, in case both the model and you fail to provide the needed configuration.

```c++
// fmr/config/yoloconfig.hpp

struct yolo_config {
    // enum representing the type of yolo support task
    typedef enum { Uknown, Detect, OBB, Pose, Segment, Classify } task_t;

    // name of the model
    std::optional<std::string> name;

    // version of the model
    std::optional<std::string> version;

    // type of task
    // [required] if the model doesn't provide this metadata
    std::optional<task_t> task = Uknown;

    // stride of the model input
    std::optional<int> stride = 32;

    // Batch size supported by the model.
    // - You may use this to enforce a batch size, in case of a dynamic model.
    //
    // A few points to note:
    // * You don't provide a batch size, the implementation defaults to,
    //   - Model's metadata.
    //   x Model doesn't have metadata.
    //     - Fixed Shape? Lookup input shape.
    //     - Dynamic Shape? Default to 1.
    //
    // * You provide the batch size,
    //   - Fixed Shape? Compare and enforce the size from the input shape.
    //   - Dynamic Shape? Use a fixed size or -1 to use the whole batch.
    // (Fixed and Dynamic refer to the input/output shape of the model)
    std::optional<int> batch;

    // number of channels supported for input images
    std::optional<int> channels;

    // imgsz used during training
    // - You may use it to enforce input shape, in case of a dynamic model or fmr will select the dimensions of the biggest image. 
    // - Varying input dimensions across different batches are considered inefficient, as they may cause re-allocation.
    std::optional<std::array<int, 2>> imgsz;

    // names/labels the model supports
    // [required] if the model doesn't provide this metadata
    std::optional<std::unordered_map<int, std::string>> names;

    // confidence for filtering predictions
    // - You may use it to enforce filtering of objects below this confidence
    std::optional<float> confidence = 0.4f;

    // IOU threshold for filtering predictions
    // - You may use it to enforce filtering of objects, below this iou threshold
    std::optional<float> iou_threshold = 0.4f;

    // kpt_shape used during training
    // [required] only during Pose Estimation task and if the model doesn't provide this metadata
    std::optional<std::array<int, 2>> kpt_shape;

    // keypoints skeleton joining the keypoints.
    // [required] only during Pose Estimation task
    std::optional<std::vector<std::pair<int, int>>> kpt_skeleton;

    // returns a task_t type for a lower-cased string based task. i.e. detect -> Detect
    static task_t taskForString(const std::string &task);
};
```

#### [accelarator](#classes-and-structs)

An abstract class, that can be inherited into subclass accelarators. `predict_raw`, `tensor_data`, and `tensor_shape` are a must to override.

```c++
// fmr/accelarators/accelarator.hpp

class accelerator {
public:
    // takes raw data, creating tensors and doing prediction
    virtual void predict_raw(std::vector<std::vector<float>> &data,
                                     std::vector<std::vector<int64_t> > customInputShapes = {}) = 0;

    // returns tensor data pointer using an index  corresponding to a single tensor
    virtual const float *tensor_data(int index) = 0;

    // returns a tensor shape using an index
    virtual const std::vector<int64_t> tensor_shape(int index) = 0;

    // print metadata of the model, including input/output tensor details
    virtual void print_metadata() const;

    // some getter methods
    const std::unordered_map<std::string, std::string>& model_metadata() const;
    const std::vector<std::vector<int64_t>> &input_shapes() const;
    const std::vector<std::vector<int64_t>> &output_shapes() const;
    size_t input_nodes() const;
    size_t output_nodes() const;
    const std::vector<const char *> &input_names() const;
    const std::vector<const char *> &output_names() const;

protected:
    // some setters for subclasses
    void set_input_shapes(const std::vector<std::vector<int64_t> > &newInput_shapes);
    void set_output_shapes(const std::vector<std::vector<int64_t> > &newOutput_shapes);
    void set_input_nodes(size_t newInput_nodes);
    void set_output_nodes(size_t newOutput_nodes);
    void set_model_metadata(const std::unordered_map<std::string, std::string> &newModel_metadata);
    void set_input_names(const std::vector<const char *> &newInput_names);
    void set_output_names(const std::vector<const char *> &newOutput_names);
};
```

> [!NOTE]
> You may introduce your own accelarator subclass and use it with the existing model classes, by overriding the necessary abstract methods. If you do so, you must use the setters to set the required private properties of the `accelarator` class.

#### [onnxruntime](#classes-and-structs)

A class inherited from accelarator, utilizing ONNXRuntime inference engine.

```c++
// fmr/accelarators/onnxruntime.hpp

class onnxruntime : public accelerator {
public:
    // takes a config, and optionally ONNXRuntime related parameters for further customization.
    // if none of the nullptr defaults are not passed, the class creates them with default values (as ONNXRuntime recommends)
    explicit onnxruntime(
        const predictor_config &config,
        std::shared_ptr<Ort::Env> env = nullptr,
        std::shared_ptr<Ort::SessionOptions> sessionOptions = nullptr,
        std::shared_ptr<custom_ort_alloc> allocator = nullptr,
        std::shared_ptr<Ort::MemoryInfo> memoryInfo = nullptr
        );

    // an override doing predictions using ONNXRuntime
    void predict_raw(std::vector<std::vector<float>> &data,
                                std::vector<std::vector<int64_t>> customInputShapes = {}) override;
    
    // sets run options for the ORT session
    // [optional] but it must be set before using predict_raw, to enforce customization
    void set_run_options(std::shared_ptr<Ort::RunOptions> runOptions);

    // returns tensor data pointer from ORT tensors
    const float *tensor_data(int index) override;

    // returns tensor output shape from ORT tensors
    const std::vector<int64_t> tensor_shape(int index) override;

    // prints metadata
    void print_metadata() const override;

    // returns the allocator used by this session
    OrtAllocator* allocator() const;

    // returns the allocator in our custom wrapper (see [custom_ort_alloc](#custom_ort_alloc))
    std::shared_ptr<custom_ort_alloc> custom_allocator() const;

    // returns memory used by this session
    std::shared_ptr<Ort::MemoryInfo> memory_info() const;
};

```

#### [yolo](#classes-and-structs)

```c++
// fmr/yolo/yolo.hpp

class yolo {
public:
    // takes an inference session, an optional config and colors for drawing
    // you must provide a config if its certain your model doesn't provide the relevent metadata (see yolo_config)
    explicit yolo(std::unique_ptr<accelerator> &inferSession, std::shared_ptr<yolo_config> config = nullptr, std::vector<cv::Scalar> colors = {});

    virtual ~yolo();

    // takes a batch of images
    // returns a list of predictions per image
    virtual std::vector<predictions_t> predict(const std::vector<cv::Mat> &batch);

    // draws a list of predictions per image
    virtual void draw(std::vector<cv::Mat> &batch, const std::vector<predictions_t>& predictionsList, float maskAlpha = 0.3f) const;

    // draws predictions on an image
    virtual void draw(cv::Mat &img, const predictions_t& predictions, float maskAlpha = 0.3f) const;

    // returns true if the model support dynamic batch
    // each implementation may vary due to the models input/output
    virtual bool has_dyn_batch();

    // returns true if the model supports taking mixed size images 
    virtual bool has_dyn_shape();

    // returns the config used by the instance
    std::shared_ptr<yolo_config> config() const;

    // returns the colors being used for drawing
    const std::vector<cv::Scalar> &colors() const;

    // sets the colors for drawing
    void set_colors(const std::vector<cv::Scalar> &newColors);

protected:
    // returns the accelarator being used by this instance
    std::unique_ptr<accelerator> &infer_session();

    // post processing methods for different tasks
    // takes batch of images, parent batch current index, sub-or-selected batch's size, and the image's resized dimensions
    // returns the predictions for this selected batch
    virtual std::vector<predictions_t> postprocess_detections(const std::vector<cv::Mat> &batch, int batch_indx, int sel_batch_size, cv::Size res_size);
    virtual std::vector<predictions_t> postprocess_obb_detections(const std::vector<cv::Mat> &batch, int batch_indx, int sel_batch_size, cv::Size res_size);
    virtual std::vector<predictions_t> postprocess_keypoints(const std::vector<cv::Mat> &batch, int batch_indx, int sel_batch_size, cv::Size res_size);
    virtual std::vector<predictions_t> postprocess_segmentations(const std::vector<cv::Mat> &batch, int batch_indx, int sel_batch_size, cv::Size res_size);
    virtual std::vector<predictions_t> postprocess_classifications(const std::vector<cv::Mat> &batch, int batch_indx, int sel_batch_size, cv::Size res_size);
};
```

#### [prediction](#classes-and-structs)

A simple struct to hold a prediction, of all sorts. Only some of the members will be valid depending on the type of model.

```c++
// fmr/core/types.hpp

struct prediction {
    float conf;         // confidence
    std::string label;  // label / name of the prediction
    int label_id = -1;  // label index of the type of label.
    int tracker_id = -1;    // tracker id, for trackers
    cv::Rect box;           // bounding box around the prediction
    cv::RotatedRect obb;    // oriented bounding box around the prediction
    cv::Mat mask;           // single-channel (8UC1) mask in full resolution
    std::vector<keypoint> points; // contains cv::Point2f, and conf
};
```

#### [keypoint](#classes-and-structs)

A Struct used during Pose Estimation tasks.

```c++
// fmr/core/types.hpp

struct keypoint {
    cv::Point2f point;  // an x & y coordinate
    float conf;         // confidence of the keypoint
};
```

#### [frames_per_second](#classes-and-structs)

Utility class for FPS counting

```c++
// fmr/core/framespersecond.hpp

class frames_per_second {
public:
    explicit frames_per_second(int max_frames = 1000, int last_n_seconds = 10);

    void start();
    
    void update();

    double fps();
};
```

### Functions

I don't think these need any explanation as their names are obivous.

```c++
// #include <fmr/core/image.hpp>

template <typename T>
clamp(const T &value, const T &low, const T &high);

// Reads labels off a .txt file.
// Note: The file should only contain straight forward per-line-name.
std::unordered_map<int, std::string> read_labels(const std::string &path);

size_t vec_product(const std::vector<int64_t> &vector);

yvoid letter_box(const cv::Mat& image, cv::Mat& outImage,
                 const cv::Size& newShape,
                 const cv::Scalar& color = cv::Scalar(114, 114, 114),
                 const bool scale = true);

void normalize(cv::Mat &image, 
               float scale = 1.0f / 255.0f);

inline void normalize_imagenet(cv::Mat &image,
                               const std::vector<float> &mean = { 0.485, 0.456, 0.406 },
                               const std::vector<float> &std = { 0.229, 0.224, 0.225 },
                               float scale = 1.0f / 255.0f);

void permute(const std::vector<cv::Mat> &batch,
             std::vector<float> &buffer);

cv::Mat sigmoid(const cv::Mat& src);

cv::Rect scale_coords(const cv::Size &resizedImageShape,
                      const cv::Rect &coords,
                      const cv::Size &originalImageShape,
                      bool clip,
                      float gain,
                      int padX,
                      int padY);

cv::Rect scale_coords(const cv::Size &resizedImageShape,
                      const cv::Rect &box,
                      const cv::Size &originalImageShape,
                      bool clip = true);

cv::RotatedRect scale_coords(const cv::Size& resizedImageShape,
                             const cv::RotatedRect& coords,
                             const cv::Size& originalImageShape,
                             bool clip = true);

cv::Point2f scale_coords(const cv::Size &resizedImageShape,
                         const cv::Point2f &point,
                         const cv::Size &originalImageShape,
                         bool clip = true);

std::vector<int> nms_bboxes(const std::vector<cv::Rect>& boxes,
                            const std::vector<float>& scores,
                            const float scoreThreshold,
                            const float iouThreshold);

float obb_iou(const cv::RotatedRect& r1, const cv::RotatedRect& r2);

std::vector<int> nms_obbs(const std::vector<cv::RotatedRect>& boxes,
                          const std::vector<float>& scores,
                          const float scoreThreshold,
                          const float iouThreshold);

std::vector<cv::Scalar> generate_colors(const std::unordered_map<int, 
                                        std::string> &classNames,
                                        int seed = 42);

std::vector<cv::Scalar> generate_colors(size_t size);

void draw_bboxes(cv::Mat &image,
                 const std::vector<prediction> &predictions,
                 const std::unordered_map<int, std::string> &labels,
                 const std::vector<cv::Scalar> &colors,
                 float maskAlpha = 0.3f);

void draw_obbs(cv::Mat &image,
               const std::vector<prediction> &predictions,
               const std::unordered_map<int, std::string> &labels,
               const std::vector<cv::Scalar> &colors,
               float maskAlpha = 0.3f);

void draw_keypoints(cv::Mat &image,
                    const std::vector<prediction> &predictions,
                    const std::vector<std::pair<int, int>> &poseSkeleton,
                    const std::unordered_map<int, std::string> &labels,
                    const std::vector<cv::Scalar> &colors,
                    bool drawBox = false,
                    float maskAlpha = 0.3f);

void draw_segmentations(cv::Mat &image,
                        const std::vector<prediction> &predictions,
                        const std::unordered_map<int, std::string> &labels,
                        const std::vector<cv::Scalar> &colors,
                        bool drawBox = false,
                        float maskAlpha = 0.3f);

void draw_classifications(cv::Mat &image,
                          const std::vector<prediction> &predictions,
                          const std::unordered_map<int, std::string> &labels,
                          const std::vector<cv::Scalar> &colors,
                          float maskAlpha = 1.0f);

```

### Examples
Look into the `examples` folder for detailed examples. Use `FMR_BUILD_EXAMPLES=ON` to build them along with the library. After build, use them like any other CLI tool. i.e.

```bash
yolo11_ort -h   # to get a detailed usage guide.

yolo11_ort --source "path/to/an/image-or-video" --model "assets/models/yolo11n-obb.onnx"
```

## CMake Integration
fmr can be integrated into a CMake through several ways

### Requirements
`fmr`'s package acquisition is designed in a way that, when [consumed as a submodule](#integrate-as-a-submodule) or [through `FetchContent`](#integrate-through-fetchcontent), it will try to acquire prebuilt libraries like `ONNXRuntime` on its own, unless you set `onnxruntime_DIR` or `onnxruntime_ROOT` to your custom builds. The builds must have the right files to work with `find_package`. The rest of the dependencies are your responsibility as a consumer.

But if you want to [build and integrate](#build-and-integrate). You'll need to provide those dependencies, when you use it. Because you can't install external libraries with the library.

- **OpenCV >= 4.10** : Used for image pre-and-post processing.
- **ONNXRuntime >= 1.22** : Used for inference of yolo models, in `.onnx` format. Optional in a sense that, it will be downloaded as a prebuilt library in some cases.
- **spdlog >= 1.15** : Used for logging throughout the library.
- **yaml-cpp >= 0.8** : Used to read metadata off of yolo models and more.
- **argparse >= 3.2** (optional): Used for argument parsing, in examples. Required when `FMR_BUILD_EXAMPLES` is `ON`.
- **GTest >= 1.16** (optional): Used for tests. Required when `FMR_BUILD_TESTS` is `ON`.

### Integrate as a Submodule

Do

```bash
git submodule add https://github.com/ahsanullah-8bit/fmr ./ext/fmr
```

in your project directory and add this in your `CMakeLists.txt`

```cmake
# set any properties before you do the next step. i.e.
set(FMR_BUILD_EXAMPLES ON) # and more

add_subdirectory(ext/fmr)
```

### Build and Integrate

Clone the project

```bash
git clone https://github.com/ahsanullah-8bit/fmr
cd fmr
```

Configure, build and install

```bash
cmake -S . -B build -DFMR_BUILD_EXAMPLES=ON -DFMR_BUILD_TESTS=ON
cmake --build build
cmake --install build --prefix <path/to/installation/dir>
```

Then in your project, either do:

```cmake
set(fmr_DIR "<path/to/installation/dir>/share/fmr")

# or
# set(fmr_ROOT "<path/to/installation/dir>") # I don't know if this works
```

or set the variables in CLI, when building your projects

```bash
cmake -S <source_dir> -B <build_dir> -Dfmr_DIR="<path/to/installation/dir>/share/fmr"
```

and finally, do the `find_package` on it

```cmake
find_package(fmr CONFIG REQUIRED)
```

### Integrate through FetchContent

```cmake
FetchContent_Declare(fmr
    GIT_REPOSITORY "https://github.com/ahsanullah-8bit/fmr"
    GIT_TAG v0.1
)

# set any properties before you do the next step. i.e.
set(FMR_BUILD_EXAMPLES ON) # and more

FetchContent_MakeAvailable(fmr)
```

## License
This project is licensed under MIT License and was done as a practice project. Feel free to use it in your projects.

## Acknowledgments
This project has bits and ideas influenced by implementations of other people, either due to lack of skills or documentation for things this project do. Some of them include,

[Geekgineer/YOLOs-CPP][yolo-cpp]

[yolo-cpp]: https://github.com/Geekgineer/YOLOs-CPP.git
