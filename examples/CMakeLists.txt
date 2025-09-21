find_package(argparse CONFIG REQUIRED)

add_executable(yolo11_ort yolo/yolo11_ort.cpp)

target_link_libraries(yolo11_ort PRIVATE
	fmr::libfmr
	spdlog::spdlog_header_only
	argparse::argparse
	onnxruntime::onnxruntime
	${OpenCV_LIBS}
)

fmr_copy_assets(yolo11_ort "${CMAKE_CURRENT_SOURCE_DIR}/../assets" "${CMAKE_CURRENT_BINARY_DIR}")
