add_executable(yolo11_ort yolo/yolo11_ort.cpp)

target_link_libraries(yolo11_ort PRIVATE
	fmr::libfmr
	onnxruntime::onnxruntime
	${OpenCV_LIBS}
)
