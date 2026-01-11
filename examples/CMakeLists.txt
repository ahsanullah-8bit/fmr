find_package(argparse CONFIG REQUIRED)

function(fmr_add_example target)
	add_executable(${target} ${ARGN})

	target_link_libraries(${target} PRIVATE
		fmr_libfmr
		spdlog::spdlog_header_only
		argparse::argparse
		onnxruntime::onnxruntime
		${OpenCV_LIBS}
	)

    target_include_directories(${target} PRIVATE ${CMAKE_CURRENT_SOURCE_DIR})
endfunction()

fmr_add_example(yolo11_ort yolo/yolo11_ort.cpp)

# Copy necessary assets
# WARNING: These checks are necessary, the GLOB_RECURSE will search the WHOLE SYSTEM, if the <globbing-expressions> is given as "/*.dll".
if (onnxruntime_ROOT AND NOT onnxruntime_ROOT STREQUAL "")
	file(GLOB_RECURSE	ORT_DLLS			LIST_DIRECTORIES FALSE "${onnxruntime_ROOT}/*.dll") # <dir>/lib/cmake/onnxruntime
endif()

if (OpenVINO_ROOT AND NOT OpenVINO_ROOT STREQUAL "")
	file(GLOB_RECURSE	OPENVINO_DLLS		LIST_DIRECTORIES FALSE "${OpenVINO_ROOT}/runtime/*.dll")
	file(GLOB			OPENVINO_JSON		LIST_DIRECTORIES FALSE "${OpenVINO_ROOT}/runtime/bin/intel64/Release/*.json")
endif()

set(temp_postbuild_runtime_files
	${ORT_DLLS}
	${OPENVINO_DLLS}
	${OPENVINO_JSON}
)

set(FMR_POSTBUILD_RUNTIME_FILES ${FMR_POSTBUILD_RUNTIME_FILES} ${temp_postbuild_runtime_files} CACHE STRING "")

fmr_copy_assets(yolo11_ort "${CMAKE_CURRENT_SOURCE_DIR}/../assets" "${CMAKE_CURRENT_BINARY_DIR}")
fmr_copy_postbuild_files(yolo11_ort "${CMAKE_CURRENT_BINARY_DIR}")
