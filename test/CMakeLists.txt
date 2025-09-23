find_package(GTest 1.16.0 CONFIG REQUIRED)

add_executable(${CMAKE_PROJECT_NAME}Test
	tst_image.cpp
	tst_ort_yolo.cpp
)

target_include_directories(${CMAKE_PROJECT_NAME}Test PRIVATE
	${CMAKE_SOURCE_DIR}
)
target_link_libraries(${CMAKE_PROJECT_NAME}Test PRIVATE
	GTest::gtest
	GTest::gtest_main
	fmr::libfmr
	onnxruntime::onnxruntime
	${OpenCV_LIBS}
)

include(GoogleTest)
gtest_add_tests(TARGET ${CMAKE_PROJECT_NAME}Test)

fmr_copy_assets(${CMAKE_PROJECT_NAME}Test "${CMAKE_CURRENT_SOURCE_DIR}/../assets" ${CMAKE_CURRENT_BINARY_DIR})
