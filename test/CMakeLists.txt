

add_executable(${CMAKE_PROJECT_NAME}Test
	tst_ort_yolo.cpp
)

target_include_directories(${CMAKE_PROJECT_NAME}Test PRIVATE
	${CMAKE_SOURCE_DIR}
)
target_precompile_headers(${CMAKE_PROJECT_NAME}Test PRIVATE ${CMAKE_SOURCE_DIR}/src/fmr_global.h)
target_link_libraries(${CMAKE_PROJECT_NAME}Test PRIVATE
	GTest::gtest
	GTest::gtest_main
	Framer::libfmr
	onnxruntime::onnxruntime
	${OpenCV_LIBS}
)

include(GoogleTest)
gtest_add_tests(TARGET ${CMAKE_PROJECT_NAME}Test)


# copy the test assets
file(GLOB TEST_ASSETS_FILES LIST_DIRECTORIES FALSE "${CMAKE_CURRENT_SOURCE_DIR}/assets/*")
set(TEST_ASSETS_BIN_DST "${CMAKE_BINARY_DIR}/test/assets")
if (TEST_ASSETS_FILES AND NOT TEST_ASSETS_FILES STREQUAL "")
	add_custom_command(TARGET ${CMAKE_PROJECT_NAME}Test POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E make_directory ${TEST_ASSETS_BIN_DST}

		COMMAND ${CMAKE_COMMAND} -E copy_if_different
		    ${TEST_ASSETS_FILES}
			${TEST_ASSETS_BIN_DST}

		VERBATIM
	)
endif()
