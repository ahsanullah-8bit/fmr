

add_executable(${CMAKE_PROJECT_NAME}Test
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


# copy the test assets
add_custom_command(TARGET ${CMAKE_PROJECT_NAME}Test POST_BUILD
	COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
	    "${CMAKE_CURRENT_SOURCE_DIR}/assets"
		"${CMAKE_BINARY_DIR}/test/assets"
	DEPENDS	"${CMAKE_CURRENT_SOURCE_DIR}/assets"
)

# file(GLOB TEST_ASSET_FILES LIST_DIRECTORIES FALSE "${CMAKE_CURRENT_SOURCE_DIR}/assets/*")

# message(STATUS "Assets: ${TEST_ASSET_FILES}")

# if (TEST_ASSET_FILES AND NOT TEST_ASSET_FILES STREQUAL "")
# 	add_custom_command(TARGET ${CMAKE_PROJECT_NAME}Test POST_BUILD
# 		COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_BINARY_DIR}/test/assets"

# 		COMMAND ${CMAKE_COMMAND} -E copy_if_different
# 		    ${TEST_ASSET_FILES}
# 			"${CMAKE_BINARY_DIR}/test/assets"

# 		VERBATIM
# 	)
# endif()
