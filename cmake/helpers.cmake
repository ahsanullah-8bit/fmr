include_guard()

# Populates all the required dependencies, that can't be done otherwise.
function(fmr_init_dependencies)
	
	include (FetchContent)
	cmake_policy(SET CMP0169 OLD)
	set(FETCHCONTENT_QUIET OFF)
	set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
	set(FETCHCONTENT_GIT_PROGRESS ON)

	include(${CMAKE_SOURCE_DIR}/cmake/3rdParty/onnxruntime.cmake)
	include(${CMAKE_SOURCE_DIR}/cmake/3rdParty/openvino.cmake)

endfunction() # fmr_init_dependencies
