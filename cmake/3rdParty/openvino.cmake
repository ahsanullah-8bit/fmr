######################################
##		OpenVINO 2025.1 (CPU, GPU)
######################################
# We're not using OpenVINO itself yet, but onnxruntime requires its dlls for OpenVINO EP.

if (WIN32)
	if (OpenVINO_DIR AND NOT OpenVINO_DIR STREQUAL "")
		set(OpenVINO_ROOT "${OpenVINO_DIR}/../../" CACHE STRING "Path to OpenVINO root directory")
	elseif(OpenVINO_ROOT AND NOT OpenVINO_ROOT STREQUAL "")
		set(OpenVINO_DIR "${OpenVINO_ROOT}/runtime/cmake" CACHE STRING "Path to OpenVINO config files")
	else()
		FetchContent_Declare(OpenVINO
			URL ""
		)
	    FetchContent_MakeAvailable(OpenVINO)
		FetchContent_GetProperties(OpenVINO)

		if (openvino_POPULATED)
			set(OpenVINO_DIR "${openvino_SOURCE_DIR}/runtime/cmake"  CACHE STRING "Path to OpenVINO config files")
			set(OpenVINO_ROOT ${openvino_SOURCE_DIR} CACHE STRING "Path to OpenVINO root directory")
		endif()
	endif()
endif()
