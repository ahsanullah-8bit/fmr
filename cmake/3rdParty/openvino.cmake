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
			URL "https://github.com/ahsanullah-8bit/fmr/releases/download/v0.0.0/openvino-2025.1.0-win-x64-auto-cpu-gpu.zip"
			URL_HASH "SHA256=a10b05047b4d140f452b564f14916a5fa4f7d6eda99c452d47ee32fcc05204c9"
		)
	    FetchContent_MakeAvailable(OpenVINO)
		FetchContent_GetProperties(OpenVINO)

		if (openvino_POPULATED)
			set(OpenVINO_DIR "${openvino_SOURCE_DIR}/runtime/cmake"  CACHE STRING "Path to OpenVINO config files")
			set(OpenVINO_ROOT ${openvino_SOURCE_DIR} CACHE STRING "Path to OpenVINO root directory")
		endif()
	endif()
endif()
