include_guard()

######################################
##		ONNXRuntime	v1.22.2
######################################

if (WIN32)
	set(ORT_URL "https://github.com/ahsanullah-8bit/fmr/releases/download/v0.0.0/ort_win64_1.22.2.zip")
	set(ORT_URL_HASH "SHA256=0a9ef10c4c15b055fc206c9783f91a51c58cd13515fb7e2034c139dfa70e7d58")
elseif(LINUX)
	set(ORT_URL "")
	set(ORT_URL_HASH "")
endif()

if (ORT_URL AND ORT_URL_HASH)
	if (onnxruntime_DIR AND NOT onnxruntime_DIR STREQUAL "")
		set(onnxruntime_ROOT "${onnxruntime_DIR}/../../../" CACHE STRING "Path to onnxruntime root directory")
	elseif (onnxruntime_ROOT AND NOT onnxruntime_ROOT STREQUAL "")
		set(onnxruntime_DIR "${onnxruntime_ROOT}/lib/cmake/onnxruntime" CACHE STRING "Path to onnxruntime config files")
	else()
		message(NOTICE "--- Setting up onnxruntime ---")

		FetchContent_Declare(onnxruntime
			URL ${ORT_URL}
			URL_HASH ${ORT_URL_HASH}
		)
	    FetchContent_MakeAvailable(onnxruntime)
		FetchContent_GetProperties(onnxruntime)

		if (onnxruntime_POPULATED)
			set(onnxruntime_DIR "${onnxruntime_SOURCE_DIR}/lib/cmake/onnxruntime" CACHE STRING "Path to onnxruntime config files")
			set(onnxruntime_ROOT ${onnxruntime_SOURCE_DIR} CACHE STRING "Path to onnxruntime root directory")
		endif()

		message(NOTICE "--- Setup onnxruntime completed ---")
	endif()

endif()
