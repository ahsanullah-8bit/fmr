include_guard()

# Populates all the required dependencies, that can't be done otherwise.
# WARNING: This can't work with integration as add_subdirectory in all cases, as
# this function will determine CMAKE_SOURCE_DIR and others from the POV of the caller.
function(fmr_init_dependencies)
	include (FetchContent)
	cmake_policy(SET CMP0169 OLD)
	set(FETCHCONTENT_QUIET OFF)
	set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
	set(FETCHCONTENT_GIT_PROGRESS ON)

	include(${CMAKE_SOURCE_DIR}/cmake/3rdParty/onnxruntime.cmake)
	include(${CMAKE_SOURCE_DIR}/cmake/3rdParty/openvino.cmake)

endfunction() # fmr_init_dependencies

function(fmr_copy_prebuild_files target dst)
	# Copy the prebuild files over to the binary dir
	if (FMR_PREBUILD_FILES AND NOT FMR_PREBUILD_FILES STREQUAL "")
		message(STATUS "--- Listing prebuild files (is being copied): ---")
		foreach(file_path IN LISTS APSS_PREBUILD_FILES)
			message(STATUS "  - ${file_path}")
		endforeach()
		message(STATUS "--- End of prebuild files list ---")

		add_custom_command(TARGET ${target} PRE_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy_if_different
			${FMR_PREBUILD_FILES}
			${dst}

			VERBATIM
		)
    endif()
endfunction() # fmr_copy_prebuild_files

function(fmr_copy_postbuild_files target dst)
	# Copy the postbuild files over to the binary dir
	if (FMR_POSTBUILD_RUNTIME_FILES AND NOT FMR_POSTBUILD_RUNTIME_FILES STREQUAL "")
		message(STATUS "--- Listing runtime files (will be copied): ---")
		foreach(file_path IN LISTS FMR_POSTBUILD_RUNTIME_FILES)
			message(STATUS "  - ${file_path}")
		endforeach()
		message(STATUS "--- End of runtime files list ---")

		add_custom_command(TARGET ${target} POST_BUILD
			COMMAND ${CMAKE_COMMAND} -E copy_if_different
			    ${FMR_POSTBUILD_RUNTIME_FILES}
				${dst}

			VERBATIM
		)
    else()
		message(WARNING "FMR_POSTBUILD_RUNTIME_FILES is empty!")
    endif()
endfunction() # fmr_copy_postbuild_files

function(fmr_copy_assets target src dst)
	message(STATUS "Copying ${src} to ${dst}/assets")
	# Model folders
	add_custom_command(TARGET ${target} POST_BUILD
		COMMAND ${CMAKE_COMMAND} -E copy_directory_if_different
		    "${src}"
			"${dst}/assets"
		DEPENDS "${src}"
		COMMENT "Copying ${src} to ${dst}/assets"
	)
endfunction() # fmr_copy_assets
