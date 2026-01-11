cmake_minimum_required(VERSION 3.24)
cmake_policy(SET CMP0169 OLD)
cmake_policy(SET CMP0135 NEW)

project(fmr VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(FMR_BUILD_EXAMPLES "Build examples" OFF)
option(FMR_BUILD_TESTS "Build tests" OFF)
option(FMR_WITH_ORT "Enable ONNXRuntime support" ON)

include(cmake/helpers.cmake)

find_package(OpenCV 4.10.0 CONFIG REQUIRED)
find_package(spdlog 1.15.3 CONFIG REQUIRED)
find_package(yaml-cpp 0.8.0 CONFIG REQUIRED)

add_library(fmr_core INTERFACE)

target_include_directories(fmr_core INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)
target_sources(fmr_core INTERFACE
   FILE_SET HEADERS
   BASE_DIRS include
   FILES
      include/fmr/accelerators/accelerator.hpp
      include/fmr/config/predictorconfig.hpp
      include/fmr/config/yoloconfig.hpp
      include/fmr/core/framespersecond.hpp
      include/fmr/core/image.hpp
      include/fmr/core/types.hpp
      include/fmr/yolo/yolo.hpp
)

target_link_libraries(fmr_core INTERFACE
   ${OpenCV_LIBS}
   spdlog::spdlog_header_only
   yaml-cpp::yaml-cpp
)

include (FetchContent)
set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
set(FETCHCONTENT_GIT_PROGRESS ON)

# Define the whole library
add_library(fmr_libfmr INTERFACE)
target_link_libraries(fmr_libfmr INTERFACE fmr_core)
set(_fmr_targets fmr_core)

# Introduce the conditional targets
if (FMR_WITH_ORT)
   # Download a pre-build onnxruntime
   include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/3rdParty/onnxruntime.cmake)
   find_package(onnxruntime CONFIG REQUIRED)

   add_library(fmr_ort INTERFACE)
   list(APPEND _fmr_targets fmr_ort)

   target_link_libraries(fmr_ort INTERFACE fmr_core onnxruntime::onnxruntime)
   target_compile_definitions(fmr_ort
       INTERFACE FMR_HAS_ORT=1
   )
   target_sources(fmr_ort INTERFACE
      FILE_SET HEADERS
      BASE_DIRS include
      FILES
         include/fmr/accelerators/onnxruntime.hpp
         include/fmr/wrappers/customortallocator.hpp
   )

   # Add to the common target
   target_link_libraries(fmr_libfmr INTERFACE fmr_ort)
endif()


include(GNUInstallDirs)
include(CMakePackageConfigHelpers)

install(TARGETS ${_fmr_targets}
    EXPORT fmrTargets
    FILE_SET HEADERS
    # INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

# For find_package
install(EXPORT fmrTargets
    FILE fmrTargets.cmake
    DESTINATION "${CMAKE_INSTALL_DATADIR}/fmr"
)

configure_package_config_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/cmake/fmrConfig.cmake.in
    ${CMAKE_CURRENT_BINARY_DIR}/fmrConfig.cmake
    INSTALL_DESTINATION ${CMAKE_INSTALL_DATADIR}/fmr
)

write_basic_package_version_file(
    ${CMAKE_CURRENT_BINARY_DIR}/fmrConfigVersion.cmake
    VERSION ${PROJECT_VERSION}
    COMPATIBILITY SameMajorVersion
)

install(FILES
    ${CMAKE_CURRENT_BINARY_DIR}/fmrConfig.cmake
    ${CMAKE_CURRENT_BINARY_DIR}/fmrConfigVersion.cmake
    DESTINATION ${CMAKE_INSTALL_DATADIR}/fmr
)

if (FMR_BUILD_EXAMPLES)
   add_subdirectory(examples)
endif()

if (FMR_BUILD_TESTS)
   add_subdirectory(test)
endif()
