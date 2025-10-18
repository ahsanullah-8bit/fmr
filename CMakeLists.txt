cmake_minimum_required(VERSION 3.16)
cmake_policy(SET CMP0169 OLD)

project(fmr VERSION 0.1 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(FMR_BUILD_EXAMPLES "Build examples" OFF)
option(FMR_BUILD_TESTS "Build tests" OFF)
option(FMR_ORT "ONNXRuntime support" ON)

include (FetchContent)
set(FETCHCONTENT_QUIET OFF)
set(FETCHCONTENT_UPDATES_DISCONNECTED ON)
set(FETCHCONTENT_GIT_PROGRESS ON)

include(${CMAKE_CURRENT_SOURCE_DIR}/cmake/3rdParty/onnxruntime.cmake)
include(cmake/helpers.cmake)

find_package(onnxruntime CONFIG REQUIRED)
find_package(OpenCV 4.10.0 CONFIG REQUIRED)
find_package(spdlog 1.15.3 CONFIG REQUIRED)
find_package(yaml-cpp 0.8.0 CONFIG REQUIRED)

add_library(libfmr INTERFACE)
add_library(fmr::libfmr ALIAS libfmr)

target_include_directories(libfmr INTERFACE
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
)

target_link_libraries(libfmr INTERFACE
   ${OpenCV_LIBS}
   onnxruntime::onnxruntime
   spdlog::spdlog_header_only
   yaml-cpp::yaml-cpp
)

set(IS_MAIN_PROJECT OFF)
if (${CMAKE_SOURCE_DIR} STREQUAL ${CMAKE_CURRENT_SOURCE_DIR})
   set(IS_MAIN_PROJECT ON)
endif()

if (IS_MAIN_PROJECT)
   include(GNUInstallDirs)
   include(CMakePackageConfigHelpers)

   install(TARGETS libfmr
       EXPORT fmrTargets
       INCLUDES DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
   )

   # For find_package
   install(EXPORT fmrTargets
       FILE fmrTargets.cmake
       NAMESPACE fmr::
       DESTINATION "${CMAKE_INSTALL_DATADIR}/fmr"
   )

   install(DIRECTORY include/ DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

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

endif()

if (FMR_BUILD_EXAMPLES)
   add_subdirectory(examples)
endif()

if (FMR_BUILD_TESTS)
   add_subdirectory(test)
endif()
