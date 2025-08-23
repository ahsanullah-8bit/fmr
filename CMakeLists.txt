cmake_minimum_required(VERSION 3.16)

project(Framer LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(cmake/helpers.cmake)
fmr_init_dependencies()

find_package(onnxruntime CONFIG REQUIRED)
find_package(OpenCV 4.10.0 CONFIG REQUIRED)
find_package(spdlog 1.15.3 CONFIG REQUIRED)
find_package(yaml-cpp 0.8.0 CONFIG REQUIRED)
find_package(GTest 1.16.0 CONFIG REQUIRED)

add_library(libfmr INTERFACE)
add_library(Framer::libfmr ALIAS libfmr)

target_include_directories(libfmr INTERFACE
   "${CMAKE_SOURCE_DIR}/include"
)
target_link_libraries(libfmr INTERFACE
   ${OpenCV_LIBS}
   onnxruntime::onnxruntime
   spdlog::spdlog_header_only
   yaml-cpp::yaml-cpp
)

add_subdirectory(test)
