cmake_minimum_required(VERSION 3.10 FATAL_ERROR)

project(VLM_NPU)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

include(CheckIncludeFile)

cmake_minimum_required(VERSION 3.10)
project(VLM_NPU)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -fexceptions -pthread -O3")

# --- Find OpenCV (works for both apt and custom /usr/local)
find_package(PkgConfig REQUIRED)
pkg_check_modules(OPENCV REQUIRED opencv4)
include_directories(${OPENCV_INCLUDE_DIRS})
link_directories(${OPENCV_LIBRARY_DIRS})

# --- Rockchip libraries (allow override via cmake -DRK_LIB_PATH=...)
set(RK_LIB_PATH "/usr/local/lib" CACHE PATH "Path to Rockchip libraries")
link_directories(${RK_LIB_PATH})

# --- Sources
set(SOURCES src/main.cpp src/RK35llm.cpp)
include_directories(include)

# --- Executable
add_executable(VLM_NPU ${SOURCES})
target_link_libraries(VLM_NPU ${OPENCV_LIBRARIES} ${RK_LIB_PATH}/librknnrt.so ${RK_LIB_PATH}/librkllmrt.so -fopenmp -ldl -lpthread -lgomp)

set(EXECUTABLE_OUTPUT_PATH "../")

