cmake_minimum_required(VERSION 3.0)
project(Matrix)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)
set(CMAKE_BUILD_TYPE Release)

file(GLOB INCLUDES "external/*/include" "include")
include_directories(${INCLUDES})

file(GLOB SOURCES "src/*.cpp")

add_executable(${PROJECT_NAME} ${SOURCES})

if(WIN32)
    set(OpenCL_DIR "${VCPKG_ROOT}/installed/x64-windows/share/opencl")
endif()

find_package(OpenCL REQUIRED)
include_directories(${OpenCL_INCLUDE_DIRS})
link_directories(${OpenCL_LIBRARY})
target_link_libraries(${PROJECT_NAME} ${OpenCL_LIBRARIES})