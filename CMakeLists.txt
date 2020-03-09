cmake_minimum_required(VERSION 3.11)
project(Matrix)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

set(EXECUTABLE_OUTPUT_PATH ./${CMAKE_BUILD_TYPE})
set(SOURCE Main.cpp)

add_executable(${PROJECT_NAME} ${SOURCE})

set(OpenCL_DIR "${VCPKG_ROOT}/installed/x64-windows/share/opencl")

find_package(OpenCL REQUIRED)
if (OpenCL_FOUND)
    include_directories(${OpenCL_INCLUDE_DIRS})
    link_directories(${OpenCL_LIBRARY})
    target_link_libraries(${PROJECT_NAME} ${OpenCL_LIBRARIES})
endif()