cmake_minimum_required(VERSION 3.25)
project(CUDAProcessing LANGUAGES CXX CUDA)

include(CTest)
enable_testing()

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(SOURCES
    main.cu
    cuda/blur.cu
    cuda/brightness.cu
    cuda/saturation.cu
    cuda/sharpness.cu
    cuda/ycrcb.cu
)

add_executable(CUDAProcessing ${SOURCES})

find_package(OpenCV REQUIRED)
if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "OpenCV is required but was not found.")
endif()

find_package(CUDAToolkit REQUIRED)
if(NOT CUDAToolkit_FOUND)
    message(FATAL_ERROR "CUDA Toolkit is required but was not found.")
endif()

target_include_directories(CUDAProcessing PUBLIC
    ${OpenCV_INCLUDE_DIRS}
    ${PROJECT_SOURCE_DIR}/include
)

target_link_libraries(CUDAProcessing PUBLIC ${OpenCV_LIBS})

set_target_properties(CUDAProcessing PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    LINKER_LANGUAGE CUDA
)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

if(CMAKE_BUILD_TYPE STREQUAL "Debug")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -G -lineinfo")
endif()

set(CPACK_PROJECT_NAME ${PROJECT_NAME})
set(CPACK_PROJECT_VERSION ${PROJECT_VERSION})
include(CPack)
