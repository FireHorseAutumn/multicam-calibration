cmake_minimum_required(VERSION 3.20)
set(CMAKE_CUDA_FLAGS_INIT "-allow-unsupported-compiler")
# nvcc 13.2 cannot parse g++-16 host headers; pin CUDA host compiler to
# g++-15 when it is available (still overridable via -DCMAKE_CUDA_HOST_COMPILER).
if(NOT DEFINED CMAKE_CUDA_HOST_COMPILER)
    find_program(MCC_CUDA_HOST_GXX g++-15)
    if(MCC_CUDA_HOST_GXX)
        set(CMAKE_CUDA_HOST_COMPILER "${MCC_CUDA_HOST_GXX}")
    endif()
endif()
project(multicam-calibration LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES native)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

find_package(CUDAToolkit REQUIRED)
find_package(Eigen3 REQUIRED NO_MODULE)
# Eigen is the only dependency (NOTES.md decision 1: no OpenCV, no Ceres).

add_library(mcc_core STATIC
    src/board.cpp
    src/detector.cpp
    src/corner_cache.cpp
    src/intrinsics.cpp
    src/pose_init.cpp
    src/bundle_adjustment.cpp
    src/colmap_export.cpp
)
target_include_directories(mcc_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(mcc_core PUBLIC Eigen3::Eigen)

# STANDALONE: vendor a copy of btl/cholesky.cuh into include/mcc/vendor/
# (see README "Vendored headers"). Never include from a sibling repo.

# TODO: add targets as sources land
add_executable(test_synthetic_e2e tests/test_synthetic_e2e.cpp)
target_link_libraries(test_synthetic_e2e PRIVATE mcc_core)

enable_testing()
add_test(NAME synthetic_e2e COMMAND test_synthetic_e2e)
