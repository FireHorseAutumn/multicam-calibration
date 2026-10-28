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
    src/batched_solve.cu
)
target_include_directories(mcc_core PUBLIC
    $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
)
target_link_libraries(mcc_core PUBLIC Eigen3::Eigen PRIVATE CUDA::cudart)

# STANDALONE: btl/cholesky.cuh is vendored into include/mcc/vendor/btl/
# (see VENDORING.md). Never include from a sibling repo.

add_executable(test_synthetic_e2e tests/test_synthetic_e2e.cpp)
target_link_libraries(test_synthetic_e2e PRIVATE mcc_core)

add_executable(test_detector tests/test_detector.cpp)
target_link_libraries(test_detector PRIVATE mcc_core)
target_include_directories(test_detector PRIVATE tests)

add_executable(test_colmap_export tests/test_colmap_export.cpp)
target_link_libraries(test_colmap_export PRIVATE mcc_core)
target_include_directories(test_colmap_export PRIVATE tests)

add_executable(test_ba_gpu_solve tests/test_ba_gpu_solve.cpp)
target_link_libraries(test_ba_gpu_solve PRIVATE mcc_core)

add_executable(bench_solve_ab benchmarks/bench_solve_ab.cpp)
target_link_libraries(bench_solve_ab PRIVATE mcc_core)

enable_testing()
add_test(NAME synthetic_e2e COMMAND test_synthetic_e2e)
add_test(NAME detector COMMAND test_detector)
add_test(NAME colmap_export COMMAND test_colmap_export)
add_test(NAME ba_gpu_solve COMMAND test_ba_gpu_solve)
