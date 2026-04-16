cmake_minimum_required(VERSION 3.20)
project(cu_x_gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 90 120)

find_package(CUDAToolkit REQUIRED)

add_executable(cu_x_gemm
    src/main.cu

    src/fp32/cublas_gemm_fp32_pedantic.cu
)

target_include_directories(cu_x_gemm PRIVATE
    src
)

target_link_libraries(cu_x_gemm PRIVATE
    CUDA::cublas
    CUDA::cudart
)

target_compile_options(cu_x_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
