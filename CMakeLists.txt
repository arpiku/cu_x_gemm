cmake_minimum_required(VERSION 3.20)
project(cu_x_gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 90 120)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(cu_x_gemm
    src/main.cu
    src/cublas_gemm.cu
    src/gemm_bf16.cu
    src/gemm_fp32_r1a.cu
    src/gemm_fp32_r1b.cu
    src/gemm_fp32_r1c.cu
    src/gemm_fp32_r1d.cu
)

target_link_libraries(cu_x_gemm PRIVATE
    CUDA::cublas
    CUDA::cudart
    OpenMP::OpenMP_CXX
)

target_compile_options(cu_x_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
