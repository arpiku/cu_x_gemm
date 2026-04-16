cmake_minimum_required(VERSION 3.20)
project(cu_x_gemm LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_ARCHITECTURES 90 120)

find_package(CUDAToolkit REQUIRED)
find_package(OpenMP REQUIRED)

add_executable(cu_x_gemm
    src/main.cu
    src/helpers.cu
    src/cublas_gemm.cu
    src/bf16/gemm_bf16.cu
    src/bf16/gemm_bf16_naive.cu
    src/bf16/gemm_bf16_wmma_h100.cu
    src/bf16/gemm_bf16_wmma_rtx5070.cu
    src/fp32/gemm_fp32_naive.cu
    src/fp32/gemm_fp32_r2z.cu
    src/fp32/gemm_fp32_master.cu
    src/fp32_tc/cublas_gemm_fp32_pedantic.cu
    src/fp32_tc/cublas_gemm_fp32_tc.cu
    src/fp32_tc/gemm_fp32_r2z_tc1.cu
)

target_include_directories(cu_x_gemm PRIVATE
    src
    src/bf16
    src/fp32
    src/fp32_tc
)

target_link_libraries(cu_x_gemm PRIVATE
    CUDA::cublas
    CUDA::cudart
    OpenMP::OpenMP_CXX
)

target_compile_options(cu_x_gemm PRIVATE
    $<$<COMPILE_LANGUAGE:CUDA>:-O3>
)
