#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

void cublas_gemm_bf16(
    cublasHandle_t handle,
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    float* C,
    int M, int N, int K,
    float alpha, float beta)
{
    cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16BF, N,
        A, CUDA_R_16BF, K,
        &beta,
        C, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void cublas_gemm_bf16_tc(
    cublasHandle_t handle,
    const __nv_bfloat16* A,
    const __nv_bfloat16* B,
    float* C,
    int M, int N, int K,
    float alpha, float beta)
{
    cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16BF, N,
        A, CUDA_R_16BF, K,
        &beta,
        C, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}

void cublas_gemm_fp32_sgemm(
    cublasHandle_t handle,
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    float alpha, float beta)
{
    cublasSgemm(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, N,
        A, K,
        &beta,
        C, N);
}

void cublas_gemm_fp32_cuda( // legacy diagnostic path
    cublasHandle_t handle,
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    float alpha, float beta)
{
    cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_32F, N,
        A, CUDA_R_32F, K,
        &beta,
        C, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
}

void cublas_gemm_fp32_pedantic( // primary CUDA-core reference (pure FP32)
    cublasHandle_t handle,
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    float alpha, float beta)
{
    cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_32F, N,
        A, CUDA_R_32F, K,
        &beta,
        C, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F_PEDANTIC, CUBLAS_GEMM_DEFAULT);
}

void cublas_gemm_fp32_tc( // TF32 / Tensor Core reference
    cublasHandle_t handle,
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    float alpha, float beta)
{
    cublasGemmEx(handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_32F, N,
        A, CUDA_R_32F, K,
        &beta,
        C, CUDA_R_32F, N,
        CUBLAS_COMPUTE_32F_FAST_TF32, CUBLAS_GEMM_DEFAULT_TENSOR_OP);
}
