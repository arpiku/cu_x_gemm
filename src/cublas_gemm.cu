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

// FP32 pedantic and TF32 reference variants live in separate source files.

// FP32 pedantic and TF32 reference variants now live in separate source files.
//
// Kept here: BF16 references plus the legacy SGEMM / CUDA-core FP32 reference
// used by the benchmark and historical comparisons.
