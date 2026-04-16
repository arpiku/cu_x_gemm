#include <cublas_v2.h>
#include <cuda_runtime.h>

void cublas_gemm_fp32_pedantic(
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

const char* get_variant_id_fp32_pedantic()
{
    return "fp32_pedantic";
}

const char* get_variant_desc_fp32_pedantic()
{
    return "cublas_compute_32f_pedantic";
}