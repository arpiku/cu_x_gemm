#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "gemm_bf16_common.cuh"

extern void launch_gemm_bf16_naive(
    const __nv_bfloat16* d_A,
    const __nv_bfloat16* d_B,
    float* d_C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream);

extern void launch_gemm_bf16_rtx5070(
    const __nv_bfloat16* d_A,
    const __nv_bfloat16* d_B,
    float* d_C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream,
    TargetArch arch);

extern void launch_gemm_bf16_h100(
    const __nv_bfloat16* d_A,
    const __nv_bfloat16* d_B,
    float* d_C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream);

constexpr const char* const BF16_VARIANT_ID   = "r1_bf16";
constexpr const char* const BF16_VARIANT_DESC = "split_bf16_dispatcher";

void launch_gemm_bf16(
    const __nv_bfloat16* d_A,
    const __nv_bfloat16* d_B,
    float*               d_C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream,
    TargetArch arch)
{
    if (bf16_gemm::requires_naive_fallback(M, N)) {
        launch_gemm_bf16_naive(d_A, d_B, d_C, M, N, K, alpha, beta, stream);
        return;
    }

    if (arch == TargetArch::H100) {
        launch_gemm_bf16_h100(d_A, d_B, d_C, M, N, K, alpha, beta, stream);
        return;
    }

    launch_gemm_bf16_rtx5070(d_A, d_B, d_C, M, N, K, alpha, beta, stream, arch);
}

const char* get_variant_id_bf16()
{
    return BF16_VARIANT_ID;
}

const char* get_variant_desc_bf16()
{
    return BF16_VARIANT_DESC;
}