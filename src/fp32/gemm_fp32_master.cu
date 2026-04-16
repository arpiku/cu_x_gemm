#include <cuda_runtime.h>

#include "gemm_fp32_common.cuh"

extern void launch_gemm_fp32_naive(const float*, const float*, float*,
    int, int, int, float, float, cudaStream_t);
extern void launch_gemm_fp32_r2z(const float*, const float*, float*,
    int, int, int, float, float, cudaStream_t, TargetArch);


constexpr const char* VARIANT_ID   = "master";
constexpr const char* VARIANT_DESC = "auto_select_naive_r2z";

void launch_gemm_fp32_master(
    const float* d_A,
    const float* d_B,
    float*       d_C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream,
    TargetArch arch)
{
    const int elements = M * N;

    if (elements <= fp32_gemm::FP32_NAIVE_MAX_ELEMENTS) {
        launch_gemm_fp32_naive(d_A, d_B, d_C, M, N, K, alpha, beta, stream);
    } else {
        launch_gemm_fp32_r2z(d_A, d_B, d_C, M, N, K, alpha, beta, stream, arch);
    }
}

const char* get_variant_id_fp32_master()   { return VARIANT_ID; }
const char* get_variant_desc_fp32_master() { return VARIANT_DESC; }
