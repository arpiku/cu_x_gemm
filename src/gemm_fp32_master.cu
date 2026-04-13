#include <cuda_runtime.h>

#include "target_arch.h"

// Kernel variants
extern void launch_gemm_fp32_naive(const float*, const float*, float*,
    int, int, int, float, float, cudaStream_t);
extern void launch_gemm_fp32_r2z(const float*, const float*, float*,
    int, int, int, float, float, cudaStream_t, TargetArch);
extern void launch_gemm_fp32_r2z_debug(const float*, const float*, float*,
    int, int, int, float, float, cudaStream_t, TargetArch, const char**);

constexpr const char* VARIANT_ID   = "master";
constexpr const char* VARIANT_DESC = "auto_select_naive_r2z";

constexpr int NAIVE_MAX_ELEMENTS = 4096;      // 64×64

struct R2ZThresholds {
    int small_max;
    int medium_max;
};

static R2ZThresholds select_r2z_thresholds(TargetArch arch) {
    switch (arch) {
        case TargetArch::H100:
            return {262144, 4194304};
        case TargetArch::RTX5070:
            return {65536, 1048576};
    }
    return {65536, 1048576};
}

static const char* select_kernel_name(int elements, TargetArch arch) {
    const auto thresholds = select_r2z_thresholds(arch);
    if (elements <= NAIVE_MAX_ELEMENTS) return "naive";
    if (elements <= thresholds.small_max) return "r2z_small";
    if (elements <= thresholds.medium_max) return "r2z_medium";
    return "r2z_large";
}

void launch_gemm_fp32_master(
    const float* d_A,
    const float* d_B,
    float*       d_C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream,
    TargetArch arch)
{
    int elements = M * N;

    if (elements <= NAIVE_MAX_ELEMENTS) {
        launch_gemm_fp32_naive(d_A, d_B, d_C, M, N, K, alpha, beta, stream);
    } else {
        launch_gemm_fp32_r2z(d_A, d_B, d_C, M, N, K, alpha, beta, stream, arch);
    }
}

void launch_gemm_fp32_master_debug(
    const float* d_A,
    const float* d_B,
    float*       d_C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream,
    TargetArch arch,
    const char** selected_kernel_out)
{
    int elements = M * N;
    const char* selected = select_kernel_name(elements, arch);

    if (elements <= NAIVE_MAX_ELEMENTS) {
        launch_gemm_fp32_naive(d_A, d_B, d_C, M, N, K, alpha, beta, stream);
    } else {
        launch_gemm_fp32_r2z_debug(d_A, d_B, d_C, M, N, K, alpha, beta, stream, arch, &selected);
    }

    if (selected_kernel_out) {
        *selected_kernel_out = selected;
    }
}

const char* get_variant_id_fp32_master()   { return VARIANT_ID; }
const char* get_variant_desc_fp32_master() { return VARIANT_DESC; }
