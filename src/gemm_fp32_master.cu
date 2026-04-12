#include <cuda_runtime.h>

// Kernel variants
extern void launch_gemm_fp32_naive(const float*, const float*, float*,
    int, int, int, float, float, cudaStream_t);
extern void launch_gemm_fp32_r1y(const float*, const float*, float*,
    int, int, int, float, float, cudaStream_t);
extern void launch_gemm_fp32_r2x(const float*, const float*, float*,
    int, int, int, float, float, cudaStream_t);
extern void launch_gemm_fp32_r3x(const float*, const float*, float*,
    int, int, int, float, float, cudaStream_t);
extern void launch_gemm_fp32_r2z2(const float*, const float*, float*,
    int, int, int, float, float, cudaStream_t);

constexpr const char* VARIANT_ID   = "master";
constexpr const char* VARIANT_DESC = "auto_select";

// Runtime-configurable thresholds (element count = M * N)
// Defaults benchmarked on RTX 5070 (see docs/r2Analysis.md)
//
//  M*N <= naive_max  (64×64):    naive  — 230-340% of cuBLAS (launch overhead wins)
//  M*N <= r1y_max   (256×256):   r1y   — 48-55%  of cuBLAS
//  M*N <= r3x_max   (512×512):   r3x   — 66% of cuBLAS (beats r2x at medium sizes)
//  M*N >  r3x_max   (≥1024×1024): r2z2 — 70-98%  of cuBLAS (double buffer + cp.async)
static int g_naive_max = 4096;    // 64×64
static int g_r1y_max   = 65536;   // 256×256
static int g_r3x_max   = 262144;  // 512×512 (r3x beats r2x here, r2z2 takes over from 1024x1024)

void set_kernel_thresholds(int naive_max, int r1y_max, int r3x_max) {
    g_naive_max = naive_max;
    g_r1y_max   = r1y_max;
    g_r3x_max   = r3x_max;
}

void get_kernel_thresholds(int* naive_max, int* r1y_max, int* r3x_max) {
    *naive_max = g_naive_max;
    *r1y_max   = g_r1y_max;
    *r3x_max   = g_r3x_max;
}

static const char* select_kernel_name(int elements) {
    if (elements <= g_naive_max) return "naive";
    if (elements <= g_r1y_max)   return "r1y";
    if (elements <= g_r3x_max)   return "r3x";
    return "r2z2";
}

void launch_gemm_fp32_master(
    const float* d_A,
    const float* d_B,
    float*       d_C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream)
{
    int elements = M * N;

    if (elements <= g_naive_max) {
        launch_gemm_fp32_naive(d_A, d_B, d_C, M, N, K, alpha, beta, stream);
    } else if (elements <= g_r1y_max) {
        launch_gemm_fp32_r1y(d_A, d_B, d_C, M, N, K, alpha, beta, stream);
    } else if (elements <= g_r3x_max) {
        launch_gemm_fp32_r3x(d_A, d_B, d_C, M, N, K, alpha, beta, stream);
    } else {
        launch_gemm_fp32_r2z2(d_A, d_B, d_C, M, N, K, alpha, beta, stream);
    }
}

void launch_gemm_fp32_master_debug(
    const float* d_A,
    const float* d_B,
    float*       d_C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream,
    const char** selected_kernel_out)
{
    int elements = M * N;
    const char* selected = select_kernel_name(elements);

    if (elements <= g_naive_max) {
        launch_gemm_fp32_naive(d_A, d_B, d_C, M, N, K, alpha, beta, stream);
    } else if (elements <= g_r1y_max) {
        launch_gemm_fp32_r1y(d_A, d_B, d_C, M, N, K, alpha, beta, stream);
    } else if (elements <= g_r3x_max) {
        launch_gemm_fp32_r3x(d_A, d_B, d_C, M, N, K, alpha, beta, stream);
    } else {
        launch_gemm_fp32_r2z2(d_A, d_B, d_C, M, N, K, alpha, beta, stream);
    }

    if (selected_kernel_out) {
        *selected_kernel_out = selected;
    }
}

const char* get_variant_id_fp32_master()   { return VARIANT_ID; }
const char* get_variant_desc_fp32_master() { return VARIANT_DESC; }
