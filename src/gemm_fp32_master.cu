#include <cuda_runtime.h>

// Kernel variants
extern void launch_gemm_fp32_naive(const float*, const float*, float*,
    int, int, int, float, float, cudaStream_t);
extern void launch_gemm_fp32_r2z_128(const float*, const float*, float*,
    int, int, int, float, float, cudaStream_t);
extern void launch_gemm_fp32_r2z_512(const float*, const float*, float*,
    int, int, int, float, float, cudaStream_t);
extern void launch_gemm_fp32_r2z_1024(const float*, const float*, float*,
    int, int, int, float, float, cudaStream_t);
extern void launch_gemm_fp32_r2z_2048(const float*, const float*, float*,
    int, int, int, float, float, cudaStream_t);
extern void launch_gemm_fp32_r2z_4096(const float*, const float*, float*,
    int, int, int, float, float, cudaStream_t);

constexpr const char* VARIANT_ID   = "master";
constexpr const char* VARIANT_DESC = "auto_select";

// Runtime-configurable thresholds (element count = M * N)
// Size-tuned r2z variants for optimal performance across matrix sizes
//
//  M*N <= 16384     (≤128×128):   naive      — launch overhead wins
//  M*N <= 65536     (256×256):    r2z_128    — 64×64 tiles, ~74%
//  M*N <= 262144    (512×512):    r2z_512    — 64×64 tiles, ~91%
//  M*N <= 1048576   (1024×1024):  r2z_1024   — 64×64 tiles, ~75%
//  M*N <= 4194304   (2048×2048):  r2z_2048   — 128×128 tiles, ~88%
//  M*N >  4194304   (>2048×2048): r2z_4096   — 128×128 tiles, ~98%

static int g_naive_max    = 16384;     // 128×128
static int g_r2z_128_max  = 65536;     // 256×256
static int g_r2z_512_max  = 262144;    // 512×512
static int g_r2z_1024_max = 1048576;   // 1024×1024
static int g_r2z_2048_max = 4194304;   // 2048×2048

void set_kernel_thresholds(int naive_max, int r2z_128_max, int r2z_512_max,
                           int r2z_1024_max, int r2z_2048_max) {
    g_naive_max = naive_max;
    g_r2z_128_max = r2z_128_max;
    g_r2z_512_max = r2z_512_max;
    g_r2z_1024_max = r2z_1024_max;
    g_r2z_2048_max = r2z_2048_max;
}

void get_kernel_thresholds(int* naive_max, int* r2z_128_max, int* r2z_512_max,
                           int* r2z_1024_max, int* r2z_2048_max) {
    *naive_max = g_naive_max;
    *r2z_128_max = g_r2z_128_max;
    *r2z_512_max = g_r2z_512_max;
    *r2z_1024_max = g_r2z_1024_max;
    *r2z_2048_max = g_r2z_2048_max;
}

static const char* select_kernel_name(int elements) {
    if (elements <= g_naive_max) return "naive";
    if (elements <= g_r2z_128_max) return "r2z_128";
    if (elements <= g_r2z_512_max) return "r2z_512";
    if (elements <= g_r2z_1024_max) return "r2z_1024";
    if (elements <= g_r2z_2048_max) return "r2z_2048";
    return "r2z_4096";
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
    } else if (elements <= g_r2z_128_max) {
        launch_gemm_fp32_r2z_128(d_A, d_B, d_C, M, N, K, alpha, beta, stream);
    } else if (elements <= g_r2z_512_max) {
        launch_gemm_fp32_r2z_512(d_A, d_B, d_C, M, N, K, alpha, beta, stream);
    } else if (elements <= g_r2z_1024_max) {
        launch_gemm_fp32_r2z_1024(d_A, d_B, d_C, M, N, K, alpha, beta, stream);
    } else if (elements <= g_r2z_2048_max) {
        launch_gemm_fp32_r2z_2048(d_A, d_B, d_C, M, N, K, alpha, beta, stream);
    } else {
        launch_gemm_fp32_r2z_4096(d_A, d_B, d_C, M, N, K, alpha, beta, stream);
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
    } else if (elements <= g_r2z_128_max) {
        launch_gemm_fp32_r2z_128(d_A, d_B, d_C, M, N, K, alpha, beta, stream);
    } else if (elements <= g_r2z_512_max) {
        launch_gemm_fp32_r2z_512(d_A, d_B, d_C, M, N, K, alpha, beta, stream);
    } else if (elements <= g_r2z_1024_max) {
        launch_gemm_fp32_r2z_1024(d_A, d_B, d_C, M, N, K, alpha, beta, stream);
    } else if (elements <= g_r2z_2048_max) {
        launch_gemm_fp32_r2z_2048(d_A, d_B, d_C, M, N, K, alpha, beta, stream);
    } else {
        launch_gemm_fp32_r2z_4096(d_A, d_B, d_C, M, N, K, alpha, beta, stream);
    }

    if (selected_kernel_out) {
        *selected_kernel_out = selected;
    }
}

const char* get_variant_id_fp32_master()   { return VARIANT_ID; }
const char* get_variant_desc_fp32_master() { return VARIANT_DESC; }
