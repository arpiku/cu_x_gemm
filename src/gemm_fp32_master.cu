#include <cuda_runtime.h>

// Kernel variants
extern void launch_gemm_fp32_naive(const float*, const float*, float*,
    int, int, int, float, float, cudaStream_t);
extern void launch_gemm_fp32_r2z(const float*, const float*, float*,
    int, int, int, float, float, cudaStream_t);
extern void launch_gemm_fp32_r2z_debug(const float*, const float*, float*,
    int, int, int, float, float, cudaStream_t, const char**);

constexpr const char* VARIANT_ID   = "master";
constexpr const char* VARIANT_DESC = "auto_select_naive_r2z";

constexpr int NAIVE_MAX_ELEMENTS      = 4096;      // 64×64
constexpr int R2Z_SMALL_MAX_ELEMENTS  = 65536;     // 256×256
constexpr int R2Z_MEDIUM_MAX_ELEMENTS = 1048576;   // 1024×1024

static const char* select_kernel_name(int elements) {
    if (elements <= NAIVE_MAX_ELEMENTS) return "naive";
    if (elements <= R2Z_SMALL_MAX_ELEMENTS) return "r2z_small";
    if (elements <= R2Z_MEDIUM_MAX_ELEMENTS) return "r2z_medium";
    return "r2z_large";
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

    if (elements <= NAIVE_MAX_ELEMENTS) {
        launch_gemm_fp32_naive(d_A, d_B, d_C, M, N, K, alpha, beta, stream);
    } else {
        launch_gemm_fp32_r2z(d_A, d_B, d_C, M, N, K, alpha, beta, stream);
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

    if (elements <= NAIVE_MAX_ELEMENTS) {
        launch_gemm_fp32_naive(d_A, d_B, d_C, M, N, K, alpha, beta, stream);
    } else {
        launch_gemm_fp32_r2z_debug(d_A, d_B, d_C, M, N, K, alpha, beta, stream, &selected);
    }

    if (selected_kernel_out) {
        *selected_kernel_out = selected;
    }
}

const char* get_variant_id_fp32_master()   { return VARIANT_ID; }
const char* get_variant_desc_fp32_master() { return VARIANT_DESC; }
