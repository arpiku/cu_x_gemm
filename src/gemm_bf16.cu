#include <cuda_bf16.h>
#include <cuda_runtime.h>

constexpr const char* const VARIANT_ID = "r0";
constexpr const char* const VARIANT_DESC = "naive";

template <int TILE>
__global__ void gemm_bf16_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        float a = __bfloat162float(A[row * K + k]);
        float b = __bfloat162float(B[k * N + col]);
        sum += a * b;
    }

    if (beta != 0.0f) {
        sum = alpha * sum + beta * C[row * N + col];
    } else {
        C[row * N + col] = alpha * sum;
    }
}

void launch_gemm_bf16(
    const __nv_bfloat16* d_A,
    const __nv_bfloat16* d_B,
    float* d_C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream)
{
    constexpr int TILE = 16;
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    dim3 block(TILE, TILE);

    gemm_bf16_kernel<TILE><<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
}

const char* get_variant_id_bf16() { return VARIANT_ID; }
const char* get_variant_desc_bf16() { return VARIANT_DESC; }
