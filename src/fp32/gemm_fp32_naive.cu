#include <cuda_runtime.h>

constexpr const char* const VARIANT_ID = "r0";
constexpr const char* const VARIANT_DESC = "naive";

template <int TILE>
__global__ void gemm_fp32_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    int row = blockIdx.y * TILE + threadIdx.y;
    int col = blockIdx.x * TILE + threadIdx.x;

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    #pragma unroll 4
    for (int k = 0; k < K; ++k) {
        sum += __ldg(&A[row * K + k]) * __ldg(&B[k * N + col]);
    }

    if (beta != 0.0f) {
        sum = alpha * sum + beta * C[row * N + col];
    } else {
        C[row * N + col] = alpha * sum;
    }
}

void launch_gemm_fp32_naive(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream)
{
    constexpr int TILE = 16;
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    dim3 block(TILE, TILE);

    gemm_fp32_kernel<TILE><<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
}

const char* get_variant_id_fp32_naive() { return VARIANT_ID; }
const char* get_variant_desc_fp32_naive() { return VARIANT_DESC; }
