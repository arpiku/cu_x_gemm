#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr const char* const VARIANT_ID = "bf16_naive";
constexpr const char* const VARIANT_DESC = "bf16_naive_row_major";

template <int TILE>
__global__ void gemm_bf16_naive_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    const int row = blockIdx.y * TILE + threadIdx.y;
    const int col = blockIdx.x * TILE + threadIdx.x;

    if (row >= M || col >= N) {
        return;
    }

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += __bfloat162float(A[row * K + k]) * __bfloat162float(B[k * N + col]);
    }

    const int c_idx = row * N + col;
    if (beta != 0.0f) {
        C[c_idx] = alpha * sum + beta * C[c_idx];
    } else {
        C[c_idx] = alpha * sum;
    }
}

} // namespace

void launch_gemm_bf16_naive(
    const __nv_bfloat16* d_A,
    const __nv_bfloat16* d_B,
    float* d_C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream)
{
    constexpr int TILE = 16;
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);

    gemm_bf16_naive_kernel<TILE><<<grid, block, 0, stream>>>(
        d_A, d_B, d_C, M, N, K, alpha, beta);
}

const char* get_variant_id_bf16_naive() {
    return VARIANT_ID;
}

const char* get_variant_desc_bf16_naive() {
    return VARIANT_DESC;
}