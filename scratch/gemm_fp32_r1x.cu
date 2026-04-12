#include <cuda_runtime.h>

namespace tile_config {
    constexpr int BM = 128;
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int THREAD_TILE_M = 4;
    constexpr int THREAD_TILE_N = 4;
}

constexpr const char* const VARIANT_ID = "r1x_ldg";
constexpr const char* const VARIANT_DESC = "128x64_ldg";

template <int BM, int BN, int BK, int TM, int TN>
__global__ void gemm_fp32_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    const int thread_col = threadIdx.x;
    const int thread_row = threadIdx.y;
    
    const int tile_row = block_row * BM;
    const int tile_col = block_col * BN;
    
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    
    float acc[TM * TN] = {0.0f};
    
    for (int bk = 0; bk < K; bk += BK) {
        const int k_end = min(bk + BK, K);
        
        for (int m = 0; m < TM; m++) {
            const int row = tile_row + thread_row * TM + m;
            if (row < M) {
                const int row_in_tile = thread_row * TM + m;
                for (int kk = bk; kk < k_end; kk++) {
                    As[row_in_tile][kk - bk] = __ldg(&A[row * K + kk]);
                }
            }
        }
        
        for (int kk = bk; kk < k_end; kk++) {
            const int kk_in_tile = kk - bk;
            for (int n = 0; n < TN; n++) {
                const int col = tile_col + thread_col * TN + n;
                if (col < N) {
                    Bs[kk_in_tile][thread_col * TN + n] = __ldg(&B[kk * N + col]);
                }
            }
        }
        
        __syncthreads();
        
        const int k_tile = k_end - bk;
        for (int kk = 0; kk < k_tile; kk++) {
            for (int m = 0; m < TM; m++) {
                const int row_in_tile = thread_row * TM + m;
                const float a_reg = As[row_in_tile][kk];
                for (int n = 0; n < TN; n++) {
                    const int col_in_tile = thread_col * TN + n;
                    acc[m * TN + n] += a_reg * Bs[kk][col_in_tile];
                }
            }
        }
        
        __syncthreads();
    }
    
    for (int m = 0; m < TM; m++) {
        const int out_row = tile_row + thread_row * TM + m;
        for (int n = 0; n < TN; n++) {
            const int out_col = tile_col + thread_col * TN + n;
            if (out_row < M && out_col < N) {
                C[out_row * N + out_col] = alpha * acc[m * TN + n];
            }
        }
    }
}

void launch_gemm_fp32_r1x(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream)
{
    constexpr int TM = tile_config::THREAD_TILE_M;
    constexpr int TN = tile_config::THREAD_TILE_N;
    
    dim3 grid(
        (N + tile_config::BN - 1) / tile_config::BN,
        (M + tile_config::BM - 1) / tile_config::BM
    );
    dim3 block(tile_config::BN / TN, tile_config::BM / TM);
    
    gemm_fp32_tiled_kernel<tile_config::BM, tile_config::BN, tile_config::BK, TM, TN>
        <<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
}

const char* get_variant_id_fp32_r1x() { return VARIANT_ID; }
const char* get_variant_desc_fp32_r1x() { return VARIANT_DESC; }
