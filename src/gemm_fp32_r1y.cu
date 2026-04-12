#include <cuda_runtime.h>

namespace tile_config {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int THREAD_TILE_M = 2;
    constexpr int THREAD_TILE_N = 2;
    constexpr int THREADS = BM * BN / (THREAD_TILE_M * THREAD_TILE_N);
}

constexpr const char* const VARIANT_ID = "r1y_1d";
constexpr const char* const VARIANT_DESC = "64x64_1Dblock_1024t";

template <int BM, int BN, int BK, int TM, int TN, int THREADS>
__global__ void gemm_fp32_tiled_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    const int tid = threadIdx.x;
    
    const int tile_row = block_row * BM;
    const int tile_col = block_col * BN;
    
    __shared__ float As[BM][BK];
    __shared__ float Bs[BK][BN];
    
    float acc[TM * TN] = {0.0f};
    
    for (int bk = 0; bk < K; bk += BK) {
        const int k_end = min(bk + BK, K);
        
        for (int load_idx = tid; load_idx < BM * BK; load_idx += THREADS) {
            const int load_row = load_idx / BK;
            const int load_col = load_idx % BK;
            const int row = tile_row + load_row;
            const int kk = bk + load_col;
            if (row < M && kk < k_end) {
                As[load_row][load_col] = __ldg(&A[row * K + kk]);
            }
        }
        
        for (int load_idx = tid; load_idx < BK * BN; load_idx += THREADS) {
            const int load_row = load_idx / BN;
            const int load_col = load_idx % BN;
            const int kk = bk + load_row;
            const int col = tile_col + load_col;
            if (kk < k_end && col < N) {
                Bs[load_row][load_col] = __ldg(&B[kk * N + col]);
            }
        }
        
        __syncthreads();
        
        for (int kk = 0; kk < BK; kk++) {
            for (int m = 0; m < TM; m++) {
                const int row_idx = tid / (BN / TN);
                const int row_in_tile = row_idx * TM + m;
                const float a_reg = As[row_in_tile][kk];
                for (int n = 0; n < TN; n++) {
                    const int col_idx = tid % (BN / TN);
                    const int col_in_tile = col_idx * TN + n;
                    acc[m * TN + n] += a_reg * Bs[kk][col_in_tile];
                }
            }
        }
        
        __syncthreads();
    }
    
    for (int m = 0; m < TM; m++) {
        const int row_idx = tid / (BN / TN);
        const int out_row = tile_row + row_idx * TM + m;
        for (int n = 0; n < TN; n++) {
            const int col_idx = tid % (BN / TN);
            const int out_col = tile_col + col_idx * TN + n;
            if (out_row < M && out_col < N) {
                C[out_row * N + out_col] = alpha * acc[m * TN + n];
            }
        }
    }
}

void launch_gemm_fp32_r1y(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream)
{
    constexpr int THREADS = tile_config::THREADS;
    dim3 grid(
        (N + tile_config::BN - 1) / tile_config::BN,
        (M + tile_config::BM - 1) / tile_config::BM
    );
    dim3 block(THREADS);
    
    gemm_fp32_tiled_kernel<tile_config::BM, tile_config::BN, tile_config::BK,
                           tile_config::THREAD_TILE_M, tile_config::THREAD_TILE_N, THREADS>
        <<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
}

const char* get_variant_id_fp32_r1y() { return VARIANT_ID; }
const char* get_variant_desc_fp32_r1y() { return VARIANT_DESC; }
