#include <cuda_runtime.h>

namespace tile_config {
    constexpr int BM = 64;           // Output tile rows per block
    constexpr int BN = 64;           // Output tile cols per block
    constexpr int BK = 64;           // K-dimension tile size
    constexpr int THREAD_TILE_M = 4; // Output elements per thread in M
    constexpr int THREAD_TILE_N = 4; // Output elements per thread in N
}

constexpr const char* const VARIANT_ID = "r1_tiled";
constexpr const char* const VARIANT_DESC = "64x64_smem";

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
        int k_end = min(bk + BK, K);
        
        // Load A: each thread loads multiple elements along K dimension
        for (int m = 0; m < TM; m++) {
            int row = tile_row + thread_row * TM + m;
            if (row < M) {
                for (int kk = bk; kk < k_end; kk++) {
                    As[thread_row * TM + m][kk - bk] = A[row * K + kk];
                }
            }
        }
        
        // Load B: each thread loads one element per K
        for (int kk = bk; kk < k_end; kk++) {
            for (int n = 0; n < TN; n++) {
                int col = tile_col + thread_col * TN + n;
                if (col < N) {
                    Bs[kk - bk][thread_col * TN + n] = B[kk * N + col];
                }
            }
        }
        
        __syncthreads();
        
        // Compute
        for (int kk = 0; kk < k_end - bk; kk++) {
            for (int m = 0; m < TM; m++) {
                float a_reg = As[thread_row * TM + m][kk];
                for (int n = 0; n < TN; n++) {
                    acc[m * TN + n] += a_reg * Bs[kk][thread_col * TN + n];
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results
    for (int m = 0; m < TM; m++) {
        int out_row = tile_row + thread_row * TM + m;
        for (int n = 0; n < TN; n++) {
            int out_col = tile_col + thread_col * TN + n;
            if (out_row < M && out_col < N) {
                C[out_row * N + out_col] = alpha * acc[m * TN + n];
            }
        }
    }
}

void launch_gemm_fp32_r1(
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

const char* get_variant_id_fp32_r1() { return VARIANT_ID; }
const char* get_variant_desc_fp32_r1() { return VARIANT_DESC; }
