#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

namespace tile_config {
    constexpr int BM = 64;
    constexpr int BN = 64;
    constexpr int BK = 64;
    constexpr int THREAD_TILE_M = 2;
    constexpr int THREAD_TILE_N = 2;
    constexpr int THREADS = BM * BN / (THREAD_TILE_M * THREAD_TILE_N);
}

constexpr const char* VARIANT_DESC = "transposed_Bs[BN][BK]";

template <int BM, int BN, int BK, int TM, int TN, int THREADS>
__global__ void gemm_transposed_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta,
    int* bank_access_count,
    int* bank_conflict_count)
{
    const int block_row = blockIdx.y;
    const int block_col = blockIdx.x;
    const int tid = threadIdx.x;
    
    const int tile_row = block_row * BM;
    const int tile_col = block_col * BN;
    
    __shared__ float As[BM][BK];
    __shared__ float Bs[BN][BK];
    
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
                
                int bank_id = (load_row * BK + load_col) % 32;
                atomicAdd(&bank_access_count[bank_id], 1);
            }
        }
        
        for (int load_idx = tid; load_idx < BN * BK; load_idx += THREADS) {
            const int load_row = load_idx / BK;
            const int load_col = load_idx % BK;
            const int kk = bk + load_col;
            const int col = tile_col + load_row;
            if (kk < k_end && col < N) {
                Bs[load_row][load_col] = __ldg(&B[kk * N + col]);
                
                int bank_id = (load_row * BK + load_col) % 32;
                atomicAdd(&bank_access_count[bank_id], 1);
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
                    acc[m * TN + n] += a_reg * Bs[col_in_tile][kk];
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

void init_matrix(float* ptr, int n, unsigned seed) {
    srand(seed);
    for (int i = 0; i < n; ++i)
        ptr[i] = (rand() / float(RAND_MAX)) * 2.0f - 1.0f;
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }
    
    int dim = atoi(argv[1]);
    int N = dim;
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("# GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("# Matrix size: %d x %d\n", N, N);
    printf("# Variant: %s\n\n", VARIANT_DESC);
    
    size_t bytes = N * N * sizeof(float);
    
    float *d_A, *d_B, *d_C;
    int *d_bank_access, *d_bank_conflict;
    
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMalloc(&d_bank_access, 32 * sizeof(int));
    cudaMalloc(&d_bank_conflict, 32 * sizeof(int));
    
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    init_matrix(h_A, N * N, 42);
    init_matrix(h_B, N * N, 43);
    
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, bytes);
    cudaMemset(d_bank_access, 0, 32 * sizeof(int));
    cudaMemset(d_bank_conflict, 0, 32 * sizeof(int));
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    constexpr int THREADS = tile_config::THREADS;
    dim3 grid(
        (N + tile_config::BN - 1) / tile_config::BN,
        (N + tile_config::BM - 1) / tile_config::BM
    );
    dim3 block(THREADS);
    
    const int WARMUP = 5;
    const int ITERATIONS = 20;
    
    for (int i = 0; i < WARMUP; ++i) {
        gemm_transposed_kernel<tile_config::BM, tile_config::BN, tile_config::BK,
                              tile_config::THREAD_TILE_M, tile_config::THREAD_TILE_N, THREADS>
            <<<grid, block, 0, stream>>>(d_A, d_B, d_C, N, N, N, 1.0f, 0.0f,
                                         d_bank_access, d_bank_conflict);
    }
    cudaStreamSynchronize(stream);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, stream);
    for (int i = 0; i < ITERATIONS; ++i) {
        gemm_transposed_kernel<tile_config::BM, tile_config::BN, tile_config::BK,
                              tile_config::THREAD_TILE_M, tile_config::THREAD_TILE_N, THREADS>
            <<<grid, block, 0, stream>>>(d_A, d_B, d_C, N, N, N, 1.0f, 0.0f,
                                         d_bank_access, d_bank_conflict);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= ITERATIONS;
    
    printf("=== Timing Results ===\n");
    printf("Average kernel time: %.4f ms\n\n", ms);
    
    int h_bank_access[32];
    cudaMemcpy(h_bank_access, d_bank_access, 32 * sizeof(int), cudaMemcpyDeviceToHost);
    
    long long total_accesses = 0;
    int max_access = 0, min_access = INT_MAX;
    for (int i = 0; i < 32; ++i) {
        total_accesses += h_bank_access[i];
        if (h_bank_access[i] > max_access) max_access = h_bank_access[i];
        if (h_bank_access[i] > 0 && h_bank_access[i] < min_access) min_access = h_bank_access[i];
    }
    
    printf("=== Bank Access Distribution (Load Phase) ===\n");
    printf("BankID  Accesses  Bar\n");
    printf("%s\n", std::string(30, '-').c_str());
    for (int i = 0; i < 32; ++i) {
        int bars = (max_access > 0) ? (h_bank_access[i] * 50) / max_access : 0;
        printf("%3d    %8d  %s\n", i, h_bank_access[i], std::string(bars, '*').c_str());
    }
    
    printf("\n=== Summary ===\n");
    printf("Total accesses: %lld\n", total_accesses);
    printf("Max bank accesses: %d\n", max_access);
    printf("Min bank accesses: %d\n", min_access);
    printf("Max/Min ratio: %.2f\n", (float)max_access / (min_access + 1));
    
    float imbalance = (total_accesses > 0) ? 
        (float)(max_access - min_access) / (total_accesses / 32) * 100.0f : 0;
    printf("Imbalance: %.1f%%\n", imbalance);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_bank_access);
    cudaFree(d_bank_conflict);
    
    free(h_A);
    free(h_B);
    
    return 0;
}