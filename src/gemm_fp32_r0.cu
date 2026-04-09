#include <cuda_runtime.h>

constexpr const char* const VARIANT_ID = "r0";

template <int TILE_M, int TILE_N, int TILE_K>
__global__ void __launch_bounds__(128) gemm_fp32_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    int cta_m = blockIdx.y * TILE_M;
    int cta_n = blockIdx.x * TILE_N;
    if (cta_m >= M || cta_n >= N) return;

    extern __shared__ float smem[];
    float* smem_a = smem;
    float* smem_b = smem + TILE_M * TILE_K;

    int tid = threadIdx.x;

    constexpr int THREADS_PER_ROW = 4;
    constexpr int ROWS_PER_THREAD = TILE_M / (128 / THREADS_PER_ROW);
    constexpr int COLS_PER_THREAD = TILE_N / THREADS_PER_ROW;

    int thread_row = tid / THREADS_PER_ROW;
    int thread_col = (tid % THREADS_PER_ROW) * COLS_PER_THREAD;

    float acc[ROWS_PER_THREAD][COLS_PER_THREAD] = {0.0f};

    const float* A_ptr = A + cta_m * K;
    const float* B_ptr = B + cta_n;

    for (int k0 = 0; k0 < K; k0 += TILE_K) {
        for (int i = tid; i < TILE_M * TILE_K; i += 128) {
            int row = i / TILE_K, col = i % TILE_K;
            smem_a[i] = (cta_m + row < M && k0 + col < K)
                ? A_ptr[row * K + k0 + col] : 0.0f;
        }
        for (int i = tid; i < TILE_K * TILE_N; i += 128) {
            int row = i / TILE_N, col = i % TILE_N;
            smem_b[i] = (k0 + row < K && cta_n + col < N)
                ? B_ptr[(k0 + row) * N + col] : 0.0f;
        }
        __syncthreads();

        for (int k = 0; k < TILE_K; ++k) {
            for (int r = 0; r < ROWS_PER_THREAD; ++r) {
                int a_row = thread_row * ROWS_PER_THREAD + r;
                float a_val = smem_a[a_row * TILE_K + k];
                for (int c = 0; c < COLS_PER_THREAD; ++c) {
                    acc[r][c] += a_val * smem_b[k * TILE_N + thread_col + c];
                }
            }
        }
        __syncthreads();
    }

    for (int r = 0; r < ROWS_PER_THREAD; ++r) {
        int gm = cta_m + thread_row * ROWS_PER_THREAD + r;
        for (int c = 0; c < COLS_PER_THREAD; ++c) {
            int gn = cta_n + thread_col + c;
            if (gm < M && gn < N) {
                float val = alpha * acc[r][c];
                if (beta != 0.0f) val += beta * C[gm * N + gn];
                C[gm * N + gn] = val;
            }
        }
    }
}

void launch_gemm_fp32(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream)
{
    constexpr int TILE_M = 64, TILE_N = 64, TILE_K = 64;
    dim3 grid((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);
    dim3 block(128);

    size_t smem = (TILE_M * TILE_K + TILE_K * TILE_N) * sizeof(float);
    cudaFuncSetAttribute(gemm_fp32_kernel<TILE_M, TILE_N, TILE_K>,
        cudaFuncAttributeMaxDynamicSharedMemorySize, smem);

    gemm_fp32_kernel<TILE_M, TILE_N, TILE_K><<<grid, block, smem, stream>>>(
        d_A, d_B, d_C, M, N, K, alpha, beta);
}

const char* get_variant_id_fp32() { return VARIANT_ID; }
