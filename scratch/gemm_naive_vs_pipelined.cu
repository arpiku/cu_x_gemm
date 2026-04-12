
#include <cuda_runtime.h>
#include <cuda/pipeline>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <chrono>

#define CUDA_CHECK(call) do {ანაი \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error at " << __FILE__ << ":" << __LINE__ \
                  << " code=" << static_cast<int>(err) \
                  << " \"" << cudaGetErrorString(err) << "\"\n"; \
        std::exit(1); \
    } \
} while (0)

constexpr int N = 1024;
constexpr int M = 1024;
constexpr int K = 1024;

constexpr int TILE_M = 32;
constexpr int TILE_N = 32;
constexpr int TILE_K = 8;

constexpr int BLOCK_THREADS = TILE_M * TILE_N; // 1024 threads per block

// ------------------------------------------------------------
// 1) Naive GEMM
// C[row, col] = sum_k A[row, k] * B[k, col]
// ------------------------------------------------------------
__global__ void gemm_naive(const float* __restrict__ A,
                           const float* __restrict__ B,
                           float* __restrict__ C,
                           int M, int N, int K)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) {
            acc += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc;
    }
}

// ------------------------------------------------------------
// 2) Tiled GEMM with async copy / pipeline-style buffering
// Each block computes one 32x32 tile of C.
// We use two shared-memory buffers for A and B.
// ------------------------------------------------------------
template<int TM, int TN, int TK>
__global__ void gemm_pipelined(const float* __restrict__ A,
                               const float* __restrict__ B,
                               float* __restrict__ C,
                               int M, int N, int K)
{
    // 32x32 threads per block
    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * TM + ty;
    const int col = blockIdx.x * TN + tx;

    // Two-stage double buffer in shared memory
    __shared__ float As[2][TM][TK];
    __shared__ float Bs[2][TK][TN];

    float acc = 0.0f;

    // We use a simple ping-pong stage index
    int stage = 0;

    // Number of tiles along K
    const int num_tiles = (K + TK - 1) / TK;

    // Helper lambda to load one tile cooperatively
    auto load_tile = [&](int tile_idx, int buf) {
        int k0 = tile_idx * TK;

        // Load A tile: each thread loads one element if in range
        if (row < M && (k0 + tx) < K && ty < TM && tx < TK) {
            As[buf][ty][tx] = A[row * K + (k0 + tx)];
        } else if (ty < TM && tx < TK) {
            As[buf][ty][tx] = 0.0f;
        }

        // Load B tile: each thread loads one element if in range
        if ((k0 + ty) < K && col < N && ty < TK && tx < TN) {
            Bs[buf][ty][tx] = B[(k0 + ty) * N + col];
        } else if (ty < TK && tx < TN) {
            Bs[buf][ty][tx] = 0.0f;
        }
    };

    // Preload first tile
    load_tile(0, stage);
    __syncthreads();

    for (int t = 0; t < num_tiles; ++t) {
        // Start loading the next tile into the other buffer
        int next = stage ^ 1;
        if (t + 1 < num_tiles) {
            load_tile(t + 1, next);
        }

        // Compute on current tile
        #pragma unroll
        for (int k = 0; k < TK; ++k) {
            if (row < M && col < N) {
                acc += As[stage][ty][k] * Bs[stage][k][tx];
            }
        }

        __syncthreads();
        stage = next;
    }

    if (row < M && col < N) {
        C[row * N + col] = acc;
    }
}

// ------------------------------------------------------------
// CPU reference
// ------------------------------------------------------------
void gemm_cpu(const std::vector<float>& A,
              const std::vector<float>& B,
              std::vector<float>& C,
              int M, int N, int K)
{
    for (int i = 0; i < M; ++i) {
        for (int j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (int k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

float max_abs_diff(const std::vector<float>& a, const std::vector<float>& b) {
    float m = 0.0f;
    for (size_t i = 0; i < a.size(); ++i) {
        m = std::max(m, std::fabs(a[i] - b[i]));
    }
    return m;
}

int main() {
    const size_t sizeA = size_t(M) * K;
    const size_t sizeB = size_t(K) * N;
    const size_t sizeC = size_t(M) * N;

    std::vector<float> hA(sizeA), hB(sizeB), hC_naive(sizeC), hC_pipe(sizeC), hC_ref(sizeC);

    std::mt19937 rng(12345);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);

    for (auto& x : hA) x = dist(rng);
    for (auto& x : hB) x = dist(rng);

    float *dA, *dB, *dC;
    CUDA_CHECK(cudaMalloc(&dA, sizeA * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dB, sizeB * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&dC, sizeC * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(dA, hA.data(), sizeA * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(dB, hB.data(), sizeB * sizeof(float), cudaMemcpyHostToDevice));

    dim3 block1(16, 16);
    dim3 grid1((N + block1.x - 1) / block1.x, (M + block1.y - 1) / block1.y);

    // For the pipelined kernel we use 32x32 threads
    dim3 block2(TILE_N, TILE_M);
    dim3 grid2((N + TILE_N - 1) / TILE_N, (M + TILE_M - 1) / TILE_M);

    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    // -------------------------
    // Naive kernel timing
    // -------------------------
    CUDA_CHECK(cudaMemset(dC, 0, sizeC * sizeof(float)));
    CUDA_CHECK(cudaEventRecord(start));
    gemm_naive<<<grid1, block1>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_naive = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_naive, start, stop));
    CUDA_CHECK(cudaMemcpy(hC_naive.data(), dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Naive kernel time: " << ms_naive << " ms\n";

    // -------------------------
    // Pipelined kernel timing
    // -------------------------
    CUDA_CHECK(cudaMemset(dC, 0, sizeC * sizeof(float)));
    CUDA_CHECK(cudaEventRecord(start));
    gemm_pipelined<TILE_M, TILE_N, TILE_K><<<grid2, block2>>>(dA, dB, dC, M, N, K);
    CUDA_CHECK(cudaEventRecord(stop));
    CUDA_CHECK(cudaEventSynchronize(stop));

    float ms_pipe = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms_pipe, start, stop));
    CUDA_CHECK(cudaMemcpy(hC_pipe.data(), dC, sizeC * sizeof(float), cudaMemcpyDeviceToHost));

    std::cout << "Pipelined kernel time: " << ms_pipe << " ms\n";

    // -------------------------
    // Reference on CPU
    // -------------------------
    std::cout << "Computing CPU reference... (may take a while)\n";
    gemm_cpu(hA, hB, hC_ref, M, N, K);

    float err_naive = max_abs_diff(hC_naive, hC_ref);
    float err_pipe  = max_abs_diff(hC_pipe, hC_ref);

    std::cout << "Max abs diff naive vs ref: " << err_naive << "\n";
    std::cout << "Max abs diff pipelined vs ref: " << err_pipe << "\n";

    CUDA_CHECK(cudaFree(dA));
    CUDA_CHECK(cudaFree(dB));
    CUDA_CHECK(cudaFree(dC));
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return 0;
}
