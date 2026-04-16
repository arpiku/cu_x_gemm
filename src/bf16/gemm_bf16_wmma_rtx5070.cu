// RTX5070-focused BF16 WMMA variants.
//
// This file splits the BF16 optimized path into smaller, size-tuned kernel
// variants while keeping the naïve fallback available for tiny matrices.
//
// Public entry points:
// - launch_gemm_bf16_rtx5070(...)
// - get_variant_id_bf16_rtx5070()
// - get_variant_desc_bf16_rtx5070()

#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "gemm_bf16_common.cuh"

namespace {

using namespace nvcuda;
using namespace bf16_gemm;

// ── Optimized WMMA kernel ─────────────────────────────────────────────────────
// BF16 input -> FP32 accumulation, WMMA m16n16k16, cp.async double-buffered
// shared memory, padded SMEM to reduce bank conflicts.

template <typename Config>
__global__ __launch_bounds__(Config::NUM_THREADS)
void gemm_bf16_wmma_kernel(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    float*               __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    constexpr int BM = Config::BM;
    constexpr int BN = Config::BN;
    constexpr int BK = Config::BK;
    constexpr int WM = Config::WM;
    constexpr int WN = Config::WN;
    constexpr int NUM_THREADS = Config::NUM_THREADS;

    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;

    constexpr int FRAGS_M = WM / WMMA_M;
    constexpr int FRAGS_N = WN / WMMA_N;
    constexpr int FRAGS_K = BK / WMMA_K;

    const uint warpIdx = threadIdx.x / 32;
    const uint warpRow = warpIdx / (BN / WN);
    const uint warpCol = warpIdx % (BN / WN);

    // Padding breaks repetitive bank aliasing for the WMMA loads.
    constexpr int PAD_A = 8;
    constexpr int PAD_B = 8;

    __shared__ __nv_bfloat16 As[2][BM * (BK + PAD_A)];
    __shared__ __nv_bfloat16 Bs[2][BK * (BN + PAD_B)];

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += (blockIdx.y * BM + warpRow * WM) * N + blockIdx.x * BN + warpCol * WN;

    // cp.async copies 16 bytes = 8 BF16 elements.
    constexpr int VEC = 8;

    const uint innerRowA = threadIdx.x / (BK / VEC);
    const uint innerColA = threadIdx.x % (BK / VEC);
    constexpr uint rowStrideA = NUM_THREADS / (BK / VEC);

    const uint innerRowB = threadIdx.x / (BN / VEC);
    const uint innerColB = threadIdx.x % (BN / VEC);
    constexpr uint rowStrideB = NUM_THREADS / (BN / VEC);

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        frag_c[FRAGS_M][FRAGS_N];
    for (int fm = 0; fm < FRAGS_M; ++fm) {
        for (int fn = 0; fn < FRAGS_N; ++fn) {
            wmma::fill_fragment(frag_c[fm][fn], 0.0f);
        }
    }

    auto async_load_tile = [&](int buf, const __nv_bfloat16* A_ptr, const __nv_bfloat16* B_ptr) {
        for (uint off = 0; off + rowStrideA <= (uint)BM; off += rowStrideA) {
            __pipeline_memcpy_async(
                &As[buf][(innerRowA + off) * (BK + PAD_A) + innerColA * VEC],
                &A_ptr[(innerRowA + off) * K + innerColA * VEC],
                sizeof(__nv_bfloat16) * VEC);
        }
        for (uint off = 0; off + rowStrideB <= (uint)BK; off += rowStrideB) {
            __pipeline_memcpy_async(
                &Bs[buf][(innerRowB + off) * (BN + PAD_B) + innerColB * VEC],
                &B_ptr[(innerRowB + off) * N + innerColB * VEC],
                sizeof(__nv_bfloat16) * VEC);
        }
    };

    auto compute_tile = [&](int buf) {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                       __nv_bfloat16, wmma::row_major> frag_a[FRAGS_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                       __nv_bfloat16, wmma::row_major> frag_b[FRAGS_N];

        for (int fk = 0; fk < FRAGS_K; ++fk) {
            for (int fm = 0; fm < FRAGS_M; ++fm) {
                wmma::load_matrix_sync(
                    frag_a[fm],
                    &As[buf][(warpRow * WM + fm * WMMA_M) * (BK + PAD_A) + fk * WMMA_K],
                    BK + PAD_A);
            }
            for (int fn = 0; fn < FRAGS_N; ++fn) {
                wmma::load_matrix_sync(
                    frag_b[fn],
                    &Bs[buf][fk * WMMA_K * (BN + PAD_B) + warpCol * WN + fn * WMMA_N],
                    BN + PAD_B);
            }
            for (int fm = 0; fm < FRAGS_M; ++fm) {
                for (int fn = 0; fn < FRAGS_N; ++fn) {
                    wmma::mma_sync(frag_c[fm][fn], frag_a[fm], frag_b[fn], frag_c[fm][fn]);
                }
            }
        }
    };

    int cur = 0;
    int nxt = 1;

    // Prologue
    async_load_tile(cur, A, B);
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    A += BK;
    B += BK * N;

    for (uint bkIdx = BK; bkIdx < (uint)K; bkIdx += BK) {
        async_load_tile(nxt, A, B);
        __pipeline_commit();

        compute_tile(cur);

        __pipeline_wait_prior(0);
        __syncthreads();

        A += BK;
        B += BK * N;
        cur ^= 1;
        nxt ^= 1;
    }

    // Final tile
    compute_tile(cur);

    // Epilogue
    for (int fm = 0; fm < FRAGS_M; ++fm) {
        for (int fn = 0; fn < FRAGS_N; ++fn) {
            float* C_ptr = C + fm * WMMA_M * N + fn * WMMA_N;
            if (beta != 0.0f) {
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> existing;
                wmma::load_matrix_sync(existing, C_ptr, N, wmma::mem_row_major);
                for (int i = 0; i < frag_c[fm][fn].num_elements; ++i) {
                    frag_c[fm][fn].x[i] =
                        alpha * frag_c[fm][fn].x[i] + beta * existing.x[i];
                }
            } else {
                for (int i = 0; i < frag_c[fm][fn].num_elements; ++i) {
                    frag_c[fm][fn].x[i] *= alpha;
                }
            }
            wmma::store_matrix_sync(C_ptr, frag_c[fm][fn], N, wmma::mem_row_major);
        }
    }
}

// ── Naïve fallback ───────────────────────────────────────────────────────────

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

    if (row >= M || col >= N) return;

    float sum = 0.0f;
    for (int k = 0; k < K; ++k) {
        sum += __bfloat162float(A[row * K + k]) * __bfloat162float(B[k * N + col]);
    }

    if (beta != 0.0f) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    } else {
        C[row * N + col] = alpha * sum;
    }
}

template <int TILE>
void launch_bf16_naive(
    const __nv_bfloat16* d_A,
    const __nv_bfloat16* d_B,
    float* d_C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream)
{
    dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
    dim3 block(TILE, TILE);
    gemm_bf16_naive_kernel<TILE><<<grid, block, 0, stream>>>(
        d_A, d_B, d_C, M, N, K, alpha, beta);
}

template <typename Config>
void launch_selected_bf16(
    const __nv_bfloat16* d_A,
    const __nv_bfloat16* d_B,
    float* d_C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream)
{
    const dim3 grid((N + Config::BN - 1) / Config::BN,
                    (M + Config::BM - 1) / Config::BM);
    gemm_bf16_wmma_kernel<Config><<<grid, Config::NUM_THREADS, 0, stream>>>(
        d_A, d_B, d_C, M, N, K, alpha, beta);
}

} // namespace

// ── Public API ───────────────────────────────────────────────────────────────

constexpr const char* const BF16_RTX5070_VARIANT_ID =
    "bf16_wmma_rtx5070";
constexpr const char* const BF16_RTX5070_VARIANT_DESC =
    "bf16_wmma_cpasync_doublebuf_padsmem_size_tuned_rtx5070";

void launch_gemm_bf16_rtx5070(
    const __nv_bfloat16* d_A,
    const __nv_bfloat16* d_B,
    float*               d_C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream,
    TargetArch arch)
{
    if (requires_naive_fallback(M, N)) {
        launch_bf16_naive<BF16_NAIVE_TILE>(d_A, d_B, d_C, M, N, K, alpha, beta, stream);
        return;
    }

    const int elements = M * N;
    switch (select_size_class(elements, arch)) {
        case SizeClass::Small:
            launch_selected_bf16<SmallBF16Config>(
                d_A, d_B, d_C, M, N, K, alpha, beta, stream);
            break;
        case SizeClass::Medium:
            launch_selected_bf16<MediumBF16Config>(
                d_A, d_B, d_C, M, N, K, alpha, beta, stream);
            break;
        case SizeClass::Large:
            launch_selected_bf16<LargeBF16Config>(
                d_A, d_B, d_C, M, N, K, alpha, beta, stream);
            break;
    }
}

const char* get_variant_id_bf16_rtx5070()
{
    return BF16_RTX5070_VARIANT_ID;
}

const char* get_variant_desc_bf16_rtx5070()
{
    return BF16_RTX5070_VARIANT_DESC;
}