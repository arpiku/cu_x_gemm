#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "gemm_bf16_common.cuh"

// H100-specific BF16 WMMA path.
// Split out from the generic BF16 launcher to keep the BF16 implementation
// smaller and to make the H100 tuning knobs explicit.

namespace {

using namespace nvcuda;

// ── H100 configs ─────────────────────────────────────────────────────────────
// H100 benefits from BK=32: fewer sync points and better pipeline overlap.
// Small keeps block count high on mid-range matrices; Large wins once the
// matrix is large enough to justify the bigger tile and thread count.

struct H100SmallBF16Config {
    static constexpr int BM = 64;
    static constexpr int BN = 64;
    static constexpr int BK = 32;
    static constexpr int WM = 32;
    static constexpr int WN = 32;
    static constexpr int NUM_THREADS = 128;
};

struct H100LargeBF16Config {
    static constexpr int BM = 128;
    static constexpr int BN = 128;
    static constexpr int BK = 32;
    static constexpr int WM = 64;
    static constexpr int WN = 64;
    static constexpr int NUM_THREADS = 256;
};



// ── Small fallback kernel ────────────────────────────────────────────────────

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

    if (beta != 0.0f) {
        C[row * N + col] = alpha * sum + beta * C[row * N + col];
    } else {
        C[row * N + col] = alpha * sum;
    }
}

// ── WMMA kernel ──────────────────────────────────────────────────────────────

template <typename Config>
__global__ __launch_bounds__(Config::NUM_THREADS)
void gemm_bf16_wmma_h100_kernel(
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

    const unsigned int warp_idx = threadIdx.x / 32;
    const unsigned int warp_row = warp_idx / (BN / WN);
    const unsigned int warp_col = warp_idx % (BN / WN);

    // Padding breaks SMEM bank aliasing for the wmma loads.
    constexpr int PAD_A = 8;
    constexpr int PAD_B = 8;

    __shared__ __nv_bfloat16 As[2][BM * (BK + PAD_A)];
    __shared__ __nv_bfloat16 Bs[2][BK * (BN + PAD_B)];

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += (blockIdx.y * BM + warp_row * WM) * N + blockIdx.x * BN + warp_col * WN;

    // cp.async copies 16 bytes = 8 BF16 values.
    constexpr int VEC = 8;

    const unsigned int inner_row_a = threadIdx.x / (BK / VEC);
    const unsigned int inner_col_a = threadIdx.x % (BK / VEC);
    constexpr unsigned int row_stride_a = NUM_THREADS / (BK / VEC);

    const unsigned int inner_row_b = threadIdx.x / (BN / VEC);
    const unsigned int inner_col_b = threadIdx.x % (BN / VEC);
    constexpr unsigned int row_stride_b = NUM_THREADS / (BN / VEC);

    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        frag_c[FRAGS_M][FRAGS_N];
    for (int fm = 0; fm < FRAGS_M; ++fm) {
        for (int fn = 0; fn < FRAGS_N; ++fn) {
            wmma::fill_fragment(frag_c[fm][fn], 0.0f);
        }
    }

    // Prologue
    for (unsigned int off = 0; off + row_stride_a <= (unsigned int)BM; off += row_stride_a) {
        __pipeline_memcpy_async(
            &As[0][(inner_row_a + off) * (BK + PAD_A) + inner_col_a * VEC],
            &A[(inner_row_a + off) * K + inner_col_a * VEC],
            sizeof(__nv_bfloat16) * VEC);
    }
    for (unsigned int off = 0; off + row_stride_b <= (unsigned int)BK; off += row_stride_b) {
        __pipeline_memcpy_async(
            &Bs[0][(inner_row_b + off) * (BN + PAD_B) + inner_col_b * VEC],
            &B[(inner_row_b + off) * N + inner_col_b * VEC],
            sizeof(__nv_bfloat16) * VEC);
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    A += BK;
    B += BK * N;

    unsigned int cur = 0;
    unsigned int nxt = 1;

    for (unsigned int bk_idx = BK; bk_idx < (unsigned int)K; bk_idx += BK) {
        for (unsigned int off = 0; off + row_stride_a <= (unsigned int)BM; off += row_stride_a) {
            __pipeline_memcpy_async(
                &As[nxt][(inner_row_a + off) * (BK + PAD_A) + inner_col_a * VEC],
                &A[(inner_row_a + off) * K + inner_col_a * VEC],
                sizeof(__nv_bfloat16) * VEC);
        }
        for (unsigned int off = 0; off + row_stride_b <= (unsigned int)BK; off += row_stride_b) {
            __pipeline_memcpy_async(
                &Bs[nxt][(inner_row_b + off) * (BN + PAD_B) + inner_col_b * VEC],
                &B[(inner_row_b + off) * N + inner_col_b * VEC],
                sizeof(__nv_bfloat16) * VEC);
        }
        __pipeline_commit();

        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                       __nv_bfloat16, wmma::row_major> frag_a[FRAGS_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                       __nv_bfloat16, wmma::row_major> frag_b[FRAGS_N];

        for (int fk = 0; fk < FRAGS_K; ++fk) {
            for (int fm = 0; fm < FRAGS_M; ++fm) {
                wmma::load_matrix_sync(
                    frag_a[fm],
                    &As[cur][(warp_row * WM + fm * WMMA_M) * (BK + PAD_A) + fk * WMMA_K],
                    BK + PAD_A);
            }
            for (int fn = 0; fn < FRAGS_N; ++fn) {
                wmma::load_matrix_sync(
                    frag_b[fn],
                    &Bs[cur][fk * WMMA_K * (BN + PAD_B) + warp_col * WN + fn * WMMA_N],
                    BN + PAD_B);
            }
            for (int fm = 0; fm < FRAGS_M; ++fm) {
                for (int fn = 0; fn < FRAGS_N; ++fn) {
                    wmma::mma_sync(frag_c[fm][fn], frag_a[fm], frag_b[fn], frag_c[fm][fn]);
                }
            }
        }

        __pipeline_wait_prior(0);
        __syncthreads();

        A += BK;
        B += BK * N;
        cur ^= 1;
        nxt ^= 1;
    }

    // Final tile
    {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                       __nv_bfloat16, wmma::row_major> frag_a[FRAGS_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                       __nv_bfloat16, wmma::row_major> frag_b[FRAGS_N];

        for (int fk = 0; fk < FRAGS_K; ++fk) {
            for (int fm = 0; fm < FRAGS_M; ++fm) {
                wmma::load_matrix_sync(
                    frag_a[fm],
                    &As[cur][(warp_row * WM + fm * WMMA_M) * (BK + PAD_A) + fk * WMMA_K],
                    BK + PAD_A);
            }
            for (int fn = 0; fn < FRAGS_N; ++fn) {
                wmma::load_matrix_sync(
                    frag_b[fn],
                    &Bs[cur][fk * WMMA_K * (BN + PAD_B) + warp_col * WN + fn * WMMA_N],
                    BN + PAD_B);
            }
            for (int fm = 0; fm < FRAGS_M; ++fm) {
                for (int fn = 0; fn < FRAGS_N; ++fn) {
                    wmma::mma_sync(frag_c[fm][fn], frag_a[fm], frag_b[fn], frag_c[fm][fn]);
                }
            }
        }
    }

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

// ── Launcher helpers ─────────────────────────────────────────────────────────

template <typename Config>
void launch_selected_bf16_h100(
    const __nv_bfloat16* d_A,
    const __nv_bfloat16* d_B,
    float*               d_C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream)
{
    dim3 grid((N + Config::BN - 1) / Config::BN, (M + Config::BM - 1) / Config::BM);
    gemm_bf16_wmma_h100_kernel<Config>
        <<<grid, Config::NUM_THREADS, 0, stream>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
}

constexpr const char* const H100_BF16_VARIANT_ID   = "bf16_h100";
constexpr const char* const H100_BF16_VARIANT_DESC = "bf16_wmma_h100_small_large";

} // namespace

// ── Public API ───────────────────────────────────────────────────────────────

void launch_gemm_bf16_h100(
    const __nv_bfloat16* d_A,
    const __nv_bfloat16* d_B,
    float*               d_C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream)
{
    if (bf16_gemm::requires_naive_fallback(M, N)) {
        constexpr int TILE = bf16_gemm::BF16_NAIVE_TILE;
        dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
        gemm_bf16_naive_kernel<TILE>
            <<<grid, dim3(TILE, TILE, 1), 0, stream>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
        return;
    }

    const int elements = M * N;
    switch (bf16_gemm::select_size_class(elements, TargetArch::H100)) {
        case bf16_gemm::SizeClass::Small:
        case bf16_gemm::SizeClass::Medium:
            launch_selected_bf16_h100<H100SmallBF16Config>(
                d_A, d_B, d_C, M, N, K, alpha, beta, stream);
            break;
        case bf16_gemm::SizeClass::Large:
            launch_selected_bf16_h100<H100LargeBF16Config>(
                d_A, d_B, d_C, M, N, K, alpha, beta, stream);
            break;
    }
}

const char* get_variant_id_bf16_h100()   { return H100_BF16_VARIANT_ID; }
const char* get_variant_desc_bf16_h100() { return H100_BF16_VARIANT_DESC; }