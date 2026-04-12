#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <mma.h>

// TF32 WMMA kernel: float inputs, hardware rounds to TF32 (10-bit mantissa),
// float accumulation.  WMMA fragment shape: M=16, N=16, K=8 (fixed by CUDA).

namespace {

// ── Compile-time configs ──────────────────────────────────────────────────────
// BK=16 keeps double-buffered SMEM within 48 KB per block:
//   2*(BM*BK + BK*BN)*4 ≤ 2*(128*16+16*128)*4 = 32 KB
// This allows more blocks per SM (better occupancy) at the cost of fewer
// K-steps per tile vs BK=32.  Occupancy beats arithmetic intensity here
// because tensor-core throughput is much higher than SMEM bandwidth.

struct SmallTCConfig {
    static constexpr int BM = 64;
    static constexpr int BN = 64;
    static constexpr int BK = 16;
    static constexpr int WM = 32;
    static constexpr int WN = 32;
    static constexpr int NUM_THREADS = 128;   // 4 warps (2M x 2N)
    static constexpr const char* ID = "tc0_small";
    // SMEM: 2*(64*16+16*64)*4 = 16 KB
};

struct MediumTCConfig {
    static constexpr int BM = 128;
    static constexpr int BN = 128;
    static constexpr int BK = 16;
    static constexpr int WM = 64;
    static constexpr int WN = 64;
    static constexpr int NUM_THREADS = 128;   // 4 warps (2M x 2N)
    static constexpr const char* ID = "tc0_medium";
    // SMEM: 2*(128*16+16*128)*4 = 32 KB
};

// 8 warps (4M x 2N): more threads hides memory latency better at large sizes.
struct LargeTCConfig {
    static constexpr int BM = 128;
    static constexpr int BN = 128;
    static constexpr int BK = 16;
    static constexpr int WM = 32;
    static constexpr int WN = 64;
    static constexpr int NUM_THREADS = 256;   // 8 warps (4M x 2N)
    static constexpr const char* ID = "tc0_large";
    // SMEM: 32 KB (same block tile, same BK, more threads)
};

// ── Size selector (same thresholds as r2z) ────────────────────────────────────

enum class SizeClass { Small, Medium, Large };

constexpr int SMALL_MAX_ELEMENTS  = 65536;    // ≤ 256×256
constexpr int MEDIUM_MAX_ELEMENTS = 1048576;  // ≤ 1024×1024

constexpr SizeClass select_size_class(int elements) {
    if (elements <= SMALL_MAX_ELEMENTS)  return SizeClass::Small;
    if (elements <= MEDIUM_MAX_ELEMENTS) return SizeClass::Medium;
    return SizeClass::Large;
}

// ── Kernel ────────────────────────────────────────────────────────────────────

template <typename Config>
__global__ __launch_bounds__(Config::NUM_THREADS)
void gemm_fp32_r2z_tc0_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    using namespace nvcuda;

    constexpr int BM = Config::BM;
    constexpr int BN = Config::BN;
    constexpr int BK = Config::BK;
    constexpr int WM = Config::WM;
    constexpr int WN = Config::WN;
    constexpr int NUM_THREADS = Config::NUM_THREADS;

    // WMMA tile shape (fixed for TF32)
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 8;

    constexpr int FRAGS_M = WM / WMMA_M;
    constexpr int FRAGS_N = WN / WMMA_N;
    constexpr int FRAGS_K = BK / WMMA_K;

    // Warp layout within block: (BN/WN) warps per row
    const uint warpIdx = threadIdx.x / 32;
    const uint warpRow = warpIdx / (BN / WN);
    const uint warpCol = warpIdx % (BN / WN);

    // Double-buffered shared memory: As[2][BM*BK], Bs[2][BK*BN]
    __shared__ float As[2][BM * BK];
    __shared__ float Bs[2][BK * BN];

    // Global pointer bases for this block
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += (blockIdx.y * BM + warpRow * WM) * N + blockIdx.x * BN + warpCol * WN;

    // Loading index decomposition (float4 vectorised)
    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;

    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

    // Accumulator fragments (persistent across K loop)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> frag_c[FRAGS_M][FRAGS_N];
    for (int fm = 0; fm < FRAGS_M; ++fm)
        for (int fn = 0; fn < FRAGS_N; ++fn)
            wmma::fill_fragment(frag_c[fm][fn], 0.0f);

    // ── Prologue: async-load first BK tile ───────────────────────────────────
    for (uint off = 0; off + rowStrideA <= (uint)BM; off += rowStrideA) {
        __pipeline_memcpy_async(
            &As[0][(innerRowA + off) * BK + innerColA * 4],
            &A [(innerRowA + off) * K  + innerColA * 4],
            sizeof(float4));
    }
    for (uint off = 0; off + rowStrideB <= (uint)BK; off += rowStrideB) {
        __pipeline_memcpy_async(
            &Bs[0][(innerRowB + off) * BN + innerColB * 4],
            &B [(innerRowB + off) * N  + innerColB * 4],
            sizeof(float4));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    A += BK;
    B += BK * N;

    uint cur = 0, nxt = 1;

    // ── Main loop: prefetch next tile, compute current tile ──────────────────
    for (uint bkIdx = BK; bkIdx < (uint)K; bkIdx += BK) {
        // Async prefetch next tile
        for (uint off = 0; off + rowStrideA <= (uint)BM; off += rowStrideA) {
            __pipeline_memcpy_async(
                &As[nxt][(innerRowA + off) * BK + innerColA * 4],
                &A      [(innerRowA + off) * K  + innerColA * 4],
                sizeof(float4));
        }
        for (uint off = 0; off + rowStrideB <= (uint)BK; off += rowStrideB) {
            __pipeline_memcpy_async(
                &Bs[nxt][(innerRowB + off) * BN + innerColB * 4],
                &B      [(innerRowB + off) * N  + innerColB * 4],
                sizeof(float4));
        }
        __pipeline_commit();

        // Compute using current buffer
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                       wmma::precision::tf32, wmma::row_major> frag_a[FRAGS_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                       wmma::precision::tf32, wmma::row_major> frag_b[FRAGS_N];

        for (int fk = 0; fk < FRAGS_K; ++fk) {
            for (int fm = 0; fm < FRAGS_M; ++fm)
                wmma::load_matrix_sync(frag_a[fm],
                    &As[cur][(warpRow * WM + fm * WMMA_M) * BK + fk * WMMA_K], BK);
            for (int fn = 0; fn < FRAGS_N; ++fn)
                wmma::load_matrix_sync(frag_b[fn],
                    &Bs[cur][fk * WMMA_K * BN + warpCol * WN + fn * WMMA_N], BN);
            for (int fm = 0; fm < FRAGS_M; ++fm)
                for (int fn = 0; fn < FRAGS_N; ++fn)
                    wmma::mma_sync(frag_c[fm][fn], frag_a[fm], frag_b[fn], frag_c[fm][fn]);
        }

        __pipeline_wait_prior(0);
        __syncthreads();

        A += BK;
        B += BK * N;
        cur ^= 1;
        nxt ^= 1;
    }

    // ── Last tile ─────────────────────────────────────────────────────────────
    {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                       wmma::precision::tf32, wmma::row_major> frag_a[FRAGS_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                       wmma::precision::tf32, wmma::row_major> frag_b[FRAGS_N];

        for (int fk = 0; fk < FRAGS_K; ++fk) {
            for (int fm = 0; fm < FRAGS_M; ++fm)
                wmma::load_matrix_sync(frag_a[fm],
                    &As[cur][(warpRow * WM + fm * WMMA_M) * BK + fk * WMMA_K], BK);
            for (int fn = 0; fn < FRAGS_N; ++fn)
                wmma::load_matrix_sync(frag_b[fn],
                    &Bs[cur][fk * WMMA_K * BN + warpCol * WN + fn * WMMA_N], BN);
            for (int fm = 0; fm < FRAGS_M; ++fm)
                for (int fn = 0; fn < FRAGS_N; ++fn)
                    wmma::mma_sync(frag_c[fm][fn], frag_a[fm], frag_b[fn], frag_c[fm][fn]);
        }
    }

    // ── Epilogue: alpha/beta scale + store to global C ────────────────────────
    for (int fm = 0; fm < FRAGS_M; ++fm) {
        for (int fn = 0; fn < FRAGS_N; ++fn) {
            float* C_ptr = C + fm * WMMA_M * N + fn * WMMA_N;
            if (beta != 0.0f) {
                wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float> existing;
                wmma::load_matrix_sync(existing, C_ptr, N, wmma::mem_row_major);
                for (int i = 0; i < frag_c[fm][fn].num_elements; ++i)
                    frag_c[fm][fn].x[i] =
                        alpha * frag_c[fm][fn].x[i] + beta * existing.x[i];
            } else {
                for (int i = 0; i < frag_c[fm][fn].num_elements; ++i)
                    frag_c[fm][fn].x[i] *= alpha;
            }
            wmma::store_matrix_sync(C_ptr, frag_c[fm][fn], N, wmma::mem_row_major);
        }
    }
}

// ── Launcher helpers ─────────────────────────────────────────────────────────

template <typename Config>
void launch_selected_tc0(
    const float* d_A, const float* d_B, float* d_C,
    int M, int N, int K, float alpha, float beta, cudaStream_t stream,
    const char** selected_variant_out)
{
    if (selected_variant_out) *selected_variant_out = Config::ID;
    dim3 grid((N + Config::BN - 1) / Config::BN, (M + Config::BM - 1) / Config::BM);
    dim3 block(Config::NUM_THREADS);
    gemm_fp32_r2z_tc0_kernel<Config><<<grid, block, 0, stream>>>(
        d_A, d_B, d_C, M, N, K, alpha, beta);
}

} // namespace

// ── Public entry points ───────────────────────────────────────────────────────

// Fallback for matrices smaller than SmallTCConfig's block tile (BM=64, BN=64).
extern void launch_gemm_fp32_naive(
    const float*, const float*, float*, int, int, int, float, float, cudaStream_t);

constexpr const char* const TC0_VARIANT_ID   = "r2z_tc0";
constexpr const char* const TC0_VARIANT_DESC = "size_tuned_tf32_small_medium_large";

void launch_gemm_fp32_r2z_tc0_debug(
    const float* d_A, const float* d_B, float* d_C,
    int M, int N, int K, float alpha, float beta,
    cudaStream_t stream, const char** selected_variant_out)
{
    // SmallTCConfig requires M >= BM=64 and N >= BN=64.
    if (M < SmallTCConfig::BM || N < SmallTCConfig::BN) {
        if (selected_variant_out) *selected_variant_out = "naive_fallback";
        launch_gemm_fp32_naive(d_A, d_B, d_C, M, N, K, alpha, beta, stream);
        return;
    }
    switch (select_size_class(M * N)) {
        case SizeClass::Small:
            launch_selected_tc0<SmallTCConfig>(
                d_A, d_B, d_C, M, N, K, alpha, beta, stream, selected_variant_out);
            break;
        case SizeClass::Medium:
            launch_selected_tc0<MediumTCConfig>(
                d_A, d_B, d_C, M, N, K, alpha, beta, stream, selected_variant_out);
            break;
        case SizeClass::Large:
            launch_selected_tc0<LargeTCConfig>(
                d_A, d_B, d_C, M, N, K, alpha, beta, stream, selected_variant_out);
            break;
    }
}

void launch_gemm_fp32_r2z_tc0(
    const float* d_A, const float* d_B, float* d_C,
    int M, int N, int K, float alpha, float beta, cudaStream_t stream)
{
    launch_gemm_fp32_r2z_tc0_debug(d_A, d_B, d_C, M, N, K, alpha, beta, stream, nullptr);
}

const char* get_variant_id_fp32_r2z_tc0()   { return TC0_VARIANT_ID; }
const char* get_variant_desc_fp32_r2z_tc0() { return TC0_VARIANT_DESC; }
