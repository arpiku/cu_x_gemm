#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <mma.h>

// gemm_bf16_r1: BF16 input -> FP32 accumulation
// WMMA m16n16k16, cp.async double-buffer, padded SMEM, size-tuned configs.
// Replaces the naive r0 kernel.

namespace {

using namespace nvcuda;

// ── Configs ───────────────────────────────────────────────────────────────────
// BK=16 matches WMMA_K=16 for BF16.
// SMEM per block (medium, double-buffered, with padding):
//   2 * (BM*(BK+PAD_A) + BK*(BN+PAD_B)) * 2 bytes
//   = 2 * (128*24 + 16*136) * 2 = ~21 KB  ← well within 48 KB

struct SmallBF16Config {
    static constexpr int BM = 64;
    static constexpr int BN = 64;
    static constexpr int BK = 16;
    static constexpr int WM = 32;
    static constexpr int WN = 32;
    static constexpr int NUM_THREADS = 128;   // 4 warps, 2M×2N warp grid

};

struct MediumBF16Config {
    static constexpr int BM = 128;
    static constexpr int BN = 128;
    static constexpr int BK = 16;
    static constexpr int WM = 64;
    static constexpr int WN = 64;
    static constexpr int NUM_THREADS = 128;   // 4 warps, 2M×2N warp grid

};

struct LargeBF16Config {
    static constexpr int BM = 128;
    static constexpr int BN = 128;
    static constexpr int BK = 16;
    static constexpr int WM = 32;
    static constexpr int WN = 64;
    static constexpr int NUM_THREADS = 256;   // 8 warps, 4M×2N warp grid

};

constexpr int SMALL_MAX_ELEMENTS  = 65536;    // <= 256×256
constexpr int MEDIUM_MAX_ELEMENTS = 1048576;  // <= 1024×1024

// ── Kernel ────────────────────────────────────────────────────────────────────

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
    constexpr int WMMA_K = 16;   // BF16 WMMA uses K=16

    constexpr int FRAGS_M = WM / WMMA_M;
    constexpr int FRAGS_N = WN / WMMA_N;
    constexpr int FRAGS_K = BK / WMMA_K;   // = 1 always (BK=16=WMMA_K)

    // Warp position within the block tile
    const uint warpIdx = threadIdx.x / 32;
    const uint warpRow = warpIdx / (BN / WN);
    const uint warpCol = warpIdx % (BN / WN);

    // Padded SMEM strides (in BF16 elements).
    // PAD=8 adds 16 bytes per row — shifts start bank by 1 per row,
    // breaking the periodic aliasing that wmma.load hits on bare power-of-2 strides.
    // Medium double-buffer footprint: 2*(128*24 + 16*136)*2 ≈ 21 KB ✓
    constexpr int PAD_A = 8;
    constexpr int PAD_B = 8;

    __shared__ __nv_bfloat16 As[2][BM * (BK + PAD_A)];
    __shared__ __nv_bfloat16 Bs[2][BK * (BN + PAD_B)];

    // Block-level base pointers
    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += (blockIdx.y * BM + warpRow * WM) * N + blockIdx.x * BN + warpCol * WN;

    // cp.async copies 16 bytes = 8 BF16 elements at a time.
    constexpr int VEC = 8;

    // A: BM×BK BF16 elements, loaded row-major
    const uint innerRowA  = threadIdx.x / (BK / VEC);
    const uint innerColA  = threadIdx.x % (BK / VEC);
    constexpr uint rowStrideA = NUM_THREADS / (BK / VEC);

    // B: BK×BN BF16 elements, loaded row-major
    const uint innerRowB  = threadIdx.x / (BN / VEC);
    const uint innerColB  = threadIdx.x % (BN / VEC);
    constexpr uint rowStrideB = NUM_THREADS / (BN / VEC);

    // Persistent FP32 accumulators (across full K)
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        frag_c[FRAGS_M][FRAGS_N];
    for (int fm = 0; fm < FRAGS_M; ++fm)
        for (int fn = 0; fn < FRAGS_N; ++fn)
            wmma::fill_fragment(frag_c[fm][fn], 0.0f);

    // ── Prologue: async-load tile[0] into buffer 0 ───────────────────────────
    for (uint off = 0; off + rowStrideA <= (uint)BM; off += rowStrideA) {
        __pipeline_memcpy_async(
            &As[0][(innerRowA + off) * (BK + PAD_A) + innerColA * VEC],
            &A [(innerRowA + off) * K                + innerColA * VEC],
            sizeof(__nv_bfloat16) * VEC);
    }
    for (uint off = 0; off + rowStrideB <= (uint)BK; off += rowStrideB) {
        __pipeline_memcpy_async(
            &Bs[0][(innerRowB + off) * (BN + PAD_B) + innerColB * VEC],
            &B [(innerRowB + off) * N                + innerColB * VEC],
            sizeof(__nv_bfloat16) * VEC);
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    A += BK;
    B += BK * N;

    uint cur = 0, nxt = 1;

    // ── Main K-loop: prefetch nxt, compute cur ────────────────────────────────
    for (uint bkIdx = BK; bkIdx < (uint)K; bkIdx += BK) {
        // Issue async loads for next tile into buffer nxt
        for (uint off = 0; off + rowStrideA <= (uint)BM; off += rowStrideA) {
            __pipeline_memcpy_async(
                &As[nxt][(innerRowA + off) * (BK + PAD_A) + innerColA * VEC],
                &A      [(innerRowA + off) * K              + innerColA * VEC],
                sizeof(__nv_bfloat16) * VEC);
        }
        for (uint off = 0; off + rowStrideB <= (uint)BK; off += rowStrideB) {
            __pipeline_memcpy_async(
                &Bs[nxt][(innerRowB + off) * (BN + PAD_B) + innerColB * VEC],
                &B      [(innerRowB + off) * N              + innerColB * VEC],
                sizeof(__nv_bfloat16) * VEC);
        }
        __pipeline_commit();

        // Compute on cur buffer — overlaps with cp.async fills of nxt
        {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           __nv_bfloat16, wmma::row_major> frag_a[FRAGS_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           __nv_bfloat16, wmma::row_major> frag_b[FRAGS_N];

            for (int fk = 0; fk < FRAGS_K; ++fk) {
                for (int fm = 0; fm < FRAGS_M; ++fm)
                    wmma::load_matrix_sync(frag_a[fm],
                        &As[cur][(warpRow * WM + fm * WMMA_M) * (BK + PAD_A) + fk * WMMA_K],
                        BK + PAD_A);
                for (int fn = 0; fn < FRAGS_N; ++fn)
                    wmma::load_matrix_sync(frag_b[fn],
                        &Bs[cur][fk * WMMA_K * (BN + PAD_B) + warpCol * WN + fn * WMMA_N],
                        BN + PAD_B);
                for (int fm = 0; fm < FRAGS_M; ++fm)
                    for (int fn = 0; fn < FRAGS_N; ++fn)
                        wmma::mma_sync(frag_c[fm][fn], frag_a[fm], frag_b[fn], frag_c[fm][fn]);
            }
        }

        __pipeline_wait_prior(0);
        __syncthreads();

        A += BK;
        B += BK * N;
        cur ^= 1;
        nxt ^= 1;
    }

    // ── Final tile ───────────────────────────────────────────────────────────
    {
        wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                       __nv_bfloat16, wmma::row_major> frag_a[FRAGS_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                       __nv_bfloat16, wmma::row_major> frag_b[FRAGS_N];

        for (int fk = 0; fk < FRAGS_K; ++fk) {
            for (int fm = 0; fm < FRAGS_M; ++fm)
                wmma::load_matrix_sync(frag_a[fm],
                    &As[cur][(warpRow * WM + fm * WMMA_M) * (BK + PAD_A) + fk * WMMA_K],
                    BK + PAD_A);
            for (int fn = 0; fn < FRAGS_N; ++fn)
                wmma::load_matrix_sync(frag_b[fn],
                    &Bs[cur][fk * WMMA_K * (BN + PAD_B) + warpCol * WN + fn * WMMA_N],
                    BN + PAD_B);
            for (int fm = 0; fm < FRAGS_M; ++fm)
                for (int fn = 0; fn < FRAGS_N; ++fn)
                    wmma::mma_sync(frag_c[fm][fn], frag_a[fm], frag_b[fn], frag_c[fm][fn]);
        }
    }

    // ── Epilogue: alpha/beta scale, store to float C ──────────────────────────
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

// ── Launcher template ─────────────────────────────────────────────────────────

template <typename Config>
void launch_selected_bf16(
    const __nv_bfloat16* d_A, const __nv_bfloat16* d_B, float* d_C,
    int M, int N, int K, float alpha, float beta, cudaStream_t stream)
{
    dim3 grid((N + Config::BN - 1) / Config::BN, (M + Config::BM - 1) / Config::BM);
    gemm_bf16_wmma_kernel<Config>
        <<<grid, Config::NUM_THREADS, 0, stream>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
}

} // namespace

// ── Naive fallback for sizes below SmallBF16Config (M or N < 64) ─────────────

template <int TILE>
__global__
void gemm_bf16_naive_kernel(
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
    for (int k = 0; k < K; ++k)
        sum += __bfloat162float(A[row * K + k]) * __bfloat162float(B[k * N + col]);
    C[row * N + col] = (beta != 0.0f)
        ? alpha * sum + beta * C[row * N + col]
        : alpha * sum;
}

// ── Public API ────────────────────────────────────────────────────────────────

constexpr const char* const BF16_VARIANT_ID   = "r1_bf16";
constexpr const char* const BF16_VARIANT_DESC = "bf16_wmma_m16n16k16_cpasync_doublebuf_padsmem";

void launch_gemm_bf16(
    const __nv_bfloat16* d_A,
    const __nv_bfloat16* d_B,
    float*               d_C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream)
{
    // WMMA kernel requires M >= BM=64 and N >= BN=64; fall back for tiny sizes.
    if (M < SmallBF16Config::BM || N < SmallBF16Config::BN) {
        constexpr int TILE = 16;
        dim3 grid((N + TILE - 1) / TILE, (M + TILE - 1) / TILE);
        gemm_bf16_naive_kernel<TILE>
            <<<grid, {TILE, TILE, 1}, 0, stream>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
        return;
    }

    const int e = M * N;
    if (e <= SMALL_MAX_ELEMENTS)
        launch_selected_bf16<SmallBF16Config>(d_A, d_B, d_C, M, N, K, alpha, beta, stream);
    else if (e <= MEDIUM_MAX_ELEMENTS)
        launch_selected_bf16<MediumBF16Config>(d_A, d_B, d_C, M, N, K, alpha, beta, stream);
    else
        launch_selected_bf16<LargeBF16Config>(d_A, d_B, d_C, M, N, K, alpha, beta, stream);
}

const char* get_variant_id_bf16()   { return BF16_VARIANT_ID; }
const char* get_variant_desc_bf16() { return BF16_VARIANT_DESC; }