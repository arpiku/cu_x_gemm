#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <mma.h>

#include "gemm_fp32_common.cuh"

// tc1: identical double-buffer + cp.async structure as tc0
// sole delta: wmma::mma_sync replaced with inline PTX wmma.mma.sync

namespace {

// ── Configs (identical to tc0) ────────────────────────────────────────────────

struct SmallTC1Config {
    static constexpr int BM = 64;
    static constexpr int BN = 64;
    static constexpr int BK = 16;
    static constexpr int WM = 32;
    static constexpr int WN = 32;
    static constexpr int NUM_THREADS = 128;

};

struct MediumTC1Config {
    static constexpr int BM = 128;
    static constexpr int BN = 128;
    static constexpr int BK = 16;
    static constexpr int WM = 64;
    static constexpr int WN = 64;
    static constexpr int NUM_THREADS = 128;

};

struct LargeTC1Config {
    static constexpr int BM = 128;
    static constexpr int BN = 128;
    static constexpr int BK = 16;
    static constexpr int WM = 32;
    static constexpr int WN = 64;
    static constexpr int NUM_THREADS = 256;

};

using namespace nvcuda;
using namespace fp32_gemm;

// ── Inline PTX MMA (sole delta vs tc0) ───────────────────────────────────────
// Wraps wmma.mma.sync.aligned.row.row.m16n16k8.f32.tf32.tf32.f32
// Fragment a/b elements are floats in TF32 bit-pattern; "r" requires uint32.

__device__ __forceinline__
void mma_tf32_ptx(
    wmma::fragment<wmma::accumulator,
                   16, 16, 8, float>& d,
    const wmma::fragment<wmma::matrix_a,
                   16, 16, 8, wmma::precision::tf32, wmma::row_major>& a,
    const wmma::fragment<wmma::matrix_b,
                   16, 16, 8, wmma::precision::tf32, wmma::row_major>& b)
{
#if __CUDA_ARCH__ >= 800
    unsigned a0, a1, a2, a3;
    unsigned b0, b1, b2, b3;
    // reinterpret float registers as uint32 for the "r" PTX constraint
    asm("mov.b32 %0, %1;" : "=r"(a0) : "f"(a.x[0]));
    asm("mov.b32 %0, %1;" : "=r"(a1) : "f"(a.x[1]));
    asm("mov.b32 %0, %1;" : "=r"(a2) : "f"(a.x[2]));
    asm("mov.b32 %0, %1;" : "=r"(a3) : "f"(a.x[3]));
    asm("mov.b32 %0, %1;" : "=r"(b0) : "f"(b.x[0]));
    asm("mov.b32 %0, %1;" : "=r"(b1) : "f"(b.x[1]));
    asm("mov.b32 %0, %1;" : "=r"(b2) : "f"(b.x[2]));
    asm("mov.b32 %0, %1;" : "=r"(b3) : "f"(b.x[3]));

    asm volatile(
        "wmma.mma.sync.aligned.row.row.m16n16k8.f32.tf32.tf32.f32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7},"
        "{%8,%9,%10,%11},"
        "{%12,%13,%14,%15},"
        "{%0,%1,%2,%3,%4,%5,%6,%7};\n"
        : "+f"(d.x[0]),"+f"(d.x[1]),"+f"(d.x[2]),"+f"(d.x[3]),
          "+f"(d.x[4]),"+f"(d.x[5]),"+f"(d.x[6]),"+f"(d.x[7])
        : "r"(a0),"r"(a1),"r"(a2),"r"(a3),
          "r"(b0),"r"(b1),"r"(b2),"r"(b3));
#else
    wmma::mma_sync(d, a, b, d);
#endif
}

// ── Kernel ────────────────────────────────────────────────────────────────────

template <typename Config>
__global__ __launch_bounds__(Config::NUM_THREADS)
void gemm_fp32_r2z_tc1_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
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
    constexpr int WMMA_K = 8;

    constexpr int FRAGS_M = WM / WMMA_M;
    constexpr int FRAGS_N = WN / WMMA_N;
    constexpr int FRAGS_K = BK / WMMA_K;

    const uint warpIdx = threadIdx.x / 32;
    const uint warpRow = warpIdx / (BN / WN);
    const uint warpCol = warpIdx % (BN / WN);

    // Padding per row to break SMEM bank conflicts.
    // A: stride BK+PAD_A — 4 rows cycle through banks 0,24,16,8 (96-byte rows)
    // B: stride BN+PAD_B — 4 rows cycle through banks 0,8,16,24 (544-byte rows)
    // Medium SMEM: 2*(128*24 + 16*136)*4 = 41 KB — fits within 48 KB limit.
    constexpr int PAD_A = 8;
    constexpr int PAD_B = 8;

    __shared__ float As[2][BM * (BK + PAD_A)];
    __shared__ float Bs[2][BK * (BN + PAD_B)];

    A += blockIdx.y * BM * K;
    B += blockIdx.x * BN;
    C += (blockIdx.y * BM + warpRow * WM) * N + blockIdx.x * BN + warpCol * WN;

    // Vectorized global-load decomposition
    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;

    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

    // Persistent accumulators
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, float>
        frag_c[FRAGS_M][FRAGS_N];
    for (int fm = 0; fm < FRAGS_M; ++fm)
        for (int fn = 0; fn < FRAGS_N; ++fn)
            wmma::fill_fragment(frag_c[fm][fn], 0.0f);

    // ── Prologue: async-load tile[0] ─────────────────────────────────────────
    for (uint off = 0; off + rowStrideA <= (uint)BM; off += rowStrideA) {
        __pipeline_memcpy_async(
            &As[0][(innerRowA + off) * (BK + PAD_A) + innerColA * 4],
            &A [(innerRowA + off) * K                + innerColA * 4],
            sizeof(float4));
    }
    for (uint off = 0; off + rowStrideB <= (uint)BK; off += rowStrideB) {
        __pipeline_memcpy_async(
            &Bs[0][(innerRowB + off) * (BN + PAD_B) + innerColB * 4],
            &B [(innerRowB + off) * N                + innerColB * 4],
            sizeof(float4));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    A += BK;
    B += BK * N;

    uint cur = 0, nxt = 1;

    // ── Main K-loop ──────────────────────────────────────────────────────────
    for (uint bkIdx = BK; bkIdx < (uint)K; bkIdx += BK) {
        // Prefetch next tile
        for (uint off = 0; off + rowStrideA <= (uint)BM; off += rowStrideA) {
            __pipeline_memcpy_async(
                &As[nxt][(innerRowA + off) * (BK + PAD_A) + innerColA * 4],
                &A      [(innerRowA + off) * K              + innerColA * 4],
                sizeof(float4));
        }
        for (uint off = 0; off + rowStrideB <= (uint)BK; off += rowStrideB) {
            __pipeline_memcpy_async(
                &Bs[nxt][(innerRowB + off) * (BN + PAD_B) + innerColB * 4],
                &B      [(innerRowB + off) * N              + innerColB * 4],
                sizeof(float4));
        }
        __pipeline_commit();

        // Compute on current buffer
        {
            wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K,
                           wmma::precision::tf32, wmma::row_major> frag_a[FRAGS_M];
            wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                           wmma::precision::tf32, wmma::row_major> frag_b[FRAGS_N];
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
                        mma_tf32_ptx(frag_c[fm][fn], frag_a[fm], frag_b[fn]);
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
                       wmma::precision::tf32, wmma::row_major> frag_a[FRAGS_M];
        wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K,
                       wmma::precision::tf32, wmma::row_major> frag_b[FRAGS_N];
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
                    mma_tf32_ptx(frag_c[fm][fn], frag_a[fm], frag_b[fn]);
        }
    }

    // ── Epilogue ─────────────────────────────────────────────────────────────
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
void launch_selected_tc1(
    const float* d_A, const float* d_B, float* d_C,
    int M, int N, int K, float alpha, float beta, cudaStream_t stream)
{
    dim3 grid((N + Config::BN - 1) / Config::BN, (M + Config::BM - 1) / Config::BM);
    gemm_fp32_r2z_tc1_kernel<Config>
        <<<grid, Config::NUM_THREADS, 0, stream>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
}

} // namespace

// ── Public API ────────────────────────────────────────────────────────────────

extern void launch_gemm_fp32_naive(
    const float*, const float*, float*, int, int, int, float, float, cudaStream_t);

constexpr const char* const TC1_VARIANT_ID   = "r2z_tc1";
constexpr const char* const TC1_VARIANT_DESC = "tf32_cpasync_doublebuf_ptx_mma_padsmem";

void launch_gemm_fp32_r2z_tc1(
    const float* d_A, const float* d_B, float* d_C,
    int M, int N, int K, float alpha, float beta, cudaStream_t stream, TargetArch arch)
{
    if (fp32_gemm::requires_naive_fallback(M, N)) {
        launch_gemm_fp32_naive(d_A, d_B, d_C, M, N, K, alpha, beta, stream);
        return;
    }

    switch (fp32_gemm::select_tc_size_class(M * N, arch)) {
        case fp32_gemm::SizeClass::Small:
            launch_selected_tc1<SmallTC1Config>(
                d_A, d_B, d_C, M, N, K, alpha, beta, stream);
            break;
        case fp32_gemm::SizeClass::Medium:
            launch_selected_tc1<MediumTC1Config>(
                d_A, d_B, d_C, M, N, K, alpha, beta, stream);
            break;
        case fp32_gemm::SizeClass::Large:
            launch_selected_tc1<LargeTC1Config>(
                d_A, d_B, d_C, M, N, K, alpha, beta, stream);
            break;
    }
}

const char* get_variant_id_fp32_r2z_tc1()   { return TC1_VARIANT_ID; }
const char* get_variant_desc_fp32_r2z_tc1() { return TC1_VARIANT_DESC; }
