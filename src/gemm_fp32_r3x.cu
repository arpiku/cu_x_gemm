#include <cuda_runtime.h>
#include <cuda_pipeline.h>

// r3x: 64x64 tiles, flat thread layout, double-buffer + cp.async
// BM=64, BN=64, BK=16, TM=8, TN=8 → 64 threads, each owns 8x8 output tile
// 4x more blocks than r2z2 at medium sizes → better occupancy for 128–512

namespace r3x_config {
    constexpr int BM         = 64;
    constexpr int BN         = 64;
    constexpr int BK         = 16;
    constexpr int TM         = 8;
    constexpr int TN         = 4;  // Tuned: 4 gives better occupancy than 8
    constexpr int NUM_THREADS = (BM / TM) * (BN / TN); // 128 threads
}

static_assert(r3x_config::NUM_THREADS == 128, "");

constexpr const char* const VARIANT_ID   = "r3x_64x64_db";
constexpr const char* const VARIANT_DESC = "64x64x16_flat_128t_doublebuf_cpasync";

template <int BM, int BN, int BK, int TM, int TN, int NUM_THREADS>
__global__ __launch_bounds__(NUM_THREADS)
void gemm_fp32_r3x_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__       C,
    int M, int N, int K,
    float alpha, float beta)
{
    // ── Thread position in the output tile ──────────────────────────────────
    const uint threadRow = threadIdx.x / (BN / TN); // [0, BM/TM)
    const uint threadCol = threadIdx.x % (BN / TN); // [0, BN/TN)

    // ── Double-buffer shared memory ─────────────────────────────────────────
    // As: transposed layout As[buf][k * BM + m] — bank-conflict-free column reads
    // Bs: row-major        Bs[buf][k * BN + n] — aligned 16-byte cp.async writes
    __shared__ float As[2][BK * BM];
    __shared__ float Bs[2][BK * BN];

    // ── Block-level pointers ─────────────────────────────────────────────────
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    // ── Global-load indices (vectorised) ─────────────────────────────────────
    // A: load BM×BK floats as float4 and scatter-transpose into As
    const uint innerRowA   = threadIdx.x / (BK / 4);          // row in [0, BM)
    const uint innerColA   = threadIdx.x % (BK / 4);          // col group in [0, BK/4)
    constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;        // rows loaded per pass = 16

    // B: load BK×BN floats as float4 via cp.async
    const uint innerRowB   = threadIdx.x / (BN / 4);          // row in [0, BK)
    const uint innerColB   = threadIdx.x % (BN / 4);          // col group in [0, BN/4)
    constexpr uint rowStrideB = NUM_THREADS / (BN / 4);        // rows loaded per pass = 4

    // ── Accumulators ─────────────────────────────────────────────────────────
    float threadResults[TM * TN] = {0.0f};
    float regM[TM];
    float regN[TN];

    // ── Prefetch tile[0] → buffer 0 ─────────────────────────────────────────
    for (uint offset = 0; offset < BM; offset += rowStrideA) {
        if (innerRowA + offset < BM) {
            float4 tmp = reinterpret_cast<const float4*>(
                &A[(innerRowA + offset) * K + innerColA * 4])[0];
            As[0][(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
            As[0][(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
            As[0][(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
            As[0][(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }
    }
    for (uint offset = 0; offset < BK; offset += rowStrideB) {
        if (innerRowB + offset < BK) {
            __pipeline_memcpy_async(
                &Bs[0][(innerRowB + offset) * BN + innerColB * 4],
                &B [(innerRowB + offset) * N  + innerColB * 4],
                sizeof(float4));
        }
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    A += BK;
    B += BK * N;

    uint cur = 0, nxt = 1;

    // ── Main K-loop ──────────────────────────────────────────────────────────
    for (uint bkIdx = BK; bkIdx < K; bkIdx += BK) {
        // Issue async loads for next tile
        for (uint offset = 0; offset < BM; offset += rowStrideA) {
            if (innerRowA + offset < BM) {
                float4 tmp = reinterpret_cast<const float4*>(
                    &A[(innerRowA + offset) * K + innerColA * 4])[0];
                As[nxt][(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
                As[nxt][(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
                As[nxt][(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
                As[nxt][(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
            }
        }
        for (uint offset = 0; offset < BK; offset += rowStrideB) {
            if (innerRowB + offset < BK) {
                __pipeline_memcpy_async(
                    &Bs[nxt][(innerRowB + offset) * BN + innerColB * 4],
                    &B [(innerRowB + offset) * N  + innerColB * 4],
                    sizeof(float4));
            }
        }
        __pipeline_commit();

        // Compute on cur buffer — overlaps with cp.async filling Bs[nxt]
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (uint i = 0; i < TM; ++i)
                regM[i] = As[cur][dotIdx * BM + threadRow * TM + i];
            for (uint i = 0; i < TN; ++i)
                regN[i] = Bs[cur][dotIdx * BN + threadCol * TN + i];
            for (uint resM = 0; resM < TM; ++resM)
                for (uint resN = 0; resN < TN; ++resN)
                    threadResults[resM * TN + resN] += regM[resM] * regN[resN];
        }

        __pipeline_wait_prior(0);
        __syncthreads();
        cur ^= 1;
        nxt ^= 1;

        A += BK;
        B += BK * N;
    }

    // ── Final tile ───────────────────────────────────────────────────────────
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
        for (uint i = 0; i < TM; ++i)
            regM[i] = As[cur][dotIdx * BM + threadRow * TM + i];
        for (uint i = 0; i < TN; ++i)
            regN[i] = Bs[cur][dotIdx * BN + threadCol * TN + i];
        for (uint resM = 0; resM < TM; ++resM)
            for (uint resN = 0; resN < TN; ++resN)
                threadResults[resM * TN + resN] += regM[resM] * regN[resN];
    }

    // ── Write-back (float4 vectorised) ───────────────────────────────────────
    for (uint resM = 0; resM < TM; ++resM) {
        float* row = &C[(threadRow * TM + resM) * N + threadCol * TN];
        for (uint resN = 0; resN < TN; resN += 4) {
            float4 tmp = reinterpret_cast<float4*>(&row[resN])[0];
            const int i = resM * TN + resN;
            tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
            tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
            tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
            tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
            reinterpret_cast<float4*>(&row[resN])[0] = tmp;
        }
    }
}

void launch_gemm_fp32_r3x(
    const float* d_A,
    const float* d_B,
    float*       d_C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream)
{
    dim3 grid(
        (N + r3x_config::BN - 1) / r3x_config::BN,
        (M + r3x_config::BM - 1) / r3x_config::BM
    );
    dim3 block(r3x_config::NUM_THREADS);

    gemm_fp32_r3x_kernel<
        r3x_config::BM, r3x_config::BN, r3x_config::BK,
        r3x_config::TM, r3x_config::TN, r3x_config::NUM_THREADS>
        <<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
}

const char* get_variant_id_fp32_r3x()   { return VARIANT_ID; }
const char* get_variant_desc_fp32_r3x() { return VARIANT_DESC; }