#include <cuda_runtime.h>
#include <cuda_pipeline.h>

// r2z_256_opt2: Option 2 - 128x64 tiles, TM=8, TN=8, 128 threads
//
// TUNING: Taller tiles (BM=128) for better M-dimension parallelism
//         TM=8, TN=8 gives 128 threads = (128/8)*(64/8) = 16*8
//         Each thread handles 8x8=64 floats (more register reuse)
//
// Key optimizations:
//   - Double-buffer SMEM: As[2][BK*BM], Bs[2][BK*BN] = 24 KB total
//   - B tiles loaded via __pipeline_memcpy_async
//   - A tiles use float4 global loads + scatter-transpose into SMEM

namespace r2z_256_opt2_config {
    constexpr int BM = 128;  // Taller tiles
    constexpr int BN = 64;
    constexpr int BK = 16;
    constexpr int WM = 64;
    constexpr int WN = 64;
    constexpr int WNITER = 2;
    constexpr int TM = 8;
    constexpr int TN = 8;
    constexpr int NUM_THREADS = 128;  // (128/8)*(64/8) = 16*8 = 128
}

constexpr const char* const VARIANT_ID   = "r2z_256_opt2";
constexpr const char* const VARIANT_DESC = "128x64_128t_db_async_opt2";

template <int BM, int BN, int BK,
          int WM, int WN, int WNITER,
          int TM, int TN, int NUM_THREADS>
__global__ __launch_bounds__(NUM_THREADS)
void gemm_fp32_r2z2_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float*       __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    // ── Block / warp / thread placement ────────────────────────────────────
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    constexpr int WMITER = (WM * WN) / (32 * TM * TN * WNITER);
    constexpr int WSUBM  = WM / WMITER;
    constexpr int WSUBN  = WN / WNITER;

    const uint warpIdx        = threadIdx.x / 32;
    const uint warpCol        = warpIdx % (BN / WN);
    const uint warpRow        = warpIdx / (BN / WN);
    const uint threadIdxInWarp  = threadIdx.x % 32;
    const uint threadColInWarp  = threadIdxInWarp % (WSUBN / TN);
    const uint threadRowInWarp  = threadIdxInWarp / (WSUBN / TN);

    // ── Double-buffered SMEM ───────────────────────────────────────────────
    // As[2][BK * BM]: A stored transposed (As[k*BM + m]) to eliminate bank conflicts
    // Bs[2][BK * BN]: B stored row-major  (Bs[k*BN + n])
    // Memory: 2*(128*16 + 16*128)*4 = 32 768 bytes = 32 KB  ← fits in 48 KB limit
    __shared__ float As[2][BK * BM];
    __shared__ float Bs[2][BK * BN];

    // ── Global-memory base pointers for each block ─────────────────────────
    A += cRow * BM * K;
    B += cCol * BN;
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    // ── Thread loading indices ──────────────────────────────────────────────
    // Each thread loads float4 (16 B) chunks; threads collectively fill one tile.
    const uint innerRowA  = threadIdx.x / (BK / 4);    // A row within tile
    const uint innerColA  = threadIdx.x % (BK / 4);    // A col group (4 floats)
    constexpr uint rowStrideA = (NUM_THREADS * 4) / BK; // rows covered per float4 pass

    const uint innerRowB  = threadIdx.x / (BN / 4);    // B row within tile
    const uint innerColB  = threadIdx.x % (BN / 4);    // B col group (4 floats)
    constexpr uint rowStrideB = NUM_THREADS / (BN / 4); // rows covered per float4 pass

    // ── Accumulators ───────────────────────────────────────────────────────
    float threadResults[WMITER * TM * WNITER * TN] = {0.0f};
    float regM[WMITER * TM]   = {0.0f};
    float regN[WNITER * TN]   = {0.0f};

    // ── Prefetch tile[0] into buffer 0 ─────────────────────────────────────
    // A: float4 load + scatter-transpose (regular; issued first to hide latency)
    for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
        float4 tmp = reinterpret_cast<const float4*>(
            &A[(innerRowA + offset) * K + innerColA * 4])[0];
        As[0][(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
        As[0][(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
        As[0][(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
        As[0][(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
    }
    // B: async 16-byte copies via cp.async → fills Bs[0] while A writes finish
    for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
        __pipeline_memcpy_async(
            &Bs[0][(innerRowB + offset) * BN + innerColB * 4],
            &B [(innerRowB + offset) * N  + innerColB * 4],
            sizeof(float4));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);   // wait for this thread's async copies
    __syncthreads();            // all threads: A scatter writes + B copies done

    A += BK;
    B += BK * N;

    uint cur = 0, nxt = 1;

    // ── Main K-loop ─────────────────────────────────────────────────────────
    // Each iteration:
    //   1. Issue loads for tile[bkIdx] into nxt buffer   \  overlapped by
    //   2. Compute on cur buffer (tile[bkIdx - BK])       /  hardware
    //   3. Wait for nxt to be ready, sync, swap
    for (uint bkIdx = BK; bkIdx < K; bkIdx += BK) {

        // ── Prefetch tile[bkIdx] into nxt ──────────────────────────────────
        // A: regular float4 loads → scatter-transpose (starts immediately,
        //    overlaps with the B async copies and with subsequent compute)
        for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            float4 tmp = reinterpret_cast<const float4*>(
                &A[(innerRowA + offset) * K + innerColA * 4])[0];
            As[nxt][(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
            As[nxt][(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
            As[nxt][(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
            As[nxt][(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }
        // B: async; hardware copy engine overlaps with the FMA loop below
        for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
            __pipeline_memcpy_async(
                &Bs[nxt][(innerRowB + offset) * BN + innerColB * 4],
                &B [(innerRowB + offset) * N  + innerColB * 4],
                sizeof(float4));
        }
        __pipeline_commit();

        // ── Compute on cur buffer ───────────────────────────────────────────
        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                for (uint i = 0; i < TM; ++i) {
                    regM[wSubRowIdx * TM + i] =
                        As[cur][dotIdx * BM + warpRow * WM +
                                wSubRowIdx * WSUBM + threadRowInWarp * TM + i];
                }
            }
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                for (uint i = 0; i < TN; ++i) {
                    regN[wSubColIdx * TN + i] =
                        Bs[cur][dotIdx * BN + warpCol * WN +
                                wSubColIdx * WSUBN + threadColInWarp * TN + i];
                }
            }
            for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
                for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                            threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                          wSubColIdx * TN + resIdxN] +=
                                regM[wSubRowIdx * TM + resIdxM] *
                                regN[wSubColIdx * TN + resIdxN];
                        }
                    }
                }
            }
        }

        // ── Wait for nxt buffer, sync all threads, swap ─────────────────────
        __pipeline_wait_prior(0);
        __syncthreads();

        A += BK;
        B += BK * N;

        cur ^= 1;
        nxt ^= 1;
    }

    // ── Compute final tile (already in cur after last swap) ─────────────────
    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            for (uint i = 0; i < TM; ++i) {
                regM[wSubRowIdx * TM + i] =
                    As[cur][dotIdx * BM + warpRow * WM +
                            wSubRowIdx * WSUBM + threadRowInWarp * TM + i];
            }
        }
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            for (uint i = 0; i < TN; ++i) {
                regN[wSubColIdx * TN + i] =
                    Bs[cur][dotIdx * BN + warpCol * WN +
                            wSubColIdx * WSUBN + threadColInWarp * TN + i];
            }
        }
        for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
            for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
                for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                    for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                        threadResults[(wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                      wSubColIdx * TN + resIdxN] +=
                            regM[wSubRowIdx * TM + resIdxM] *
                            regN[wSubColIdx * TN + resIdxN];
                    }
                }
            }
        }
    }

    // ── Write results to C (float4 vectorised stores, alpha/beta scaling) ───
    for (uint wSubRowIdx = 0; wSubRowIdx < WMITER; ++wSubRowIdx) {
        for (uint wSubColIdx = 0; wSubColIdx < WNITER; ++wSubColIdx) {
            float* C_interim = C + (wSubRowIdx * WSUBM) * N + wSubColIdx * WSUBN;
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                    float4 tmp = reinterpret_cast<float4*>(
                        &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                                   threadColInWarp * TN + resIdxN])[0];
                    const int i = (wSubRowIdx * TM + resIdxM) * (WNITER * TN) +
                                  wSubColIdx * TN + resIdxN;
                    tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
                    tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
                    tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
                    tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
                    reinterpret_cast<float4*>(
                        &C_interim[(threadRowInWarp * TM + resIdxM) * N +
                                   threadColInWarp * TN + resIdxN])[0] = tmp;
                }
            }
        }
    }
}

void launch_gemm_fp32_r2z_256_opt2(
    const float* d_A,
    const float* d_B,
    float*       d_C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream)
{
    using namespace r2z_256_opt2_config;
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(NUM_THREADS);

    gemm_fp32_r2z2_kernel<BM, BN, BK, WM, WN, WNITER, TM, TN, NUM_THREADS>
        <<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
}

const char* get_variant_id_fp32_r2z_256_opt2()   { return VARIANT_ID; }
const char* get_variant_desc_fp32_r2z_256_opt2() { return VARIANT_DESC; }
