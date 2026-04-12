#include <cuda_runtime.h>

namespace tile_config {
    constexpr int BM = 128;
    constexpr int BN = 128;
    constexpr int BK = 16;
    constexpr int TM = 8;
    constexpr int TN = 8;
    constexpr int THREADS = 256;
}

constexpr const char* const VARIANT_ID = "r2x_float4";
constexpr const char* const VARIANT_DESC = "128x128x16_f4_transA_256t";

template <int BM, int BN, int BK, int TM, int TN, int NUM_THREADS>
__global__ void gemm_fp32_r2x_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    constexpr int WM = TM * 16;
    constexpr int WN = TN * 16;
    constexpr int WMITER = (BM + WM - 1) / WM;
    constexpr int WNITER = (BN + WN - 1) / WN;

    const int threadCol = threadIdx.x % (WN / TN);
    const int threadRow = threadIdx.x / (WN / TN);

    __shared__ float As[BM * BK];
    __shared__ float Bs[BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;

    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

    float threadResults[WMITER * WNITER * TM * TN] = {0.0f};
    float regM[TM] = {0.0f};
    float regN[TN] = {0.0f};

    for (uint bkIdx = 0; bkIdx < K; bkIdx += BK) {
        for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            float4 tmp = reinterpret_cast<const float4*>(&A[(innerRowA + offset) * K + innerColA * 4])[0];
            As[(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
            As[(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
            As[(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
            As[(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }

        for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
            reinterpret_cast<float4*>(&Bs[(innerRowB + offset) * BN + innerColB * 4])[0] =
                reinterpret_cast<const float4*>(&B[(innerRowB + offset) * N + innerColB * 4])[0];
        }
        __syncthreads();

        for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
            for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
                for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
                    for (uint i = 0; i < TM; ++i) {
                        regM[i] = As[dotIdx * BM + (wmIdx * WM) + threadRow * TM + i];
                    }
                    for (uint i = 0; i < TN; ++i) {
                        regN[i] = Bs[dotIdx * BN + (wnIdx * WN) + threadCol * TN + i];
                    }
                    for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                        for (uint resIdxN = 0; resIdxN < TN; ++resIdxN) {
                            threadResults[(wmIdx * TM + resIdxM) * (WNITER * TN) +
                                          wnIdx * TN + resIdxN] +=
                                regM[resIdxM] * regN[resIdxN];
                        }
                    }
                }
            }
        }
        __syncthreads();

        A += BK;
        B += BK * N;
    }

    for (uint wmIdx = 0; wmIdx < WMITER; ++wmIdx) {
        for (uint wnIdx = 0; wnIdx < WNITER; ++wnIdx) {
            float* C_interim = C + (wmIdx * WM * N) + (wnIdx * WN);
            for (uint resIdxM = 0; resIdxM < TM; ++resIdxM) {
                for (uint resIdxN = 0; resIdxN < TN; resIdxN += 4) {
                    float4 tmp = reinterpret_cast<float4*>(&C_interim[(threadRow * TM + resIdxM) * N +
                                                                       threadCol * TN + resIdxN])[0];
                    const int i = (wmIdx * TM + resIdxM) * (WNITER * TN) + wnIdx * TN + resIdxN;
                    tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
                    tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
                    tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
                    tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
                    reinterpret_cast<float4*>(&C_interim[(threadRow * TM + resIdxM) * N +
                                                         threadCol * TN + resIdxN])[0] = tmp;
                }
            }
        }
    }
}

void launch_gemm_fp32_r2x(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream)
{
    dim3 grid(
        (N + tile_config::BN - 1) / tile_config::BN,
        (M + tile_config::BM - 1) / tile_config::BM
    );
    dim3 block(tile_config::THREADS);

    gemm_fp32_r2x_kernel<tile_config::BM, tile_config::BN, tile_config::BK,
                         tile_config::TM, tile_config::TN, tile_config::THREADS>
        <<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
}

const char* get_variant_id_fp32_r2x() { return VARIANT_ID; }
const char* get_variant_desc_fp32_r2x() { return VARIANT_DESC; }