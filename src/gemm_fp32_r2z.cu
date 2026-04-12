#include <cuda_pipeline.h>
#include <cuda_runtime.h>

namespace {

struct SmallConfig {
    static constexpr int BM = 64;
    static constexpr int BN = 64;
    static constexpr int BK = 32;
    static constexpr int WM = 32;
    static constexpr int WN = 32;
    static constexpr int WNITER = 1;
    static constexpr int TM = 8;
    static constexpr int TN = 4;
    static constexpr int NUM_THREADS = 128;
    static constexpr const char* ID = "r2z_small";
};

struct MediumConfig {
    static constexpr int BM = 64;
    static constexpr int BN = 64;
    static constexpr int BK = 16;
    static constexpr int WM = 32;
    static constexpr int WN = 32;
    static constexpr int WNITER = 1;
    static constexpr int TM = 8;
    static constexpr int TN = 4;
    static constexpr int NUM_THREADS = 128;
    static constexpr const char* ID = "r2z_medium";
};

struct LargeConfig {
    static constexpr int BM = 128;
    static constexpr int BN = 128;
    static constexpr int BK = 16;
    static constexpr int WM = 64;
    static constexpr int WN = 64;
    static constexpr int WNITER = 4;
    static constexpr int TM = 8;
    static constexpr int TN = 4;
    static constexpr int NUM_THREADS = 128;
    static constexpr const char* ID = "r2z_large";
};

enum class SizeClass {
    Small,
    Medium,
    Large,
};

constexpr int SMALL_MAX_ELEMENTS = 65536;      // 256x256
constexpr int MEDIUM_MAX_ELEMENTS = 1048576;   // 1024x1024

constexpr SizeClass select_size_class(int elements) {
    if (elements <= SMALL_MAX_ELEMENTS) {
        return SizeClass::Small;
    }
    if (elements <= MEDIUM_MAX_ELEMENTS) {
        return SizeClass::Medium;
    }
    return SizeClass::Large;
}

template <typename Config>
__global__ __launch_bounds__(Config::NUM_THREADS)
void gemm_fp32_r2z_kernel(
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
    constexpr int WNITER = Config::WNITER;
    constexpr int TM = Config::TM;
    constexpr int TN = Config::TN;
    constexpr int NUM_THREADS = Config::NUM_THREADS;

    constexpr int WMITER = (WM * WN) / (32 * TM * TN * WNITER);
    constexpr int WSUBM = WM / WMITER;
    constexpr int WSUBN = WN / WNITER;

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;

    const uint warpIdx = threadIdx.x / 32;
    const uint warpCol = warpIdx % (BN / WN);
    const uint warpRow = warpIdx / (BN / WN);
    const uint threadIdxInWarp = threadIdx.x % 32;
    const uint threadColInWarp = threadIdxInWarp % (WSUBN / TN);
    const uint threadRowInWarp = threadIdxInWarp / (WSUBN / TN);

    // A is stored transposed in SMEM to avoid bank conflicts on column-wise reads.
    __shared__ float As[2][BK * BM];
    __shared__ float Bs[2][BK * BN];

    A += cRow * BM * K;
    B += cCol * BN;
    C += (cRow * BM + warpRow * WM) * N + cCol * BN + warpCol * WN;

    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;

    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

    float threadResults[WMITER * TM * WNITER * TN] = {0.0f};
    float regM[WMITER * TM] = {0.0f};
    float regN[WNITER * TN] = {0.0f};

    for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
        float4 tmp = reinterpret_cast<const float4*>(
            &A[(innerRowA + offset) * K + innerColA * 4])[0];
        As[0][(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
        As[0][(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
        As[0][(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
        As[0][(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
    }

    for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
        __pipeline_memcpy_async(
            &Bs[0][(innerRowB + offset) * BN + innerColB * 4],
            &B[(innerRowB + offset) * N + innerColB * 4],
            sizeof(float4));
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    A += BK;
    B += BK * N;

    uint cur = 0;
    uint nxt = 1;

    for (uint bkIdx = BK; bkIdx < K; bkIdx += BK) {
        for (uint offset = 0; offset + rowStrideA <= BM; offset += rowStrideA) {
            float4 tmp = reinterpret_cast<const float4*>(
                &A[(innerRowA + offset) * K + innerColA * 4])[0];
            As[nxt][(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
            As[nxt][(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
            As[nxt][(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
            As[nxt][(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }

        for (uint offset = 0; offset + rowStrideB <= BK; offset += rowStrideB) {
            __pipeline_memcpy_async(
                &Bs[nxt][(innerRowB + offset) * BN + innerColB * 4],
                &B[(innerRowB + offset) * N + innerColB * 4],
                sizeof(float4));
        }
        __pipeline_commit();

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

        __pipeline_wait_prior(0);
        __syncthreads();

        A += BK;
        B += BK * N;

        cur ^= 1;
        nxt ^= 1;
    }

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

template <typename Config>
void launch_gemm_fp32_r2z_impl(
    const float* d_A,
    const float* d_B,
    float*       d_C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream)
{
    dim3 grid((N + Config::BN - 1) / Config::BN, (M + Config::BM - 1) / Config::BM);
    dim3 block(Config::NUM_THREADS);

    gemm_fp32_r2z_kernel<Config><<<grid, block, 0, stream>>>(d_A, d_B, d_C, M, N, K, alpha, beta);
}

template <typename Config>
void launch_selected_r2z(
    const float* d_A,
    const float* d_B,
    float*       d_C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream,
    const char** selected_variant_out)
{
    if (selected_variant_out) {
        *selected_variant_out = Config::ID;
    }
    launch_gemm_fp32_r2z_impl<Config>(d_A, d_B, d_C, M, N, K, alpha, beta, stream);
}

} // namespace

constexpr const char* const VARIANT_ID = "r2z";
constexpr const char* const VARIANT_DESC = "size_tuned_small_medium_large";

void launch_gemm_fp32_r2z_debug(
    const float* d_A,
    const float* d_B,
    float*       d_C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream,
    const char** selected_variant_out);

void launch_gemm_fp32_r2z(
    const float* d_A,
    const float* d_B,
    float*       d_C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream)
{
    launch_gemm_fp32_r2z_debug(d_A, d_B, d_C, M, N, K, alpha, beta, stream, nullptr);
}

void launch_gemm_fp32_r2z_debug(
    const float* d_A,
    const float* d_B,
    float*       d_C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream,
    const char** selected_variant_out)
{
    const int elements = M * N;
    switch (select_size_class(elements)) {
        case SizeClass::Small:
            launch_selected_r2z<SmallConfig>(d_A, d_B, d_C, M, N, K, alpha, beta, stream,
                                            selected_variant_out);
            break;
        case SizeClass::Medium:
            launch_selected_r2z<MediumConfig>(d_A, d_B, d_C, M, N, K, alpha, beta, stream,
                                             selected_variant_out);
            break;
        case SizeClass::Large:
            launch_selected_r2z<LargeConfig>(d_A, d_B, d_C, M, N, K, alpha, beta, stream,
                                             selected_variant_out);
            break;
    }
}

const char* get_variant_id_fp32_r2z() { return VARIANT_ID; }
const char* get_variant_desc_fp32_r2z() { return VARIANT_DESC; }
