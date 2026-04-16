#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "target_arch.h"

// Shared BF16 tuning metadata and dispatch helpers.
//
// This header intentionally contains only the BF16-specific configuration and
// size-selection logic so the BF16 implementation can be split into smaller
// source files without duplicating thresholds, tile shapes, and launch math.

namespace bf16_gemm {

enum class SizeClass {
    Small,
    Medium,
    Large,
};

struct SizeThresholds {
    int small_max;
    int medium_max;
};

// Fallback threshold for kernels that require at least a 64x64 matrix tile.
inline constexpr int BF16_MIN_WMMA_DIM = 64;
inline constexpr int BF16_NAIVE_TILE   = 16;

// RTX5070 baseline configs.
// These match the current best-performing BF16 WMMA shapes.
struct SmallBF16Config {
    static constexpr int BM = 64;
    static constexpr int BN = 64;
    static constexpr int BK = 16;
    static constexpr int WM = 32;
    static constexpr int WN = 32;
    static constexpr int NUM_THREADS = 128;
};

struct MediumBF16Config {
    static constexpr int BM = 128;
    static constexpr int BN = 128;
    static constexpr int BK = 16;
    static constexpr int WM = 64;
    static constexpr int WN = 64;
    static constexpr int NUM_THREADS = 128;
};

struct LargeBF16Config {
    static constexpr int BM = 128;
    static constexpr int BN = 128;
    static constexpr int BK = 16;
    static constexpr int WM = 32;
    static constexpr int WN = 64;
    static constexpr int NUM_THREADS = 256;
};

// H100-specific configs.
// BK=32 reduces synchronization overhead and is currently the preferred BF16
// choice on Hopper for these matrix ranges.
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

inline constexpr SizeThresholds select_size_thresholds(TargetArch arch) {
    switch (arch) {
        case TargetArch::H100:
            // Keep the smaller 64x64 tile longer on H100, then move straight to
            // the larger 128x128 tile once the matrix is large enough.
            return {1048576, 4194304}; // 1024^2, 2048^2
        case TargetArch::RTX5070:
            return {65536, 1048576};   // 256^2, 1024^2
    }
    return {65536, 1048576};
}

inline constexpr bool requires_naive_fallback(int M, int N) {
    return (M < BF16_MIN_WMMA_DIM) || (N < BF16_MIN_WMMA_DIM);
}

inline constexpr SizeClass select_size_class(int elements, TargetArch arch) {
    const SizeThresholds thresholds = select_size_thresholds(arch);
    if (elements <= thresholds.small_max) {
        return SizeClass::Small;
    }
    if (elements <= thresholds.medium_max) {
        return SizeClass::Medium;
    }
    return SizeClass::Large;
}

template <typename Config>
inline constexpr dim3 make_grid(int M, int N) {
    return dim3((N + Config::BN - 1) / Config::BN,
                (M + Config::BM - 1) / Config::BM);
}

template <typename Config>
inline constexpr dim3 make_block() {
    return dim3(Config::NUM_THREADS);
}

template <typename Config, typename LaunchFn>
inline void launch_configured_bf16(
    LaunchFn&& launch_fn,
    const __nv_bfloat16* d_A,
    const __nv_bfloat16* d_B,
    float* d_C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream)
{
    (void)launch_fn;
    const dim3 grid = make_grid<Config>(M, N);
    const dim3 block = make_block<Config>();
    launch_fn(grid, block, d_A, d_B, d_C, M, N, K, alpha, beta, stream);
}

template <typename SmallFn, typename MediumFn, typename LargeFn>
inline void dispatch_by_size(
    int elements,
    TargetArch arch,
    SmallFn&& small_fn,
    MediumFn&& medium_fn,
    LargeFn&& large_fn)
{
    switch (select_size_class(elements, arch)) {
        case SizeClass::Small:
            small_fn();
            break;
        case SizeClass::Medium:
            medium_fn();
            break;
        case SizeClass::Large:
            large_fn();
            break;
    }
}

} // namespace bf16_gemm