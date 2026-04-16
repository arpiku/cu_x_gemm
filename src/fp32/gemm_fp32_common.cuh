#pragma once

#include <cuda_runtime.h>

#include "target_arch.h"

// Shared FP32 tuning metadata and dispatch helpers.
//
// This header centralizes the size thresholds and launch-selection utilities
// used by the FP32 CUDA-core and tensor-core paths, so each implementation file
// can stay focused on the kernel body instead of repeating the same dispatch
// logic.

namespace fp32_gemm {

enum class SizeClass {
    Small,
    Medium,
    Large,
};

struct SizeThresholds {
    int small_max;
    int medium_max;
};

// Tiny matrices stay on the naïve kernel.
inline constexpr int FP32_NAIVE_MAX_ELEMENTS = 4096; // 64 x 64

// Smallest matrix size that makes sense for the tuned WMMA/tiled kernels.
inline constexpr int FP32_MIN_WMMA_DIM = 64;

// Default thresholds for the CUDA-core r2z family.
inline constexpr SizeThresholds select_r2z_thresholds(TargetArch arch) {
    switch (arch) {
        case TargetArch::H100:
            // On H100 the 64x64 small kernel remains good much longer.
            return {4194304, 4194304};
        case TargetArch::RTX5070:
            return {65536, 1048576};
    }
    return {65536, 1048576};
}

// Thresholds for the tensor-core FP32 family.
// These are intentionally different from the CUDA-core r2z path.
inline constexpr SizeThresholds select_tc_thresholds(TargetArch arch) {
    switch (arch) {
        case TargetArch::H100:
            return {262144, 4194304};
        case TargetArch::RTX5070:
            return {65536, 1048576};
    }
    return {65536, 1048576};
}

inline constexpr bool requires_naive_fallback(int M, int N) {
    return (M < FP32_MIN_WMMA_DIM) || (N < FP32_MIN_WMMA_DIM);
}

inline constexpr SizeClass select_size_class(int elements, const SizeThresholds& thresholds) {
    if (elements <= thresholds.small_max) {
        return SizeClass::Small;
    }
    if (elements <= thresholds.medium_max) {
        return SizeClass::Medium;
    }
    return SizeClass::Large;
}

inline constexpr SizeClass select_r2z_size_class(int elements, TargetArch arch) {
    return select_size_class(elements, select_r2z_thresholds(arch));
}

inline constexpr SizeClass select_tc_size_class(int elements, TargetArch arch) {
    return select_size_class(elements, select_tc_thresholds(arch));
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

template <typename SmallFn, typename MediumFn, typename LargeFn>
inline void dispatch_by_size(
    int elements,
    const SizeThresholds& thresholds,
    SmallFn&& small_fn,
    MediumFn&& medium_fn,
    LargeFn&& large_fn)
{
    switch (select_size_class(elements, thresholds)) {
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

template <typename SmallFn, typename LargeFn>
inline void dispatch_two_way(
    int elements,
    const SizeThresholds& thresholds,
    SmallFn&& small_fn,
    LargeFn&& large_fn)
{
    if (elements <= thresholds.small_max) {
        small_fn();
    } else {
        large_fn();
    }
}

} // namespace fp32_gemm