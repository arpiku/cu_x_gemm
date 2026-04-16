// Shared benchmark constants and helper declarations.
//
// This header is intended to collect the common benchmark configuration and the
// utility functions that are currently embedded in `main.cu`, so the benchmark
// driver can be split into smaller pieces.

#pragma once

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <string>

#include "target_arch.h"

// Benchmark dimensions and iteration counts.
inline constexpr int DIMENSIONS[] = {32, 64, 128, 256, 512, 1024, 2048, 4096};
inline constexpr int NUM_DIMS = static_cast<int>(sizeof(DIMENSIONS) / sizeof(DIMENSIONS[0]));

inline constexpr int WARMUP_ITERATIONS = 10;
inline constexpr int MEASURE_ITERATIONS = 50;

// Benchmark result records.
struct BF16Result {
    float custom_ms;
    float cublas_ms;
    float cublas_tc_ms;
    float l2_error;
};

struct FP32Result {
    float custom_ms;
    float pedantic_ms;
    float l2_error;
};

struct FP32TCResult {
    float custom_ms;
    float tc_ms;
    float l2_error;
};

// Dimension lookup helper.
int find_dimension_index(int dim);

// Input initialization helpers.
void init_bf16(__nv_bfloat16* ptr, int n, unsigned seed);
void init_fp32(float* ptr, int n, unsigned seed);

// Error metric helper.
float compute_l2_error(const float* a, const float* b, int n);

// Reusable timing helper for benchmark launches.
template <typename LaunchFn>
float time_launch_ms(cudaStream_t stream, int warmup_iterations, int measure_iterations,
                     LaunchFn&& launch_fn) {
    for (int i = 0; i < warmup_iterations; ++i) {
        launch_fn();
    }
    cudaStreamSynchronize(stream);

    cudaEvent_t start;
    cudaEvent_t stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    for (int i = 0; i < measure_iterations; ++i) {
        launch_fn();
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return elapsed_ms / measure_iterations;
}

// Benchmark path helpers.
std::string benchmark_output_dir(TargetArch arch);
std::string benchmark_csv_path(TargetArch arch);

// Device/architecture helper.
TargetArch detect_target_arch_from_device(const cudaDeviceProp& prop);