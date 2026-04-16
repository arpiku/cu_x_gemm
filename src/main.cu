#include <array>
#include <cublas_v2.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include <cuda_runtime.h>

#include "target_arch.h"

extern void cublas_gemm_fp32_pedantic(
    cublasHandle_t handle,
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K,
    float alpha, float beta);

extern const char* get_variant_id_fp32_pedantic();
extern const char* get_variant_desc_fp32_pedantic();

static constexpr int DIMENSIONS[] = {32, 64, 128, 256, 512, 1024, 2048, 4096};
static constexpr int NUM_DIMS = static_cast<int>(sizeof(DIMENSIONS) / sizeof(DIMENSIONS[0]));
static constexpr int WARMUP_ITERATIONS = 10;
static constexpr int MEASURE_ITERATIONS = 50;

struct PedanticResult {
    float pedantic_ms;
};

static void init_fp32(float* ptr, int n, unsigned seed) {
    std::srand(seed);
    for (int i = 0; i < n; ++i) {
        ptr[i] = (std::rand() / float(RAND_MAX)) * 2.0f - 1.0f;
    }
}

template <typename LaunchFn>
static float time_launch_ms(cudaStream_t stream, int warmup_iterations, int measure_iterations,
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

static TargetArch detect_target_arch_from_device(const cudaDeviceProp& prop) {
    if (prop.major == 9) return TargetArch::H100;
    if (prop.major == 12) return TargetArch::RTX5070;

    std::printf("Unsupported SM %d.%d; defaulting to RTX5070\n", prop.major, prop.minor);
    return TargetArch::RTX5070;
}

static std::string benchmark_output_dir(TargetArch arch) {
    return std::string("results/") + target_arch_name(arch);
}

static std::string benchmark_csv_path(TargetArch arch) {
    return benchmark_output_dir(arch) + "/benchmark_results.csv";
}

static void benchmark_fp32_pedantic(int dim, cublasHandle_t handle, cudaStream_t stream,
                                    PedanticResult* out) {
    const int elements = dim * dim;
    const size_t bytes = static_cast<size_t>(elements) * sizeof(float);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    float* h_A = static_cast<float*>(std::malloc(bytes));
    float* h_B = static_cast<float*>(std::malloc(bytes));
    init_fp32(h_A, elements, 42);
    init_fp32(h_B, elements, 43);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, bytes);

    out->pedantic_ms = time_launch_ms(stream, WARMUP_ITERATIONS, MEASURE_ITERATIONS, [&] {
        cublas_gemm_fp32_pedantic(handle, d_A, d_B, d_C, dim, dim, dim, 1.0f, 0.0f);
    });

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    std::free(h_A);
    std::free(h_B);
}

static void print_report(const PedanticResult* results, const char* variant_id) {
    std::printf("=== FP32 (Pedantic / CUDA Cores) ===\n");
    std::printf("# Reference: cuBLAS COMPUTE_32F_PEDANTIC\n");
    std::printf("%-6s %-14s %10s\n", "Dim", "Variant", "Pedantic(ms)");
    std::printf("%s\n", std::string(36, '-').c_str());

    for (int i = 0; i < NUM_DIMS; ++i) {
        std::printf("%-6d %-14s %10.4f\n",
                    DIMENSIONS[i], variant_id, results[i].pedantic_ms);
    }
}

static void write_csv_with_arch(
    const std::string& path,
    const int* dims,
    int num_dims,
    const PedanticResult* results,
    const char* target_arch,
    const char* gpu_name,
    int sm_major,
    int sm_minor,
    const char* variant,
    const char* desc)
{
    std::ofstream f(path);
    f << "dim,section,target_arch,gpu_name,sm,variant,desc,pedantic_ms\n";
    for (int i = 0; i < num_dims; ++i) {
        const auto& r = results[i];
        f << dims[i] << ",FP32_PEDANTIC," << target_arch << "," << gpu_name << ","
          << sm_major << "." << sm_minor << "," << variant << "," << desc << ","
          << r.pedantic_ms << "\n";
    }
}

int main(int argc, char** argv) {
    cudaDeviceProp prop{};
    cudaGetDeviceProperties(&prop, 0);
    TargetArch arch = detect_target_arch_from_device(prop);

    for (int i = 1; i < argc; ++i) {
        if (!parse_target_arch_flag(argv[i], &arch)) {
            std::printf("Invalid arg: %s\n", argv[i]);
            std::printf("Use -h100 or -rtx5070\n");
            return 1;
        }
    }

    std::printf("# GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    std::printf("# Target arch: %s\n", target_arch_name(arch));
    std::printf("# Benchmark: FP32 pedantic only\n\n");

    const std::string out_dir = benchmark_output_dir(arch);
    std::filesystem::create_directories(out_dir);
    const std::string csv_path = benchmark_csv_path(arch);

    cublasHandle_t handle = nullptr;
    cublasCreate(&handle);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    cublasSetStream(handle, stream);

    const char* variant = get_variant_id_fp32_pedantic();
    const char* desc = get_variant_desc_fp32_pedantic();

    std::array<PedanticResult, NUM_DIMS> results{};
    for (int i = 0; i < NUM_DIMS; ++i) {
        benchmark_fp32_pedantic(DIMENSIONS[i], handle, stream, &results[i]);
    }

    print_report(results.data(), variant);

    write_csv_with_arch(csv_path, DIMENSIONS, NUM_DIMS, results.data(),
                        target_arch_name(arch), prop.name, prop.major, prop.minor,
                        variant, desc);

    std::printf("\n# CSV written to %s\n", csv_path.c_str());

    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    return 0;
}