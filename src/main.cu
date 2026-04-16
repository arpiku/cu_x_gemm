#include <array>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>

#include "helpers.h"

// Kernel entry points.
extern void launch_gemm_bf16(const __nv_bfloat16*, const __nv_bfloat16*, float*,
    int, int, int, float, float, cudaStream_t, TargetArch);
extern const char* get_variant_id_bf16();
extern const char* get_variant_desc_bf16();

extern void launch_gemm_fp32_master(const float*, const float*, float*,
    int, int, int, float, float, cudaStream_t, TargetArch);
extern const char* get_variant_id_fp32_master();
extern const char* get_variant_desc_fp32_master();

extern void launch_gemm_fp32_r2z_tc1(const float*, const float*, float*,
    int, int, int, float, float, cudaStream_t, TargetArch);
extern const char* get_variant_id_fp32_r2z_tc1();
extern const char* get_variant_desc_fp32_r2z_tc1();

extern void cublas_gemm_bf16(cublasHandle_t, const __nv_bfloat16*, const __nv_bfloat16*, float*,
    int, int, int, float, float);
extern void cublas_gemm_bf16_tc(cublasHandle_t, const __nv_bfloat16*, const __nv_bfloat16*, float*,
    int, int, int, float, float);

extern void cublas_gemm_fp32_pedantic(cublasHandle_t, const float*, const float*, float*,
    int, int, int, float, float);
extern void cublas_gemm_fp32_tc(cublasHandle_t, const float*, const float*, float*,
    int, int, int, float, float);

template <typename LaunchFn>
static float benchmark_ms(cudaStream_t stream, LaunchFn&& launch_fn) {
    return time_launch_ms(stream, WARMUP_ITERATIONS, MEASURE_ITERATIONS, launch_fn);
}

static void benchmark_bf16(int dim, cublasHandle_t handle, cudaStream_t stream,
                           TargetArch arch, BF16Result* out) {
    const int elements = dim * dim;
    const size_t bytes_a = static_cast<size_t>(elements) * sizeof(__nv_bfloat16);
    const size_t bytes_c = static_cast<size_t>(elements) * sizeof(float);

    __nv_bfloat16* d_A = nullptr;
    __nv_bfloat16* d_B = nullptr;
    float* d_C = nullptr;
    float* d_C_ref = nullptr;
    cudaMalloc(&d_A, bytes_a);
    cudaMalloc(&d_B, bytes_a);
    cudaMalloc(&d_C, bytes_c);
    cudaMalloc(&d_C_ref, bytes_c);

    __nv_bfloat16* h_A = static_cast<__nv_bfloat16*>(std::malloc(bytes_a));
    __nv_bfloat16* h_B = static_cast<__nv_bfloat16*>(std::malloc(bytes_a));
    init_bf16(h_A, elements, 42);
    init_bf16(h_B, elements, 43);

    cudaMemcpy(d_A, h_A, bytes_a, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_a, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, bytes_c);
    cudaMemset(d_C_ref, 0, bytes_c);

    out->custom_ms = benchmark_ms(stream, [&] {
        launch_gemm_bf16(d_A, d_B, d_C, dim, dim, dim, 1.0f, 0.0f, stream, arch);
    });

    out->cublas_ms = benchmark_ms(stream, [&] {
        cublas_gemm_bf16(handle, d_A, d_B, d_C_ref, dim, dim, dim, 1.0f, 0.0f);
    });

    cudaMemset(d_C_ref, 0, bytes_c);
    out->cublas_tc_ms = benchmark_ms(stream, [&] {
        cublas_gemm_bf16_tc(handle, d_A, d_B, d_C_ref, dim, dim, dim, 1.0f, 0.0f);
    });

    float* h_C = static_cast<float*>(std::malloc(bytes_c));
    float* h_C_ref = static_cast<float*>(std::malloc(bytes_c));
    cudaMemcpy(h_C, d_C, bytes_c, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_ref, d_C_ref, bytes_c, cudaMemcpyDeviceToHost);
    out->l2_error = compute_l2_error(h_C, h_C_ref, elements);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C_ref);
    std::free(h_A);
    std::free(h_B);
    std::free(h_C);
    std::free(h_C_ref);
}

static void benchmark_fp32(int dim, cublasHandle_t handle, cudaStream_t stream,
                           TargetArch arch, FP32Result* out) {
    const int elements = dim * dim;
    const size_t bytes = static_cast<size_t>(elements) * sizeof(float);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    float* d_C_ref = nullptr;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMalloc(&d_C_ref, bytes);

    float* h_A = static_cast<float*>(std::malloc(bytes));
    float* h_B = static_cast<float*>(std::malloc(bytes));
    init_fp32(h_A, elements, 42);
    init_fp32(h_B, elements, 43);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, bytes);
    cudaMemset(d_C_ref, 0, bytes);

    out->custom_ms = benchmark_ms(stream, [&] {
        launch_gemm_fp32_master(d_A, d_B, d_C, dim, dim, dim, 1.0f, 0.0f, stream, arch);
    });

    out->pedantic_ms = benchmark_ms(stream, [&] {
        cublas_gemm_fp32_pedantic(handle, d_A, d_B, d_C_ref, dim, dim, dim, 1.0f, 0.0f);
    });

    float* h_C = static_cast<float*>(std::malloc(bytes));
    float* h_C_ref = static_cast<float*>(std::malloc(bytes));
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_ref, d_C_ref, bytes, cudaMemcpyDeviceToHost);
    out->l2_error = compute_l2_error(h_C, h_C_ref, elements);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C_ref);
    std::free(h_A);
    std::free(h_B);
    std::free(h_C);
    std::free(h_C_ref);
}

static void benchmark_fp32_tc(int dim, cublasHandle_t handle, cudaStream_t stream,
                              TargetArch arch, FP32TCResult* out) {
    const int elements = dim * dim;
    const size_t bytes = static_cast<size_t>(elements) * sizeof(float);

    float* d_A = nullptr;
    float* d_B = nullptr;
    float* d_C = nullptr;
    float* d_C_ref = nullptr;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMalloc(&d_C_ref, bytes);

    float* h_A = static_cast<float*>(std::malloc(bytes));
    float* h_B = static_cast<float*>(std::malloc(bytes));
    init_fp32(h_A, elements, 42);
    init_fp32(h_B, elements, 43);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, bytes);
    cudaMemset(d_C_ref, 0, bytes);

    out->custom_ms = benchmark_ms(stream, [&] {
        launch_gemm_fp32_r2z_tc1(d_A, d_B, d_C, dim, dim, dim, 1.0f, 0.0f, stream, arch);
    });

    cublas_gemm_fp32_pedantic(handle, d_A, d_B, d_C_ref, dim, dim, dim, 1.0f, 0.0f);
    cudaStreamSynchronize(stream);

    float* h_C = static_cast<float*>(std::malloc(bytes));
    float* h_C_ref = static_cast<float*>(std::malloc(bytes));
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_ref, d_C_ref, bytes, cudaMemcpyDeviceToHost);
    out->l2_error = compute_l2_error(h_C, h_C_ref, elements);

    cudaMemset(d_C_ref, 0, bytes);
    out->tc_ms = benchmark_ms(stream, [&] {
        cublas_gemm_fp32_tc(handle, d_A, d_B, d_C_ref, dim, dim, dim, 1.0f, 0.0f);
    });

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C_ref);
    std::free(h_A);
    std::free(h_B);
    std::free(h_C);
    std::free(h_C_ref);
}

static void run_bf16_benchmarks(cublasHandle_t handle, cudaStream_t stream,
                                TargetArch arch, BF16Result* results) {
    for (int i = 0; i < NUM_DIMS; ++i) {
        benchmark_bf16(DIMENSIONS[i], handle, stream, arch, &results[i]);
    }
}

static void run_fp32_benchmarks(cublasHandle_t handle, cudaStream_t stream,
                                TargetArch arch, FP32Result* results) {
    for (int i = 0; i < NUM_DIMS; ++i) {
        benchmark_fp32(DIMENSIONS[i], handle, stream, arch, &results[i]);
    }
}

static void run_fp32_tc_benchmarks(cublasHandle_t handle, cudaStream_t stream,
                                   TargetArch arch, FP32TCResult* results) {
    for (int i = 0; i < NUM_DIMS; ++i) {
        benchmark_fp32_tc(DIMENSIONS[i], handle, stream, arch, &results[i]);
    }
}

static void print_bf16_report(const char* variant_bf16, const BF16Result* results) {
    std::printf("=== BF16 ===\n");
    std::printf("%-6s %-12s %10s %10s %12s %7s %10s\n",
                "Dim", "Selected", "Custom(ms)", "CuBLAS(ms)", "CuBLAS-TC(ms)",
                "Ratio%", "L2_Error");
    std::printf("%s\n", std::string(78, '-').c_str());

    for (int i = 0; i < NUM_DIMS; ++i) {
        const auto& r = results[i];
        const float ratio = (r.cublas_ms / r.custom_ms) * 100.0f;
        std::printf("%-6d %-12s %10.4f %10.4f %12.4f %6.1f%% %10.2e\n",
                    DIMENSIONS[i], variant_bf16, r.custom_ms, r.cublas_ms,
                    r.cublas_tc_ms, ratio, r.l2_error);
    }
}

static void print_fp32_report(const FP32Result* results) {
    std::printf("\n=== FP32 (CU Cores) ===\n");
    std::printf("# Primary reference: cuBLAS pedantic (pure FP32, CUDA cores)\n");
    std::printf("%-6s %-14s %10s %12s %8s %10s\n",
                "Dim", "Variant", "Custom(ms)", "Pedantic(ms)", "Ratio%", "L2_Err");
    std::printf("%s\n", std::string(76, '-').c_str());

    for (int i = 0; i < NUM_DIMS; ++i) {
        const auto& r = results[i];
        const float ratio = (r.pedantic_ms / r.custom_ms) * 100.0f;
        std::printf("%-6d %-14s %10.4f %12.4f %7.1f%% %10.2e\n",
                    DIMENSIONS[i], get_variant_id_fp32_master(),
                    r.custom_ms, r.pedantic_ms, ratio, r.l2_error);
    }
}

static void print_fp32_tc_report(const FP32TCResult* results) {
    std::printf("\n=== FP32 (TF32 / Tensor Cores) ===\n");
    std::printf("# Custom kernel: TF32 WMMA/PTX (r2z_tc1); reference: cuBLAS COMPUTE_32F_FAST_TF32\n");
    std::printf("%-6s %-14s %10s %10s %8s %10s\n",
                "Dim", "Variant", "Custom(ms)", "TF32(ms)", "Ratio%", "L2_Err");
    std::printf("%s\n", std::string(76, '-').c_str());

    for (int i = 0; i < NUM_DIMS; ++i) {
        const auto& r = results[i];
        const float ratio = (r.tc_ms / r.custom_ms) * 100.0f;
        std::printf("%-6d %-14s %10.4f %10.4f %7.1f%% %10.2e\n",
                    DIMENSIONS[i], get_variant_id_fp32_r2z_tc1(),
                    r.custom_ms, r.tc_ms, ratio, r.l2_error);
    }
}

static void write_csv_with_arch(
    const std::string& path,
    const int* dims,
    int num_dims,
    const BF16Result* bf16_results,
    const FP32Result* fp32_results,
    const FP32TCResult* fp32_tc_results,
    const char* target_arch,
    const char* gpu_name,
    int sm_major,
    int sm_minor,
    const char* variant_bf16,
    const char* desc_bf16,
    const char* variant_fp32,
    const char* desc_fp32,
    const char* variant_tc,
    const char* desc_tc)
{
    std::ofstream f(path);
    f << "dim,section,target_arch,gpu_name,sm,variant,desc,custom_ms,reference_ms,l2_error\n";

    for (int i = 0; i < num_dims; ++i) {
        const auto& r = bf16_results[i];
        f << dims[i] << ",BF16," << target_arch << "," << gpu_name << ","
          << sm_major << "." << sm_minor << "," << variant_bf16 << "," << desc_bf16 << ","
          << r.custom_ms << "," << r.cublas_ms << "," << r.l2_error << "\n";
    }

    for (int i = 0; i < num_dims; ++i) {
        const auto& r = fp32_results[i];
        f << dims[i] << ",FP32_PEDANTIC," << target_arch << "," << gpu_name << ","
          << sm_major << "." << sm_minor << "," << variant_fp32 << "," << desc_fp32 << ","
          << r.custom_ms << "," << r.pedantic_ms << "," << r.l2_error << "\n";
    }

    for (int i = 0; i < num_dims; ++i) {
        const auto& r = fp32_tc_results[i];
        f << dims[i] << ",FP32_TF32," << target_arch << "," << gpu_name << ","
          << sm_major << "." << sm_minor << "," << variant_tc << "," << desc_tc << ","
          << r.custom_ms << "," << r.tc_ms << "," << r.l2_error << "\n";
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
    std::printf("# Comparison variants: BF16 tensor core, FP32 pedantic, FP32 TF32 tensor core\n\n");

    const std::string out_dir = benchmark_output_dir(arch);
    std::filesystem::create_directories(out_dir);
    const std::string csv_path = benchmark_csv_path(arch);

    cublasHandle_t handle = nullptr;
    cublasCreate(&handle);

    cudaStream_t stream = nullptr;
    cudaStreamCreate(&stream);
    cublasSetStream(handle, stream);

    const char* variant_bf16 = get_variant_id_bf16();
    const char* desc_bf16 = get_variant_desc_bf16();
    const char* variant_fp32 = get_variant_id_fp32_master();
    const char* desc_fp32 = get_variant_desc_fp32_master();
    const char* variant_tc = get_variant_id_fp32_r2z_tc1();
    const char* desc_tc = get_variant_desc_fp32_r2z_tc1();

    std::array<BF16Result, NUM_DIMS> bf16_results{};
    std::array<FP32Result, NUM_DIMS> fp32_results{};
    std::array<FP32TCResult, NUM_DIMS> fp32_tc_results{};

    run_bf16_benchmarks(handle, stream, arch, bf16_results.data());
    run_fp32_benchmarks(handle, stream, arch, fp32_results.data());
    run_fp32_tc_benchmarks(handle, stream, arch, fp32_tc_results.data());

    print_bf16_report(variant_bf16, bf16_results.data());
    print_fp32_report(fp32_results.data());
    print_fp32_tc_report(fp32_tc_results.data());

    write_csv_with_arch(csv_path, DIMENSIONS, NUM_DIMS,
                        bf16_results.data(), fp32_results.data(), fp32_tc_results.data(),
                        target_arch_name(arch), prop.name, prop.major, prop.minor,
                        variant_bf16, desc_bf16, variant_fp32, desc_fp32, variant_tc, desc_tc);
    std::printf("\n# CSV written to %s\n", csv_path.c_str());

    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    return 0;
}