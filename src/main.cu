#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>

// ============ BENCHMARK CONFIG ============
// ACTIVE VARIANTS:
//   1 = naive   (gemm_fp32.cu)              - __ldg + unroll4
//   2 = r2z_128 (gemm_fp32_r2z_128.cu)      - 64x64 tiles, for size 128
//   3 = r2z_256 (gemm_fp32_r2z_256.cu)      - 64x64 tiles, 256 threads, for size 256
//   4 = r2z_512 (gemm_fp32_r2z_512.cu)      - 64x64 tiles, for size 512 (~91%)
//   5 = r2z_1024 (gemm_fp32_r2z_1024.cu)    - 64x64 tiles, for size 1024
//   6 = r2z_2048 (gemm_fp32_r2z_2048.cu)    - 128x128 tiles, for size 2048
//   7 = r2z_4096 (gemm_fp32_r2z_4096.cu)    - 128x128 tiles, for size 4096 (~98%)
//   8 = master  (gemm_fp32_master.cu)       - auto-select by size
//
// SCRATCH VARIANTS (archived in scratch/):
//   r1x, r1y, r2x, r2y, r2z, r2z2, r2z2_small, r3x
//   See scratch/SCRATCH_INDEX.md for details
//
#define FP32_VARIANT 8

constexpr bool TEST_ALL_VARIANTS = true;

constexpr int DIMENSIONS[] = {32, 64, 128, 256, 512, 1024, 2048, 4096};
constexpr int NUM_DIMS = sizeof(DIMENSIONS) / sizeof(DIMENSIONS[0]);

constexpr int WARMUP_ITERATIONS = 10;
constexpr int MEASURE_ITERATIONS = 50;

constexpr const char* CSV_PATH = "results/benchmark_results.csv";
// ==========================================

extern void launch_gemm_bf16(const __nv_bfloat16*, const __nv_bfloat16*, float*,
    int, int, int, float, float, cudaStream_t);
extern const char* get_variant_id_bf16();
extern const char* get_variant_desc_bf16();

#if FP32_VARIANT == 1
extern void launch_gemm_fp32_naive(const float*, const float*, float*,
    int, int, int, float, float, cudaStream_t);
extern const char* get_variant_id_fp32_naive();
extern const char* get_variant_desc_fp32_naive();
#define launch_gemm_fp32 launch_gemm_fp32_naive
#define get_variant_id_fp32 get_variant_id_fp32_naive
#define get_variant_desc_fp32 get_variant_desc_fp32_naive
#elif FP32_VARIANT == 2
extern void launch_gemm_fp32_r2z_128(const float*, const float*, float*,
    int, int, int, float, float, cudaStream_t);
extern const char* get_variant_id_fp32_r2z_128();
extern const char* get_variant_desc_fp32_r2z_128();
#define launch_gemm_fp32 launch_gemm_fp32_r2z_128
#define get_variant_id_fp32 get_variant_id_fp32_r2z_128
#define get_variant_desc_fp32 get_variant_desc_fp32_r2z_128
#elif FP32_VARIANT == 3
extern void launch_gemm_fp32_r2z_256(const float*, const float*, float*,
    int, int, int, float, float, cudaStream_t);
extern const char* get_variant_id_fp32_r2z_256();
extern const char* get_variant_desc_fp32_r2z_256();
#define launch_gemm_fp32 launch_gemm_fp32_r2z_256
#define get_variant_id_fp32 get_variant_id_fp32_r2z_256
#define get_variant_desc_fp32 get_variant_desc_fp32_r2z_256
#elif FP32_VARIANT == 4
extern void launch_gemm_fp32_r2z_512(const float*, const float*, float*,
    int, int, int, float, float, cudaStream_t);
extern const char* get_variant_id_fp32_r2z_512();
extern const char* get_variant_desc_fp32_r2z_512();
#define launch_gemm_fp32 launch_gemm_fp32_r2z_512
#define get_variant_id_fp32 get_variant_id_fp32_r2z_512
#define get_variant_desc_fp32 get_variant_desc_fp32_r2z_512
#elif FP32_VARIANT == 5
extern void launch_gemm_fp32_r2z_1024(const float*, const float*, float*,
    int, int, int, float, float, cudaStream_t);
extern const char* get_variant_id_fp32_r2z_1024();
extern const char* get_variant_desc_fp32_r2z_1024();
#define launch_gemm_fp32 launch_gemm_fp32_r2z_1024
#define get_variant_id_fp32 get_variant_id_fp32_r2z_1024
#define get_variant_desc_fp32 get_variant_desc_fp32_r2z_1024
#elif FP32_VARIANT == 6
extern void launch_gemm_fp32_r2z_2048(const float*, const float*, float*,
    int, int, int, float, float, cudaStream_t);
extern const char* get_variant_id_fp32_r2z_2048();
extern const char* get_variant_desc_fp32_r2z_2048();
#define launch_gemm_fp32 launch_gemm_fp32_r2z_2048
#define get_variant_id_fp32 get_variant_id_fp32_r2z_2048
#define get_variant_desc_fp32 get_variant_desc_fp32_r2z_2048
#elif FP32_VARIANT == 7
extern void launch_gemm_fp32_r2z_4096(const float*, const float*, float*,
    int, int, int, float, float, cudaStream_t);
extern const char* get_variant_id_fp32_r2z_4096();
extern const char* get_variant_desc_fp32_r2z_4096();
#define launch_gemm_fp32 launch_gemm_fp32_r2z_4096
#define get_variant_id_fp32 get_variant_id_fp32_r2z_4096
#define get_variant_desc_fp32 get_variant_desc_fp32_r2z_4096
#elif FP32_VARIANT == 8
extern void launch_gemm_fp32_master(const float*, const float*, float*,
    int, int, int, float, float, cudaStream_t);
extern void launch_gemm_fp32_master_debug(const float*, const float*, float*,
    int, int, int, float, float, cudaStream_t, const char**);
extern const char* get_variant_id_fp32_master();
extern const char* get_variant_desc_fp32_master();
#define launch_gemm_fp32 launch_gemm_fp32_master
#define get_variant_id_fp32 get_variant_id_fp32_master
#define get_variant_desc_fp32 get_variant_desc_fp32_master
#define MASTER_MODE 1
#endif

extern void cublas_gemm_bf16(cublasHandle_t, const __nv_bfloat16*, const __nv_bfloat16*, float*,
    int, int, int, float, float);
extern void cublas_gemm_bf16_tc(cublasHandle_t, const __nv_bfloat16*, const __nv_bfloat16*, float*,
    int, int, int, float, float);

extern void cublas_gemm_fp32_sgemm(cublasHandle_t, const float*, const float*, float*,
    int, int, int, float, float);
extern void cublas_gemm_fp32_cuda(cublasHandle_t, const float*, const float*, float*,
    int, int, int, float, float);
extern void cublas_gemm_fp32_pedantic(cublasHandle_t, const float*, const float*, float*,
    int, int, int, float, float);
extern void cublas_gemm_fp32_tc(cublasHandle_t, const float*, const float*, float*,
    int, int, int, float, float);

static void init_bf16(__nv_bfloat16* ptr, int n, unsigned seed) {
    srand(seed);
    for (int i = 0; i < n; ++i)
        ptr[i] = __float2bfloat16((rand() / float(RAND_MAX)) * 2.0f - 1.0f);
}

static void init_fp32(float* ptr, int n, unsigned seed) {
    srand(seed);
    for (int i = 0; i < n; ++i)
        ptr[i] = (rand() / float(RAND_MAX)) * 2.0f - 1.0f;
}

static float compute_l2_error(const float* a, const float* b, int n) {
    double sum = 0.0, ref = 0.0;
    for (int i = 0; i < n; ++i) {
        double diff = double(a[i]) - double(b[i]);
        sum += diff * diff;
        ref += double(b[i]) * double(b[i]);
    }
    return float(sqrt(sum) / (sqrt(ref) + 1e-10));
}

struct BF16Result {
    float custom_ms;
    float cublas_ms;
    float cublas_tc_ms;
    float l2_error;
};

struct FP32Result {
    float custom_ms;
    float sgemm_ms;
    float cuda_ms;
    float pedantic_ms;
    float tc_ms;
    float l2_error;
};

static void benchmark_bf16(int dim, cublasHandle_t handle, cudaStream_t stream, BF16Result* out) {
    int N = dim;
    size_t bytes_A = N * N * sizeof(__nv_bfloat16);
    size_t bytes_C = N * N * sizeof(float);

    __nv_bfloat16 *d_A, *d_B;
    float *d_C, *d_C_ref;
    cudaMalloc(&d_A, bytes_A);
    cudaMalloc(&d_B, bytes_A);
    cudaMalloc(&d_C, bytes_C);
    cudaMalloc(&d_C_ref, bytes_C);

    __nv_bfloat16* h_A = (__nv_bfloat16*)malloc(bytes_A);
    __nv_bfloat16* h_B = (__nv_bfloat16*)malloc(bytes_A);
    init_bf16(h_A, N * N, 42);
    init_bf16(h_B, N * N, 43);

    cudaMemcpy(d_A, h_A, bytes_A, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes_A, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, bytes_C);
    cudaMemset(d_C_ref, 0, bytes_C);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    for (int i = 0; i < WARMUP_ITERATIONS; ++i)
        launch_gemm_bf16(d_A, d_B, d_C, N, N, N, 1.0f, 0.0f, stream);
    cudaStreamSynchronize(stream);

    cudaEventRecord(start, stream);
    for (int i = 0; i < MEASURE_ITERATIONS; ++i)
        launch_gemm_bf16(d_A, d_B, d_C, N, N, N, 1.0f, 0.0f, stream);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&out->custom_ms, start, stop);
    out->custom_ms /= MEASURE_ITERATIONS;

    auto bench_cublas = [&](auto func, float* out_ms) {
        cudaMemset(d_C_ref, 0, bytes_C);
        for (int i = 0; i < WARMUP_ITERATIONS; ++i)
            func(handle, d_A, d_B, d_C_ref, N, N, N, 1.0f, 0.0f);
        cudaStreamSynchronize(stream);

        cudaEventRecord(start, stream);
        for (int i = 0; i < MEASURE_ITERATIONS; ++i)
            func(handle, d_A, d_B, d_C_ref, N, N, N, 1.0f, 0.0f);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(out_ms, start, stop);
        *out_ms /= MEASURE_ITERATIONS;
    };

    bench_cublas(cublas_gemm_bf16, &out->cublas_ms);
    bench_cublas(cublas_gemm_bf16_tc, &out->cublas_tc_ms);

    float* h_C = (float*)malloc(bytes_C);
    float* h_C_ref = (float*)malloc(bytes_C);
    cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_ref, d_C_ref, bytes_C, cudaMemcpyDeviceToHost);
    out->l2_error = compute_l2_error(h_C, h_C_ref, N * N);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_ref);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
}

#if FP32_VARIANT == 6
static const char* selected_kernel_names[NUM_DIMS];
#endif

static void benchmark_fp32(int dim, cublasHandle_t handle, cudaStream_t stream, FP32Result* out) {
    int N = dim;
    size_t bytes = N * N * sizeof(float);

    float *d_A, *d_B, *d_C, *d_C_ref;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMalloc(&d_C_ref, bytes);

    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    init_fp32(h_A, N * N, 42);
    init_fp32(h_B, N * N, 43);

    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, bytes);
    cudaMemset(d_C_ref, 0, bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

#if FP32_VARIANT == 6
    const char* selected_kernel = nullptr;
    for (int i = 0; i < WARMUP_ITERATIONS; ++i)
        launch_gemm_fp32_master_debug(d_A, d_B, d_C, N, N, N, 1.0f, 0.0f, stream, &selected_kernel);
    cudaStreamSynchronize(stream);

    cudaEventRecord(start, stream);
    for (int i = 0; i < MEASURE_ITERATIONS; ++i)
        launch_gemm_fp32_master_debug(d_A, d_B, d_C, N, N, N, 1.0f, 0.0f, stream, &selected_kernel);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&out->custom_ms, start, stop);
    out->custom_ms /= MEASURE_ITERATIONS;
    
    // Store selected kernel name for this dimension
    int dim_idx = -1;
    for (int i = 0; i < NUM_DIMS; ++i) {
        if (DIMENSIONS[i] == dim) {
            dim_idx = i;
            break;
        }
    }
    if (dim_idx >= 0 && selected_kernel != nullptr) {
        static char kernel_copy[32];
        strncpy(kernel_copy, selected_kernel, 31);
        kernel_copy[31] = '\0';
        selected_kernel_names[dim_idx] = kernel_copy;
    }
#else
    for (int i = 0; i < WARMUP_ITERATIONS; ++i)
        launch_gemm_fp32(d_A, d_B, d_C, N, N, N, 1.0f, 0.0f, stream);
    cudaStreamSynchronize(stream);

    cudaEventRecord(start, stream);
    for (int i = 0; i < MEASURE_ITERATIONS; ++i)
        launch_gemm_fp32(d_A, d_B, d_C, N, N, N, 1.0f, 0.0f, stream);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&out->custom_ms, start, stop);
    out->custom_ms /= MEASURE_ITERATIONS;
#endif

    auto bench_cublas = [&](auto func, float* out_ms) {
        cudaMemset(d_C_ref, 0, bytes);
        for (int i = 0; i < WARMUP_ITERATIONS; ++i)
            func(handle, d_A, d_B, d_C_ref, N, N, N, 1.0f, 0.0f);
        cudaStreamSynchronize(stream);

        cudaEventRecord(start, stream);
        for (int i = 0; i < MEASURE_ITERATIONS; ++i)
            func(handle, d_A, d_B, d_C_ref, N, N, N, 1.0f, 0.0f);
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(out_ms, start, stop);
        *out_ms /= MEASURE_ITERATIONS;
    };

    bench_cublas(cublas_gemm_fp32_sgemm, &out->sgemm_ms);
    bench_cublas(cublas_gemm_fp32_cuda, &out->cuda_ms);
    bench_cublas(cublas_gemm_fp32_pedantic, &out->pedantic_ms);
    bench_cublas(cublas_gemm_fp32_tc, &out->tc_ms);

    float* h_C = (float*)malloc(bytes);
    float* h_C_ref = (float*)malloc(bytes);
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_ref, d_C_ref, bytes, cudaMemcpyDeviceToHost);
    out->l2_error = compute_l2_error(h_C, h_C_ref, N * N);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_ref);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
}

static void write_csv(const std::string& path,
                      const int* dims, int num_dims,
                      const BF16Result* bf16_results,
                      const FP32Result* fp32_results,
                      const char* variant_bf16, const char* desc_bf16,
                      const char* variant_fp32, const char* desc_fp32) {
    std::ofstream f(path);
    f << "dim,dtype,variant,desc,custom_ms,cublas_32f_ms,cublas_tc_ms,"
       << "sgemm_ms,cuda_ms,pedantic_ms,tc_ms,l2_error\n";
    for (int i = 0; i < num_dims; ++i) {
        const auto& r = bf16_results[i];
        f << dims[i] << ",BF16," << variant_bf16 << "," << desc_bf16 << ","
          << r.custom_ms << "," << r.cublas_ms << "," << r.cublas_tc_ms << ",,,,"
          << r.l2_error << "\n";
    }
    for (int i = 0; i < num_dims; ++i) {
        const auto& r = fp32_results[i];
        f << dims[i] << ",FP32," << variant_fp32 << "," << desc_fp32 << ","
          << r.custom_ms << ",,,"
          << r.sgemm_ms << "," << r.cuda_ms << "," << r.pedantic_ms << "," << r.tc_ms << ","
          << r.l2_error << "\n";
    }
}

int main(int argc, char** argv) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("# GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("# TEST_ALL_VARIANTS: %s\n", TEST_ALL_VARIANTS ? "true" : "false");
#if FP32_VARIANT == 1
    printf("# FP32_VARIANT: naive (baseline)\n\n");
#elif FP32_VARIANT == 2
    printf("# FP32_VARIANT: r2z_128 (64x64 tiles, for size 128)\n\n");
#elif FP32_VARIANT == 3
    printf("# FP32_VARIANT: r2z_256 (64x64 tiles, 256 threads, for size 256)\n\n");
#elif FP32_VARIANT == 4
    printf("# FP32_VARIANT: r2z_512 (64x64 tiles, ~91%% at 512)\n\n");
#elif FP32_VARIANT == 5
    printf("# FP32_VARIANT: r2z_1024 (64x64 tiles, for size 1024)\n\n");
#elif FP32_VARIANT == 6
    printf("# FP32_VARIANT: r2z_2048 (128x128 tiles, for size 2048)\n\n");
#elif FP32_VARIANT == 7
    printf("# FP32_VARIANT: r2z_4096 (128x128 tiles, ~98%% at 4096)\n\n");
#elif FP32_VARIANT == 8
    printf("# FP32_VARIANT: master (auto-select by size)\n\n");
#else
    printf("# FP32_VARIANT: unknown (check config)\n\n");
#endif

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasSetStream(handle, stream);

    const char* variant_bf16 = get_variant_id_bf16();
    const char* desc_bf16 = get_variant_desc_bf16();
    const char* variant_fp32 = get_variant_id_fp32();
    const char* desc_fp32 = get_variant_desc_fp32();

    BF16Result bf16_results[NUM_DIMS];
    FP32Result fp32_results[NUM_DIMS];

    printf("=== BF16 ===\n");
    printf("%-6s %-4s %-8s %10s %10s %10s %7s %10s\n",
        "Dim", "Ver", "Desc", "Custom(ms)", "CuBLAS(ms)", "CuBLAS-TC(ms)", "Ratio%", "L2_Error");
    printf("%s\n", std::string(78, '-').c_str());

    for (int d = 0; d < NUM_DIMS; ++d) {
        int dim = DIMENSIONS[d];
        benchmark_bf16(dim, handle, stream, &bf16_results[d]);
        const auto& r = bf16_results[d];
        float ratio = (r.cublas_ms / r.custom_ms) * 100.0f;
        printf("%-6d %-4s %-8s %10.4f %10.4f %10.4f %6.1f%% %10.2e\n",
            dim, variant_bf16, desc_bf16, r.custom_ms, r.cublas_ms, r.cublas_tc_ms, ratio, r.l2_error);
    }

    printf("\n=== FP32 ===\n");
#if FP32_VARIANT == 6
    printf("%-6s %-8s %-10s %10s %10s %10s %10s %10s %6s %6s %6s %6s %10s\n",
        "Dim", "Selected", "Kernel", "Custom", "Sgemm", "CUDA", "Pedant", "TC(ms)",
        "Sg%", "CU%", "Pd%", "TC%", "L2_Err");
    printf("%s\n", std::string(100, '-').c_str());

    for (int d = 0; d < NUM_DIMS; ++d) {
        int dim = DIMENSIONS[d];
        benchmark_fp32(dim, handle, stream, &fp32_results[d]);
        const auto& r = fp32_results[d];

        float r_sgemm = (r.sgemm_ms / r.custom_ms) * 100.0f;
        float r_cuda = (r.cuda_ms / r.custom_ms) * 100.0f;
        float r_pedantic = (r.pedantic_ms / r.custom_ms) * 100.0f;
        float r_tc = (r.tc_ms / r.custom_ms) * 100.0f;

        const char* sel = selected_kernel_names[d] ? selected_kernel_names[d] : "unknown";
        printf("%-6d %-8s %-10s %10.4f %10.4f %10.4f %10.4f %10.4f %5.1f%% %5.1f%% %5.1f%% %5.1f%% %10.2e\n",
            dim, sel, desc_fp32, r.custom_ms, r.sgemm_ms, r.cuda_ms, r.pedantic_ms, r.tc_ms,
            r_sgemm, r_cuda, r_pedantic, r_tc, r.l2_error);
    }
#else
    printf("%-6s %-4s %-8s %10s %10s %10s %10s %10s %6s %6s %6s %6s %10s\n",
        "Dim", "Ver", "Desc", "Custom(ms)", "Sgemm(ms)", "CUDA(ms)", "Pedant(ms)", "TC(ms)",
        "Sg%", "CU%", "Pd%", "TC%", "L2_Err");
    printf("%s\n", std::string(115, '-').c_str());

    for (int d = 0; d < NUM_DIMS; ++d) {
        int dim = DIMENSIONS[d];
        benchmark_fp32(dim, handle, stream, &fp32_results[d]);
        const auto& r = fp32_results[d];

        float r_sgemm = (r.sgemm_ms / r.custom_ms) * 100.0f;
        float r_cuda = (r.cuda_ms / r.custom_ms) * 100.0f;
        float r_pedantic = (r.pedantic_ms / r.custom_ms) * 100.0f;
        float r_tc = (r.tc_ms / r.custom_ms) * 100.0f;

        printf("%-6d %-4s %-8s %10.4f %10.4f %10.4f %10.4f %10.4f %5.1f%% %5.1f%% %5.1f%% %5.1f%% %10.2e\n",
            dim, variant_fp32, desc_fp32, r.custom_ms, r.sgemm_ms, r.cuda_ms, r.pedantic_ms, r.tc_ms,
            r_sgemm, r_cuda, r_pedantic, r_tc, r.l2_error);
    }
#endif

    write_csv(CSV_PATH, DIMENSIONS, NUM_DIMS, bf16_results, fp32_results,
              variant_bf16, desc_bf16, variant_fp32, desc_fp32);
    printf("\n# CSV written to %s\n", CSV_PATH);

    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    return 0;
}
