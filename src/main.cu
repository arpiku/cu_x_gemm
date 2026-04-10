#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <string>

// ============ BENCHMARK CONFIG ============
// SELECT FP32 VARIANT: 1 = r1a (32x32), 2 = r1b (64x64), 3 = r1c (128x64), 4 = r1d (128x128)
#define FP32_VARIANT 2

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
extern void launch_gemm_fp32_r1a(const float*, const float*, float*,
    int, int, int, float, float, cudaStream_t);
extern const char* get_variant_id_fp32_r1a();
extern const char* get_variant_desc_fp32_r1a();
#define launch_gemm_fp32 launch_gemm_fp32_r1a
#define get_variant_id_fp32 get_variant_id_fp32_r1a
#define get_variant_desc_fp32 get_variant_desc_fp32_r1a
#elif FP32_VARIANT == 2
extern void launch_gemm_fp32_r1b(const float*, const float*, float*,
    int, int, int, float, float, cudaStream_t);
extern const char* get_variant_id_fp32_r1b();
extern const char* get_variant_desc_fp32_r1b();
#define launch_gemm_fp32 launch_gemm_fp32_r1b
#define get_variant_id_fp32 get_variant_id_fp32_r1b
#define get_variant_desc_fp32 get_variant_desc_fp32_r1b
#elif FP32_VARIANT == 3
extern void launch_gemm_fp32_r1c(const float*, const float*, float*,
    int, int, int, float, float, cudaStream_t);
extern const char* get_variant_id_fp32_r1c();
extern const char* get_variant_desc_fp32_r1c();
#define launch_gemm_fp32 launch_gemm_fp32_r1c
#define get_variant_id_fp32 get_variant_id_fp32_r1c
#define get_variant_desc_fp32 get_variant_desc_fp32_r1c
#else
extern void launch_gemm_fp32_r1d(const float*, const float*, float*,
    int, int, int, float, float, cudaStream_t);
extern const char* get_variant_id_fp32_r1d();
extern const char* get_variant_desc_fp32_r1d();
#define launch_gemm_fp32 launch_gemm_fp32_r1d
#define get_variant_id_fp32 get_variant_id_fp32_r1d
#define get_variant_desc_fp32 get_variant_desc_fp32_r1d
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
    printf("# FP32_VARIANT: r1a (32x32 tile)\n\n");
#elif FP32_VARIANT == 2
    printf("# FP32_VARIANT: r1b (64x64 tile)\n\n");
#elif FP32_VARIANT == 3
    printf("# FP32_VARIANT: r1c (128x64 tile)\n\n");
#else
    printf("# FP32_VARIANT: r1d (128x128 tile)\n\n");
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

    write_csv(CSV_PATH, DIMENSIONS, NUM_DIMS, bf16_results, fp32_results,
              variant_bf16, desc_bf16, variant_fp32, desc_fp32);
    printf("\n# CSV written to %s\n", CSV_PATH);

    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    return 0;
}
