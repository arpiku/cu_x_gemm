#include <cublas_v2.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

// ============ BENCHMARK CONFIG ============
constexpr bool TEST_ALL_VARIANTS = true;

constexpr int DIMENSIONS[] = {32, 64, 128, 256, 512, 1024, 2048, 4096};
constexpr int NUM_DIMS = sizeof(DIMENSIONS) / sizeof(DIMENSIONS[0]);

constexpr int WARMUP_ITERATIONS = 10;
constexpr int MEASURE_ITERATIONS = 50;
// ==========================================

extern void launch_gemm_bf16(const __nv_bfloat16*, const __nv_bfloat16*, float*,
    int, int, int, float, float, cudaStream_t);
extern const char* get_variant_id_bf16();

extern void launch_gemm_fp32(const float*, const float*, float*,
    int, int, int, float, float, cudaStream_t);
extern const char* get_variant_id_fp32();

extern void cublas_gemm_bf16(cublasHandle_t, const __nv_bfloat16*, const __nv_bfloat16*, float*,
    int, int, int, float, float);
extern void cublas_gemm_fp32(cublasHandle_t, const float*, const float*, float*,
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

static void benchmark_bf16(int dim, cublasHandle_t handle, cudaStream_t stream,
    float* out_custom_ms, float* out_cublas_ms, float* out_l2_error) {
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

    for (int i = 0; i < WARMUP_ITERATIONS; ++i)
        launch_gemm_bf16(d_A, d_B, d_C, N, N, N, 1.0f, 0.0f, stream);
    cudaStreamSynchronize(stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    for (int i = 0; i < MEASURE_ITERATIONS; ++i)
        launch_gemm_bf16(d_A, d_B, d_C, N, N, N, 1.0f, 0.0f, stream);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(out_custom_ms, start, stop);
    *out_custom_ms /= MEASURE_ITERATIONS;

    for (int i = 0; i < WARMUP_ITERATIONS; ++i)
        cublas_gemm_bf16(handle, d_A, d_B, d_C_ref, N, N, N, 1.0f, 0.0f);
    cudaStreamSynchronize(stream);

    cudaEventRecord(start, stream);
    for (int i = 0; i < MEASURE_ITERATIONS; ++i)
        cublas_gemm_bf16(handle, d_A, d_B, d_C_ref, N, N, N, 1.0f, 0.0f);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(out_cublas_ms, start, stop);
    *out_cublas_ms /= MEASURE_ITERATIONS;

    float* h_C = (float*)malloc(bytes_C);
    float* h_C_ref = (float*)malloc(bytes_C);
    cudaMemcpy(h_C, d_C, bytes_C, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_ref, d_C_ref, bytes_C, cudaMemcpyDeviceToHost);

    *out_l2_error = compute_l2_error(h_C, h_C_ref, N * N);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_ref);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
}

static void benchmark_fp32(int dim, cublasHandle_t handle, cudaStream_t stream,
    float* out_custom_ms, float* out_cublas_ms, float* out_l2_error) {
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

    for (int i = 0; i < WARMUP_ITERATIONS; ++i)
        launch_gemm_fp32(d_A, d_B, d_C, N, N, N, 1.0f, 0.0f, stream);
    cudaStreamSynchronize(stream);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    cudaEventRecord(start, stream);
    for (int i = 0; i < MEASURE_ITERATIONS; ++i)
        launch_gemm_fp32(d_A, d_B, d_C, N, N, N, 1.0f, 0.0f, stream);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(out_custom_ms, start, stop);
    *out_custom_ms /= MEASURE_ITERATIONS;

    for (int i = 0; i < WARMUP_ITERATIONS; ++i)
        cublas_gemm_fp32(handle, d_A, d_B, d_C_ref, N, N, N, 1.0f, 0.0f);
    cudaStreamSynchronize(stream);

    cudaEventRecord(start, stream);
    for (int i = 0; i < MEASURE_ITERATIONS; ++i)
        cublas_gemm_fp32(handle, d_A, d_B, d_C_ref, N, N, N, 1.0f, 0.0f);
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(out_cublas_ms, start, stop);
    *out_cublas_ms /= MEASURE_ITERATIONS;

    float* h_C = (float*)malloc(bytes);
    float* h_C_ref = (float*)malloc(bytes);
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(h_C_ref, d_C_ref, bytes, cudaMemcpyDeviceToHost);

    *out_l2_error = compute_l2_error(h_C, h_C_ref, N * N);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_A); cudaFree(d_B); cudaFree(d_C); cudaFree(d_C_ref);
    free(h_A); free(h_B); free(h_C); free(h_C_ref);
}

int main(int argc, char** argv) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("# GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("# TEST_ALL_VARIANTS: %s\n\n", TEST_ALL_VARIANTS ? "true" : "false");

    cublasHandle_t handle;
    cublasCreate(&handle);

    cudaStream_t stream;
    cudaStreamCreate(&stream);
    cublasSetStream(handle, stream);

    printf("%-6s %-6s %-12s %10s %10s %7s %10s\n",
        "Dim", "Type", "Variant", "Custom(ms)", "CuBLAS(ms)", "Ratio%", "L2_Error");
    printf("%s\n", std::string(70, '-').c_str());

    const char* variant_bf16 = get_variant_id_bf16();
    const char* variant_fp32 = get_variant_id_fp32();

    for (int d = 0; d < NUM_DIMS; ++d) {
        int dim = DIMENSIONS[d];
        float custom_ms, cublas_ms, l2_error;

        benchmark_bf16(dim, handle, stream, &custom_ms, &cublas_ms, &l2_error);
        float ratio = (cublas_ms / custom_ms) * 100.0f;
        printf("%-6d %-6s %-12s %10.4f %10.4f %6.1f%% %10.2e\n",
            dim, "BF16", variant_bf16, custom_ms, cublas_ms, ratio, l2_error);
    }

    printf("\n");

    for (int d = 0; d < NUM_DIMS; ++d) {
        int dim = DIMENSIONS[d];
        float custom_ms, cublas_ms, l2_error;

        benchmark_fp32(dim, handle, stream, &custom_ms, &cublas_ms, &l2_error);
        float ratio = (cublas_ms / custom_ms) * 100.0f;
        printf("%-6d %-6s %-12s %10.4f %10.4f %6.1f%% %10.2e\n",
            dim, "FP32", variant_fp32, custom_ms, cublas_ms, ratio, l2_error);
    }

    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    return 0;
}
