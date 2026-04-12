#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>

extern void launch_test_kernel(
    const float* d_A,
    const float* d_B,
    float* d_C,
    int M, int N, int K,
    float alpha, float beta,
    cudaStream_t stream,
    int* d_bank_access,
    int* d_bank_conflict);

extern const char* get_variant_desc();

void init_matrix(float* ptr, int n, unsigned seed) {
    srand(seed);
    for (int i = 0; i < n; ++i)
        ptr[i] = (rand() / float(RAND_MAX)) * 2.0f - 1.0f;
}

float compute_l2_error(const float* a, const float* b, int n) {
    double sum = 0.0, ref = 0.0;
    for (int i = 0; i < n; ++i) {
        double diff = double(a[i]) - double(b[i]);
        sum += diff * diff;
        ref += double(b[i]) * double(b[i]);
    }
    return float(sqrt(sum) / (sqrt(ref) + 1e-10));
}

int main(int argc, char** argv) {
    if (argc < 2) {
        printf("Usage: %s <matrix_size>\n", argv[0]);
        return 1;
    }
    
    int dim = atoi(argv[1]);
    int N = dim;
    
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    printf("# GPU: %s (SM %d.%d)\n", prop.name, prop.major, prop.minor);
    printf("# Matrix size: %d x %d\n", N, N);
    printf("# Variant: %s\n\n", get_variant_desc());
    
    size_t bytes = N * N * sizeof(float);
    
    float *d_A, *d_B, *d_C, *d_C_ref;
    int *d_bank_access, *d_bank_conflict;
    
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);
    cudaMalloc(&d_C_ref, bytes);
    cudaMalloc(&d_bank_access, 32 * sizeof(int));
    cudaMalloc(&d_bank_conflict, 32 * sizeof(int));
    
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    init_matrix(h_A, N * N, 42);
    init_matrix(h_B, N * N, 43);
    
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);
    cudaMemset(d_C, 0, bytes);
    cudaMemset(d_C_ref, 0, bytes);
    cudaMemset(d_bank_access, 0, 32 * sizeof(int));
    cudaMemset(d_bank_conflict, 0, 32 * sizeof(int));
    
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    
    const int WARMUP = 5;
    const int ITERATIONS = 20;
    
    for (int i = 0; i < WARMUP; ++i) {
        launch_test_kernel(d_A, d_B, d_C, N, N, N, 1.0f, 0.0f, stream,
                          d_bank_access, d_bank_conflict);
    }
    cudaStreamSynchronize(stream);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    cudaEventRecord(start, stream);
    for (int i = 0; i < ITERATIONS; ++i) {
        launch_test_kernel(d_A, d_B, d_C, N, N, N, 1.0f, 0.0f, stream,
                          d_bank_access, d_bank_conflict);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= ITERATIONS;
    
    printf("=== Timing Results ===\n");
    printf("Average kernel time: %.4f ms\n", ms);
    
    int h_bank_access[32];
    int h_bank_conflict[32];
    cudaMemcpy(h_bank_access, d_bank_access, 32 * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_bank_conflict, d_bank_conflict, 32 * sizeof(int), cudaMemcpyDeviceToHost);
    
    long long total_accesses = 0;
    int max_access = 0, min_access = INT_MAX;
    for (int i = 0; i < 32; ++i) {
        total_accesses += h_bank_access[i];
        if (h_bank_access[i] > max_access) max_access = h_bank_access[i];
        if (h_bank_access[i] > 0 && h_bank_access[i] < min_access) min_access = h_bank_access[i];
    }
    
    printf("\n=== Bank Access Distribution ===\n");
    printf("BankID  Accesses  Bar\n");
    printf("%s\n", std::string(30, '-').c_str());
    for (int i = 0; i < 32; ++i) {
        int bars = (h_bank_access[i] * 50) / (max_access + 1);
        printf("%3d    %8d  %s\n", i, h_bank_access[i], std::string(bars, '*').c_str());
    }
    
    printf("\n=== Summary ===\n");
    printf("Total accesses: %lld\n", total_accesses);
    printf("Max bank accesses: %d\n", max_access);
    printf("Min bank accesses: %d\n", min_access);
    printf("Max/Min ratio: %.2f\n", (float)max_access / (min_access + 1));
    
    float imbalance = (float)(max_access - min_access) / (total_accesses / 32) * 100.0f;
    printf("Imbalance: %.1f%%\n", imbalance);
    
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaStreamDestroy(stream);
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    cudaFree(d_C_ref);
    cudaFree(d_bank_access);
    cudaFree(d_bank_conflict);
    
    free(h_A);
    free(h_B);
    
    return 0;
}