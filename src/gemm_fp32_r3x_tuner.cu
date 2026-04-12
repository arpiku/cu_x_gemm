#include <cuda_runtime.h>
#include <cuda_pipeline.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>
#include <algorithm>

// CSV parsing for existing results
struct CudaBLASResult {
    int dim;
    float sgemm_ms;
    bool valid = false;
};

// Read cuBLAS times from existing CSV
std::vector<CudaBLASResult> load_cublas_results(const char* csv_path) {
    std::vector<CudaBLASResult> results;
    printf("    Trying to open: %s\n", csv_path);
    fflush(stdout);
    std::ifstream file(csv_path);
    if (!file.is_open()) {
        printf("    Warning: Could not open %s\n", csv_path);
        return results;
    }
    printf("    Successfully opened %s\n", csv_path);
    fflush(stdout);
    
    std::string line;
    // Skip header
    std::getline(file, line);
    
    while (std::getline(file, line)) {
        std::stringstream ss(line);
        std::string token;
        std::vector<std::string> tokens;
        
        while (std::getline(ss, token, ',')) {
            tokens.push_back(token);
        }
        
        // Look for FP32 rows with valid sgemm_ms
        if (tokens.size() > 7 && tokens[1] == "FP32") {
            try {
                CudaBLASResult r;
                r.dim = std::stoi(tokens[0]);
                float sgemm = std::stof(tokens[7]); // sgemm_ms column
                r.sgemm_ms = sgemm;
                // Valid if reasonable time (0.001ms to 10000ms)
                r.valid = (sgemm > 0.001f && sgemm < 10000.0f);
                if (r.valid) {
                    results.push_back(r);
                }
            } catch (...) {
                // Skip malformed rows
            }
        }
    }
    return results;
}

// r3x kernel template (double buffer + cp.async)
template <int BM, int BN, int BK, int TM, int TN, int NUM_THREADS>
__global__ __launch_bounds__(NUM_THREADS)
void gemm_fp32_r3x_tuner_kernel(
    const float* __restrict__ A,
    const float* __restrict__ B,
    float* __restrict__ C,
    int M, int N, int K,
    float alpha, float beta)
{
    const uint threadRow = threadIdx.x / (BN / TN);
    const uint threadCol = threadIdx.x % (BN / TN);

    __shared__ float As[2][BK * BM];
    __shared__ float Bs[2][BK * BN];

    const uint cRow = blockIdx.y;
    const uint cCol = blockIdx.x;
    A += cRow * BM * K;
    B += cCol * BN;
    C += cRow * BM * N + cCol * BN;

    const uint innerRowA = threadIdx.x / (BK / 4);
    const uint innerColA = threadIdx.x % (BK / 4);
    constexpr uint rowStrideA = (NUM_THREADS * 4) / BK;

    const uint innerRowB = threadIdx.x / (BN / 4);
    const uint innerColB = threadIdx.x % (BN / 4);
    constexpr uint rowStrideB = NUM_THREADS / (BN / 4);

    float threadResults[TM * TN] = {0.0f};
    float regM[TM];
    float regN[TN];

    // Prefetch tile[0]
    for (uint offset = 0; offset < BM; offset += rowStrideA) {
        if (innerRowA + offset < BM) {
            float4 tmp = reinterpret_cast<const float4*>(
                &A[(innerRowA + offset) * K + innerColA * 4])[0];
            As[0][(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
            As[0][(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
            As[0][(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
            As[0][(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
        }
    }
    for (uint offset = 0; offset < BK; offset += rowStrideB) {
        if (innerRowB + offset < BK) {
            __pipeline_memcpy_async(
                &Bs[0][(innerRowB + offset) * BN + innerColB * 4],
                &B[(innerRowB + offset) * N + innerColB * 4],
                sizeof(float4));
        }
    }
    __pipeline_commit();
    __pipeline_wait_prior(0);
    __syncthreads();

    A += BK;
    B += BK * N;

    uint cur = 0, nxt = 1;

    for (uint bkIdx = BK; bkIdx < K; bkIdx += BK) {
        for (uint offset = 0; offset < BM; offset += rowStrideA) {
            if (innerRowA + offset < BM) {
                float4 tmp = reinterpret_cast<const float4*>(
                    &A[(innerRowA + offset) * K + innerColA * 4])[0];
                As[nxt][(innerColA * 4 + 0) * BM + innerRowA + offset] = tmp.x;
                As[nxt][(innerColA * 4 + 1) * BM + innerRowA + offset] = tmp.y;
                As[nxt][(innerColA * 4 + 2) * BM + innerRowA + offset] = tmp.z;
                As[nxt][(innerColA * 4 + 3) * BM + innerRowA + offset] = tmp.w;
            }
        }
        for (uint offset = 0; offset < BK; offset += rowStrideB) {
            if (innerRowB + offset < BK) {
                __pipeline_memcpy_async(
                    &Bs[nxt][(innerRowB + offset) * BN + innerColB * 4],
                    &B[(innerRowB + offset) * N + innerColB * 4],
                    sizeof(float4));
            }
        }
        __pipeline_commit();

        for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
            for (uint i = 0; i < TM; ++i)
                regM[i] = As[cur][dotIdx * BM + threadRow * TM + i];
            for (uint i = 0; i < TN; ++i)
                regN[i] = Bs[cur][dotIdx * BN + threadCol * TN + i];
            for (uint resM = 0; resM < TM; ++resM)
                for (uint resN = 0; resN < TN; ++resN)
                    threadResults[resM * TN + resN] += regM[resM] * regN[resN];
        }

        __pipeline_wait_prior(0);
        __syncthreads();
        cur ^= 1;
        nxt ^= 1;

        A += BK;
        B += BK * N;
    }

    for (uint dotIdx = 0; dotIdx < BK; ++dotIdx) {
        for (uint i = 0; i < TM; ++i)
            regM[i] = As[cur][dotIdx * BM + threadRow * TM + i];
        for (uint i = 0; i < TN; ++i)
            regN[i] = Bs[cur][dotIdx * BN + threadCol * TN + i];
        for (uint resM = 0; resM < TM; ++resM)
            for (uint resN = 0; resN < TN; ++resN)
                threadResults[resM * TN + resN] += regM[resM] * regN[resN];
    }

    for (uint resM = 0; resM < TM; ++resM) {
        float* row = &C[(threadRow * TM + resM) * N + threadCol * TN];
        for (uint resN = 0; resN < TN; resN += 4) {
            float4 tmp = reinterpret_cast<float4*>(&row[resN])[0];
            const int i = resM * TN + resN;
            tmp.x = alpha * threadResults[i + 0] + beta * tmp.x;
            tmp.y = alpha * threadResults[i + 1] + beta * tmp.y;
            tmp.z = alpha * threadResults[i + 2] + beta * tmp.z;
            tmp.w = alpha * threadResults[i + 3] + beta * tmp.w;
            reinterpret_cast<float4*>(&row[resN])[0] = tmp;
        }
    }
}

// Config definitions
struct TunerConfig {
    const char* name;
    int BM, BN, BK, TM, TN;
};

constexpr TunerConfig CONFIGS[] = {
    {"r3x_64_64_16_8_8",   64, 64, 16, 8, 8},    // 0: Baseline (64 threads)
    {"r3x_64_64_16_4_4",   64, 64, 16, 4, 4},    // 1: High occupancy (256 threads)
    {"r3x_64_64_16_4_8",   64, 64, 16, 4, 8},    // 2: Balanced (128 threads)
    {"r3x_64_64_16_8_4",   64, 64, 16, 8, 4},    // 3: Balanced (128 threads)
    {"r3x_64_64_32_8_8",   64, 64, 32, 8, 8},    // 4: Higher BK (64 threads)
    {"r3x_64_96_16_4_4",   64, 96, 16, 4, 4},    // 5: Wider (384 threads)
    {"r3x_64_128_16_4_4",  64, 128, 16, 4, 4},   // 6: Max width (512 threads)
    {"r3x_96_64_16_4_4",   96, 64, 16, 4, 4},    // 7: Taller (384 threads)
    {"r3x_128_64_16_4_4",  128, 64, 16, 4, 4},   // 8: Max height (512 threads)
    {"r3x_96_96_16_4_4",   96, 96, 16, 4, 4},    // 9: Large square (576 threads)
    {"r3x_128_128_8_8_8",  128, 128, 8, 8, 8},   // 10: Large, low BK (256 threads)
    {"r3x_128_128_16_8_8", 128, 128, 16, 8, 8},  // 11: Large (256 threads)
};
constexpr int NUM_CONFIGS = sizeof(CONFIGS) / sizeof(CONFIGS[0]);

// Launch helper
template<int BM, int BN, int BK, int TM, int TN>
void launch_config(const float* A, const float* B, float* C,
                   int M, int N, int K, float alpha, float beta,
                   cudaStream_t stream) {
    constexpr int THREADS = (BM / TM) * (BN / TN);
    dim3 grid((N + BN - 1) / BN, (M + BM - 1) / BM);
    dim3 block(THREADS);
    gemm_fp32_r3x_tuner_kernel<BM, BN, BK, TM, TN, THREADS>
        <<<grid, block, 0, stream>>>(A, B, C, M, N, K, alpha, beta);
}

// Dispatch based on config index
void dispatch_config(int config_idx, const float* A, const float* B, float* C,
                     int M, int N, int K, float alpha, float beta,
                     cudaStream_t stream) {
    switch (config_idx) {
        case 0: launch_config<64, 64, 16, 8, 8>(A, B, C, M, N, K, alpha, beta, stream); break;
        case 1: launch_config<64, 64, 16, 4, 4>(A, B, C, M, N, K, alpha, beta, stream); break;
        case 2: launch_config<64, 64, 16, 4, 8>(A, B, C, M, N, K, alpha, beta, stream); break;
        case 3: launch_config<64, 64, 16, 8, 4>(A, B, C, M, N, K, alpha, beta, stream); break;
        case 4: launch_config<64, 64, 32, 8, 8>(A, B, C, M, N, K, alpha, beta, stream); break;
        case 5: launch_config<64, 96, 16, 4, 4>(A, B, C, M, N, K, alpha, beta, stream); break;
        case 6: launch_config<64, 128, 16, 4, 4>(A, B, C, M, N, K, alpha, beta, stream); break;
        case 7: launch_config<96, 64, 16, 4, 4>(A, B, C, M, N, K, alpha, beta, stream); break;
        case 8: launch_config<128, 64, 16, 4, 4>(A, B, C, M, N, K, alpha, beta, stream); break;
        case 9: launch_config<96, 96, 16, 4, 4>(A, B, C, M, N, K, alpha, beta, stream); break;
        case 10: launch_config<128, 128, 8, 8, 8>(A, B, C, M, N, K, alpha, beta, stream); break;
        case 11: launch_config<128, 128, 16, 8, 8>(A, B, C, M, N, K, alpha, beta, stream); break;
    }
}

static void init_fp32(float* ptr, int n, unsigned seed) {
    srand(seed);
    for (int i = 0; i < n; ++i)
        ptr[i] = (rand() / float(RAND_MAX)) * 2.0f - 1.0f;
}

// Benchmark cuBLAS sgemm
float benchmark_cublas(cublasHandle_t handle, const float* d_A, const float* d_B, float* d_C,
                       int size, cudaStream_t stream) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    float alpha = 1.0f, beta = 0.0f;
    
    // Warmup
    for (int i = 0; i < 10; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size,
                    &alpha, d_B, size, d_A, size, &beta, d_C, size);
    }
    cudaStreamSynchronize(stream);
    
    // Benchmark
    cudaEventRecord(start, stream);
    for (int i = 0; i < 30; i++) {
        cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, size, size, size,
                    &alpha, d_B, size, d_A, size, &beta, d_C, size);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return ms / 30.0f;
}

// Benchmark one config at one size
float benchmark_config(int config_idx, int size, const float* d_A, const float* d_B, 
                       float* d_C, cudaStream_t stream) {
    //printf("      Starting benchmark_config for config %d, size %d\n", config_idx, size);
    //fflush(stdout);
    
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    
    // Warmup
    //printf("      Warmup...\n");
    //fflush(stdout);
    for (int i = 0; i < 10; i++) {
        dispatch_config(config_idx, d_A, d_B, d_C, size, size, size, 1.0f, 0.0f, stream);
    }
    cudaStreamSynchronize(stream);
    //printf("      Warmup done\n");
    //fflush(stdout);
    
    // Benchmark
    cudaEventRecord(start, stream);
    for (int i = 0; i < 30; i++) {
        dispatch_config(config_idx, d_A, d_B, d_C, size, size, size, 1.0f, 0.0f, stream);
    }
    cudaEventRecord(stop, stream);
    cudaEventSynchronize(stop);
    
    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    
    return ms / 30.0f;
}

// Main benchmark loop
void run_tuner(cublasHandle_t handle, cudaStream_t stream) {
    printf("  Entering run_tuner...\n");
    fflush(stdout);
    
    const int sizes[] = {128, 256, 512, 1024};
    const int num_sizes = 4;
    const int num_configs = NUM_CONFIGS;
    
    printf("  Loading CSV...\n");
    fflush(stdout);
    // Load existing cuBLAS results - try multiple paths
    auto cublas_results = load_cublas_results("../results/benchmark_results.csv");
    if (cublas_results.empty()) {
        cublas_results = load_cublas_results("results/benchmark_results.csv");
    }
    printf("  Loaded %zu results from CSV\n", cublas_results.size());
    fflush(stdout);
    
    // Get cuBLAS times
    printf("  Getting cuBLAS times for sizes...\n");
    fflush(stdout);
    std::vector<float> cublas_times(num_sizes, 0.0f);
    for (int i = 0; i < num_sizes; i++) {
        printf("    Looking for size %d...\n", sizes[i]);
        fflush(stdout);
        auto it = std::find_if(cublas_results.begin(), cublas_results.end(),
            [sizes, i](const CudaBLASResult& r) { return r.dim == sizes[i] && r.valid; });
        if (it != cublas_results.end()) {
            cublas_times[i] = it->sgemm_ms;
            printf("    Loaded: %.4f ms\n", cublas_times[i]);
        } else {
            printf("    Not found, will benchmark\n");
        }
        fflush(stdout);
    }
    printf("  Done loading cuBLAS times\n");
    fflush(stdout);
    
    // Results storage
    struct Result {
        int config_idx;
        int size_idx;
        float custom_ms;
        float ratio_pct;
    };
    std::vector<Result> results;
    
    // Run all configs
    printf("\n=== R3X Tuner ===\n");
    printf("Testing %d configurations on sizes: 128, 256, 512, 1024\n\n", num_configs);
    fflush(stdout);
    
    for (int s = 0; s < num_sizes; s++) {
        int size = sizes[s];
        printf("  Processing size %d...\n", size);
        fflush(stdout);
        
        // Allocate matrices for this size
        printf("    Allocating GPU memory...\n");
        fflush(stdout);
        float *d_A, *d_B, *d_C;
        cudaError_t err;
        err = cudaMalloc(&d_A, size * size * sizeof(float));
        if (err != cudaSuccess) { printf("    cudaMalloc d_A failed: %s\n", cudaGetErrorString(err)); continue; }
        err = cudaMalloc(&d_B, size * size * sizeof(float));
        if (err != cudaSuccess) { printf("    cudaMalloc d_B failed: %s\n", cudaGetErrorString(err)); cudaFree(d_A); continue; }
        err = cudaMalloc(&d_C, size * size * sizeof(float));
        if (err != cudaSuccess) { printf("    cudaMalloc d_C failed: %s\n", cudaGetErrorString(err)); cudaFree(d_A); cudaFree(d_B); continue; }
        
        printf("    Allocating CPU memory...\n");
        fflush(stdout);
        // Initialize with random data
        float* h_A = (float*)malloc(size * size * sizeof(float));
        float* h_B = (float*)malloc(size * size * sizeof(float));
        if (!h_A || !h_B) {
            printf("    CPU malloc failed!\n");
            cudaFree(d_A); cudaFree(d_B); cudaFree(d_C);
            continue;
        }
        printf("    Initializing data...\n");
        fflush(stdout);
        init_fp32(h_A, size * size, 42);
        init_fp32(h_B, size * size, 43);
        printf("    Copying to GPU...\n");
        fflush(stdout);
        cudaMemcpy(d_A, h_A, size * size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_B, h_B, size * size * sizeof(float), cudaMemcpyHostToDevice);
        free(h_A);
        free(h_B);
        printf("    Ready to benchmark configs...\n");
        fflush(stdout);
        
        // Get cuBLAS time if needed
        if (cublas_times[s] < 0.001f) {
            printf("Benchmarking cuBLAS for size %d...\n", size);
            cublas_times[s] = benchmark_cublas(handle, d_A, d_B, d_C, size, stream);
            printf("  cuBLAS: %.4f ms\n", cublas_times[s]);
        }
        
        printf("\n=== Size %d (cuBLAS: %.4f ms) ===\n", size, cublas_times[s]);
        printf("%-22s %10s %10s %10s\n", "Config", "Custom(ms)", "Ratio%", "Status");
        printf("%s\n", std::string(55, '-').c_str());
        
        for (int c = 0; c < num_configs; c++) {
            const auto& cfg = CONFIGS[c];
            printf("    Testing config %d: %s\n", c, cfg.name);
            fflush(stdout);
            
            // Check SMEM
            int smem = 2 * (cfg.BM * cfg.BK + cfg.BK * cfg.BN) * 4;
            if (smem > 49152) {
                printf("%-22s %10s %10s %10s\n", cfg.name, "-", "-", "SKIP(>48KB)");
                continue;
            }
            
            float custom_ms = benchmark_config(c, size, d_A, d_B, d_C, stream);
            float ratio = (cublas_times[s] / custom_ms) * 100.0f;
            
            printf("%-22s %10.4f %9.1f%% %10s\n", cfg.name, custom_ms, ratio, "OK");
            
            results.push_back({c, s, custom_ms, ratio});
        }
        
        cudaFree(d_A);
        cudaFree(d_B);
        cudaFree(d_C);
    }
    
    // Write CSV
    FILE* csv = fopen("results/r3x_tuner.csv", "w");
    fprintf(csv, "config,bm,bn,bk,tm,tn,size,custom_ms,cublas_ms,ratio_pct\n");
    for (const auto& r : results) {
        const auto& cfg = CONFIGS[r.config_idx];
        fprintf(csv, "%s,%d,%d,%d,%d,%d,%d,%.6f,%.6f,%.2f\n",
                cfg.name, cfg.BM, cfg.BN, cfg.BK, cfg.TM, cfg.TN,
                sizes[r.size_idx], r.custom_ms, cublas_times[r.size_idx], r.ratio_pct);
    }
    fclose(csv);
    
    // Print summary
    printf("\n=== Summary ===\n");
    printf("Results written to: results/r3x_tuner.csv\n\n");
    
    // Find best per size
    printf("Best by size:\n");
    for (int s = 0; s < num_sizes; s++) {
        float best_ratio = 0;
        int best_idx = -1;
        for (const auto& r : results) {
            if (r.size_idx == s && r.ratio_pct > best_ratio) {
                best_ratio = r.ratio_pct;
                best_idx = r.config_idx;
            }
        }
        if (best_idx >= 0) {
            printf("  Size %d: %-22s (%.1f%% of cuBLAS)\n", 
                   sizes[s], CONFIGS[best_idx].name, best_ratio);
        }
    }
    
    // Find overall best (weighted by size importance, or just best at 512)
    printf("\nBest at target size 512:\n");
    float best_512 = 0;
    int best_512_idx = -1;
    for (const auto& r : results) {
        if (sizes[r.size_idx] == 512 && r.ratio_pct > best_512) {
            best_512 = r.ratio_pct;
            best_512_idx = r.config_idx;
        }
    }
    if (best_512_idx >= 0) {
        const auto& cfg = CONFIGS[best_512_idx];
        printf("  %s (BM=%d, BN=%d, BK=%d, TM=%d, TN=%d) = %.1f%%\n",
               cfg.name, cfg.BM, cfg.BN, cfg.BK, cfg.TM, cfg.TN, best_512);
    }
}

int main() {
    printf("R3X Tuner for Medium Sizes (128, 256, 512, 1024)\n");
    printf("===============================================\n\n");
    fflush(stdout);
    
    printf("Initializing cuBLAS...\n");
    fflush(stdout);
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    if (status != CUBLAS_STATUS_SUCCESS) {
        printf("Failed to create cuBLAS handle: %d\n", status);
        return 1;
    }
    
    printf("Creating CUDA stream...\n");
    fflush(stdout);
    cudaStream_t stream;
    cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess) {
        printf("Failed to create CUDA stream: %s\n", cudaGetErrorString(err));
        cublasDestroy(handle);
        return 1;
    }
    
    printf("Setting cuBLAS stream...\n");
    fflush(stdout);
    cublasSetStream(handle, stream);
    
    printf("Running tuner...\n");
    fflush(stdout);
    run_tuner(handle, stream);
    
    printf("Cleaning up...\n");
    fflush(stdout);
    cublasDestroy(handle);
    cudaStreamDestroy(stream);
    
    printf("\nDone.\n");
    return 0;
}
