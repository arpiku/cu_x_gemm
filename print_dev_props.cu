#include <cstdio>
#include <cuda_runtime.h>

int main() {
    int device_count;
    cudaGetDeviceCount(&device_count);
    for (int d = 0; d < device_count; d++) {
        cudaDeviceProp p;
        cudaGetDeviceProperties(&p, d);
        printf("=== Device %d: %s (SM %d.%d) ===\n", d, p.name, p.major, p.minor);
        printf("  multiProcessorCount       = %d\n", p.multiProcessorCount);
        printf("  maxThreadsPerMultiProcessor = %d\n", p.maxThreadsPerMultiProcessor);
        printf("  maxThreadsPerBlock          = %d\n", p.maxThreadsPerBlock);
        printf("  maxThreadsDim               = %d x %d x %d\n", p.maxThreadsDim[0], p.maxThreadsDim[1], p.maxThreadsDim[2]);
        printf("  maxGridSize                 = %d x %d x %d\n", p.maxGridSize[0], p.maxGridSize[1], p.maxGridSize[2]);
        printf("  clockRate                   = %d kHz (%.2f GHz boost)\n", p.clockRate, p.clockRate / 1e6);
        printf("  memoryClockRate              = %d kHz (%.2f GHz)\n", p.memoryClockRate, p.memoryClockRate / 1e6);
        printf("  memoryBusWidth               = %d bits\n", p.memoryBusWidth);
        printf("  totalGlobalMem              = %.0f MB\n", p.totalGlobalMem / (1024.0 * 1024.0));
        printf("  sharedMemPerBlock           = %zu KB\n", p.sharedMemPerBlock / 1024);
        printf("  sharedMemPerMultiprocessor  = %zu KB\n", p.sharedMemPerMultiprocessor / 1024);
        printf("  l2CacheSize                  = %d KB\n", p.l2CacheSize / 1024);
        printf("  regsPerBlock                = %d\n", p.regsPerBlock);
        printf("  regsPerMultiprocessor       = %d\n", p.regsPerMultiprocessor);
        printf("  warpSize                    = %d\n", p.warpSize);
        printf("  totalConstMem               = %zu KB\n", p.totalConstMem / 1024);
        printf("  memoryBandwidth             = %.1f GB/s\n",
               2.0 * p.memoryBusWidth / 8.0 * p.memoryClockRate * 1e3 / 1e9);
    }
    return 0;
}