#include "helpers.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>

void init_bf16(__nv_bfloat16* ptr, int n, unsigned seed) {
    srand(seed);
    for (int i = 0; i < n; ++i) {
        ptr[i] = __float2bfloat16((rand() / float(RAND_MAX)) * 2.0f - 1.0f);
    }
}

void init_fp32(float* ptr, int n, unsigned seed) {
    srand(seed);
    for (int i = 0; i < n; ++i) {
        ptr[i] = (rand() / float(RAND_MAX)) * 2.0f - 1.0f;
    }
}

float compute_l2_error(const float* a, const float* b, int n) {
    double sum = 0.0;
    double ref = 0.0;
    for (int i = 0; i < n; ++i) {
        const double diff = double(a[i]) - double(b[i]);
        sum += diff * diff;
        ref += double(b[i]) * double(b[i]);
    }
    return float(std::sqrt(sum) / (std::sqrt(ref) + 1e-10));
}

int find_dimension_index(int dim) {
    for (int i = 0; i < NUM_DIMS; ++i) {
        if (DIMENSIONS[i] == dim) {
            return i;
        }
    }
    return -1;
}

TargetArch detect_target_arch_from_device(const cudaDeviceProp& prop) {
    if (prop.major == 9) return TargetArch::H100;
    if (prop.major == 12) return TargetArch::RTX5070;

    std::printf("Unsupported SM %d.%d; defaulting to RTX5070\n", prop.major, prop.minor);
    return TargetArch::RTX5070;
}

std::string benchmark_output_dir(TargetArch arch) {
    return std::string("results/") + target_arch_name(arch);
}

std::string benchmark_csv_path(TargetArch arch) {
    return benchmark_output_dir(arch) + "/benchmark_results.csv";
}