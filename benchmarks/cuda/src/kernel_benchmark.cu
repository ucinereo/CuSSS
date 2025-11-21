#include <iostream>
#include <vector>
#include <functional>
#include "kernels/all_kernels_import.hpp"

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    }

// ------------------------------------------------------------
// A wrapper for ANY kernel with signature:
// __global__ void kernel(const float*, float*, int);
// ------------------------------------------------------------
struct KernelWrapper {
    std::string name;
    std::function<void (const float*, float*, int)> func;  // kernel symbol pointer
    dim3 grid;
    dim3 block;
};

// ------------------------------------------------------------
// Launch helper (kernel function pointer cannot be invoked directly)
// We wrap the <<<grid,block>>> inside a templated ‘launcher’.
// ------------------------------------------------------------
template <void (*kernel)(const float*, float*, int)>
void launch_kernel(dim3 grid, dim3 block,
                   const float* x, float* y, int n) {
    kernel<<<grid, block>>>(x, y, n);
}

// ------------------------------------------------------------
// Benchmark function
// ------------------------------------------------------------
float benchmark(const KernelWrapper& k,
                const float* x, float* y, int n, int iters)
{
    // Warmup
    for (int i = 0; i < 10; i++) {
        k.func(x, y, n);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    cudaEvent_t start, end;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&end));

    CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < iters; i++) {
        k.func(x, y, n);
    }
    CUDA_CHECK(cudaEventRecord(end));
    CUDA_CHECK(cudaEventSynchronize(end));

    float ms = 0;
    CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));

    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(end));

    return ms / iters;
}

// ------------------------------------------------------------
// MAIN
// ------------------------------------------------------------



int main() {

    int n = 1 << 20;  // 1M
    int bytes = n * sizeof(float);

    float *x, *y;
    CUDA_CHECK(cudaMalloc(&x, bytes));
    CUDA_CHECK(cudaMalloc(&y, bytes));

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);

    const int iters = 500;

    // Register kernels in a vector
    std::vector<KernelWrapper> kernels;

    kernels.push_back({
        "Sigmoid",
        [grid, block] (const float* x, float* y, int m) {
            sigmoid_forward_kernel<<<grid, block>>>(x, y, m);
        },
        grid, block
    });

    kernels.push_back({
        "Identity",
        [grid, block] (const float* x, float* y, int m) {
            identity_kernel<<<grid, block>>>(x, y, m);
        },
        grid, block
    });

    kernels.push_back({
        "SSS",
        [grid, block] (const float* x, float* y, int m) {
            sss_forward_kernel<<<grid, block>>>(x, y, m);
        },
        grid, block
    });

    for (auto& k : kernels) {
        float per_ms = benchmark(k, x, y, n, iters);
        std::cout << k.name << ": " << per_ms * 1000
                  << " µs per launch" << std::endl;
    }

    cudaFree(x);
    cudaFree(y);
    return 0;
}