#include <iostream>
#include <vector>
#include <functional>
#include <iomanip>
#include "../cusss/utils/template_utils.hpp"

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    }


template <typename scalar_t>
__global__ void identity_kernel(const scalar_t* x, scalar_t* output, int size) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < size) {
    output[idx] = x[idx];
  }
}

// ------------------------------------------------------------
// A wrapper for ANY kernel with signature:
// __global__ void kernel(const scalar_t*, scalar_t*, int);
// ------------------------------------------------------------
template <typename scalar_t>
struct KernelWrapper {
    std::string name;
    std::function<void (const scalar_t*, scalar_t*, int)> func;  // kernel symbol pointer
    dim3 grid;
    dim3 block;
};

// ------------------------------------------------------------
// Launch helper (kernel function pointer cannot be invoked directly)
// We wrap the <<<grid,block>>> inside a templated ‘launcher’.
// ------------------------------------------------------------
// template <typename scalar_t>
// template <void (*kernel)(const scalar_t*, scalar_t*, int)>
// void launch_kernel(dim3 grid, dim3 block,
//                    const scalar_t* x, scalar_t* y, int n) {
//     kernel<<<grid, block>>>(x, y, n);
// }

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


using scalar_t = cu10::Bfloat16;
int main() {

    int n = 1 << 20;  // 1M
    int bytes = n * sizeof(scalar_t);

    scalar_t *x, *y;
    CUDA_CHECK(cudaMalloc(&x, bytes));
    CUDA_CHECK(cudaMalloc(&y, bytes));

    dim3 block(256);
    dim3 grid((n + block.x - 1) / block.x);

    const int iters = 500;

    // Register kernels in a vector
    std::vector<KernelWrapper> kernels;

    kernels.push_back({
        "Identity",
        [grid, block] (const scalar_t* x, scalar_t* y, int m) {
            identity_kernel<<<grid, block>>>(x, y, m);
        },
        grid, block
    });

    kernels.push_back({
        "SSS_bf16",
        [grid, block] (const scalar_t* x, scalar_t* y, int m) {
            sss_elementwise_op<<<grid, block>>>(x, y, m);
        },
        grid, block
    });

    kernels.push_back({
        "SSS_bf16_with_conversion",
        [block] (const scalar_t* x, scalar_t* y, int m) {
            int num4 = m / 4;
            int numBlocks = (num4 + block.x - 1) / block.x;
            sss_elementwise_bf16_with_conv<<<numBlocks, block>>>(x, y, m);
        },
        grid, block
    });
    
    std::cout << std::left;
    for (auto& k : kernels) {
        float per_ms = benchmark(k, x, y, n, iters);
        std::cout << std::setw(10) << k.name
            << std::right << std::setw(10) << per_ms * 1000
            << " µs\n"
            << std::left;
    }

    cudaFree(x);
    cudaFree(y);
    return 0;
}