#include <iostream>
#include <vector>
#include <functional>
#include <iomanip>
#include "../../cusss/csrc/utils/template_utils.hpp"
#include "kernels/all_kernels_import.hpp"
#include <torch/script.h>
#include <cuda_bf16.h>


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
template <typename scalar_t>
float benchmark(const KernelWrapper<scalar_t>& k, const scalar_t* x, scalar_t* y, int n, int iters)
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

template <typename scalar_t>
void value_comparison(const KernelWrapper<scalar_t>& k1, const KernelWrapper<scalar_t>& k2,
                      const scalar_t* x, scalar_t* y1, scalar_t* y2, int n)
{
    k1.func(x, y1, n);
    k2.func(x, y2, n);
    CUDA_CHECK(cudaDeviceSynchronize());

    std::vector<scalar_t> h_y1(n);
    std::vector<scalar_t> h_y2(n);
    CUDA_CHECK(cudaMemcpy(h_y1.data(), y1, n * sizeof(scalar_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_y2.data(), y2, n * sizeof(scalar_t), cudaMemcpyDeviceToHost));

    for (int i = 0; i < n; i++) {
        if (h_y1[i] != h_y2[i]) {
            std::cerr << "Mismatch at index " << i << ": "
                      << static_cast<float>(h_y1[i]) << " vs "
                      << static_cast<float>(h_y2[i]) << std::endl;
            return;
        }
    }
    std::cout << "Outputs match!" << std::endl;
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

    const int iters = 10000;

    auto k0 = KernelWrapper<float>{
        "Identity",
        [grid, block] (const float* x, float* y, int m) {
            identity_kernel<<<grid, block>>>(x, y, m);
        },
        grid, block
    };

    auto k1 = KernelWrapper<c10::BFloat16>{
        "SSS_bf16",
        [grid, block] (const c10::BFloat16* x, c10::BFloat16* y, int m) {
            sss_forward_kernel<c10::BFloat16><<<grid, block>>>(x, y, m);
        },
        grid, block
    };

    auto k2 = KernelWrapper<c10::Half>{
        "SSS_fp16",
        [grid, block] (const c10::Half* x, c10::Half* y, int m) {
            sss_forward_kernel<c10::Half><<<grid, block>>>(x, y, m);
        },
        grid, block
    };
    float per_ms;
    per_ms = benchmark(k0, x, y, n, iters);
    std::cout << std::left
        << std::setw(10) << k0.name
        << std::right << std::setw(10) << per_ms * 1000
        << " µs\n"
        << std::left;
    per_ms = benchmark(k1, reinterpret_cast<const c10::BFloat16*>(x), reinterpret_cast<c10::BFloat16*>(y), n, iters);
    std::cout << std::left
        << std::setw(10) << k1.name
        << std::right << std::setw(10) << per_ms * 1000
        << " µs\n"
        << std::left;
    per_ms = benchmark(k2, reinterpret_cast<const c10::Half*>(x), reinterpret_cast<c10::Half*>(y), n, iters);
    std::cout << std::left
        << std::setw(10) << k2.name
        << std::right << std::setw(10) << per_ms * 1000
        << " µs\n"
        << std::left;
    // std::cout << std::left;
    // for (auto& k : kernels) {
    //     float per_ms = benchmark(k, x, y, n, iters);
    //     std::cout << std::setw(10) << k.name
    //         << std::right << std::setw(10) << per_ms * 1000
    //         << " µs\n"
    //         << std::left;
    // }

    // value_comparison(kernels[1], kernels[2], x, y, y, n);

    cudaFree(x);
    cudaFree(y);
    return 0;
}