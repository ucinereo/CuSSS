#include <iostream>
#include <vector>
#include <functional>
#include "kernels/all_kernels_import.hpp"
#include <algorithm>
#include <numeric>
#include <string>
#include <nlohmann/json.hpp>

#define CUDA_CHECK(err) \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(err) \
                  << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    }

// ------------------------------------------------------------
// Helper functions to compute and print statistics
// ------------------------------------------------------------
struct Stats {
    float mean, median, p5, p95, min, median_throughput;
};

Stats compute_stats(std::vector<float>& v, int n) {
    std::sort(v.begin(), v.end());
    Stats s;
    s.mean = std::accumulate(v.begin(), v.end(), 0.0f) / v.size();
    s.median = v[v.size()/2];
    s.p5 = v[std::max(int(0.05*v.size()), 0)];
    s.p95 = v[std::min(int(0.95*v.size()), int(v.size()-1))];
    s.min = v[0];
    s.median_throughput = 2.0f * n * sizeof(float) / (s.median * 1e6f); // GB/s
    return s;
}

void print_block_divider() {
    std::cout << std::string(87, '-') << std::endl;  // separator
}

void print_header() {

    std::cout << std::left
              << std::setw(10) << "Kernel"
              << std::setw(12) << "TensorSize"
              << std::setw(10) << "Block"
              << std::setw(10) << "Mean(ms)"
              << std::setw(10) << "Median(ms)"
              << std::setw(10) << "P5(ms)"
              << std::setw(10) << "P95(ms)"
              << std::setw(27) << "Median Throughput(GB/s)"
              << std::endl;

    print_block_divider();
}


std::string human_size_mem(int n_elements) {
    size_t bytes = n_elements * sizeof(float); // total bytes for one tensor
    float mb = bytes / (1024.0f * 1024.0f);
    if (mb < 1024.0f) {
        char buf[16];
        snprintf(buf, sizeof(buf), "%.1f MB", mb);
        return std::string(buf);
    } else {
        float gb = mb / 1024.0f;
        char buf[16];
        snprintf(buf, sizeof(buf), "%.2f GB", gb);
        return std::string(buf);
    }
}

void print_result(const std::string& name, int n, int block,
                  const Stats& s) {
    std::cout << std::left
              << std::setw(10) << name
              << std::setw(12) << human_size_mem(n)  // <-- human-readable
              << std::setw(10) << block
              << std::setw(10) << std::fixed << std::setprecision(3) << s.mean
              << std::setw(10) << s.median
              << std::setw(10) << s.p5
              << std::setw(10) << s.p95
              << std::setw(27) << s.median_throughput
              << std::endl;
}

// ------------------------------------------------------------
// A wrapper for ANY kernel with signature:
// __global__ void kernel(const float*, float*, int);
// ------------------------------------------------------------
template <typename scalar_t>
struct KernelWrapper {
    std::string name;
    std::function<void (const scalar_t*, scalar_t*, int, dim3, dim3)> func;
    dim3 grid;
    dim3 block;
    int num_floats_per_kernel;
};

// ------------------------------------------------------------
// Launch helper for template kernels
// ------------------------------------------------------------
template<typename scalar_t>
__host__ void launch_sss_forward(const scalar_t* x, scalar_t* y, int n, dim3 grid, dim3 block) {
    sss_forward_kernel<scalar_t><<<grid, block>>>(x, y, n);
}

// ------------------------------------------------------------
// Benchmark function
// ------------------------------------------------------------
template <typename scalar_t>
std::vector<scalar_t> benchmark(const KernelWrapper<scalar_t>& k,
                const scalar_t* x, scalar_t* y, int n, int iters)
{
    std::vector<float> measurements(iters, 0);

    // Warmup
    for (int i = 0; i < 100; i++) {
        k.func(x, y, n, k.grid, k.block);
    }

    for (int i = 0; i < iters; i++) {
        CUDA_CHECK(cudaDeviceSynchronize());

        cudaEvent_t start, end;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&end));

        CUDA_CHECK(cudaEventRecord(start));

        k.func(x, y, n, k.grid, k.block);

        CUDA_CHECK(cudaEventRecord(end));
        CUDA_CHECK(cudaEventSynchronize(end));

        float ms = 0;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, end));
        measurements[i] = ms;

        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(end));
    }

    return measurements;
}

// ------------------------------------------------------------
// MAIN
// ------------------------------------------------------------


int main() {

    // Benchmarking parameters:
    const int iters = 500;

    std::vector<int> tensor_sizes = {1<<18, 1<<20, 1<<22, 1<<24, 1<<26, 1<<28};
    std::vector<int> block_sizes = {128, 256, 512, 1024};

    // Register kernels in a vector
    std::vector<KernelWrapper<float>> kernels;

    // Initialize block and grid to value 0 (gets adjusted in the benchmarking-for-loop anyways)
    dim3 block(0);
    dim3 grid(0);

    kernels.push_back({
        "Sigmoid",
        [] (const float* x, float* y, int m, dim3 grid, dim3 block) {
            sigmoid_forward_kernel<<<grid, block>>>(x, y, m);
        },
        grid, block, 1
    });

    kernels.push_back({
        "Identity",
        [] (const float* x, float* y, int m, dim3 grid, dim3 block) {
            identity_kernel<<<grid, block>>>(x, y, m);
        },
        grid, block, 1
    });

    kernels.push_back({
        "SSS",
        [] (const float* x, float* y, int m, dim3 grid, dim3 block) {
            launch_sss_forward<float>(x, y, m, grid, block);
        },
        grid, block, 4
    });

    nlohmann::json results_json;

    results_json["results"] = nlohmann::json::array();

    float *x, *y;
    // For each tensor-size, go over all block-sizes and benchmark each kernel
    // Then print the results to the command line and save them for the JSON-file
    for (int n : tensor_sizes) {
        int bytes = n * sizeof(float);
        CUDA_CHECK(cudaMalloc(&x, bytes));
        CUDA_CHECK(cudaMalloc(&y, bytes));

        print_header();

        for (int b : block_sizes) {           
            dim3 block(b); 
            
            for (auto& k : kernels) {

                dim3 grid((n / k.num_floats_per_kernel + b - 1) / b);

                k.grid = grid;
                k.block = block;
                std::vector<float> measurements = benchmark(k, x, y, n, iters);

                // Prints the command-line stats
                Stats s = compute_stats(measurements, n);
                print_result(k.name, n, b, s);

                // Creates and entry for the JSON
                nlohmann::json entry;
                entry["kernel"] = k.name;
                entry["tensor_size_bytes"] = n * sizeof(float);
                entry["block_size"] = block.x;
                entry["measurements_ms"] = measurements;
                results_json["results"].push_back(entry);

            }

            print_block_divider();

        }
        print_block_divider();

        cudaFree(x);
        cudaFree(y);
    }

    // Save results in JSON-file for further visualizations
    std::string json_file_path = "benchmarks/results/cuda.json";
    std::cout << "Storing results in " << json_file_path << std::endl;
    std::ofstream f(json_file_path);
    f << results_json.dump(2);   // pretty printed
    f.close();

    return 0;
}
