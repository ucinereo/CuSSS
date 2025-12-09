# Benchmarking

## PyTorch Benchmarking
### Benchmark our modules in PyTorch
```bash
pip install megatron-core
python3 benchmarks/pytorch/sss_sigmoid_benchmark.py
```
or other benchmark files if applicable
### Add other kernel implementations
Create a PyTorch Wrapper in cuss/ops/
In benchmarks/pytorch_benchmarks:
- Copy the file sss_sigmoid_benchmark.py and adjust according to your needs

## CUDA Benchmarking
### Directly benchmark the CUDA forward kernels
```bash
cmake -S benchmarks/cuda/ -B build && cmake --build build --parallel --target cuda_benchmark
./build/cuda_benchmark
```

### Visualize CUDA & PyTorch results
```bash
python3 benchmarks/plotting/create_boxplots.py GH200
python3 benchmarks/plotting/create_graph.py
```
The resulting plot will be generated at benchmarks/results/ <br>
If you have executed the benchmark on another GPU:
- Add the GPU to benchmarks/device_specs/gpus.json
- Adjust the argument "GH200" in the CUDA benchmarking call

### Add other existing kernel implementations
In benchmarks/cuda_benchmarks/all_kernels_import.hpp: 
- Add an import for the header file which references the kernel (such as #include "../../../../cusss/csrc/cuda/sss_cuda.hpp" for SSS)
- Add the kernel in the kernel_benchmark.cu main() function (note that some kernels may have more parameters, such as parameter "a" in xSSS. Here is an example of how that may look like:)
```cpp
    kernels.push_back({
        "xSSS",
        [grid, block] (const float* x, float* y, int m) { // The inputs for the wrapper function should stay the same
            xsss_forward_kernel<<<grid, block>>>(x, y, m, 1.0f); // Add some static value 1.0f for the parameter "a"
        },
        grid, block
    });
```

### Alternatively: Add a kernel directly, without cusss implementation:
- In benchmarks/cuda/src/kernels/baseline_kernels add your kernel in baseline_kernels.cu and baseline_kernels.hpp
- Add the kernel in the kernel_benchmark.cu main() function like above
