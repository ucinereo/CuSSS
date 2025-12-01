# CuSSS: Cuda implementation of SSS variants
<div align="center">
Nicola Studer
&nbsp;&nbsp;&nbsp;&nbsp;
Marino Eisenegger
&nbsp;&nbsp;&nbsp;&nbsp;
Tristan Gabl
&nbsp;&nbsp;&nbsp;&nbsp;
Benedict Armstrong
&nbsp;&nbsp;&nbsp;&nbsp;
Valentin Vogt

ETH Zurich, Switzerland

Large-Scale AI Engineering
&#8226;
HS2025
</div>

# Getting Started

**Requirements**
- Python >= 3.10
- PyTorch >= 2.10
- CUDA Toolkit
- CMake >= 3.10

## Installation

```bash
pip install . --no-build-isolation --no-deps
```

## Testing

```bash
pytest tests/ -v
```

## Benchmarking
### PyTorch benchmarking
```bash
pip install megatron-core
python3 benchmarks/pytorch/sss_sigmoid_benchmark.py
python3 benchmarks/pytorch/sssglu_swiglu_benchmark.py
```
or other benchmark files if applicable

### CUDA benchmarking
```bash
cmake -S benchmarks/cuda/ -B build && cmake --build build --parallel --target cuda_benchmark
./build/cuda_benchmark
```

### Visualize CUDA results
```bash
python3 benchmarks/plotting/create_boxplots.py GH200
```
The resulting plot will be generated at benchmarks/results/boxplots_cuda.pdf <br>
If you have executed the benchmark on another GPU:
- Add the GPU to benchmarks/device_specs/gpus.json
- Adjust the argument "GH200" in the CUDA benchmarking call

# Acknowledgments
- Cuda implementation of [xielu](https://github.com/rubber-duck-debug/xielu) for repository structure.
