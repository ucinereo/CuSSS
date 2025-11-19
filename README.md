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
### Benchmark our SSS Implementations vs PyTorch Sigmoid
```bash
pip install megatron-core
python3 benchmarks/sss_sigmoid_benchmark.py
```

### Benchmark SSS forward vs Sigmoid forward kernels
```bash
nvcc benchmarks/kernel_benchmark.cu -O3 -o ./build/cuda_benchmark
./build/cuda_benchmark
```

# Acknowledgments
- Cuda implementation of [xielu](https://github.com/rubber-duck-debug/xielu) for repository structure.
