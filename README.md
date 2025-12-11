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

Needs CUDA toolkit to be available. (see CSCS.md for instructions on how to set up the environment on CSCS cluster)

```bash
pip install . --no-build-isolation --no-deps
```

or using uv:

```bash
make install
```

## Testing

```bash
pytest tests/ -v
```

or using makefile:

```bash
make test
```

## Benchmarking
### PyTorch benchmarking
```bash
pip install megatron-core
python3 benchmarks/pytorch/sss_sigmoid_benchmark.py
python3 benchmarks/pytorch/xsss_scaledSigmoid_benchmark.py
```
and/or other benchmark files if applicable

#### Visualize PyTorch benchmarking
After executing 1 or more PyTorch benchmarks:
```bash
python3 benchmarks/plotting/create_graph.py
```
The resulting plots will be generated at benchmarks/results/pytorch_plots <br>

### CUDA benchmarking
```bash
cmake -S benchmarks/cuda/ -B build && cmake --build build --parallel --target cuda_benchmark
./build/cuda_benchmark
```

#### Visualize CUDA benchmarking
```bash
python3 benchmarks/plotting/create_boxplots.py GH200
```
The resulting plots will be generated at benchmarks/results/cuda_plots <br>
If you have executed the benchmark on another GPU:
- Add the GPU to benchmarks/device_specs/gpus.json
- Adjust the argument "GH200" in the CUDA benchmarking call

# Acknowledgments
- Cuda implementation of [xielu](https://github.com/rubber-duck-debug/xielu) for repository structure.
