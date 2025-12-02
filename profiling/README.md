# Profiling
```
nsys profile   --trace=cuda,nvtx,osrt,cudnn   --capture-range=cudaProfilerApi   --python-sampling=true   --python-backtrace=cuda   --force-overwrite=true   --output=my_kernel_trace_py   python profiling/trace_kernel.py
```