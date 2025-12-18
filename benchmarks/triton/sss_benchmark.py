import torch
import triton
import triton.testing

import cusss
import cusss.triton as tcusss
from sss_triton_no_autograd import sss_triton_forward, sss_triton_backward


@triton.testing.perf_report(
    triton.testing.Benchmark(
        x_names=["N"],
        x_vals=[128 * 1024 * i for i in range(2, 100, 10)],
        line_arg="provider",
        line_vals=["torch-sigmoid", "cuda-sss", "triton-sss", "triton-sss-direct"],
        line_names=[
            "PyTorch Sigmoid",
            "Custom CUDA",
            "Triton SSS",
            "Triton SSS Direct",
        ],
        styles=[("blue", "-"), ("green", "-"), ("red", "-"), ("purple", "-")],
        ylabel="GB/s",
        plot_name="sss-checkpointing-performance",
        args={},
    )
)
def benchmark(N, provider):
    x = torch.randn(N, device="cuda", dtype=torch.float32, requires_grad=True)
    grad_out = torch.randn(N, device="cuda", dtype=torch.float32)

    # Define execution steps (2 FWD + 1 BWD)
    def run_torch():
        _ = torch.sigmoid(x)
        y = torch.sigmoid(x)
        y.backward(grad_out)
        x.grad = None

    def run_cuda():
        _ = cusss.SSS.forward(None, x)
        y = cusss.SSS.forward(None, x)
        y.backward(grad_out)
        x.grad = None

    def run_triton():
        _ = tcusss.SSS.forward(None, x)
        y = tcusss.SSS.forward(None, x)
        y.backward(grad_out)
        x.grad = None

    def run_triton_direct():
        _ = sss_triton_forward(x)
        y = sss_triton_forward(x)
        _ = sss_triton_backward(y, grad_out)
        x.grad = None

    quantiles = [0.5, 0.2, 0.8]
    if provider == "torch-sigmoid":
        func = run_torch
    elif provider == "cuda-sss":
        func = run_cuda
    elif provider == "triton-sss":
        func = run_triton
    elif provider == "triton-sss-direct":
        func = run_triton_direct

    ms, min_ms, max_ms = triton.testing.do_bench(func, quantiles=quantiles)

    # Throughput Calculation:
    # Forward 1: Read X, Write Y (2 ops)
    # Forward 2: Read X, Write Y (2 ops)
    # Backward : Read Y, Read GradOut, Write GradX (3 ops)
    # Total    : 7 memory accesses per element
    total_ops = 7

    gbps = lambda ms: total_ops * x.numel() * x.element_size() * 1e-9 / (ms * 1e-3)  # noqa
    return gbps(ms), gbps(max_ms), gbps(min_ms)


if __name__ == "__main__":
    df = benchmark.run(show_plots=False, print_data=True, save_path=".", return_df=True)
    df.to_csv("sss_checkpointing_results.csv", index=False)
