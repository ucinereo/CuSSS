import torch
import numpy as np
import json
from pathlib import Path
import argparse


def summarize_timings(t):
    t = np.array(t, dtype=float)
    # convert numpy scalars to native python types for JSON serialization
    return {
        "mean": float(t.mean()),
        "std": float(t.std()),
        "median": float(np.median(t)),
        "p5": float(np.percentile(t, 5)),
        "p95": float(np.percentile(t, 95)),
        "n": int(len(t)),
    }


# ANSI escape sequences
RED = "\033[91m"
GREEN = "\033[92m"
RESET = "\033[0m"


def compare_and_print_results(results, baseline_key, func_type):
    stats = {k: summarize_timings(v) for k, v in results.items()}
    baseline = stats[baseline_key]["median"]

    # Header widths:
    # Name = 25 chars
    # Mean, p5, p95, Median, Δ Median = 13 chars each
    # Δ % = 10 chars
    header = (
        f"{func_type:25s}  "
        f"{'Mean':>13s}  "
        f"{'p5':>13s}  "
        f"{'p95':>13s}  "
        f"{'Median':>13s}  "
        f"{'Δ Median':>13s}  "
        f"{'Δ %':>10s}"
    )
    print(header)
    print("-" * len(header))

    for name, s in stats.items():
        median = s["median"]
        d_abs = median - baseline
        d_pct = (median / baseline - 1) * 100

        # format Δ % with colors
        pct_str = f"{d_pct:.2f}%"
        pct_fmt = f"{pct_str:>10}"  # align first

        if d_pct < 0:
            pct_colored = f"{GREEN}{pct_fmt}{RESET}"
        elif d_pct > 0:
            pct_colored = f"{RED}{pct_fmt}{RESET}"
        else:
            pct_colored = pct_fmt

        print(
            f"{name:25s}  "
            f"{s['mean']:10.3f} ms  "
            f"{s['p5']:10.3f} ms  "
            f"{s['p95']:10.3f} ms  "
            f"{median:10.3f} ms  "
            f"{d_abs:10.3f} ms  "
            f"{pct_colored:10s}"
        )

    print()

    # Return stats so callers can use them programmatically (e.g., to save JSON)
    return stats


def benchmark_on_cuda(
    modules: dict[str, torch.nn.Module],
    baseline: tuple[str, torch.nn.Module],
    tensor_sizes: list[int] = [1_000, 10_000, 100_000, 1_000_000, 10_000_000],
    mode: str = "SSS"
):
    """
    Generic benchmark function which takes some modules (here for the activation functions) and records the time
    it takes to apply the forward and backward functions each 100 times. On cuda-device.
    """


    parser = argparse.ArgumentParser(description="Plot PyTorch benchmark JSON: mean time vs tensor size")
    parser.add_argument("input", type=Path, help="Path for the benchmark JSON (from generic_benchmark.py)")
    args = parser.parse_args()

    device = torch.device("cuda")

    WARMUP_PASSES = 100
    MEASUREMENTS = 10
    PASSES_PER_MEASUREMENT = 100

    baseline_name, baseline_module = baseline
    baseline_name = f"{baseline_name} [Baseline]"
    modules[baseline_name] = baseline_module

    json_data = {}

    # Iterate over tensor sizes
    for size in tensor_sizes:
        batch_size = 64
        x = torch.randn(batch_size, size, device=device, requires_grad=True)
        a = torch.randn(1, device=device, requires_grad=True) # for xSSS

        title = f"| Tensor size ({batch_size}, {size:_}) |"
        print("-" * len(title))
        print(title)
        print("-" * len(title))
        print()

        all_forward_results = {}
        all_backward_results = {}

        for module_name in modules:
            activ_fn = modules[module_name].to(device)

            # Forward pass:
            # Warm-up
            for _ in range(WARMUP_PASSES):
                if mode == "SSS":
                    y = activ_fn(x)
                elif mode == "xSSS":
                    y = activ_fn(x, a)

            forward_passes_times = []

            # Take multiple measurements
            for _ in range(MEASUREMENTS):
                # Measure time for multiple forward passes
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(PASSES_PER_MEASUREMENT):
                    if mode == "SSS":
                        y = activ_fn(x)
                    elif mode == "xSSS":
                        y = activ_fn(x, a)
                end.record()
                torch.cuda.synchronize()
                forward_passes_times.append(start.elapsed_time(end))

            all_forward_results[module_name] = forward_passes_times

            # # Backward pass:
            loss = y.sum()

            # Warm-up
            for _ in range(WARMUP_PASSES):
                loss.backward(retain_graph=True)

            backward_passes_times = []

            # Take multiple measurements
            for _ in range(MEASUREMENTS):
                # Measure time for multiple backward passes
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(PASSES_PER_MEASUREMENT):
                    loss.backward(retain_graph=True)
                end.record()
                torch.cuda.synchronize()
                backward_passes_times.append(start.elapsed_time(end))

            all_backward_results[module_name] = backward_passes_times

        forward_stats = compare_and_print_results(
            all_forward_results,
            baseline_key=baseline_name,
            func_type=f"{MEASUREMENTS} x {PASSES_PER_MEASUREMENT} Forward passes",
        )

        backward_stats = compare_and_print_results(
            all_backward_results,
            baseline_key=baseline_name,
            func_type=f"{MEASUREMENTS} x {PASSES_PER_MEASUREMENT} Backward passes",
        )

        json_data[size] = {
            "forward": forward_stats,
            "backward": backward_stats
        }

    # If requested, save a JSON with raw timings and computed stats for this tensor size
    if args.output is not None:
        out_path = Path(args.output)

        # write back
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(json_data, f, indent=2)
        
        print(f"Wrote to {out_path}")

    return
