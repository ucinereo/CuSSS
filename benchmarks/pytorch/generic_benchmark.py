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
    mode: str,
    out_filename: str,
    tensor_sizes: list[int] = [10_752, 21_504, 43_008], # 2nd and 3rd correspond to Apertus 8B, 70B
):
    """
    Generic benchmark function which takes some modules (here for the activation functions) and records the time
    it takes to apply the forward and backward functions each 100 times. On cuda-device.
    """

    device = torch.device("cuda")

    WARMUP_PASSES = 100
    MEASUREMENTS = 100
    PASSES_PER_MEASUREMENT = 10

    baseline_name, baseline_module = baseline
    baseline_name = f"{baseline_name} [Baseline]"
    modules[baseline_name] = baseline_module

    json_data = {}

    # Iterate over tensor sizes
    for size in tensor_sizes:
        size = (1, 4096, size)

        x = torch.randn(size, device=device, requires_grad=True)
        x2 = torch.randn(size, device=device, requires_grad=True)

        target = torch.randn(size, device=device, requires_grad=True)
        a = torch.randn(1, device=device, requires_grad=True) # for xSSS

        num_giga_bytes = x.nbytes / 1_000_000_000

        title = f"| Tensor size {size} |"
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
                elif mode == "SSSGLU":
                    y = activ_fn(x, x2)

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
                    elif mode == "SSSGLU":
                        y = activ_fn(x, x2)
                end.record()
                torch.cuda.synchronize()
                time_per_gigabyte = start.elapsed_time(end) / (num_giga_bytes * PASSES_PER_MEASUREMENT)
                forward_passes_times.append(time_per_gigabyte)

            all_forward_results[module_name] = forward_passes_times

            # # Backward pass:
            loss_fn = torch.nn.MSELoss()
            loss_fn2 = torch.nn.L1Loss()
            loss = loss_fn(y, target) + loss_fn2(y, target)

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
                time_per_gigabyte = start.elapsed_time(end) / (num_giga_bytes * PASSES_PER_MEASUREMENT)
                backward_passes_times.append(time_per_gigabyte)

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

        # Compute combined timings (element-wise sum of forward + backward measurements)
        combined_timings = {}
        # Only combine modules that have both forward and backward measurements
        for name in all_forward_results.keys():
            f_times = all_forward_results[name]
            b_times = all_backward_results[name]
            # Align lengths (use min length) and sum element-wise
            min_len = min(len(f_times), len(b_times))
            if min_len <= 0:
                continue
            combined = [float(f_times[i]) * 2 + float(b_times[i]) for i in range(min_len)]
            combined_timings[name] = combined

        # Print combined results to terminal (use baseline if available)
        if combined_timings:
            if baseline_name in combined_timings:
                baseline_for_combined = baseline_name
            else:
                # fallback to any available module as baseline and warn
                baseline_for_combined = next(iter(combined_timings))
                print(
                    f"Warning: baseline '{baseline_name}' has no combined timings; using '{baseline_for_combined}' as baseline for combined results."
                )

            combined_stats = compare_and_print_results(
                combined_timings,
                baseline_key=baseline_for_combined,
                func_type=f"{MEASUREMENTS} x {PASSES_PER_MEASUREMENT} Combined (2xF + 1xB) passes",
            )
        else:
            combined_stats = {}

        json_data[x.nbytes] = {
            "forward": forward_stats,
            "backward": backward_stats,
            "combined": combined_stats,
        }

    # If requested, save a JSON with raw timings and computed stats for this tensor size
    out_path = Path(f"benchmarks/results/pytorch_jsons/{out_filename}.json")

    # write back
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(json_data, f, indent=2)
    
    print(f"Wrote to {out_path}")

    return
