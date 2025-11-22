import torch
import numpy as np


def summarize_timings(t):
    t = np.array(t, dtype=float)
    return {
        "mean": t.mean(),
        "std": t.std(),
        "median": np.median(t),
        "p5": np.percentile(t, 5),
        "p95": np.percentile(t, 95),
        "n": len(t),
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
        pct_fmt = f"{pct_str:>10}"   # align first

        if d_pct < 0:
            pct_colored = f"{GREEN}{pct_fmt}{RESET}"
        elif d_pct > 0:
            pct_colored = f"{RED}{pct_fmt}{RESET}"
        else:
            pct_colored = pct_fmt

        print(
            f"{name:35s}  "
            f"{s['mean']:10.3f} ms  "
            f"{s['p5']:10.3f} ms  "
            f"{s['p95']:10.3f} ms  "
            f"{median:10.3f} ms  "
            f"{d_abs:10.3f} ms  "
            f"{pct_colored:10s}"
        )

    print()

def benchmark_on_cuda(modules : dict[str, torch.nn.Module], baseline : tuple[str, torch.nn.Module], tensor_sizes : list[int] = [1_000, 10_000, 100_000, 1_000_000, 10_000_000], number_inputs: int = 1):
    """
    Generic benchmark function which takes some modules (here for the activation functions) and records the time 
    it takes to apply the forward and backward functions each 100 times. On cuda-device.
    """
    device = torch.device("cuda")

    WARMUP_PASSES = 100
    MEASUREMENTS = 10
    PASSES_PER_MEASUREMENT = 100

    baseline_name, baseline_module = baseline
    baseline_name = f"{baseline_name} [Baseline]"
    modules[baseline_name] = baseline_module

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
                if number_inputs == 2:
                    y = activ_fn(x, a)
                else:
                    y = activ_fn(x)

            forward_passes_times = []

            # Take multiple measurements
            for _ in range(MEASUREMENTS):
                # Measure time for multiple forward passes
                torch.cuda.synchronize()
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(PASSES_PER_MEASUREMENT):
                    if number_inputs == 2:
                        y = activ_fn(x, a)
                    else:
                        y = activ_fn(x)
                end.record()
                torch.cuda.synchronize()
                forward_passes_times.append(start.elapsed_time(end))

            all_forward_results[module_name] = forward_passes_times

            # Backward pass:
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

        compare_and_print_results(all_forward_results, baseline_key=baseline_name, func_type=f"{MEASUREMENTS} x {PASSES_PER_MEASUREMENT} Forward passes")
        compare_and_print_results(all_backward_results, baseline_key=baseline_name, func_type=f"{MEASUREMENTS} x {PASSES_PER_MEASUREMENT} Backward passes")

    return

