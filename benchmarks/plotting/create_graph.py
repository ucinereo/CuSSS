"""Create a line plot of mean execution time vs tensor size from a benchmark JSON.

Expected JSON format (produced by `generic_benchmark.py`):
{
  "1000": {"forward": {"ModuleA": {"mean": 0.1, ...}, ...}, "backward": {...}},
  "10000": { ... }
}

This script plots the chosen stat (default: mean) for each module across tensor sizes.
"""
from __future__ import annotations

import json
from pathlib import Path
import argparse
import math
import matplotlib.pyplot as plt
from typing import Dict, Any


def load_benchmark_json(path: Path) -> Dict[int, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # JSON keys may be strings; convert to ints where possible
    parsed: Dict[int, Any] = {}
    for k, v in data.items():
        try:
            key = int(k)
        except Exception:
            # If key is non-numeric (e.g. contains comma), try splitting
            try:
                key = int(k.split(",")[-1])
            except Exception:
                # skip unknown key
                continue
        parsed[key] = v
    return parsed


def prepare_series(data: Dict[int, Any], direction: str = "forward", stat: str = "mean"):
    # sort sizes
    sizes = sorted(data.keys())

    # gather module names across sizes
    module_names = set()
    for s in sizes:
        block = data[s]
        dir_block = block.get(direction, {})
        module_names.update(dir_block.keys())

    # prepare mapping module -> list of stat values aligned with sizes
    series: Dict[str, list[float | None]] = {m: [] for m in sorted(module_names)}
    for s in sizes:
        block = data[s]
        dir_block = block.get(direction, {})
        for m in series:
            stats = dir_block.get(m)
            if stats is None:
                series[m].append(None)
            else:
                val = stats.get(stat)
                series[m].append(float(val) if val is not None else None)

    return sizes, series


def prepare_combined_series(
    data: Dict[int, Any], f_mult: float = 2.0, b_mult: float = 1.0, stat: str = "mean"
):
    """Prepare a series that is a linear combination of forward and backward stats.

    For each module and size the value is computed as: f_mult * forward[stat] + b_mult * backward[stat].
    If either forward or backward stat is missing for a module at a size, the combined value is None
    (so matplotlib will show a gap).
    """
    sizes = sorted(data.keys())

    # gather module names across forward and backward
    module_names = set()
    for s in sizes:
        block = data[s]
        module_names.update(block.get("forward", {}).keys())
        module_names.update(block.get("backward", {}).keys())

    series: Dict[str, list[float | None]] = {m: [] for m in sorted(module_names)}
    for s in sizes:
        block = data[s]
        f_block = block.get("forward", {})
        b_block = block.get("backward", {})
        for m in series:
            f_stats = f_block.get(m)
            b_stats = b_block.get(m)
            if f_stats is None or b_stats is None:
                series[m].append(None)
            else:
                fval = f_stats.get(stat)
                bval = b_stats.get(stat)
                if fval is None or bval is None:
                    series[m].append(None)
                else:
                    try:
                        combined = float(f_mult * float(fval) + b_mult * float(bval))
                    except Exception:
                        combined = None
                    series[m].append(combined)

    return sizes, series


def plot_mean_vs_size(
    sizes: list[int],
    series: Dict[str, list[float | None]],
    out_path: Path,
    title: str | None = None,
    logx: bool = True,
):
    plt.figure(figsize=(8, 5))
    for name, vals in series.items():
        # replace None with nan so matplotlib can handle gaps
        y = [float(v) if v is not None else math.nan for v in vals]
        plt.plot(sizes, y, marker="o", label=name)

    plt.xlabel("Tensor size (features)")
    plt.ylabel("Mean time (ms)")
    if title:
        plt.title(title)
    plt.grid(True, which="both", ls="--", lw=0.5)
    if logx:
        plt.xscale("log")
    plt.legend(loc="best", fontsize="small")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Decide extension(s) to save
    if out_path.suffix.lower() in [".pdf", ".svg"]:
        plt.savefig(out_path, bbox_inches="tight")
    else:
        # save both pdf and svg with provided base name
        pdf = out_path.with_suffix(".pdf")
        svg = out_path.with_suffix(".svg")
        plt.savefig(pdf, bbox_inches="tight")
        plt.savefig(svg, bbox_inches="tight")

    plt.close()


def main() -> None:
    input_path = Path("benchmarks/results/pytorch.json")
    output_path = Path("benchmarks/results/plots/graph_pytorch/")

    
    data = load_benchmark_json(input_path)
    if not data:
        raise SystemExit(f"No data found in {input_path}")

    out_forward = output_path / "forward"
    sizes, series = prepare_series(data, direction="forward")
    title = f"Forward mean time vs tensor size"
    plot_mean_vs_size(sizes, series, out_forward, title=title, logx=True)

    print(f"Wrote forward plot to {out_forward} (or same name with .pdf/.svg)")

    out_backward = output_path / "backward"
    sizes, series = prepare_series(data, direction="backward")
    title = f"Backward mean time vs tensor size"
    plot_mean_vs_size(sizes, series, out_backward, title=title, logx=True)

    print(f"Wrote forward plot to {out_backward} (or same name with .pdf/.svg)")

    sizes, series = prepare_series(data, direction="combined")
    out_combined = output_path / "combined"
    title = f"Combined (2x forward + 1x backward) mean time vs tensor size"
    plot_mean_vs_size(sizes, series, out_combined, title=title, logx=True)  
    print(f"Wrote combined plot to {out_combined} (or same name with .pdf/.svg)")

if __name__ == "__main__":
    main()
