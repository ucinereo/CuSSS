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
        plt.yscale("log")
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
    parser = argparse.ArgumentParser(description="Plot PyTorch benchmark JSON: mean time vs tensor size")
    parser.add_argument("input", type=Path, help="Path to benchmark JSON (from generic_benchmark.py)")
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=Path("benchmarks/results/plots/pytorch_mean_vs_size"),
        help="Output file path (pdf/svg extension optional). If no extension given, both .pdf and .svg are written.",
    )

    parser.add_argument(
        "--stat",
        default="mean",
        help="Which statistic to plot from the JSON (default: mean)",
    )
    parser.add_argument("--no-logx", action="store_true", help="Disable log scale on x axis")

    args = parser.parse_args()

    data = load_benchmark_json(args.input)
    if not data:
        raise SystemExit(f"No data found in {args.input}")

    out_forward = args.output / "forward"
    sizes, series = prepare_series(data, direction=out_forward, stat=args.stat)
    title = f"Forward mean time vs tensor size ({args.stat})"
    plot_mean_vs_size(sizes, series, out_forward, title=title, logx=not args.no_logx)

    print(f"Wrote forward plot to {out_forward} (or same name with .pdf/.svg)")

    out_backward = args.output / "backward"
    sizes, series = prepare_series(data, direction="backward", stat=args.stat)
    title = f"Backward mean time vs tensor size ({args.stat})"
    plot_mean_vs_size(sizes, series, out_backward, title=title, logx=not args.no_logx)

    print(f"Wrote forward plot to {out_backward} (or same name with .pdf/.svg)")


if __name__ == "__main__":
    main()
