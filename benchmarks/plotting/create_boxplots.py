import json
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.ticker import FuncFormatter
from collections import defaultdict
import os
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
import math
import sys


# Creates a multipage-pdf with a page for each tensor-size, in each page graphs for each block-size, with different kernels side-by-side in the same graph
def boxplots_cuda(device_name):
    # -----------------------------
    # Load JSON with results
    # -----------------------------

    base = os.path.dirname(os.path.abspath(__file__))
    device_specs_path = os.path.join(base, "../device_specs/gpus.json")
    cuda_results_path = os.path.join(base, "../results/cuda.json")

    # Check if the file exists
    if os.path.exists(cuda_results_path):
        print("CUDA results: Exist")
        with open(cuda_results_path, "r") as f:
            data = json.load(f)["results"]
    else:
        print("CUDA results: Don't exist")
        return

    device_bandwidth_bytes_s = None
    with open(device_specs_path, "r") as f:
        devices = json.load(f)
        for gpu in devices:
            if gpu["device"] == device_name:
                device_bandwidth_bytes_s = gpu["memory_bandwith_gb_s"] * 1e9

    if device_bandwidth_bytes_s is None:
        raise Exception(f"Device not found in {device_specs_path}")

    # --------------------------------------------------------------
    # First: reorganize everything globally by tensor_size
    # structure: tensor_size -> block_size -> kernel -> list of meas
    # --------------------------------------------------------------

    output_path = os.path.join(base, "../results/cuda_plots/boxplots_cuda.pdf")
    with PdfPages(output_path) as pdf:
        tensor_groups = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))

        for r in data:
            kernel = r["kernel"]
            tsize = r["tensor_size_bytes"]
            block = r["block_size"]
            meas = r["measurements_ms"]  # list of lists

            # flatten each measurement list (they are list of lists)
            flat = meas
            tensor_groups[tsize][block][kernel].extend(flat)

        # --------------------------------------------------------------
        # Now create ONE PAGE PER TENSOR SIZE
        # --------------------------------------------------------------

        for tensor_bytes, blocks in tensor_groups.items():
            # human-readable tensor size
            mb = tensor_bytes / (1024**2)
            size_str = f"{mb / 1024:.2f}GB" if mb >= 1024 else f"{mb:.1f}MB"

            sorted_blocks = sorted(blocks.keys())
            all_kernels = sorted({k for b in blocks.values() for k in b.keys()})

            n_blocks = len(sorted_blocks)
            blocks_per_page = 4
            n_pages = math.ceil(n_blocks / blocks_per_page)

            tensor_bytes_current = tensor_bytes

            # transforms
            def ms_to_util(ms):
                ms = np.asarray(ms)
                t = ms / 1000.0
                BW = np.zeros_like(t)
                np.divide(tensor_bytes_current, t, out=BW, where=t > 0)
                util = BW / device_bandwidth_bytes_s
                return util * 100.0

            def util_to_ms(util_percent):
                util = np.asarray(util_percent) / 100.0
                BW = util * device_bandwidth_bytes_s
                t = np.zeros_like(BW)
                np.divide(tensor_bytes_current, BW, out=t, where=BW > 0)
                return t * 1000.0

            # --- Loop over pages ---
            for page_idx in range(n_pages):
                # block indices for this page
                start = page_idx * blocks_per_page
                end = min(start + blocks_per_page, n_blocks)
                page_blocks = sorted_blocks[start:end]

                # create 2×2 grid (even if fewer graphs on last page)
                fig, axes = plt.subplots(2, 2, figsize=(12, 10))
                fig.suptitle(f"{device_name} - Tensor {size_str}", fontsize=16)

                axes = axes.flatten()  # flatten 2x2 → list of 4 axes

                for ax, block in zip(axes, page_blocks):
                    plot_data = []
                    labels = []

                    for kernel in all_kernels:
                        meas = blocks[block].get(kernel, [])
                        plot_data.append(meas)
                        labels.append(kernel)

                    ax.boxplot(plot_data, vert=True, patch_artist=True)
                    ax.set_xticks(range(1, len(labels) + 1))
                    ax.set_xticklabels(labels)
                    ax.set_title(f"Block size {block}")
                    ax.set_ylabel("Time [ms]")
                    ax.grid(True, linestyle="--", alpha=0.5)

                    secax = ax.secondary_yaxis(
                        "right", functions=(ms_to_util, util_to_ms)
                    )
                    secax.set_ylabel("GPU Utilization [%]")

                    fig.canvas.draw()

                    # color gradient from red → green
                    cmap = plt.get_cmap("RdYlGn")
                    norm = mcolors.Normalize(vmin=0, vmax=100)

                    # formatter: just show percent
                    secax.yaxis.set_major_formatter(
                        FuncFormatter(lambda val, pos: f"{val:.0f}%")
                    )

                    # now color the tick labels
                    for tick in secax.get_yticklabels():
                        try:
                            val = float(tick.get_text().strip("%"))
                            tick.set_color(cmap(norm(val)))
                        except ValueError:
                            pass

                # hide empty subplots (if fewer than 4 blocks on last page)
                for remaining_ax in axes[len(page_blocks) :]:
                    remaining_ax.set_visible(False)

                plt.tight_layout(rect=(0, 0, 1, 0.96))
                svg_path = os.path.join(
                    base,
                    f"../results/cuda_plots/svgs/boxplot_{size_str.replace('.', '_')}.svg",
                )
                fig.savefig(svg_path)
                pdf.savefig(fig)
                plt.close(fig)


if __name__ == "__main__":
    args = sys.argv
    if len(args) < 2:
        raise Exception("No GPU name specified")
    boxplots_cuda(args[1])
