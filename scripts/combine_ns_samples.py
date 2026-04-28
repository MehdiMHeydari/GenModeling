"""
Tile per-method NS sample grids into a single combined image.

Reads the PNGs that scripts/generate_ns_samples.py wrote and stacks them
vertically — one row per (method, step_count) combo, ordered to match
config/ns_paper_eval.yaml. Mirrors scripts/summarize_ns_diag.py.

Usage:
    python scripts/combine_ns_samples.py
    python scripts/combine_ns_samples.py --input_dir paper_figures/ns_samples \\
        --output paper_figures/ns_samples/combined.png
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import yaml


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, default="paper_figures/ns_samples")
    p.add_argument("--config", type=str, default="config/ns_paper_eval.yaml")
    p.add_argument("--output", type=str, default=None,
                   help="default: <input_dir>/combined.png")
    args = p.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    rows = []
    for entry in cfg["methods"]:
        step_counts = entry.get("step_counts", [entry.get("student_steps")])
        for n_steps in step_counts:
            tag = f"{entry['name']}_steps{n_steps}"
            png = os.path.join(args.input_dir, f"{tag}.png")
            if os.path.exists(png):
                rows.append((tag, png))
            else:
                print(f"[skip] missing {png}")

    if not rows:
        raise SystemExit(f"No PNGs found in {args.input_dir}")

    n = len(rows)
    fig, axes = plt.subplots(n, 1, figsize=(15, 3.0 * n), squeeze=False)
    for i, (tag, png) in enumerate(rows):
        ax = axes[i, 0]
        ax.axis("off")
        ax.imshow(mpimg.imread(png))
        ax.set_title(tag, fontsize=10, loc="left")

    fig.tight_layout()
    out = args.output or os.path.join(args.input_dir, "combined.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}  ({n} method-rows)")


if __name__ == "__main__":
    main()
