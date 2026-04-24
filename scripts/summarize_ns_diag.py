"""
Tile all NS teacher diagnostic PNGs into one summary image.

Reads an output dir produced by diagnose_ns_teacher.py and composes a
single summary image with one row per (ckpt, sampler, n_steps) combo
from the CSV, and four columns: sample grid, |v| hist, Vx hist, Vy hist.

Usage:
    python scripts/summarize_ns_diag.py --input_dir diagnostics/ns_teacher_v3
    python scripts/summarize_ns_diag.py --input_dir diagnostics/ns_teacher_v3 \\
        --output summary.png
"""

import argparse
import csv
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


COLS = ["grid", "hist", "hist_vxvy"]
COL_TITLES = ["Samples (GT top, Gen bottom)", "|v| histogram", "Vx / Vy histograms"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input_dir", type=str, required=True)
    p.add_argument("--output", type=str, default=None,
                   help="Output image path (default: <input_dir>/summary.png)")
    args = p.parse_args()

    csv_path = os.path.join(args.input_dir, "ns_teacher_diag.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")

    rows = []
    with open(csv_path) as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(r)
    if not rows:
        raise RuntimeError("CSV has no rows")

    n_rows = len(rows)
    n_cols = len(COLS)
    fig, axes = plt.subplots(n_rows, n_cols,
                             figsize=(5.5 * n_cols, 2.3 * n_rows),
                             squeeze=False)

    for i, row in enumerate(rows):
        tag = f"ckpt{row['ckpt_epoch']}_{row['sampler']}_{row['n_steps']}steps"
        label = (f"ckpt {row['ckpt_epoch']}  {row['sampler'].upper()} "
                 f"{row['n_steps']}s  |  WD {float(row['wasserstein_mag']):.4f}  "
                 f"MSE {float(row['pixel_mse']):.4f}")

        for j, col in enumerate(COLS):
            png_path = os.path.join(args.input_dir, f"{tag}_{col}.png")
            ax = axes[i, j]
            ax.axis("off")
            if os.path.exists(png_path):
                img = mpimg.imread(png_path)
                ax.imshow(img)
            else:
                ax.text(0.5, 0.5, f"missing\n{os.path.basename(png_path)}",
                        ha="center", va="center")
            if i == 0:
                ax.set_title(COL_TITLES[j], fontsize=11)
            if j == 0:
                ax.set_ylabel(label, fontsize=9, rotation=0,
                              labelpad=110, ha="right", va="center")

    fig.suptitle(os.path.basename(os.path.normpath(args.input_dir)),
                 fontsize=13, y=1.0)
    fig.tight_layout()

    out = args.output or os.path.join(args.input_dir, "summary.png")
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote {out}")

    # Side-by-side comparison strips per plot type.
    for col in ["hist", "hist_vxvy", "grid"]:
        fig2, axes2 = plt.subplots(n_rows, 1,
                                   figsize=(11, 3.5 * n_rows),
                                   squeeze=False)
        for i, row in enumerate(rows):
            tag = f"ckpt{row['ckpt_epoch']}_{row['sampler']}_{row['n_steps']}steps"
            label = (f"ckpt {row['ckpt_epoch']}  |  WD {float(row['wasserstein_mag']):.4f}"
                     f"  |  MSE {float(row['pixel_mse']):.4f}")
            png_path = os.path.join(args.input_dir, f"{tag}_{col}.png")
            ax = axes2[i, 0]
            ax.axis("off")
            if os.path.exists(png_path):
                img = mpimg.imread(png_path)
                ax.imshow(img)
            ax.set_title(label, fontsize=10, loc="left")
        fig2.tight_layout()
        strip_out = os.path.join(args.input_dir, f"compare_{col}.png")
        fig2.savefig(strip_out, dpi=110, bbox_inches="tight")
        plt.close(fig2)
        print(f"Wrote {strip_out}")


if __name__ == "__main__":
    main()
