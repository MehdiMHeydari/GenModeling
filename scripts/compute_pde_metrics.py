"""
PDE-aware metrics for the CD-vs-RF side paper.

Two metrics, one per dataset:
    - NS: mean abs divergence of the velocity field. True incompressible NS has
          div(v) = 0; if generated samples violate this, they are physically
          inconsistent regardless of how good they look.
    - Darcy: L1 distance between the radially-averaged power spectrum of
          generated and ground-truth solutions. Elliptic PDEs have a
          characteristic spectral decay; matching the spectrum is a PDE-aware
          consistency check that doesn't require knowing the coefficient field.

Re-uses the model loaders and samplers from evaluate_paper.py so the set of
methods stays in lockstep with the main eval. Runs one seed (SINGLE_SEED) since
PDE metrics are deterministic averages, not distribution tails.

Outputs:
    results/pde_metrics.csv  (dataset, method, nfe, pde_metric, metric_name)

Usage (on the GPU server):
    python scripts/compute_pde_metrics.py --gpu 0 --dataset ns
    python scripts/compute_pde_metrics.py --gpu 0 --dataset darcy
"""

import argparse
import csv
import os

import h5py
import numpy as np
import torch as th

from src.eval.constants import N_TEST_SAMPLES, SINGLE_SEED, get_dataset, load_test_indices
from scripts.evaluate_paper import (
    KIND_SAMPLERS, denormalize, make_noise, make_unet_cfg, resolve_ckpt,
)
import yaml


def ns_divergence(samples):
    """Mean |div(v)| over samples. samples: (N, 2, H, W) with channels (vx, vy).
    Central differences with periodic boundary (PDEBench NS is periodic)."""
    vx = samples[:, 0]
    vy = samples[:, 1]
    dvx_dx = (np.roll(vx, -1, axis=2) - np.roll(vx, 1, axis=2)) / 2.0
    dvy_dy = (np.roll(vy, -1, axis=1) - np.roll(vy, 1, axis=1)) / 2.0
    div = dvx_dx + dvy_dy
    return float(np.mean(np.abs(div)))


def radial_power_spectrum(field):
    """Radially-averaged 2D power spectrum of a single H x W field."""
    F = np.fft.fft2(field)
    P = np.abs(F) ** 2
    h, w = field.shape
    cy, cx = h // 2, w // 2
    P = np.fft.fftshift(P)
    y, x = np.indices((h, w))
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)
    r_max = min(cy, cx)
    radial = np.bincount(r.ravel(), P.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    return radial[:r_max]


def darcy_spectral_l1(gen_samples, real_samples):
    """Mean L1 distance between gen and real radially-averaged power spectra.
    Both inputs (N, 1, H, W). Spectra are averaged over samples, then compared."""
    gen_field = gen_samples[:, 0]
    real_field = real_samples[:, 0]
    gen_spec = np.mean([radial_power_spectrum(f) for f in gen_field], axis=0)
    real_spec = np.mean([radial_power_spectrum(f) for f in real_field], axis=0)
    # Normalize so the L1 isn't dominated by overall energy scale
    gen_spec = gen_spec / (gen_spec.sum() + 1e-12)
    real_spec = real_spec / (real_spec.sum() + 1e-12)
    return float(np.mean(np.abs(gen_spec - real_spec)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--config", type=str, default=None,
                   help="Defaults to config/paper_eval_<dataset>.yaml")
    p.add_argument("--dataset", type=str, required=True, choices=["darcy", "ns"])
    p.add_argument("--output", type=str, default="results/pde_metrics.csv")
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated method names to run")
    args = p.parse_args()

    if args.config:
        config_path = args.config
    elif args.dataset == "ns":
        config_path = "config/ns_paper_eval.yaml"
    else:
        config_path = "config/paper_eval.yaml"
    print(f"Using config: {config_path}")

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if th.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    methods = cfg["methods"]
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        methods = [m for m in methods if m["name"] in wanted]
        if not methods:
            raise SystemExit(f"No methods matched --only={args.only}")

    ds = get_dataset(args.dataset)
    data_shape = ds["data_shape"]
    unet_cfg = make_unet_cfg(data_shape)
    print(f"Dataset: {args.dataset}  data_shape={data_shape}")

    data_min = np.load(os.path.join(ds["stats_dir"], "data_min.npy"))
    data_max = np.load(os.path.join(ds["stats_dir"], "data_max.npy"))
    test_idx = load_test_indices(args.dataset)[:N_TEST_SAMPLES]
    with h5py.File(ds["data_path"], "r") as f:
        data = f["tensor"][:]
    if data.ndim == 3:
        data = data[:, None]
    real_denorm = data[test_idx]

    metric_name = "ns_divergence" if args.dataset == "ns" else "darcy_spectral_l1"
    if args.dataset == "ns":
        real_metric = ns_divergence(real_denorm)
    else:
        real_metric = 0.0  # Spectral L1 is gen-vs-real; real-vs-real is trivially 0
    print(f"Reference (real) {metric_name} = {real_metric:.6f}")

    header = ["dataset", "method", "kind", "nfe", "pde_metric", "metric_name"]
    new_file = not os.path.exists(args.output)
    out_f = open(args.output, "a", newline="")
    writer = csv.writer(out_f)
    if new_file:
        writer.writerow(header)
        # Log the real-data reference so the table is self-documenting
        writer.writerow([args.dataset, "REAL", "real", 0, real_metric, metric_name])
        out_f.flush()

    for entry in methods:
        kind = entry["kind"]
        if kind not in KIND_SAMPLERS:
            print(f"[skip] unknown kind: {kind} ({entry['name']})")
            continue
        try:
            ckpt = resolve_ckpt(entry)
        except FileNotFoundError as e:
            print(f"[skip] {entry['name']}: {e}")
            continue

        sampler = KIND_SAMPLERS[kind]
        step_counts = entry.get("step_counts", [entry.get("student_steps")])

        for n_steps in step_counts:
            noise = make_noise(SINGLE_SEED, N_TEST_SAMPLES, data_shape)
            if kind == "cd":
                samples, nfe = sampler(ckpt, entry["student_steps"], noise, device, unet_cfg)
            else:
                samples, nfe = sampler(ckpt, n_steps, noise, device, unet_cfg)
            gen_denorm = denormalize(samples, data_min, data_max)

            if args.dataset == "ns":
                metric = ns_divergence(gen_denorm)
            else:
                metric = darcy_spectral_l1(gen_denorm, real_denorm)

            writer.writerow([args.dataset, entry["name"], kind, nfe, metric, metric_name])
            out_f.flush()
            print(f"  [{entry['name']}] steps={n_steps}  {metric_name}={metric:.6f}")

    out_f.close()
    print(f"\nDone. Results in {args.output}")


if __name__ == "__main__":
    main()
