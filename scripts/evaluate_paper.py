"""
Unified paper-facing evaluation.

Reads a YAML config listing every model to evaluate. For each
(method, step_count, seed) combination:

  - Loads the model + checkpoint
  - Samples N_TEST_SAMPLES from a fixed noise seed (from src.eval.constants)
  - Compares against the locked test indices from data/test_indices.npy
  - Records pixel MSE, 1-Wasserstein, moment errors, NFE, wall-clock

All settings (seeds, step counts, test indices) are pulled from
src.eval.constants so results are reproducible across reruns.

Usage:
    # Evaluate everything in the config
    python scripts/evaluate_paper.py --gpu 0 --config config/paper_eval.yaml

    # Evaluate just one or two methods (names must match 'name' in config)
    python scripts/evaluate_paper.py --gpu 0 \\
        --only "CD-4step,MM-exp18 (mu=8, var=150)"

    # Quick test with fewer seeds
    python scripts/evaluate_paper.py --gpu 0 --only "Teacher" --seeds 0

Results append to results/eval_all.csv by default; safe to re-run
(each row is a single (method, step_count, seed) evaluation).
"""

import argparse
import csv
import glob
import os
import re
import time

import h5py
import numpy as np
import torch as th
import yaml
from scipy.stats import wasserstein_distance

from src.eval.constants import (
    DATA_PATH, DATA_SHAPE, N_TEST_SAMPLES, PAPER_SEEDS, SCHEDULE_S, STATS_DIR,
    load_test_indices,
)
from src.models.networks.unet.unet import UNetModelWrapper as UNetModel
from src.models.vp_diffusion import VPDiffusionModel
from src.models.consistency_models import MultistepConsistencyModel
from src.models.flow_models import MeanFlowMatching, RectifiedFlowMatching
from src.inference.samplers import MultistepCMSampler, MeanSampler, RectifiedFlowSampler
from src.models.diffusion_utils import ddim_step


UNET_CFG = dict(
    dim=list(DATA_SHAPE),
    channel_mult="1, 2, 4, 4",
    num_channels=64,
    num_res_blocks=2,
    num_head_channels=32,
    attention_resolutions="32",
    dropout=0.0,
    use_new_attention_order=True,
    use_scale_shift_norm=True,
    class_cond=False,
    num_classes=None,
)


def denormalize(samples, data_min, data_max):
    if isinstance(samples, th.Tensor):
        samples = samples.cpu().numpy()
    return (samples + 1.0) / 2.0 * (data_max - data_min) + data_min


def latest_checkpoint(save_dir):
    ckpts = glob.glob(os.path.join(save_dir, "checkpoint_*.pt"))
    if not ckpts:
        return None
    def epoch_num(p):
        m = re.search(r"checkpoint_(\d+)\.pt", p)
        return int(m.group(1)) if m else -1
    return max(ckpts, key=epoch_num)


def make_noise(seed, n_samples, shape):
    gen = th.Generator().manual_seed(int(seed))
    return th.randn(n_samples, *shape, generator=gen)


# ---------------------------------------------------------------------------
# Samplers per kind
# ---------------------------------------------------------------------------

@th.no_grad()
def sample_teacher(ckpt, n_steps, noise, device, batch_size=64):
    network = UNetModel(**UNET_CFG)
    model = VPDiffusionModel(network=network, schedule_s=SCHEDULE_S, infer=True)
    state = th.load(ckpt, map_location="cpu", weights_only=True)
    model.network.load_state_dict(state["model_state_dict"])
    model.to(device).eval()

    ts = th.linspace(1.0, 0.0, n_steps + 1, device=device)
    out = []
    for i in range(0, noise.shape[0], batch_size):
        z = noise[i:i+batch_size].to(device)
        n = z.shape[0]
        for step in range(n_steps):
            t = th.full((n,), ts[step].item(), device=device)
            s = th.full((n,), ts[step + 1].item(), device=device)
            x_hat = model.predict_x(z, t)
            z = ddim_step(x_hat, z, t, s, SCHEDULE_S)
        out.append(z.cpu())
    del model, network
    th.cuda.empty_cache()
    return th.cat(out, dim=0), n_steps


@th.no_grad()
def sample_cd(ckpt, student_steps, noise, device, batch_size=64):
    network = UNetModel(**UNET_CFG)
    model = MultistepConsistencyModel(
        network=network, student_steps=student_steps,
        schedule_s=SCHEDULE_S, infer=True,
    )
    state = th.load(ckpt, map_location="cpu", weights_only=True)
    model.network.load_state_dict(state["model_state_dict"])
    if "ema_state_dict" in state and model.ema_network is not None:
        model.ema_network.load_state_dict(state["ema_state_dict"])
    model.to(device).eval()

    sampler = MultistepCMSampler(model)
    out = []
    for i in range(0, noise.shape[0], batch_size):
        z = noise[i:i+batch_size].to(device)
        out.append(sampler.sample(z).cpu())
    del model, network
    th.cuda.empty_cache()
    return th.cat(out, dim=0), student_steps


@th.no_grad()
def sample_pd(ckpt, n_steps, noise, device, batch_size=64):
    network = UNetModel(**UNET_CFG)
    model = VPDiffusionModel(network=network, schedule_s=SCHEDULE_S, infer=True)
    state = th.load(ckpt, map_location="cpu", weights_only=True)
    model.network.load_state_dict(state["model_state_dict"])
    model.to(device).eval()

    ts = th.linspace(1.0, 0.0, n_steps + 1, device=device)
    out = []
    for i in range(0, noise.shape[0], batch_size):
        z = noise[i:i+batch_size].to(device)
        n = z.shape[0]
        for step in range(n_steps):
            t = th.full((n,), ts[step].item(), device=device).clamp(1e-4, 1 - 1e-4)
            s = th.full((n,), ts[step + 1].item(), device=device).clamp(0, 1 - 1e-4)
            x_hat = model.predict_x(z, t, use_ema=True)
            z = ddim_step(x_hat, z, t, s, SCHEDULE_S)
        out.append(z.cpu())
    del model, network
    th.cuda.empty_cache()
    return th.cat(out, dim=0), n_steps


@th.no_grad()
def sample_rf(ckpt, n_steps, noise, device, batch_size=64):
    network = UNetModel(**UNET_CFG)
    model = RectifiedFlowMatching(network=network, add_heavy_noise=False)
    state = th.load(ckpt, map_location="cpu", weights_only=True)
    model.network.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.infer = True
    model.network.eval()

    sampler = RectifiedFlowSampler(model)
    t_span_kwargs = {"start": 0, "end": 1, "steps": n_steps + 1}
    out = []
    for i in range(0, noise.shape[0], batch_size):
        z = noise[i:i+batch_size].to(device)
        out.append(sampler.sample(z, t_span_kwargs=t_span_kwargs).cpu())
    del model, network
    th.cuda.empty_cache()
    return th.cat(out, dim=0), n_steps


@th.no_grad()
def sample_mfm(ckpt, n_steps, noise, device, batch_size=64):
    cfg = dict(UNET_CFG)
    cfg["use_future_time_emb"] = True
    network = UNetModel(**cfg)
    model = MeanFlowMatching(network=network)
    state = th.load(ckpt, map_location="cpu", weights_only=True)
    model.network.load_state_dict(state["model_state_dict"])
    model.to(device)
    model.infer = True
    model.network.eval()

    sampler = MeanSampler(model)
    t_span_kwargs = {"start": 0, "end": 1, "steps": n_steps + 1}
    out = []
    for i in range(0, noise.shape[0], batch_size):
        z = noise[i:i+batch_size].to(device)
        out.append(sampler.sample(z, t_span_kwargs=t_span_kwargs).cpu())
    del model, network
    th.cuda.empty_cache()
    return th.cat(out, dim=0), n_steps


KIND_SAMPLERS = {
    "teacher": sample_teacher,
    "cd": sample_cd,
    "pd": sample_pd,
    "rf": sample_rf,
    "mfm": sample_mfm,
}


def compute_metrics(gen_denorm, real_denorm):
    gen_flat = gen_denorm.flatten()
    real_flat = real_denorm.flatten()
    cap = min(len(gen_flat), len(real_flat), 100_000)

    def skew(x):
        m = x.mean(); s = x.std()
        return float(((x - m) ** 3).mean() / (s ** 3 + 1e-12))

    def kurt(x):
        m = x.mean(); s = x.std()
        return float(((x - m) ** 4).mean() / (s ** 4 + 1e-12) - 3.0)

    return {
        "pixel_mse": float(((gen_denorm - real_denorm) ** 2).mean()),
        "wasserstein": float(wasserstein_distance(gen_flat[:cap], real_flat[:cap])),
        "mean_err": abs(float(gen_denorm.mean()) - float(real_denorm.mean())),
        "std_err": abs(float(gen_denorm.std()) - float(real_denorm.std())),
        "skew_err": abs(skew(gen_flat[:cap]) - skew(real_flat[:cap])),
        "kurt_err": abs(kurt(gen_flat[:cap]) - kurt(real_flat[:cap])),
    }


def load_real_data():
    data_min = np.load(os.path.join(STATS_DIR, "data_min.npy"))
    data_max = np.load(os.path.join(STATS_DIR, "data_max.npy"))
    test_idx = load_test_indices()[:N_TEST_SAMPLES]
    with h5py.File(DATA_PATH, "r") as f:
        data = f["tensor"][:]
    if data.ndim == 3:
        data = data[:, None]
    real_denorm = data[test_idx]
    return real_denorm, data_min, data_max


def resolve_ckpt(entry):
    if "ckpt" in entry and entry["ckpt"]:
        return entry["ckpt"]
    exp_dir = entry.get("exp_dir")
    if exp_dir:
        ckpt = latest_checkpoint(os.path.join(exp_dir, "saved_state"))
        if ckpt is None:
            raise FileNotFoundError(f"No checkpoint in {exp_dir}/saved_state")
        return ckpt
    raise ValueError(f"Entry {entry['name']} has neither 'ckpt' nor 'exp_dir'")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--gpu", type=int, default=0)
    p.add_argument("--config", type=str, default="config/paper_eval.yaml")
    p.add_argument("--output", type=str, default="results/eval_all.csv")
    p.add_argument("--only", type=str, default=None,
                   help="Comma-separated method names to run (subset of config)")
    p.add_argument("--seeds", type=int, nargs="+", default=None,
                   help="Override PAPER_SEEDS (for quick tests)")
    args = p.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    device = "cuda" if th.cuda.is_available() else "cpu"
    os.makedirs(os.path.dirname(args.output), exist_ok=True)

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    methods = cfg["methods"]
    if args.only:
        wanted = {s.strip() for s in args.only.split(",")}
        methods = [m for m in methods if m["name"] in wanted]
        if not methods:
            raise SystemExit(f"No methods matched --only={args.only}")

    seeds = args.seeds if args.seeds is not None else list(PAPER_SEEDS)
    real_denorm, data_min, data_max = load_real_data()

    new_file = not os.path.exists(args.output)
    out_f = open(args.output, "a", newline="")
    writer = csv.writer(out_f)
    if new_file:
        writer.writerow([
            "method", "kind", "ckpt", "step_count", "seed", "nfe",
            "pixel_mse", "wasserstein", "mean_err", "std_err",
            "skew_err", "kurt_err", "wall_clock_s", "notes",
        ])

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
            for seed in seeds:
                noise = make_noise(seed, N_TEST_SAMPLES, DATA_SHAPE)
                t0 = time.time()
                if kind == "cd":
                    samples, nfe = sampler(
                        ckpt, entry["student_steps"], noise, device,
                    )
                else:
                    samples, nfe = sampler(ckpt, n_steps, noise, device)
                wall = time.time() - t0

                gen_denorm = denormalize(samples, data_min, data_max)
                m = compute_metrics(gen_denorm, real_denorm)

                writer.writerow([
                    entry["name"], kind, ckpt, n_steps, seed, nfe,
                    m["pixel_mse"], m["wasserstein"], m["mean_err"],
                    m["std_err"], m["skew_err"], m["kurt_err"],
                    wall, entry.get("notes", ""),
                ])
                out_f.flush()
                print(f"  [{entry['name']}] steps={n_steps} seed={seed} "
                      f"WD={m['wasserstein']:.4f} "
                      f"MSE={m['pixel_mse']:.4f} ({wall:.1f}s)")

    out_f.close()
    print(f"\nDone. Results in {args.output}")


if __name__ == "__main__":
    main()
