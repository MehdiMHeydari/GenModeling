"""
Locked evaluation constants for the paper.

All paper-facing eval, regeneration, and sampling must import from here so
results across methods are directly comparable. Changing any of these values
invalidates prior runs — do not change without re-running everything.
"""

import os

# ---------------------------------------------------------------------------
# Test set
# ---------------------------------------------------------------------------

# Path to the file holding 500 locked held-out Darcy indices.
# Generated once via scripts/lock_test_indices.py and committed.
TEST_INDICES_PATH = "data/test_indices.npy"

# Number of held-out Darcy samples used for every paper-facing evaluation.
# 1000 = use all available held-out samples (total 10k, 9k train). Matches
# DiffusionPDE (NeurIPS 2024) which reports metrics "averaged across 1,000
# random scenes and observations for each PDE".
N_TEST_SAMPLES = 1000

# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------

# Canonical teacher DDIM step count (see tickets/KNOWLEDGE.md for rationale).
# Confirmed via the teacher diagnostic sweep: 250 matches the GT marginal
# distribution near-perfectly, 75 undershoots the mode, Heun doesn't help.
TEACHER_DDIM_STEPS = 250

# Teacher checkpoint used across the project. Raw weights beat EMA;
# checkpoint_200 beats later checkpoints (raw weights drift after 200).
TEACHER_CKPT = "darcy_teacher/exp_1/saved_state/checkpoint_200.pt"

# Cosine schedule offset (matches teacher training config).
SCHEDULE_S = 0.008

# ---------------------------------------------------------------------------
# Seeds
# ---------------------------------------------------------------------------

# Noise seeds used for every eval run. Three seeds → mean ± std in tables.
# This follows the diffusion-paper convention of averaging across seeds.
PAPER_SEEDS = (0, 1, 2)

# Seed used when we need a deterministic single draw (e.g. precomputing
# teacher moments, which is a single one-shot artifact).
SINGLE_SEED = 0

# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

DATA_PATH = "data/2D_DarcyFlow_beta1.0_Train.hdf5"
DATA_SHAPE = (1, 128, 128)
STATS_DIR = "darcy_teacher/exp_1/saved_state"

# ---------------------------------------------------------------------------
# Per-dataset configs (multi-dataset eval support)
# ---------------------------------------------------------------------------
#
# Every key above is the Darcy default. The `_DATASETS` dict lets paper-facing
# scripts parametrize over dataset. Darcy values remain accessible as module
# constants for backward compatibility with pre-existing scripts.
#
# NS canonical teacher = ckpt 75 + DDIM 75. Decided from
# `diagnostics/ns_teacher_v3/`: ckpt 75 has WD 0.014 (best), ckpt 200 WD
# 0.033, training is non-monotone so latest is not best. DDIM 75 matches 250
# within 3% — no reason to pay 3.3× NFE.

_DATASETS = {
    "darcy": {
        "data_path": "data/2D_DarcyFlow_beta1.0_Train.hdf5",
        "data_shape": (1, 128, 128),
        "stats_dir": "darcy_teacher/exp_1/saved_state",
        "test_indices_path": "data/test_indices.npy",
        "teacher_ckpt": "darcy_teacher/exp_1/saved_state/checkpoint_200.pt",
        "teacher_ddim_steps": 250,
        "train_samples": 9000,
    },
    "ns": {
        "data_path": "data/ns_incom_128_merged.h5",
        "data_shape": (2, 128, 128),
        "stats_dir": "ns_teacher/exp_1/saved_state",
        "test_indices_path": "data/ns_test_indices.npy",
        "teacher_ckpt": "ns_teacher/exp_1/saved_state/checkpoint_75.pt",
        "teacher_ddim_steps": 75,
        "train_samples": 7000,
    },
    "rd": {
        # 2D Reaction-Diffusion from PDEBench (FitzHugh-Nagumo). 2 channels
        # (u, v concentrations), 128x128, ~10k frames after preprocessing.
        # Canonical teacher = ckpt 75 + DDIM 75. Same pattern as NS — early
        # checkpoint beats late ones (training is non-monotone). RD teacher
        # WD ~0.042 (weaker than Darcy 0.023 / NS 0.014); RD has sharp
        # pattern boundaries which diffusion models partially smooth.
        # See diagnostics/rd_teacher_v1/.
        "data_path": "data/rd_128_merged.h5",
        "data_shape": (2, 128, 128),
        "stats_dir": "rd_teacher/exp_1/saved_state",
        "test_indices_path": "data/rd_test_indices.npy",
        "teacher_ckpt": "rd_teacher/exp_1/saved_state/checkpoint_75.pt",
        "teacher_ddim_steps": 75,
        "train_samples": 9000,
    },
    "cns": {
        # 2D Compressible Navier-Stokes from PDEBench (2D_CFD_Rand M1.0
        # Eta0.01 Zeta0.01 periodic, 128x128). 4 channels: density, pressure,
        # Vx, Vy. Preprocessed to 10k single snapshots from 210k available
        # (10k trajectories x 21 timesteps). Random shuffled 9k/1k split
        # (NOT contiguous tail — fixes the NS temporal-OOD issue). Teacher
        # canonical checkpoint TBD, will be locked after diagnose_cns_teacher.
        #
        # norm="zscore": CNS channels span disparate scales with heavy tails
        # (pressure median 29 / max 557; velocity bulk ±0.5 / tails ±12), so
        # global per-channel min-max compresses the bulk into 7-33% of [-1,1]
        # (diagnostics/cns_mfm_v1/). We standardize per channel instead,
        # following The Well. Stats files: data_mean.npy / data_std.npy in
        # stats_dir (data_min/max.npy deliberately NOT written so legacy
        # min-max code fails loudly instead of silently mis-denormalizing).
        "data_path": "data/cns_128_merged.h5",
        "data_shape": (4, 128, 128),
        "stats_dir": "cns_teacher/exp_1/saved_state",
        "test_indices_path": "data/cns_test_indices.npy",
        "teacher_ckpt": "cns_teacher/exp_1/saved_state/checkpoint_75.pt",  # placeholder
        "teacher_ddim_steps": 75,
        "train_samples": 9000,
        "norm": "zscore",
    },
}


def get_dataset(name):
    """Fetch dataset-specific config dict."""
    if name not in _DATASETS:
        raise ValueError(
            f"Unknown dataset '{name}'. Options: {list(_DATASETS)}"
        )
    return _DATASETS[name]


def load_test_indices(dataset="darcy"):
    """Load the locked test indices for a dataset, erroring loudly if missing."""
    import numpy as np
    cfg = get_dataset(dataset)
    path = cfg["test_indices_path"]
    if not os.path.exists(path):
        suffix = "ns_" if dataset == "ns" else ""
        raise FileNotFoundError(
            f"Locked test indices for {dataset} not found at {path}. "
            f"Run scripts/lock_{suffix}test_indices.py first."
        )
    return np.load(path)
