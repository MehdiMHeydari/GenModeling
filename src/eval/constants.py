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


def load_test_indices():
    """Load the locked test indices, erroring loudly if missing."""
    import numpy as np
    if not os.path.exists(TEST_INDICES_PATH):
        raise FileNotFoundError(
            f"Locked test indices not found at {TEST_INDICES_PATH}. "
            f"Run scripts/lock_test_indices.py first."
        )
    return np.load(TEST_INDICES_PATH)
