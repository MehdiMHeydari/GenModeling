"""
Evaluate CD student samples vs teacher vs real data.
Saves plots as PNG files and prints statistics.

Usage:
    python scripts/evaluate.py
"""

import os
import sys
import numpy as np
import h5py
import torch as th
import matplotlib
matplotlib.use('Agg')  # No display needed on server
import matplotlib.pyplot as plt

from src.models.networks.unet.unet import UNetModelWrapper as UNetModel
from src.models.vp_diffusion import VPDiffusionModel
from src.models.consistency_models import MultistepConsistencyModel
from src.inference.samplers import MultistepCMSampler
from src.models.diffusion_utils import addim_step, snr

# ============================================================
# CONFIG — adjust these if your paths differ
# ============================================================
DATA_SHAPE = (1, 128, 128)
DATA_PATH = "data/2D_DarcyFlow_beta1.0_Train.hdf5"
TEACHER_SAVE_DIR = "darcy_teacher/exp_1/saved_state"
CD_SAVE_DIR = "darcy_student/exp_1/saved_state"
SCHEDULE_S = 0.008
STUDENT_STEPS = 4
DEVICE = "cuda"

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

NUM_SAMPLES = 16
BATCH_SIZE = 8
DDIM_STEPS = 50

OUTPUT_DIR = "eval_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# FIND BEST CHECKPOINT
# ============================================================
cd_ckpts = sorted(
    [f for f in os.listdir(CD_SAVE_DIR)
     if f.startswith('checkpoint_') and f.endswith('.pt')],
    key=lambda f: int(f.split('_')[1].split('.')[0])  # sort by epoch number
)
print("Available CD checkpoints:", cd_ckpts)
CD_CKPT = os.path.join(CD_SAVE_DIR, cd_ckpts[-1])
print(f"Using: {CD_CKPT}")

TEACHER_CKPT = os.path.join(TEACHER_SAVE_DIR, "checkpoint_175.pt")
print(f"Teacher: {TEACHER_CKPT}")

# ============================================================
# LOAD REAL DATA
# ============================================================
print("\n--- Loading real data ---")
with h5py.File(DATA_PATH, 'r') as f:
    outputs = np.array(f['tensor']).astype(np.float32)
if outputs.ndim == 3:
    outputs = outputs[:, np.newaxis, :, :]

# Norm stats saved by whichever training script ran with save_dir
# Check both teacher and student dirs
for stats_dir in [TEACHER_SAVE_DIR, CD_SAVE_DIR]:
    if os.path.exists(os.path.join(stats_dir, "data_min.npy")):
        break
data_min = float(np.load(os.path.join(stats_dir, "data_min.npy")))
data_max = float(np.load(os.path.join(stats_dir, "data_max.npy")))
print(f"Loaded norm stats from {stats_dir}")


def denormalize(x_norm):
    if isinstance(x_norm, th.Tensor):
        x_norm = x_norm.numpy()
    return (x_norm + 1.0) / 2.0 * (data_max - data_min) + data_min


real_norm = 2.0 * (outputs - data_min) / (data_max - data_min) - 1.0
test_data = real_norm[9500:]  # test set
print(f"Test data: {test_data.shape[0]} samples")

# ============================================================
# LOAD & SAMPLE FROM CD STUDENT
# ============================================================
print("\n--- Loading CD student ---")
network = UNetModel(**UNET_CFG)
cm = MultistepConsistencyModel(
    network=network,
    student_steps=STUDENT_STEPS,
    schedule_s=SCHEDULE_S,
    infer=True,
)
state = th.load(CD_CKPT, map_location='cpu', weights_only=True)
cm.network.load_state_dict(state['model_state_dict'])
if 'ema_state_dict' in state:
    cm.ema_network.load_state_dict(state['ema_state_dict'])
    print("Loaded EMA weights")
cm.to(DEVICE)
cm.eval()
print(f"Loaded student from epoch {state['epoch']}")

print(f"\nGenerating {NUM_SAMPLES} CM samples ({STUDENT_STEPS} steps)...")
C, H, W = DATA_SHAPE
sampler = MultistepCMSampler(cm)
rounds = (NUM_SAMPLES + BATCH_SIZE - 1) // BATCH_SIZE

th.manual_seed(42)
cm_noise = [th.randn(min(BATCH_SIZE, NUM_SAMPLES - r * BATCH_SIZE), C, H, W)
            for r in range(rounds)]

all_samples = []
with th.no_grad():
    for r in range(rounds):
        z = cm_noise[r].to(DEVICE)
        samples = sampler.sample(z)
        all_samples.append(samples.cpu())
cm_samples = th.cat(all_samples, dim=0)[:NUM_SAMPLES]
print(f"CM samples done. Range: [{cm_samples.min():.4f}, {cm_samples.max():.4f}]")

# Free GPU memory
del cm, network
th.cuda.empty_cache()

# ============================================================
# LOAD & SAMPLE FROM TEACHER (aDDIM)
# ============================================================
print(f"\nGenerating {NUM_SAMPLES} teacher samples ({DDIM_STEPS} aDDIM steps)...")
teacher_net = UNetModel(**UNET_CFG)
teacher = VPDiffusionModel(network=teacher_net, schedule_s=SCHEDULE_S, infer=True)
t_state = th.load(TEACHER_CKPT, map_location='cpu', weights_only=True)
teacher.network.load_state_dict(t_state['model_state_dict'])
teacher.to(DEVICE)
teacher.eval()

ts = th.linspace(1.0, 0.0, DDIM_STEPS + 1, device=DEVICE)
teacher_samples = []

with th.no_grad():
    for r in range(rounds):
        z = cm_noise[r].to(DEVICE)  # Same noise as CM
        n = z.shape[0]
        for i in range(DDIM_STEPS):
            t_batch = th.full((n,), ts[i].item(), device=DEVICE)
            s_batch = th.full((n,), ts[i + 1].item(), device=DEVICE)
            x_hat = teacher.predict_x(z, t_batch)
            x_var = 0.1 / (2.0 + snr(t_batch, SCHEDULE_S))
            z = addim_step(x_hat, z, x_var, t_batch, s_batch, SCHEDULE_S)
        teacher_samples.append(z.cpu())

teacher_samples = th.cat(teacher_samples, dim=0)[:NUM_SAMPLES]
print(f"Teacher samples done. Range: [{teacher_samples.min():.4f}, {teacher_samples.max():.4f}]")

del teacher, teacher_net
th.cuda.empty_cache()

# ============================================================
# DENORMALIZE
# ============================================================
cm_denorm = denormalize(cm_samples)
teacher_denorm = denormalize(teacher_samples)
real_denorm = denormalize(test_data)

# ============================================================
# STATISTICS
# ============================================================
print("\n" + "=" * 60)
print("STATISTICS COMPARISON (physical units)")
print("=" * 60)
print(f"{'':20s} {'CM (4-step)':>12s} {'Teacher (50)':>12s} {'Real (test)':>12s}")
print("-" * 60)
print(f"{'Mean':20s} {cm_denorm.mean():12.6f} {teacher_denorm.mean():12.6f} {real_denorm.mean():12.6f}")
print(f"{'Std':20s} {cm_denorm.std():12.6f} {teacher_denorm.std():12.6f} {real_denorm.std():12.6f}")
print(f"{'Min':20s} {cm_denorm.min():12.6f} {teacher_denorm.min():12.6f} {real_denorm.min():12.6f}")
print(f"{'Max':20s} {cm_denorm.max():12.6f} {teacher_denorm.max():12.6f} {real_denorm.max():12.6f}")

cm_means = cm_denorm.mean(axis=(1, 2, 3))
t_means = teacher_denorm.mean(axis=(1, 2, 3))
r_means = real_denorm.mean(axis=(1, 2, 3))
cm_stds = cm_denorm.std(axis=(1, 2, 3))
t_stds = teacher_denorm.std(axis=(1, 2, 3))
r_stds = real_denorm.std(axis=(1, 2, 3))

print(f"\nPer-sample mean:")
print(f"  CM:       {cm_means.mean():.6f} +/- {cm_means.std():.6f}")
print(f"  Teacher:  {t_means.mean():.6f} +/- {t_means.std():.6f}")
print(f"  Real:     {r_means.mean():.6f} +/- {r_means.std():.6f}")

print(f"\nPer-sample std (spatial structure):")
print(f"  CM:       {cm_stds.mean():.6f} +/- {cm_stds.std():.6f}")
print(f"  Teacher:  {t_stds.mean():.6f} +/- {t_stds.std():.6f}")
print(f"  Real:     {r_stds.mean():.6f} +/- {r_stds.std():.6f}")

# ============================================================
# PLOT 1: 3-row comparison (CM vs Teacher vs Real)
# ============================================================
n_show = 4
fig, axes = plt.subplots(3, n_show, figsize=(4 * n_show, 12))

vmin = min(cm_denorm[:n_show].min(), teacher_denorm[:n_show].min(), real_denorm[:n_show].min())
vmax = max(cm_denorm[:n_show].max(), teacher_denorm[:n_show].max(), real_denorm[:n_show].max())

for i in range(n_show):
    axes[0, i].imshow(cm_denorm[i, 0], cmap='viridis', vmin=vmin, vmax=vmax)
    axes[0, i].set_title(f'CM Student {i + 1}')
    axes[0, i].axis('off')

    axes[1, i].imshow(teacher_denorm[i, 0], cmap='viridis', vmin=vmin, vmax=vmax)
    axes[1, i].set_title(f'Teacher {i + 1}')
    axes[1, i].axis('off')

    im = axes[2, i].imshow(real_denorm[i, 0], cmap='viridis', vmin=vmin, vmax=vmax)
    axes[2, i].set_title(f'Real {i + 1}')
    axes[2, i].axis('off')

axes[0, 0].set_ylabel(f'CM ({STUDENT_STEPS} steps)', fontsize=12)
axes[1, 0].set_ylabel('Teacher (50 steps)', fontsize=12)
axes[2, 0].set_ylabel('Real', fontsize=12)

fig.colorbar(im, ax=axes, shrink=0.5, label='u(x,y)')
plt.suptitle('Consistency Model vs Teacher vs Real Darcy Flow', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/comparison.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nSaved {OUTPUT_DIR}/comparison.png")

# ============================================================
# PLOT 2: Grid of all CM samples
# ============================================================
n_grid = min(NUM_SAMPLES, 16)
rows = (n_grid + 3) // 4
fig, axes = plt.subplots(rows, 4, figsize=(16, 4 * rows))
axes = axes.flatten()

for i in range(len(axes)):
    if i < n_grid:
        axes[i].imshow(cm_denorm[i, 0], cmap='viridis')
        axes[i].set_title(f'CM Sample {i + 1}')
    axes[i].axis('off')

plt.suptitle(f'All CM Student Samples ({STUDENT_STEPS}-step generation)', fontsize=14)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/cm_grid.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {OUTPUT_DIR}/cm_grid.png")

# ============================================================
# PLOT 3: Histogram of pixel distributions
# ============================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.hist(cm_denorm.flatten(), bins=100, alpha=0.5, density=True,
        label=f'CM ({STUDENT_STEPS}-step)', color='tab:green')
ax.hist(teacher_denorm.flatten(), bins=100, alpha=0.5, density=True,
        label='Teacher (50-step)', color='tab:blue')
ax.hist(real_denorm.flatten(), bins=100, alpha=0.5, density=True,
        label='Real (test)', color='tab:orange')
ax.set_xlabel('u(x,y)')
ax.set_ylabel('Density')
ax.set_title('Pixel Value Distribution: CM vs Teacher vs Real')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{OUTPUT_DIR}/histogram.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Saved {OUTPUT_DIR}/histogram.png")

print(f"\nAll results saved to {OUTPUT_DIR}/")
print("Download them with: scp server:{OUTPUT_DIR}/*.png ./")
