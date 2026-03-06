# Complete Experiment Log

This document contains every detail needed to reproduce all experiments from scratch.

---

## Project Overview

**Goal:** Train fast generative models for 2D Darcy Flow fields.

**Data:** `2D_DarcyFlow_beta1.0_Train.hdf5` containing pressure fields from Darcy flow simulations.
- Shape: `(N, 128, 128)` loaded as `(N, 1, 128, 128)`
- 9000 samples used for training, remainder for test
- Normalization: min-max to `[-1, 1]`: `x_norm = 2 * (x - x_min) / (x_max - x_min) - 1`
- Denormalization: `x = (x_norm + 1) / 2 * (x_max - x_min) + x_min`
- Normalization stats (`data_min.npy`, `data_max.npy`) saved in each experiment's `saved_state/` directory

**Dataset class:** `VF_FM` from `src/utils/dataset.py`
- `__getitem__` returns `(np.empty(shape), data[index])` — a tuple `(x0, x1)` where `x0` is an empty placeholder and `x1` is the real data
- This matters for loss functions: losses unpack `x0, x1 = batch` and use `x1` as the real data

**Dataloader:** `get_darcy_loader()` from `src/utils/dataloader.py`
- Loads HDF5, applies min-max normalization, creates `VF_FM` dataset, returns `DataLoader` + normalization stats

---

## Shared UNet Architecture

All experiments use `UNetModelWrapper` from `src/models/networks/unet/unet.py` with these settings:

```yaml
dim: [1, 128, 128]          # (channels, height, width)
channel_mult: "1, 2, 4, 4"  # Channel multipliers per resolution level
num_channels: 64             # Base channel count
num_res_blocks: 2            # ResBlocks per resolution level
num_head_channels: 32        # Channels per attention head
attention_resolutions: "32"  # Apply attention when spatial dim = 32
dropout: 0.0
use_new_attention_order: True
use_scale_shift_norm: True
class_cond: False
num_classes: null
```

**Exception:** Mean Flow Matching adds `use_future_time_emb: True` for dual time (t, r) embedding.

The UNet accepts `forward(t, x, y=None, r=None)`:
- `t`: timestep embedding (always used)
- `x`: input tensor `[B, 1, 128, 128]`
- `y`: class labels (unused, `class_cond=False`)
- `r`: future time embedding (only for Mean Flow Matching)

---

## How to Run Experiments

All experiments are run on a GPU server via tmux:

```bash
tmux new -s <session_name>
conda activate gen-modeling
cd ~/GenModeling-main
git pull
CUDA_VISIBLE_DEVICES=<gpu_id> python scripts/<script>.py config/<config>.yaml
```

To detach: `Ctrl+B, D`. To reattach: `tmux attach -t <session_name>`. To kill: `tmux kill-session -t <session_name>`.

---

## Experiment 1: Teacher (VP Diffusion)

**Paper reference:** "Improved Denoising Diffusion Probabilistic Models" (Nichol & Dhariwal 2021) for the cosine schedule.

**Directory:** `darcy_teacher/exp_1/`

### What it does
Trains a VP diffusion model with cosine noise schedule. The network predicts clean data `x_0` from noisy `z_t` using preconditioning:
```
z_t = alpha_t * x + sigma_t * eps       (forward diffusion)
x_hat = c_skip * z_t + c_out * F(z_t, t)  (prediction with preconditioning)
c_skip = alpha_t^2
c_out  = alpha_t * sigma_t
```

### Training
- **Script:** `python scripts/train_vp_diffusion.py config/unet_vp_diffusion.yaml`
- **Config:** `config/unet_vp_diffusion.yaml`
- **Loss:** `VPDiffusionLoss` — weighted MSE: `mean(w * (x_hat - x)^2)` with `w = SNR(t) + 1`
- **Model class:** `VPDiffusionModel` from `src/models/vp_diffusion.py`

### Parameters
| Parameter | Value |
|-----------|-------|
| batch_size | 64 |
| lr | 1e-4 (Adam) |
| epochs | 400 |
| schedule_s | 0.008 (cosine schedule offset) |
| ema_rate | 0.9999 |
| th_seed | 1 |
| np_seed | 1 |
| checkpoint interval | every 25 epochs |

### Sampling
The teacher samples via DDIM (50 steps):
```python
ts = linspace(1.0, 0.0, 51)  # 50 steps from t=1 to t=0
for i in range(50):
    x_hat = model.predict_x(z, t[i])
    z = ddim_step(x_hat, z, t[i], t[i+1], schedule_s=0.008)
```

### Result
- **Best checkpoint:** `checkpoint_200.pt` (raw weights, NOT EMA)
- **Key finding:** EMA weights (0.9999 decay) performed worse than raw weights across all epochs. All downstream experiments use the raw-weight checkpoint as teacher.
- **Checkpoints saved:** `darcy_teacher/exp_1/saved_state/checkpoint_{0,25,50,...,399}.pt`
- **Wandb project:** `darcy-teacher`

---

## Experiment 2: Consistency Distillation (CD)

**Paper reference:** "Multistep Consistency Models" (Heek, Hoogeboom, Salimans 2024)

**Directory:** `darcy_student/exp_{1,2,3,4,5}/`

### What it does
Trains a student model to be "self-consistent" across segments of the noise schedule, using a frozen teacher to generate training signal. The student learns to jump across each segment in one step, eventually generating samples in `student_steps` total network evaluations.

### How the loss works (`MultistepCDLoss` in `src/training/objectives.py`)
1. Sample noise `eps`, segment index `step`, and relative position `n_rel`
2. Forward diffuse: `z_t = alpha_t * x + sigma_t * eps`
3. Teacher predicts `x_teacher` from `z_t`
4. aDDIM step: `z_s = aDDIM(x_teacher, z_t, x_var, t -> s)` where `x_var = 0.75 * ||x_teacher - x||^2 / d`
5. Online student predicts `x_hat_online` from `(z_t, t)` — WITH gradient
6. EMA target predicts `x_hat_target` from `(z_s, s)` — no gradient
7. DDIM from `s -> t_step` using target, then invDDIM to get reference `x_ref`
8. Loss: `mean(w * pseudo_huber(x_ref - x_hat_online))` with `w = SNR(t) + 1`

### Teacher step schedule
N_teacher anneals from 64 to 1280 over 100k iterations:
```python
N_teacher = exp(log(64) + clip(iteration/100000, 0, 1) * (log(1280) - log(64)))
```

### Sampling (`MultistepCMSampler` in `src/inference/samplers.py`)
```python
z = initial_noise  # z_1 ~ N(0, I)
for i in range(T, 0, -1):  # T = student_steps
    t = i/T - 1e-4
    s = (i-1)/T
    x_hat = model.predict_x(z, t, use_ema=True)
    z = ddim_step(x_hat, z, t, s, schedule_s=0.008)
return z
```

### Shared parameters across all CD experiments
| Parameter | Value |
|-----------|-------|
| teacher_checkpoint | `darcy_teacher/exp_1/saved_state/checkpoint_200.pt` (raw weights) |
| schedule_s | 0.008 |
| ema_rate | 0.9999 |
| init_from_teacher | True (student weights initialized from teacher) |
| x_var_frac | 0.75 |
| huber_epsilon | 1e-4 |
| lr | 1e-4 (Adam) |
| th_seed | 1 |
| np_seed | 1 |
| train_samples | 9000 |
| checkpoint interval | every 25 epochs |
| Wandb project | `darcy-student` |

### CD exp_1
| Parameter | Value |
|-----------|-------|
| student_steps | 4 |
| batch_size | 64 |
| grad_accum | 1 (none) |
| effective batch | 64 |
| epochs | 400 |
| **Result** | **Mode collapse — all samples are centered blobs (data mean)** |

**Diagnosis:** student_steps=4 means each segment covers 25% of the noise schedule, making the consistency constraint too hard.

### CD exp_2
| Parameter | Value |
|-----------|-------|
| student_steps | 8 |
| batch_size | 64 |
| grad_accum | 1 (none) |
| effective batch | 64 |
| epochs | 1000 |
| **Result** | Diversity emerging, loss 0.04 -> 0.02. Better than exp_1 but still not matching teacher. |

### CD exp_3
| Parameter | Value |
|-----------|-------|
| student_steps | 16 |
| batch_size | 64 |
| grad_accum | 1 (none) |
| effective batch | 64 |
| epochs | 1000 |
| **Result** | Similar to exp_2. 8 and 16 step students match each other but not teacher. |

Config file: `config/unet_cm_cd.yaml` (exp_num was manually changed between runs, currently set to 3)

### CD exp_4
| Parameter | Value |
|-----------|-------|
| student_steps | 16 |
| batch_size | 64 |
| grad_accum | 4 |
| effective batch | 256 |
| epochs | 1000 |
| **Result** | TBD |

Config file: `config/unet_cm_cd_exp4.yaml`

### CD exp_5
| Parameter | Value |
|-----------|-------|
| student_steps | 16 |
| batch_size | 64 |
| grad_accum | 8 |
| effective batch | 512 |
| epochs | 1000 |
| **Result** | Currently running. Produces recognizable Darcy patterns but not as sharp as teacher. |

Config file: `config/unet_cm_cd_exp5.yaml`

### Sampling config for CD experiments
Config file: `config/unet_cm_sample.yaml`
```yaml
cd:
  checkpoint_path: "darcy_student/exp_1/saved_state/checkpoint_175.pt"  # change per experiment
  student_steps: 4  # change to match experiment
inference:
  total_samples_to_generate: 500
  shape_of_sample: "1,128,128"
  batch_size: 8
  use_ema: True
  save_path: "darcy_student/exp_1/gen_samples/samples.pt"
  norm_stats_dir: "darcy_teacher/exp_1/saved_state"
```
Script: `python scripts/sample_cm.py config/unet_cm_sample.yaml`

---

## Experiment 3: Mean Flow Matching (MFM)

**Paper reference:** "Mean Flows for One-step Generative Modeling" (Geng & Deng, 2025, arXiv:2505.13447)

**Source code origin:** Ported from `~/Downloads/JianXun Repo/` (Prof. JianXun Wang's research group codebase).

**Directory:** `darcy_mean_flow/exp_{1,2,3}/`

### What it does
Trains a model to predict the **average velocity** over a time interval `[t, r]` (dual time variables), rather than instantaneous velocity. This enables 1-step generation because the model directly predicts the full displacement from noise to data.

The model `u_theta(z_t, r, t)` takes both current time `t` and future time `r` as inputs. The UNet encodes `t` via the standard time embedding and `r` via a separate "future time embedding" (`use_future_time_emb: True`), which are summed together.

### How the training works (`MeanFlowMatching.get_training_objective` in `src/models/flow_models.py`)

1. Sample `t, r` with `r >= t` (both uniform or logit-normal on `[0, 1]`)
2. Sample noise `eps ~ N(0, I)` and create `x_t = t * x1 + (1 - t) * eps` (rectified flow interpolation)
3. Compute conditional velocity `v_t = x1 - eps`
4. Use `torch.autograd.functional.jvp` to compute both the model prediction and its time derivative:
   ```python
   u_pred, du_dt = jvp(
       model.sample,            # function: (r, t, x) -> u_theta
       inputs=(r, t, xt),
       v=(zeros, ones, vt),     # tangent: dr=0, dt=1, dx=v_t
       create_graph=True
   )
   ```
5. Compute target with stop-gradient: `u_target = ((r - t) * du_dt + v_t).detach()`
6. Loss: adaptive MSE with gamma weighting

### Loss function (`MeanFlowMatchingLoss` in `src/training/objectives.py`)
```python
delta = u_pred - u_target
delta_l2_sq = delta.view(B, -1).pow(2).sum(dim=1)  # per-sample L2 squared
w = (1 / (delta_l2_sq + 1e-3)^(1 - gamma)).detach()  # adaptive weight
loss = (w * delta_l2_sq).mean()  # mean over batch
```
- `gamma=0.0` gives the strongest adaptive weighting (best in paper ablation)
- The `.mean()` is critical — the JianXun source uses `.sum()` because they train with `batch_size=1` per GPU

### Sampling (`MeanSampler` in `src/inference/samplers.py`)
Simple Euler integration:
```python
t_span = linspace(0, 1, steps)  # default: steps=2 (1 Euler step)
x = initial_noise  # z_0 ~ N(0, I)
for t_prev, t_next in zip(t_span[:-1], t_span[1:]):
    drift = model(r=t_next, t=t_prev, x=x)  # predict mean velocity over [t_prev, t_next]
    x = x + (t_next - t_prev) * drift
return x
```
For 1-step: `x = noise + 1.0 * model(r=1, t=0, noise)` — single network evaluation.

### MFM exp_1 (FAILED)
| Parameter | Value |
|-----------|-------|
| batch_size | 16 |
| grad_accum | 4 |
| effective batch | 64 |
| lr | 1e-4 |
| epochs | 1000 |
| gamma | 0.0 |
| t_schedule | uniform |
| loss reduction | `.sum()` (BUG) |
| **Result** | **FAILED — noisy blurry blobs. Root cause: loss used `.sum()` with batch_size=16, making gradients ~16x too large.** |

**Bug explanation:** The JianXun source uses `.sum()` with `batch_size=1` per GPU, so `.sum()` = `.mean()` for a single sample. With our `batch_size=16`, `.sum()` makes the loss 16x larger. Combined with 4x higher learning rate (1e-4 vs 2.5e-5) and grad_accum=4, the effective parameter updates were ~256x larger than JianXun's. The model could never converge.

### MFM exp_2 (loss fix, low lr)
| Parameter | Value |
|-----------|-------|
| batch_size | 16 |
| grad_accum | 1 (none) |
| effective batch | 16 |
| lr | 2.5e-5 (matches JianXun base lr, but JianXun scales by world_size=4 → effective 1e-4) |
| epochs | 1000 |
| gamma | 0.0 |
| t_schedule | uniform |
| log_norm_args | [-0.4, 1.0] (available but not used with uniform) |
| loss reduction | `.mean()` (FIXED) |
| **Result** | **Still blurry blobs at epoch 25.** Diagnostics at epoch 25: `||delta||^2=27.13`, `adaptive_w=0.065`, loss saturated at `0.999935 ≈ 1.0`. The gamma=0 adaptive weight crushes gradients when errors are large. Additionally, lr=2.5e-5 is 4x lower than JianXun's effective lr (they use `lr * world_size = 2.5e-5 * 4 = 1e-4`). |

### MFM exp_3 (corrected lr to match JianXun effective lr)
| Parameter | Value |
|-----------|-------|
| batch_size | 16 |
| grad_accum | 1 (none) |
| effective batch | 16 |
| lr | 1e-4 (matches JianXun's effective lr: 2.5e-5 * 4 GPUs) |
| epochs | 1000 |
| gamma | 0.0 |
| t_schedule | uniform |
| log_norm_args | [-0.4, 1.0] (available but not used with uniform) |
| loss reduction | `.mean()` |
| **Result** | TBD |

Config file: `config/unet_mean_flow.yaml` (exp_num=3)

### MFM sampling config
Config file: `config/unet_mean_flow_sample.yaml`
```yaml
mean_flow:
  checkpoint_path: "darcy_mean_flow/exp_1/saved_state/checkpoint_999.pt"  # update per experiment
inference:
  total_samples_to_generate: 500
  shape_of_sample: "1,128,128"
  batch_size: 8
  save_path: "darcy_mean_flow/exp_1/gen_samples/samples.pt"
  norm_stats_dir: "darcy_mean_flow/exp_1/saved_state"
  solver_kwargs:
    t_span_kwargs:
      start: 0
      end: 1
      steps: 2    # 1 Euler step (2 time points = 1 interval)
```
Script: `python scripts/sample_mean_flow.py config/unet_mean_flow_sample.yaml`

### JianXun Repo reference parameters (for comparison)
| Parameter | JianXun Value |
|-----------|---------------|
| batch_size | 1 per GPU (distributed) |
| lr | 2.5e-5 |
| loss reduction | `.sum()` (equivalent to `.mean()` with batch=1) |
| data | 3-channel channel flow, 320x200 |
| normalization | mean/std (z-score) |
| num_channels | 128 |
| gamma | 0.0 |
| t_schedule | uniform |

---

## Experiment 4: Progressive Distillation (PD)

**Paper reference:** "Progressive Distillation for Fast Sampling of Diffusion Models" (Salimans & Ho, ICLR 2022)

**Directory:** `darcy_pd/exp_1/`

### What it does
Iteratively halves the number of DDIM sampling steps. Each round trains a student to match 2 teacher DDIM steps in 1 forward pass. After each round, the student becomes the new teacher and the step count halves.

### Round structure
Starting from 128 steps, halving to 2 steps = `log2(128/2) = 6 rounds`:

| Round | Teacher Steps (N) | Student Steps (N/2) |
|-------|-------------------|---------------------|
| 0 | 128 | 64 |
| 1 | 64 | 32 |
| 2 | 32 | 16 |
| 3 | 16 | 8 |
| 4 | 8 | 4 |
| 5 | 4 | 2 |

### How the loss works (`ProgressiveDistillationLoss` in `src/training/objectives.py`)

For teacher with N steps, student with N/2 steps:
1. Sample student step index `i ~ Uniform{1, ..., N/2}`
2. Compute times: `t = 2i/N`, `t_mid = (2i-1)/N`, `t_target = (2i-2)/N`
3. Forward diffuse: `z_t = alpha_t * x + sigma_t * eps`
4. Teacher does 2 DDIM steps (no gradient):
   ```
   x1 = teacher.predict_x(z_t, t)
   z_mid = ddim_step(x1, z_t, t, t_mid)
   x2 = teacher.predict_x(z_mid, t_mid)
   z_target = ddim_step(x2, z_mid, t_mid, t_target)
   ```
5. Recover target via invDDIM: `x_target = inv_ddim(z_target, z_t, t, t_target)`
6. Student predicts: `x_student = student.predict_x(z_t, t)`
7. Loss: `mean((SNR(t) + 1) * (x_student - x_target)^2)` — weighted MSE, same as teacher

### Training procedure per round
1. Student initialized from teacher weights
2. Fresh Adam optimizer each round
3. Train for `epochs_per_round` epochs
4. Save checkpoint: `pd_round{i}_steps{N/2}.pt`
5. Student becomes next round's teacher (frozen, eval mode, no grad)

### Parameters
| Parameter | Value |
|-----------|-------|
| teacher_checkpoint | `darcy_teacher/exp_1/saved_state/checkpoint_200.pt` (raw weights) |
| schedule_s | 0.008 |
| start_steps | 128 |
| end_steps | 2 |
| epochs_per_round | 100 |
| batch_size | 64 |
| grad_accum | 1 (none) |
| lr | 1e-4 (Adam) |
| student ema_rate | 0 (no EMA — student is a VPDiffusionModel with ema disabled) |
| th_seed | 1 |
| np_seed | 1 |
| Wandb project | `darcy-pd` |
| **Result** | Currently running. Produces recognizable Darcy patterns. |

Config file: `config/unet_pd.yaml`
Script: `python scripts/train_pd.py config/unet_pd.yaml`

### PD Sampling
Uses plain DDIM with the student's step count:
```python
ts = linspace(1.0, 0.0, student_steps + 1)
z = initial_noise
for i in range(student_steps):
    t = clamp(ts[i], 1e-4, 1-1e-4)
    s = clamp(ts[i+1], 0, 1-1e-4)
    x_hat = model.predict_x(z, t, use_ema=True)
    z = ddim_step(x_hat, z, t, s, schedule_s=0.008)
```

### PD sampling config
Config file: `config/unet_pd_sample.yaml`
```yaml
pd:
  schedule_s: 0.008
  checkpoint_path: "darcy_pd/exp_1/saved_state/pd_round6_steps2.pt"  # update per round
  student_steps: 2  # match the checkpoint's step count
inference:
  total_samples_to_generate: 500
  shape_of_sample: "1,128,128"
  batch_size: 8
  save_path: "darcy_pd/exp_1/gen_samples/samples.pt"
  norm_stats_dir: "darcy_teacher/exp_1/saved_state"
```

---

## Experiment Comparison Script

**Script:** `python scripts/check_experiments.py --gpu 0 --n_samples 8`
**Config:** `config/` (no separate config, hardcoded paths in script)

Generates a side-by-side grid comparing all running experiments. Samples from:
1. Teacher (50-step DDIM)
2. CD exp_5 (latest checkpoint, 16 steps)
3. MFM (latest checkpoint, 1 Euler step)
4. PD (latest round checkpoint)

Uses **shared noise** for fair comparison (seed=42).

Output: `experiment_check.png`

---

## Evaluation Script

**Script:** `python scripts/evaluate_all.py --num-samples 256 --output-dir eval_results`

Produces 4 output files:
1. `sample_grids.png` — visual sample comparison
2. `mean_std_fields.png` — pixelwise mean and std fields
3. `power_spectral_density.png` — radially averaged 2D PSD
4. `sampling_speed.txt` — timing comparison table

---

## File Structure

```
GenModeling-main/
├── config/
│   ├── unet_vp_diffusion.yaml       # Teacher training config
│   ├── unet_cm_cd.yaml              # CD exp_1/2/3 config (change exp_num, student_steps)
│   ├── unet_cm_cd_exp4.yaml         # CD exp_4 config (grad_accum=4)
│   ├── unet_cm_cd_exp5.yaml         # CD exp_5 config (grad_accum=8)
│   ├── unet_cm_sample.yaml          # CD sampling config
│   ├── unet_mean_flow.yaml          # MFM training config
│   ├── unet_mean_flow_sample.yaml   # MFM sampling config
│   ├── unet_pd.yaml                 # PD training config
│   └── unet_pd_sample.yaml          # PD sampling config
├── scripts/
│   ├── train_vp_diffusion.py        # Teacher training
│   ├── train_cm.py                  # CD student training
│   ├── sample_cm.py                 # CD sampling
│   ├── train_mean_flow.py           # MFM training
│   ├── sample_mean_flow.py          # MFM sampling
│   ├── train_pd.py                  # PD training
│   ├── check_experiments.py         # Side-by-side comparison
│   ├── evaluate_all.py              # Full evaluation
│   └── evaluate_teacher.py          # Teacher-only evaluation
├── src/
│   ├── models/
│   │   ├── base.py                  # GenerativeModel base class
│   │   ├── vp_diffusion.py          # VP Diffusion model (teacher)
│   │   ├── consistency_models.py    # Multistep Consistency Model (CD student)
│   │   ├── flow_models.py           # FlowMatcher -> ClassicFM -> RectifiedFM -> MeanFM
│   │   ├── diffusion_utils.py       # Cosine schedule, DDIM, invDDIM, aDDIM
│   │   └── networks/unet/unet.py   # UNet with time/future-time/class embeddings
│   ├── training/
│   │   ├── objectives.py            # VPDiffusionLoss, MultistepCDLoss, MeanFlowMatchingLoss, ProgressiveDistillationLoss
│   │   └── trainer.py               # Generic Trainer class (used by JianXun, not by our scripts)
│   ├── inference/
│   │   └── samplers.py              # MultistepCMSampler, MeanSampler
│   └── utils/
│       ├── dataset.py               # VF_FM dataset class
│       └── dataloader.py            # get_darcy_loader (HDF5, min-max norm)
└── data/
    └── 2D_DarcyFlow_beta1.0_Train.hdf5
```

---

## Known Issues and Fixes

### 1. MFM loss scaling (FIXED in exp_2)
- **Bug:** `MeanFlowMatchingLoss` used `.sum()` over batch, causing gradients to scale with batch size
- **Source:** JianXun Repo uses `batch_size=1` where `.sum()` = `.mean()`
- **Fix:** Changed to `.mean()` in `src/training/objectives.py` line 207

### 4. MFM learning rate too low in exp_2
- **Issue:** exp_2 used lr=2.5e-5, matching JianXun's *base* lr. But JianXun scales lr by world_size (4 GPUs): `lr * world_size = 1e-4`
- **Impact:** exp_2 was learning 4x slower than JianXun, compounded by gamma=0 adaptive weight suppression
- **Fix:** exp_3 uses lr=1e-4 to match JianXun's effective learning rate

### 2. Teacher EMA vs raw weights
- EMA weights (0.9999 decay) consistently performed worse than raw weights for this dataset
- All experiments use the raw-weight checkpoint (`model_state_dict`, not `ema_state_dict`)

### 3. CD mode collapse at low step counts
- student_steps=4 causes mode collapse (exp_1)
- student_steps=8 and 16 produce diversity but don't match teacher quality
- Larger effective batch sizes (grad_accum) help but don't solve the fundamental issue
