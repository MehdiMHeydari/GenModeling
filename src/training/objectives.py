import torch


class Loss:

    def __init__(self, class_conditional : bool):
        self.class_conditional =  class_conditional

    def __call__(self, *args, **kwargs):
        return NotImplementedError


# --------------------------------------------------------------------------- #
# VP Diffusion Loss                                                            #
# --------------------------------------------------------------------------- #

class VPDiffusionLoss(Loss):
    """Weighted MSE loss for VP diffusion model training.

    The model's get_training_objective returns (pred, target, weight),
    and the loss is: mean(weight * (pred - target)^2).
    Same interface as ScoreMatchingLoss.
    """

    def __init__(self, class_conditional):
        super().__init__(class_conditional)

    def __call__(self, model, batch, device):
        if self.class_conditional:
            _, x, y = batch
            x, y = x.to(device), y.to(device)
        else:
            _, x = batch
            y = None
            x = x.to(device)
        pred, target, weight = model(x, y=y) if self.class_conditional else model(x)
        loss = torch.mean(weight * (pred - target) ** 2)
        return loss


# --------------------------------------------------------------------------- #
# Multistep Consistency Distillation Loss (Algorithm 1)                        #
# --------------------------------------------------------------------------- #

import math

class MultistepCDLoss(Loss):
    """
    Multistep Consistency Distillation loss from Algorithm 1 of
    "Multistep Consistency Models" (Heek, Hoogeboom, Salimans 2024).

    Orchestrates the teacher model, online student, and EMA target to
    compute the consistency distillation objective.

    Optionally includes a sampling-based moment-matching regularizer that
    runs the full student sampling chain every `moment_every` iterations
    and penalizes deviation from pre-computed teacher distribution moments.

    Args:
        class_conditional: Whether the model is class-conditional.
        teacher_model: Frozen VPDiffusionModel (the teacher).
        student_steps: Number of consistency segments.
        x_var_frac: Eta parameter for aDDIM (default 0.75).
        huber_epsilon: Epsilon for pseudo-Huber loss (default 1e-4).
        schedule_s: Cosine schedule offset.
        moment_weight_mu: Weight for mean-matching loss (0 = off).
        moment_weight_var: Weight for variance-matching loss (0 = off).
        teacher_moments_path: Path to pre-computed teacher_moments.pt.
        moment_every: Run sampling-based moment loss every N iterations.
        moment_batch_size: Number of samples to generate for moment computation.
        moment_normalized: If True, divide each squared moment error by its
            teacher-target scale so every term is dimensionless O(1) and the
            weights transfer across datasets/normalizations. The raw
            (unnormalized) loss is scale-sensitive: on z-scored CNS the
            variance targets are up to ~60x Darcy's, so the squared terms
            are ~3,700x larger and Darcy-tuned weights destabilize training
            (cns_student/exp_22 diverged to structured noise). Default False
            to keep the Darcy/NS paper runs exactly reproducible.
        moment_per_channel: If True, compute the four moment statistics per
            CHANNEL against per-channel teacher targets (keys *_ch in the
            moments file, from precompute_teacher_moments --per_channel) and
            average the normalized terms over channels. The pooled stats
            cannot see per-channel failures (a dead Vy channel leaves the
            pooled mean/var almost unchanged on CNS because Vy carries ~2%
            of the pooled variance). Implies moment_normalized.
    """

    def __init__(self, class_conditional, teacher_model, student_steps=2,
                 x_var_frac=0.75, huber_epsilon=1e-4, schedule_s=0.008,
                 moment_weight_mu=0.0, moment_weight_var=0.0,
                 teacher_moments_path=None, moment_every=50,
                 moment_batch_size=32, sample_shape=(1, 128, 128),
                 moment_normalized=False, moment_per_channel=False,
                 moment_spectral=False, moment_weight_spec=0.0,
                 spectral_band_edges=(1, 4, 16, 48, 64)):
        super().__init__(class_conditional)
        self.teacher = teacher_model
        self.student_steps = student_steps
        self.x_var_frac = x_var_frac
        self.huber_epsilon = huber_epsilon
        self.schedule_s = schedule_s
        self.moment_weight_mu = moment_weight_mu
        self.moment_weight_var = moment_weight_var
        self.moment_every = moment_every
        self.moment_batch_size = moment_batch_size
        self.moment_normalized = moment_normalized or moment_per_channel
        self.moment_per_channel = moment_per_channel
        # Spectral-band moments: the amplitude moments (spatial mean/var)
        # are blind to spatial structure — a sample can satisfy them with any
        # texture (the exp_22 noise divergence and the exp_23-27 mid-scale
        # textures both did). Band log-powers of the radial spectrum SEE
        # structure: the low-k band forces field-scale commitment and the
        # across-sample variance of band power forces regime diversity.
        self.moment_spectral = moment_spectral
        self.moment_weight_spec = moment_weight_spec
        self.spectral_band_edges = tuple(spectral_band_edges)
        self._band_masks = None  # lazy, cached per device
        self.sample_shape = tuple(sample_shape)
        self._iteration = 0
        self.last_moment_mu = 0.0
        self.last_moment_var = 0.0
        self.last_moment_spec = 0.0

        _any_moment = (moment_weight_mu > 0 or moment_weight_var > 0
                       or (moment_spectral and moment_weight_spec > 0))
        # Load pre-computed teacher moments
        if teacher_moments_path is not None and _any_moment:
            import os
            assert os.path.exists(teacher_moments_path), \
                f"Teacher moments not found: {teacher_moments_path}"
            teacher_moments = torch.load(teacher_moments_path, map_location="cpu",
                                         weights_only=True)
            self.teacher_mu_mean = teacher_moments["mu_mean"]
            self.teacher_mu_var = teacher_moments["mu_var"]
            self.teacher_var_mean = teacher_moments["var_mean"]
            self.teacher_var_var = teacher_moments["var_var"]
            print(f"Loaded teacher moments from {teacher_moments_path}")
            print(f"  target mu:  mean={self.teacher_mu_mean:.6f}, var={self.teacher_mu_var:.6f}")
            print(f"  target var: mean={self.teacher_var_mean:.6f}, var={self.teacher_var_var:.6f}")
            if self.moment_per_channel:
                for k in ("mu_mean_ch", "mu_var_ch", "var_mean_ch", "var_var_ch"):
                    assert k in teacher_moments, (
                        f"moment_per_channel=True but '{k}' missing from "
                        f"{teacher_moments_path}. Re-run "
                        "precompute_teacher_moments.py --per_channel."
                    )
                self.teacher_mu_mean_ch = teacher_moments["mu_mean_ch"].float()
                self.teacher_mu_var_ch = teacher_moments["mu_var_ch"].float()
                self.teacher_var_mean_ch = teacher_moments["var_mean_ch"].float()
                self.teacher_var_var_ch = teacher_moments["var_var_ch"].float()
                print(f"  per-channel targets loaded "
                      f"({len(self.teacher_mu_mean_ch)} channels)")
            if self.moment_spectral:
                for k in ("spec_mean_ch", "spec_var_ch", "spec_band_edges"):
                    assert k in teacher_moments, (
                        f"moment_spectral=True but '{k}' missing from "
                        f"{teacher_moments_path}. Re-run "
                        "precompute_teacher_moments.py (spectral targets are "
                        "computed automatically in the current version)."
                    )
                self.teacher_spec_mean = teacher_moments["spec_mean_ch"].float()
                self.teacher_spec_var = teacher_moments["spec_var_ch"].float()
                loaded_edges = tuple(int(x) for x in teacher_moments["spec_band_edges"])
                assert loaded_edges == self.spectral_band_edges, (
                    f"spectral band edges mismatch: targets {loaded_edges} vs "
                    f"config {self.spectral_band_edges}")
                print(f"  spectral targets loaded "
                      f"{tuple(self.teacher_spec_mean.shape)} (C, bands)")
        else:
            self.teacher_mu_mean = None

    def _teacher_step_schedule(self):
        """
        Annealing schedule for teacher discretization steps.
        N_teacher(i) = exp(log(64) + clip(i/100000, 0, 1) * (log(1280) - log(64)))
        """
        progress = min(self._iteration / 100000.0, 1.0)
        log_N = math.log(64) + progress * (math.log(1280) - math.log(64))
        return max(1, round(math.exp(log_N)))

    def _pseudo_huber(self, x):
        """Pseudo-Huber loss: sqrt(x^2 + eps^2) - eps."""
        return torch.sqrt(x ** 2 + self.huber_epsilon ** 2) - self.huber_epsilon

    def _band_logpower(self, z):
        """Per-sample, per-channel log mean power in radial wavenumber bands.

        z: (B, C, H, W) -> (B, C, K) with K = len(band_edges) - 1. The DC
        bin is excluded (band edges start at k=1); the spatial mean is
        already constrained by the amplitude moments. Differentiable
        (torch.fft). Band masks are cached per device/shape.
        """
        B, C, H, W = z.shape
        P = torch.fft.fftshift(torch.fft.fft2(z).abs() ** 2, dim=(-2, -1))
        if self._band_masks is None or self._band_masks[0].shape != (H, W) \
                or self._band_masks[0].device != z.device:
            cy, cx = H // 2, W // 2
            yy, xx = torch.meshgrid(
                torch.arange(H, device=z.device),
                torch.arange(W, device=z.device), indexing="ij")
            r = torch.sqrt((yy - cy).float() ** 2 + (xx - cx).float() ** 2)
            self._band_masks = [
                ((r >= lo) & (r < hi)).float()
                for lo, hi in zip(self.spectral_band_edges[:-1],
                                  self.spectral_band_edges[1:])
            ]
        stats = []
        for m in self._band_masks:
            band_mean = (P * m).sum(dim=(-2, -1)) / m.sum().clamp(min=1.0)
            stats.append(torch.log(band_mean + 1e-12))
        return torch.stack(stats, dim=-1)  # (B, C, K)

    def sample_moment_loss(self, model, device):
        """Run the full student sampling chain and compare moments to teacher.

        Uses gradient checkpointing to backprop through ALL T steps without
        storing all intermediate activations. Each step's activations are
        recomputed during backward, so peak memory is ~2x one forward pass
        instead of 16x.

        Should be called AFTER cd_loss.backward() so the CD graph is freed.

        Returns weighted moment loss tensor, or None if not a moment iteration.
        """
        if not (self.moment_weight_mu > 0 or self.moment_weight_var > 0
                or (self.moment_spectral and self.moment_weight_spec > 0)):
            return None
        if self.teacher_mu_mean is None:
            return None
        if (self._iteration - 1) % self.moment_every != 0:
            return None

        from src.models.diffusion_utils import ddim_step
        from torch.utils.checkpoint import checkpoint

        T = self.student_steps
        schedule_s = self.schedule_s
        z = torch.randn(self.moment_batch_size, *self.sample_shape, device=device)

        def _step_fn(z_in, t_val, s_val):
            x_hat = model.predict_x(z_in, t_val, use_ema=False)
            return ddim_step(x_hat, z_in, t_val, s_val, schedule_s)

        for i in range(T, 0, -1):
            t_val = torch.full((z.shape[0],), i / T - 1e-4, device=device)
            s_val = torch.full((z.shape[0],), (i - 1) / T, device=device)
            z = checkpoint(_step_fn, z, t_val, s_val, use_reentrant=False)

        total = None

        # Spectral-band moments (composable with any amplitude variant)
        if self.moment_spectral and self.moment_weight_spec > 0:
            eps = 1e-8
            s = self._band_logpower(z)          # (B, C, K)
            t_mean = self.teacher_spec_mean.to(z.device)
            t_var = self.teacher_spec_var.to(z.device)
            loss_spec = (((s.mean(0) - t_mean) ** 2 / (t_var + eps))
                         + ((s.var(0) - t_var) ** 2 / (t_var ** 2 + eps))).mean()
            self.last_moment_spec = loss_spec.item()
            total = self.moment_weight_spec * loss_spec

        # Per-channel amplitude variant: stats per channel vs per-channel
        # targets, normalized terms averaged over channels.
        if self.moment_per_channel and (self.moment_weight_mu > 0
                                        or self.moment_weight_var > 0):
            eps = 1e-8
            B, C = z.shape[0], z.shape[1]
            flat_ch = z.reshape(B, C, -1)
            mu_c = flat_ch.mean(dim=2)     # (B, C)
            var_c = flat_ch.var(dim=2)     # (B, C)
            t_mu_mean = self.teacher_mu_mean_ch.to(z.device)
            t_mu_var = self.teacher_mu_var_ch.to(z.device)
            t_var_mean = self.teacher_var_mean_ch.to(z.device)
            t_var_var = self.teacher_var_var_ch.to(z.device)
            loss_mu = (((mu_c.mean(0) - t_mu_mean) ** 2 / (t_mu_var + eps))
                       + ((mu_c.var(0) - t_mu_var) ** 2 / (t_mu_var ** 2 + eps))).mean()
            loss_var = (((var_c.mean(0) - t_var_mean) ** 2 / (t_var_var + eps))
                        + ((var_c.var(0) - t_var_var) ** 2 / (t_var_var ** 2 + eps))).mean()
            self.last_moment_mu = loss_mu.item()
            self.last_moment_var = loss_var.item()
            amp = self.moment_weight_mu * loss_mu + self.moment_weight_var * loss_var
            return amp if total is None else total + amp
        if self.moment_per_channel:
            return total

        if not (self.moment_weight_mu > 0 or self.moment_weight_var > 0):
            return total

        # Compute moments of generated samples
        flat = z.flatten(1)
        mu_student = flat.mean(dim=1)
        var_student = flat.var(dim=1)

        # Compare to teacher targets
        if self.moment_normalized:
            # Dimensionless form: each squared error is divided by its
            # natural teacher scale. E[stat] errors are scaled by the
            # teacher's variance of that stat (a squared z-score of the
            # batch statistic); Var[stat] errors by the target variance
            # squared (a squared relative error). Every term is O(1) when
            # the student sits within the teacher's own spread.
            eps = 1e-8
            loss_mu = (mu_student.mean() - self.teacher_mu_mean) ** 2 \
                / (self.teacher_mu_var + eps) \
                + (mu_student.var() - self.teacher_mu_var) ** 2 \
                / (self.teacher_mu_var ** 2 + eps)
            loss_var = (var_student.mean() - self.teacher_var_mean) ** 2 \
                / (self.teacher_var_var + eps) \
                + (var_student.var() - self.teacher_var_var) ** 2 \
                / (self.teacher_var_var ** 2 + eps)
        else:
            loss_mu = (mu_student.mean() - self.teacher_mu_mean) ** 2 \
                    + (mu_student.var() - self.teacher_mu_var) ** 2

            loss_var = (var_student.mean() - self.teacher_var_mean) ** 2 \
                     + (var_student.var() - self.teacher_var_var) ** 2

        self.last_moment_mu = loss_mu.item()
        self.last_moment_var = loss_var.item()

        amp = self.moment_weight_mu * loss_mu + self.moment_weight_var * loss_var
        return amp if total is None else total + amp

    def __call__(self, model, batch, device):
        """
        Compute the multistep consistency distillation loss.

        Args:
            model: MultistepConsistencyModel (online student + EMA target).
            batch: From dataloader, (x0_unused, x1) or (x0_unused, x1, y).
            device: Torch device.

        Returns:
            Scalar loss tensor.
        """
        from src.models.diffusion_utils import (
            alpha_t as _alpha_t, sigma_t as _sigma_t, snr as _snr,
            q_sample, ddim_step, inv_ddim, addim_step,
            _broadcast_to_spatial,
        )

        # 1. Unpack batch
        if self.class_conditional:
            _, x, y = batch
            x, y = x.to(device), y.to(device)
            net_kwargs = {'y': y}
        else:
            _, x = batch
            y = None
            x = x.to(device)
            net_kwargs = {}

        B = x.shape[0]
        d = float(x[0].numel())  # C * H * W

        # 2. Compute teacher step schedule
        N_teacher = self._teacher_step_schedule()
        N_per_segment = max(1, round(N_teacher / self.student_steps))
        T_total = N_per_segment * self.student_steps

        # 3. Sample noise and segment indices
        eps = torch.randn_like(x)
        step = torch.randint(0, self.student_steps, (B,), device=device)
        n_rel = torch.randint(1, N_per_segment + 1, (B,), device=device)

        # 4. Compute times
        t_step = step.float() / self.student_steps
        t = t_step + n_rel.float() / T_total
        s = t - 1.0 / T_total

        # Clamp to valid range
        t = t.clamp(1e-5, 1.0 - 1e-5)
        s = s.clamp(1e-5, 1.0 - 1e-5)
        t_step = t_step.clamp(0.0, 1.0 - 1e-5)

        # 5. Forward diffuse: z_t = alpha_t * x + sigma_t * eps
        z_t, _ = q_sample(x, t, noise=eps, s=self.schedule_s)

        # 6. Teacher prediction at z_t (no gradient)
        with torch.no_grad():
            x_teacher = self.teacher.predict_x(z_t, t, **net_kwargs)

        # 7. Compute x_var = eta * ||x_teacher - x||^2 / d
        x_var = self.x_var_frac * (x_teacher - x).flatten(1).pow(2).sum(1) / d

        # 8. aDDIM teacher step: z_t -> z_s (no gradient)
        with torch.no_grad():
            z_s = addim_step(x_teacher, z_t, x_var, t, s, self.schedule_s)

        # 9. Online student prediction at (z_t, t) — WITH gradient
        x_hat_online = model.predict_x(z_t, t, use_ema=False, **net_kwargs)

        # 10. EMA target prediction at (z_s, s) — no gradient
        with torch.no_grad():
            x_hat_target = model.predict_x(z_s, s, use_ema=True, **net_kwargs)

        # 11. DDIM from s -> t_step using target prediction (no gradient)
        with torch.no_grad():
            z_ref_t_step = ddim_step(
                x_hat_target, z_s, s, t_step, self.schedule_s
            )

        # 12. invDDIM: recover reference x from (z_ref_t_step, z_t, t -> t_step)
        with torch.no_grad():
            x_ref = inv_ddim(z_ref_t_step, z_t, t, t_step, self.schedule_s)

        # 13. Consistency loss with pseudo-Huber and v-loss weighting
        x_diff = x_ref.detach() - x_hat_online

        w_t = _snr(t, self.schedule_s) + 1.0
        w_t = _broadcast_to_spatial(w_t, x)

        loss = torch.mean(w_t * self._pseudo_huber(x_diff))

        # 14. Increment iteration counter
        self._iteration += 1

        return loss


# --------------------------------------------------------------------------- #
# Mean Flow Matching Loss                                                      #
# --------------------------------------------------------------------------- #

class MeanFlowMatchingLoss(Loss):
    """Adaptive MSE loss for mean flow matching."""

    def __init__(self, class_conditional, gamma=0.):
        super().__init__(class_conditional)
        self.gamma = gamma

    def __call__(self, model, batch, device):
        if self.class_conditional:
            x0, x1, y = batch
            x0, x1, y = x0.to(device), x1.to(device), y.to(device)
        else:
            x0, x1 = batch
            y = None
            x0, x1 = x0.to(device), x1.to(device)
        ut_pred, ut = model(x0, x1, y)
        delta = ut_pred - ut
        delta_l2_sq = delta.view(delta.shape[0], -1).pow(2).sum(dim=1)
        w = (1./ (delta_l2_sq + 1e-3)**(1 - self.gamma)).detach()
        loss = (w * delta_l2_sq).mean()
        return loss


# --------------------------------------------------------------------------- #
# Rectified Flow Loss                                                          #
# --------------------------------------------------------------------------- #

class RectifiedFlowLoss(Loss):
    """MSE loss for Rectified Flow training.

    Supports two modes:
    - Standard (round 1): data comes as (_, x1), noise is sampled fresh.
    - Reflow (round 2+): data comes as (z, x) paired, uses stored noise.
    """

    def __init__(self, class_conditional, reflow=False):
        super().__init__(class_conditional)
        self.reflow = reflow

    def __call__(self, model, batch, device):
        if self.reflow:
            # Paired data from reflow: (z_noise, x_data)
            z, x = batch
            z, x = z.to(device), x.to(device)
            B = x.shape[0]
            t = torch.rand(B, device=device)
            t_broad = t.reshape(-1, *([1] * (x.dim() - 1)))
            xt = t_broad * x + (1 - t_broad) * z
            ut_target = x - z
            ut_pred = model.network(t=t, x=xt)
            loss = torch.mean((ut_pred - ut_target) ** 2)
        else:
            # Standard: (placeholder, data)
            _, x = batch
            x = x.to(device)
            # model.forward -> get_training_objective(x0, x1) which ignores x0
            ut_pred, ut = model(torch.zeros_like(x), x)
            loss = torch.mean((ut_pred - ut) ** 2)
        return loss


# --------------------------------------------------------------------------- #
# Progressive Distillation Loss (Salimans & Ho, ICLR 2022)                    #
# --------------------------------------------------------------------------- #

class ProgressiveDistillationLoss(Loss):
    """
    Progressive distillation loss: trains a student to match 2 teacher DDIM
    steps in 1 forward pass. The target x_hat is computed via inv_ddim from
    the result of 2 teacher steps.

    Args:
        teacher: Frozen VPDiffusionModel.
        num_teacher_steps: Number of DDIM steps the teacher uses (N).
        schedule_s: Cosine schedule offset.
    """

    def __init__(self, teacher, num_teacher_steps, schedule_s=0.008):
        super().__init__(class_conditional=False)
        self.teacher = teacher
        self.N = num_teacher_steps
        self.schedule_s = schedule_s

    def __call__(self, model, batch, device):
        from src.models.diffusion_utils import (
            snr as _snr, q_sample, ddim_step, inv_ddim,
            _broadcast_to_spatial,
        )

        _, x = batch
        x = x.to(device)
        B = x.shape[0]

        half_N = self.N // 2

        # Sample student step index: i in {1, ..., N/2}
        step_idx = torch.randint(1, half_N + 1, (B,), device=device)

        # Compute times in [0, 1]
        t        = (2 * step_idx).float() / self.N
        t_mid    = (2 * step_idx - 1).float() / self.N
        t_target = (2 * step_idx - 2).float() / self.N

        # Clamp to avoid schedule singularities
        t = t.clamp(1e-4, 1 - 1e-4)
        t_mid = t_mid.clamp(1e-4, 1 - 1e-4)
        t_target = t_target.clamp(0, 1 - 1e-4)

        # Forward diffuse: z_t = alpha_t * x + sigma_t * eps
        z_t, _ = q_sample(x, t, s=self.schedule_s)

        # Teacher: 2 DDIM steps (no grad)
        with torch.no_grad():
            x_hat_1 = self.teacher.predict_x(z_t, t, use_ema=True)
            z_mid = ddim_step(x_hat_1, z_t, t, t_mid, self.schedule_s)

            x_hat_2 = self.teacher.predict_x(z_mid, t_mid, use_ema=True)
            z_target = ddim_step(x_hat_2, z_mid, t_mid, t_target, self.schedule_s)

            # Target: x_hat that makes DDIM(x_hat, z_t, t -> t_target) = z_target
            x_target = inv_ddim(z_target, z_t, t, t_target, self.schedule_s)

        # Student prediction (with grad)
        x_hat_student = model.predict_x(z_t, t)

        # Weighted MSE loss (v-loss weighting: SNR + 1)
        w = _snr(t, self.schedule_s) + 1.0
        w = _broadcast_to_spatial(w, x)

        loss = torch.mean(w * (x_hat_student - x_target.detach()) ** 2)
        return loss
