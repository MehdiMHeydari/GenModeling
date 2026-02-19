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

    Args:
        class_conditional: Whether the model is class-conditional.
        teacher_model: Frozen VPDiffusionModel (the teacher).
        student_steps: Number of consistency segments.
        x_var_frac: Eta parameter for aDDIM (default 0.75).
        huber_epsilon: Epsilon for pseudo-Huber loss (default 1e-4).
        schedule_s: Cosine schedule offset.
    """

    def __init__(self, class_conditional, teacher_model, student_steps=2,
                 x_var_frac=0.75, huber_epsilon=1e-4, schedule_s=0.008):
        super().__init__(class_conditional)
        self.teacher = teacher_model
        self.student_steps = student_steps
        self.x_var_frac = x_var_frac
        self.huber_epsilon = huber_epsilon
        self.schedule_s = schedule_s
        self._iteration = 0

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
