"""
Multistep Consistency Model for Consistency Distillation.

Implements the model parameterization, EMA target management, and
sampling algorithm (Algorithm 2) from:
"Multistep Consistency Models" (Heek, Hoogeboom, Salimans 2024).
"""

import copy
import torch
import torch.nn as nn
from .base import GenerativeModel
from .diffusion_utils import (
    alpha_t, sigma_t, ddim_step, _broadcast_to_spatial
)
from .networks.shared.layers import update_ema


class MultistepConsistencyModel(GenerativeModel):
    """
    Multistep Consistency Model (MCM).

    The same network weights are shared across all segments. At each
    segment boundary t_step = step/student_steps, the model is
    parameterized to satisfy a boundary condition via a skip connection.

    An EMA (exponential moving average) copy of the network is maintained
    internally for use as the target model during consistency distillation.

    Args:
        network: The backbone UNet/Transformer.
        student_steps: Number of segments to split [0, 1] into.
        schedule_s: Cosine schedule offset parameter.
        ema_rate: EMA decay rate for the target network.
    """

    def __init__(self, network, student_steps=2, schedule_s=0.008,
                 ema_rate=0.9999, *args, **kwargs):
        super().__init__(network, infer=kwargs.get('infer', False))
        self.student_steps = student_steps
        self.schedule_s = schedule_s
        self.ema_rate = ema_rate

        # EMA (target) network: deep copy with frozen parameters
        self.ema_network = copy.deepcopy(network)
        for p in self.ema_network.parameters():
            p.requires_grad_(False)
        self.ema_network.eval()

    def predict_x(self, z_t, t, use_ema=False, **kwargs):
        """
        Predict clean data x_hat from (z_t, t).

        Uses a boundary-respecting parameterization: at the start of each
        segment (t = t_step), the prediction is constrained to recover x
        from z_t via a skip connection. As t increases within the segment,
        the network output is blended in.

        Parameterization:
          x_hat = c_skip(t) * (z_t / alpha_t) + c_out(t) * F_theta(z_t, t)
        where c_skip -> 1 and c_out -> 0 as t -> t_step (boundary condition).

        Args:
            z_t: Noisy input [B, C, H, W].
            t: Timestep [B] in [0, 1].
            use_ema: If True, use the EMA (target) network.
            **kwargs: Passed to network (e.g., y for class conditioning).

        Returns:
            x_hat: Predicted clean data [B, C, H, W].
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.shape[0] != z_t.shape[0]:
            t = t.expand(z_t.shape[0])

        net = self.ema_network if use_ema else self.network

        # Determine which segment each sample falls in
        step = torch.floor(t * self.student_steps).clamp(
            0, self.student_steps - 1
        ).long()
        t_step = step.float() / self.student_steps

        # Fraction through the current segment: 0 at boundary, ~1 at end
        segment_frac = (t - t_step) * self.student_steps
        segment_frac = segment_frac.clamp(0.0, 1.0)
        segment_frac_b = _broadcast_to_spatial(segment_frac, z_t)

        # Network forward pass
        raw = net(t=t, x=z_t, **kwargs)

        # Skip connection: at boundary, recover x = z_t / alpha_t
        a_t = _broadcast_to_spatial(alpha_t(t, self.schedule_s), z_t)
        x_identity = z_t / a_t.clamp(min=1e-4)

        # Blend: at boundary c_skip=1 (identity), away from boundary c_out grows
        x_hat = (1.0 - segment_frac_b) * x_identity + segment_frac_b * raw

        return x_hat

    def update_ema(self):
        """Update EMA network parameters from online network."""
        update_ema(
            self.ema_network.parameters(),
            self.network.parameters(),
            rate=self.ema_rate,
        )

    def get_training_objective(self, *args, **kwargs):
        """Not used directly. See MultistepCDLoss which orchestrates training."""
        raise NotImplementedError(
            "Use MultistepCDLoss which orchestrates teacher + student + EMA."
        )

    def sample(self, z_T, **kwargs):
        """
        Algorithm 2: Multistep sampling.

        Starting from z_1 ~ N(0, I), chain DDIM steps through segments:
          for t in (T/T, ..., 1/T):
            s = t - 1/T
            x_hat = f(z_t, t)
            z_s = DDIM(x_hat, z_t, t -> s)

        Args:
            z_T: Initial noise [B, C, H, W].
            **kwargs: Passed to predict_x (e.g., y for class conditioning).

        Returns:
            z_0: Generated samples [B, C, H, W].
        """
        z = z_T
        T = self.student_steps

        for i in range(T, 0, -1):
            # Offset t below the segment boundary so floor(t*T) assigns it
            # to segment i-1 (top of segment, full network output).
            t_val = torch.full(
                (z.shape[0],), i / T - 1e-4, device=z.device, dtype=z.dtype
            )
            s_val = torch.full(
                (z.shape[0],), (i - 1) / T, device=z.device, dtype=z.dtype
            )

            x_hat = self.predict_x(z, t_val, use_ema=True, **kwargs)
            z = ddim_step(x_hat, z, t_val, s_val, self.schedule_s)

        return z
