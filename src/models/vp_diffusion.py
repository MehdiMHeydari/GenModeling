"""
VP Diffusion Model (Cosine Schedule) for use as a teacher in
Multistep Consistency Distillation.

Follows the same pattern as src/models/score_models.py (ScoreModel).
The network predicts clean data x_0 from noisy z_t using preconditioning.
"""

import copy
import torch
from .base import GenerativeModel
from .diffusion_utils import alpha_t, sigma_t, snr, q_sample, _broadcast_to_spatial
from .networks.shared.layers import update_ema


class VPDiffusionModel(GenerativeModel):
    """
    A variance-preserving diffusion model with cosine noise schedule.

    The network receives (t, z_t) and outputs a raw prediction. Preconditioning
    coefficients combine this with a skip connection to produce x_hat.

    Training objective: weighted MSE ||x_hat - x||^2, with w_t = SNR(t) + 1.

    Args:
        network: The backbone UNet/Transformer. Must accept forward(t, x, y=None).
        schedule_s: Cosine schedule offset parameter (default 0.008).
        ema_rate: EMA decay rate (default 0.9999). Set to 0 to disable.
    """

    def __init__(self, network, schedule_s=0.008, ema_rate=0.9999, *args, **kwargs):
        super().__init__(network, infer=kwargs.get('infer', False))
        self.schedule_s = schedule_s
        self.ema_rate = ema_rate

        # EMA network for higher-quality sampling
        if ema_rate > 0 and not kwargs.get('infer', False):
            self.ema_network = copy.deepcopy(network)
            for p in self.ema_network.parameters():
                p.requires_grad_(False)
            self.ema_network.eval()
        else:
            self.ema_network = None

    def update_ema(self):
        """Update EMA network parameters from online network."""
        if self.ema_network is not None:
            update_ema(
                self.ema_network.parameters(),
                self.network.parameters(),
                rate=self.ema_rate,
            )

    def predict_x(self, z_t, t, use_ema=False, **kwargs):
        """
        Predict clean data x_0 from z_t using the preconditioned network.

        x_hat = c_skip * z_t + c_out * F_theta(z_t, t)

        For VP diffusion (alpha^2 + sigma^2 = 1):
          c_skip = alpha_t^2          (at t=0: ~1, at t=1: ~0)
          c_out  = alpha_t * sigma_t  (at t=0: ~0, at t=1: ~0, peak at mid)

        Args:
            z_t: Noisy input [B, C, H, W].
            t: Timestep [B] in [0, 1].
            use_ema: If True, use the EMA network for prediction.
            **kwargs: Passed to network (e.g., y for class conditioning).

        Returns:
            x_hat: Predicted clean data [B, C, H, W].
        """
        if t.dim() == 0:
            t = t.unsqueeze(0)
        if t.shape[0] != z_t.shape[0]:
            t = t.expand(z_t.shape[0])

        a = alpha_t(t, self.schedule_s)
        sig = sigma_t(t, self.schedule_s)

        c_skip = _broadcast_to_spatial(a ** 2, z_t)
        c_out = _broadcast_to_spatial(a * sig, z_t)

        net = self.ema_network if (use_ema and self.ema_network is not None) else self.network
        raw = net(t=t, x=z_t, **kwargs)
        x_hat = c_skip * z_t + c_out * raw
        return x_hat

    def get_training_objective(self, x1, *args, **kwargs):
        """
        Compute tensors for the VP diffusion training loss.

        Samples t ~ U(0,1), creates z_t, returns (pred, target, weight) so
        that the loss is: mean(weight * (pred - target)^2).

        This matches the interface of ScoreModel.get_training_objective(),
        so ScoreMatchingLoss (or VPDiffusionLoss) can be used directly.

        Args:
            x1: Clean data [B, C, H, W].

        Returns:
            (x_hat, x1, weight): Prediction, target, and per-sample weight.
        """
        B = x1.shape[0]
        # Sample t uniformly in (0, 1), avoiding exact endpoints
        t = torch.rand(B, device=x1.device) * 0.998 + 0.001  # [0.001, 0.999]

        z_t, _ = q_sample(x1, t, s=self.schedule_s)
        x_hat = self.predict_x(z_t, t, **kwargs)

        # v-loss weighting: w_t = SNR(t) + 1
        w = _broadcast_to_spatial(snr(t, self.schedule_s) + 1.0, x1)

        return x_hat, x1, w

    def sample(self, t, z_t, **kwargs):
        """Alias for predict_x. Used during teacher queries in CD training."""
        return self.predict_x(z_t, t, **kwargs)
