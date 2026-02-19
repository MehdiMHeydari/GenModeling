import torch


class MultistepCMSampler:
    """
    Sampler for Multistep Consistency Models (Algorithm 2).

    Chains DDIM steps through student_steps segments, each using the
    consistency model's prediction.

    Reference: "Multistep Consistency Models" (Heek et al. 2024)
    """

    def __init__(self, model):
        """
        Args:
            model: MultistepConsistencyModel instance.
        """
        self.model = model

    @torch.no_grad()
    def sample(self, initial_noise, **kwargs):
        """
        Generate samples starting from z_1 ~ N(0, I).

        Algorithm 2:
          for t in (T/T, ..., 1/T):
            s = t - 1/T
            x_hat = f(z_t, t)
            z_s = DDIM(x_hat, z_t, t -> s)

        Args:
            initial_noise: Tensor [B, C, H, W] of Gaussian noise.
            **kwargs: Additional args (e.g., y for class conditioning).

        Returns:
            z_0: Generated samples [B, C, H, W].
        """
        from src.models.diffusion_utils import ddim_step

        z = initial_noise
        T = self.model.student_steps
        schedule_s = self.model.schedule_s

        # Extract class labels if provided
        y = kwargs.get('y', None)
        model_kwargs = {'y': y} if y is not None else {}

        for i in range(T, 0, -1):
            t_val = torch.full(
                (z.shape[0],), i / T,
                device=z.device, dtype=z.dtype
            )
            s_val = t_val - 1.0 / T

            x_hat = self.model.predict_x(
                z, t_val, use_ema=False, **model_kwargs
            )
            z = ddim_step(x_hat, z, t_val, s_val, schedule_s)

        return z
