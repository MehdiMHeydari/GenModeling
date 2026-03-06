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
            # Offset t slightly below i/T to avoid exact t=1.0 where alpha_t≈0
            t_val = torch.full(
                (z.shape[0],), i / T - 1e-4,
                device=z.device, dtype=z.dtype
            )
            s_val = torch.full(
                (z.shape[0],), (i - 1) / T,
                device=z.device, dtype=z.dtype
            )

            x_hat = self.model.predict_x(
                z, t_val, use_ema=True, **model_kwargs
            )
            z = ddim_step(x_hat, z, t_val, s_val, schedule_s)

        return z


class RectifiedFlowSampler:
    """
    Euler ODE integrator for Rectified Flow models.

    Integrates from t=0 (noise) to t=1 (data) using the learned velocity field.
    Supports variable number of Euler steps for few-step generation.
    """

    def __init__(self, model):
        self.model = model

    @torch.no_grad()
    def sample(self, initial_noise, num_steps=100, **kwargs):
        """
        Generate samples starting from z ~ N(0, I).

        Args:
            initial_noise: Tensor [B, C, H, W] of Gaussian noise.
            num_steps: Number of Euler steps (more = higher quality).

        Returns:
            x: Generated samples [B, C, H, W].
        """
        x = initial_noise
        dt = 1.0 / num_steps

        for i in range(num_steps):
            t = torch.full(
                (x.shape[0],), i / num_steps,
                device=x.device, dtype=x.dtype,
            )
            # model.sample(t, xt) returns the predicted velocity
            v = self.model.sample(t, x, **kwargs)
            x = x + dt * v

        return x


class MeanSampler:
    """
    A wrapper for mean flow models for few step generation
    """

    def __init__(self, model):
        self.model = model

    def drift(self, t_prev, t_next, x):
        return self.model(t_next, t_prev, x)

    @torch.no_grad()
    def sample(self, initial_noise, **kwargs):

        if "t_span_kwargs" not in kwargs.keys():
            t_span = torch.linspace(0, 1, 2, device=initial_noise.device)

        else:
            t_sample = kwargs["t_span_kwargs"]
            t_span = torch.linspace(**t_sample, device=initial_noise.device)

        x = initial_noise

        for t_prev, t_next in zip(t_span[:-1], t_span[1:]):
            x = x + (t_next - t_prev) * self.drift(t_prev, t_next, x)

        return x
