"""
VP Diffusion utilities for Multistep Consistency Models.

Implements the cosine noise schedule, DDIM, inverse DDIM, and adjusted DDIM (aDDIM)
from "Multistep Consistency Models" (Heek, Hoogeboom, Salimans 2024).

All functions operate on batched tensors and handle broadcasting from [B] scalars
to [B, C, H, W] spatial tensors.
"""

import torch
import math


# --------------------------------------------------------------------------- #
# Helper: broadcast a [B] tensor to match a [B, C, H, W] tensor              #
# --------------------------------------------------------------------------- #

def _broadcast_to_spatial(val, ref):
    """Expand a [B] tensor to match the number of dims of ref (e.g. [B,C,H,W])."""
    while val.dim() < ref.dim():
        val = val.unsqueeze(-1)
    return val


# --------------------------------------------------------------------------- #
# Cosine Noise Schedule (Nichol & Dhariwal 2021)                              #
# --------------------------------------------------------------------------- #

def cosine_alpha_bar(t, s=0.008):
    """
    Cosine schedule for alpha_bar(t).

    alpha_bar(t) = cos^2( (t + s) / (1 + s) * pi/2 )

    Args:
        t: Timesteps in [0, 1], any shape.
        s: Small offset to avoid singularity near t=0.

    Returns:
        alpha_bar values, same shape as t.
    """
    return torch.cos(((t + s) / (1.0 + s)) * (math.pi / 2.0)) ** 2


def alpha_t(t, s=0.008):
    """alpha_t = sqrt(alpha_bar(t))."""
    return torch.sqrt(cosine_alpha_bar(t, s).clamp(min=1e-8))


def sigma_t(t, s=0.008):
    """sigma_t = sqrt(1 - alpha_bar(t)), so that alpha_t^2 + sigma_t^2 = 1."""
    return torch.sqrt((1.0 - cosine_alpha_bar(t, s)).clamp(min=1e-8))


def snr(t, s=0.008):
    """Signal-to-Noise Ratio: SNR(t) = alpha_t^2 / sigma_t^2."""
    ab = cosine_alpha_bar(t, s)
    return ab / (1.0 - ab).clamp(min=1e-8)


# --------------------------------------------------------------------------- #
# Forward Diffusion: q(z_t | x)                                               #
# --------------------------------------------------------------------------- #

def q_sample(x, t, noise=None, s=0.008):
    """
    Sample from the forward VP diffusion process.

    z_t = alpha_t * x + sigma_t * epsilon

    Args:
        x: Clean data [B, C, H, W].
        t: Timestep [B] in [0, 1].
        noise: Optional pre-sampled noise, same shape as x.
        s: Cosine schedule offset.

    Returns:
        (z_t, noise): Noisy data and the noise used.
    """
    if noise is None:
        noise = torch.randn_like(x)

    a = _broadcast_to_spatial(alpha_t(t, s), x)
    sig = _broadcast_to_spatial(sigma_t(t, s), x)
    z_t = a * x + sig * noise
    return z_t, noise


# --------------------------------------------------------------------------- #
# DDIM Step (Eq. 4 in the paper)                                              #
# --------------------------------------------------------------------------- #

def ddim_step(x_hat, z_t, t, s_target, schedule_s=0.008):
    """
    Standard DDIM step: given x_hat prediction at time t, produce z_{s_target}.

    z_s = alpha_s * x_hat + (sigma_s / sigma_t) * (z_t - alpha_t * x_hat)

    Args:
        x_hat: Predicted clean data [B, C, H, W].
        z_t: Noisy data at time t [B, C, H, W].
        t: Current time [B].
        s_target: Target time [B], with s_target < t.
        schedule_s: Cosine schedule offset.

    Returns:
        z_s: Data at time s_target.
    """
    a_t = _broadcast_to_spatial(alpha_t(t, schedule_s), x_hat)
    sig_t = _broadcast_to_spatial(sigma_t(t, schedule_s), x_hat)
    a_s = _broadcast_to_spatial(alpha_t(s_target, schedule_s), x_hat)
    sig_s = _broadcast_to_spatial(sigma_t(s_target, schedule_s), x_hat)

    z_s = a_s * x_hat + (sig_s / sig_t) * (z_t - a_t * x_hat)
    return z_s


# --------------------------------------------------------------------------- #
# Inverse DDIM (Eq. 5 in the paper)                                           #
# --------------------------------------------------------------------------- #

def inv_ddim(z_s, z_t, t, s_target, schedule_s=0.008):
    """
    Inverse DDIM: recover x from z_s and z_t such that DDIM(x, z_t, t->s) = z_s.

    x = (z_s - (sigma_s / sigma_t) * z_t) / (alpha_s - alpha_t * sigma_s / sigma_t)

    Args:
        z_s: Data at time s_target [B, C, H, W].
        z_t: Data at time t [B, C, H, W].
        t: The "from" time [B].
        s_target: The "to" time [B].
        schedule_s: Cosine schedule offset.

    Returns:
        x: Recovered clean-data prediction.
    """
    a_t = _broadcast_to_spatial(alpha_t(t, schedule_s), z_s)
    sig_t = _broadcast_to_spatial(sigma_t(t, schedule_s), z_s)
    a_s = _broadcast_to_spatial(alpha_t(s_target, schedule_s), z_s)
    sig_s = _broadcast_to_spatial(sigma_t(s_target, schedule_s), z_s)

    ratio = sig_s / sig_t
    denom = (a_s - a_t * ratio).clamp(min=1e-8)
    x = (z_s - ratio * z_t) / denom
    return x


# --------------------------------------------------------------------------- #
# Adjusted DDIM (Algorithm 3 in the paper)                                     #
# --------------------------------------------------------------------------- #

def addim_step(x_hat, z_t, x_var, t, s_target, schedule_s=0.008):
    """
    Adjusted DDIM (aDDIM) step from Algorithm 3 of the MCM paper.
    Corrects for the missing variance in standard DDIM by inflating the
    noise prediction.

    eps_hat = (z_t - alpha_t * x_hat) / sigma_t
    z_s_var = (alpha_s - alpha_t * sigma_s / sigma_t)^2 * x_var
    z_s = alpha_s * x_hat + sqrt(sigma_s^2 + (d / ||eps_hat||^2) * z_s_var) * eps_hat

    Args:
        x_hat: Teacher prediction of clean data [B, C, H, W].
        z_t: Noisy data [B, C, H, W].
        x_var: Per-sample variance, shape [B]. Typically eta * ||x_teacher - x||^2 / d.
        t: Current time [B].
        s_target: Target time [B].
        schedule_s: Cosine schedule offset.

    Returns:
        z_s: Result of the aDDIM step.
    """
    a_t = _broadcast_to_spatial(alpha_t(t, schedule_s), x_hat)
    sig_t = _broadcast_to_spatial(sigma_t(t, schedule_s), x_hat)
    a_s = _broadcast_to_spatial(alpha_t(s_target, schedule_s), x_hat)
    sig_s = _broadcast_to_spatial(sigma_t(s_target, schedule_s), x_hat)

    # Predicted noise
    eps_hat = (z_t - a_t * x_hat) / sig_t

    # Variance contribution coefficient
    coeff = a_s - a_t * sig_s / sig_t
    x_var_b = _broadcast_to_spatial(x_var, x_hat)
    z_s_var = coeff ** 2 * x_var_b

    # Data dimensionality: C * H * W
    d = float(x_hat[0].numel())

    # ||eps_hat||^2 per sample
    eps_norm_sq = eps_hat.flatten(1).pow(2).sum(dim=1)
    eps_norm_sq = _broadcast_to_spatial(eps_norm_sq, x_hat)

    # aDDIM formula
    inner = sig_s ** 2 + (d / eps_norm_sq.clamp(min=1e-8)) * z_s_var
    z_s = a_s * x_hat + torch.sqrt(inner.clamp(min=1e-8)) * eps_hat
    return z_s
