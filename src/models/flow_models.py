"""Implements Conditional Flow Matcher Models."""

# Author: Meet Parikh
#         +++

from typing import Union

import torch
from torch.distributions import Chi2
from .base import GenerativeModel


def pad_t_like_x(t, x):
    """Function to reshape the time vector t by the number of dimensions of x.

    Parameters
    ----------
    x : Tensor, shape (bs, *dim)
        represents the source minibatch
    t : FloatTensor, shape (bs)

    Returns
    -------
    t : Tensor, shape (bs, number of x dimensions)

    Example
    -------
    x: Tensor (bs, C, W, H)
    t: Vector (bs)
    pad_t_like_x(t, x): Tensor (bs, 1, 1, 1)
    """
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))

class FlowMatcher(GenerativeModel):
    """
    Base Class for Flow Matching Algorithm with heavy tail noise functionality
    """

    def __init__(self, network, sigma: Union[float, int], add_heavy_noise : bool, *args, **kwargs):
        r"""Initialize the ConditionalFlowMatcher class. It requires the hyper-parameter $\sigma$.

        Parameters
        ----------
        sigma : Union[float, int]
        """
        super().__init__(network, infer=kwargs['infer'] if 'infer' in kwargs else False)
        self.sigma = sigma
        self.heavy_noise = False
        if add_heavy_noise:
            self.nu = kwargs.get("nu", torch.inf)
            self.heavy_noise = add_heavy_noise
            if self.nu == torch.inf:
                print("Heavy noise is set to True but nu is set to infinity. Falling back to normal noise.")
                self.heavy_noise = False
            else:
                self.chi2 = Chi2(self.nu)

    def sample_noise_like(self, x):
        if self.heavy_noise:
            z = torch.randn_like(x)
            kappa = self.chi2.sample((x.shape[0],)).to(x.device)/self.nu
            kappa = pad_t_like_x(kappa, x)
            return z / torch.sqrt(kappa)
        return torch.randn_like(x)

    def compute_mu_t(self, *args, **kwargs):
        raise NotImplementedError

    def compute_sigma_t(self, *args, **kwargs):
        raise NotImplementedError

    def sample_xt(self, *args, **kwargs):
        raise NotImplementedError

    def compute_conditional_flow(self, *args, **kwargs):
        raise NotImplementedError

    def sample_location_and_conditional_flow(self, *args, **kwargs):
        raise NotImplementedError

class ClassicFlowMatching(FlowMatcher):

    """
    Classic Flow Matching implementation:
    From Gaussian/Heavy Noise to Target Distribution
    Forward Noising process: mu_t = t * x1 and sigma_t = 1 - t + t*sigma
    Conditional Velocity: u_t = x1 - eps
    """

    def __init__(self, sigma, network, add_heavy_noise, use_denoiser=False, *args, **kwargs):
        super().__init__(network, sigma, add_heavy_noise=add_heavy_noise, *args, **kwargs)
        self.use_denoiser = use_denoiser

    def compute_mu_t(self, t, x1):
        return pad_t_like_x(t, x1) * x1

    def compute_sigma_t(self, t):
        return 1. - t + t*self.sigma

    def sample_xt(self, t, x1, epsilon):
        mu_t = self.compute_mu_t(t, x1)
        sigma_t = self.compute_sigma_t(t)
        sigma_t = pad_t_like_x(sigma_t, x1)
        return mu_t + sigma_t * epsilon

    def compute_conditional_flow(self, epsilon, x1):
        return x1  + (self.sigma - 1) * epsilon

    def sample_location_and_conditional_flow(self, x1, t=None):
        if t is None:
            t = torch.rand(x1.shape[0]).type_as(x1)
        assert len(t) == x1.shape[0], "t has to have batch size dimension"

        eps = self.sample_noise_like(x1)
        xt = self.sample_xt(t, x1, eps)
        ut = self.compute_conditional_flow(eps, x1)
        return t, xt, ut

    def sample(self, t, xt, *args, **kwargs):

        if t.dim() == 0:
             t = t[..., None]

        if t.shape[0] != xt.shape[0]:
            t = t.expand(xt.shape[0])

        if self.use_denoiser:
            t_den = pad_t_like_x(t, xt)
            return (self.network(t=t, x=xt, *args, **kwargs) - xt) / (1. - t_den).clip(min=0.05)
        else:
            return self.network(t=t, x=xt, *args, **kwargs)

    def get_training_objective(self, x0, x1, t=None, *args, **kwargs):

        del x0

        t, xt, ut  = self.sample_location_and_conditional_flow(x1, t)

        ut_pred = self.sample(t, xt, *args, **kwargs)

        return ut_pred, ut

class RectifiedFlowMatching(ClassicFlowMatching):

    """
    Rectified Flow implementation:
    From Gaussian/Heavy Noise to Target Distribution
    Forward Noising process: mu_t = t * x1 and sigma_t = 1 - t
    Conditional Velocity: u_t = x1 - eps
    """

    def __init__(self, network, add_heavy_noise, use_denoiser=False, *args, **kwargs):
        super().__init__(network=network, sigma=0.0, add_heavy_noise=add_heavy_noise, use_denoiser=use_denoiser, *args, **kwargs)

class MeanFlowMatching(RectifiedFlowMatching):

    """
    Mean Flow Matching implementation:
    From Arbitrary Source Distribution to Target Distribution
    Forward Noising process: mu_t = t*x1 + (1-t)*x_0 and sigma_t = 0
    Conditional Velocity: u_t = x1 - x0
    """

    def __init__(self, network, t_schedule="uniform", log_norm_args=(-0.4, 1), *args, **kwargs):
        # Force use_denoiser to False for MeanFlowMatching, ignoring any passed value
        kwargs.pop('use_denoiser', None)
        super().__init__(network=network, add_heavy_noise=False, use_denoiser=False, *args, **kwargs)
        self.t_schedule = t_schedule
        self.log_norm_args = log_norm_args

    def sample_t_and_r(self, x1):
        B = x1.shape[0]
        if self.t_schedule == "uniform":
            t = torch.rand(B).type_as(x1)
            r = torch.rand(B).type_as(x1)
            r = torch.where(r > t, r, t)
        elif self.t_schedule == "log_norm":
            mu, sigma = self.log_norm_args
            t = torch.empty(B).type_as(x1).log_normal_(mean=mu, std=sigma)
            t = torch.sigmoid(t)
            r = torch.empty(B).type_as(x1).log_normal_(mean=mu, std=sigma)
            r = torch.sigmoid(r)
            r = torch.where(r > t, r, t)
        else:
            raise ValueError(f"schedule {self.t_schedule} not recognized. Supported types are 'uniform' and 'log_norm'.")
        return t, r

    def sample_location_and_conditional_flow(self, x0, x1):

        del x0

        eps = self.sample_noise_like(x1)
        t, r = self.sample_t_and_r(x1)
        xt = self.sample_xt(t, x1, eps)
        ut = self.compute_conditional_flow(eps, x1)

        return t, r, xt, ut

    def sample(self, r, t, xt, *args, **kwargs):

        return self.network(r=r, t=t, x=xt, *args, **kwargs)

    def get_training_objective(self, x0, x1, t=None, *args, **kwargs):

        t, r, xt, vt  = self.sample_location_and_conditional_flow(x0, x1)
        u_pred, du_dt = torch.autograd.functional.jvp(self.sample, inputs=(r, t, xt), v=(torch.zeros_like(r).to(x1), torch.ones_like(t).to(x1), vt), create_graph=True, strict=True)
        r, t = pad_t_like_x(r, x1), pad_t_like_x(t, x1)
        u = ((r - t) * du_dt + vt).detach()
        return u_pred, u
