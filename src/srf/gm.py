"""Gaussian-mixture utilities for the SRF flow prior.

The flow prior of SRF predicts the displacement (velocity) ``u`` of the additive
path ``H_t = H_0 + t * u``, where ``u`` is a channel-domain noise realisation.
The network output head parameterises ``p(u | H_t, t)`` as a Gaussian mixture
with the following tensor layout (kept from the legacy implementation):

* ``means``      -- ``(B, K, C, H, W)``
* ``logstds``    -- ``(B, 1, 1, 1, 1)`` for the single shared scale used by the
  SRF/DPR prior, or variants accepted by :func:`sample_gm`
* ``logweights`` -- ``(B, K, 1, H, W)``, log-softmax normalised over ``K``

The DPR clean-channel centre is obtained from the mixture of ``u`` as
``H_0 = H_t - t * mu_theta(H_t, t)`` (paper eq. (25)); see :func:`u_to_x0_gm`.
"""

from __future__ import annotations

import math
from typing import Dict, Optional

import torch

__all__ = [
    "GMDict",
    "expand_shared_logstd",
    "gm_effective_variance",
    "gm_global_stats",
    "gm_logprob",
    "gm_nll_loss",
    "gm_second_moment",
    "gm_tau2_from_x0_gm",
    "gm_to_dtype",
    "gm_to_mean",
    "reverse_transition_additive",
    "sample_gm",
    "u_to_x0_gm",
]

GMDict = Dict[str, torch.Tensor]


def expand_shared_logstd(logstds: torch.Tensor, out_shape: torch.Size) -> torch.Tensor:
    """Broadcast a shared log-standard-deviation to ``out_shape = (B, C, H, W)``.

    Two legacy layouts are supported: a single scalar per sample
    ``(B, 1, 1, 1, 1)`` and a single scalar per pixel ``(B, 1, 1, H, W)``.
    """
    if logstds.ndim != 5:
        raise ValueError(f"Expected 5D logstds, got shape={tuple(logstds.shape)}")
    if logstds.shape[1:] == (1, 1, 1, 1):
        b = logstds.shape[0]
        _, c, h, w = out_shape
        return logstds.view(b, 1, 1, 1).expand(b, c, h, w)
    if logstds.shape[1] == 1 and logstds.shape[2] == 1:
        return logstds.expand(-1, 1, out_shape[1], -1, -1).squeeze(1)
    raise ValueError(f"Unsupported shared logstd shape: {tuple(logstds.shape)}")


def gm_to_dtype(gm: GMDict, dtype: torch.dtype) -> GMDict:
    """Cast every mixture tensor to ``dtype`` (used for fp16-safe inference)."""
    return {key: value.to(dtype) for key, value in gm.items()}


def gm_to_mean(gm: GMDict) -> torch.Tensor:
    """Mixture mean ``sum_k pi_k mu_k`` with shape ``(B, C, H, W)``."""
    weights = gm["logweights"].exp()
    return (weights * gm["means"]).sum(dim=1)


def gm_second_moment(gm: GMDict, eps: float = 1e-8) -> torch.Tensor:
    """Second moment ``sum_k pi_k (mu_k^2 + sigma_k^2)`` of the mixture."""
    weights = gm["logweights"].exp()
    var = (2.0 * gm["logstds"]).exp().clamp_min(eps)
    return (weights * (gm["means"].square() + var)).sum(dim=1)


def gm_effective_variance(gm: GMDict, eps: float = 1e-8) -> torch.Tensor:
    """Isotropic variance proxy ``E[x^2] - E[x]^2`` of the mixture."""
    mean = gm_to_mean(gm)
    second = gm_second_moment(gm, eps=eps)
    return (second - mean.square()).clamp_min(eps)


def sample_gm(gm: GMDict, generator: Optional[torch.Generator] = None) -> torch.Tensor:
    """Draw one sample per pixel from the mixture, shape ``(B, C, H, W)``."""
    means = gm["means"]
    logweights = gm["logweights"]
    logstds = gm["logstds"]
    b, k, c, h, w = means.shape

    probs = logweights[:, :, 0].permute(0, 2, 3, 1).reshape(-1, k).exp()
    comp_idx = torch.multinomial(probs, 1, replacement=True, generator=generator).view(b, h, w)
    gather_idx = comp_idx[:, None, None, :, :].expand(-1, 1, c, -1, -1)
    chosen_means = torch.gather(means, 1, gather_idx).squeeze(1)

    # Shared logstd (``(B, 1, 1, 1, 1)`` or ``(B, 1, 1, H, W)``) takes priority so
    # that a single-component mixture is not mistaken for a per-component scale.
    if logstds.shape[1] == 1 and logstds.shape[2] == 1:
        chosen_logstds = expand_shared_logstd(logstds, chosen_means.shape)
    elif logstds.shape[1] == k:
        chosen_logstds = torch.gather(
            logstds.expand(-1, -1, c, -1, -1), 1, gather_idx
        ).squeeze(1)
    else:
        raise ValueError(f"Unsupported logstd shape: {tuple(logstds.shape)}")

    if generator is None:
        noise = torch.randn_like(chosen_means)
    else:
        noise = torch.randn(
            chosen_means.shape,
            device=chosen_means.device,
            dtype=chosen_means.dtype,
            generator=generator,
        )
    return chosen_means + chosen_logstds.exp() * noise


def u_to_x0_gm(
    gm_u: GMDict, x_t: torch.Tensor, sigma: torch.Tensor, eps: float = 1e-6
) -> GMDict:
    """Map the velocity mixture ``p(u | x_t, t)`` to the clean-channel mixture.

    For the additive path ``x_t = x_0 + sigma * u`` the mixture transforms to
    ``x_0 = x_t - sigma * u`` with standard deviations scaled by ``sigma``.
    With ``sigma = t`` this is the DPR centre of paper eq. (25).
    """
    if sigma.dim() == 1:
        sigma = sigma.view(-1, 1, 1, 1)
    sigma = sigma.clamp_min(eps)
    x_t = x_t.unsqueeze(1)
    sigma_expanded = sigma.unsqueeze(1)
    return {
        "means": x_t - sigma_expanded * gm_u["means"],
        "logstds": gm_u["logstds"] + sigma.log().view(-1, 1, 1, 1, 1),
        "logweights": gm_u["logweights"],
    }


def reverse_transition_additive(
    gm_u: GMDict,
    x_t_high: torch.Tensor,
    t_low: torch.Tensor,
    t_high: torch.Tensor,
    eps: float = 1e-6,
) -> GMDict:
    """Finite-transition target mixture for DPR training.

    For ``x_t = x_0 + t * u`` the transition ``t_high -> t_low`` reads
    ``x_t_low = x_t_high - (t_high - t_low) * u``; the mixture means are shifted
    and the standard deviations scaled by ``delta = t_high - t_low``.
    """
    if t_low.dim() == 1:
        t_low = t_low.view(-1, 1, 1, 1)
    if t_high.dim() == 1:
        t_high = t_high.view(-1, 1, 1, 1)
    delta = (t_high - t_low).clamp_min(eps)
    x_t_high = x_t_high.unsqueeze(1)
    return {
        "means": x_t_high - delta.unsqueeze(1) * gm_u["means"],
        "logstds": gm_u["logstds"] + delta.log().view(-1, 1, 1, 1, 1),
        "logweights": gm_u["logweights"],
    }


def gm_logprob(gm: GMDict, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Log-density ``log p(target)`` of the mixture for ``target: (B, C, H, W)``."""
    target = target.unsqueeze(1)
    logstd = gm["logstds"]
    var = (2.0 * logstd).exp().clamp_min(eps)
    centered = target - gm["means"]
    c_dim = centered.shape[2]
    quad = centered.square().sum(dim=2, keepdim=True) / var
    log_det = c_dim * logstd
    log_comp = (
        gm["logweights"]
        - 0.5 * quad
        - log_det
        - 0.5 * c_dim * math.log(2.0 * math.pi)
    )
    return torch.logsumexp(log_comp, dim=1)


def gm_nll_loss(
    gm: GMDict, target: torch.Tensor, reduction: str = "mean"
) -> torch.Tensor:
    """Negative log-likelihood of ``target`` under the mixture."""
    nll = -gm_logprob(gm, target)
    if reduction == "none":
        return nll
    if reduction == "mean":
        return nll.mean()
    raise ValueError(f"Unsupported reduction: {reduction}")


def gm_global_stats(gm: GMDict) -> Dict[str, torch.Tensor]:
    """Scalar mixture statistics used for evaluation diagnostics."""
    weights = gm["logweights"].exp()
    entropy = -(weights.clamp_min(1e-12) * gm["logweights"]).sum(dim=1).mean()
    max_weight = weights.max(dim=1).values.mean()
    return {
        "entropy_mean": entropy,
        "max_weight_mean": max_weight,
        "logstd_mean": gm["logstds"].mean(),
    }


def gm_tau2_from_x0_gm(gm_x0: GMDict, eps: float = 1e-8) -> torch.Tensor:
    """Mean clean-channel mixture variance per sample, shape ``(B,)``."""
    tau2 = (
        (2.0 * gm_x0["logstds"])
        .exp()
        .reshape(gm_x0["means"].shape[0], -1)
        .mean(dim=1)
    )
    return tau2.clamp_min(eps)
