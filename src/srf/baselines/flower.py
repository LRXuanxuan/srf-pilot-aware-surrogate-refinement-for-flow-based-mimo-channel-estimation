"""FLOWER full-posterior sampler (comparison method of the SRF paper).

Extracted from the legacy ``test_flower_full_gm1tr05.py``.  Each reverse step
uses the flow prior to build ``q(x_0 | x_t)``, combines it with the pilot
likelihood to obtain the Gaussian posterior ``p(x_0 | x_t, y)``, optionally
draws from that posterior, and advances time with fresh source noise.

Supported modes (``FLOWER_MODES``):

* ``"mean"``   -- posterior-mean solver (``gamma = 0``);
* ``"sample"`` -- one posterior sample per step (full FLOWER);
* ``"mmse"``   -- Monte-Carlo MMSE, i.e. the average of several independent
  posterior-sampling trajectories.
"""

from __future__ import annotations

import math
from typing import Any, Dict

import torch

from ..gm import gm_to_dtype, gm_to_mean, u_to_x0_gm
from ..utils import (
    complex_dtype,
    get_dynamic_nfe,
    model_device,
    model_dtype,
    real_dtype,
)

__all__ = ["FLOWER_MODES", "flower_sample_channels"]

FLOWER_MODES = ("mean", "sample", "mmse")


def _complex_randn(
    shape: tuple[int, ...],
    device: torch.device,
    dtype: torch.dtype = torch.complex64,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """Circularly symmetric complex Gaussian sample ``CN(0, 1)``."""
    real_dtype_ = torch.float32 if dtype == torch.complex64 else torch.float64
    real = torch.randn(shape, device=device, dtype=real_dtype_, generator=generator)
    imag = torch.randn(shape, device=device, dtype=real_dtype_, generator=generator)
    return (real + 1j * imag) / math.sqrt(2.0)


def _safe_cholesky_hermitian(
    mat: torch.Tensor, jitter: float = 1e-6, max_tries: int = 5
) -> torch.Tensor:
    """Cholesky factorisation with increasing diagonal jitter for stability."""
    eye = torch.eye(mat.shape[-1], device=mat.device, dtype=mat.dtype).expand_as(mat)
    current = mat
    for k in range(max_tries):
        try:
            return torch.linalg.cholesky(current)
        except RuntimeError:
            current = mat + (10.0**k) * jitter * eye
    return torch.linalg.cholesky(current)


def _single_trajectory(
    model: torch.nn.Module,
    y_cplx: torch.Tensor,
    p_cplx: torch.Tensor,
    noise_scale: torch.Tensor,
    total_steps: int,
    return_mode: str,
    seed_offset: int,
    seed: int,
) -> torch.Tensor:
    """One FLOWER posterior trajectory with fresh-noise time progression."""
    device = model_device(model)
    batch_size, nr, _ = y_cplx.shape
    nt = p_cplx.shape[1]
    out_real_dtype = real_dtype(y_cplx)
    out_complex_dtype = complex_dtype(y_cplx)
    mdtype = model_dtype(model)

    generator = torch.Generator(device=device)
    generator.manual_seed(seed + seed_offset)

    p_h = p_cplx.mH
    y_p_h = torch.bmm(y_cplx, p_h)
    p_p_h = torch.bmm(p_cplx, p_h)
    eye_nt = torch.eye(nt, device=device, dtype=out_complex_dtype).expand(batch_size, nt, nt)
    sigma2 = noise_scale.view(batch_size, 1, 1).pow(2).clamp_min(1e-8)

    h_t = _complex_randn(
        (batch_size, nr, nt), device=device, dtype=out_complex_dtype, generator=generator
    )

    for step in range(total_steps):
        t_curr = max(1.0 - step / total_steps, 1e-4)
        t_batch = torch.full((batch_size,), t_curr, device=device, dtype=out_real_dtype)
        t_batch_model = t_batch.to(mdtype)

        # Step 1: flow-consistent destination distribution q(x_0 | x_t).
        h_t_real_model = (
            torch.view_as_real(h_t).permute(0, 3, 1, 2).contiguous().to(mdtype)
        )
        gm_u = model(h_t_real_model, t_batch_model)
        gm_u = gm_to_dtype(gm_u, out_real_dtype)
        gm_x0 = u_to_x0_gm(gm_u, h_t_real_model.to(out_real_dtype), t_batch)

        mean_x0_real = gm_to_mean(gm_x0)
        mean_x0 = torch.view_as_complex(mean_x0_real.permute(0, 2, 3, 1).contiguous())
        tau2 = torch.exp(2.0 * gm_x0["logstds"].view(batch_size, 1, 1)).clamp_min(1e-6)

        # Step 2: measurement-aware posterior p(x_0 | x_t, y) = CN(mu_t, Sigma_t).
        lhs = p_p_h / sigma2 + eye_nt / tau2
        rhs = y_p_h / sigma2 + mean_x0 / tau2
        sigma_post = torch.linalg.solve(lhs, eye_nt)
        mu_post = torch.bmm(rhs, sigma_post)

        if return_mode == "mean":
            x0_post = mu_post
        else:
            chol = _safe_cholesky_hermitian(sigma_post)
            eps_post = _complex_randn(
                (batch_size, nr, nt),
                device=device,
                dtype=out_complex_dtype,
                generator=generator,
            )
            # Each row is independent; right-multiplying by chol^H gives the
            # posterior covariance of that row.
            x0_post = mu_post + torch.bmm(eps_post, chol.mH)

        # Step 3: FLOWER-style time progression with fresh source noise.
        t_next = max(1.0 - (step + 1) / total_steps, 0.0)
        if t_next > 0.0:
            eps_fresh = _complex_randn(
                (batch_size, nr, nt),
                device=device,
                dtype=out_complex_dtype,
                generator=generator,
            )
            h_t = t_next * eps_fresh + (1.0 - t_next) * x0_post
        else:
            h_t = x0_post

    return h_t


@torch.no_grad()
def flower_sample_channels(
    model: torch.nn.Module,
    y_cplx: torch.Tensor,
    p_cplx: torch.Tensor,
    noise_scale: torch.Tensor,
    mode: str = "sample",
    return_mode: str | None = None,
    *,
    total_steps: int | None = None,
    mmse_num_samples: int = 8,
    match_rcflow_budget: bool = True,
    nfe1: int | None = None,
    nfe2: int | None = None,
    seed: int = 42,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Estimate MIMO channels with the FLOWER full-posterior sampler.

    Args:
        model: Flow prior with a Gaussian-mixture head.
        y_cplx: Noisy pilot observations, shape ``(B, N_r, N_p)``, complex.
        p_cplx: Pilot matrix, shape ``(B, N_t, N_p)``, complex.
        noise_scale: Per-sample noise standard deviation, ``(B,)`` or ``(B, 1, 1)``.
        mode: One of :data:`FLOWER_MODES`.
        return_mode: Per-trajectory return mode (``"mean"`` or ``"sample"``);
            defaults to ``mode`` and is forced to ``"sample"`` for ``"mmse"``.
        total_steps: Number of reverse steps; defaults to ``nfe1 * nfe2`` when
            ``match_rcflow_budget`` is true and to ``nfe2`` otherwise.
        mmse_num_samples: Number of trajectories averaged in ``"mmse"`` mode.
        match_rcflow_budget: Match the RC-Flow total step budget (default true).
        nfe1: Outer loop override for the default step count.
        nfe2: Inner loop override for the default step count.
        seed: Seed of the internal random generator (legacy default 42).

    Returns:
        ``(h_est_cplx, diagnostics)`` with ``h_est_cplx`` shaped ``(B, N_r, N_t)``.
    """
    if mode not in FLOWER_MODES:
        raise ValueError(f"Unsupported FLOWER mode {mode!r}; expected one of {FLOWER_MODES}")
    if return_mode is None:
        return_mode = mode
    effective_return_mode = "sample" if mode == "mmse" else return_mode
    if effective_return_mode not in ("mean", "sample"):
        raise ValueError(
            "return_mode must be 'mean' or 'sample' for FLOWER trajectories"
        )

    nt = p_cplx.shape[1]
    resolved_nfe1: int | None = None
    resolved_nfe2: int | None = None
    if total_steps is None:
        dynamic_nfe1, dynamic_nfe2 = get_dynamic_nfe(noise_scale, nt)
        resolved_nfe1 = int(nfe1) if nfe1 is not None else int(dynamic_nfe1)
        resolved_nfe2 = int(nfe2) if nfe2 is not None else int(dynamic_nfe2)
        total_steps = resolved_nfe1 * resolved_nfe2 if match_rcflow_budget else resolved_nfe2
    total_steps = int(total_steps)
    if total_steps < 1:
        raise ValueError("total_steps must be positive")

    if mode == "mmse":
        trajectories = [
            _single_trajectory(
                model,
                y_cplx,
                p_cplx,
                noise_scale,
                total_steps,
                return_mode="sample",
                seed_offset=1000 * k,
                seed=seed,
            )
            for k in range(int(mmse_num_samples))
        ]
        h_est = torch.stack(trajectories, dim=0).mean(dim=0)
    else:
        h_est = _single_trajectory(
            model,
            y_cplx,
            p_cplx,
            noise_scale,
            total_steps,
            return_mode=effective_return_mode,
            seed_offset=0,
            seed=seed,
        )

    diag: Dict[str, Any] = {
        "sampler_mode": "flower",
        "flower_mode": mode,
        "return_mode": effective_return_mode,
        "total_steps": total_steps,
        "nfe1": resolved_nfe1,
        "nfe2": resolved_nfe2,
        "mmse_num_samples": int(mmse_num_samples) if mode == "mmse" else 0,
    }
    target_dtype = complex_dtype(y_cplx)
    if h_est.dtype != target_dtype:
        h_est = h_est.to(target_dtype)
    return h_est, diag
