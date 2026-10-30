"""RC-Flow baseline sampler (comparison method of the SRF paper).

The sampler ports the legacy ``rcflow_inference`` of ``test_rcflow.py``: the same
anchored reverse-time loop as SRF, but without DPR's finite-transition prior
refinement and without PBG.  It consumes a model that predicts the point
displacement ``u`` of the additive path ``H_t = H_0 + t u``.  Both head types
found in the legacy code are supported:

* a Gaussian-mixture head (e.g. :class:`srf.models.GMUNet` with ``K = 1``), in
  which case the point prediction is the mixture mean or a mixture sample;
* a plain velocity head returning a ``(B, C, H, W)`` tensor, which is used as
  the displacement directly.
"""

from __future__ import annotations

from typing import Any, Dict

import torch

from ..gm import gm_to_dtype, gm_to_mean, sample_gm, u_to_x0_gm
from ..physics import build_physics_cache, proximal_update
from ..utils import complex_dtype, get_dynamic_nfe, model_device, model_dtype, real_dtype

__all__ = ["rcflow_sample_channels"]


@torch.no_grad()
def rcflow_sample_channels(
    model: torch.nn.Module,
    y_cplx: torch.Tensor,
    p_cplx: torch.Tensor,
    noise_scale: torch.Tensor,
    *,
    lambda_exp: float = 2.0,
    beta_exp: float = 2.0,
    gm_output_mode: str = "sample",
    nfe1: int | None = None,
    nfe2: int | None = None,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Estimate MIMO channels with the RC-Flow point-velocity baseline.

    Args:
        model: Point-velocity model (tensor output) or flow prior with a
            Gaussian-mixture head (dictionary output).
        y_cplx: Noisy pilot observations, shape ``(B, N_r, N_p)``, complex.
        p_cplx: Pilot matrix, shape ``(B, N_t, N_p)``, complex.
        noise_scale: Per-sample noise standard deviation, ``(B,)`` or ``(B, 1, 1)``.
        lambda_exp: Time-warp exponent of ``t_curr`` (``LAMBDA = 2``).
        beta_exp: Time-warp exponent of ``t_next`` (``BETA = 2``).
        gm_output_mode: ``"mean"`` or ``"sample"`` for Gaussian-mixture heads.
        nfe1: Outer loop override (``None`` uses :func:`srf.utils.get_dynamic_nfe`).
        nfe2: Inner loop override (``None`` uses :func:`srf.utils.get_dynamic_nfe`).

    Returns:
        ``(h_est_cplx, diagnostics)`` with ``h_est_cplx`` shaped ``(B, N_r, N_t)``.
    """
    device = model_device(model)
    batch_size = y_cplx.shape[0]
    nt = p_cplx.shape[1]

    if nfe1 is None or nfe2 is None:
        dynamic_nfe1, dynamic_nfe2 = get_dynamic_nfe(noise_scale, nt)
        nfe1 = int(dynamic_nfe1) if nfe1 is None else int(nfe1)
        nfe2 = int(dynamic_nfe2) if nfe2 is None else int(nfe2)
    nfe1, nfe2 = int(nfe1), int(nfe2)

    out_real_dtype = real_dtype(y_cplx)
    mdtype = model_dtype(model)
    cache = build_physics_cache(y_cplx, p_cplx, noise_scale)
    noise_scale_b = noise_scale.view(batch_size, 1, 1)

    dt = 1.0 / nfe2
    h_t = (torch.randn_like(cache["y_p_h"]) + 1j * torch.randn_like(cache["y_p_h"])) / (
        2**0.5
    )
    model_output_kind = "gm"

    for outer_idx in range(nfe1):
        epsilon_anchor = h_t
        for inner_idx in range(nfe2):
            t_curr = (1.0 - inner_idx * dt) ** lambda_exp
            t_batch_model = torch.full((batch_size,), t_curr, device=device, dtype=mdtype)
            t_batch = torch.full((batch_size,), t_curr, device=device, dtype=out_real_dtype)

            h_t_real_model = (
                torch.view_as_real(h_t).permute(0, 3, 1, 2).contiguous().to(mdtype)
            )
            model_output = model(h_t_real_model, t_batch_model)
            h_t_real = h_t_real_model.to(out_real_dtype)

            if isinstance(model_output, dict):
                gm_u = gm_to_dtype(model_output, out_real_dtype)
                gm_x0 = u_to_x0_gm(gm_u, h_t_real, t_batch)
                if gm_output_mode == "sample":
                    h0_real = sample_gm(gm_x0)
                else:
                    h0_real = gm_to_mean(gm_x0)
            else:
                model_output_kind = "velocity"
                h0_real = h_t_real - t_curr * model_output.to(out_real_dtype)
            h0 = torch.view_as_complex(h0_real.permute(0, 2, 3, 1).contiguous())

            h0_y = proximal_update(
                cache["y_cplx"],
                h0,
                cache["p_cplx"],
                noise_scale_b,
                t_curr,
                cache=cache,
            )

            t_next = (1.0 - (inner_idx + 1) * dt) ** beta_exp
            if outer_idx == 0:
                epsilon_step = (
                    torch.randn_like(h_t) + 1j * torch.randn_like(h_t)
                ) / (2**0.5)
            else:
                epsilon_step = epsilon_anchor
            h_t = t_next * epsilon_step + (1 - t_next) * h0_y

    diag: Dict[str, Any] = {
        "nfe1": int(nfe1),
        "nfe2": int(nfe2),
        "sampler_mode": "rcflow",
        "gm_output_mode": gm_output_mode,
        "model_output": model_output_kind,
    }
    target_dtype = complex_dtype(y_cplx)
    if h_t.dtype != target_dtype:
        h_t = h_t.to(target_dtype)
    return h_t, diag
