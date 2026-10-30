"""Pilot-space physics used by DPR and PBG.

The observation model is ``Y = H P + N`` with ``H in C^{N_r x N_t}``,
``P in C^{N_t x N_p}`` and per-sample noise standard deviation ``sigma``.
All functions operate on batched complex tensors and on the cache produced by
:func:`build_physics_cache`.

Legacy names (kept for traceability, see ``docs/NAMING.md``):

* :func:`build_physics_cache` -- ``build_physics_cache``
* :func:`apply_measurement` -- ``apply_measurement``
* :func:`apply_range_projection` -- ``apply_range_projection``
* :func:`apply_null_projection` -- ``apply_null_projection``
* :func:`pbg_target` -- ``fig_measurement_interpolant``
* :func:`pbg_gradient` -- ``fig_data_consistency_gradient``
* :func:`proximal_update` -- closed-form pilot-consistency step (paper eq. (27))
"""

from __future__ import annotations

import torch

__all__ = [
    "apply_measurement",
    "apply_null_projection",
    "apply_range_projection",
    "build_physics_cache",
    "pbg_gradient",
    "pbg_target",
    "proximal_update",
]


def build_physics_cache(
    y_cplx: torch.Tensor, p_cplx: torch.Tensor, noise_scale: torch.Tensor
) -> dict[str, torch.Tensor]:
    """Pre-compute every pilot-space quantity needed by the SRF samplers.

    Args:
        y_cplx: Noisy pilot observations, shape ``(B, N_r, N_p)``.
        p_cplx: Pilot matrix, shape ``(B, N_t, N_p)``.
        noise_scale: Per-sample noise standard deviation, ``(B,)`` or ``(B, 1, 1)``.

    Returns:
        Cache with legacy-compatible keys (paper symbol in parentheses):
        ``sigma2`` (:math:`\\sigma^2`), ``p_h`` (:math:`P^H`), ``p_p_h``
        (:math:`P P^H`), ``p_pinv`` (:math:`P^\\dagger`), ``proj``
        (:math:`P P^\\dagger`), ``fig_lipschitz`` (largest eigenvalue of
        :math:`P P^H`, used to normalise the PBG gradient), ``pilot_ratio``
        (:math:`N_p / N_t`), ``y_p_h`` (:math:`Y P^H`), ``i_nt``
        (:math:`I_{N_t}`), ``y_cplx`` and ``p_cplx``.
    """
    device = y_cplx.device
    batch_size, _, npilots = y_cplx.shape
    nt = p_cplx.shape[1]
    if noise_scale.dim() == 1:
        noise_scale = noise_scale.view(batch_size, 1, 1)
    sigma2 = noise_scale.square().clamp_min(1e-12)

    p_h = p_cplx.mH
    y_p_h = torch.bmm(y_cplx, p_h)
    p_p_h = torch.bmm(p_cplx, p_h)
    p_pinv = torch.linalg.pinv(p_cplx)
    proj = torch.bmm(p_cplx, p_pinv)
    i_nt = torch.eye(nt, device=device, dtype=y_cplx.dtype).expand(batch_size, nt, nt)
    eigvals = torch.linalg.eigvalsh(p_p_h).real
    fig_lipschitz = eigvals.max(dim=1).values.clamp_min(1e-8).view(batch_size, 1, 1)

    return {
        "sigma2": sigma2,
        "p_h": p_h,
        "p_p_h": p_p_h,
        "p_pinv": p_pinv,
        "proj": proj,
        "fig_lipschitz": fig_lipschitz,
        "pilot_ratio": torch.full(
            (batch_size, 1, 1),
            float(npilots / nt),
            device=device,
            dtype=sigma2.dtype,
        ),
        "y_p_h": y_p_h,
        "i_nt": i_nt,
        "y_cplx": y_cplx,
        "p_cplx": p_cplx,
    }


def apply_measurement(h_cplx: torch.Tensor, cache: dict[str, torch.Tensor]) -> torch.Tensor:
    """Apply the pilot measurement operator ``H -> H P``."""
    return torch.bmm(h_cplx, cache["p_cplx"])


def apply_range_projection(
    h_cplx: torch.Tensor, cache: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Project ``h`` onto the row space of ``P``: ``h P P^\\dagger``."""
    return torch.bmm(h_cplx, cache["proj"])


def apply_null_projection(
    h_cplx: torch.Tensor, cache: dict[str, torch.Tensor]
) -> torch.Tensor:
    """Project ``h`` onto the null space of ``P`` (paper PBG recombination)."""
    return h_cplx - apply_range_projection(h_cplx, cache)


def pbg_target(
    y_cplx: torch.Tensor,
    epsilon_cplx: torch.Tensor,
    cache: dict[str, torch.Tensor],
    sigma_t: torch.Tensor,
    w: float,
) -> torch.Tensor:
    """Pilot-space target of PBG at time ``sigma_t`` (paper eqs. (29)-(36)).

    ``y_t = (1 - sigma_t) y + w * sigma_t * P epsilon_t`` interpolates between
    the clean observation and the current noise sample.
    """
    alpha_t = (1.0 - sigma_t).view(-1, 1, 1)
    sigma_t = sigma_t.view(-1, 1, 1)
    return alpha_t * y_cplx + w * sigma_t * apply_measurement(epsilon_cplx, cache)


def pbg_gradient(
    h_cplx: torch.Tensor,
    y_interp: torch.Tensor,
    cache: dict[str, torch.Tensor],
    normalize_by_operator: bool = True,
) -> torch.Tensor:
    """Gradient of ``||P H - y_t||^2`` used by PBG guidance.

    The gradient is normalised by the largest eigenvalue of ``P P^H`` (the
    operator Lipschitz constant) unless ``normalize_by_operator`` is false.
    """
    residual = apply_measurement(h_cplx, cache) - y_interp
    grad = torch.bmm(residual, cache["p_h"])
    if normalize_by_operator:
        grad = grad / cache["fig_lipschitz"]
    return grad


def _pilot_terms(p_cplx: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return ``(P^H, P P^H, I)`` for the closed-form proximal update."""
    batch_size, nt, _ = p_cplx.shape
    device = p_cplx.device
    p_h = p_cplx.mH
    p_p_h = torch.bmm(p_cplx, p_h)
    i_nt = torch.eye(nt, device=device, dtype=p_cplx.dtype).expand(batch_size, nt, nt)
    return p_h, p_p_h, i_nt


def _proximal_apply(
    y_p_h: torch.Tensor,
    p_p_h: torch.Tensor,
    i_nt: torch.Tensor,
    noise_scale: torch.Tensor,
    h0_cplx: torch.Tensor,
    t: float | torch.Tensor,
) -> torch.Tensor:
    """Closed-form pilot-consistency proximal update (paper eq. (27)).

    ``H_proj = (Y P^H / sigma^2 + H_0 / (v_t^2 + eps))
    (P P^H / sigma^2 + I / (v_t^2 + eps))^{-1}``
    with ``v_t^2 = t^2 / (t^2 + (1 - t)^2 + 1e-6)`` and ``H_0`` the DPR centre.
    """
    batch_size = y_p_h.shape[0]
    if not torch.is_tensor(t):
        t = torch.tensor(float(t), device=y_p_h.device, dtype=noise_scale.dtype)
    noise_scale = noise_scale.view(batch_size, 1, 1)
    vt_sq = t**2 / (t**2 + (1 - t) ** 2 + 1e-6)
    rhs = y_p_h / (noise_scale**2) + h0_cplx / (vt_sq + 1e-6)
    lhs = p_p_h / (noise_scale**2) + i_nt / (vt_sq + 1e-6)
    lhs_inv = torch.linalg.solve(lhs, i_nt)
    return torch.bmm(rhs, lhs_inv)


def proximal_update(
    y_cplx: torch.Tensor,
    h0_cplx: torch.Tensor,
    p_cplx: torch.Tensor,
    noise_scale: torch.Tensor,
    t: float | torch.Tensor,
    cache: dict[str, torch.Tensor] | None = None,
) -> torch.Tensor:
    """Closed-form pilot-consistency proximal update, paper eq. (27).

    Args:
        y_cplx: Pilot observations ``(B, N_r, N_p)``.
        h0_cplx: DPR clean-channel centre ``(B, N_r, N_t)`` (paper eq. (25)).
        p_cplx: Pilot matrix ``(B, N_t, N_p)``.
        noise_scale: Per-sample noise standard deviation, ``(B,)`` or ``(B, 1, 1)``.
        t: Diffusion time (Python scalar or tensor).
        cache: Optional :func:`build_physics_cache` output, used by the sampler to
            reuse the pre-computed ``y_p_h``, ``p_p_h`` and ``i_nt`` terms.

    Returns:
        The posterior-consistency estimate ``H_proj``, same shape as ``h0_cplx``.
    """
    if cache is not None:
        return _proximal_apply(
            cache["y_p_h"], cache["p_p_h"], cache["i_nt"], noise_scale, h0_cplx, t
        )
    p_h, p_p_h, i_nt = _pilot_terms(p_cplx)
    y_p_h = torch.bmm(y_cplx, p_h)
    return _proximal_apply(y_p_h, p_p_h, i_nt, noise_scale, h0_cplx, t)
