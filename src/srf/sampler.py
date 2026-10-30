"""SRF / SRF-P / PBG-only / RC-Flow channel-estimation samplers.

The sampler ports the legacy ``rcflow_inference`` reference implementation
line for line: an outer anchoring loop (``nfe1`` passes) wraps an inner
reverse-time loop (``nfe2`` steps), and every step performs

1. DPR: the flow prior predicts the velocity mixture ``p(u | H_t, t)``; the
   clean-channel centre is ``H_0 = H_t - t mu_theta(H_t, t)`` (paper eq. (25)),
   either taken as the mixture mean or drawn from the mixture;
2. the closed-form pilot-consistency proximal update (paper eq. (27));
3. optional PBG: a pilot-space residual correction after trajectory
   re-embedding, with null/range recombination for the full SRF mode
   (paper eqs. (29)-(36) and Algorithm 1).

Paper naming versus legacy naming (see ``docs/NAMING.md``):

* ``srf_p``    <- legacy ``baseline``: DPR + proximal update, no PBG.
* ``pbg_only`` <- legacy ``fig``: PBG guidance without null/range mixing.
* ``srf``      <- legacy ``fig_plus``: full SRF, PBG + mixing with ``rho``.
* ``rcflow``   <- legacy RC-Flow sampler (point velocity, transition ratio 1.0).
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Dict

import torch

from .baselines.flower import flower_sample_channels
from .baselines.rcflow import rcflow_sample_channels
from .gm import (
    gm_global_stats,
    gm_tau2_from_x0_gm,
    gm_to_dtype,
    gm_to_mean,
    sample_gm,
    u_to_x0_gm,
)
from .physics import (
    apply_measurement,
    apply_null_projection,
    apply_range_projection,
    build_physics_cache,
    pbg_gradient,
    pbg_target,
    proximal_update,
)
from .utils import complex_dtype, get_dynamic_nfe, model_device, model_dtype, real_dtype

__all__ = ["SamplerConfig", "sample_channels"]

VALID_MODES = ("srf", "srf_p", "pbg_only", "rcflow", "flower")
_LEGACY_MODE_ALIASES = {
    "baseline": "srf_p",
    "fig": "pbg_only",
    "fig_plus": "srf",
}
_GM_OUTPUT_MODES = ("mean", "sample")


@dataclass
class SamplerConfig:
    """Configuration of :func:`sample_channels`.

    Args:
        mode: ``"srf"`` (full SRF), ``"srf_p"`` (DPR + proximal update),
            ``"pbg_only"`` (PBG without null/range mixing), ``"rcflow"``
            (RC-Flow baseline) or ``"flower"`` (convenience delegation to
            :func:`srf.baselines.flower.flower_sample_channels`, with
            ``gm_output_mode`` selecting the FLOWER mode). Legacy names
            ``baseline`` / ``fig`` / ``fig_plus`` are accepted as aliases.
        gm_output_mode: ``"mean"`` uses the mixture mean as the DPR centre,
            ``"sample"`` draws one sample from the mixture (default; the legacy
            evaluation scripts also defaulted to ``"sample"``).
        lambda_exp: Time-warp exponent of ``t_curr`` (``LAMBDA = 2`` in the
            legacy code and in the paper setting).
        beta_exp: Time-warp exponent of ``t_next`` (``BETA = 2`` in the legacy
            code and in the paper setting).
        pbg_iters: Number of PBG guidance iterations per reverse step
            (``FIG_K = 1``).
        pbg_step: PBG step scale (``FIG_C = 0.10``; legacy name ``fig_c``).
        pbg_w: Weight of the noise term in the PBG pilot interpolant
            (``FIG_W = 1``).
        pbg_rho: Null/range recombination weight of the full SRF mode
            (``FIG_PLUS_MIX = 0.08``; legacy name ``fig_plus_mix``).
        pbg_noise_coef: Noise gate coefficient of the PBG step
            (``FIG_NOISE_COEF = 1``).
        pbg_eps: Numerical epsilon of the PBG step.
        pbg_step_max: Upper bound of the PBG step (``FIG_SCALE_MAX = 0.5``).
        nfe1: Override of the outer loop count (``None`` uses
            :func:`srf.utils.get_dynamic_nfe`).
        nfe2: Override of the inner loop count (``None`` uses
            :func:`srf.utils.get_dynamic_nfe`).
    """

    mode: str = "srf"
    gm_output_mode: str = "sample"
    lambda_exp: float = 2.0
    beta_exp: float = 2.0
    pbg_iters: int = 1
    pbg_step: float = 0.10
    pbg_w: float = 1.0
    pbg_rho: float = 0.08
    pbg_noise_coef: float = 1.0
    pbg_eps: float = 1e-6
    pbg_step_max: float = 0.5
    nfe1: int | None = None
    nfe2: int | None = None

    def __post_init__(self) -> None:
        mode = str(self.mode).strip().lower()
        mode = _LEGACY_MODE_ALIASES.get(mode, mode)
        if mode not in VALID_MODES:
            raise ValueError(f"Unsupported sampler mode {self.mode!r}; expected one of {VALID_MODES}")
        self.mode = mode
        if self.gm_output_mode not in _GM_OUTPUT_MODES:
            raise ValueError(
                f"Unsupported gm_output_mode {self.gm_output_mode!r}; expected one of {_GM_OUTPUT_MODES}"
            )
        if self.pbg_iters < 0:
            raise ValueError("pbg_iters must be non-negative")
        for name in ("nfe1", "nfe2"):
            value = getattr(self, name)
            if value is not None and int(value) < 1:
                raise ValueError(f"{name} must be positive when provided")


def _legacy_complex_noise(reference: torch.Tensor) -> torch.Tensor:
    """Complex noise initialisation reproduced from the legacy implementation.

    The legacy expression is ``(randn_like(x) + 1j * randn_like(x)) / sqrt(2)``
    with a complex reference tensor.  It is kept verbatim (see
    ``docs/NAMING.md``) so that inference matches the reference numbers.
    """
    return (torch.randn_like(reference) + 1j * torch.randn_like(reference)) / (2**0.5)


def _mean_accumulator_update(acc: Dict[str, float], stats: Dict[str, Any]) -> None:
    for key, value in stats.items():
        if torch.is_tensor(value):
            acc[key] += float(value.detach().mean().cpu())
        else:
            acc[key] += float(value)
    acc["_count"] += 1.0


def _mean_accumulator_finalize(acc: Dict[str, float]) -> Dict[str, float]:
    count = max(acc.pop("_count", 1.0), 1.0)
    return {key: value / count for key, value in acc.items()}


def _zero_pbg_stats(device: torch.device) -> Dict[str, torch.Tensor]:
    zero = torch.tensor(0.0, device=device)
    return {
        "pbg_residual_mean": zero,
        "pbg_grad_norm_mean": zero,
        "pbg_step_mean": zero,
    }


def _apply_pbg_guidance(
    h_next: torch.Tensor,
    h0_mean: torch.Tensor,
    epsilon_step: torch.Tensor,
    cache: Dict[str, torch.Tensor],
    sigma_next: torch.Tensor,
    sigma_curr: torch.Tensor,
    cfg: SamplerConfig,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Apply PBG guidance to the re-embedded trajectory (paper eqs. (29)-(36))."""
    if cfg.pbg_iters <= 0:
        return h_next, {
            **_zero_pbg_stats(h_next.device),
            "pbg_lipschitz_mean": cache["fig_lipschitz"].mean(),
        }

    alpha_curr = (1.0 - sigma_curr).clamp_min(cfg.pbg_eps).view(-1, 1, 1)
    sigma_curr = sigma_curr.view(-1, 1, 1)
    y_interp = pbg_target(
        y_cplx=cache["y_cplx"],
        epsilon_cplx=epsilon_step,
        cache=cache,
        sigma_t=sigma_next,
        w=cfg.pbg_w,
    )

    noise_gate = 1.0 / (1.0 + cfg.pbg_noise_coef * cache["sigma2"])
    pilot_gate = cache["pilot_ratio"].clamp(max=1.0)

    grad_norms = []
    step_means = []
    for _ in range(int(cfg.pbg_iters)):
        grad = pbg_gradient(h_next, y_interp, cache)
        step = (cfg.pbg_step * sigma_curr / alpha_curr * noise_gate * pilot_gate).clamp_max(
            cfg.pbg_step_max
        )
        h_next = h_next - step * grad
        grad_norms.append(torch.linalg.norm(grad.reshape(grad.shape[0], -1), dim=1).mean())
        step_means.append(step.mean())

    if cfg.mode == "srf":
        # Null/range recombination of the full SRF mode (paper Algorithm 1).
        h_tilde = (1.0 - sigma_next).view(-1, 1, 1) * h0_mean + sigma_next.view(
            -1, 1, 1
        ) * epsilon_step
        null_part = apply_null_projection(h_tilde, cache)
        range_part = apply_range_projection(h_next, cache)
        h_next = (1.0 - cfg.pbg_rho) * h_next + cfg.pbg_rho * (null_part + range_part)

    residual = apply_measurement(h_next, cache) - y_interp
    residual_norm = torch.linalg.norm(residual.reshape(residual.shape[0], -1), dim=1).mean()
    grad_norm = torch.stack(grad_norms).mean()
    step_mean = torch.stack(step_means).mean()
    return h_next, {
        "pbg_residual_mean": residual_norm,
        "pbg_grad_norm_mean": grad_norm,
        "pbg_step_mean": step_mean,
        "pbg_lipschitz_mean": cache["fig_lipschitz"].mean(),
    }


def _resolve_nfe(
    noise_scale: torch.Tensor, nt: int, cfg: SamplerConfig
) -> tuple[int, int]:
    """Return ``(nfe1, nfe2)`` using the dynamic schedule unless overridden."""
    if cfg.nfe1 is not None and cfg.nfe2 is not None:
        return int(cfg.nfe1), int(cfg.nfe2)
    dynamic_nfe1, dynamic_nfe2 = get_dynamic_nfe(noise_scale, nt)
    nfe1 = int(cfg.nfe1) if cfg.nfe1 is not None else int(dynamic_nfe1)
    nfe2 = int(cfg.nfe2) if cfg.nfe2 is not None else int(dynamic_nfe2)
    return nfe1, nfe2


@torch.no_grad()
def sample_channels(
    model: torch.nn.Module,
    y_cplx: torch.Tensor,
    p_cplx: torch.Tensor,
    noise_scale: torch.Tensor,
    cfg: SamplerConfig | None = None,
) -> tuple[torch.Tensor, Dict[str, Any]]:
    """Estimate MIMO channels from noisy pilots with SRF (paper Algorithm 1).

    Args:
        model: Flow prior with a Gaussian-mixture output head (e.g.
            :class:`srf.models.GMUNet`).  For ``mode="rcflow"`` a point-velocity
            head is also accepted.
        y_cplx: Noisy pilot observations, shape ``(B, N_r, N_p)``, complex.
        p_cplx: Pilot matrix, shape ``(B, N_t, N_p)``, complex.
        noise_scale: Per-sample noise standard deviation, ``(B,)`` or ``(B, 1, 1)``.
        cfg: Sampler configuration; defaults to the full SRF setting.

    Returns:
        ``(h_est_cplx, diagnostics)`` where ``h_est_cplx`` has shape
        ``(B, N_r, N_t)`` and ``diagnostics`` holds mean scalar statistics
        (mixture entropy, learned variance, PBG residual/gradient/step, the
        used ``nfe1`` / ``nfe2`` and the effective sampler settings).

    Mode semantics:
        * ``"srf_p"`` -- DPR centre plus proximal update, no PBG;
        * ``"pbg_only"`` -- PBG guidance without null/range mixing;
        * ``"srf"`` -- full SRF: PBG plus ``rho`` null/range recombination;
        * ``"rcflow"`` -- legacy RC-Flow point-velocity sampler;
        * ``"flower"`` -- FLOWER full-posterior baseline.
    """
    if cfg is None:
        cfg = SamplerConfig()
    if cfg.mode == "rcflow":
        return rcflow_sample_channels(
            model,
            y_cplx,
            p_cplx,
            noise_scale,
            lambda_exp=cfg.lambda_exp,
            beta_exp=cfg.beta_exp,
            gm_output_mode=cfg.gm_output_mode,
            nfe1=cfg.nfe1,
            nfe2=cfg.nfe2,
        )
    if cfg.mode == "flower":
        return flower_sample_channels(
            model,
            y_cplx,
            p_cplx,
            noise_scale,
            mode=cfg.gm_output_mode,
            nfe1=cfg.nfe1,
            nfe2=cfg.nfe2,
        )

    device = model_device(model)
    batch_size = y_cplx.shape[0]
    nt = p_cplx.shape[1]
    nfe1, nfe2 = _resolve_nfe(noise_scale, nt, cfg)
    out_real_dtype = real_dtype(y_cplx)
    mdtype = model_dtype(model)

    cache = build_physics_cache(y_cplx, p_cplx, noise_scale)
    y_p_h = cache["y_p_h"]
    noise_scale_b = noise_scale.view(batch_size, 1, 1)

    dt = 1.0 / nfe2
    h_t = _legacy_complex_noise(y_p_h)

    diag_acc: Dict[str, float] = defaultdict(float)

    for outer_idx in range(nfe1):
        epsilon_anchor = h_t
        for inner_idx in range(nfe2):
            t_curr = (1.0 - inner_idx * dt) ** cfg.lambda_exp
            t_batch_model = torch.full((batch_size,), t_curr, device=device, dtype=mdtype)
            t_batch = torch.full((batch_size,), t_curr, device=device, dtype=out_real_dtype)

            h_t_real_model = (
                torch.view_as_real(h_t).permute(0, 3, 1, 2).contiguous().to(mdtype)
            )
            gm_u = model(h_t_real_model, t_batch_model)
            if not isinstance(gm_u, dict):
                raise TypeError(
                    "sample_channels expects a Gaussian-mixture model output; "
                    "use SamplerConfig(mode='rcflow') for point-velocity models"
                )
            gm_u = gm_to_dtype(gm_u, out_real_dtype)
            h_t_real = h_t_real_model.to(out_real_dtype)

            gm_stats = gm_global_stats(gm_u)
            # DPR clean-channel centre H_0 = H_t - t mu_theta(H_t, t), eq. (25).
            gm_x0 = u_to_x0_gm(gm_u, h_t_real, t_batch)
            if cfg.gm_output_mode == "sample":
                h0_real = sample_gm(gm_x0)
            else:
                h0_real = gm_to_mean(gm_x0)
            h0 = torch.view_as_complex(h0_real.permute(0, 2, 3, 1).contiguous())
            tau2_learned = gm_tau2_from_x0_gm(gm_x0)

            # Closed-form pilot-consistency proximal update, eq. (27).
            vt_sq = t_curr**2 / (t_curr**2 + (1 - t_curr) ** 2 + 1e-6)
            h0_y = proximal_update(
                cache["y_cplx"],
                h0,
                cache["p_cplx"],
                noise_scale_b,
                t_curr,
                cache=cache,
            )

            t_next = max((1.0 - (inner_idx + 1) * dt) ** cfg.beta_exp, 0.0)
            sigma_next = torch.full((batch_size,), t_next, device=device, dtype=out_real_dtype)
            sigma_curr = torch.full((batch_size,), t_curr, device=device, dtype=out_real_dtype)

            if outer_idx == 0:
                epsilon_step = _legacy_complex_noise(h_t)
            else:
                epsilon_step = epsilon_anchor

            h_next = t_next * epsilon_step + (1 - t_next) * h0_y

            if cfg.mode in ("pbg_only", "srf") and t_next > 0.0:
                h_next, pbg_stats = _apply_pbg_guidance(
                    h_next=h_next,
                    h0_mean=h0,
                    epsilon_step=epsilon_step,
                    cache=cache,
                    sigma_next=sigma_next,
                    sigma_curr=sigma_curr,
                    cfg=cfg,
                )
            else:
                pbg_stats = {
                    **_zero_pbg_stats(device),
                    "pbg_lipschitz_mean": cache["fig_lipschitz"].mean(),
                }

            h_t = h_next

            _mean_accumulator_update(
                diag_acc,
                {
                    **gm_stats,
                    "tau2_learned_mean": tau2_learned.mean(),
                    "rc_var_mean": torch.tensor(vt_sq, device=device, dtype=out_real_dtype),
                    "t_curr_mean": torch.tensor(t_curr, device=device, dtype=out_real_dtype),
                    "t_next_mean": torch.tensor(t_next, device=device, dtype=out_real_dtype),
                    **pbg_stats,
                },
            )

    diag = _mean_accumulator_finalize(diag_acc)
    diag["nfe1"] = int(nfe1)
    diag["nfe2"] = int(nfe2)
    diag["sampler_mode"] = cfg.mode
    diag["pbg_iters"] = int(cfg.pbg_iters)
    diag["pbg_step"] = float(cfg.pbg_step)
    diag["pbg_w"] = float(cfg.pbg_w)
    diag["pbg_rho"] = float(cfg.pbg_rho)
    diag["pbg_noise_coef"] = float(cfg.pbg_noise_coef)

    target_dtype = complex_dtype(y_cplx)
    if h_t.dtype != target_dtype:
        h_t = h_t.to(target_dtype)
    return h_t, diag
