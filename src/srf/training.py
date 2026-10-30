"""Training objectives and checkpoint helpers for the SRF flow prior.

Two objectives are provided:

* :class:`GMFlowObjective` -- the DPR objective actually used for SRF: a
  Gaussian-mixture negative log-likelihood of the finite-transition target
  ``t_high -> t_low`` (transition ratio ``r = 0.5``), with the SNR drawn
  uniformly from ``{-10, -5, ..., 30}`` dB and the time sampled as
  ``sigmoid(N(time_mu, time_sigma))``.
* :class:`FlowMatchingObjective` -- the classic point-velocity flow-matching
  objective of the RC-Flow baseline (additive path, unweighted MSE between the
  predicted displacement and the noise realisation).

Both are plain callables ``objective(model, batch) -> dict`` with the scalar
``"loss"`` plus detached metrics, so they can be used by any training loop.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np
import torch

from .gm import gm_nll_loss, gm_to_mean, reverse_transition_additive
from .utils import model_device

__all__ = [
    "FlowMatchingObjective",
    "GMFlowObjective",
    "extract_state_dict",
    "load_checkpoint",
    "strip_state_dict_prefix",
]


def _draw_snr_db(snr_list_db: np.ndarray) -> float:
    """Draw one SNR value (dB) from the discrete paper grid."""
    return float(np.random.choice(snr_list_db))


def _additive_noise(h_true_cplx: torch.Tensor, snr_db: float) -> torch.Tensor:
    """Channel-domain noise ``u ~ CN(0, 1/SNR)`` for the additive path."""
    snr_linear = 10 ** (snr_db / 10)
    return (1 / snr_linear) ** 0.5 * (
        torch.randn_like(h_true_cplx) + 1j * torch.randn_like(h_true_cplx)
    ) / (2 ** 0.5)


def _snr_grid(snr_min_db: float, snr_max_db: float, snr_step_db: float) -> np.ndarray:
    """Inclusive SNR grid matching the legacy ``np.arange`` construction."""
    return np.arange(snr_min_db, snr_max_db + snr_step_db, snr_step_db)


class GMFlowObjective:
    """DPR finite-transition Gaussian-mixture NLL objective.

    Args:
        snr_min_db: Lowest SNR of the uniform training grid (dB).
        snr_max_db: Highest SNR of the uniform training grid (dB).
        snr_step_db: SNR grid step (dB).
        transition_ratio: Transition ratio ``r``; the target time is
            ``t_low = (1 - r) * t_high`` (paper setting ``r = 0.5``).
        time_mu: Mean of the logit-normal time sampler.  The legacy entry
            points used ``0.0`` (the ``-0.4`` class default of the old
            ``GMFlowMatchingLoss`` was never passed in training).
        time_sigma: Standard deviation of the logit-normal time sampler.
    """

    def __init__(
        self,
        snr_min_db: float = -10.0,
        snr_max_db: float = 30.0,
        snr_step_db: float = 5.0,
        transition_ratio: float = 0.5,
        time_mu: float = 0.0,
        time_sigma: float = 1.0,
    ):
        self.snr_list_db = _snr_grid(snr_min_db, snr_max_db, snr_step_db)
        self.transition_ratio = transition_ratio
        self.time_mu = time_mu
        self.time_sigma = time_sigma

    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample training times ``sigmoid(N(time_mu, time_sigma))``."""
        normal_samples = torch.randn(batch_size, device=device) * self.time_sigma + self.time_mu
        return torch.sigmoid(normal_samples)

    def draw_snr_db(self) -> float:
        """Draw one SNR (dB) uniformly from the discrete training grid."""
        return _draw_snr_db(self.snr_list_db)

    def __call__(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = model_device(model)
        h_true_cplx = batch["H"].to(device)
        batch_size = h_true_cplx.shape[0]

        snr_db = self.draw_snr_db()
        noise_cplx = _additive_noise(h_true_cplx, snr_db)

        t_high = self.sample_time(batch_size, device).clamp(min=1e-4)
        t_low = (t_high * (1.0 - self.transition_ratio)).clamp(min=0.0)

        t_high_b = t_high.view(-1, 1, 1)
        t_low_b = t_low.view(-1, 1, 1)

        x_t_high_real = (
            torch.view_as_real(h_true_cplx + t_high_b * noise_cplx)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        x_t_low_real = (
            torch.view_as_real(h_true_cplx + t_low_b * noise_cplx)
            .permute(0, 3, 1, 2)
            .contiguous()
        )
        u_target_real = (
            torch.view_as_real(noise_cplx).permute(0, 3, 1, 2).contiguous()
        )

        gm_u = model(x_t_high_real, t_high)
        gm_xt_low = reverse_transition_additive(gm_u, x_t_high_real, t_low, t_high)
        nll = gm_nll_loss(gm_xt_low, x_t_low_real)

        # Mean-field velocity MSE, kept as a detached training/validation metric.
        mean_u = gm_to_mean(gm_u)
        error = mean_u - u_target_real
        error_norm = torch.norm(error.reshape(batch_size, -1), dim=1)
        unweighted_mse = torch.mean(error_norm**2)

        return {
            "loss": nll,
            "nll": nll.detach(),
            "unweighted_mse": unweighted_mse.detach(),
        }


class FlowMatchingObjective:
    """Classic point-velocity flow-matching objective (RC-Flow baseline).

    The model predicts the displacement ``u`` of the additive path
    ``H_t = H_0 + t u``; the loss is the unweighted mean squared error between
    the predicted displacement and the sampled noise realisation.

    Args:
        snr_min_db: Lowest SNR of the uniform training grid (dB).
        snr_max_db: Highest SNR of the uniform training grid (dB).
        snr_step_db: SNR grid step (dB).
        time_mu: Mean of the logit-normal time sampler.
        time_sigma: Standard deviation of the logit-normal time sampler.
    """

    def __init__(
        self,
        snr_min_db: float = -10.0,
        snr_max_db: float = 30.0,
        snr_step_db: float = 5.0,
        time_mu: float = 0.0,
        time_sigma: float = 1.0,
    ):
        self.snr_list_db = _snr_grid(snr_min_db, snr_max_db, snr_step_db)
        self.time_mu = time_mu
        self.time_sigma = time_sigma

    def sample_time(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Sample training times ``sigmoid(N(time_mu, time_sigma))``."""
        normal_samples = torch.randn(batch_size, device=device) * self.time_sigma + self.time_mu
        return torch.sigmoid(normal_samples)

    def draw_snr_db(self) -> float:
        """Draw one SNR (dB) uniformly from the discrete training grid."""
        return _draw_snr_db(self.snr_list_db)

    def __call__(self, model: torch.nn.Module, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        device = model_device(model)
        h_true_cplx = batch["H"].to(device)
        batch_size = h_true_cplx.shape[0]

        snr_db = self.draw_snr_db()
        u_target_cplx = _additive_noise(h_true_cplx, snr_db)

        t = self.sample_time(batch_size, device)
        x_t_cplx = h_true_cplx + t.view(-1, 1, 1) * u_target_cplx

        x_t_real = torch.view_as_real(x_t_cplx).permute(0, 3, 1, 2).contiguous()
        u_target_real = (
            torch.view_as_real(u_target_cplx).permute(0, 3, 1, 2).contiguous()
        )

        model_output = model(x_t_real, t)
        # ``GMUNet`` returns a mixture dictionary; a pure velocity head returns a
        # tensor. In both cases the point prediction is the displacement ``u``.
        pred_u = gm_to_mean(model_output) if isinstance(model_output, dict) else model_output
        error = pred_u - u_target_real
        mse = torch.mean(torch.norm(error.reshape(batch_size, -1), dim=1) ** 2)

        return {"loss": mse, "mse": mse.detach()}


def extract_state_dict(ckpt_obj: Any) -> Dict[str, torch.Tensor]:
    """Extract a state dict from a raw or wrapped checkpoint object."""
    if isinstance(ckpt_obj, dict):
        for key in ("state_dict", "model", "ema_model", "model_state_dict", "ema_state_dict"):
            if key in ckpt_obj and isinstance(ckpt_obj[key], dict):
                return ckpt_obj[key]
        return ckpt_obj
    raise TypeError(f"Unsupported checkpoint object type: {type(ckpt_obj)}")


def strip_state_dict_prefix(state_dict: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Strip a uniform ``module.`` prefix produced by ``DataParallel``/DDP."""
    if state_dict and all(key.startswith("module.") for key in state_dict):
        return {key[len("module."):]: value for key, value in state_dict.items()}
    return state_dict


def load_checkpoint(
    model: torch.nn.Module,
    path: str,
    device: str | torch.device,
    strict: bool = True,
) -> tuple[list[str], list[str]]:
    """Load a checkpoint into ``model`` and return ``(missing, unexpected)`` keys."""
    try:
        ckpt = torch.load(path, map_location=device, weights_only=False)
    except TypeError:  # torch < 1.13 has no ``weights_only`` keyword
        ckpt = torch.load(path, map_location=device)
    state_dict = strip_state_dict_prefix(extract_state_dict(ckpt))
    return model.load_state_dict(state_dict, strict=strict)
