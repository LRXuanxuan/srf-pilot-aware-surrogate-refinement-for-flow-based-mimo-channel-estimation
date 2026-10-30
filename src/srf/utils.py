"""Small runtime helpers shared by the SRF samplers and training objectives."""

from __future__ import annotations

import math

import torch

__all__ = [
    "complex_dtype",
    "get_dynamic_nfe",
    "model_device",
    "model_dtype",
    "real_dtype",
]


def get_dynamic_nfe(noise_sigma: float | torch.Tensor, nt: int) -> tuple[int, int]:
    """Adaptive number of reverse-time steps used by the SRF samplers.

    The outer loop is fixed (``nfe1 = 100``) while the inner loop ``nfe2`` grows
    with the channel noise level; the schedule is unchanged from the legacy
    implementation.

    Args:
        noise_sigma: Noise standard deviation (scalar or batch tensor).
        nt: Number of transmit antennas.

    Returns:
        ``(nfe1, nfe2)`` with ``nfe1 = 100`` and ``nfe2 >= 2``.
    """
    sigma_min = (nt / 1000) ** 0.5
    sigma_max = (nt * 10) ** 0.5
    if torch.is_tensor(noise_sigma):
        noise_sigma = noise_sigma.mean().item()
    ratio = (
        (math.log10(sigma_max / noise_sigma)) / (math.log10(sigma_max / sigma_min))
    ) ** 2
    n_max = 50
    n_min = 3
    nfe2 = int(n_min + ratio * (n_max - n_min))
    return 100, max(nfe2, 2)


def model_device(model: torch.nn.Module) -> torch.device:
    """Return the device of a model's first parameter (CPU if parameterless)."""
    param = next(model.parameters(), None)
    return param.device if param is not None else torch.device("cpu")


def model_dtype(model: torch.nn.Module) -> torch.dtype:
    """Return the floating-point parameter dtype of a model (float32 fallback)."""
    param = next(model.parameters(), None)
    if param is None or not param.dtype.is_floating_point:
        return torch.float32
    return param.dtype


def real_dtype(x: torch.Tensor) -> torch.dtype:
    """Return the real floating-point dtype matching ``x``."""
    return torch.float64 if x.dtype == torch.complex128 else torch.float32


def complex_dtype(x: torch.Tensor) -> torch.dtype:
    """Return the complex dtype matching ``x`` (complex64 unless double)."""
    return torch.complex128 if x.dtype == torch.complex128 else torch.complex64
