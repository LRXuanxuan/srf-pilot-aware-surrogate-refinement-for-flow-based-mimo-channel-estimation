"""Channel dataset for the SRF experiments.

The dataset loads MATLAB ``.mat`` files that store three complex variables
``H_train``, ``H_val`` and ``H_test``, each shaped ``(N_r, N_t, N_samples)``,
and keeps the legacy pipeline: QPSK pilots, normalised channels, additive
complex Gaussian noise whose power follows ``N_t / SNR``.

``hdf5storage`` is imported lazily so that ``import srf`` works in
environments where the MATLAB I/O dependency is not installed.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch.utils.data import Dataset

__all__ = ["ChannelConfig", "MatChannelDataset", "save_mat_dataset"]


@dataclass
class ChannelConfig:
    """Minimal configuration object expected by :class:`MatChannelDataset`."""

    image_size: tuple[int, int]
    num_pilots: int
    snr_db: float = 1.0


def save_mat_dataset(
    path: str | Path,
    H_train: np.ndarray,
    H_val: np.ndarray,
    H_test: np.ndarray,
) -> None:
    """Write a synthetic ``.mat`` dataset in the layout read by the dataset.

    Args:
        path: Destination ``.mat`` file.
        H_train: Training channels, shaped ``(N_r, N_t, N_samples)``.
        H_val: Validation channels, same layout.
        H_test: Test channels, same layout.
    """
    import hdf5storage

    arrays = {
        "H_train": np.asarray(H_train, dtype=np.complex64),
        "H_val": np.asarray(H_val, dtype=np.complex64),
        "H_test": np.asarray(H_test, dtype=np.complex64),
    }
    hdf5storage.savemat(str(path), arrays, format="7.3", oned_as="row")


class MatChannelDataset(Dataset):
    """Pilot-based MIMO channel dataset used by training and evaluation.

    Args:
        file_list: ``.mat`` files containing ``H_train`` / ``H_val`` / ``H_test``.
        split: One of ``"train"``, ``"val"`` or ``"test"``.
        config: Object exposing ``image_size`` ``(N_r, N_t)``, ``num_pilots`` and
            ``snr_db``.  :class:`ChannelConfig` is provided for convenience.
        seed: Seed of the pilot generator (same convention as the legacy code).

    The attributes ``snr_db``, ``noise_power`` and ``noise_sigma`` are public and
    may be overwritten after construction to sweep the SNR (legacy behaviour).
    """

    def __init__(
        self,
        file_list: Sequence[str | Path],
        split: str,
        config: Any,
        seed: int = 42,
    ):
        import hdf5storage

        self.split = split
        self.config = config
        self.n_rx = config.image_size[0]
        self.n_tx = config.image_size[1]
        self.snr_db = config.snr_db

        channels_list = []
        for fpath in file_list:
            try:
                data = hdf5storage.loadmat(str(fpath))
            except Exception as exc:  # noqa: BLE001 - re-raised with context
                raise RuntimeError(f"Failed to load {fpath}: {exc}") from exc

            key = f"H_{split}"
            if key not in data:
                raise KeyError(f"{fpath} does not contain variable {key}")

            h = data[key].astype(np.complex64)
            if h.ndim == 3 and h.shape[0] == self.n_rx and h.shape[1] == self.n_tx:
                h = np.transpose(h, (2, 0, 1))
            elif h.ndim == 3 and h.shape[1] == self.n_rx and h.shape[2] == self.n_tx:
                pass
            else:
                raise ValueError(f"Unexpected shape {h.shape} in {fpath}")
            channels_list.append(h)

        self.channels = np.concatenate(channels_list, axis=0)
        self.channel_scale = np.std(self.channels)

        np.random.seed(seed)
        num_samples = self.channels.shape[0]
        num_pilots = config.num_pilots
        pilot_bits_real = (
            2 * np.random.binomial(1, 0.5, size=(num_samples, self.n_tx, num_pilots)) - 1
        )
        pilot_bits_imag = (
            2 * np.random.binomial(1, 0.5, size=(num_samples, self.n_tx, num_pilots)) - 1
        )
        self.pilots = (pilot_bits_real + 1j * pilot_bits_imag) / np.sqrt(2.0)

        snr_linear = 10 ** (self.snr_db / 10)
        self.noise_power = self.n_tx / snr_linear
        self.noise_sigma = np.sqrt(self.noise_power)

    def __len__(self) -> int:
        return len(self.channels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        h = self.channels[idx] / self.channel_scale
        p = self.pilots[idx]
        y = np.matmul(h, p)

        noise = (
            self.noise_sigma
            * (np.random.normal(size=y.shape) + 1j * np.random.normal(size=y.shape))
            / np.sqrt(2.0)
        )
        y_noisy = y + noise

        return {
            "H": torch.from_numpy(h.astype(np.complex64)),
            "P": torch.from_numpy(p.astype(np.complex64)),
            "Y": torch.from_numpy(y_noisy.astype(np.complex64)),
            "channel_scale": torch.tensor(self.channel_scale, dtype=torch.float32),
            "noise_scale": torch.tensor(self.noise_sigma, dtype=torch.float32),
        }
