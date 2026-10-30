#!/usr/bin/env python
"""Generate small synthetic CDL-like channel files for smoke tests.

This is NOT the CDL generator used for the paper. It creates spatially
correlated Rayleigh channels (Kronecker model with exponential correlation at
both ends) so that the training and evaluation entry points can be exercised
end-to-end without any external data.

The files follow the data contract in ``docs/DATA.md``: each ``channel_CDL-X.mat``
contains ``H_train``, ``H_val`` and ``H_test`` of shape ``(N_r, N_t, N)``.
"""
from __future__ import annotations

import argparse
import os

import numpy as np


def _exp_corr_matrix(n: int, rho: float) -> np.ndarray:
    idx = np.arange(n)
    return rho ** np.abs(idx[:, None] - idx[None, :])


def _make_channels(
    num_rx: int,
    num_tx: int,
    num_samples: int,
    rng: np.random.Generator,
    rx_corr: float,
    tx_corr: float,
) -> np.ndarray:
    rr = _exp_corr_matrix(num_rx, rx_corr)
    rt = _exp_corr_matrix(num_tx, tx_corr)
    # Cholesky factors of the receive and transmit correlation matrices.
    lr = np.linalg.cholesky(rr + 1e-6 * np.eye(num_rx))
    lt = np.linalg.cholesky(rt + 1e-6 * np.eye(num_tx))
    z = (rng.standard_normal((num_samples, num_rx, num_tx))
         + 1j * rng.standard_normal((num_samples, num_rx, num_tx))) / np.sqrt(2.0)
    # H = Lr Z Lt^T  (Kronecker correlation model).
    h = np.einsum("ij,njk,kl->nil", lr, z, lt.T)
    return np.transpose(h, (1, 2, 0)).astype(np.complex64)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="data/synthetic")
    parser.add_argument("--num-rx", type=int, default=8)
    parser.add_argument("--num-tx", type=int, default=16)
    parser.add_argument("--n-train", type=int, default=128)
    parser.add_argument("--n-val", type=int, default=16)
    parser.add_argument("--n-test", type=int, default=32)
    parser.add_argument("--cdl-types", nargs="+", default=["A", "B", "C", "D"])
    parser.add_argument("--rx-corr", type=float, default=0.6)
    parser.add_argument("--tx-corr", type=float, default=0.7)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    try:
        import hdf5storage
    except ImportError as exc:  # pragma: no cover - friendly message
        raise SystemExit(
            "hdf5storage is required to write .mat files. Install it with "
            "`pip install hdf5storage`."
        ) from exc

    os.makedirs(args.output_dir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    for offset, cdl in enumerate(args.cdl_types):
        h_train = _make_channels(args.num_rx, args.num_tx, args.n_train, rng,
                                 args.rx_corr, args.tx_corr)
        h_val = _make_channels(args.num_rx, args.num_tx, args.n_val, rng,
                               args.rx_corr, args.tx_corr)
        h_test = _make_channels(args.num_rx, args.num_tx, args.n_test, rng,
                                args.rx_corr, args.tx_corr)
        path = os.path.join(args.output_dir, f"channel_CDL-{cdl}.mat")
        hdf5storage.savemat(
            path,
            {"H_train": h_train, "H_val": h_val, "H_test": h_test},
            format="7.3",
            matlab_compatible=True,
        )
        print(
            f"wrote {path}: N_r={args.num_rx}, N_t={args.num_tx}, "
            f"train={h_train.shape[-1]}, val={h_val.shape[-1]}, test={h_test.shape[-1]}"
        )


if __name__ == "__main__":
    main()
