# Data preparation

This repository does **not** ship the CDL channel datasets used in the paper
(they are large and were generated externally). This document defines the exact
data contract expected by `src/srf/data.py`, and how to create a tiny synthetic
dataset for smoke tests.

## 1. File contract (`.mat`)

For each CDL profile place one MATLAB v7.3 (HDF5) file in one directory:

```
data/channel_data/
├── channel_CDL-A.mat
├── channel_CDL-B.mat
├── channel_CDL-C.mat
└── channel_CDL-D.mat
```

Each file must contain three complex variables:

| Variable  | Shape               | Meaning                                  |
|-----------|---------------------|------------------------------------------|
| `H_train` | `(N_r, N_t, N_tr)`  | clean training channel realizations      |
| `H_val`   | `(N_r, N_t, N_va)`  | clean validation realizations            |
| `H_test`  | `(N_r, N_t, N_te)`  | clean test realizations (100 in the paper)|

Notes:

- `N_r` and `N_t` are inferred from the configuration (`data.num_rx`,
  `data.num_tx`; the paper main setting is `N_r = 16`, `N_t = 64`).
- The loader also accepts files whose arrays are ordered `(N, N_r, N_t)`; it
  transposes them automatically.
- Files are read with `hdf5storage.loadmat`, so plain MATLAB v7.3 files work.
  `scipy.io.loadmat` files (v5/v7) are **not** read by
  `hdf5storage.loadmat` unless saved through `hdf5storage.savemat`.

## 2. Pilots, observation model and noise

The loader generates, once per split and seed, i.i.d. QPSK pilots
`P ∈ C^{N_t × N_p}` (`P[i, k] = (±1 ± j)/√2`) and forms the noisy pilot
observation

```
Y = H P + N,     vec(N) ~ CN(0, σ_p² I),     σ_p² = N_t / SNR_linear.
```

Each channel realization is normalized by the standard deviation of the split
(`H / std(H)`), exactly as in the legacy implementation. During training the
SNR is re-drawn per batch from `{snr_min, snr_min+step, …, snr_max}` dB; during
evaluation the SNR is set explicitly (see `configs/eval_*.yaml`).

`N_p` follows the pilot-density definition `α = N_p / N_t`: the paper main
setting uses `α = 0.6` for `N_t = 64`, i.e. `N_p = 38`.

## 3. Paper dataset and the included generator

The accepted paper evaluates on 3GPP CDL-A/B/C/D channels with
`N_r = 16`, `N_t = 64`, `α = 0.6`, `SNR ∈ {-10, -5, …, 30} dB`, and 100 test
realizations per profile. Those exact realizations are not committed to this
repository (they are large and were generated externally).

Two ways to obtain compatible data:

1. **Included MATLAB generator.** [`data_generation/`](../data_generation/) is
   the self-contained CKM-SRF unified dataset generator used for prototyping
   (MATLAB, no 5G Toolbox). With `cfg.cfo.enable = false` it produces the same
   limited-pilot observation model `Y = H P + N` as SRF, together with
   CKM/BIM context features and path-level ground truth. Run
   `data_generation/run_demo.m` (or `configs/research_config.m` for formal
   runs), then convert the shards to this repository's `.mat` contract:

   ```bash
   python data_generation/prepare_srf_mat.py \
       --input-dir data_generation/output_demo \
       --output-dir data/channel_data
   ```

   The generator produces geometrically defined scenes rather than 3GPP CDL
   profile realisations, so results on generated data are not bit-exact
   reproductions of the paper table — they are suitable for training and for
   physics-consistency prototyping. See
   [`data_generation/README.md`](../data_generation/README.md) for the schema
   and the HDF5 export path.
2. **Your own 3GPP CDL generator.** Any CDL-compliant generator (e.g.
   QuaDRiGa, MATLAB 5G Toolbox, Sionna RT for geometry-based channels) can be
   used as long as it follows the contract above. Minor differences between
   generators lead to small NMSE deviations from the paper table.

Do **not** commit large `.mat` files to git — the `.gitignore` excludes them;
share generated datasets via a data repository instead.

## 4. Synthetic smoke-test data (no external data needed)

`scripts/generate_synthetic_data.py` writes CDL-`A`…`D` files with a simple
spatially-correlated Gaussian channel model. It is only meant to check that
installation/training/evaluation run end-to-end, not to reproduce paper
numbers.

```bash
python scripts/generate_synthetic_data.py \
    --output-dir data/synthetic \
    --num-rx 8 --num-tx 16 \
    --n-train 32 --n-val 8 --n-test 16
```

Then use `configs/smoke_test.yaml` for a CPU-friendly run.
