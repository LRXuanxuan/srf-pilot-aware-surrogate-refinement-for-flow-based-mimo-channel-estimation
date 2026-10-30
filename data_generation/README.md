# CKM-SRF unified dataset generator

This directory contains the MATLAB dataset generator used to synthesise MIMO
channel data compatible with the SRF observation model `Y = H P + N`. It is
self-contained (no 5G Toolbox) and produces scene/shard files with a unified
channel layout

```
H(rx, tx, subcarrier, snapshot, sample),
```

plus pilot observations, CKM/BIM context features and path-level ground truth.
With `cfg.cfo.enable = false` the generated data follow the same limited-pilot
observation model as the SRF paper; with `cfg.cfo.enable = true` the generator
also supports a CFO/ICI generalised operator (this requires the companion
adjoint operator, see `docs/SRF_INTEGRATION.md`).

The original Chinese documentation is kept as [`README.zh.md`](README.zh.md).

## Layout

```
data_generation/
├── run_demo.m                     # small end-to-end demo
├── generate_ckm_srf_dataset.m     # top-level generator
├── ckmsrf_default_config.m        # defaults
├── configs/research_config.m      # suggested formal-experiment configuration
├── export_scene_to_hdf5.m         # MATLAB .mat -> explicit HDF5 export
├── export_shard_to_hdf5.m
├── src/*.m                        # geometry, CKM, pilot, CFO and I/O helpers
├── tests/run_all_tests.m          # physics and schema tests
├── docs/SCHEMA.md                 # field-by-field data schema (Chinese)
├── docs/SRF_INTEGRATION.md        # how to condition SRF on CKM features
├── python/                        # HDF5 loader + example
├── prepare_srf_mat.py             # shard -> release `.mat` contract converter
└── README.md / README.zh.md       # this file / original Chinese README
```

## Quick start (MATLAB)

```matlab
cd('<repo>/data_generation')
run_demo          % writes output_demo/ with 1 scene, 12/6/6 train/val/test samples
run('tests/run_all_tests.m')
```

Formal experiments use `configs/research_config.m`:

```matlab
addpath('configs')
cfg = research_config('D:/datasets/ckm_srf_64x16');
manifest = generate_ckm_srf_dataset(cfg);
```

Start with a small sample count (`cfg.dataset.samplesPerScene.train = 10`) to
check disk/memory compatibility, then scale up. Demo outputs (`output*/`) are
excluded from git.

## Converting generator output to the SRF training format

The SRF training/evaluation scripts in this repository expect one `.mat` file
per profile with `H_train` / `H_val` / `H_test` of shape `(N_r, N_t, N)` (see
`docs/DATA.md`). Convert the generated shards with:

```bash
python data_generation/prepare_srf_mat.py \
    --input-dir data_generation/output_demo \
    --output-dir data/channel_data
```

The converter picks one subcarrier (default: middle), flattens snapshots and
samples into independent realizations, and writes `channel_CDL-A.mat`,
`channel_CDL-B.mat`, … (scene ids are mapped to labels `A`, `B`, `C`, `D` in
ascending order; use `--scene-map` to override). The converter only needs
`hdf5storage` (`pip install hdf5storage`).

`--num-pilots` is deliberately not passed: the converter only transfers clean
channels, and the release pipeline regenerates QPSK pilots / noise according to
`configs/*.yaml` (`data.num_pilots`, per-batch SNR sampling).

**Important:** the generated scenes are geometrically defined scenes rather than
3GPP CDL-A/B/C/D profile realisations. They are suitable for training and for
physics-consistency prototyping, but small numerical differences from the paper
table are expected. The paper's CDL evaluation data were generated externally,
so this repository does not claim bit-exact reproduction of Table I from the
generator output alone.

## Python / HDF5 export

MATLAB `-v7.3` structs are awkward to read from Python; the generator provides
explicit HDF5 exports:

```matlab
export_scene_to_hdf5('output_demo/scene_01_ckm.mat', 'output_demo_h5/scene_01_ckm.h5')
export_shard_to_hdf5('output_demo/scene_01_train_shard_001.mat', 'output_demo_h5/scene_01_train_shard_001.h5')
```

Complex arrays are stored as `*_real` / `*_imag` dataset pairs. Use
`python/ckmsrf_h5_dataset.py` (it restores the MATLAB axis order) or install the
small reader requirements:

```bash
python -m pip install -r data_generation/python/requirements.txt
python data_generation/python/test_h5_loader.py \
    data_generation/output_demo_h5/scene_01_train_shard_001.h5 \
    data_generation/output_demo_h5/scene_01_ckm.h5
```

## Scope and data boundaries

- `truth` fields are supervision labels / analysis variables only; online model
  conditions must come from `state` and `context`.
- `context` is interpolated from an independent historical CKM and does not
  contain test-time instantaneous path gains.
- `aux` (BIM / beamforming labels) is auxiliary and must not be used as an
  oracle input.
- All snapshots of one trajectory stay in the same spatial split.
