#!/usr/bin/env python
"""Convert CKM-SRF generator shards into the SRF `.mat` dataset contract.

The MATLAB generator in this directory (``generate_ckm_srf_dataset.m``) writes
scene/shard files whose channels have shape ``(N_r, N_t, N_sc, T, N)`` and store
extra CKM / path / observation fields. The SRF training and evaluation scripts
of this repository expect one ``.mat`` file per CDL profile containing
``H_train``, ``H_val`` and ``H_test`` of shape ``(N_r, N_t, N_samples)``.

This script bridges the two formats:

* it reads every ``scene_<id>_<split>_shard_<k>.mat`` file produced by the
  generator (``shard.H_true``),
* keeps one subcarrier (``--subcarrier``) and the requested snapshots,
* flattens ``(T, N)`` into independent channel realizations, and
* writes ``channel_CDL-<L>.mat`` files with ``H_train`` / ``H_val`` / ``H_test``.

Example (after running ``run_demo.m`` in MATLAB):

    python data_generation/prepare_srf_mat.py \
        --input-dir CKM_SRF_unified_generator/output_demo \
        --output-dir data/channel_data \
        --num-pilots 38

Note: the generator produces geometrically generated scenes rather than the
3GPP CDL-A/B/C/D profiles. Scene ids are mapped to the labels ``A``, ``B``,
``C``, ``D`` in ascending order (override with ``--scene-map 3:A,7:B``); the
resulting files therefore provide data *in the release format*, not an exact
copy of the paper dataset. Use ``cfg.cfo.enable = false`` in MATLAB to keep the
SRF observation model ``Y = H P + N``.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, List

import numpy as np

SHARD_RE = re.compile(r"scene_(\d+)_(train|val|test)_shard_\d+\.mat$", re.IGNORECASE)
SPLITS = ("train", "val", "test")


def _load_h_true(path: Path) -> np.ndarray:
    """Return the ``(N_r, N_t, N_sc, T, N)`` channel array of one shard."""
    import hdf5storage

    data = hdf5storage.loadmat(str(path))
    if "shard" not in data:
        raise ValueError(f"{path} does not contain a 'shard' variable")
    shard = data["shard"]
    if not hasattr(shard, "dtype") or shard.dtype.names is None:
        raise ValueError(f"{path}: unexpected shard layout")
    if "H_true" not in shard.dtype.names:
        raise ValueError(f"{path}: shard has no H_true field")
    h = np.asarray(shard["H_true"])
    while h.ndim > 5:  # drop leading MATLAB struct dimensions
        h = h[0]
    if h.ndim != 5:
        raise ValueError(f"{path}: H_true is {h.shape}, expected (Nr, Nt, Nsc, T, N)")
    return h.astype(np.complex64)


def _parse_scene_map(text: str | None) -> Dict[int, str]:
    mapping: Dict[int, str] = {}
    if not text:
        return mapping
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        scene, label = item.split(":")
        mapping[int(scene)] = label.strip().upper()
    return mapping


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", required=True,
                        help="generator output directory containing scene_*_shard_*.mat")
    parser.add_argument("--output-dir", default="data/channel_data")
    parser.add_argument("--subcarrier", type=int, default=None,
                        help="subcarrier index to extract (default: middle subcarrier)")
    parser.add_argument("--snapshots", default=None,
                        help="comma-separated snapshot indices (default: all)")
    parser.add_argument("--scene-map", default=None,
                        help="explicit scene->label map, e.g. '3:A,7:B'")
    parser.add_argument("--max-scenes", type=int, default=4,
                        help="maximum number of scenes to convert (labels A..D)")
    args = parser.parse_args()

    import hdf5storage

    input_dir = Path(args.input_dir)
    if not input_dir.is_dir():
        raise SystemExit(f"input directory not found: {input_dir}")

    shards: Dict[int, Dict[str, List[Path]]] = {}
    for path in sorted(input_dir.glob("scene_*_shard_*.mat")):
        match = SHARD_RE.search(path.name)
        if not match:
            continue
        scene, split = int(match.group(1)), match.group(2).lower()
        shards.setdefault(scene, {s: [] for s in SPLITS})[split].append(path)

    if not shards:
        raise SystemExit(
            f"no scene_*_shard_*.mat files found in {input_dir}; run run_demo.m first"
        )

    scene_map = _parse_scene_map(args.scene_map)
    scene_ids = sorted(scene_map) if scene_map else sorted(shards)[: args.max_scenes]
    if len(scene_ids) > 4:
        raise SystemExit("at most 4 scenes can be mapped to labels A/B/C/D")
    labels = "ABCD"[: len(scene_ids)]

    snapshot_idx = None
    if args.snapshots:
        snapshot_idx = [int(s) for s in args.snapshots.split(",")]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    for label, scene in zip(labels, scene_ids):
        split_channels: Dict[str, List[np.ndarray]] = {s: [] for s in SPLITS}
        for split in SPLITS:
            for path in shards[scene][split]:
                h = _load_h_true(path)
                num_sc = h.shape[2]
                sc = args.subcarrier if args.subcarrier is not None else num_sc // 2
                if not 0 <= sc < num_sc:
                    raise SystemExit(f"subcarrier {sc} out of range for {path}")
                h = h[:, :, sc, :, :]  # (Nr, Nt, T, N)
                if snapshot_idx is not None:
                    h = h[:, :, snapshot_idx, :]
                h = h.reshape(h.shape[0], h.shape[1], -1)  # (Nr, Nt, T*N)
                split_channels[split].append(h)
        if not split_channels["train"]:
            raise SystemExit(f"scene {scene} has no training shards")

        arrays = {
            f"H_{split}": np.concatenate(split_channels[split], axis=-1)
            if split_channels[split]
            else np.zeros((1, 1, 0), dtype=np.complex64)
            for split in SPLITS
        }
        out_path = output_dir / f"channel_CDL-{label}.mat"
        hdf5storage.savemat(
            str(out_path),
            arrays,
            format="7.3",
            matlab_compatible=True,
        )
        print(
            f"scene {scene} -> {out_path}: "
            f"train={arrays['H_train'].shape}, "
            f"val={arrays['H_val'].shape}, test={arrays['H_test'].shape}"
        )

    print(f"\nwrote {len(labels)} file(s) to {output_dir}")
    print("next: python scripts/train.py --config configs/srf_cdl_16x64_alpha06.yaml "
          f"--data-dir {output_dir}")


if __name__ == "__main__":
    main()
