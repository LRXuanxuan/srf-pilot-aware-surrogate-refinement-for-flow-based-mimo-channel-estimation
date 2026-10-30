#!/usr/bin/env python
"""Evaluate SRF / SRF-P / baseline samplers on CDL channel data.

Prints per-CDL and averaged NMSE (dB) for every requested sampler and SNR,
and optionally writes JSON/CSV results. With a checkpoint trained by
``scripts/train.py`` on the paper data, ``--samplers srf_p srf`` reproduces the
SRF-P / SRF rows of Table I.

Example:
    python scripts/evaluate.py \
        --config configs/eval_cdl_16x64_alpha06.yaml \
        --checkpoint outputs/srf_cdl_16x64_alpha06/best_model.pt \
        --samplers srf_p srf \
        --output-json results/table1_srf.json
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srf import (  # noqa: E402
    ChannelConfig,
    GMUNet,
    MatChannelDataset,
    SamplerConfig,
    flower_sample_channels,
    sample_channels,
)
from srf.training import extract_state_dict, strip_state_dict_prefix  # noqa: E402

KNOWN_SAMPLERS = ("srf", "srf_p", "pbg_only", "rcflow", "flower")


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    out = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = deep_update(out[key], value)
        else:
            out[key] = value
    return out


def load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_model(ckpt: Dict[str, Any], device: torch.device) -> torch.nn.Module:
    meta = ckpt.get("model") if isinstance(ckpt, dict) else None
    if meta is None:
        raise SystemExit(
            "Checkpoint does not carry model metadata. Use a checkpoint written by "
            "scripts/train.py (it stores the GMUNet constructor arguments)."
        )
    kwargs = dict(meta)
    kwargs["channel_mult"] = tuple(kwargs["channel_mult"])
    model = GMUNet(**kwargs).to(device)
    state_dict = strip_state_dict_prefix(extract_state_dict(ckpt))
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return model


def make_sampler_cfg(sampler: str, sampler_cfg: Dict[str, Any]) -> SamplerConfig:
    return SamplerConfig(
        mode=sampler,
        gm_output_mode=sampler_cfg.get("gm_output_mode", "sample"),
        lambda_exp=float(sampler_cfg.get("lambda_exp", 2.0)),
        beta_exp=float(sampler_cfg.get("beta_exp", 2.0)),
        pbg_iters=int(sampler_cfg.get("pbg_iters", 1)),
        pbg_step=float(sampler_cfg.get("pbg_step", 0.10)),
        pbg_w=float(sampler_cfg.get("pbg_w", 1.0)),
        pbg_rho=float(sampler_cfg.get("pbg_rho", 0.08)),
        pbg_noise_coef=float(sampler_cfg.get("pbg_noise_coef", 1.0)),
        pbg_eps=float(sampler_cfg.get("pbg_eps", 1e-6)),
        pbg_step_max=float(sampler_cfg.get("pbg_step_max", 0.5)),
    )


def estimate(
    sampler: str,
    model: torch.nn.Module,
    batch: Dict[str, torch.Tensor],
    sampler_cfg: SamplerConfig,
    device: torch.device,
    seed: int,
    flower_cfg: Dict[str, Any] | None = None,
) -> torch.Tensor:
    y = batch["Y"].to(device)
    p = batch["P"].to(device)
    noise_scale = batch["noise_scale"].to(device)
    if sampler == "flower":
        flower_cfg = flower_cfg or {}
        h_est, _ = flower_sample_channels(
            model, y, p, noise_scale,
            mode=sampler_cfg.gm_output_mode,
            total_steps=flower_cfg.get("total_steps"),
            mmse_num_samples=int(flower_cfg.get("mmse_num_samples", 8)),
            match_rcflow_budget=bool(flower_cfg.get("match_rcflow_budget", True)),
            seed=seed,
        )
    else:
        h_est, _ = sample_channels(model, y, p, noise_scale, sampler_cfg)
    return h_est


def nmse_ratios(h_est: torch.Tensor, h_true: torch.Tensor) -> List[float]:
    """Per-sample linear NMSE ``||H_hat - H||^2 / ||H||^2`` (paper metric)."""
    num = torch.sum(torch.abs(h_est - h_true) ** 2, dim=(-1, -2))
    den = torch.sum(torch.abs(h_true) ** 2, dim=(-1, -2))
    return (num / den).cpu().numpy().tolist()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/eval_cdl_16x64_alpha06.yaml")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--samplers", nargs="+", default=["srf_p", "srf"])
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-json", default=None)
    parser.add_argument("--output-csv", default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--max-batches", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    for sampler in args.samplers:
        if sampler not in KNOWN_SAMPLERS:
            raise SystemExit(f"Unknown sampler {sampler!r}; choose from {KNOWN_SAMPLERS}")

    cfg = load_config(args.config)
    overrides: Dict[str, Any] = {}
    if args.data_dir:
        overrides.setdefault("data", {})["data_dir"] = args.data_dir
    if args.batch_size is not None:
        overrides.setdefault("test", {})["batch_size"] = args.batch_size
    if args.max_batches is not None:
        overrides.setdefault("test", {})["max_batches"] = args.max_batches
    if args.seed is not None:
        overrides.setdefault("test", {})["seed"] = args.seed
    if args.num_workers is not None:
        overrides.setdefault("data", {})["num_workers"] = args.num_workers
    cfg = deep_update(cfg, overrides)

    data_cfg, test_cfg, sampler_cfg = cfg["data"], cfg["test"], cfg["sampler"]
    flower_cfg = sampler_cfg.get("flower", {}) if isinstance(sampler_cfg.get("flower"), dict) else {}
    seed = int(test_cfg.get("seed", 42))
    set_seed(seed)
    device = resolve_device(args.device)

    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(ckpt, device)
    model_meta = ckpt.get("model", {})
    print(f"loaded checkpoint: {args.checkpoint}")
    print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")

    num_rx = int(data_cfg["num_rx"])
    num_tx = int(data_cfg["num_tx"])
    num_pilots = int(data_cfg["num_pilots"])
    snr_list = [int(s) for s in test_cfg["snr_db"]]
    cdl_types = [str(c).upper() for c in test_cfg["cdl_types"]]
    batch_size = int(test_cfg["batch_size"])
    max_batches = test_cfg.get("max_batches")
    num_workers = int(data_cfg.get("num_workers", 0))

    results: Dict[str, Dict[str, Dict[str, float]]] = {}
    for sampler in args.samplers:
        sampler_cfg_obj = make_sampler_cfg(sampler, sampler_cfg)
        results[sampler] = {}
        for cdl in cdl_types:
            path = os.path.join(data_cfg["data_dir"], f"channel_CDL-{cdl}.mat")
            if not os.path.exists(path):
                raise FileNotFoundError(f"missing data file {path} (see docs/DATA.md)")
            dataset = MatChannelDataset(
                [path],
                split="test",
                config=ChannelConfig(
                    image_size=(num_rx, num_tx),
                    num_pilots=num_pilots,
                    snr_db=float(snr_list[0]),
                ),
                seed=seed,
            )
            results[sampler][cdl] = {}
            for snr in snr_list:
                dataset.snr_db = snr
                snr_linear = 10 ** (snr / 10.0)
                dataset.noise_power = num_tx / snr_linear
                dataset.noise_sigma = np.sqrt(dataset.noise_power)
                loader = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                )
                estimates: List[float] = []
                for index, batch in enumerate(
                    tqdm(loader, desc=f"{sampler} CDL-{cdl} SNR {snr}dB", leave=False)
                ):
                    if max_batches is not None and index >= int(max_batches):
                        break
                    h_est = estimate(
                        sampler, model, batch, sampler_cfg_obj, device, seed,
                        flower_cfg=flower_cfg,
                    )
                    h_true = batch["H"].to(device)
                    estimates.extend(nmse_ratios(h_est, h_true))
                results[sampler][cdl][str(snr)] = float(
                    10.0 * math.log10(np.mean(estimates))
                )

    # Averages over CDL profiles.
    for sampler in results:
        avg = {}
        for snr in snr_list:
            values = [results[sampler][cdl][str(snr)] for cdl in cdl_types]
            avg[str(snr)] = float(np.mean(values))
        results[sampler]["average"] = avg

    # Print tables.
    for sampler, per_cdl in results.items():
        print(f"\n=== {sampler} (NMSE dB) ===")
        header = "CDL".ljust(8) + "".join(f"{snr:>8}" for snr in snr_list)
        print(header)
        for cdl in cdl_types + ["average"]:
            row = str(cdl).ljust(8) + "".join(f"{per_cdl[cdl][str(snr)]:>8.2f}" for snr in snr_list)
            print(row)

    payload = {
        "checkpoint": os.path.abspath(args.checkpoint),
        "model": model_meta,
        "config": str(args.config),
        "samplers": args.samplers,
        "cdl_types": cdl_types,
        "snr_db": snr_list,
        "results": results,
    }
    if args.output_json:
        Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_json, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        print(f"\nsaved JSON: {args.output_json}")

    if args.output_csv:
        Path(args.output_csv).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_csv, "w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["sampler", "cdl_type", "snr_db", "nmse_db"])
            for sampler, per_cdl in results.items():
                for cdl in cdl_types + ["average"]:
                    for snr in snr_list:
                        writer.writerow([sampler, cdl, snr, per_cdl[cdl][str(snr)]])
        print(f"saved CSV: {args.output_csv}")


if __name__ == "__main__":
    main()
