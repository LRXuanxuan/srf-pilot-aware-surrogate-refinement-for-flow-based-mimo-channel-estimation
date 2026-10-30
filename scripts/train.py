#!/usr/bin/env python
"""Train the SRF flow prior on CDL MIMO channel data.

The default objective is the DPR finite-transition Gaussian-mixture NLL
(``objective: gm_nll``, paper eq. 24 with ``r = 0.5``); ``flow_matching``
trains the classic point-velocity baseline instead.

Example:
    python scripts/train.py --config configs/srf_cdl_16x64_alpha06.yaml
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import random
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

# Make ``src/`` importable when running from an uninstalled checkout.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from srf.data import ChannelConfig, MatChannelDataset  # noqa: E402
from srf.models import GMUNet  # noqa: E402
from srf.training import FlowMatchingObjective, GMFlowObjective  # noqa: E402


def deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively merge ``override`` into ``base`` (returns a new dict)."""
    out = copy.deepcopy(base)
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def build_file_list(data_dir: str, dataset_type: str) -> list[str]:
    if dataset_type.lower() == "mixed":
        types = ["A", "B", "C", "D"]
    else:
        types = [dataset_type.upper()]
    files = [os.path.join(data_dir, f"channel_CDL-{t}.mat") for t in types]
    missing = [f for f in files if not os.path.exists(f)]
    if missing:
        raise FileNotFoundError(
            "Missing data files: " + ", ".join(missing) + ". See docs/DATA.md."
        )
    return files


def build_objective(cfg: Dict[str, Any]):
    train_cfg = cfg["training"]
    kwargs = dict(
        snr_min_db=train_cfg["snr_min_db"],
        snr_max_db=train_cfg["snr_max_db"],
        snr_step_db=train_cfg["snr_step_db"],
        time_mu=train_cfg["time_mu"],
        time_sigma=train_cfg["time_sigma"],
    )
    objective = train_cfg["objective"].lower()
    if objective in ("gm_nll", "gm-nll", "dpr", "srf"):
        return GMFlowObjective(transition_ratio=train_cfg["transition_ratio"], **kwargs)
    if objective in ("flow_matching", "flow-matching", "rcflow"):
        return FlowMatchingObjective(**kwargs)
    raise ValueError(f"Unknown training.objective: {train_cfg['objective']!r}")


def validation_metric(model: torch.nn.Module, loader: DataLoader, objective, device) -> float:
    model.eval()
    total, count = 0.0, 0
    with torch.no_grad():
        for batch in loader:
            out = objective(model, batch)
            key = "unweighted_mse" if "unweighted_mse" in out else "mse" if "mse" in out else "loss"
            total += float(out[key])
            count += 1
    model.train()
    return total / max(count, 1)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/srf_cdl_16x64_alpha06.yaml")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", default=None, choices=["auto", "cpu", "cuda"])
    parser.add_argument("--num-workers", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    overrides: Dict[str, Any] = {}
    if args.data_dir:
        overrides.setdefault("data", {})["data_dir"] = args.data_dir
    if args.output_dir:
        overrides.setdefault("training", {})["output_dir"] = args.output_dir
    if args.epochs is not None:
        overrides.setdefault("training", {})["epochs"] = args.epochs
    if args.batch_size is not None:
        overrides.setdefault("training", {})["batch_size"] = args.batch_size
    if args.lr is not None:
        overrides.setdefault("training", {})["lr"] = args.lr
    if args.seed is not None:
        overrides.setdefault("training", {})["seed"] = args.seed
    if args.device is not None:
        overrides.setdefault("training", {})["device"] = args.device
    if args.num_workers is not None:
        overrides.setdefault("data", {})["num_workers"] = args.num_workers
    cfg = deep_update(cfg, overrides)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    train_cfg = cfg["training"]
    seed = int(train_cfg["seed"])
    set_seed(seed)

    device = resolve_device(train_cfg.get("device", "auto"))
    output_dir = Path(train_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    file_list = build_file_list(data_cfg["data_dir"], data_cfg["dataset_type"])
    dataset_cfg = ChannelConfig(
        image_size=(int(data_cfg["num_rx"]), int(data_cfg["num_tx"])),
        num_pilots=int(data_cfg["num_pilots"]),
        snr_db=float(data_cfg["snr_db"]),
    )
    train_dataset = MatChannelDataset(file_list, split="train", config=dataset_cfg, seed=seed)
    val_dataset = MatChannelDataset(file_list, split="val", config=dataset_cfg, seed=seed)
    print(f"train samples: {len(train_dataset)} | val samples: {len(val_dataset)}")

    num_workers = int(data_cfg.get("num_workers", 0))
    train_loader = DataLoader(
        train_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=num_workers,
        drop_last=True,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    gm_kwargs = {
        "model_channels": int(model_cfg["model_channels"]),
        "channel_mult": tuple(model_cfg["channel_mult"]),
        "num_res_blocks": int(model_cfg["num_res_blocks"]),
        "gm_num_components": int(model_cfg["gm_num_components"]),
        "gm_min_logstd": float(model_cfg["gm_min_logstd"]),
        "gm_max_logstd": float(model_cfg["gm_max_logstd"]),
        "gm_logstd_layers": int(model_cfg.get("gm_logstd_layers", 2)),
    }
    model = GMUNet(**gm_kwargs).to(device)
    print(f"model parameters: {sum(p.numel() for p in model.parameters()):,}")

    objective = build_objective(cfg)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    amp_mode = str(train_cfg.get("amp", "none")).lower()
    use_amp = amp_mode in ("bf16", "fp16") and device.type == "cuda"
    amp_dtype = torch.bfloat16 if amp_mode == "bf16" else torch.float16
    scaler = torch.amp.GradScaler(device.type, enabled=(use_amp and amp_dtype == torch.float16))

    grad_clip = float(train_cfg.get("grad_clip", 0.0))
    val_every = max(int(train_cfg.get("val_every", 1)), 1)
    epochs = int(train_cfg["epochs"])
    best_metric = float("inf")

    metadata = {
        "model": {**gm_kwargs, "channel_mult": list(gm_kwargs["channel_mult"])},
        "data": {
            "num_rx": dataset_cfg.image_size[0],
            "num_tx": dataset_cfg.image_size[1],
            "num_pilots": dataset_cfg.num_pilots,
            "dataset_type": data_cfg["dataset_type"],
        },
        "training": {
            "objective": train_cfg["objective"],
            "transition_ratio": train_cfg.get("transition_ratio"),
            "snr_min_db": train_cfg["snr_min_db"],
            "snr_max_db": train_cfg["snr_max_db"],
            "snr_step_db": train_cfg["snr_step_db"],
            "time_mu": train_cfg["time_mu"],
            "time_sigma": train_cfg["time_sigma"],
            "batch_size": int(train_cfg["batch_size"]),
            "lr": float(train_cfg["lr"]),
            "seed": seed,
        },
        "format_version": 1,
    }
    with open(output_dir / "config.json", "w", encoding="utf-8") as handle:
        json.dump(cfg, handle, indent=2, ensure_ascii=False)

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, running_metric, steps = 0.0, 0.0, 0
        progress = tqdm(train_loader, desc=f"epoch {epoch}/{epochs}", leave=False)
        for batch in progress:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type=device.type, dtype=amp_dtype, enabled=use_amp):
                out = objective(model, batch)
                loss = out["loss"]
            if scaler.is_enabled():
                scaler.scale(loss).backward()
                if grad_clip > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                optimizer.step()

            metric_value = float(out.get("unweighted_mse", out.get("mse", loss.detach())))
            running_loss += float(loss.detach())
            running_metric += metric_value
            steps += 1
            progress.set_postfix(loss=f"{float(loss.detach()):.4f}")

        train_loss = running_loss / max(steps, 1)
        if epoch % val_every == 0 or epoch == epochs:
            val_metric = validation_metric(model, val_loader, objective, device)
            print(
                f"epoch {epoch:4d} | train loss {train_loss:.4f} | "
                f"val metric {val_metric:.6f}"
            )
            if val_metric < best_metric:
                best_metric = val_metric
                torch.save(
                    {**metadata, "state_dict": model.state_dict(),
                     "epoch": epoch, "val_metric": val_metric},
                    output_dir / "best_model.pt",
                )
                print(f"  saved new best checkpoint (val metric {val_metric:.6f})")
        else:
            print(f"epoch {epoch:4d} | train loss {train_loss:.4f}")

        torch.save(
            {**metadata, "state_dict": model.state_dict(),
             "epoch": epoch, "val_metric": None},
            output_dir / "last_model.pt",
        )

    print(f"done. best val metric: {best_metric:.6f}")
    print(f"checkpoints: {output_dir / 'best_model.pt'}")


if __name__ == "__main__":
    main()
