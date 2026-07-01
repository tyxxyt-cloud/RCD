from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from crcdm.config import (
    ensure_dir,
    latest_run_dir,
    load_config,
    resolve_device,
    write_json,
)
from crcdm.data import CRCDMDataset, collate_samples, load_processed_dataset
from crcdm.models import CRCDMModel
from crcdm.utils.metrics import binary_metrics


DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained CRCDM checkpoint.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--split", default=None, choices=["train", "val", "test"])
    parser.add_argument("--output-dir", default=None)
    return parser.parse_args()


@torch.no_grad()
def collect_predictions(
    model: CRCDMModel,
    loader: DataLoader,
    device: torch.device,
    lambda_grl: float,
) -> Dict[str, np.ndarray]:
    model.eval()
    collected: Dict[str, list[float]] = {
        "y_closure": [],
        "y_congestion": [],
        "closure_prob": [],
        "congestion_prob": [],
    }
    for batch in loader:
        output = model(batch["samples"], device=device, lambda_grl=lambda_grl)
        collected["y_closure"].extend(batch["y_closure"].numpy().tolist())
        collected["y_congestion"].extend(batch["y_congestion"].numpy().tolist())
        collected["closure_prob"].extend(output["closure_prob"].cpu().numpy().tolist())
        collected["congestion_prob"].extend(
            output["congestion_prob"].cpu().numpy().tolist()
        )
    return {
        "y_closure": np.asarray(collected["y_closure"], dtype=np.int8),
        "y_congestion": np.asarray(collected["y_congestion"], dtype=np.int8),
        "closure_prob": np.asarray(collected["closure_prob"], dtype=float),
        "congestion_prob": np.asarray(collected["congestion_prob"], dtype=float),
    }


def evaluate_checkpoint(
    cfg: Dict[str, Any],
    checkpoint_path: Path,
    split: str,
    output_dir: Path,
) -> Dict[str, Any]:
    train_cfg = cfg["training"]
    device = resolve_device(str(train_cfg["device"]))
    payload = load_processed_dataset(cfg)
    dataset = CRCDMDataset(payload, split)
    loader = DataLoader(
        dataset,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_samples,
    )
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)
    checkpoint_cfg = checkpoint.get("config", cfg)
    model = CRCDMModel(checkpoint_cfg, checkpoint["meta"]).to(device)
    model.load_state_dict(checkpoint["model_state"])
    predictions = collect_predictions(
        model,
        loader,
        device,
        float(checkpoint_cfg["training"]["lambda_grl"]),
    )
    best = checkpoint.get("best", {})
    closure_threshold = float(best.get("threshold", 0.5))
    congestion_threshold = float(best.get("congestion_threshold", 0.5))
    metrics = {
        "method": "CRCDM",
        "split": split,
        "checkpoint": str(checkpoint_path),
        "best_epoch": best.get("epoch"),
        "closure": binary_metrics(
            predictions["y_closure"], predictions["closure_prob"], closure_threshold
        ),
        "congestion_auxiliary": binary_metrics(
            predictions["y_congestion"],
            predictions["congestion_prob"],
            congestion_threshold,
        ),
    }
    ensure_dir(output_dir)
    write_json(output_dir / f"{split}_metrics.json", metrics)
    rows = []
    for sample, closure_prob, congestion_prob in zip(
        dataset.samples,
        predictions["closure_prob"],
        predictions["congestion_prob"],
    ):
        rows.append(
            {
                "sample_id": sample.get("sample_id"),
                "road_id": sample.get("road_id"),
                "label_hour": sample.get("label_hour"),
                "y_closure": sample.get("y_closure"),
                "closure_probability": float(closure_prob),
                "closure_prediction": int(closure_prob >= closure_threshold),
                "y_congestion": sample.get("y_congestion"),
                "congestion_probability": float(congestion_prob),
                "congestion_prediction": int(congestion_prob >= congestion_threshold),
            }
        )
    pd.DataFrame(rows).to_csv(
        output_dir / f"{split}_predictions.csv.gz",
        index=False,
        compression="gzip",
    )
    return metrics


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    run_dir = latest_run_dir(cfg)
    checkpoint_path = (
        Path(args.checkpoint)
        if args.checkpoint
        else run_dir / "checkpoints" / "best.pt"
    )
    split = args.split or str(cfg.get("evaluation", {}).get("split", "test"))
    output_dir = Path(args.output_dir) if args.output_dir else run_dir / "evaluation"
    metrics = evaluate_checkpoint(cfg, checkpoint_path, split, output_dir)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
