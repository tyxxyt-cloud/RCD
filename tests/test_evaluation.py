from __future__ import annotations

from pathlib import Path

import pandas as pd
import torch

from tests.helpers import tiny_config, tiny_payload
from trdmu.evaluation.evaluate import evaluate_checkpoint
from trdmu.models import CRCDMModel


def test_checkpoint_evaluation_exports_metrics_and_predictions(tmp_path: Path) -> None:
    cfg = tiny_config(tmp_path)
    payload = tiny_payload()
    processed_dir = Path(cfg["paths"]["processed_dir"])
    processed_dir.mkdir(parents=True)
    torch.save(payload, processed_dir / "trdmu_dataset.pt")
    model = CRCDMModel(cfg, payload["meta"])
    checkpoint_path = tmp_path / "best.pt"
    torch.save(
        {
            "model_state": model.state_dict(),
            "meta": payload["meta"],
            "config": cfg,
            "best": {"epoch": 1, "threshold": 0.5, "congestion_threshold": 0.5},
        },
        checkpoint_path,
    )
    output_dir = tmp_path / "evaluation"
    metrics = evaluate_checkpoint(cfg, checkpoint_path, "test", output_dir)
    assert metrics["closure"]["rows"] == 2
    assert metrics["closure"]["pr_auc"] is not None
    assert (output_dir / "test_metrics.json").exists()
    predictions = pd.read_csv(output_dir / "test_predictions.csv.gz")
    assert len(predictions) == 2
    assert {"closure_probability", "closure_prediction"} <= set(predictions.columns)
