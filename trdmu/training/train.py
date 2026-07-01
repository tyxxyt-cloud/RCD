from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Dict

import numpy as np
import torch
from torch.utils.data import DataLoader

from trdmu.config import load_config, make_run_dir, resolve_device, set_seed, write_json
from trdmu.data import TRDMUDataset, collate_samples, load_processed_dataset, pos_weight
from trdmu.models import CRCDMModel, compute_loss
from trdmu.utils.metrics import best_f1_threshold, binary_metrics

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train CRCDM.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


@torch.no_grad()
def predict_split(
    model: CRCDMModel,
    loader: DataLoader,
    device: torch.device,
    lambda_grl: float,
) -> Dict[str, np.ndarray]:
    model.eval()
    y_closure = []
    y_congestion = []
    closure_prob = []
    congestion_prob = []
    for batch in loader:
        out = model(batch["samples"], device=device, lambda_grl=lambda_grl)
        y_closure.extend(batch["y_closure"].numpy().tolist())
        y_congestion.extend(batch["y_congestion"].numpy().tolist())
        closure_prob.extend(out["closure_prob"].detach().cpu().numpy().tolist())
        congestion_prob.extend(out["congestion_prob"].detach().cpu().numpy().tolist())
    return {
        "y_closure": np.asarray(y_closure, dtype=np.int8),
        "y_congestion": np.asarray(y_congestion, dtype=np.int8),
        "closure_prob": np.asarray(closure_prob, dtype=float),
        "congestion_prob": np.asarray(congestion_prob, dtype=float),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    train_cfg = cfg["training"]
    set_seed(int(train_cfg["seed"]))
    device = resolve_device(str(train_cfg["device"]))

    payload = load_processed_dataset(cfg)
    train_ds = TRDMUDataset(payload, "train")
    val_ds = TRDMUDataset(payload, "val")

    train_loader = DataLoader(
        train_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=True,
        num_workers=int(train_cfg.get("num_workers", 0)),
        collate_fn=collate_samples,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=int(train_cfg["batch_size"]),
        shuffle=False,
        num_workers=0,
        collate_fn=collate_samples,
    )

    run_dir = make_run_dir(cfg)
    (run_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    write_json(run_dir / "config.json", cfg)
    shutil.copy2(args.config, run_dir / "default.yaml")

    model = CRCDMModel(cfg, payload["meta"]).to(device)
    optimizer = torch.optim.Adam(
        model.main_parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    mi_optimizer = torch.optim.Adam(
        model.mi_estimator.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    closure_pos_weight = torch.tensor(
        pos_weight(payload["splits"]["train"], "y_closure"), dtype=torch.float32
    )
    congestion_pos_weight = torch.tensor(
        pos_weight(payload["splits"]["train"], "y_congestion"), dtype=torch.float32
    )

    best = {"epoch": -1, "val_f1": -1.0, "threshold": 0.5, "congestion_threshold": 0.5}
    patience = 0
    history = []
    for epoch in range(1, int(train_cfg["epochs"]) + 1):
        model.train()
        totals: Dict[str, float] = {}
        rows = 0
        for step, batch in enumerate(train_loader, start=1):
            mi_optimizer.zero_grad(set_to_none=True)
            with torch.no_grad():
                representations = model.representations(batch["samples"], device)
            loss_mi_estimator = model.mi_estimator_loss(
                representations["z"],
                representations["c"],
            )
            loss_mi_estimator.backward()
            mi_optimizer.step()

            for parameter in model.mi_estimator.parameters():
                parameter.requires_grad_(False)
            optimizer.zero_grad(set_to_none=True)
            out = model(
                batch["samples"],
                device=device,
                lambda_grl=float(train_cfg["lambda_grl"]),
            )
            losses = compute_loss(
                out,
                batch["y_closure"],
                batch["y_congestion"],
                closure_pos_weight,
                congestion_pos_weight,
                lambda_con=float(train_cfg["lambda_con"]),
                lambda_mi=float(train_cfg["lambda_mi"]),
                lambda_dis=float(train_cfg.get("lambda_dis", 1.0)),
            )
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), float(train_cfg["grad_clip_norm"])
            )
            optimizer.step()
            for parameter in model.mi_estimator.parameters():
                parameter.requires_grad_(True)
            losses["loss_mi_estimator"] = loss_mi_estimator.detach()
            batch_size = len(batch["samples"])
            rows += batch_size
            for key, value in losses.items():
                totals[key] = (
                    totals.get(key, 0.0) + float(value.detach().cpu()) * batch_size
                )
            if step % int(train_cfg["log_every"]) == 0:
                avg = totals["loss"] / max(rows, 1)
                print(
                    f"epoch={epoch:03d} step={step:04d} train_loss={avg:.6f}",
                    flush=True,
                )

        val_pred = predict_split(
            model, val_loader, device, float(train_cfg["lambda_grl"])
        )
        threshold, threshold_metrics = best_f1_threshold(
            val_pred["y_closure"], val_pred["closure_prob"]
        )
        congestion_threshold, congestion_threshold_metrics = best_f1_threshold(
            val_pred["y_congestion"], val_pred["congestion_prob"]
        )
        val_metrics = binary_metrics(
            val_pred["y_closure"], val_pred["closure_prob"], threshold
        )
        val_con_metrics = binary_metrics(
            val_pred["y_congestion"],
            val_pred["congestion_prob"],
            congestion_threshold,
        )
        epoch_record = {
            "epoch": epoch,
            "train": {key: value / max(rows, 1) for key, value in totals.items()},
            "validation": val_metrics,
            "validation_congestion": val_con_metrics,
            "threshold_search": threshold_metrics,
            "congestion_threshold_search": congestion_threshold_metrics,
        }
        history.append(epoch_record)
        write_json(run_dir / "train_history.json", history)
        print(
            "epoch=%03d val_f1=%.6f val_p=%.6f val_r=%.6f threshold=%.6f"
            % (
                epoch,
                float(val_metrics["f1"]),
                float(val_metrics["precision"]),
                float(val_metrics["recall"]),
                float(threshold),
            ),
            flush=True,
        )
        if float(val_metrics["f1"]) > float(best["val_f1"]):
            best = {
                "epoch": epoch,
                "val_f1": float(val_metrics["f1"]),
                "threshold": float(threshold),
                "congestion_threshold": float(congestion_threshold),
            }
            patience = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "meta": payload["meta"],
                    "config": cfg,
                    "model_class": "CRCDMModel",
                    "format_version": 2,
                    "best": best,
                    "closure_pos_weight": float(closure_pos_weight.item()),
                    "congestion_pos_weight": float(congestion_pos_weight.item()),
                },
                run_dir / "checkpoints" / "best.pt",
            )
            write_json(run_dir / "best.json", best)
        else:
            patience += 1
        if patience >= int(train_cfg["early_stop_patience"]):
            print(f"early stopping at epoch {epoch}", flush=True)
            break

    write_json(
        run_dir / "run_info.json",
        {"run_dir": str(run_dir), "best": best, "device": str(device)},
    )
    print(
        f"Training complete. Best checkpoint: {run_dir / 'checkpoints' / 'best.pt'}",
        flush=True,
    )


if __name__ == "__main__":
    main()
