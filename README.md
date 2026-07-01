# CRCDM

Engineering reproduction of **Causality-Aware Robust Road Closure Detection from
Trajectories under External Perturbations**



## Installation

Python 3.9 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Configuration

Edit `configs/default.yaml` to point at the local data0524 files. The checked-in
defaults follow the paper where specified:

- hidden dimension `D=512`;
- perturbation dimension `d=256`;
- `K=14` perturbation bases;
- at most `M=24` target-road trajectories;
- four attention heads, two R-GCN layers, dropout `0.3`;
- Adam learning rate `1e-3`, weight decay `1e-5`, batch size `64`;
- `lambda_mi=0.1`, `lambda_con=0.5`.

The paper does not specify `lambda_dis`; this implementation exposes it in the
configuration and defaults it to `1.0`.

## Run

Complete build/train/evaluate workflow:

```bash
bash scripts/run_all.sh
```

Individual stages:

```bash
python3 -m crcdm.data.build_dataset --config configs/default.yaml
python3 -m crcdm.training.train --config configs/default.yaml
python3 -m crcdm.evaluation.evaluate --config configs/default.yaml --split test
```

Training outputs are written under `outputs/run_YYYYMMDD_HHMMSS/`. Evaluation adds:

- `evaluation/test_metrics.json` with Precision, Recall, F1, PR-AUC, ROC-AUC,
  thresholds, and confusion matrices;
- `evaluation/test_predictions.csv.gz` with sample identifiers, labels,
  probabilities, and binary predictions for both paths.

## Project structure

```text
configs/             Experiment and road-feature fallback configuration
scripts/             Build/train/evaluate shell entry points
crcdm/data/          data0524 preprocessing and PyTorch datasets
crcdm/models/        CRCDM architecture and losses
crcdm/training/      Alternating CLUB/main-model training
crcdm/evaluation/    Checkpoint evaluation and prediction export
crcdm/utils/         Metrics and shared helpers
```
