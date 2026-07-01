# CRCDM

Code for **Causality-Aware Robust Road Closure Detection from Trajectories under
External Perturbations**.

## Requirements

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Data

Set the input paths in `configs/default.yaml`. The model uses road-level flow and
speed sequences, actual/planned trajectory paths, road topology, static road
attributes, closure labels, and congestion labels.

## Training

```bash
bash scripts/train.sh
```

## Evaluation

```bash
bash scripts/evaluate.sh
```

To run data preparation, training, and evaluation sequentially:

```bash
bash scripts/run_all.sh
```
