# CRCDM

Engineering reproduction of **Causality-Aware Robust Road Closure Detection from
Trajectories under External Perturbations**, focused on the main CRCDM method in
Sections 4.2-4.4.

The implementation provides a complete data0524 workflow:

1. build road-flow and trajectory-deviation samples;
2. train CRCDM with adversarial and CLUB disentanglement;
3. select closure and congestion thresholds on the validation split;
4. evaluate the best checkpoint and export test predictions.

This repository does not claim to reproduce the paper's BJ13, SH18, or CD18 table
values. The nine baselines, ablations, perturbation subsets, and sensitivity plots
are outside the current scope.

## Method coverage

- Traffic-flow semantics: GRU over each road's flow/speed history, static road
  feature fusion, and four-head target-specific graph attention.
- Trajectory-deviation semantics: actual/planned role encoding, static road
  feature fusion, two-layer relation-aware GCN, and trajectory-level LSTM.
- Adaptive perturbation learning: learnable basis bank, cross-attention,
  gradient-reversal adversarial disentanglement, and conditional Gaussian CLUB.
- Perturbation-attributed discrimination: counterfactual removal for every basis,
  soft causal routing, and separate closure/congestion prediction paths.

The `H` and `Z` road encoders have the same architecture but independent parameters.

## Road metadata contract

`data.method_data/tclosure/dataset/artifacts/road_metadata.csv.gz` must contain:

- required: `road_id`, `raw_eid`, `highway`, `length_m`, `start_lon`, `start_lat`,
  `end_lon`, `end_lat`;
- optional but preferred: `lanes`, `oneway`.

The node representation uses road type, normalized lane count, normalized road
length, direction as `sin(bearing)`/`cos(bearing)`, one-way state, and flags that
identify inferred lane/one-way values. Real `lanes` and `oneway` values take
precedence. Missing values use the configurable highway defaults under
`data.road_features`.

The generated manifest reports observed/inferred counts, coverage ratios, static
feature names, and normalization statistics. If all lane or one-way fields are
missing, the model remains runnable but the report makes the fallback explicit.

## Installation

Python 3.9 or newer is required.

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
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

## Tests

```bash
python3 -m compileall -q crcdm tests
pytest -q
```

The tests use a synthetic road network and trajectory set. They cover road feature
parsing/fallbacks, both semantic encoders, gradient reversal, CLUB estimation,
counterfactual routing, optimization, checkpoint restoration, metrics, and prediction
export without requiring the private data0524 files.

## Project structure

```text
configs/             Experiment and road-feature fallback configuration
scripts/             Build/train/evaluate shell entry points
crcdm/data/          data0524 preprocessing and PyTorch datasets
crcdm/models/        CRCDM architecture and losses
crcdm/training/      Alternating CLUB/main-model training
crcdm/evaluation/    Checkpoint evaluation and prediction export
crcdm/utils/         Metrics and shared helpers
tests/               Synthetic CPU tests
```
