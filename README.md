# CRCDM

Implementation for road closure representation learning and training.

## Structure

```text
configs/             YAML experiment configuration
scripts/             Shell entry points
trdmu/config/        Configuration, paths, seeds, and device helpers
trdmu/data/          Dataset construction and PyTorch dataset wrappers
trdmu/models/        TRDMU neural architecture and training losses
trdmu/training/      Training entry points
trdmu/utils/         Shared metrics and utilities
```

## Training

```bash
pip install -r requirements.txt
bash scripts/train.sh
```
