# CRCDM

Code for **Causality-aware Robust Road Closure Detection from Trajectories under
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

The complete trajectory datasets are available from:

- **BJ13:** [Baidu Netdisk](https://pan.baidu.com/s/1nFGB60NNARCqp__Q6igv9A?pwd=hh8h)
- **SH18:** [Shanghai Open Data Platform](https://soda.data.sh.gov.cn/competitionData.html)
- **CD18:** [Baidu Netdisk](https://pan.baidu.com/s/1NQB3qsFugmbRs2dOGApL8w?pwd=kkft)

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

The included one-batch example can be run with `bash scripts/run_example.sh`.

## Project Structure

```text
RCD/
├── configs/
│   ├── default.yaml
│   └── example.yaml
├── crcdm/
│   ├── config/
│   ├── data/
│   ├── evaluation/
│   ├── models/
│   ├── training/
│   └── utils/
├── examples/
│   └── data/
├── scripts/
│   ├── evaluate.sh
│   ├── run_all.sh
│   ├── run_example.sh
│   └── train.sh
├── LICENSE
├── README.md
├── pyproject.toml
└── requirements.txt
```
