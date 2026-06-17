from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset

from trdmu.config import processed_path


class TRDMUDataset(Dataset):
    def __init__(self, payload: Dict[str, Any], split: str):
        self.payload = payload
        self.split = split
        self.samples = payload["splits"][split]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        return self.samples[idx]


def load_processed_dataset(cfg: Dict[str, Any]) -> Dict[str, Any]:
    return torch.load(processed_path(cfg), map_location="cpu")


def collate_samples(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    y_closure = torch.tensor([float(x["y_closure"]) for x in batch], dtype=torch.float32)
    y_congestion = torch.tensor([float(x["y_congestion"]) for x in batch], dtype=torch.float32)
    return {
        "samples": batch,
        "y_closure": y_closure,
        "y_congestion": y_congestion,
    }


def pos_weight(samples: List[Dict[str, Any]], key: str) -> float:
    y = np.asarray([int(s[key]) for s in samples], dtype=np.int64)
    pos = int(y.sum())
    neg = int(len(y) - pos)
    if pos <= 0:
        return 1.0
    return float(max(1.0, neg / max(pos, 1)))
