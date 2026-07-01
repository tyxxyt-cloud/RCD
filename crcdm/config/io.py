from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import yaml


def load_config(path: str | os.PathLike[str]) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["_config_path"] = str(path)
    return cfg


def ensure_dir(path: str | os.PathLike[str]) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def processed_path(cfg: Dict[str, Any]) -> Path:
    return ensure_dir(cfg["paths"]["processed_dir"]) / "crcdm_dataset.pt"


def manifest_path(cfg: Dict[str, Any]) -> Path:
    return ensure_dir(cfg["paths"]["processed_dir"]) / "manifest.json"


def make_run_dir(cfg: Dict[str, Any]) -> Path:
    root = ensure_dir(cfg["paths"]["output_root"])
    run_dir = root / time.strftime("run_%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=False)
    latest = root / "latest"
    if latest.is_symlink() or latest.exists():
        try:
            latest.unlink()
        except IsADirectoryError:
            pass
    try:
        latest.symlink_to(run_dir, target_is_directory=True)
    except OSError:
        pass
    return run_dir


def latest_run_dir(cfg: Dict[str, Any]) -> Path:
    latest = Path(cfg["paths"]["output_root"]) / "latest"
    if latest.exists():
        return latest.resolve()
    candidates = sorted(Path(cfg["paths"]["output_root"]).glob("run_*"))
    if not candidates:
        raise FileNotFoundError("No run directory found under outputs.")
    return candidates[-1]


def write_json(path: str | os.PathLike[str], payload: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_json(path: str | os.PathLike[str]) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
