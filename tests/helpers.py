from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict

import numpy as np


def tiny_config(tmp_path: Any = None) -> Dict[str, Any]:
    root = str(tmp_path) if tmp_path is not None else "."
    return {
        "paths": {
            "processed_dir": f"{root}/processed",
            "output_root": f"{root}/outputs",
        },
        "data": {},
        "model": {
            "hidden_dim": 16,
            "perturbation_dim": 8,
            "perturbation_bases": 3,
            "attention_heads": 4,
            "highway_emb_dim": 4,
            "role_emb_dim": 4,
            "dropout": 0.0,
            "rgcn_layers": 2,
        },
        "training": {
            "device": "cpu",
            "batch_size": 2,
            "lambda_grl": 1.0,
        },
        "evaluation": {"split": "test"},
    }


def tiny_meta() -> Dict[str, Any]:
    return {
        "highway_vocab": {"unknown": 0, "primary": 1},
        "role_vocab_size": 7,
        "turn_vocab_size": 4,
        "static_dim": 7,
        "static_feature_names": [
            "length_log_z",
            "lanes_z",
            "bearing_sin",
            "bearing_cos",
            "oneway",
            "lanes_inferred",
            "oneway_inferred",
        ],
        "flow_dim": 2,
    }


def tiny_sample(sample_id: str, label: int, offset: float) -> Dict[str, Any]:
    static = np.asarray(
        [
            [offset, 0.5 * offset, 0.0, 1.0, 1.0, 0.0, 0.0],
            [0.2 + offset, -0.2, 1.0, 0.0, 0.0, 1.0, 1.0],
            [-0.1, 0.1 + offset, -1.0, 0.0, 0.0, 1.0, 1.0],
        ],
        dtype=np.float32,
    )
    flow = np.asarray(
        [
            [[offset, 0.1], [offset + 0.1, 0.2], [offset + 0.2, 0.3]],
            [[0.1, offset], [0.2, offset + 0.1], [0.3, offset + 0.2]],
            [[-offset, 0.2], [-offset, 0.1], [-offset, 0.0]],
        ],
        dtype=np.float32,
    )
    graph = {
        "road_ids": [10, 11, 12],
        "highway": np.asarray([1, 1, 0], dtype=np.int64),
        "role": np.asarray([6, 1, 0], dtype=np.int64),
        "static": static,
        "edge_index": np.asarray([[0, 1, 0, 2], [1, 2, 2, 0]], dtype=np.int64),
        "edge_type": np.asarray([0, 1, 2, 3], dtype=np.int64),
        "target_idx": 0,
    }
    return {
        "sample_id": sample_id,
        "split": "test",
        "road_id": 10 + label,
        "label_hour": "2026-01-01 00:00:00",
        "y_closure": label,
        "y_congestion": 1 - label,
        "traffic_nodes": [10, 11, 12],
        "traffic_flow": flow,
        "traffic_highway": np.asarray([1, 1, 0], dtype=np.int64),
        "traffic_static": static,
        "traj_graphs": [deepcopy(graph), deepcopy(graph)],
    }


def tiny_payload() -> Dict[str, Any]:
    samples = [tiny_sample("negative", 0, -1.0), tiny_sample("positive", 1, 1.0)]
    return {
        "splits": {
            "train": deepcopy(samples),
            "val": deepcopy(samples),
            "test": deepcopy(samples),
        },
        "meta": tiny_meta(),
    }
