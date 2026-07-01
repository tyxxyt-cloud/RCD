from __future__ import annotations

import gzip
import json
import pickle
import subprocess
import sys
from pathlib import Path

import networkx as nx
import pandas as pd
import torch
import yaml


def test_synthetic_data0524_build_includes_static_features_and_report(
    tmp_path: Path,
) -> None:
    data_root = tmp_path / "data"
    artifacts = data_root / "artifacts"
    samples_root = data_root / "samples"
    processed = tmp_path / "processed"
    artifacts.mkdir(parents=True)
    samples_root.mkdir(parents=True)

    label_hour = pd.Timestamp("2026-01-01 01:00:00")
    common_rows = []
    for road_id, closure, congestion in [(1, 1, 0), (2, 0, 1)]:
        common_rows.append(
            {
                "road_id": road_id,
                "raw_eid": road_id,
                "highway": "primary" if road_id == 1 else "residential",
                "hour": label_hour,
                "obs_flow": 10.0,
                "obs_speed_mean": 20.0,
                "baseline_flow_hod": 12.0,
                "baseline_speed_hod": 25.0,
                "speed_ratio": 0.8,
                "flow_ratio": 0.7,
                "tv": 0.2,
                "ru": 0.1,
                "deviation_intensity": 0.3,
                "difficulty_type": "synthetic",
                "is_closure": closure,
                "is_congestion": congestion,
                "label_state": "closure" if closure else "normal",
            }
        )
    common_path = data_root / "common.csv.gz"
    pd.DataFrame(common_rows).to_csv(common_path, index=False, compression="gzip")

    pd.DataFrame(
        [
            {
                "road_id": 1,
                "raw_eid": 1,
                "highway": "primary",
                "length_m": 100.0,
                "lanes": 2,
                "oneway": "yes",
                "start_lon": 120.0,
                "start_lat": 31.0,
                "end_lon": 120.001,
                "end_lat": 31.0,
            },
            {
                "road_id": 2,
                "raw_eid": 2,
                "highway": "residential",
                "length_m": 80.0,
                "lanes": None,
                "oneway": None,
                "start_lon": 120.001,
                "start_lat": 31.0,
                "end_lon": 120.001,
                "end_lat": 31.001,
            },
        ]
    ).to_csv(artifacts / "road_metadata.csv", index=False)

    graph = nx.Graph()
    graph.add_edge(1, 2)
    with (artifacts / "road_graph.pkl").open("wb") as file:
        pickle.dump(graph, file)

    flow_rows = []
    for road_id in [1, 2]:
        for step in range(6):
            flow_rows.append(
                {
                    "road_id": road_id,
                    "bin_key": label_hour - pd.Timedelta(minutes=(5 - step) * 10),
                    "flow_count": 5 + road_id + step,
                    "mean_speed": 20 + step,
                }
            )
    pd.DataFrame(flow_rows).to_csv(artifacts / "road_flow.csv", index=False)

    trajectory = {
        "traj_id": 101,
        "actual_roads": [1, 2],
        "planned_roads": [1, 2],
        "actual_times": [],
        "actual_speeds": [],
    }
    with gzip.open(
        samples_root / "selected_trajectories.jsonl.gz", "wt", encoding="utf-8"
    ) as file:
        file.write(json.dumps(trajectory) + "\n")
    split_row = {
        "sample_id": "sample-1",
        "road_id": 1,
        "label_hour": label_hour,
        "traj_ids_json": "[101]",
        "event_ts_json": "[]",
        "event_types_json": "[]",
    }
    for split in ["train", "val", "test"]:
        pd.DataFrame([split_row]).to_csv(
            samples_root / f"{split}_samples.csv.gz",
            index=False,
            compression="gzip",
        )

    config = {
        "paths": {"processed_dir": str(processed)},
        "data": {
            "common_samples": str(common_path),
            "tclosure_samples_root": str(samples_root),
            "road_graph": str(artifacts / "road_graph.pkl"),
            "road_metadata": str(artifacts / "road_metadata.csv"),
            "road_flow": str(artifacts / "road_flow.csv"),
            "flow_steps": 6,
            "flow_bin_minutes": 10,
            "max_traj": 24,
            "k_hop": 1,
            "rebuild": True,
            "road_features": {
                "lane_defaults": {"primary": 2, "residential": 1, "default": 1},
                "oneway_defaults": {"default": False},
            },
        },
    }
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "trdmu.data.build_dataset",
            "--config",
            str(config_path),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    payload = torch.load(
        processed / "trdmu_dataset.pt", map_location="cpu", weights_only=False
    )
    assert payload["splits"]["train"][0]["traffic_static"].shape[1] == 7
    assert payload["splits"]["train"][0]["traj_graphs"][0]["static"].shape[1] == 7
    report = payload["meta"]["road_feature_report"]
    assert report["lanes_observed"] == 1
    assert report["lanes_inferred"] == 1
    assert report["oneway_observed"] == 1
    assert report["oneway_inferred"] == 1
    assert (
        json.loads((processed / "manifest.json").read_text())["road_feature_report"]
        == report
    )
