from __future__ import annotations

import argparse
import gzip
import json
import math
import pickle
from collections import deque
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import torch

from trdmu.config import ensure_dir, load_config, manifest_path, processed_path, write_json


TURN_STRAIGHT = 0
TURN_LEFT = 1
TURN_RIGHT = 2
TURN_UTURN = 3

DEFAULT_CONFIG = Path(__file__).resolve().parents[2] / "configs" / "default.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build processed TRDMU dataset from data0524.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def bearing_deg(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    x = lon2 - lon1
    y = lat2 - lat1
    if abs(x) + abs(y) < 1e-12:
        return 0.0
    return (math.degrees(math.atan2(x, y)) + 360.0) % 360.0


def angle_diff(a: float, b: float) -> float:
    return ((b - a + 180.0) % 360.0) - 180.0


def turn_type(u: int, v: int, bearings: Dict[int, float]) -> int:
    if u not in bearings or v not in bearings:
        return TURN_STRAIGHT
    diff = angle_diff(bearings[u], bearings[v])
    adiff = abs(diff)
    if adiff <= 35.0:
        return TURN_STRAIGHT
    if adiff >= 145.0:
        return TURN_UTURN
    return TURN_LEFT if diff > 0 else TURN_RIGHT


def hop_nodes(graph: nx.Graph, target: int, k_hop: int) -> List[int]:
    target = int(target)
    if target not in graph:
        return [target]
    seen = {target: 0}
    q: deque[int] = deque([target])
    while q:
        u = q.popleft()
        if seen[u] >= k_hop:
            continue
        for v in graph.neighbors(u):
            if v not in seen:
                seen[v] = seen[u] + 1
                q.append(v)
    return [target] + sorted([n for n in seen if n != target])


def z_norm(value: float, mean: float, std: float) -> float:
    std = std if abs(std) > 1e-9 else 1.0
    return float((value - mean) / std)


def load_trajectories(jsonl_gz: Path) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    with gzip.open(jsonl_gz, "rt", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            out[int(obj["traj_id"])] = obj
    return out


def parse_json_list(text: Any) -> List[Any]:
    if isinstance(text, list):
        return text
    if text is None or (isinstance(text, float) and math.isnan(text)):
        return []
    try:
        return json.loads(str(text))
    except Exception:
        return []


def ordered_union(*seqs: Iterable[int], target: int) -> List[int]:
    out = [int(target)]
    seen = {int(target)}
    for seq in seqs:
        for item in seq:
            try:
                rid = int(item)
            except Exception:
                continue
            if rid not in seen:
                out.append(rid)
                seen.add(rid)
    return out


def role_id(road_id: int, target: int, actual_set: set[int], planned_set: set[int]) -> int:
    in_actual = road_id in actual_set
    in_planned = road_id in planned_set
    is_target = road_id == target
    if is_target and in_actual and in_planned:
        return 6
    if is_target and in_actual:
        return 4
    if is_target and in_planned:
        return 5
    if is_target:
        return 3
    if in_actual and in_planned:
        return 2
    if in_planned:
        return 1
    return 0


def build_edges(
    actual: List[int],
    planned: List[int],
    node_to_idx: Dict[int, int],
    bearings: Dict[int, float],
    target_idx: int,
) -> Tuple[np.ndarray, np.ndarray]:
    edges: List[Tuple[int, int]] = []
    types: List[int] = []
    for seq in (actual, planned):
        for u, v in zip(seq[:-1], seq[1:]):
            if u in node_to_idx and v in node_to_idx:
                edges.append((node_to_idx[u], node_to_idx[v]))
                types.append(turn_type(int(u), int(v), bearings))
    if not edges:
        edges.append((target_idx, target_idx))
        types.append(TURN_STRAIGHT)
    return np.asarray(edges, dtype=np.int64).T, np.asarray(types, dtype=np.int64)


def make_static_feature(road_id: int, meta: Dict[int, Dict[str, Any]], length_mean: float, length_std: float) -> List[float]:
    item = meta.get(int(road_id), {})
    length = float(item.get("length_m", 0.0) or 0.0)
    bearing = float(item.get("bearing_deg", 0.0) or 0.0)
    rad = math.radians(bearing)
    return [
        z_norm(math.log1p(max(length, 0.0)), length_mean, length_std),
        math.sin(rad),
        math.cos(rad),
    ]


def make_flow_sequence(
    road_id: int,
    label_hour: pd.Timestamp,
    flow_lookup: Dict[Tuple[int, str], Tuple[float, float]],
    flow_steps: int,
    bin_minutes: int,
    flow_mean: float,
    flow_std: float,
    speed_mean: float,
    speed_std: float,
) -> List[List[float]]:
    seq: List[List[float]] = []
    for offset in range(flow_steps - 1, -1, -1):
        ts = label_hour - pd.Timedelta(minutes=offset * bin_minutes)
        key = (int(road_id), ts.strftime("%Y-%m-%d %H:%M:%S"))
        raw_flow, raw_speed = flow_lookup.get(key, (0.0, 0.0))
        seq.append(
            [
                z_norm(math.log1p(max(float(raw_flow), 0.0)), flow_mean, flow_std),
                z_norm(float(raw_speed), speed_mean, speed_std),
            ]
        )
    return seq


def build_one_traj_graph(
    payload: Dict[str, Any],
    target: int,
    meta: Dict[int, Dict[str, Any]],
    highway_vocab: Dict[str, int],
    bearings: Dict[int, float],
    length_mean: float,
    length_std: float,
) -> Dict[str, Any]:
    actual = [int(x) for x in payload.get("actual_roads", [])]
    planned = [int(x) for x in payload.get("planned_roads", [])]
    nodes = ordered_union(actual, planned, target=int(target))
    node_to_idx = {rid: i for i, rid in enumerate(nodes)}
    actual_set = set(actual)
    planned_set = set(planned)
    highway = [highway_vocab.get(str(meta.get(rid, {}).get("highway", "unknown")), 0) for rid in nodes]
    roles = [role_id(rid, int(target), actual_set, planned_set) for rid in nodes]
    static = [make_static_feature(rid, meta, length_mean, length_std) for rid in nodes]
    edge_index, edge_type = build_edges(actual, planned, node_to_idx, bearings, node_to_idx[int(target)])
    return {
        "road_ids": nodes,
        "highway": np.asarray(highway, dtype=np.int64),
        "role": np.asarray(roles, dtype=np.int64),
        "static": np.asarray(static, dtype=np.float32),
        "edge_index": edge_index,
        "edge_type": edge_type,
        "target_idx": int(node_to_idx[int(target)]),
        "actual_roads": actual,
        "planned_roads": planned,
        "start_ts": payload.get("start_ts"),
        "end_ts": payload.get("end_ts"),
        "divergence_ts": payload.get("divergence_ts"),
    }


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    out_path = processed_path(cfg)
    if out_path.exists() and not bool(cfg["data"].get("rebuild", False)):
        print(f"Processed dataset exists: {out_path}")
        return

    data_cfg = cfg["data"]
    ensure_dir(cfg["paths"]["processed_dir"])

    common = pd.read_csv(data_cfg["common_samples"], parse_dates=["hour"])
    common["road_id"] = common["road_id"].astype(int)
    common["label_hour"] = common["hour"].dt.floor("h").dt.strftime("%Y-%m-%d %H:%M:%S")

    road_meta_df = pd.read_csv(data_cfg["road_metadata"])
    road_meta_df["road_id"] = road_meta_df["road_id"].astype(int)
    road_meta_df["bearing_deg"] = road_meta_df.apply(
        lambda r: bearing_deg(float(r["start_lon"]), float(r["start_lat"]), float(r["end_lon"]), float(r["end_lat"])),
        axis=1,
    )
    highway_values = sorted({"unknown"} | {str(x) for x in road_meta_df["highway"].fillna("unknown").unique()})
    highway_vocab = {name: idx for idx, name in enumerate(highway_values)}
    meta = {
        int(r["road_id"]): {
            "raw_eid": int(r["raw_eid"]),
            "highway": str(r["highway"]),
            "length_m": float(r["length_m"]),
            "start_lon": float(r["start_lon"]),
            "start_lat": float(r["start_lat"]),
            "end_lon": float(r["end_lon"]),
            "end_lat": float(r["end_lat"]),
            "bearing_deg": float(r["bearing_deg"]),
        }
        for _, r in road_meta_df.iterrows()
    }
    bearings = {rid: item["bearing_deg"] for rid, item in meta.items()}

    length_logs = np.log1p(np.maximum(road_meta_df["length_m"].fillna(0.0).to_numpy(dtype=float), 0.0))
    length_mean = float(length_logs.mean())
    length_std = float(length_logs.std() or 1.0)

    flow_df = pd.read_csv(data_cfg["road_flow"], parse_dates=["bin_key"])
    flow_logs = np.log1p(np.maximum(flow_df["flow_count"].fillna(0.0).to_numpy(dtype=float), 0.0))
    speeds = flow_df["mean_speed"].fillna(0.0).to_numpy(dtype=float)
    flow_mean = float(flow_logs.mean())
    flow_std = float(flow_logs.std() or 1.0)
    speed_mean = float(speeds.mean())
    speed_std = float(speeds.std() or 1.0)
    flow_df["bin_key_str"] = flow_df["bin_key"].dt.strftime("%Y-%m-%d %H:%M:%S")
    flow_lookup = {
        (int(r.road_id), str(r.bin_key_str)): (float(r.flow_count), float(r.mean_speed))
        for r in flow_df.itertuples(index=False)
    }

    with open(data_cfg["road_graph"], "rb") as f:
        graph = pickle.load(f)
    trajectories = load_trajectories(Path(data_cfg["tclosure_samples_root"]) / "selected_trajectories.jsonl.gz")

    common_cols = [
        "road_id",
        "label_hour",
        "raw_eid",
        "highway",
        "obs_flow",
        "obs_speed_mean",
        "baseline_flow_hod",
        "baseline_speed_hod",
        "speed_ratio",
        "flow_ratio",
        "tv",
        "ru",
        "deviation_intensity",
        "difficulty_type",
        "is_closure",
        "is_congestion",
        "label_state",
    ]
    common_small = common[common_cols].copy()

    splits: Dict[str, List[Dict[str, Any]]] = {}
    missing_traj = 0
    skipped = 0
    for split in ["train", "val", "test"]:
        samples_path = Path(data_cfg["tclosure_samples_root"]) / f"{split}_samples.csv.gz"
        df = pd.read_csv(samples_path, parse_dates=["label_hour"])
        df["road_id"] = df["road_id"].astype(int)
        df["label_hour"] = df["label_hour"].dt.floor("h").dt.strftime("%Y-%m-%d %H:%M:%S")
        merged = df.merge(common_small, on=["road_id", "label_hour"], how="left", suffixes=("", "_common"))
        split_samples: List[Dict[str, Any]] = []
        for row in merged.itertuples(index=False):
            if pd.isna(getattr(row, "is_closure")) or pd.isna(getattr(row, "is_congestion")):
                skipped += 1
                continue
            target = int(row.road_id)
            label_hour = pd.Timestamp(row.label_hour)
            local_nodes = hop_nodes(graph, target, int(data_cfg["k_hop"]))
            traffic_flow = [
                make_flow_sequence(
                    rid,
                    label_hour,
                    flow_lookup,
                    int(data_cfg["flow_steps"]),
                    int(data_cfg["flow_bin_minutes"]),
                    flow_mean,
                    flow_std,
                    speed_mean,
                    speed_std,
                )
                for rid in local_nodes
            ]
            traffic_highway = [
                highway_vocab.get(str(meta.get(rid, {}).get("highway", "unknown")), 0) for rid in local_nodes
            ]
            traffic_static = [
                make_static_feature(rid, meta, length_mean, length_std) for rid in local_nodes
            ]

            traj_ids = [int(x) for x in parse_json_list(getattr(row, "traj_ids_json"))][: int(data_cfg["max_traj"])]
            event_ts = parse_json_list(getattr(row, "event_ts_json", "[]"))
            event_types = parse_json_list(getattr(row, "event_types_json", "[]"))
            traj_graphs: List[Dict[str, Any]] = []
            traj_payloads: List[Dict[str, Any]] = []
            for tid in traj_ids:
                payload = trajectories.get(tid)
                if payload is None:
                    missing_traj += 1
                    continue
                traj_graphs.append(
                    build_one_traj_graph(
                        payload,
                        target,
                        meta,
                        highway_vocab,
                        bearings,
                        length_mean,
                        length_std,
                    )
                )
                traj_payloads.append(
                    {
                        "traj_id": int(tid),
                        "actual_roads": [int(x) for x in payload.get("actual_roads", [])],
                        "planned_roads": [int(x) for x in payload.get("planned_roads", [])],
                        "actual_times": payload.get("actual_times", []),
                        "actual_speeds": payload.get("actual_speeds", []),
                        "start_ts": payload.get("start_ts"),
                        "end_ts": payload.get("end_ts"),
                        "divergence_ts": payload.get("divergence_ts"),
                    }
                )
            if not traj_graphs:
                skipped += 1
                continue

            road_geo = meta.get(target, {})
            sample = {
                "sample_id": str(row.sample_id),
                "split": split,
                "road_id": target,
                "label_hour": str(row.label_hour),
                "y_closure": int(row.is_closure),
                "y_congestion": int(row.is_congestion),
                "difficulty_type": str(row.difficulty_type),
                "label_state": str(row.label_state),
                "traffic_nodes": local_nodes,
                "traffic_flow": np.asarray(traffic_flow, dtype=np.float32),
                "traffic_highway": np.asarray(traffic_highway, dtype=np.int64),
                "traffic_static": np.asarray(traffic_static, dtype=np.float32),
                "traj_graphs": traj_graphs,
                "traj_ids": traj_ids,
                "event_ts": event_ts[: int(data_cfg["max_traj"])],
                "event_types": event_types[: int(data_cfg["max_traj"])],
                "trajectories": traj_payloads,
                "road_geo": {
                    **road_geo,
                    "line_string": [
                        [road_geo.get("start_lon"), road_geo.get("start_lat")],
                        [road_geo.get("end_lon"), road_geo.get("end_lat")],
                    ],
                },
                "features": {
                    "obs_flow": None if pd.isna(row.obs_flow) else float(row.obs_flow),
                    "obs_speed_mean": None if pd.isna(row.obs_speed_mean) else float(row.obs_speed_mean),
                    "baseline_flow_hod": None if pd.isna(row.baseline_flow_hod) else float(row.baseline_flow_hod),
                    "baseline_speed_hod": None if pd.isna(row.baseline_speed_hod) else float(row.baseline_speed_hod),
                    "speed_ratio": None if pd.isna(row.speed_ratio) else float(row.speed_ratio),
                    "flow_ratio": None if pd.isna(row.flow_ratio) else float(row.flow_ratio),
                    "tv": None if pd.isna(row.tv) else float(row.tv),
                    "ru": None if pd.isna(row.ru) else float(row.ru),
                    "deviation_intensity": None
                    if pd.isna(row.deviation_intensity)
                    else float(row.deviation_intensity),
                },
            }
            split_samples.append(sample)
        splits[split] = split_samples
        print(f"{split}: {len(split_samples)} samples built from {len(df)} rows")

    dataset = {
        "splits": splits,
        "meta": {
            "highway_vocab": highway_vocab,
            "role_vocab_size": 7,
            "turn_vocab_size": 4,
            "static_dim": 3,
            "flow_dim": 2,
            "length_mean": length_mean,
            "length_std": length_std,
            "flow_mean": flow_mean,
            "flow_std": flow_std,
            "speed_mean": speed_mean,
            "speed_std": speed_std,
            "missing_traj": int(missing_traj),
            "skipped_rows": int(skipped),
            "source_rows": {k: int(len(v)) for k, v in splits.items()},
        },
    }
    torch.save(dataset, out_path)
    write_json(
        manifest_path(cfg),
        {
            "processed_path": str(out_path),
            "splits": {k: len(v) for k, v in splits.items()},
            "missing_traj": int(missing_traj),
            "skipped_rows": int(skipped),
            "highway_vocab": highway_vocab,
        },
    )
    print(f"Saved processed dataset to {out_path}")


if __name__ == "__main__":
    main()
