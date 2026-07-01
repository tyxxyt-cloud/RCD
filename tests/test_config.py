from pathlib import Path

import yaml


def test_default_configuration_matches_reported_paper_settings() -> None:
    path = Path(__file__).resolve().parents[1] / "configs" / "default.yaml"
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    assert config["model"]["hidden_dim"] == 512
    assert config["model"]["perturbation_dim"] == 256
    assert config["model"]["perturbation_bases"] == 14
    assert config["model"]["attention_heads"] == 4
    assert config["model"]["rgcn_layers"] == 2
    assert config["data"]["max_traj"] == 24
    assert config["training"]["batch_size"] == 64
    assert config["training"]["learning_rate"] == 1e-3
    assert config["training"]["weight_decay"] == 1e-5
    assert config["training"]["lambda_mi"] == 0.1
    assert config["training"]["lambda_con"] == 0.5
