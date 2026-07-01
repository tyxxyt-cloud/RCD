from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List

import torch
import torch.nn as nn
import torch.nn.functional as F


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, x: torch.Tensor, lambd: float) -> torch.Tensor:
        ctx.lambd = float(lambd)
        return x.view_as(x)

    @staticmethod
    def backward(ctx: Any, grad_output: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.lambd * grad_output, None


def grl(x: torch.Tensor, lambd: float) -> torch.Tensor:
    return GradientReversal.apply(x, lambd)


class MLP(nn.Module):
    def __init__(
        self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0
    ):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TrafficEncoder(nn.Module):
    """Equation (2): GRU flow encoding followed by target-specific multi-head GAT."""

    def __init__(self, num_highways: int, cfg: Dict[str, Any], static_dim: int):
        super().__init__()
        hidden = int(cfg["model"]["hidden_dim"])
        highway_dim = int(cfg["model"]["highway_emb_dim"])
        heads = int(cfg["model"].get("attention_heads", 4))
        dropout = float(cfg["model"]["dropout"])
        if hidden % heads:
            raise ValueError(
                "model.hidden_dim must be divisible by model.attention_heads"
            )
        self.highway_emb = nn.Embedding(num_highways, highway_dim)
        self.flow_in = nn.Linear(2, hidden)
        self.flow_gru = nn.GRU(hidden, hidden, batch_first=True)
        self.node_proj = nn.Sequential(
            nn.Linear(hidden + highway_dim + static_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.target_gat = nn.MultiheadAttention(
            hidden,
            heads,
            dropout=dropout,
            batch_first=True,
        )
        self.out = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout)
        )

    def forward_one(self, sample: Dict[str, Any], device: torch.device) -> torch.Tensor:
        flow = torch.as_tensor(
            sample["traffic_flow"], dtype=torch.float32, device=device
        )
        highway = torch.as_tensor(
            sample["traffic_highway"], dtype=torch.long, device=device
        )
        static = torch.as_tensor(
            sample["traffic_static"], dtype=torch.float32, device=device
        )
        x = self.flow_in(flow)
        _, final_state = self.flow_gru(x)
        flow_h = final_state[-1]
        node_h = self.node_proj(
            torch.cat([flow_h, self.highway_emb(highway), static], dim=-1)
        )
        query = node_h[:1].unsqueeze(0)
        key_value = node_h.unsqueeze(0)
        target_h, _ = self.target_gat(query, key_value, key_value, need_weights=False)
        return self.out(target_h[0, 0])


class RGCNLayer(nn.Module):
    def __init__(self, hidden: int, rel_count: int, dropout: float):
        super().__init__()
        self.rel_lins = nn.ModuleList(
            [nn.Linear(hidden, hidden, bias=False) for _ in range(rel_count)]
        )
        self.self_lin = nn.Linear(hidden, hidden)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor
    ) -> torch.Tensor:
        aggregated = torch.zeros_like(x)
        degree = torch.zeros(x.size(0), dtype=x.dtype, device=x.device)
        src_all, dst_all = edge_index[0], edge_index[1]
        for relation, transform in enumerate(self.rel_lins):
            mask = edge_type == relation
            if not torch.any(mask):
                continue
            src = src_all[mask]
            dst = dst_all[mask]
            aggregated.index_add_(0, dst, transform(x[src]))
            degree.index_add_(0, dst, torch.ones_like(dst, dtype=x.dtype))
        aggregated = aggregated / degree.clamp_min(1.0).unsqueeze(-1)
        return self.dropout(F.relu(self.norm(self.self_lin(x) + aggregated)))


class TrajectoryDeviationEncoder(nn.Module):
    """Equation (3): static/role-aware R-GCN followed by trajectory-level LSTM."""

    def __init__(
        self,
        num_highways: int,
        cfg: Dict[str, Any],
        role_vocab_size: int,
        static_dim: int,
    ):
        super().__init__()
        hidden = int(cfg["model"]["hidden_dim"])
        highway_dim = int(cfg["model"]["highway_emb_dim"])
        role_dim = int(cfg["model"]["role_emb_dim"])
        dropout = float(cfg["model"]["dropout"])
        self.highway_emb = nn.Embedding(num_highways, highway_dim)
        self.role_emb = nn.Embedding(role_vocab_size, role_dim)
        self.node_proj = nn.Sequential(
            nn.Linear(highway_dim + role_dim + static_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList(
            [
                RGCNLayer(hidden, 4, dropout)
                for _ in range(int(cfg["model"]["rgcn_layers"]))
            ]
        )
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)

    def encode_graph(self, graph: Dict[str, Any], device: torch.device) -> torch.Tensor:
        highway = torch.as_tensor(graph["highway"], dtype=torch.long, device=device)
        role = torch.as_tensor(graph["role"], dtype=torch.long, device=device)
        static = torch.as_tensor(graph["static"], dtype=torch.float32, device=device)
        edge_index = torch.as_tensor(
            graph["edge_index"], dtype=torch.long, device=device
        )
        edge_type = torch.as_tensor(graph["edge_type"], dtype=torch.long, device=device)
        node_h = self.node_proj(
            torch.cat([self.highway_emb(highway), self.role_emb(role), static], dim=-1)
        )
        for layer in self.layers:
            node_h = layer(node_h, edge_index, edge_type)
        return node_h[int(graph["target_idx"])]

    def forward_one(self, sample: Dict[str, Any], device: torch.device) -> torch.Tensor:
        graph_embeddings = [
            self.encode_graph(graph, device) for graph in sample["traj_graphs"]
        ]
        if not graph_embeddings:
            return torch.zeros(
                self.lstm.hidden_size, dtype=torch.float32, device=device
            )
        sequence = torch.stack(graph_embeddings, dim=0).unsqueeze(0)
        _, (final_state, _) = self.lstm(sequence)
        return final_state[-1, 0]


class RoadEncoder(nn.Module):
    """Equation (4): gated fusion of flow and deviation semantics."""

    def __init__(
        self,
        num_highways: int,
        cfg: Dict[str, Any],
        role_vocab_size: int,
        static_dim: int,
    ):
        super().__init__()
        hidden = int(cfg["model"]["hidden_dim"])
        self.traffic = TrafficEncoder(num_highways, cfg, static_dim=static_dim)
        self.deviation = TrajectoryDeviationEncoder(
            num_highways,
            cfg,
            role_vocab_size=role_vocab_size,
            static_dim=static_dim,
        )
        self.gate = nn.Linear(hidden * 2, hidden)

    def forward_one(
        self, sample: Dict[str, Any], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        flow_h = self.traffic.forward_one(sample, device)
        deviation_h = self.deviation.forward_one(sample, device)
        gate = torch.sigmoid(self.gate(torch.cat([flow_h, deviation_h], dim=-1)))
        fused = gate * flow_h + (1.0 - gate) * deviation_h
        return {"fused": fused, "flow": flow_h, "deviation": deviation_h, "gate": gate}

    def forward(
        self, samples: List[Dict[str, Any]], device: torch.device
    ) -> Dict[str, torch.Tensor]:
        encoded = [self.forward_one(sample, device) for sample in samples]
        return {
            key: torch.stack([item[key] for item in encoded], dim=0)
            for key in encoded[0]
        }


class ConditionalGaussian(nn.Module):
    """Variational q_phi(Z|C) used by the CLUB objective in Equation (8)."""

    def __init__(self, c_dim: int, z_dim: int, hidden_dim: int):
        super().__init__()
        self.backbone = MLP(c_dim, hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, z_dim)
        self.logvar = nn.Linear(hidden_dim, z_dim)

    def parameters_for(self, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        hidden = self.backbone(c)
        return self.mean(hidden), self.logvar(hidden).clamp(-8.0, 8.0)

    def log_prob(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        mean, logvar = self.parameters_for(c)
        return -0.5 * (
            (z - mean).pow(2) * torch.exp(-logvar) + logvar + math.log(2.0 * math.pi)
        ).sum(dim=-1)

    def pairwise_log_prob(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        mean, logvar = self.parameters_for(c)
        difference = z.unsqueeze(0) - mean.unsqueeze(1)
        return -0.5 * (
            difference.pow(2) * torch.exp(-logvar).unsqueeze(1)
            + logvar.unsqueeze(1)
            + math.log(2.0 * math.pi)
        ).sum(dim=-1)


class CRCDMModel(nn.Module):
    """CRCDM Sections 4.2-4.4, with independent H and Z road encoders."""

    def __init__(self, cfg: Dict[str, Any], meta: Dict[str, Any]):
        super().__init__()
        model_cfg = cfg["model"]
        hidden = int(model_cfg["hidden_dim"])
        perturbation_dim = int(model_cfg.get("perturbation_dim", hidden))
        dropout = float(model_cfg["dropout"])
        basis_count = int(model_cfg["perturbation_bases"])
        num_highways = len(meta["highway_vocab"])
        role_vocab_size = int(meta["role_vocab_size"])
        static_dim = int(meta["static_dim"])
        self.hidden = hidden
        self.perturbation_dim = perturbation_dim
        self.k = basis_count
        self.encoder_h = RoadEncoder(num_highways, cfg, role_vocab_size, static_dim)
        self.encoder_z = RoadEncoder(num_highways, cfg, role_vocab_size, static_dim)
        self.basis = nn.Parameter(torch.randn(basis_count, perturbation_dim) * 0.02)
        self.query = nn.Linear(hidden, perturbation_dim, bias=False)
        self.key = nn.Linear(perturbation_dim, perturbation_dim, bias=False)
        self.discriminator = MLP(hidden, hidden, perturbation_dim, dropout)
        self.mi_estimator = ConditionalGaussian(perturbation_dim, hidden, hidden)
        route_dim = hidden + perturbation_dim
        self.expert_rel = MLP(perturbation_dim, hidden, perturbation_dim, dropout)
        self.expert_irr = MLP(perturbation_dim, hidden, perturbation_dim, dropout)
        self.closure_head = MLP(route_dim, hidden, 1, dropout)
        self.congestion_head = MLP(route_dim, hidden, 1, dropout)

    def main_parameters(self) -> Iterable[nn.Parameter]:
        for name, parameter in self.named_parameters():
            if not name.startswith("mi_estimator."):
                yield parameter

    def perturbation_distribution(
        self, h: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query(h)
        keys = self.key(self.basis)
        alpha = torch.softmax(
            (query @ keys.t()) / math.sqrt(float(self.perturbation_dim)), dim=-1
        )
        return alpha, alpha @ self.basis

    def representations(
        self,
        samples: List[Dict[str, Any]],
        device: torch.device,
    ) -> Dict[str, torch.Tensor]:
        h_pack = self.encoder_h(samples, device)
        z_pack = self.encoder_z(samples, device)
        alpha, c = self.perturbation_distribution(h_pack["fused"])
        return {
            "h_pack": h_pack,
            "z_pack": z_pack,
            "h": h_pack["fused"],
            "z": z_pack["fused"],
            "alpha": alpha,
            "c": c,
        }

    def mi_estimator_loss(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        return -self.mi_estimator.log_prob(z.detach(), c.detach()).mean()

    def mi_upper_bound(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        pairwise = self.mi_estimator.pairwise_log_prob(z, c)
        matched = torch.diagonal(pairwise)
        return (matched - pairwise.mean(dim=1)).mean()

    def route(
        self, z: torch.Tensor, alpha: torch.Tensor, c: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        full_input = torch.cat([z, c], dim=-1)
        full_closure = torch.sigmoid(self.closure_head(full_input)).squeeze(-1)
        full_irrelevant = torch.sigmoid(self.congestion_head(full_input)).squeeze(-1)
        delta_closure = []
        delta_irrelevant = []
        for basis_index in range(self.k):
            component = (
                alpha[:, basis_index : basis_index + 1] * self.basis[basis_index]
            )
            counterfactual = torch.cat([z, c - component], dim=-1)
            delta_closure.append(
                full_closure
                - torch.sigmoid(self.closure_head(counterfactual)).squeeze(-1)
            )
            delta_irrelevant.append(
                full_irrelevant
                - torch.sigmoid(self.congestion_head(counterfactual)).squeeze(-1)
            )
        delta_closure_t = torch.stack(delta_closure, dim=-1)
        delta_irrelevant_t = torch.stack(delta_irrelevant, dim=-1)
        causal_gate = torch.sigmoid(delta_closure_t - delta_irrelevant_t)
        rel_weight = alpha * causal_gate
        irr_weight = alpha * (1.0 - causal_gate)
        c_rel = self.expert_rel(rel_weight @ self.basis)
        c_irr = self.expert_irr(irr_weight @ self.basis)
        return {
            "causal_gate": causal_gate,
            "delta_closure": delta_closure_t,
            "delta_irrelevant": delta_irrelevant_t,
            "rel_weight": rel_weight,
            "irr_weight": irr_weight,
            "c_rel": c_rel,
            "c_irr": c_irr,
        }

    def forward(
        self,
        samples: List[Dict[str, Any]],
        device: torch.device,
        lambda_grl: float = 1.0,
    ) -> Dict[str, torch.Tensor]:
        representation = self.representations(samples, device)
        z = representation["z"]
        c = representation["c"]
        adv_pred = self.discriminator(grl(z, lambda_grl))
        routed = self.route(z, representation["alpha"], c)
        closure_logit = self.closure_head(
            torch.cat([z, routed["c_rel"]], dim=-1)
        ).squeeze(-1)
        congestion_logit = self.congestion_head(
            torch.cat([z, routed["c_irr"]], dim=-1)
        ).squeeze(-1)
        return {
            **representation,
            **routed,
            "adv_pred": adv_pred,
            "adv_loss": F.mse_loss(adv_pred, c.detach()),
            "mi_loss": self.mi_upper_bound(z, c),
            "closure_logit": closure_logit,
            "congestion_logit": congestion_logit,
            "closure_prob": torch.sigmoid(closure_logit),
            "congestion_prob": torch.sigmoid(congestion_logit),
        }


# Backward-compatible public name used by the first repository release.
TRDMUModel = CRCDMModel


def compute_loss(
    out: Dict[str, torch.Tensor],
    y_closure: torch.Tensor,
    y_congestion: torch.Tensor,
    closure_pos_weight: torch.Tensor,
    congestion_pos_weight: torch.Tensor,
    lambda_con: float,
    lambda_mi: float,
    lambda_dis: float = 1.0,
) -> Dict[str, torch.Tensor]:
    y_closure = y_closure.to(out["closure_logit"].device)
    y_congestion = y_congestion.to(out["congestion_logit"].device)
    loss_closure = F.binary_cross_entropy_with_logits(
        out["closure_logit"],
        y_closure,
        pos_weight=closure_pos_weight.to(out["closure_logit"].device),
    )
    loss_congestion = F.binary_cross_entropy_with_logits(
        out["congestion_logit"],
        y_congestion,
        pos_weight=congestion_pos_weight.to(out["congestion_logit"].device),
    )
    loss_disentanglement = out["adv_loss"] + float(lambda_mi) * out["mi_loss"]
    loss = (
        loss_closure
        + float(lambda_con) * loss_congestion
        + float(lambda_dis) * loss_disentanglement
    )
    return {
        "loss": loss,
        "loss_closure": loss_closure.detach(),
        "loss_congestion": loss_congestion.detach(),
        "loss_adv": out["adv_loss"].detach(),
        "loss_mi": out["mi_loss"].detach(),
        "loss_disentanglement": loss_disentanglement.detach(),
    }
