from __future__ import annotations

import math
from typing import Any, Dict, List

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
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, dropout: float = 0.0):
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
    def __init__(self, num_highways: int, cfg: Dict[str, Any], static_dim: int = 3):
        super().__init__()
        hidden = int(cfg["model"]["hidden_dim"])
        highway_dim = int(cfg["model"]["highway_emb_dim"])
        dropout = float(cfg["model"]["dropout"])
        self.highway_emb = nn.Embedding(num_highways, highway_dim)
        self.flow_in = nn.Linear(2, hidden)
        self.flow_gru = nn.GRU(hidden, hidden, batch_first=True)
        self.node_proj = nn.Sequential(
            nn.Linear(hidden + highway_dim + static_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.q = nn.Linear(hidden, hidden)
        self.k = nn.Linear(hidden, hidden)
        self.v = nn.Linear(hidden, hidden)
        self.out = nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Dropout(dropout))

    def forward_one(self, sample: Dict[str, Any], device: torch.device) -> torch.Tensor:
        flow = torch.as_tensor(sample["traffic_flow"], dtype=torch.float32, device=device)
        highway = torch.as_tensor(sample["traffic_highway"], dtype=torch.long, device=device)
        static = torch.as_tensor(sample["traffic_static"], dtype=torch.float32, device=device)
        x = self.flow_in(flow)
        _, h = self.flow_gru(x)
        h = h.squeeze(0)
        node_h = self.node_proj(torch.cat([h, self.highway_emb(highway), static], dim=-1))
        q = self.q(node_h[:1])
        k = self.k(node_h)
        v = self.v(node_h)
        attn = torch.softmax((q @ k.t()).squeeze(0) / math.sqrt(k.size(-1)), dim=-1)
        return self.out(torch.sum(attn.unsqueeze(-1) * v, dim=0))


class RGCNLayer(nn.Module):
    def __init__(self, hidden: int, rel_count: int, dropout: float):
        super().__init__()
        self.rel_lins = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(rel_count)])
        self.self_lin = nn.Linear(hidden, hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_type: torch.Tensor) -> torch.Tensor:
        out = torch.zeros_like(x)
        deg = torch.zeros(x.size(0), device=x.device).clamp_min_(0.0)
        src_all, dst_all = edge_index[0], edge_index[1]
        for rel, lin in enumerate(self.rel_lins):
            mask = edge_type == rel
            if not torch.any(mask):
                continue
            src = src_all[mask]
            dst = dst_all[mask]
            msg = lin(x[src])
            out.index_add_(0, dst, msg)
            deg.index_add_(0, dst, torch.ones_like(dst, dtype=torch.float32))
        out = out / deg.clamp_min(1.0).unsqueeze(-1)
        return self.dropout(F.relu(self.self_lin(x) + out))


class TrajectoryDeviationEncoder(nn.Module):
    def __init__(self, num_highways: int, cfg: Dict[str, Any], role_vocab_size: int, static_dim: int = 3):
        super().__init__()
        hidden = int(cfg["model"]["hidden_dim"])
        highway_dim = int(cfg["model"]["highway_emb_dim"])
        role_dim = int(cfg["model"]["role_emb_dim"])
        dropout = float(cfg["model"]["dropout"])
        rel_count = 4
        self.highway_emb = nn.Embedding(num_highways, highway_dim)
        self.role_emb = nn.Embedding(role_vocab_size + 1, role_dim)
        self.node_proj = nn.Sequential(
            nn.Linear(highway_dim + role_dim + static_dim, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.layers = nn.ModuleList(
            [RGCNLayer(hidden, rel_count, dropout) for _ in range(int(cfg["model"]["rgcn_layers"]))]
        )
        self.lstm = nn.LSTM(hidden, hidden, batch_first=True)

    def encode_graph(self, graph: Dict[str, Any], device: torch.device) -> torch.Tensor:
        highway = torch.as_tensor(graph["highway"], dtype=torch.long, device=device)
        role = torch.as_tensor(graph["role"], dtype=torch.long, device=device)
        static = torch.as_tensor(graph["static"], dtype=torch.float32, device=device)
        edge_index = torch.as_tensor(graph["edge_index"], dtype=torch.long, device=device)
        edge_type = torch.as_tensor(graph["edge_type"], dtype=torch.long, device=device)
        x = self.node_proj(torch.cat([self.highway_emb(highway), self.role_emb(role), static], dim=-1))
        for layer in self.layers:
            x = layer(x, edge_index, edge_type)
        return x[int(graph["target_idx"])]

    def forward_one(self, sample: Dict[str, Any], device: torch.device) -> torch.Tensor:
        graph_embs = [self.encode_graph(g, device) for g in sample["traj_graphs"]]
        if not graph_embs:
            hidden = self.lstm.hidden_size
            return torch.zeros(hidden, dtype=torch.float32, device=device)
        seq = torch.stack(graph_embs, dim=0).unsqueeze(0)
        out, _ = self.lstm(seq)
        return out[0, -1]


class RoadEncoder(nn.Module):
    def __init__(self, num_highways: int, cfg: Dict[str, Any], role_vocab_size: int, static_dim: int):
        super().__init__()
        hidden = int(cfg["model"]["hidden_dim"])
        self.traffic = TrafficEncoder(num_highways, cfg, static_dim=static_dim)
        self.deviation = TrajectoryDeviationEncoder(
            num_highways, cfg, role_vocab_size=role_vocab_size, static_dim=static_dim
        )
        self.gate = nn.Linear(hidden * 2, hidden)

    def forward_one(self, sample: Dict[str, Any], device: torch.device) -> torch.Tensor:
        h_f = self.traffic.forward_one(sample, device)
        h_d = self.deviation.forward_one(sample, device)
        g = torch.sigmoid(self.gate(torch.cat([h_f, h_d], dim=-1)))
        return g * h_f + (1.0 - g) * h_d

    def forward(self, samples: List[Dict[str, Any]], device: torch.device) -> torch.Tensor:
        return torch.stack([self.forward_one(sample, device) for sample in samples], dim=0)


class TRDMUModel(nn.Module):
    def __init__(self, cfg: Dict[str, Any], meta: Dict[str, Any]):
        super().__init__()
        hidden = int(cfg["model"]["hidden_dim"])
        dropout = float(cfg["model"]["dropout"])
        k = int(cfg["model"]["perturbation_bases"])
        num_highways = len(meta["highway_vocab"])
        role_vocab_size = int(meta["role_vocab_size"])
        static_dim = int(meta["static_dim"])
        self.hidden = hidden
        self.k = k
        self.encoder_h = RoadEncoder(num_highways, cfg, role_vocab_size, static_dim)
        self.encoder_z = RoadEncoder(num_highways, cfg, role_vocab_size, static_dim)
        self.basis = nn.Parameter(torch.randn(k, hidden) * 0.02)
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.discriminator = MLP(hidden, hidden, hidden, dropout)
        self.mi_estimator = MLP(hidden, hidden, hidden, dropout)
        self.route_clo = MLP(hidden * 2, hidden, 1, dropout)
        self.route_irr = MLP(hidden * 2, hidden, 1, dropout)
        self.expert_rel = MLP(hidden, hidden, hidden, dropout)
        self.expert_irr = MLP(hidden, hidden, hidden, dropout)
        self.closure_head = MLP(hidden * 2, hidden, 1, dropout)
        self.congestion_head = MLP(hidden * 2, hidden, 1, dropout)

    def perturbation_distribution(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        q = self.query(h)
        keys = self.key(self.basis)
        alpha = torch.softmax((q @ keys.t()) / math.sqrt(keys.size(-1)), dim=-1)
        c = alpha @ self.basis
        return alpha, c

    def route(self, z: torch.Tensor, alpha: torch.Tensor, c: torch.Tensor) -> Dict[str, torch.Tensor]:
        full = torch.cat([z, c], dim=-1)
        base_clo = torch.sigmoid(self.route_clo(full))
        base_irr = torch.sigmoid(self.route_irr(full))
        tau_clo = []
        tau_irr = []
        for idx in range(self.k):
            c_minus = c - alpha[:, idx : idx + 1] * self.basis[idx].unsqueeze(0)
            cf = torch.cat([z, c_minus], dim=-1)
            tau_clo.append((base_clo - torch.sigmoid(self.route_clo(cf))).squeeze(-1))
            tau_irr.append((base_irr - torch.sigmoid(self.route_irr(cf))).squeeze(-1))
        tau_clo_t = torch.stack(tau_clo, dim=-1)
        tau_irr_t = torch.stack(tau_irr, dim=-1)
        s_rel = F.relu(tau_clo_t)
        s_irr = F.relu(tau_irr_t) + F.relu(-tau_clo_t)
        pi = torch.softmax(torch.stack([s_rel, s_irr], dim=-1), dim=-1)
        pi_rel = pi[..., 0]
        pi_irr = pi[..., 1]
        rel_weight = alpha * pi_rel
        irr_weight = alpha * pi_irr
        c_rel = self.expert_rel(rel_weight @ self.basis)
        c_irr = self.expert_irr(irr_weight @ self.basis)
        return {
            "pi_rel": pi_rel,
            "pi_irr": pi_irr,
            "rel_weight_sum": rel_weight.sum(dim=-1),
            "irr_weight_sum": irr_weight.sum(dim=-1),
            "c_rel": c_rel,
            "c_irr": c_irr,
        }

    def forward(self, samples: List[Dict[str, Any]], device: torch.device, lambda_grl: float = 1.0) -> Dict[str, torch.Tensor]:
        h = self.encoder_h(samples, device)
        z = self.encoder_z(samples, device)
        alpha, c = self.perturbation_distribution(h)
        adv_pred = self.discriminator(grl(z, lambda_grl))
        mi_mean = self.mi_estimator(c)
        perm = torch.randperm(z.size(0), device=z.device)
        mi_joint = -F.mse_loss(z, mi_mean, reduction="none").mean(dim=-1)
        mi_product = -F.mse_loss(z[perm], mi_mean, reduction="none").mean(dim=-1)
        # The variational difference can become negative and dominate BCE losses.
        # Softplus keeps the MI penalty non-negative while preserving the same ordering.
        mi_loss = F.softplus(mi_joint - mi_product).mean()
        routed = self.route(z, alpha, c)
        closure_logit = self.closure_head(torch.cat([z, routed["c_rel"]], dim=-1)).squeeze(-1)
        congestion_logit = self.congestion_head(torch.cat([z, routed["c_irr"]], dim=-1)).squeeze(-1)
        return {
            "h": h,
            "z": z,
            "alpha": alpha,
            "c": c,
            "c_norm": torch.linalg.norm(c, dim=-1),
            "adv_pred": adv_pred,
            "adv_loss": F.mse_loss(adv_pred, c.detach()),
            "mi_loss": mi_loss,
            "closure_logit": closure_logit,
            "congestion_logit": congestion_logit,
            "closure_prob": torch.sigmoid(closure_logit),
            "congestion_prob": torch.sigmoid(congestion_logit),
            **routed,
        }


def compute_loss(
    out: Dict[str, torch.Tensor],
    y_closure: torch.Tensor,
    y_congestion: torch.Tensor,
    closure_pos_weight: torch.Tensor,
    congestion_pos_weight: torch.Tensor,
    lambda_con: float,
    lambda_mi: float,
) -> Dict[str, torch.Tensor]:
    y_closure = y_closure.to(out["closure_logit"].device)
    y_congestion = y_congestion.to(out["congestion_logit"].device)
    loss_clo = F.binary_cross_entropy_with_logits(
        out["closure_logit"], y_closure, pos_weight=closure_pos_weight.to(out["closure_logit"].device)
    )
    loss_con = F.binary_cross_entropy_with_logits(
        out["congestion_logit"],
        y_congestion,
        pos_weight=congestion_pos_weight.to(out["congestion_logit"].device),
    )
    loss = loss_clo + float(lambda_con) * loss_con + out["adv_loss"] + float(lambda_mi) * out["mi_loss"]
    return {
        "loss": loss,
        "loss_closure": loss_clo.detach(),
        "loss_congestion": loss_con.detach(),
        "loss_adv": out["adv_loss"].detach(),
        "loss_mi": out["mi_loss"].detach(),
    }
