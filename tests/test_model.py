from __future__ import annotations

import torch

from tests.helpers import tiny_config, tiny_meta, tiny_sample
from crcdm.models import CRCDMModel, compute_loss, grl


def test_gradient_reversal() -> None:
    x = torch.tensor([2.0], requires_grad=True)
    (grl(x, 0.25) * 4.0).sum().backward()
    assert torch.allclose(x.grad, torch.tensor([-1.0]))


def test_forward_routes_and_both_semantic_encoders_receive_gradients() -> None:
    torch.manual_seed(7)
    model = CRCDMModel(tiny_config(), tiny_meta())
    samples = [tiny_sample("a", 0, -1.0), tiny_sample("b", 1, 1.0)]
    output = model(samples, torch.device("cpu"), lambda_grl=0.5)
    assert output["closure_logit"].shape == (2,)
    assert output["congestion_logit"].shape == (2,)
    assert output["alpha"].shape == (2, 3)
    assert output["c"].shape == (2, 8)
    assert output["causal_gate"].shape == (2, 3)
    assert torch.all(output["causal_gate"] >= 0.0)
    assert torch.all(output["causal_gate"] <= 1.0)
    assert torch.allclose(output["alpha"].sum(dim=-1), torch.ones(2), atol=1e-6)
    losses = compute_loss(
        output,
        torch.tensor([0.0, 1.0]),
        torch.tensor([1.0, 0.0]),
        torch.tensor(1.0),
        torch.tensor(1.0),
        lambda_con=0.5,
        lambda_mi=0.1,
        lambda_dis=1.0,
    )
    losses["loss"].backward()
    assert model.encoder_h.traffic.node_proj[0].weight.grad is not None
    assert model.encoder_h.deviation.node_proj[0].weight.grad is not None
    assert model.encoder_z.traffic.node_proj[0].weight.grad is not None
    assert model.encoder_z.deviation.node_proj[0].weight.grad is not None
    assert model.basis.grad is not None


def test_conditional_estimator_and_tiny_batch_optimization() -> None:
    torch.manual_seed(11)
    model = CRCDMModel(tiny_config(), tiny_meta())
    samples = [tiny_sample("a", 0, -1.0), tiny_sample("b", 1, 1.0)]
    representation = model.representations(samples, torch.device("cpu"))
    estimator_loss = model.mi_estimator_loss(representation["z"], representation["c"])
    estimator_loss.backward()
    assert model.mi_estimator.mean.weight.grad is not None

    model.zero_grad(set_to_none=True)
    optimizer = torch.optim.Adam(model.main_parameters(), lr=0.02)
    labels = torch.tensor([0.0, 1.0])
    with torch.no_grad():
        initial = torch.nn.functional.binary_cross_entropy_with_logits(
            model(samples, torch.device("cpu"))["closure_logit"], labels
        )
    for _ in range(20):
        optimizer.zero_grad(set_to_none=True)
        output = model(samples, torch.device("cpu"))
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            output["closure_logit"], labels
        )
        loss.backward()
        optimizer.step()
    with torch.no_grad():
        final = torch.nn.functional.binary_cross_entropy_with_logits(
            model(samples, torch.device("cpu"))["closure_logit"], labels
        )
    assert final < initial
