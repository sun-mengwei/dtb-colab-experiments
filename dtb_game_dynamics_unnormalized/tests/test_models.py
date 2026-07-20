import torch

from game_dtb.models import TangentMLP, TangentMMNN, TangentNODE, count_trainable
from game_dtb.parameters import flat_params


def test_mmnn_freezes_random_features_and_selects_only_a_c() -> None:
    model = TangentMMNN(
        dim=3, width=5, rank=2, depth=2, activation="tanh", dtype=torch.float64
    )
    trainable = {
        name for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    frozen = {
        name for name, parameter in model.named_parameters()
        if not parameter.requires_grad
    }
    assert trainable == {
        "blocks.0.combination_weight",
        "blocks.0.component_bias",
        "blocks.1.combination_weight",
        "blocks.1.component_bias",
    }
    assert frozen == {
        "blocks.0.feature_weight",
        "blocks.0.feature_bias",
        "blocks.1.feature_weight",
        "blocks.1.feature_bias",
    }

    flat, structure = flat_params(model)
    assert [name for name, _ in structure] == [
        "blocks.0.combination_weight",
        "blocks.0.component_bias",
        "blocks.1.combination_weight",
        "blocks.1.component_bias",
    ]
    assert flat.numel() == (2 * 5 + 2) + (3 * 5 + 3)


def test_mmnn_forward_and_tanh_diagnostics_are_finite() -> None:
    torch.manual_seed(17)
    model = TangentMMNN(
        dim=3, width=6, rank=4, depth=2, activation="tanh", dtype=torch.float64
    )
    x = torch.randn(7, 3, dtype=torch.float64)
    output = model(x)
    diagnostics = model.tanh_diagnostics(x)
    assert output.shape == (7, 3)
    assert bool(torch.isfinite(output).all())
    assert len(diagnostics) == 2
    for layer in diagnostics:
        assert 0.0 <= layer["fraction_tanh_derivative_lt_0.05"] <= 1.0
        assert 0.0 <= layer["mean_tanh_derivative"] <= 1.0


def test_node_is_a_differentiable_terminal_flow_with_matched_parameter_count() -> None:
    torch.manual_seed(23)
    mlp = TangentMLP(dim=2, width=5, depth=2, dtype=torch.float64)
    node = TangentNODE(
        dim=2, width=5, depth=2, inner_steps=4, dtype=torch.float64
    )
    assert count_trainable(node) == count_trainable(mlp)

    x = torch.randn(3, 2, dtype=torch.float64)
    output = node(x)
    input_jacobian = torch.func.jacrev(lambda value: node(value))(x[0])
    assert output.shape == x.shape
    assert bool(torch.isfinite(output).all())
    assert bool(torch.isfinite(input_jacobian).all())


def test_zero_node_vector_field_returns_the_input() -> None:
    node = TangentNODE(dim=2, width=4, depth=1, inner_steps=2)
    with torch.no_grad():
        for parameter in node.parameters():
            parameter.zero_()
    x = torch.randn(6, 2, dtype=torch.float64)
    assert torch.equal(node(x), x)
