"""Neural vector field whose parameter tangents form the DTB basis."""

from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F


class TangentMLP(nn.Module):
    """A smooth map ``f_theta: R^d -> R^d`` used only for its tangents.

    ``tanh`` is the default because the score equation requires second
    spatial derivatives of the tangent velocity.  ReLU should not be used:
    its classical second derivative is zero almost everywhere and undefined
    at the kink.
    """

    def __init__(
        self,
        dim: int,
        width: int = 32,
        depth: int = 2,
        activation: str = "tanh",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if dim < 1:
            raise ValueError("dim must be positive")
        if width < 1 or depth < 1:
            raise ValueError("width and depth must be positive")

        activations = {
            "tanh": nn.Tanh,
            "gelu": nn.GELU,
            "silu": nn.SiLU,
        }
        if activation not in activations:
            raise ValueError(f"activation must be one of {tuple(activations)}")

        layers: List[nn.Module] = []
        input_dim = dim
        for _ in range(depth):
            linear = nn.Linear(input_dim, width, dtype=dtype)
            nn.init.xavier_uniform_(linear.weight)
            nn.init.zeros_(linear.bias)
            layers.extend((linear, activations[activation]()))
            input_dim = width
        output = nn.Linear(input_dim, dim, dtype=dtype)
        nn.init.xavier_uniform_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def tanh_diagnostics(self, x: torch.Tensor) -> list[dict[str, float | int]]:
        """Measure how often hidden tanh units are in their flat region."""

        if not any(isinstance(layer, nn.Tanh) for layer in self.net):
            return []
        values = x
        diagnostics: list[dict[str, float | int]] = []
        layer_index = 0
        for layer in self.net:
            if isinstance(layer, nn.Linear):
                preactivation = layer(values)
                values = preactivation
            elif isinstance(layer, nn.Tanh):
                derivative = 1.0 - torch.tanh(preactivation).square()
                diagnostics.append(
                    _tanh_summary(preactivation, derivative, layer_index)
                )
                layer_index += 1
                values = layer(values)
            else:
                values = layer(values)
        return diagnostics


class RandomFeatureBlock(nn.Module):
    """One MMNN block ``A activation(Wx+b)+c``.

    ``W`` and ``b`` are registered as non-trainable parameters so they are
    present in the model state but excluded from the DTB tangent-coordinate
    pool.  Only the component-combination matrix ``A`` and bias ``c`` have
    ``requires_grad=True``.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        width: int,
        activation: str,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()
        activations = {
            "tanh": torch.tanh,
            "gelu": F.gelu,
            "silu": F.silu,
        }
        if activation not in activations:
            raise ValueError(f"activation must be one of {tuple(activations)}")

        feature_weight = torch.empty(width, in_features, dtype=dtype)
        nn.init.xavier_uniform_(feature_weight)
        self.feature_weight = nn.Parameter(feature_weight, requires_grad=False)
        self.feature_bias = nn.Parameter(
            torch.zeros(width, dtype=dtype), requires_grad=False
        )

        self.combination_weight = nn.Parameter(
            torch.empty(out_features, width, dtype=dtype)
        )
        nn.init.xavier_uniform_(self.combination_weight)
        self.component_bias = nn.Parameter(torch.zeros(out_features, dtype=dtype))
        self.activation_name = activation
        self.activation = activations[activation]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        features = self.activation(
            F.linear(x, self.feature_weight, self.feature_bias)
        )
        return F.linear(features, self.combination_weight, self.component_bias)


class TangentMMNN(nn.Module):
    """Multicomponent/multilayer tangent model with frozen random features.

    A depth-``L`` model composes ``L`` shallow random-feature blocks.  The
    intermediate block outputs have dimension ``rank`` and the final block
    maps to the requested vector-field dimension.  This implements the MMNN
    separation relevant to the experiment: frozen ``W,b`` and tangent
    coordinates drawn only from trainable ``A,c``.
    """

    def __init__(
        self,
        dim: int,
        width: int = 32,
        rank: int = 8,
        depth: int = 2,
        activation: str = "tanh",
        dtype: torch.dtype = torch.float64,
    ) -> None:
        super().__init__()
        if dim < 1 or width < 1 or rank < 1 or depth < 1:
            raise ValueError("dim, width, rank, and depth must be positive")

        blocks: list[RandomFeatureBlock] = []
        input_dim = dim
        for index in range(depth):
            output_dim = dim if index == depth - 1 else rank
            blocks.append(
                RandomFeatureBlock(
                    input_dim,
                    output_dim,
                    width,
                    activation,
                    dtype,
                )
            )
            input_dim = output_dim
        self.blocks = nn.ModuleList(blocks)
        self.activation_name = activation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        values = x
        for block in self.blocks:
            values = block(values)
        return values

    def tanh_diagnostics(self, x: torch.Tensor) -> list[dict[str, float | int]]:
        """Measure frozen-feature saturation in every MMNN block."""

        if self.activation_name != "tanh":
            return []
        values = x
        diagnostics: list[dict[str, float | int]] = []
        for index, block in enumerate(self.blocks):
            preactivation = F.linear(
                values, block.feature_weight, block.feature_bias
            )
            activated = torch.tanh(preactivation)
            derivative = 1.0 - activated.square()
            diagnostics.append(_tanh_summary(preactivation, derivative, index))
            values = F.linear(
                activated, block.combination_weight, block.component_bias
            )
        return diagnostics


def _tanh_summary(
    preactivation: torch.Tensor,
    derivative: torch.Tensor,
    layer_index: int,
) -> dict[str, float | int]:
    preactivation = preactivation.detach()
    derivative = derivative.detach()
    return {
        "layer": layer_index + 1,
        "unit_values": preactivation.numel(),
        "mean_abs_preactivation": float(preactivation.abs().mean()),
        "fraction_abs_preactivation_gt_2": float(
            (preactivation.abs() > 2.0).to(torch.float32).mean()
        ),
        "mean_tanh_derivative": float(derivative.mean()),
        "fraction_tanh_derivative_lt_0.05": float(
            (derivative < 0.05).to(torch.float32).mean()
        ),
    }


def count_trainable(model: nn.Module) -> int:
    """Number ``M`` of trainable scalar parameters."""

    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
