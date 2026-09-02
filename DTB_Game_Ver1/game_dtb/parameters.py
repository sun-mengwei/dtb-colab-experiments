"""Flat-parameter helpers reused from the original DTB implementation.

The design is adapted from ``DTB_rep/dtb.py`` in
https://github.com/sun-mengwei/dtb-colab-experiments.  Keeping these helpers
in this new package makes the game code standalone and leaves the original
Deep Tangent Bundle files unchanged.
"""

from __future__ import annotations

from typing import Dict, List, Sequence, Tuple

import torch
import torch.nn as nn
from torch.func import functional_call

ParameterStructure = List[Tuple[str, Tuple[int, ...]]]


def flat_params(model: nn.Module) -> Tuple[torch.Tensor, ParameterStructure]:
    """Return a detached flat vector and the shapes of trainable parameters.

    This follows the original DTB code's ``flat_params`` representation.  A
    separate flat vector is useful because ``torch.func.jacrev`` can then
    differentiate only with respect to selected scalar coordinates.
    """

    parts: List[torch.Tensor] = []
    structure: ParameterStructure = []
    for name, parameter in model.named_parameters():
        if parameter.requires_grad:
            parts.append(parameter.detach().reshape(-1).clone())
            structure.append((name, tuple(parameter.shape)))
    if not parts:
        raise ValueError("the tangent model must have trainable parameters")
    return torch.cat(parts), structure


def unflatten(
    flat: torch.Tensor, structure: Sequence[Tuple[str, Tuple[int, ...]]]
) -> Dict[str, torch.Tensor]:
    """Reconstruct the named parameter dictionary from a flat vector."""

    result: Dict[str, torch.Tensor] = {}
    offset = 0
    for name, shape in structure:
        count = 1
        for size in shape:
            count *= size
        result[name] = flat[offset : offset + count].reshape(shape)
        offset += count
    if offset != flat.numel():
        raise ValueError("flat parameter size does not match model structure")
    return result


def functional_model_call(
    model: nn.Module,
    flat: torch.Tensor,
    structure: ParameterStructure,
    x: torch.Tensor,
) -> torch.Tensor:
    """Evaluate ``model(x)`` at explicit flat parameter values.

    Frozen parameters and buffers stay fixed at their live values.  Trainable
    parameters are replaced by tensors reconstructed from ``flat``.
    """

    parameters_and_buffers = dict(model.named_parameters())
    parameters_and_buffers.update(dict(model.named_buffers()))
    parameters_and_buffers.update(unflatten(flat, structure))
    return functional_call(model, parameters_and_buffers, (x,))
