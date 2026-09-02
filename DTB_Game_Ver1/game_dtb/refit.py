"""Source-matching periodic reset for the game-dynamics DTB solver.

This module follows ``DTB_rep/run_game_dtb.py`` without modifying it:

1. freeze the block parameters and selected parameter coordinates;
2. accumulate ``s = sum(alpha)`` for exactly ``L`` Euler steps;
3. define ``f_target = f_theta_block + h J_theta_block,S s``;
4. precompute that target on fresh samples from the reference distribution;
5. fit every trainable parameter with Adam and cosine learning-rate decay;
6. report RMSE on another fresh reference sample.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import torch
import torch.nn as nn

from .parameters import ParameterStructure, functional_model_call
from .projection import selected_parameter_jacobian

ReferenceSampler = Callable[[int, torch.Generator], torch.Tensor]
TargetFunction = Callable[[torch.Tensor], torch.Tensor]


@dataclass(frozen=True)
class RefitDiagnostics:
    """Training-set error before fitting and fresh-test error afterwards."""

    rmse_before: float
    rmse_after: float
    optimizer_steps: int
    train_samples: int
    test_samples: int


class AccumulatedTangentTarget:
    """Callable accumulated DTB state at the end of a tangent block.

    This is the vector-valued counterpart of ``CurrentSolution`` in
    ``DTB_rep/run_5d_ac.py`` and matches ``CurrentGameMap`` in the original
    repository's game driver.
    """

    def __init__(
        self,
        model: nn.Module,
        theta_block: torch.Tensor,
        structure: ParameterStructure,
        selected: torch.Tensor,
        accumulated_alpha: torch.Tensor,
        *,
        step_size: float,
        jacobian_chunk_size: int,
    ) -> None:
        self.model = model
        self.theta_block = theta_block
        self.structure = structure
        self.selected = selected
        self.accumulated_alpha = accumulated_alpha
        self.step_size = step_size
        self.jacobian_chunk_size = jacobian_chunk_size

    def __call__(self, samples: torch.Tensor) -> torch.Tensor:
        base_values = functional_model_call(
            self.model, self.theta_block, self.structure, samples
        )
        jacobians = selected_parameter_jacobian(
            self.theta_block,
            self.selected,
            samples,
            self.model,
            self.structure,
            chunk_size=self.jacobian_chunk_size,
        )
        increment = torch.einsum(
            "ndm,m->nd", jacobians, self.accumulated_alpha
        )
        return base_values + self.step_size * increment


def fit_model_to_target(
    model: nn.Module,
    target_fn: TargetFunction,
    reference_sampler: ReferenceSampler,
    *,
    n_samples: int,
    optimizer_steps: int,
    learning_rate: float,
    batch_size: int,
    test_samples: int,
    generator: torch.Generator,
) -> RefitDiagnostics:
    """Apply the original DTB reset optimizer to a precomputed snapshot.

    The minibatches use sampling with replacement, exactly as in
    ``fit_map_to_target``. Parameters with ``requires_grad=False`` remain
    frozen because Adam receives only trainable parameters.
    """

    trainable = [parameter for parameter in model.parameters() if parameter.requires_grad]
    if not trainable:
        raise ValueError("the tangent model has no trainable parameters")
    if min(n_samples, optimizer_steps, batch_size, test_samples) < 1:
        raise ValueError("refit sample counts, steps, and batch size must be positive")
    if learning_rate <= 0:
        raise ValueError("refit learning rate must be positive")

    # One-shot training set and precomputed target snapshot, as in the source.
    with torch.no_grad():
        z_train = reference_sampler(n_samples, generator)
        y_train = target_fn(z_train).detach()
        rmse_before = torch.mean((model(z_train) - y_train).square()).sqrt()

    optimizer = torch.optim.Adam(trainable, lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=optimizer_steps
    )
    loss_fn = nn.MSELoss()
    was_training = model.training
    model.train()
    for _ in range(optimizer_steps):
        indices = torch.randint(
            0,
            n_samples,
            (batch_size,),
            device=z_train.device,
            generator=generator,
        )
        prediction = model(z_train[indices])
        loss = loss_fn(prediction, y_train[indices])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

    # The source reports reset RMSE on a fresh reference-distribution batch.
    with torch.no_grad():
        z_test = reference_sampler(test_samples, generator)
        rmse_after = torch.mean(
            (model(z_test) - target_fn(z_test)).square()
        ).sqrt()
    model.train(was_training)
    return RefitDiagnostics(
        rmse_before=float(rmse_before),
        rmse_after=float(rmse_after),
        optimizer_steps=optimizer_steps,
        train_samples=n_samples,
        test_samples=test_samples,
    )
