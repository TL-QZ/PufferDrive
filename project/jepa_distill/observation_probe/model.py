"""Frozen Condition B loading and the learned observation decoder."""
from __future__ import annotations

from collections.abc import Sequence
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING, Mapping

import torch
from torch import Tensor, nn

from .contracts import DecoderOutput
from .schema import ObservationSchema

if TYPE_CHECKING:
    from ..model import ConditionBModel


def load_frozen_jepa(checkpoint_path: str | Path, *, device: str) -> ConditionBModel:
    """Restore one Condition B checkpoint, set eval and freeze all parameters.

    Return the existing model with context/target encoders, predictor and action
    decoder restored exactly. Do not load a driving teacher or initialize an
    environment. This same frozen model serves every collection. Loading for
    training is unimplemented; CPU metadata inspection belongs to preflight.
    """
    from ..model import ConditionBModel

    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Condition B checkpoint does not exist: {checkpoint_path}")
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(payload, Mapping) or payload.get("format") != "condition_b_v1":
        raise ValueError("Expected a Condition B checkpoint with format condition_b_v1")
    model_config = payload.get("model_config")
    model_state = payload.get("model_state")
    if not isinstance(model_config, Mapping):
        raise ValueError("Condition B checkpoint is missing model_config metadata")
    if not isinstance(model_state, Mapping):
        raise ValueError("Condition B checkpoint is missing model_state")

    # Rebuild only from exported metadata. ConditionBModel creates the encoder
    # modules directly and does not construct a driving teacher or simulator.
    model = ConditionBModel(model_config)
    model.load_state_dict(model_state, strict=True)
    model.to(device)
    model.requires_grad_(False)
    model.eval()
    model.probe_checkpoint_metadata = {
        "format": payload["format"],
        "step": payload.get("step"),
        "model_config": deepcopy(dict(model_config)),
    }
    return model


class ObservationDecoder(nn.Module):
    """Read normalized latents through one MLP into typed observation heads."""

    def __init__(
        self,
        latent_dim: int,
        observation_layout: Mapping[str, int],
        *,
        hidden_sizes: Sequence[int] = (1024, 1024),
    ) -> None:
        super().__init__()
        if isinstance(latent_dim, bool) or not isinstance(latent_dim, int) or latent_dim <= 0:
            raise ValueError("latent_dim must be a positive integer")
        if not isinstance(hidden_sizes, Sequence) or isinstance(hidden_sizes, (str, bytes)) or not hidden_sizes:
            raise ValueError("hidden_sizes must be a non-empty sequence of positive integers")
        hidden_widths = []
        for width in hidden_sizes:
            if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
                raise ValueError("hidden_sizes entries must be positive integers")
            hidden_widths.append(width)

        self.latent_dim = latent_dim
        self.schema = ObservationSchema(observation_layout)
        self.observation_layout = dict(observation_layout)
        self.observation_dim = self.schema.layout["observation_dim"]
        layers: list[nn.Module] = []
        current_width = latent_dim
        for width in hidden_widths:
            layers.extend((nn.Linear(current_width, width), nn.ReLU()))
            current_width = width
        self.trunk = nn.Sequential(*layers)

        self.continuous_heads = nn.ModuleDict(
            {
                name: nn.Linear(
                    current_width,
                    width if name in self.schema.fixed_groups
                    else width * self.schema.capacities[name],
                )
                for name, width in self.schema.continuous_widths.items()
            }
        )
        self._categorical_head_names = {
            field: field.replace(".", "__") for field in self.schema.categorical_classes
        }
        self.categorical_heads = nn.ModuleDict(
            {
                self._categorical_head_names[field]: nn.Linear(
                    current_width,
                    classes * self.schema.capacities[field.split(".", 1)[0]],
                )
                for field, classes in self.schema.categorical_classes.items()
            }
        )
        self.presence_heads = nn.ModuleDict(
            {
                group: nn.Linear(current_width, self.schema.capacities[group])
                for group in self.schema.groups
            }
        )

    def forward(self, normalized_latents: Tensor) -> DecoderOutput:
        """Decode unit-normalized [B,Z] inputs; no action sequence is supplied.

        Training inputs come from real future target latents. Evaluation also
        supplies predicted endpoint latents, normalized by the same rule.
        """
        if not isinstance(normalized_latents, Tensor) or not normalized_latents.is_floating_point():
            raise TypeError("normalized_latents must be a floating-point torch.Tensor")
        if normalized_latents.ndim != 2 or normalized_latents.shape[1] != self.latent_dim:
            raise ValueError(
                f"normalized_latents must have shape [B, {self.latent_dim}], "
                f"got {tuple(normalized_latents.shape)}"
            )
        if not torch.isfinite(normalized_latents).all():
            raise ValueError("normalized_latents contain non-finite values")
        features = self.trunk(normalized_latents)
        batch_size = features.shape[0]
        continuous: dict[str, Tensor] = {}
        for name, head in self.continuous_heads.items():
            values = head(features)
            continuous[name] = (
                values if name in self.schema.fixed_groups
                else values.reshape(
                    batch_size,
                    self.schema.capacities[name],
                    self.schema.continuous_widths[name],
                )
            )
        categorical_logits = {
            field: self.categorical_heads[self._categorical_head_names[field]](features).reshape(
                batch_size, self.schema.capacities[field.split(".", 1)[0]], classes
            )
            for field, classes in self.schema.categorical_classes.items()
        }
        presence_logits = {
            group: head(features) for group, head in self.presence_heads.items()
        }
        return DecoderOutput(continuous, categorical_logits, presence_logits)
