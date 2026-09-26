"""Typed view of the flat Drive observation vector used by the probe."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from typing import Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from .contracts import DecoderOutput


@dataclass(frozen=True)
class ObservationTargets:
    """Continuous values, integer labels, and masks unpacked from ``[B,D]``."""

    continuous: Mapping[str, Tensor]
    categorical: Mapping[str, Tensor]
    presence: Mapping[str, Tensor]
    continuous_valid: Mapping[str, Tensor]
    categorical_valid: Mapping[str, Tensor]
    valid_counts: Mapping[str, Tensor]


class ObservationSchema:
    """Offsets, field types, and count masks for the audited C observation."""

    groups = ("partners", "lanes", "boundaries", "traffic_controls")
    count_order = ("lanes", "boundaries", "partners", "traffic_controls")
    fixed_groups = ("ego", "context")
    categorical_classes = {"traffic_controls.type": 4, "traffic_controls.state": 5}

    _required_layout = (
        "observation_dim",
        "ego_features",
        "partner_features",
        "lane_features",
        "boundary_features",
        "traffic_control_features",
        "num_reward_coefs",
        "goal_dim",
        "obs_slots_partners_n",
        "obs_slots_lane_kept",
        "obs_slots_boundary_kept",
        "obs_slots_traffic_controls_n",
        "obs_valid_count_features",
    )

    def __init__(self, layout: Mapping[str, int]) -> None:
        if not isinstance(layout, Mapping):
            raise TypeError("observation_layout must be a mapping")
        missing = [key for key in self._required_layout if key not in layout]
        if missing:
            raise ValueError("observation_layout is missing keys: " + ", ".join(missing))
        self.layout = dict(layout)
        for key in self._required_layout:
            value = self.layout[key]
            if isinstance(value, bool) or not isinstance(value, int):
                raise ValueError(f"observation_layout.{key} must be an integer")

        expected_widths = {
            "ego_features": 10,
            "partner_features": 9,
            "lane_features": 9,
            "boundary_features": 9,
            "traffic_control_features": 7,
            "obs_valid_count_features": 4,
        }
        for key, expected in expected_widths.items():
            if self.layout[key] != expected:
                raise ValueError(f"observation_layout.{key} must equal {expected}")
        if self.layout["num_reward_coefs"] < 0 or self.layout["goal_dim"] < 0:
            raise ValueError("reward and goal dimensions must be non-negative")
        if self.layout["goal_dim"] % 3:
            raise ValueError("observation_layout.goal_dim must contain complete xyz goals")
        for key in (
            "obs_slots_partners_n",
            "obs_slots_lane_kept",
            "obs_slots_boundary_kept",
            "obs_slots_traffic_controls_n",
        ):
            if self.layout[key] <= 0:
                raise ValueError(f"observation_layout.{key} must be positive")

        self.context_dim = self.layout["num_reward_coefs"] + self.layout["goal_dim"]
        if self.context_dim <= 0:
            raise ValueError("observation context must include reward or goal features")
        if "context_dim" in self.layout and self.layout["context_dim"] != self.context_dim:
            raise ValueError("observation_layout.context_dim disagrees with reward and goal widths")
        if "goal_features" in self.layout and self.layout["goal_features"] != 3:
            raise ValueError("observation_layout.goal_features must equal 3")
        self.capacities = {
            "partners": self.layout["obs_slots_partners_n"],
            "lanes": self.layout["obs_slots_lane_kept"],
            "boundaries": self.layout["obs_slots_boundary_kept"],
            "traffic_controls": self.layout["obs_slots_traffic_controls_n"],
        }
        self.continuous_widths = {
            "ego": self.layout["ego_features"],
            "context": self.context_dim,
            "partners": self.layout["partner_features"],
            "lanes": self.layout["lane_features"],
            "boundaries": self.layout["boundary_features"],
            "traffic_controls": self.layout["traffic_control_features"] - 2,
        }

        offsets: dict[str, slice] = {}
        cursor = 0

        def add(name: str, width: int) -> slice:
            nonlocal cursor
            result = slice(cursor, cursor + width)
            offsets[name] = result
            cursor += width
            return result

        add("ego", self.continuous_widths["ego"])
        reward_start = cursor
        add("reward_coefs", self.layout["num_reward_coefs"])
        add("goals", self.layout["goal_dim"])
        offsets["context"] = slice(reward_start, cursor)
        for group in self.groups:
            full_width = 7 if group == "traffic_controls" else self.continuous_widths[group]
            add(group, self.capacities[group] * full_width)
        offsets["counts"] = slice(cursor, cursor + 4)
        count_cursor = cursor
        for group in self.count_order:
            offsets[f"{group}_count"] = slice(count_cursor, count_cursor + 1)
            count_cursor += 1
        cursor += 4
        if cursor != self.layout["observation_dim"]:
            raise ValueError(
                "observation_layout.observation_dim does not match its declared field widths: "
                f"expected {cursor}, got {self.layout['observation_dim']}"
            )
        self.offsets = offsets
        self.observation_dim = self.layout["observation_dim"]

    def unpack(self, observations: Tensor) -> ObservationTargets:
        """Split ``[B,D]`` observations, validating count and category tails."""

        if not isinstance(observations, Tensor) or not observations.is_floating_point():
            raise TypeError("observations must be a floating-point torch.Tensor")
        if observations.ndim != 2 or observations.shape[1] != self.layout["observation_dim"]:
            raise ValueError(
                "observations must have shape [B, observation_dim], got "
                f"{tuple(observations.shape)}"
            )
        if not torch.isfinite(observations).all():
            raise ValueError("observations contain non-finite values")

        batch_size = observations.shape[0]
        count_values = observations[:, self.offsets["counts"]]
        rounded_counts = count_values.round()
        if not torch.equal(count_values, rounded_counts):
            raise ValueError("observation valid counts must be integers")
        valid_counts: dict[str, Tensor] = {}
        for index, group in enumerate(self.count_order):
            counts = rounded_counts[:, index].to(torch.long)
            if torch.any(counts < 0) or torch.any(counts > self.capacities[group]):
                raise ValueError(f"observation {group} count is outside [0, capacity]")
            valid_counts[group] = counts

        continuous: dict[str, Tensor] = {
            "ego": observations[:, self.offsets["ego"]],
            "context": observations[:, self.offsets["context"]],
        }
        continuous_valid: dict[str, Tensor] = {
            key: torch.ones_like(value, dtype=torch.bool) for key, value in continuous.items()
        }
        presence: dict[str, Tensor] = {}
        for group in self.groups:
            group_values = observations[:, self.offsets[group]].reshape(
                batch_size, self.capacities[group], -1
            )
            row_valid = (
                torch.arange(self.capacities[group], device=observations.device)[None, :]
                < valid_counts[group][:, None]
            )
            presence[group] = row_valid
            if group == "traffic_controls":
                continuous[group] = group_values[:, :, :5]
            else:
                continuous[group] = group_values
            continuous_valid[group] = row_valid[:, :, None].expand_as(continuous[group])

        traffic_values = observations[:, self.offsets["traffic_controls"]].reshape(
            batch_size, self.capacities["traffic_controls"], 7
        )
        categorical: dict[str, Tensor] = {}
        categorical_valid: dict[str, Tensor] = {}
        for key, feature_index, class_count in (
            ("traffic_controls.type", 5, 4),
            ("traffic_controls.state", 6, 5),
        ):
            raw_values = traffic_values[:, :, feature_index]
            if not torch.equal(raw_values, raw_values.round()):
                raise ValueError(f"observation categorical field {key} must contain integer IDs")
            if torch.any(raw_values < 0) or torch.any(raw_values >= class_count):
                raise ValueError(f"observation categorical field {key} is outside [0, {class_count})")
            categorical[key] = raw_values.to(torch.long)
            categorical_valid[key] = presence["traffic_controls"]

        return ObservationTargets(
            continuous=continuous,
            categorical=categorical,
            presence=presence,
            continuous_valid=continuous_valid,
            categorical_valid=categorical_valid,
            valid_counts=valid_counts,
        )

    def as_prediction(self, observations: Tensor) -> DecoderOutput:
        """Represent exact observations as decoder outputs for baseline scoring."""

        targets = self.unpack(observations)
        categorical_logits = {
            key: F.one_hot(labels, num_classes=self.categorical_classes[key]).to(observations.dtype)
            .mul(40.0)
            .sub(20.0)
            for key, labels in targets.categorical.items()
        }
        presence_logits = {
            group: torch.where(mask, 20.0, -20.0).to(observations.dtype)
            for group, mask in targets.presence.items()
        }
        return DecoderOutput(targets.continuous, categorical_logits, presence_logits)

    def pack(self, prediction: DecoderOutput, *, presence_threshold: float = 0.5) -> Tensor:
        """Flatten typed predictions to ``[B,D]`` and derive counts from presence."""

        if not isfinite(float(presence_threshold)) or not 0.0 < float(presence_threshold) < 1.0:
            raise ValueError("presence_threshold must be finite and between 0 and 1")
        continuous_keys = set(self.continuous_widths)
        categorical_keys = set(self.categorical_classes)
        group_keys = set(self.groups)
        if set(prediction.continuous) != continuous_keys:
            raise ValueError("prediction continuous fields do not match the observation schema")
        if set(prediction.categorical_logits) != categorical_keys:
            raise ValueError("prediction categorical fields do not match the observation schema")
        if set(prediction.presence_logits) != group_keys:
            raise ValueError("prediction presence fields do not match the observation schema")

        reference = prediction.continuous["ego"]
        if reference.ndim != 2 or reference.shape[1] != self.continuous_widths["ego"]:
            raise ValueError("prediction ego must have shape [B, ego_features]")
        batch_size = reference.shape[0]
        output = reference.new_zeros((batch_size, self.layout["observation_dim"]))
        compacted_slots: dict[str, tuple[Tensor, Tensor, Tensor]] = {}
        counts_by_group: dict[str, Tensor] = {}
        for group in self.groups:
            logits = prediction.presence_logits[group]
            expected_shape = (batch_size, self.capacities[group])
            if tuple(logits.shape) != expected_shape:
                raise ValueError(f"prediction presence {group} must have shape {expected_shape}")
            if (
                logits.device != reference.device
                or logits.dtype != reference.dtype
                or not logits.is_floating_point()
                or not torch.isfinite(logits).all()
            ):
                raise ValueError(f"prediction presence {group} must be finite and match the output dtype/device")
            retained = torch.sigmoid(logits) >= float(presence_threshold)
            batch_indices, source_slots = retained.nonzero(as_tuple=True)
            destination_slots = retained.to(torch.long).cumsum(dim=1)[batch_indices, source_slots] - 1
            compacted_slots[group] = (batch_indices, source_slots, destination_slots)
            counts_by_group[group] = retained.sum(dim=1).to(reference.dtype)

        for key, value in prediction.continuous.items():
            width = self.continuous_widths[key]
            expected_shape = (batch_size, width) if key in self.fixed_groups else (
                batch_size, self.capacities[key], width
            )
            if tuple(value.shape) != expected_shape:
                raise ValueError(f"prediction continuous {key} must have shape {expected_shape}")
            if value.device != reference.device or value.dtype != reference.dtype:
                raise ValueError("all prediction fields must share a device and dtype")
            if not torch.isfinite(value).all():
                raise ValueError(f"prediction continuous {key} contains non-finite values")
            if key in self.fixed_groups:
                output[:, self.offsets[key]] = value
                continue
            flattened = output[:, self.offsets[key]].reshape(batch_size, self.capacities[key], -1)
            batch_indices, source_slots, destination_slots = compacted_slots[key]
            flattened[batch_indices, destination_slots, :width] = value[batch_indices, source_slots]

        for key, class_count in self.categorical_classes.items():
            logits = prediction.categorical_logits[key]
            group = key.split(".", 1)[0]
            expected_shape = (batch_size, self.capacities[group], class_count)
            if tuple(logits.shape) != expected_shape:
                raise ValueError(f"prediction categorical {key} must have shape {expected_shape}")
            if (
                logits.device != reference.device
                or logits.dtype != reference.dtype
                or not logits.is_floating_point()
                or not torch.isfinite(logits).all()
            ):
                raise ValueError(f"prediction categorical {key} must be finite and match the output dtype/device")
            feature_index = 5 if key.endswith(".type") else 6
            traffic_rows = output[:, self.offsets[group]].reshape(batch_size, self.capacities[group], 7)
            batch_indices, source_slots, destination_slots = compacted_slots[group]
            predicted_labels = logits.argmax(dim=-1)
            traffic_rows[batch_indices, destination_slots, feature_index] = predicted_labels[
                batch_indices, source_slots
            ].to(reference.dtype)
        for index, group in enumerate(self.count_order):
            output[:, self.offsets["counts"].start + index] = counts_by_group[group]
        return output
