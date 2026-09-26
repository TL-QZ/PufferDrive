"""Permutation-aware object matching and observation reconstruction losses."""
from __future__ import annotations

import math
from typing import Any, Mapping

import torch
from torch import Tensor
from torch.nn import functional as F

from .contracts import DecoderOutput, ObjectAssignment, ReconstructionLoss
from .schema import ObservationSchema


_MATCHING_COST_KEYS = {"continuous", "categorical", "presence"}
_LOSS_GROUPS = ("ego", "context", "partners", "lanes", "boundaries", "traffic_controls")


def _finite_weights(
    config: Mapping[str, Any], *, name: str, expected: set[str], strictly_positive: bool
) -> dict[str, float]:
    if not isinstance(config, Mapping):
        raise ValueError(f"{name} must be a mapping")
    unknown = set(config) - expected
    missing = expected - set(config)
    if unknown or missing:
        raise ValueError(
            f"{name} keys must be exactly {sorted(expected)}; "
            f"missing={sorted(missing)}, unknown={sorted(unknown)}"
        )
    values: dict[str, float] = {}
    for key, value in config.items():
        if isinstance(value, bool):
            raise ValueError(f"{name}.{key} must be a finite number")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{name}.{key} must be a finite number") from exc
        if not math.isfinite(numeric) or (numeric <= 0 if strictly_positive else numeric < 0):
            qualifier = "finite and positive" if strictly_positive else "finite and non-negative"
            raise ValueError(f"{name}.{key} must be {qualifier}")
        values[key] = numeric
    return values


def _validate_prediction(prediction: DecoderOutput, schema: ObservationSchema) -> tuple[int, Tensor]:
    if not isinstance(prediction, DecoderOutput):
        raise TypeError("prediction must be a DecoderOutput")
    expected_continuous = set(schema.continuous_widths)
    expected_categorical = set(schema.categorical_classes)
    expected_presence = set(schema.groups)
    if set(prediction.continuous) != expected_continuous:
        raise ValueError("prediction continuous fields do not match the observation schema")
    if set(prediction.categorical_logits) != expected_categorical:
        raise ValueError("prediction categorical fields do not match the observation schema")
    if set(prediction.presence_logits) != expected_presence:
        raise ValueError("prediction presence fields do not match the observation schema")
    reference = prediction.continuous["ego"]
    if reference.ndim != 2 or reference.shape[1] != schema.continuous_widths["ego"]:
        raise ValueError("prediction ego has an incompatible shape")
    batch_size = reference.shape[0]
    if batch_size <= 0:
        raise ValueError("prediction batch must contain at least one window")

    for name, values in prediction.continuous.items():
        width = schema.continuous_widths[name]
        shape = (batch_size, width) if name in schema.fixed_groups else (
            batch_size, schema.capacities[name], width
        )
        if tuple(values.shape) != shape:
            raise ValueError(f"prediction continuous {name} must have shape {shape}")
        if values.device != reference.device or values.dtype != reference.dtype:
            raise ValueError("prediction tensors must share device and floating-point dtype")
        if not values.is_floating_point() or not torch.isfinite(values).all():
            raise ValueError(f"prediction continuous {name} must be finite and floating point")
    for name, classes in schema.categorical_classes.items():
        group = name.split(".", 1)[0]
        logits = prediction.categorical_logits[name]
        shape = (batch_size, schema.capacities[group], classes)
        if tuple(logits.shape) != shape:
            raise ValueError(f"prediction categorical {name} must have shape {shape}")
        if logits.device != reference.device or logits.dtype != reference.dtype:
            raise ValueError("prediction tensors must share device and floating-point dtype")
        if not logits.is_floating_point() or not torch.isfinite(logits).all():
            raise ValueError(f"prediction categorical {name} must be finite and floating point")
    for group in schema.groups:
        logits = prediction.presence_logits[group]
        shape = (batch_size, schema.capacities[group])
        if tuple(logits.shape) != shape:
            raise ValueError(f"prediction presence {group} must have shape {shape}")
        if logits.device != reference.device or logits.dtype != reference.dtype:
            raise ValueError("prediction tensors must share device and floating-point dtype")
        if not logits.is_floating_point() or not torch.isfinite(logits).all():
            raise ValueError(f"prediction presence {group} must be finite and floating point")
    return batch_size, reference


def _per_window_mean(values: Tensor, batch_indices: Tensor, batch_size: int) -> Tensor:
    """Average matched values per window, assigning empty windows zero loss."""
    sums = values.new_zeros((batch_size,)).index_add(0, batch_indices, values)
    counts = torch.bincount(batch_indices, minlength=batch_size)
    denominators = counts.clamp_min(1).to(values.dtype)
    return torch.where(counts > 0, sums / denominators, torch.zeros_like(sums))


def match_objects(
    prediction: DecoderOutput,
    target_observations: Tensor,
    *,
    observation_layout: Mapping[str, int],
    matching_config: Mapping[str, Any],
) -> Mapping[str, ObjectAssignment]:
    """Match slots by detached typed-field costs, independently per window/group."""
    try:
        from scipy.optimize import linear_sum_assignment
    except ImportError as exc:
        raise RuntimeError("permutation-aware matching requires SciPy") from exc

    schema = ObservationSchema(observation_layout)
    batch_size, reference = _validate_prediction(prediction, schema)
    if not isinstance(matching_config, Mapping):
        raise ValueError("matching_config must be a mapping")
    if "costs" in matching_config:
        if matching_config.get("method") != "permutation_aware":
            raise ValueError("matching.method must be 'permutation_aware'")
        cost_config = matching_config["costs"]
    else:
        cost_config = matching_config
    costs = _finite_weights(
        cost_config, name="matching.costs", expected=_MATCHING_COST_KEYS, strictly_positive=False
    )
    if not isinstance(target_observations, Tensor):
        raise TypeError("target_observations must be a torch.Tensor")
    targets = schema.unpack(
        target_observations.to(device=reference.device, dtype=reference.dtype)
    )
    if target_observations.shape[0] != batch_size:
        raise ValueError("prediction and target batch sizes differ")

    assignments: dict[str, ObjectAssignment] = {}
    for group in schema.groups:
        predicted_values = prediction.continuous[group].detach()
        true_values = targets.continuous[group]
        continuous_cost = torch.cdist(predicted_values, true_values, p=2).square()
        continuous_cost = continuous_cost / predicted_values.shape[-1]
        categorical_cost = torch.zeros_like(continuous_cost)
        if group == "traffic_controls":
            for field in ("traffic_controls.type", "traffic_controls.state"):
                log_probabilities = F.log_softmax(prediction.categorical_logits[field].detach(), dim=-1)
                target_labels = targets.categorical[field]
                batch_index = torch.arange(batch_size, device=reference.device)[:, None, None]
                predicted_index = torch.arange(schema.capacities[group], device=reference.device)[None, :, None]
                categorical_cost = categorical_cost - log_probabilities[
                    batch_index, predicted_index, target_labels[:, None, :]
                ] / 2.0
        positive_presence_cost = F.softplus(
            -prediction.presence_logits[group].detach()
        ).unsqueeze(-1)
        pair_cost = (
            costs["continuous"] * continuous_cost
            + costs["categorical"] * categorical_cost
            + costs["presence"] * positive_presence_cost
        )
        if not torch.isfinite(pair_cost).all():
            raise ValueError(f"matching costs for {group} contain non-finite values")

        # Include each window's valid count in the same single group transfer.
        count_column = targets.valid_counts[group].to(pair_cost.dtype)[:, None, None]
        count_column = count_column.expand(-1, schema.capacities[group], 1)
        cpu_costs = torch.cat((pair_cost, count_column), dim=-1).cpu().numpy()
        index_rows: list[tuple[int, int, int]] = []
        for batch_index in range(batch_size):
            valid_count = int(cpu_costs[batch_index, 0, schema.capacities[group]])
            if valid_count == 0:
                continue
            predicted_slots, true_slots = linear_sum_assignment(
                cpu_costs[batch_index, :, :valid_count]
            )
            index_rows.extend(
                (batch_index, int(predicted_slot), int(true_slot))
                for predicted_slot, true_slot in zip(predicted_slots, true_slots)
            )
        indices = torch.tensor(index_rows, dtype=torch.long, device=reference.device)
        if not index_rows:
            indices = torch.empty((0, 3), dtype=torch.long, device=reference.device)
        assignments[group] = ObjectAssignment(indices)
    return assignments


def reconstruction_loss(
    prediction: DecoderOutput,
    target_observations: Tensor,
    assignments: Mapping[str, ObjectAssignment],
    *,
    observation_layout: Mapping[str, int],
    loss_config: Mapping[str, Any],
) -> ReconstructionLoss:
    """Combine matched attributes and all-slot presence loss with window means."""
    schema = ObservationSchema(observation_layout)
    batch_size, reference = _validate_prediction(prediction, schema)
    if not isinstance(loss_config, Mapping):
        raise ValueError("loss_config must be a mapping")
    group_weight_config = loss_config.get("group_weights", loss_config)
    group_weights = _finite_weights(
        group_weight_config,
        name="loss.group_weights",
        expected=set(_LOSS_GROUPS),
        strictly_positive=True,
    )
    if not isinstance(target_observations, Tensor):
        raise TypeError("target_observations must be a torch.Tensor")
    targets = schema.unpack(
        target_observations.to(device=reference.device, dtype=reference.dtype)
    )
    if target_observations.shape[0] != batch_size:
        raise ValueError("prediction and target batch sizes differ")
    if set(assignments) != set(schema.groups):
        raise ValueError("assignments must include every object group exactly once")

    components: dict[str, Tensor] = {}
    valid_counts: dict[str, int] = {
        "ego": batch_size * schema.continuous_widths["ego"],
        "context": batch_size * schema.continuous_widths["context"],
    }
    group_totals: dict[str, Tensor] = {}
    for group in schema.fixed_groups:
        error = F.smooth_l1_loss(
            prediction.continuous[group], targets.continuous[group], reduction="none"
        ).mean(dim=-1)
        group_loss = error.mean()
        components[f"{group}/continuous"] = group_loss
        group_totals[group] = group_loss

    for group in schema.groups:
        indices = assignments[group].indices
        if not isinstance(indices, Tensor) or indices.dtype != torch.long or indices.ndim != 2 or indices.shape[1] != 3:
            raise ValueError(f"assignment {group} must be an integer [N,3] tensor")
        if indices.device != reference.device:
            raise ValueError(f"assignment {group} must be on the prediction device")
        valid_counts[group] = int(sum(targets.valid_counts[group]).item())
        if indices.shape[0] != valid_counts[group]:
            raise ValueError(f"assignment {group} must contain one match per valid target object")
        if indices.shape[0]:
            batch_indices, predicted_slots, true_slots = indices.unbind(dim=1)
            if (
                torch.any(batch_indices < 0)
                or torch.any(batch_indices >= batch_size)
                or torch.any(predicted_slots < 0)
                or torch.any(predicted_slots >= schema.capacities[group])
                or torch.any(true_slots < 0)
                or torch.any(true_slots >= schema.capacities[group])
            ):
                raise ValueError(f"assignment {group} contains out-of-range indices")
            if torch.any(true_slots >= targets.valid_counts[group][batch_indices]):
                raise ValueError(f"assignment {group} refers to padded target slots")
            unique_matches = torch.stack(
                (batch_indices * schema.capacities[group] + predicted_slots,
                 batch_indices * schema.capacities[group] + true_slots),
                dim=1,
            )
            if torch.unique(unique_matches[:, 0]).numel() != indices.shape[0] or torch.unique(
                unique_matches[:, 1]
            ).numel() != indices.shape[0]:
                raise ValueError(f"assignment {group} must be one-to-one within each window")

            predicted = prediction.continuous[group][batch_indices, predicted_slots]
            target = targets.continuous[group][batch_indices, true_slots]
            per_object = F.smooth_l1_loss(predicted, target, reduction="none").mean(dim=-1)
            continuous_per_window = _per_window_mean(per_object, batch_indices, batch_size)
        else:
            batch_indices = torch.empty((0,), dtype=torch.long, device=reference.device)
            continuous_per_window = prediction.continuous[group].sum(dim=(1, 2)) * 0.0
        continuous_loss = continuous_per_window.mean()
        components[f"{group}/continuous"] = continuous_loss

        categorical_loss = reference.sum() * 0.0
        if group == "traffic_controls":
            categorical_loss = sum(
                prediction.categorical_logits[field].sum() * 0.0
                for field in ("traffic_controls.type", "traffic_controls.state")
            )
            if indices.shape[0]:
                categorical_errors = []
                for field in ("traffic_controls.type", "traffic_controls.state"):
                    logits = prediction.categorical_logits[field][batch_indices, predicted_slots]
                    labels = targets.categorical[field][batch_indices, true_slots]
                    categorical_errors.append(F.cross_entropy(logits, labels, reduction="none"))
                per_object_category = torch.stack(categorical_errors, dim=-1).mean(dim=-1)
                categorical_per_window = _per_window_mean(
                    per_object_category, batch_indices, batch_size
                )
                categorical_loss = categorical_per_window.mean()
            components[f"{group}/categorical"] = categorical_loss

        presence_targets = torch.zeros_like(prediction.presence_logits[group])
        if indices.shape[0]:
            presence_targets[batch_indices, predicted_slots] = 1.0
        presence_per_window = F.binary_cross_entropy_with_logits(
            prediction.presence_logits[group], presence_targets, reduction="none"
        ).mean(dim=1)
        presence_loss = presence_per_window.mean()
        components[f"{group}/presence"] = presence_loss
        group_totals[group] = continuous_loss + categorical_loss + presence_loss

    total = sum(group_weights[group] * group_totals[group] for group in _LOSS_GROUPS)
    return ReconstructionLoss(total=total, components=components, valid_counts=valid_counts)
