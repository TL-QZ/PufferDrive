"""Typed decoder, matching, and frozen checkpoint tests for the observation probe."""

from __future__ import annotations

import pytest
import torch

from project.jepa_distill.model import ConditionBModel
from project.jepa_distill.observation_probe.contracts import DecoderOutput
from project.jepa_distill.observation_probe.losses import match_objects, reconstruction_loss
from project.jepa_distill.observation_probe.model import ObservationDecoder, load_frozen_jepa
from project.jepa_distill.observation_probe.schema import ObservationSchema


def _layout() -> dict[str, int | str]:
    return {
        "observation_dim": 96,
        "dtype": "float32",
        "ego_features": 10,
        "num_reward_coefs": 2,
        "goal_dim": 3,
        "goal_features": 3,
        "context_dim": 5,
        "partner_features": 9,
        "lane_features": 9,
        "boundary_features": 9,
        "traffic_control_features": 7,
        "obs_slots_partners_n": 3,
        "obs_slots_lane_kept": 2,
        "obs_slots_boundary_kept": 2,
        "obs_slots_traffic_controls_n": 2,
        "obs_valid_count_features": 4,
    }


def _observations(schema: ObservationSchema) -> torch.Tensor:
    observations = torch.zeros((2, schema.observation_dim), dtype=torch.float32)
    observations[:, schema.offsets["ego"]] = torch.tensor(
        [[0.0] * 10, [0.1] * 10], dtype=torch.float32
    )
    observations[:, schema.offsets["context"]] = torch.tensor(
        [[0.2] * 5, [0.3] * 5], dtype=torch.float32
    )
    group_rows = {
        "partners": [
            [[0.1] * 9, [0.2] * 9],
            [[0.4] * 9],
        ],
        "lanes": [[[0.5] * 9], []],
        "boundaries": [[], []],
        "traffic_controls": [
            [[0.11, 0.12, 0.13, 0.14, 0.15, 1.0, 3.0],
             [0.21, 0.22, 0.23, 0.24, 0.25, 2.0, 0.0]],
            [],
        ],
    }
    for group, per_window_rows in group_rows.items():
        width = 7 if group == "traffic_controls" else schema.continuous_widths[group]
        flat = observations[:, schema.offsets[group]].reshape(2, schema.capacities[group], width)
        for batch_index, rows in enumerate(per_window_rows):
            if rows:
                flat[batch_index, : len(rows)] = torch.tensor(rows, dtype=torch.float32)
    for batch_index, count_by_group in enumerate(
        (
            {"lanes": 1, "boundaries": 0, "partners": 2, "traffic_controls": 2},
            {"lanes": 0, "boundaries": 0, "partners": 1, "traffic_controls": 0},
        )
    ):
        for group in schema.count_order:
            count_offset = schema.offsets["counts"].start + schema.count_order.index(group)
            observations[batch_index, count_offset] = count_by_group[group]
    return observations


def _target_prediction(schema: ObservationSchema, observations: torch.Tensor) -> DecoderOutput:
    targets = schema.unpack(observations)
    categorical_logits = {
        key: torch.nn.functional.one_hot(
            labels, num_classes=schema.categorical_classes[key]
        ).to(observations.dtype) * 16.0 - 8.0
        for key, labels in targets.categorical.items()
    }
    presence_logits = {
        group: torch.where(mask, 8.0, -8.0) for group, mask in targets.presence.items()
    }
    return DecoderOutput(
        {key: value.clone() for key, value in targets.continuous.items()},
        categorical_logits,
        presence_logits,
    )


def _reverse_valid_rows(
    value: torch.Tensor, counts: torch.Tensor
) -> torch.Tensor:
    result = value.clone()
    for batch_index, count in enumerate(counts.tolist()):
        if count > 1:
            result[batch_index, :count] = result[batch_index, :count].flip(0)
    return result


def _permuted_prediction(
    schema: ObservationSchema, prediction: DecoderOutput, observations: torch.Tensor
) -> DecoderOutput:
    targets = schema.unpack(observations)
    continuous = {
        key: value.clone() for key, value in prediction.continuous.items()
    }
    for group in schema.groups:
        continuous[group] = _reverse_valid_rows(
            continuous[group], targets.valid_counts[group]
        )
    categorical = {
        key: value.clone() for key, value in prediction.categorical_logits.items()
    }
    categorical["traffic_controls.type"] = _reverse_valid_rows(
        categorical["traffic_controls.type"],
        targets.valid_counts["traffic_controls"],
    )
    categorical["traffic_controls.state"] = _reverse_valid_rows(
        categorical["traffic_controls.state"],
        targets.valid_counts["traffic_controls"],
    )
    return DecoderOutput(continuous, categorical, prediction.presence_logits)


def _matching_config() -> dict[str, float]:
    return {"continuous": 1.0, "categorical": 1.0, "presence": 0.2}


def _loss_config() -> dict[str, dict[str, float]]:
    return {
        "group_weights": {
            "ego": 1.0,
            "context": 1.0,
            "partners": 1.0,
            "lanes": 1.0,
            "boundaries": 1.0,
            "traffic_controls": 1.0,
        }
    }


def test_schema_unpack_masks_counts_categories_and_roundtrips_baseline():
    schema = ObservationSchema(_layout())
    observations = _observations(schema)

    targets = schema.unpack(observations)

    assert schema.observation_dim == 96
    assert [schema.offsets[group].start for group in schema.groups] == [15, 42, 60, 78]
    assert schema.count_order == ("lanes", "boundaries", "partners", "traffic_controls")
    assert targets.valid_counts["partners"].tolist() == [2, 1]
    assert targets.valid_counts["traffic_controls"].tolist() == [2, 0]
    assert targets.presence["partners"].tolist() == [[True, True, False], [True, False, False]]
    assert targets.categorical["traffic_controls.type"].tolist() == [[1, 2], [0, 0]]
    assert targets.categorical["traffic_controls.state"].tolist() == [[3, 0], [0, 0]]
    assert targets.continuous_valid["traffic_controls"].shape == (2, 2, 5)
    assert targets.continuous_valid["ego"].all()

    reconstructed = schema.pack(schema.as_prediction(observations))

    torch.testing.assert_close(reconstructed, observations)


def test_pack_compacts_presence_rows_and_zeros_rejected_geometry():
    schema = ObservationSchema(_layout())
    observations = _observations(schema)
    prediction = _target_prediction(schema, observations)
    prediction.continuous["partners"][0, 0] = torch.full((9,), 1.0)
    prediction.continuous["partners"][0, 1] = torch.full((9,), 2.0)
    prediction.continuous["partners"][0, 2] = torch.full((9,), 3.0)
    prediction.presence_logits["partners"][0] = torch.tensor([8.0, -8.0, 8.0])
    prediction.continuous["traffic_controls"][0, 0] = torch.full((5,), 4.0)
    prediction.continuous["traffic_controls"][0, 1] = torch.full((5,), 5.0)
    prediction.categorical_logits["traffic_controls.type"][0] = torch.tensor(
        [[0.0, 9.0, 0.0, 0.0], [0.0, 0.0, 9.0, 0.0]]
    )
    prediction.categorical_logits["traffic_controls.state"][0] = torch.tensor(
        [[0.0, 0.0, 0.0, 9.0, 0.0], [9.0, 0.0, 0.0, 0.0, 0.0]]
    )
    prediction.presence_logits["traffic_controls"][0] = torch.tensor([-8.0, 8.0])

    packed = schema.pack(prediction, presence_threshold=0.5)
    targets = schema.unpack(packed)

    assert targets.valid_counts["partners"].tolist() == [2, 1]
    assert targets.continuous["partners"][0, :, 0].tolist() == [1.0, 3.0, 0.0]
    assert targets.categorical["traffic_controls.type"][0].tolist() == [2, 0]
    assert targets.categorical["traffic_controls.state"][0].tolist() == [0, 0]
    assert targets.continuous["traffic_controls"][0, :, 0].tolist() == [5.0, 0.0]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("partner_count", 4.0, "outside"),
        ("partner_count", 1.5, "integers"),
        ("traffic_type", 4.0, "outside"),
        ("traffic_state", 1.5, "integer IDs"),
    ],
)
def test_schema_rejects_invalid_counts_and_categories(field, value, message):
    schema = ObservationSchema(_layout())
    observations = _observations(schema)
    if field == "partner_count":
        observations[0, schema.offsets["partners_count"]] = value
    else:
        feature_index = 5 if field == "traffic_type" else 6
        rows = observations[:, schema.offsets["traffic_controls"]].reshape(2, 2, 7)
        rows[0, 0, feature_index] = value

    with pytest.raises(ValueError, match=message):
        schema.unpack(observations)


def test_decoder_head_shapes_and_gradients():
    schema = ObservationSchema(_layout())
    decoder = ObservationDecoder(4, _layout(), hidden_sizes=(8, 6))

    output = decoder(torch.randn(3, 4))
    total = sum(value.square().mean() for value in output.continuous.values())
    total = total + sum(value.square().mean() for value in output.categorical_logits.values())
    total = total + sum(value.square().mean() for value in output.presence_logits.values())
    total.backward()

    assert output.continuous["ego"].shape == (3, 10)
    assert output.continuous["context"].shape == (3, 5)
    assert output.continuous["partners"].shape == (3, 3, 9)
    assert output.continuous["traffic_controls"].shape == (3, 2, 5)
    assert output.categorical_logits["traffic_controls.type"].shape == (3, 2, 4)
    assert output.categorical_logits["traffic_controls.state"].shape == (3, 2, 5)
    assert output.presence_logits["lanes"].shape == (3, 2)
    assert decoder.trunk[0].weight.grad is not None
    assert decoder.categorical_heads["traffic_controls__type"].weight.grad is not None
    assert decoder.presence_heads["lanes"].weight.grad is not None
    assert schema.pack(output).shape == (3, schema.observation_dim)


def test_hungarian_matching_and_reconstruction_are_permutation_invariant():
    schema = ObservationSchema(_layout())
    observations = _observations(schema)
    prediction = _target_prediction(schema, observations)
    baseline_assignments = match_objects(
        prediction,
        observations,
        observation_layout=_layout(),
        matching_config=_matching_config(),
    )
    baseline_loss = reconstruction_loss(
        prediction,
        observations,
        baseline_assignments,
        observation_layout=_layout(),
        loss_config=_loss_config(),
    )
    permuted = _permuted_prediction(schema, prediction, observations)

    assignments = match_objects(
        permuted, observations, observation_layout=_layout(), matching_config=_matching_config()
    )
    loss = reconstruction_loss(
        permuted,
        observations,
        assignments,
        observation_layout=_layout(),
        loss_config=_loss_config(),
    )

    partners = assignments["partners"].indices
    assert set(map(tuple, partners.tolist())) >= {(0, 0, 1), (0, 1, 0)}
    assert all(indices.shape == (0, 3) for indices in [assignments["boundaries"].indices])
    assert torch.isfinite(loss.total)
    assert loss.total.item() < 0.01
    torch.testing.assert_close(loss.total, baseline_loss.total)
    for component in loss.components:
        torch.testing.assert_close(loss.components[component], baseline_loss.components[component])
    assert loss.valid_counts["partners"] == 3
    assert loss.valid_counts["traffic_controls"] == 2
    assert "traffic_controls/categorical" in loss.components


def test_empty_object_groups_have_finite_loss_and_gradient():
    schema = ObservationSchema(_layout())
    observations = torch.zeros((2, schema.observation_dim), dtype=torch.float32)
    decoder = ObservationDecoder(4, _layout(), hidden_sizes=(8,))
    prediction = decoder(torch.randn(2, 4))
    assignments = match_objects(
        prediction, observations, observation_layout=_layout(), matching_config=_matching_config()
    )
    loss = reconstruction_loss(
        prediction,
        observations,
        assignments,
        observation_layout=_layout(),
        loss_config=_loss_config(),
    )
    loss.total.backward()

    assert torch.isfinite(loss.total)
    assert all(indices.indices.shape == (0, 3) for indices in assignments.values())
    assert loss.valid_counts["partners"] == 0
    assert decoder.presence_heads["partners"].weight.grad is not None
    assert all(parameter.grad is not None for parameter in decoder.parameters())


def test_loss_reduction_is_additive_over_microbatches():
    schema = ObservationSchema(_layout())
    observations = _observations(schema)
    prediction = _target_prediction(schema, observations)
    prediction.continuous["ego"][0] += 0.4
    prediction.continuous["ego"][1] += 1.2
    prediction.continuous["context"][0] -= 0.3
    prediction.continuous["partners"][0] += 0.25
    prediction.continuous["partners"][1] += 0.8

    full_assignments = match_objects(
        prediction, observations, observation_layout=_layout(), matching_config=_matching_config()
    )
    full = reconstruction_loss(
        prediction,
        observations,
        full_assignments,
        observation_layout=_layout(),
        loss_config=_loss_config(),
    )
    microbatch_losses = []
    for index in range(2):
        micro_observations = observations[index : index + 1]
        micro_prediction = DecoderOutput(
            {key: value[index : index + 1] for key, value in prediction.continuous.items()},
            {key: value[index : index + 1] for key, value in prediction.categorical_logits.items()},
            {key: value[index : index + 1] for key, value in prediction.presence_logits.items()},
        )
        micro_assignments = match_objects(
            micro_prediction,
            micro_observations,
            observation_layout=_layout(),
            matching_config=_matching_config(),
        )
        microbatch_losses.append(
            reconstruction_loss(
                micro_prediction,
                micro_observations,
                micro_assignments,
                observation_layout=_layout(),
                loss_config=_loss_config(),
            )
        )

    torch.testing.assert_close(full.total, torch.stack([loss.total for loss in microbatch_losses]).mean())
    for component in full.components:
        torch.testing.assert_close(
            full.components[component],
            torch.stack([loss.components[component] for loss in microbatch_losses]).mean(),
        )


def _tiny_condition_b_config() -> dict:
    return {
        "model": {
            "chunk_length": 4,
            "execution_horizon": 1,
            "latent_dim": 8,
            "num_action_classes": 12,
            "decoder_hidden_sizes": [5],
            "predictor_hidden_sizes": [7, 6],
            "activation": "relu",
        },
        "observation_layout": {
            "observation_dim": 10,
            "ego_features": 6,
            "partner_features": 2,
            "lane_features": 2,
            "boundary_features": 2,
            "traffic_control_features": 2,
            "num_reward_coefs": 0,
            "goal_dim": 0,
            "obs_slots_partners_n": 0,
            "obs_slots_lane_kept": 0,
            "obs_slots_boundary_kept": 0,
            "obs_slots_traffic_controls_n": 0,
        },
        "teacher_config": {
            "env": {"dynamics_model": "jerk"},
            "policy": {
                "ego_input_size": 4,
                "partner_input_size": 3,
                "lane_input_size": 3,
                "boundary_input_size": 3,
                "traffic_control_input_size": 3,
                "context_input_size": 3,
                "backbone_hidden_size": 8,
                "backbone_num_layers": 1,
                "encoder_activation": "relu",
                "encoder_layer_norm": False,
                "backbone_activation": "relu",
                "backbone_layer_norm": False,
                "mask_padded_features": False,
            },
        },
    }


def test_load_frozen_jepa_strictly_reconstructs_checkpoint_without_teacher(tmp_path):
    model = ConditionBModel(_tiny_condition_b_config())
    checkpoint_path = tmp_path / "condition_b.pt"
    torch.save(
        {
            "format": "condition_b_v1",
            "model_config": model.export_metadata(),
            "model_state": model.state_dict(),
            "step": 17,
        },
        checkpoint_path,
    )

    loaded = load_frozen_jepa(checkpoint_path, device="cpu")

    assert isinstance(loaded, ConditionBModel)
    assert loaded.training is False
    assert all(not parameter.requires_grad for parameter in loaded.parameters())
    assert loaded.probe_checkpoint_metadata["step"] == 17
    input_observations = torch.zeros((2, 10), requires_grad=True)
    loaded.encode_context(input_observations).sum().backward()
    assert input_observations.grad is not None
    assert all(parameter.grad is None for parameter in loaded.parameters())
    for key, value in model.state_dict().items():
        torch.testing.assert_close(loaded.state_dict()[key], value)


def test_load_frozen_jepa_rejects_teacher_checkpoint_format(tmp_path):
    checkpoint_path = tmp_path / "teacher.pt"
    torch.save({"format": "ppo_teacher_v1"}, checkpoint_path)

    with pytest.raises(ValueError, match="Condition B checkpoint"):
        load_frozen_jepa(checkpoint_path, device="cpu")
