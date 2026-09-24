"""Focused Condition B model invariants."""

import copy

import pytest
import torch
from torch import nn

from project.jepa_distill.model import ConditionBModel, ModelOutputs

torch.set_num_threads(1)


class TinyBackbone(nn.Module):
    def __init__(self, input_dim=6, latent_dim=8):
        super().__init__()
        self.ego_dim = input_dim
        self.out_dim = latent_dim
        self.projection = nn.Linear(input_dim, latent_dim)

    def forward(self, observations, ego_dim):
        assert ego_dim == self.ego_dim
        return self.projection(observations)


def make_config():
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
        "loss": {
            "temperature": 2.0,
            "distillation_weight": 1.0,
            "jepa_weight": 1.0,
            "variance_weight": 0.1,
            "variance_floor": 1.0,
            "epsilon": 1e-6,
        },
        "ema": {"tau": 0.5},
        "observation_layout": {"observation_dim": 6, "ego_features": 6},
    }


def make_drive_config(encoder_initialization=None):
    """Small resolved Drive architecture metadata for encoder initialization tests."""
    config = make_config()
    if encoder_initialization is not None:
        config["model"]["encoder_initialization"] = encoder_initialization
    config["teacher_config"] = {
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
    }
    config["observation_layout"] = {
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
    }
    return config


def make_frozen_drive_teacher(config):
    teacher = nn.Module()
    teacher.actor_backbone = ConditionBModel._build_context_encoder(
        config=config,
        observation_layout=config["observation_layout"],
        teacher=None,
    )
    teacher.eval()
    teacher.requires_grad_(False)
    return teacher


def test_forward_shapes_and_target_is_frozen():
    teacher = nn.Module()
    teacher.actor_backbone = TinyBackbone()
    model = ConditionBModel(make_config(), teacher=teacher)
    observations = torch.randn(4, 5, 6)
    controls = torch.randn(4, 4, 2)

    outputs = model(observations, controls)

    assert outputs.context_latents.shape == (4, 8)
    assert outputs.chunk_logits.shape == (4, 4, 12)
    assert outputs.predicted_endpoint.shape == (4, 8)
    assert outputs.target_endpoint.shape == (4, 8)
    assert not outputs.target_endpoint.requires_grad
    assert all(not parameter.requires_grad for parameter in model.target_encoder.parameters())


def test_teacher_encoder_initialization_remains_default_and_checkpoint_metadata_compatible():
    config = make_drive_config()
    teacher = make_frozen_drive_teacher(config)
    teacher_state = copy.deepcopy(teacher.actor_backbone.state_dict())

    model = ConditionBModel(config, teacher=teacher)

    for name, teacher_parameter in teacher.actor_backbone.named_parameters():
        torch.testing.assert_close(dict(model.context_encoder.named_parameters())[name], teacher_parameter)
        torch.testing.assert_close(dict(model.target_encoder.named_parameters())[name], teacher_parameter)
    assert all(parameter.requires_grad for parameter in model.context_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.target_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert model.export_metadata()["model"].get("encoder_initialization") is None
    for name, value in teacher_state.items():
        torch.testing.assert_close(teacher.actor_backbone.state_dict()[name], value)


def test_random_encoder_initialization_is_independent_and_target_starts_as_student():
    config = make_drive_config("random")
    teacher = make_frozen_drive_teacher(config)
    teacher_state = copy.deepcopy(teacher.actor_backbone.state_dict())

    torch.manual_seed(2718)
    model = ConditionBModel(config, teacher=teacher)

    student_state = model.context_encoder.state_dict()
    target_state = model.target_encoder.state_dict()
    assert student_state.keys() == teacher_state.keys()
    assert any(not torch.equal(student_state[name], teacher_state[name]) for name in teacher_state)
    for name in student_state:
        torch.testing.assert_close(target_state[name], student_state[name])
        torch.testing.assert_close(teacher.actor_backbone.state_dict()[name], teacher_state[name])
    assert all(parameter.requires_grad for parameter in model.context_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in model.target_encoder.parameters())
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert model.export_metadata()["model"]["encoder_initialization"] == "random"


def test_encoder_initialization_rejects_unknown_mode():
    config = make_drive_config("student")
    teacher = make_frozen_drive_teacher(config)

    with pytest.raises(ValueError, match="model.encoder_initialization"):
        ConditionBModel(config, teacher=teacher)


def test_gradient_ownership_and_ema_are_explicit():
    teacher = nn.Module()
    teacher.actor_backbone = TinyBackbone()
    model = ConditionBModel(make_config(), teacher=teacher)
    outputs = model(torch.randn(4, 5, 6), torch.randn(4, 4, 2))
    teacher_logits = torch.randn(4, 4, 12)
    losses = model.compute_losses(outputs, teacher_logits)
    losses.total.backward()

    assert model.context_encoder.projection.weight.grad is not None
    assert model.decoder[0].weight.grad is not None
    assert model.predictor[0].weight.grad is not None
    assert all(parameter.grad is None for parameter in model.target_encoder.parameters())

    previous_target = copy.deepcopy(model.target_encoder.state_dict())
    with torch.no_grad():
        for parameter in model.context_encoder.parameters():
            parameter.add_(1.0)
    model.update_target_encoder(tau=0.5)
    for name, previous in previous_target.items():
        expected = 0.5 * previous + 0.5 * model.context_encoder.state_dict()[name]
        torch.testing.assert_close(model.target_encoder.state_dict()[name], expected)
    model.train()
    assert not model.target_encoder.training


def test_loss_reductions_match_reference_and_reject_singleton_variance_batch():
    teacher = nn.Module()
    teacher.actor_backbone = TinyBackbone()
    model = ConditionBModel(make_config(), teacher=teacher)
    student_logits = torch.tensor([[[1.0, 0.0] * 6] * 4])
    teacher_logits = torch.tensor([[[0.0, 1.0] * 6] * 4])
    # The public method is independent of the configured number of classes.
    loss = model.distillation_loss(student_logits, teacher_logits, temperature=2.0)
    target_probability = torch.softmax(teacher_logits / 2.0, dim=-1)
    expected = -(target_probability * torch.log_softmax(student_logits / 2.0, dim=-1)).sum(-1).mean() * 4
    torch.testing.assert_close(loss, expected)
    with pytest.raises(ValueError, match="at least two"):
        model.variance_loss(torch.zeros(1, 8))


def test_exported_metadata_reconstructs_when_teacher_config_is_present():
    teacher = nn.Module()
    teacher.actor_backbone = TinyBackbone()
    config = make_config()
    config["teacher_config"] = {"env": {}, "policy": {}}
    model = ConditionBModel(config, teacher=teacher)
    reconstructed = ConditionBModel(model.export_metadata(), teacher=teacher)
    assert reconstructed.chunk_length == model.chunk_length
    assert reconstructed.observation_dim == model.observation_dim
