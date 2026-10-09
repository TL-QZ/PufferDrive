"""Drive networks for feed-forward discrete SAC."""

import torch
from torch import nn

import pufferlib
from pufferlib.ocean.drive import binding
from pufferlib.ocean.torch import Drive, DriveBackbone


def _action_values(env):
    if env.dynamics_model_flag == binding.DYNAMICS_MODEL_JERK:
        action_long_values, action_lat_values = binding.JERK_LONG, binding.JERK_LAT
    elif env.dynamics_model_flag == binding.DYNAMICS_MODEL_CLASSIC:
        action_long_values, action_lat_values = binding.ACCELERATION_VALUES, binding.STEERING_VALUES
    else:
        raise ValueError(f"Unsupported dynamics model: {env.dynamics_model}")
    return action_long_values, action_lat_values


def _make_head(input_size, hidden_size, num_layers, layer_norm, output_size, output_std):
    layers = []
    layer_input_size = input_size
    for _ in range(num_layers):
        layers.append(pufferlib.pytorch.layer_init(nn.Linear(layer_input_size, hidden_size)))
        if layer_norm:
            layers.append(nn.LayerNorm(hidden_size))
        layers.append(nn.ReLU())
        layer_input_size = hidden_size
    layers.append(pufferlib.pytorch.layer_init(nn.Linear(layer_input_size, output_size), std=output_std))
    return nn.Sequential(*layers)


class DriveSAC(nn.Module):
    """Discrete Drive actor returning action logits with shape ``[B, K]``."""

    discrete_actions_to_continuous = Drive.discrete_actions_to_continuous
    discrete_probs_to_continuous_mean = Drive.discrete_probs_to_continuous_mean
    pool_slot_counts = Drive.pool_slot_counts

    def __init__(
        self,
        env,
        ego_input_size: int,
        partner_input_size: int,
        lane_input_size: int,
        boundary_input_size: int,
        traffic_control_input_size: int,
        context_input_size: int,
        backbone_hidden_size: int,
        backbone_num_layers: int,
        actor_hidden_size: int,
        actor_num_layers: int,
        critic_hidden_size: int,
        critic_num_layers: int,
        encoder_activation: str,
        encoder_layer_norm: bool,
        backbone_activation: str,
        backbone_layer_norm: bool,
        shared_network: bool,
        mask_padded_features: bool,
        action_type: str,
        actor_head_layer_norm: bool = False,
        critic_head_layer_norm: bool = False,
    ):
        super().__init__()
        if action_type != "discrete":
            raise ValueError(f"DriveSAC supports only action_type='discrete', got: {action_type}")
        if shared_network:
            raise ValueError("DriveSAC requires shared_network=False so SAC models own separate parameters")

        action_long_values, action_lat_values = _action_values(env)
        action_long = torch.tensor(action_long_values, dtype=torch.float32)
        action_lat = torch.tensor(action_lat_values, dtype=torch.float32)

        long_norm = torch.where(
            action_long < 0.0,
            action_long / -action_long[0],
            action_long / action_long[-1],
        )
        lat_norm = action_lat / action_lat[-1]
        self.register_buffer("action_long_norm", long_norm, persistent=False)
        self.register_buffer("action_lat_norm", lat_norm, persistent=False)

        # Keep class ordering aligned with Drive's decoder: k = long_idx * num_lat + lat_idx.
        num_lat = lat_norm.numel()
        self.action_count = long_norm.numel() * num_lat
        action_indices = torch.arange(self.action_count)
        action_table = torch.stack(
            [long_norm[action_indices // num_lat], lat_norm[action_indices % num_lat]], dim=-1
        )
        self.register_buffer("action_table", action_table, persistent=False)
        action_table_physical = torch.stack(
            [action_long[action_indices // num_lat], action_lat[action_indices % num_lat]], dim=-1
        )
        self.register_buffer("action_table_physical", action_table_physical, persistent=False)
        self.action_long_neg_scale = float(-action_long[0])
        self.action_long_pos_scale = float(action_long[-1])
        self.action_lat_scale = float(action_lat[-1])
        self.is_continuous = False
        self.ego_dim = env.ego_features

        backbone_args = {
            "env": env,
            "ego_input_size": ego_input_size,
            "partner_input_size": partner_input_size,
            "lane_input_size": lane_input_size,
            "boundary_input_size": boundary_input_size,
            "traffic_control_input_size": traffic_control_input_size,
            "context_input_size": context_input_size,
            "backbone_hidden_size": backbone_hidden_size,
            "backbone_num_layers": backbone_num_layers,
            "ego_dim": self.ego_dim,
            "encoder_activation": encoder_activation,
            "encoder_layer_norm": encoder_layer_norm,
            "backbone_activation": backbone_activation,
            "backbone_layer_norm": backbone_layer_norm,
            "mask_padded_features": mask_padded_features,
        }
        self.actor_backbone = DriveBackbone(**backbone_args)
        self.actor_head = _make_head(
            self.actor_backbone.out_dim,
            actor_hidden_size,
            actor_num_layers,
            actor_head_layer_norm,
            self.action_count,
            output_std=0.01,
        )

    def forward(self, observations):
        hidden = self.actor_backbone(observations, self.ego_dim)
        return self.actor_head(hidden)


class DriveSACCritic(nn.Module):
    """Twin discrete Drive Q networks returning two ``[B, K]`` tensors."""

    def __init__(
        self,
        env,
        ego_input_size: int,
        partner_input_size: int,
        lane_input_size: int,
        boundary_input_size: int,
        traffic_control_input_size: int,
        context_input_size: int,
        backbone_hidden_size: int,
        backbone_num_layers: int,
        actor_hidden_size: int,
        actor_num_layers: int,
        critic_hidden_size: int,
        critic_num_layers: int,
        encoder_activation: str,
        encoder_layer_norm: bool,
        backbone_activation: str,
        backbone_layer_norm: bool,
        shared_network: bool,
        mask_padded_features: bool,
        action_type: str,
        actor_head_layer_norm: bool = False,
        critic_head_layer_norm: bool = False,
    ):
        super().__init__()
        if action_type != "discrete":
            raise ValueError(f"DriveSACCritic supports only action_type='discrete', got: {action_type}")
        if shared_network:
            raise ValueError("DriveSACCritic requires shared_network=False so SAC models own separate parameters")

        action_long_values, action_lat_values = _action_values(env)
        action_count = len(action_long_values) * len(action_lat_values)
        self.ego_dim = env.ego_features
        backbone_args = {
            "env": env,
            "ego_input_size": ego_input_size,
            "partner_input_size": partner_input_size,
            "lane_input_size": lane_input_size,
            "boundary_input_size": boundary_input_size,
            "traffic_control_input_size": traffic_control_input_size,
            "context_input_size": context_input_size,
            "backbone_hidden_size": backbone_hidden_size,
            "backbone_num_layers": backbone_num_layers,
            "ego_dim": self.ego_dim,
            "encoder_activation": encoder_activation,
            "encoder_layer_norm": encoder_layer_norm,
            "backbone_activation": backbone_activation,
            "backbone_layer_norm": backbone_layer_norm,
            "mask_padded_features": mask_padded_features,
        }

        self.q1_backbone = DriveBackbone(**backbone_args)
        self.q2_backbone = DriveBackbone(**backbone_args)
        self.q1_head = _make_head(
            self.q1_backbone.out_dim,
            critic_hidden_size,
            critic_num_layers,
            critic_head_layer_norm,
            action_count,
            output_std=1.0,
        )
        self.q2_head = _make_head(
            self.q2_backbone.out_dim,
            critic_hidden_size,
            critic_num_layers,
            critic_head_layer_norm,
            action_count,
            output_std=1.0,
        )

    def forward(self, observations):
        q1_hidden = self.q1_backbone(observations, self.ego_dim)
        q2_hidden = self.q2_backbone(observations, self.ego_dim)
        return self.q1_head(q1_hidden), self.q2_head(q2_hidden)
