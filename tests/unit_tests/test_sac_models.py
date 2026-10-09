from types import SimpleNamespace

import pytest
import torch

from pufferlib.ocean.drive import binding
from pufferlib.ocean.torch import Drive
from pufferlib.ocean.torch_sac import DriveSAC, DriveSACCritic


def _mock_env(dynamics_model):
    dynamics_model_flag = {
        "classic": binding.DYNAMICS_MODEL_CLASSIC,
        "jerk": binding.DYNAMICS_MODEL_JERK,
    }[dynamics_model]
    return SimpleNamespace(
        dynamics_model=dynamics_model,
        dynamics_model_flag=dynamics_model_flag,
        ego_features=10,
        num_reward_coefs=17,
        goal_dim=9,
        obs_slots_partners_n=1,
        partner_features=9,
        obs_slots_lane_kept=1,
        lane_features=9,
        obs_slots_boundary_kept=1,
        boundary_features=9,
        obs_slots_traffic_controls_n=1,
        traffic_control_features=7,
        single_action_space=SimpleNamespace(shape=(2,)),
    )


def _policy_kwargs():
    return {
        "ego_input_size": 8,
        "partner_input_size": 8,
        "lane_input_size": 8,
        "boundary_input_size": 8,
        "traffic_control_input_size": 8,
        "context_input_size": 8,
        "backbone_hidden_size": 16,
        "backbone_num_layers": 1,
        "actor_hidden_size": 16,
        "actor_num_layers": 1,
        "critic_hidden_size": 16,
        "critic_num_layers": 1,
        "encoder_activation": "relu",
        "encoder_layer_norm": False,
        "backbone_activation": "gelu",
        "backbone_layer_norm": False,
        "shared_network": False,
        "mask_padded_features": True,
        "action_type": "discrete",
        "actor_head_layer_norm": False,
        "critic_head_layer_norm": False,
    }


def _mock_observations(batch_size=2):
    # 10 ego + 26 context + 9 partner + 9 lane + 9 boundary + 7 traffic + 4 counts.
    observations = torch.randn(batch_size, 74)
    observations[:, 68:70] = 0  # valid traffic-control type and state IDs
    observations[:, 70:74] = 1  # one valid row for each pooled object group
    return observations


@pytest.mark.parametrize("dynamics_model", ["classic", "jerk"])
def test_drive_sac_shapes_gradients_parameter_independence_and_action_parity(dynamics_model):
    previous_thread_count = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        torch.manual_seed(23)
        env = _mock_env(dynamics_model)
        policy_kwargs = _policy_kwargs()
        actor = DriveSAC(env, **policy_kwargs)
        critic = DriveSACCritic(env, **policy_kwargs)

        # PPO Drive is the action mapping oracle; dimensions are intentionally tiny.
        ppo_drive = Drive(env, **policy_kwargs)
        observations = _mock_observations()
        logits = actor(observations)
        q1, q2 = critic(observations)
        action_count = ppo_drive.action_table.shape[0]
        assert logits.shape == (2, action_count)
        assert q1.shape == (2, action_count)
        assert q2.shape == (2, action_count)
        assert actor.is_continuous is False

        actor_parameter_ids = {id(parameter) for parameter in actor.parameters()}
        q1_parameter_ids = {id(parameter) for parameter in critic.q1_backbone.parameters()} | {
            id(parameter) for parameter in critic.q1_head.parameters()
        }
        q2_parameter_ids = {id(parameter) for parameter in critic.q2_backbone.parameters()} | {
            id(parameter) for parameter in critic.q2_head.parameters()
        }
        assert actor_parameter_ids.isdisjoint(q1_parameter_ids)
        assert actor_parameter_ids.isdisjoint(q2_parameter_ids)
        assert q1_parameter_ids.isdisjoint(q2_parameter_ids)
        assert any(
            not torch.equal(q1_parameter, q2_parameter)
            for q1_parameter, q2_parameter in zip(critic.q1_backbone.parameters(), critic.q2_backbone.parameters())
        )

        action_indices = torch.arange(action_count)
        probabilities = torch.softmax(torch.randn(2, action_count), dim=-1)
        torch.testing.assert_close(actor.action_table, ppo_drive.action_table)
        torch.testing.assert_close(actor.action_table_physical, ppo_drive.action_table_physical)
        torch.testing.assert_close(
            actor.discrete_actions_to_continuous(action_indices),
            ppo_drive.discrete_actions_to_continuous(action_indices),
        )
        torch.testing.assert_close(
            actor.discrete_probs_to_continuous_mean(probabilities),
            ppo_drive.discrete_probs_to_continuous_mean(probabilities),
        )

        (logits.square().mean() + q1.square().mean() + q2.square().mean()).backward()
        assert all(parameter.grad is not None for parameter in actor.parameters())
        assert all(parameter.grad is not None for parameter in critic.parameters())
        assert all(torch.isfinite(parameter.grad).all() for parameter in actor.parameters())
        assert all(torch.isfinite(parameter.grad).all() for parameter in critic.parameters())
    finally:
        torch.set_num_threads(previous_thread_count)


def test_drive_sac_masked_models_support_bfloat16_autocast_and_empty_objects():
    previous_thread_count = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        env = _mock_env("jerk")
        policy_kwargs = _policy_kwargs()
        actor = DriveSAC(env, **policy_kwargs)
        critic = DriveSACCritic(env, **policy_kwargs)
        observations = _mock_observations(batch_size=2)
        observations[1, 70:74] = 0

        with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
            logits = actor(observations)
            q1, q2 = critic(observations)
            assert logits.dtype == torch.bfloat16
            assert q1.dtype == torch.bfloat16
            assert q2.dtype == torch.bfloat16
            assert torch.isfinite(logits).all()
            assert torch.isfinite(q1).all()
            assert torch.isfinite(q2).all()
            loss = logits.square().mean() + q1.square().mean() + q2.square().mean()

        loss.backward()
        assert all(parameter.grad is not None for parameter in actor.parameters())
        assert all(parameter.grad is not None for parameter in critic.parameters())
        assert all(torch.isfinite(parameter.grad).all() for parameter in actor.parameters())
        assert all(torch.isfinite(parameter.grad).all() for parameter in critic.parameters())
    finally:
        torch.set_num_threads(previous_thread_count)


@pytest.mark.parametrize("model_class", [DriveSAC, DriveSACCritic])
def test_drive_sac_models_reject_unsupported_construction_options(model_class):
    env = _mock_env("jerk")
    policy_kwargs = _policy_kwargs()

    with pytest.raises(ValueError, match="action_type='discrete'"):
        model_class(env, **{**policy_kwargs, "action_type": "continuous"})
    with pytest.raises(ValueError, match="shared_network=False"):
        model_class(env, **{**policy_kwargs, "shared_network": True})
