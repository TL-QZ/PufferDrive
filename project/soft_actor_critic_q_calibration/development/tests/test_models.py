from types import SimpleNamespace

import numpy as np
import torch
import yaml

from pufferlib.ocean.drive import binding
from pufferlib.ocean.torch_sac import DriveSAC, DriveSACCritic
from project.soft_actor_critic_q_calibration.development.models import FrozenModels


def _template_env():
    return SimpleNamespace(
        dynamics_model="jerk",
        dynamics_model_flag=binding.DYNAMICS_MODEL_JERK,
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
    )


def _policy_config():
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


def _write_checkpoint(directory, env, policy_config):
    actor = DriveSAC(env, **policy_config)
    critic = DriveSACCritic(env, **policy_config)
    target = DriveSACCritic(env, **policy_config)
    checkpoint_path = directory / "trainer_state.pt"
    torch.save(
        {
            "config": {
                "gamma": 0.99,
                "return_estimator": "n_step",
                "n_steps": 1,
                "critic_entropy_in_target": True,
            },
            "epoch": 4,
            "agent_steps": 123,
            "log_alpha": torch.tensor(-1.0),
            "policy_state_dict": actor.state_dict(),
            "critic_state_dict": critic.state_dict(),
            "target_critic_state_dict": target.state_dict(),
        },
        checkpoint_path,
    )
    (directory / "config.yaml").write_text(
        yaml.safe_dump({"policy": policy_config, "train": {"amp": False}}),
        encoding="utf-8",
    )
    return checkpoint_path, actor, critic, target


def test_frozen_models_strict_checkpoint_load_and_soft_target(tmp_path):
    previous_thread_count = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        env = _template_env()
        policy_config = _policy_config()
        checkpoint_path, actor, critic, target = _write_checkpoint(tmp_path, env, policy_config)
        models = FrozenModels(checkpoint_path, env, device="cpu")

        assert models.actor.state_dict().keys() == actor.state_dict().keys()
        assert models.critic.state_dict().keys() == critic.state_dict().keys()
        assert models.target.state_dict().keys() == target.state_dict().keys()
        for name, original in (("actor", actor), ("critic", critic), ("target", target)):
            for loaded_parameter, original_parameter in zip(
                getattr(models, name).parameters(), original.parameters()
            ):
                torch.testing.assert_close(loaded_parameter, original_parameter)
                assert not loaded_parameter.requires_grad

        observations = np.zeros((2, 74), dtype=np.float32)
        observations[:, 68:70] = 0  # valid traffic-control type and state IDs
        probabilities, log_probabilities = models.policy(observations)
        q1_values, q2_values = models.q_values(observations)
        target_values = models.next_values(observations)

        assert probabilities.shape == log_probabilities.shape == (2, 12)
        assert q1_values.shape == q2_values.shape == (2, 12)
        assert target_values.shape == (2,)
        assert np.isfinite(probabilities).all()
        assert np.isfinite(log_probabilities).all()
        assert np.isfinite(q1_values).all() and np.isfinite(q2_values).all()
        assert np.isfinite(target_values).all()
        np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0, atol=1e-6)

        with torch.inference_mode():
            observation_tensor = torch.as_tensor(observations)
            actor_logits = models.actor(observation_tensor).float()
            target_q1, target_q2 = models.target(observation_tensor)
            expected_values = (
                actor_logits.log_softmax(dim=-1).exp()
                * (
                    torch.minimum(target_q1.float(), target_q2.float())
                    - models.alpha * actor_logits.log_softmax(dim=-1)
                )
            ).sum(dim=-1)
        np.testing.assert_allclose(target_values, expected_values.numpy(), rtol=1e-6, atol=1e-6)

        models.verify_frozen()
    finally:
        torch.set_num_threads(previous_thread_count)
