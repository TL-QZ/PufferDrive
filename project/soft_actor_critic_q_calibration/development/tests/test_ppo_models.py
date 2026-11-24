from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml

from project.soft_actor_critic_q_calibration.development.ppo_models import FrozenPPO


class FakeDrive(torch.nn.Module):
    def __init__(self, model_env, **policy_config):
        super().__init__()
        assert policy_config["action_type"] == "discrete"
        self.logit_scale = torch.nn.Parameter(torch.tensor(1.0))
        self.value_bias = torch.nn.Parameter(torch.tensor(0.5))
        self.register_buffer("action_table", torch.zeros((12, 2)))
        self.register_buffer("action_table_physical", torch.ones((12, 2)))

    def forward(self, observations):
        logits = observations[:, :1] * self.logit_scale + torch.arange(
            12, device=observations.device, dtype=observations.dtype
        )
        logits = logits.expand(-1, 12)
        values = observations[:, :1] + self.value_bias
        return [logits], values


def _checkpoint(tmp_path, monkeypatch, config_overrides=None):
    import pufferlib.ocean.torch as ocean_torch

    monkeypatch.setattr(ocean_torch, "Drive", FakeDrive, raising=False)
    model_env = SimpleNamespace(dynamics_model="jerk", num_obs=3)
    config = {
        "package": "ocean",
        "policy_name": "Drive",
        "rnn_name": None,
        "policy": {"action_type": "discrete"},
        "env": {"dynamics_model": "jerk"},
        "train": {"gamma": 0.999, "gae_lambda": 0.95, "amp": False},
    }
    if config_overrides:
        config.update(config_overrides)
    (tmp_path / "config.yaml").write_text(yaml.safe_dump(config), encoding="utf-8")
    torch.save(FakeDrive(model_env, action_type="discrete").state_dict(), tmp_path / "final_model.pt")
    return tmp_path / "final_model.pt", model_env


def test_frozen_ppo_loads_discrete_policy_and_values(tmp_path, monkeypatch):
    checkpoint_path, model_env = _checkpoint(tmp_path, monkeypatch)
    model = FrozenPPO(checkpoint_path, model_env)

    observations = np.asarray([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=np.float32)
    probabilities, log_probabilities = model.policy(observations)
    values = model.values(observations)

    assert probabilities.shape == log_probabilities.shape == (2, 12)
    assert values.shape == (2,)
    np.testing.assert_allclose(probabilities.sum(axis=-1), 1.0, atol=1e-6)
    np.testing.assert_allclose(np.log(probabilities), log_probabilities, atol=1e-6)
    np.testing.assert_allclose(values, [1.5, 2.5])
    assert model.action_table.shape == model.action_table_physical.shape == (12, 2)
    assert model.gamma == 0.999
    assert all(not parameter.requires_grad for parameter in model.policy_model.parameters())
    model.verify_frozen()

    with torch.no_grad():
        model.policy_model.value_bias.add_(1.0)
    with pytest.raises(AssertionError, match="parameters changed"):
        model.verify_frozen()


@pytest.mark.parametrize(
    "config_overrides,model_env,match",
    [
        ({"rnn_name": "LSTM"}, SimpleNamespace(dynamics_model="jerk", num_obs=3), "feed-forward"),
        ({"policy": {"action_type": "continuous"}}, SimpleNamespace(dynamics_model="jerk", num_obs=3), "discrete"),
        ({}, SimpleNamespace(dynamics_model="acceleration", num_obs=3), "different dynamics"),
    ],
)
def test_frozen_ppo_rejects_unsupported_model_configs(
    tmp_path, monkeypatch, config_overrides, model_env, match
):
    checkpoint_path, _ = _checkpoint(tmp_path, monkeypatch, config_overrides)
    with pytest.raises(ValueError, match=match):
        FrozenPPO(checkpoint_path, model_env)


def test_frozen_ppo_rejects_bad_observation_shape(tmp_path, monkeypatch):
    checkpoint_path, model_env = _checkpoint(tmp_path, monkeypatch)
    model = FrozenPPO(checkpoint_path, model_env)
    with pytest.raises(ValueError, match="Expected observations"):
        model.policy(np.zeros((3,), dtype=np.float32))
    with pytest.raises(ValueError, match="must be finite"):
        model.values(np.full((1, 3), np.nan, dtype=np.float32))
