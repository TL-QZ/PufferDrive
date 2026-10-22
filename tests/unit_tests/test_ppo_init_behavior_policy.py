import copy
from contextlib import redirect_stdout
import io
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import yaml

from pufferlib import pytorch
from pufferlib.pufferl_sac_ppo_init_behavior_policy import (
    FrozenPPOBehavior,
    validate_config,
)
from project.soft_actor_critic_tuning.scripts import run_sac16_grid


REPO_ROOT = Path(__file__).resolve().parents[2]
PPO_DRIVE_CONFIG = REPO_ROOT / "pufferlib/config/puffer_drive.yaml"
ROUND4_1_CONFIG_DIR = REPO_ROOT / "project/soft_actor_critic_tuning/configs/round4_1"


class _TinyTeacherBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(1))
        self.calls = 0

    def forward(self, observations, ego_dim):
        self.calls += 1
        return observations[:, :ego_dim] + self.bias


class _TinyTeacherHead(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = nn.Parameter(torch.as_tensor(logits, dtype=torch.float32))

    def forward(self, hidden):
        return self.logits.expand(hidden.shape[0], -1)


class _TinyTeacher(nn.Module):
    def __init__(self, logits=(80.0, -80.0)):
        super().__init__()
        self.actor_backbone = _TinyTeacherBackbone()
        self.actor_head = _TinyTeacherHead(logits)
        self.ego_dim = 3

    def forward(self, observations):
        return self.actor_head(self.actor_backbone(observations, self.ego_dim))


class _TinySACActor(nn.Module):
    is_continuous = False

    def __init__(self, logits=(-80.0, 80.0)):
        super().__init__()
        self.logits = nn.Parameter(torch.as_tensor(logits, dtype=torch.float32))

    def forward(self, observations):
        return self.logits.expand(observations.shape[0], -1)


class _FakeLearner:
    def __init__(self, *, rank=0, total_agents=4, batch_size=8, logits=(-80.0, 80.0)):
        self.device = torch.device("cpu")
        self.rank = rank
        self.total_agents = total_agents
        self.batch_size = batch_size
        self.agent_steps = 0
        self.policy = _TinySACActor(logits)
        self.uncompiled_policy = self.policy
        self.pytorch = pytorch
        self.env_continuous = False
        self.amp_context = torch.autocast(device_type="cpu", enabled=False)
        self.pending = torch.ones(total_agents, dtype=torch.bool)
        self.distributed = False
        self.replay = []


def _settings(*, ppo_only_steps=250, decay_steps=500, total_steps=1000, seed=31):
    return {
        "ppo_only_steps": ppo_only_steps,
        "decay_steps": decay_steps,
        "total_steps": total_steps,
        "seed": seed,
    }


def _round4_1_trial_97(tmp_path_factory):
    output_dir = tmp_path_factory.mktemp("round4_1_generation") / "trial_97_run"
    with redirect_stdout(io.StringIO()):
        result = run_sac16_grid.main([
            "--config-dir", str(ROUND4_1_CONFIG_DIR),
            "--trial-ids", "97",
            "--gpu-groups", "0,1",
            "--max-concurrent", "1",
            "--dry-run",
            "--output-dir", str(output_dir),
        ])
    assert result == 0
    with (output_dir / "configs/trial_97.yaml").open(encoding="utf-8") as source:
        return yaml.safe_load(source)


@pytest.fixture(scope="module")
def round4_1_trial_97(tmp_path_factory):
    return _round4_1_trial_97(tmp_path_factory)


def _valid_config(tmp_path, round4_1_trial_97):
    config = copy.deepcopy(round4_1_trial_97)
    checkpoint = tmp_path / "teacher.pt"
    checkpoint.write_bytes(b"test checkpoint placeholder")
    config["behavior_policy"] = {
        "ppo_only_steps": 250,
        "decay_steps": 500,
        "total_steps": config["train"]["total_timesteps"],
        "seed": 31,
        "checkpoint": str(checkpoint),
        "teacher_config": str(PPO_DRIVE_CONFIG),
    }
    return config


def test_received_step_schedule_has_teacher_only_decay_and_sac_only_boundaries():
    learner = _FakeLearner()
    provider = FrozenPPOBehavior(learner, _TinyTeacher(), _settings())

    expected = (
        (0, 250, 1.0),
        (250, 750, 1.0),
        (375, 750, 0.75),
        (500, 750, 0.5),
        (749, 750, 0.002),
        (750, 1000, 0.0),
        (999, 1000, 0.0),
        (1000, 1000, 0.0),
    )
    for agent_steps, expected_boundary, expected_beta in expected:
        learner.agent_steps = agent_steps
        assert provider.begin_collection(learner) == expected_boundary
        assert provider.beta == pytest.approx(expected_beta)


def test_teacher_is_frozen_and_beta_one_samples_teacher_output():
    learner = _FakeLearner()
    teacher = _TinyTeacher(logits=(80.0, -80.0))
    provider = FrozenPPOBehavior(learner, teacher, _settings())
    assert not teacher.training
    assert all(not parameter.requires_grad for parameter in teacher.parameters())

    learner.agent_steps = 0
    provider.begin_collection(learner)
    observations = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    torch.manual_seed(14)
    expected_actions, _, _, _ = pytorch.sample_logits(
        teacher(observations).float(),
        env_continuous=learner.env_continuous,
        policy=learner.uncompiled_policy,
    )

    torch.manual_seed(14)
    actions, continuous_actions = provider.sample_actions(learner, observations)

    assert torch.equal(actions, expected_actions)
    assert continuous_actions is None
    assert provider.last_sources.tolist() == [True] * len(observations)
    assert all(parameter.grad is None for parameter in teacher.parameters())


def test_beta_zero_matches_native_sampling_output_and_rng_state():
    learner = _FakeLearner()
    teacher = _TinyTeacher()
    provider = FrozenPPOBehavior(
        learner, teacher, _settings(ppo_only_steps=0, decay_steps=1, total_steps=2),
    )
    learner.agent_steps = 1
    provider.begin_collection(learner)
    observations = torch.arange(24, dtype=torch.float32).reshape(8, 3)

    torch.manual_seed(2026)
    with learner.amp_context:
        native_logits = learner.policy(observations)
    native_actions, _, _, native_continuous = pytorch.sample_logits(
        native_logits.float(),
        env_continuous=learner.env_continuous,
        policy=learner.uncompiled_policy,
    )
    native_rng_state = torch.random.get_rng_state().clone()

    torch.manual_seed(2026)
    actions, continuous_actions = provider.sample_actions(learner, observations)
    provider_rng_state = torch.random.get_rng_state().clone()

    assert torch.equal(actions, native_actions)
    assert continuous_actions is native_continuous is None
    assert torch.equal(provider_rng_state, native_rng_state)
    assert provider.last_sources.tolist() == [False] * len(observations)
    assert teacher.actor_backbone.calls == 0


def test_mixed_source_mask_is_seeded_and_selects_the_labeled_actor():
    observations = torch.zeros(128, 3)
    first_learner = _FakeLearner(rank=2)
    second_learner = _FakeLearner(rank=2)
    first = FrozenPPOBehavior(first_learner, _TinyTeacher(), _settings())
    second = FrozenPPOBehavior(second_learner, _TinyTeacher(), _settings())
    first_learner.agent_steps = second_learner.agent_steps = 500
    first.begin_collection(first_learner)
    second.begin_collection(second_learner)

    first_actions, _ = first.sample_actions(first_learner, observations)
    second_actions, _ = second.sample_actions(second_learner, observations)

    assert torch.equal(first.last_sources, second.last_sources)
    assert 0 < first.last_sources.sum().item() < len(observations)
    assert torch.equal(first_actions, second_actions)
    assert torch.equal(first_actions, torch.where(
        first.last_sources, torch.zeros_like(first_actions), torch.ones_like(first_actions),
    ))


def test_transition_source_uses_previous_action_after_schedule_switches_to_sac():
    learner = _FakeLearner(total_agents=4)
    provider = FrozenPPOBehavior(learner, _TinyTeacher(), _settings())
    observations = torch.zeros(4, 3)
    env_ids = torch.arange(4)

    learner.agent_steps = 0
    provider.begin_collection(learner)
    provider.sample_actions(learner, observations)
    sent_sources = provider.last_sources.clone()
    provider.record_action_sources(env_ids)

    learner.agent_steps = 750
    provider.begin_collection(learner)
    learner.pending = torch.tensor([True, True, True, False])
    masks = torch.tensor([True, False, True, True])
    transition_sources = provider.transition_sources(env_ids, masks)
    provider.sample_actions(learner, observations)
    provider.record_action_sources(env_ids)

    assert sent_sources.tolist() == [True] * 4
    assert transition_sources.tolist() == [True, True]
    assert provider.pending_sources.tolist() == [False] * 4


def test_replay_source_replacement_and_metrics_follow_stored_labels():
    learner = _FakeLearner(total_agents=4, batch_size=4)
    provider = FrozenPPOBehavior(learner, _TinyTeacher(), _settings())
    learner.agent_steps = 500
    provider.begin_collection(learner)
    observations = torch.zeros(4, 3)
    provider.sample_actions(learner, observations)
    action_teacher_count = int(provider.last_sources.sum().item())

    initial_sources = torch.tensor([True, True, False])
    provider.record_replay_sources(initial_sources, torch.tensor([0, 1, 2]))
    provider.record_replay_sources(torch.tensor([False]), torch.tensor([1]))
    learner.replay = [object(), object(), object()]
    provider.end_collection(learner)

    metrics = provider.metrics()
    assert provider.replay_sources[:3].tolist() == [True, False, False]
    assert metrics["behavior/ppo_probability"] == pytest.approx(0.5)
    assert metrics["behavior/ppo_action_fraction"] == pytest.approx(action_teacher_count / 4)
    assert metrics["behavior/ppo_insert_fraction"] == pytest.approx(2 / 4)
    assert metrics["behavior/ppo_replay_fraction"] == pytest.approx(1 / 3)
    assert metrics["behavior/ppo_actions"] == action_teacher_count
    assert metrics["behavior/sac_actions"] == 4 - action_teacher_count
    assert metrics["behavior/ppo_replay_transitions"] == 1
    assert metrics["behavior/collection_start_steps"] == 500
    assert metrics["behavior/stage"] == 1


def test_validate_config_accepts_real_round4_1_trial_and_repo_ppo_drive_config(
    tmp_path, round4_1_trial_97,
):
    config = _valid_config(tmp_path, round4_1_trial_97)

    teacher_config = validate_config(config)

    assert teacher_config["policy_name"] == "Drive"
    assert teacher_config["rnn_name"] is None
    assert teacher_config["policy"]["action_type"] == "discrete"
    assert config["env"]["dt"] == teacher_config["env"]["dt"] == 0.3


@pytest.mark.parametrize("missing_key", ["checkpoint", "teacher_config"])
def test_validate_config_rejects_missing_teacher_files(
    tmp_path, round4_1_trial_97, missing_key,
):
    config = _valid_config(tmp_path, round4_1_trial_97)
    config["behavior_policy"][missing_key] = str(tmp_path / "missing-file")

    with pytest.raises(FileNotFoundError, match=f"Teacher {missing_key} not found"):
        validate_config(config)


def test_validate_config_rejects_teacher_environment_representation_mismatch(
    tmp_path, round4_1_trial_97,
):
    config = _valid_config(tmp_path, round4_1_trial_97)
    teacher_config = yaml.safe_load(PPO_DRIVE_CONFIG.read_text(encoding="utf-8"))
    teacher_config["env"]["dt"] = 0.1
    mismatched_config_path = tmp_path / "teacher_mismatch.yaml"
    mismatched_config_path.write_text(yaml.safe_dump(teacher_config), encoding="utf-8")
    config["behavior_policy"]["teacher_config"] = str(mismatched_config_path)

    with pytest.raises(ValueError, match="representation mismatch: dt"):
        validate_config(config)


@pytest.mark.parametrize(
    ("setting", "value"),
    [
        ("load_id", "old-run"),
        ("load_model_path", "old-actor.pt"),
        ("resume_state_path", "old-state.pt"),
        ("fresh_run", False),
    ],
)
def test_validate_config_requires_a_fresh_sac_learner(
    tmp_path, round4_1_trial_97, setting, value,
):
    config = _valid_config(tmp_path, round4_1_trial_97)
    if setting == "fresh_run":
        config["train"][setting] = value
    elif setting == "resume_state_path":
        config["train"][setting] = value
    else:
        config[setting] = value

    with pytest.raises(ValueError, match="fresh SAC|fresh_run=true"):
        validate_config(config)
