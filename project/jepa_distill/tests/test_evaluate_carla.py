"""Chunked CARLA policy scheduling and configuration tests."""

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from torch import nn
import yaml

from project.jepa_distill.evaluate import StudentPolicyAdapter
from project.jepa_distill.evaluate_carla import (
    ChunkPolicyAdapter,
    build_protocol_config,
    main,
    parse_execution_ks,
    validate_checkpoint_metadata,
)


class ObservationEncodedChunkStudent(nn.Module):
    """Make the planned observation and action slot visible in each output."""

    def __init__(self):
        super().__init__()
        self.observation_batches = []

    def forward(self, observations):
        self.observation_batches.append(observations.detach().clone())
        batch_size = observations.shape[0]
        chunk_logits = torch.full((batch_size, 4, 12), -1000.0, device=observations.device)
        for slot_idx in range(4):
            chunk_logits[:, slot_idx, slot_idx] = observations[:, 0] * 100.0 + slot_idx + 1.0
        return chunk_logits


class InvalidShapeStudent(nn.Module):
    def forward(self, observations):
        return torch.zeros((observations.shape[0], 3, 12), device=observations.device)


class LinearChunkStudent(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 48)

    def forward(self, observations):
        return self.projection(observations).reshape(observations.shape[0], 4, 12)


class ResetTrackingChunkPolicyAdapter(ChunkPolicyAdapter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.reset_masks = []

    def reset_eval_state(self, mask=None):
        self.reset_masks.append(None if mask is None else np.asarray(mask, dtype=bool).copy())
        super().reset_eval_state(mask)


def _scheduled_logits(adapter, observations):
    (logits,), _ = adapter.forward_eval(observations)
    return logits


@pytest.mark.parametrize("execution_k", [1, 2, 3, 4])
def test_chunk_policy_replans_and_selects_the_expected_action_slot(execution_k):
    student = ObservationEncodedChunkStudent()
    adapter = ChunkPolicyAdapter(student, execution_k=execution_k)
    cycle_length = 1 if execution_k == 1 else 3 * execution_k

    for tick_idx in range(3 * execution_k + 1):
        observations = torch.full((1, 3), float(tick_idx))
        logits = _scheduled_logits(adapter, observations)
        if execution_k == 1:
            selected_slot = 0
            planned_tick = tick_idx
        else:
            selected_slot = (tick_idx % cycle_length) // 3
            planned_tick = (tick_idx // cycle_length) * cycle_length

        assert int(logits.argmax(dim=-1).item()) == selected_slot
        assert logits[0, selected_slot].item() == pytest.approx(
            planned_tick * 100.0 + selected_slot + 1.0
        )

    expected_calls = 3 * execution_k + 1 if execution_k == 1 else 2
    assert len(student.observation_batches) == expected_calls
    if execution_k > 1:
        assert student.observation_batches[-1][0, 0].item() == 3 * execution_k


def test_done_mask_replans_only_reset_agents_from_the_new_observation():
    student = ObservationEncodedChunkStudent()
    adapter = ChunkPolicyAdapter(student, execution_k=2)
    _scheduled_logits(adapter, torch.tensor([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]))

    adapter.reset_eval_state(torch.tensor([True, False]))
    logits = _scheduled_logits(adapter, torch.tensor([[30.0, 0.0, 0.0], [40.0, 0.0, 0.0]]))

    assert student.observation_batches[0].shape[0] == 2
    assert student.observation_batches[1].shape[0] == 1
    assert logits[0, 0].item() == pytest.approx(3001.0)
    assert logits[1, 0].item() == pytest.approx(2001.0)


def test_rollout_reset_discards_all_cached_chunk_logits():
    student = ObservationEncodedChunkStudent()
    adapter = ChunkPolicyAdapter(student, execution_k=4)
    _scheduled_logits(adapter, torch.tensor([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]]))

    adapter.reset_eval_state()
    logits = _scheduled_logits(adapter, torch.tensor([[30.0, 0.0, 0.0], [40.0, 0.0, 0.0]]))

    assert [batch.shape[0] for batch in student.observation_batches] == [2, 2]
    assert logits[:, 0].tolist() == pytest.approx([3001.0, 4001.0])


@pytest.mark.parametrize("execution_k", [1, 2, 3, 4])
def test_first_chunk_action_and_mean_control_match_slot_zero_adapter(execution_k):
    observations = torch.tensor([[0.2, -0.1, 0.4], [0.7, 0.3, -0.2]])
    student = LinearChunkStudent()
    slot_zero_adapter = StudentPolicyAdapter(student)
    chunk_adapter = ChunkPolicyAdapter(student, execution_k=execution_k)

    (slot_zero_logits,), _ = slot_zero_adapter.forward_eval(observations)
    (chunk_logits,), _ = chunk_adapter.forward_eval(observations)
    torch.testing.assert_close(chunk_logits, slot_zero_logits)

    action_probabilities = torch.softmax(slot_zero_logits, dim=-1)
    torch.testing.assert_close(
        chunk_adapter.discrete_probs_to_continuous_mean(action_probabilities),
        slot_zero_adapter.discrete_probs_to_continuous_mean(action_probabilities),
    )


def test_protocol_defaults_match_the_baseline_carla_benchmark_and_launcher():
    teacher_args = {
        "env": {"dt": 0.3},
        "policy": {},
        "eval": {},
        "train": {"device": "cpu", "amp": False, "compile": False},
        "vec": {"seed": 0, "num_envs": 1},
        "package": "ocean",
        "policy_name": "Drive",
    }
    checkpoint = {
        "format": "condition_b_v1",
        "model_config": {"chunk_length": 4, "num_action_classes": 12},
        "config": {"teacher_config": teacher_args},
    }
    protocol = build_protocol_config(
        checkpoint,
        execution_k=4,
        checkpoint_path="student.pt",
        output_dir="out",
        device="cpu",
    )
    baseline_path = "project/baseline_run_sync_2026-08-24/override_config/evaluation_benchmarks.yaml"
    with open(baseline_path, encoding="utf-8") as config_file:
        baseline = yaml.safe_load(config_file)
    carla = next(item for item in baseline["benchmarks"] if item["name"] == "carla")

    puffer_args = protocol["puffer_args"]
    from pufferlib.ocean.evaluation_utils import evaluation_utils as drive_benchmark

    resolved_args = drive_benchmark.build_benchmark_args(puffer_args, carla, baseline["env"])
    assert protocol["evaluation"]["benchmark_config"] == str(Path(baseline_path).resolve())
    assert protocol["evaluation"]["benchmarks"] == ["carla"]
    assert protocol["evaluation"]["env_overrides"]["dt"] == baseline["env"]["dt"] == 0.1
    assert protocol["evaluation"]["env_overrides"]["action_type"] == "continuous"
    assert protocol["evaluation"]["num_scenarios"] == carla["num_scenarios"] == 1000
    assert protocol["evaluation"]["episode_timesteps"] == carla["env"]["scenario_length"] == 6000
    for key, value in carla["env"].items():
        assert resolved_args["env"][key] == value
    assert resolved_args["env"]["dt"] == baseline["env"]["dt"] == 0.1
    assert resolved_args["env"]["num_agents"] == 300
    assert resolved_args["env"]["resample_frequency"] == 6000
    assert resolved_args["num_scenarios"] == 1000
    assert puffer_args["eval"]["num_agents"] == 300
    assert puffer_args["eval"]["action_selection"] == "mean"
    assert puffer_args["vec"]["num_envs"] == 20
    assert puffer_args["eval"]["render_scenarios"] is False

    smoke_protocol = build_protocol_config(
        checkpoint,
        execution_k=4,
        checkpoint_path="student.pt",
        output_dir="out",
        device="cpu",
        smoke=True,
    )
    assert smoke_protocol["evaluation"]["num_scenarios"] == 2
    assert smoke_protocol["evaluation"]["episode_timesteps"] == 13
    assert smoke_protocol["evaluation"]["num_agents"] == 50
    assert smoke_protocol["evaluation"]["vec_overrides"]["num_envs"] == 1


def test_invalid_execution_k_cli_and_checkpoint_metadata_fail_fast(tmp_path):
    assert parse_execution_ks("all") == (1, 2, 3, 4)
    with pytest.raises(argparse.ArgumentTypeError, match="--k"):
        parse_execution_ks("0")
    with pytest.raises(ValueError, match="Condition B|format|checkpoint"):
        validate_checkpoint_metadata({"format": "pufferlib_ppo"})
    with pytest.raises(ValueError, match="chunk_length|four|4"):
        validate_checkpoint_metadata(
            {
                "format": "condition_b_v1",
                "model_config": {"chunk_length": 3, "num_action_classes": 12},
                "config": {"teacher_config": {"env": {"dt": 0.3}}},
            }
        )
    with pytest.raises(ValueError, match="training dt"):
        validate_checkpoint_metadata(
            {
                "model_config": {"chunk_length": 4, "num_action_classes": 12},
                "config": {"teacher_config": {"env": {"dt": 0.1}}},
            }
        )
    with pytest.raises(SystemExit) as error:
        main(["--checkpoint", "student.pt", "--output-dir", str(tmp_path), "--k", "5"])
    assert error.value.code == 2
    with pytest.raises(SystemExit) as error:
        main(
            [
                "--checkpoint",
                str(tmp_path / "missing-checkpoint.pt"),
                "--output-dir",
                str(tmp_path / "missing-output"),
                "--k",
                "1",
            ]
        )
    assert error.value.code == 2

    adapter = ChunkPolicyAdapter(InvalidShapeStudent(), execution_k=1)
    with pytest.raises(ValueError, match="4|slots|shape"):
        adapter.forward_eval(torch.zeros((1, 3)))


def test_run_eval_rollout_resets_done_agent_chunk_cache_before_next_action(monkeypatch):
    import pufferlib.pufferl as pufferl

    class FakeProgress:
        def __init__(self, **kwargs):
            pass

        def update(self, amount):
            pass

        def close(self):
            pass

    class FakeVectorEnv:
        agents_per_batch = 2
        single_action_space = pufferl.pufferlib.spaces.Box(
            low=-1.0, high=1.0, shape=(2,), dtype=np.float32
        )
        action_space = type(
            "BatchedBox",
            (),
            {"shape": (2, 2), "low": np.full((2, 2), -1.0), "high": np.full((2, 2), 1.0)},
        )()

        def __init__(self):
            self.step_idx = 0

        def reset(self, seed):
            return np.array([[10.0, 0.0, 0.0], [20.0, 0.0, 0.0]], dtype=np.float32), {}

        def step(self, actions):
            self.step_idx += 1
            next_observations = np.array([[30.0, 0.0, 0.0], [40.0, 0.0, 0.0]], dtype=np.float32)
            if self.step_idx == 1:
                terminals = np.array([True, False])
                infos = [{}]
            else:
                terminals = np.array([False, False])
                infos = [[{"summary_type": "evaluation_episode"}]]
            return next_observations, np.zeros(2), terminals, np.zeros(2, dtype=bool), infos

        def close(self):
            pass

    vector_env = FakeVectorEnv()
    student = ObservationEncodedChunkStudent()
    policy = ResetTrackingChunkPolicyAdapter(student, execution_k=2)
    monkeypatch.setattr(
        pufferl.importlib,
        "import_module",
        lambda name: SimpleNamespace(env_creator=lambda env_name: object()),
    )
    monkeypatch.setattr(pufferl.pufferlib.vector, "make", lambda *args, **kwargs: vector_env)
    monkeypatch.setattr(pufferl, "tqdm", FakeProgress)

    summaries = pufferl._run_eval_rollout(
        args={
            "package": "ocean",
            "vec": {"seed": 42},
            "train": {
                "seed": 42,
                "compile": False,
                "device": "cpu",
                "amp": False,
                "precision": "bfloat16",
                "use_rnn": False,
            },
            "eval": {"action_selection": "mean"},
        },
        env_name="fake_carla",
        worker_env_kwargs=[{"resample_frequency": 13, "num_eval_scenarios": 2}],
        total_steps=2,
        desc="test",
        expected_episodes=1,
        policy=policy,
    )

    assert len(summaries) == 1
    assert policy.reset_masks[0] is None
    np.testing.assert_array_equal(policy.reset_masks[1], [True, False])
    assert [batch.shape[0] for batch in student.observation_batches] == [2, 1]
    assert student.observation_batches[1][0, 0].item() == 30.0
