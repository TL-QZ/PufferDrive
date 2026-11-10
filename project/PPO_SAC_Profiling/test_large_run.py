"""CPU-only checks for large-run budget and source-config controls."""

import copy
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import yaml

from project.PPO_SAC_Profiling import profile_training as harness


def _parse_sac_args(output_dir, *extra):
    return harness.parse_args([
        "--algorithm", "sac",
        "--output-dir", str(output_dir),
        "--device", "cpu",
        *extra,
    ])


def test_large_run_controls_parse_with_sac_only_positive_budget(tmp_path):
    source_config = harness.SAC_CONFIG
    args = _parse_sac_args(
        tmp_path / "profile",
        "--source-config", str(source_config),
        "--total-agent-steps", "12345",
        "--save-learner-state", "on",
        "--snapshot-interval-cycles", "7",
    )

    assert args.source_config == source_config.resolve()
    assert args.total_agent_steps == 12345
    assert args.save_learner_state == "on"
    assert args.snapshot_interval_cycles == 7


@pytest.mark.parametrize(
    "extra",
    [
        ("--total-agent-steps", "0"),
        ("--total-agent-steps", "-1"),
        ("--snapshot-interval-cycles", "0"),
    ],
)
def test_large_run_rejects_nonpositive_step_or_snapshot_intervals(tmp_path, extra):
    with pytest.raises(SystemExit):
        _parse_sac_args(tmp_path / "profile", *extra)


@pytest.mark.parametrize(
    "extra",
    [
        ("--total-agent-steps", "123"),
        ("--save-learner-state", "on"),
    ],
)
def test_large_run_sac_only_controls_reject_ppo(tmp_path, extra):
    with pytest.raises(SystemExit):
        harness.parse_args([
            "--algorithm", "ppo",
            "--output-dir", str(tmp_path / "ppo-profile"),
            "--device", "cpu",
            *extra,
        ])


def test_default_large_run_budget_and_snapshot_policy_preserve_existing_profile(tmp_path):
    args = _parse_sac_args(tmp_path / "profile")

    assert args.total_agent_steps is None
    assert args.save_learner_state == "off"
    assert args.snapshot_interval_cycles == 25
    assert harness.resolve_total_timesteps(args) == 1_000_000_000_000_000

    config, source_path = harness.prepare_config(args, args.output_dir)

    assert config["train"]["total_timesteps"] == 1_000_000_000_000_000
    assert source_path == harness.SAC_CONFIG.resolve()
    assert not args.output_dir.exists()


def test_source_config_is_copied_and_requested_budget_propagates(tmp_path):
    original_config, _ = harness.load_source_config("sac")
    custom_config = copy.deepcopy(original_config)
    custom_config["env"]["reward_goal"] = 1.2345
    custom_config["policy"]["actor_hidden_size"] += 1
    custom_config["train"]["target_entropy_force_decay_steps"] = 9876
    source_path = tmp_path / "source_sac.yaml"
    source_path.write_text(yaml.safe_dump(custom_config, sort_keys=False))
    source_bytes = source_path.read_bytes()

    args = _parse_sac_args(
        tmp_path / "profile",
        "--source-config", str(source_path),
        "--total-agent-steps", "54321",
    )
    config, resolved_source = harness.prepare_config(args, args.output_dir)

    assert resolved_source == source_path.resolve()
    assert config["train"]["total_timesteps"] == 54321
    assert config["env"]["reward_goal"] == 1.2345
    assert config["policy"]["actor_hidden_size"] == custom_config["policy"]["actor_hidden_size"]
    assert config["train"]["target_entropy_force_decay_steps"] == 9876
    assert source_path.read_bytes() == source_bytes
    assert not args.output_dir.exists()


@pytest.mark.parametrize(
    ("preset_name", "save_state", "snapshot_interval"),
    [
        ("optimized.yaml", "on", 25),
        ("stock_baseline.yaml", "on", 25),
    ],
)
def test_large_run_presets_propagate_source_budget_and_snapshot_controls(
    tmp_path, preset_name, save_state, snapshot_interval,
):
    preset_path = Path(__file__).parent / "large_run_1b" / preset_name
    args = harness.parse_args([
        "--profile-config", str(preset_path),
        "--output-dir", str(tmp_path / "profile"),
    ])

    assert args.algorithm == "sac"
    assert args.source_config == (
        Path(__file__).resolve().parents[2]
        / "project/PPO_SAC_Profiling/large_run_1b/source_sac.yaml"
    ).resolve()
    assert args.total_agent_steps == 1_000_000_000
    assert args.save_learner_state == save_state
    assert args.snapshot_interval_cycles == snapshot_interval


def test_progress_record_allows_cycle_cap_before_budget():
    progress = harness.build_progress_record(
        status="cycle_limit",
        algorithm="sac",
        cycle_index=3,
        cycles_completed=4,
        cycle_cap=4,
        budget_agent_steps=1_000,
        global_agent_steps=320,
        cumulative_received_rows=320,
        cumulative_retained_rows=240,
        cumulative_candidate_rows=320,
        elapsed_seconds=10,
        collection_seconds=6,
        optimization_seconds=4,
        checkpoint_seconds=0.5,
        stop_reason="cycle_limit",
    )

    assert progress["budget_agent_steps"] == 1_000
    assert progress["global_agent_steps"] == 320
    assert progress["budget_reached"] is False
    assert progress["cycles_completed"] == progress["cycle_cap"] == 4
    assert progress["stop_reason"] == "cycle_limit"
    assert progress["cumulative_retention_fraction"] == 0.75
    assert progress["received_steps_per_second"] == 32
    assert progress["eta_seconds"] == 21.25


def _assert_nested_equal(actual, expected):
    if isinstance(expected, torch.Tensor):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    elif isinstance(expected, dict):
        assert actual.keys() == expected.keys()
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
    elif isinstance(expected, (tuple, list)):
        assert len(actual) == len(expected)
        for actual_value, expected_value in zip(actual, expected):
            _assert_nested_equal(actual_value, expected_value)
    else:
        assert actual == expected


class _SnapshotController:
    def state_dict(self):
        return {"target_entropy": 0.75, "decay_count": 3}


def _make_snapshot_trainer():
    policy = nn.Linear(2, 3)
    critic = nn.Linear(2, 3)
    target_critic = nn.Linear(2, 3)
    log_alpha = nn.Parameter(torch.tensor(-1.25))
    optimizers = {
        "optimizer": torch.optim.Adam(policy.parameters(), lr=0.01),
        "critic_optimizer": torch.optim.Adam(critic.parameters(), lr=0.02),
        "alpha_optimizer": torch.optim.Adam([log_alpha], lr=0.03),
    }
    for optimizer in optimizers.values():
        parameters = [parameter for group in optimizer.param_groups for parameter in group["params"]]
        sum(parameter.square().sum() for parameter in parameters).backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

    schedulers = {
        "scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(optimizers["optimizer"], T_max=9),
        "critic_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizers["critic_optimizer"], T_max=9,
        ),
        "alpha_scheduler": torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizers["alpha_optimizer"], T_max=9,
        ),
    }

    trainer = SimpleNamespace(
        uncompiled_policy=policy,
        uncompiled_critic=critic,
        uncompiled_target_critic=target_critic,
        log_alpha=log_alpha,
        target_entropy_controller=_SnapshotController(),
        config={
            "total_timesteps": 123456,
            "target_entropy_schedule": "adaptive",
            "profile_filter_score": "target_advantage",
            "profile_filter_mode": "ewma_threshold",
            "profile_filter_ewma_beta": 0.25,
            "profile_filter_threshold_scale": 0.01,
        },
        helpers=SimpleNamespace(capture_rng_state=lambda: {
            "torch": torch.random.get_rng_state(),
            "marker": "captured-before-save",
        }),
        replay={"must_not_be_serialized": True},
        global_step=1024,
        agent_steps=1024,
        epoch=17,
        valid_inserts=640,
        learner_updates=23,
        sampled_transitions=8192,
        best_score=1.25,
        filter_current_max=0.9,
        filter_ema_max=0.7,
        filter_threshold=0.007,
        filter_retained_before_sync=512,
        filter_retained_rows=480,
        filter_candidate_rows=640,
        **optimizers,
        **schedulers,
    )
    return trainer


def test_final_learner_snapshot_is_atomic_portable_and_explicitly_not_resumable(
    monkeypatch, tmp_path,
):
    trainer = _make_snapshot_trainer()
    output_dir = tmp_path / "snapshot-output"
    source_states = {
        name: copy.deepcopy(getattr(trainer, name).state_dict())
        for name in (
            "uncompiled_policy", "uncompiled_critic", "uncompiled_target_critic",
            "optimizer", "critic_optimizer", "alpha_optimizer",
            "scheduler", "critic_scheduler", "alpha_scheduler",
        )
    }
    log_alpha = trainer.log_alpha.detach().clone()
    controller_state = trainer.target_entropy_controller.state_dict()
    rng_state = trainer.helpers.capture_rng_state()
    original_replace = harness.os.replace
    atomic_replacements = []

    def observe_atomic_replace(source, destination):
        source = Path(source)
        destination = Path(destination)
        assert source.is_file()
        assert source.name.endswith(".tmp")
        assert not destination.exists()
        atomic_replacements.append((source, destination))
        original_replace(source, destination)

    monkeypatch.setattr(harness.os, "replace", observe_atomic_replace)
    metadata = {"stop_reason": "budget_reached", "budget_agent_steps": 1024}
    result = harness.save_learner_snapshot(
        trainer, output_dir, torch, False, 0, 1, 3,
        final=True, metadata=metadata,
    )

    learner_path = output_dir / "learner_state_final.pt"
    actor_path = output_dir / "final_model.pt"
    assert result["learner_state_path"] == str(learner_path)
    assert result["actor_artifact_path"] == str(actor_path)
    assert result["learner_state_bytes"] == learner_path.stat().st_size > 0
    assert result["actor_artifact_bytes"] == actor_path.stat().st_size > 0
    assert result["checkpoint_seconds"] >= 0
    assert len(atomic_replacements) == 2
    assert all(not source.exists() for source, _destination in atomic_replacements)
    assert not list(output_dir.rglob("*.tmp"))

    snapshot = torch.load(learner_path, map_location="cpu", weights_only=False)
    actor_artifact = torch.load(actor_path, map_location="cpu", weights_only=False)
    assert snapshot["algorithm"] == "sac"
    assert snapshot["scope"] == "profiling_learner_snapshot"
    assert snapshot["resume_supported"] is False
    assert snapshot["snapshot_metadata"] == metadata
    assert snapshot["config"] == trainer.config
    assert "replay" not in snapshot
    assert "environment" not in snapshot
    assert snapshot["excluded_state"] == [
        "replay", "environment", "pending_transitions", "worker_processes",
    ]

    for payload_name, trainer_name in (
        ("policy_state_dict", "uncompiled_policy"),
        ("critic_state_dict", "uncompiled_critic"),
        ("target_critic_state_dict", "uncompiled_target_critic"),
        ("optimizer_state_dict", "optimizer"),
        ("critic_optimizer_state_dict", "critic_optimizer"),
        ("alpha_optimizer_state_dict", "alpha_optimizer"),
        ("scheduler_state_dict", "scheduler"),
        ("critic_scheduler_state_dict", "critic_scheduler"),
        ("alpha_scheduler_state_dict", "alpha_scheduler"),
    ):
        _assert_nested_equal(snapshot[payload_name], source_states[trainer_name])
    _assert_nested_equal(actor_artifact, source_states["uncompiled_policy"])
    _assert_nested_equal(snapshot["log_alpha"], log_alpha)
    _assert_nested_equal(snapshot["target_entropy_controller_state_dict"], controller_state)
    assert snapshot["counters"] == {
        "global_step": 1024,
        "agent_steps": 1024,
        "epoch": 17,
        "valid_inserts": 640,
        "learner_updates": 23,
        "sampled_transitions": 8192,
        "best_score": 1.25,
    }
    assert len(snapshot["rank_local_state"]) == 1
    rank_state = snapshot["rank_local_state"][0]
    assert rank_state["rank"] == 0
    assert rank_state["filter_score_state"] == {
        "mode": "ewma_threshold",
        "score": "target_advantage",
        "ewma_beta": 0.25,
        "threshold_scale": 0.01,
        "current_max": 0.9,
        "ema_max": 0.7,
        "threshold": 0.007,
        "retained_before_sync": 512,
        "retained_rows": 480,
        "candidate_rows": 640,
    }
    _assert_nested_equal(rank_state["rng_state"], rng_state)
