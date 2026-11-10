"""Focused CPU checks for the bounded profiling harness helpers."""

import copy
from types import SimpleNamespace

import pytest
import torch
from torch import nn
import yaml

from pufferlib.sac_replay import TensorReplay
from project.PPO_SAC_Profiling import profile_training as harness


def test_null_profile_is_iterable_and_all_profiler_hooks_are_noops():
    profile = harness.NullProfile()

    # Learner mean_and_log() unpacks the profiler as (name, measurements) pairs.
    assert dict(profile) == {}
    assert profile("collection", 3) is None
    assert profile("optimization", 3, nest=True) is None
    assert profile.end() is None
    assert profile.clear() is None
    assert dict(profile) == {}


def test_replay_instrumentation_tracks_real_add_replace_and_batch_data():
    replay = TensorReplay(capacity=4, observation_shape=(2,), device="cpu")
    counters = harness.instrument_replay(replay)

    observations = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    actions = torch.tensor([0, 1], dtype=torch.int64)
    rewards = torch.tensor([0.1, -0.2])
    next_observations = torch.tensor([[2.0, 3.0], [4.0, 5.0]])
    discounts = torch.tensor([0.9, 0.0])
    replay.add(observations, actions, rewards, next_observations, discounts)

    replacement = (
        torch.tensor([[7.0, 8.0]]),
        torch.tensor([2], dtype=torch.int64),
        torch.tensor([0.3]),
        torch.tensor([[8.0, 9.0]]),
        torch.tensor([0.5]),
    )
    replay.replace(torch.tensor([1]), *replacement)

    batch = replay.get_batch(torch.tensor([0, 1]))
    expected = (
        torch.tensor([[1.0, 2.0], [7.0, 8.0]]),
        torch.tensor([0, 2], dtype=torch.int64),
        torch.tensor([0.1, 0.3]),
        torch.tensor([[2.0, 3.0], [8.0, 9.0]]),
        torch.tensor([0.9, 0.5]),
    )

    assert counters == {
        "get_batch_calls": 1,
        "get_batch_bytes": 64,
        "inserted_rows": 3,
    }
    assert len(replay) == 2
    for actual, expected_tensor in zip(batch, expected):
        torch.testing.assert_close(actual, expected_tensor, rtol=0, atol=0)


@pytest.mark.parametrize("algorithm", ["ppo", "sac"])
def test_prepare_config_preserves_saved_model_and_environment_settings(algorithm, tmp_path):
    source_config, source_path = harness.load_source_config(algorithm)
    source_bytes = source_path.read_bytes()
    source_snapshot = yaml.safe_load(source_bytes)
    output_dir = tmp_path / f"{algorithm}-profile"
    args = harness.parse_args([
        "--algorithm", algorithm,
        "--output-dir", str(output_dir),
        "--device", "cpu",
        "--num-envs", "4",
        "--agents", "17",
        "--horizon", "4",
    ])

    config, resolved_source = harness.prepare_config(args, output_dir)

    expected_env = copy.deepcopy(source_snapshot["env"])
    expected_env["num_agents"] = args.agents
    if algorithm == "sac":
        expected_env["capture_final_observations"] = True
    expected_vec = copy.deepcopy(source_snapshot["vec"])
    expected_vec.update(num_envs=4, batch_size=2, seed=args.seed)

    assert resolved_source == source_path
    assert config["env"] == expected_env
    assert config["vec"] == expected_vec
    assert config["policy_name"] == source_snapshot["policy_name"]
    assert config["policy"] == source_snapshot["policy"]
    assert config["train"]["batch_size"] == (
        args.replay_capacity if algorithm == "sac" else args.num_envs * args.agents * args.horizon
    )
    # Both saved experiments use 20 environments grouped in batches of 10.
    assert source_config["vec"]["num_envs"] == 20
    assert source_config["vec"]["batch_size"] == 10
    assert source_path.read_bytes() == source_bytes
    assert not output_dir.exists()


def test_existing_output_directory_is_rejected_before_benchmark_start(tmp_path):
    output_dir = tmp_path / "already-used"
    output_dir.mkdir()
    marker = output_dir / "keep.txt"
    marker.write_text("existing profile data\n")

    with pytest.raises(harness.ProfileSetupError, match="output directory already exists"):
        harness.main([
            "--algorithm", "ppo",
            "--output-dir", str(output_dir),
            "--device", "cpu",
        ])

    assert marker.read_text() == "existing profile data\n"
    assert sorted(path.name for path in output_dir.iterdir()) == ["keep.txt"]


def test_torch_profiler_cycle_is_excluded_from_measurement():
    included = harness.measurement_included

    assert included(cycle_index=0, warmup=1, trace_cycle=None) is False
    assert included(cycle_index=1, warmup=1, trace_cycle=None) is True
    assert included(cycle_index=2, warmup=1, trace_cycle=2) is False
    assert included(cycle_index=3, warmup=1, trace_cycle=2) is True


def test_torch_profiler_trace_output_is_unique_per_rank(tmp_path):
    trace_path = harness.trace_output_path(tmp_path, cycle_index=2, rank=1)

    assert trace_path == tmp_path / "trace_cycle_002_rank01.json"
    assert harness.trace_output_path(tmp_path, cycle_index=2, rank=0) != trace_path


def test_fixed_target_audit_reuses_precompute_chunk_context_and_tail_indices():
    row_count = 11
    replay = TensorReplay(row_count, (1,), device="cpu")
    observations = torch.arange(row_count, dtype=torch.float32).reshape(-1, 1)
    next_observations = observations + 100
    rewards = observations[:, 0] / 10
    discounts = torch.full((row_count,), 0.9)
    replay.add(
        observations,
        torch.zeros(row_count, dtype=torch.int64),
        rewards,
        next_observations,
        discounts,
    )
    batch_contexts = []

    def compute_targets(chunk_rewards, chunk_next_observations, chunk_discounts):
        batch_contexts.append({
            "next_observations": chunk_next_observations.clone(),
            "row_count": chunk_rewards.numel(),
        })
        # Depend on batch composition so a row-wise or differently sized audit
        # forward cannot accidentally appear equal to the original cache.
        batch_context = chunk_next_observations[:, 0].mean() / 100
        next_values = chunk_next_observations[:, 0] + batch_context
        return chunk_rewards + chunk_discounts * next_values

    precompute_batch_size = 4
    fixed_cycle_targets = torch.empty(row_count)
    for chunk_start in range(0, row_count, precompute_batch_size):
        chunk_end = min(chunk_start + precompute_batch_size, row_count)
        fixed_cycle_targets[chunk_start:chunk_end] = compute_targets(
            replay.rewards[chunk_start:chunk_end],
            replay.next_observations[chunk_start:chunk_end],
            replay.discounts[chunk_start:chunk_end],
        )
    batch_contexts.clear()

    trainer = SimpleNamespace(
        replay=replay,
        config={"return_estimator": "n_step"},
        fixed_cycle_targets=fixed_cycle_targets,
        max_minibatch_size=precompute_batch_size,
        device=torch.device("cpu"),
        _compute_targets=compute_targets,
    )
    audit = harness.audit_fixed_targets(trainer, requested_rows=5, torch=torch)

    assert [context["row_count"] for context in batch_contexts] == [4, 4, 3]
    assert [context["next_observations"][:, 0].tolist() for context in batch_contexts] == [
        [100.0, 101.0, 102.0, 103.0],
        [104.0, 105.0, 106.0, 107.0],
        [108.0, 109.0, 110.0],
    ]
    assert audit["rows"] == 5
    assert audit["precompute_batch_size"] == 4
    assert audit["forward_rows_for_audit"] == row_count
    assert audit["matching_precompute_batch_context"] is True
    assert audit["mean_absolute_difference"] == 0
    assert audit["max_absolute_difference"] == 0
    assert audit["all_finite"] is True


def _set_parameters(module, value):
    with torch.no_grad():
        for parameter_index, parameter in enumerate(module.parameters()):
            parameter.fill_(value + parameter_index)


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


def _make_learner_stub():
    torch.manual_seed(11)
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

    trainer = SimpleNamespace(
        uncompiled_policy=policy,
        uncompiled_critic=critic,
        uncompiled_target_critic=target_critic,
        log_alpha=log_alpha,
        replay={"rows": torch.tensor([4, 5, 6])},
        epoch=7,
        agent_steps=101,
        learner_updates=13,
        sampled_transitions=29,
        **optimizers,
    )
    return trainer


def test_sac_weights_load_into_three_models_without_restoring_training_state(tmp_path):
    trainer = _make_learner_stub()
    source_modules = (nn.Linear(2, 3), nn.Linear(2, 3), nn.Linear(2, 3))
    for value, module in enumerate(source_modules, start=1):
        _set_parameters(module, value * 10)
    source_states = [copy.deepcopy(module.state_dict()) for module in source_modules]
    weights_path = tmp_path / "trainer_state.pt"
    torch.save({
        "algorithm": "sac",
        "scope": "learner_only",
        "policy_state_dict": source_states[0],
        "critic_state_dict": source_states[1],
        "target_critic_state_dict": source_states[2],
        # These full-checkpoint fields are deliberately present and must be ignored.
        "optimizer_state_dict": {"unexpected": torch.tensor([999])},
        "critic_optimizer_state_dict": {"unexpected": torch.tensor([999])},
        "alpha_optimizer_state_dict": {"unexpected": torch.tensor([999])},
        "epoch": 999,
        "agent_steps": 999,
        "learner_updates": 999,
        "sampled_transitions": 999,
        "log_alpha": torch.tensor(7.0),
        "replay_state": {"rows": torch.tensor([999])},
    }, weights_path)

    optimizer_states = {
        name: copy.deepcopy(getattr(trainer, name).state_dict())
        for name in ("optimizer", "critic_optimizer", "alpha_optimizer")
    }
    replay_object = trainer.replay
    replay_state = copy.deepcopy(trainer.replay)
    counter_values = {
        name: getattr(trainer, name)
        for name in ("epoch", "agent_steps", "learner_updates", "sampled_transitions")
    }
    log_alpha = trainer.log_alpha.detach().clone()
    random_state = torch.random.get_rng_state().clone()

    metadata = harness.load_profile_weights(
        trainer, "sac", weights_path, torch, SimpleNamespace(),
    )

    for module, expected_state in zip(
        (trainer.uncompiled_policy, trainer.uncompiled_critic, trainer.uncompiled_target_critic),
        source_states,
    ):
        _assert_nested_equal(module.state_dict(), expected_state)
    for name, expected_state in optimizer_states.items():
        _assert_nested_equal(getattr(trainer, name).state_dict(), expected_state)
    assert trainer.replay is replay_object
    _assert_nested_equal(trainer.replay, replay_state)
    assert {
        name: getattr(trainer, name) for name in counter_values
    } == counter_values
    torch.testing.assert_close(trainer.log_alpha, log_alpha, rtol=0, atol=0)
    torch.testing.assert_close(torch.random.get_rng_state(), random_state, rtol=0, atol=0)
    assert metadata["model_state_keys_loaded"] == [
        "policy_state_dict", "critic_state_dict", "target_critic_state_dict",
    ]
    assert metadata["source_format"] == "sac_learner_only_trainer_state"
    assert metadata["source_sha256"] == harness.sha256_file(weights_path)
    assert metadata["optimizer_state_restored"] is False
    assert metadata["training_counters_restored"] is False
    assert metadata["rng_state_restored"] is False
    assert metadata["replay_restored"] is False
    assert metadata["log_alpha_restored"] is False


def test_ppo_weights_load_only_policy_parameters(tmp_path):
    trainer = _make_learner_stub()
    source_policy = nn.Linear(2, 3)
    _set_parameters(source_policy, 42)
    source_state = copy.deepcopy(source_policy.state_dict())
    weights_path = tmp_path / "policy.pt"
    torch.save(source_state, weights_path)
    optimizer_state = copy.deepcopy(trainer.optimizer.state_dict())
    replay_object = trainer.replay
    counters = (trainer.epoch, trainer.agent_steps, trainer.learner_updates, trainer.sampled_transitions)
    torch_state = torch.random.get_rng_state().clone()
    ppo = SimpleNamespace(clean_policy_state_dict=lambda state: state)

    metadata = harness.load_profile_weights(trainer, "ppo", weights_path, torch, ppo)

    _assert_nested_equal(trainer.uncompiled_policy.state_dict(), source_state)
    _assert_nested_equal(trainer.optimizer.state_dict(), optimizer_state)
    assert trainer.replay is replay_object
    assert (trainer.epoch, trainer.agent_steps, trainer.learner_updates, trainer.sampled_transitions) == counters
    torch.testing.assert_close(torch.random.get_rng_state(), torch_state, rtol=0, atol=0)
    assert metadata["model_state_keys_loaded"] == ["policy_state_dict"]
    assert metadata["source_format"] == "ppo_policy_state_dict"
    assert metadata["optimizer_state_restored"] is False
    assert metadata["replay_restored"] is False


@pytest.mark.parametrize("algorithm", ["ppo", "sac"])
def test_weights_loader_rejects_incompatible_model_keys(tmp_path, algorithm):
    trainer = _make_learner_stub()
    invalid_state = {"unexpected_key": torch.ones(2)}
    weights_path = tmp_path / f"invalid-{algorithm}.pt"
    if algorithm == "ppo":
        torch.save(invalid_state, weights_path)
        ppo = SimpleNamespace(clean_policy_state_dict=lambda state: state)
    else:
        torch.save({
            "algorithm": "sac",
            "scope": "learner_only",
            "policy_state_dict": invalid_state,
            "critic_state_dict": trainer.uncompiled_critic.state_dict(),
            "target_critic_state_dict": trainer.uncompiled_target_critic.state_dict(),
        }, weights_path)
        ppo = SimpleNamespace()

    with pytest.raises(RuntimeError):
        harness.load_profile_weights(trainer, algorithm, weights_path, torch, ppo)
