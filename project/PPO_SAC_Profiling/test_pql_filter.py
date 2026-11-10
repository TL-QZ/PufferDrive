"""Semantic checks for fixed PQL targets and cycle-local SAC row filtering."""

from collections import Counter, defaultdict
from datetime import timedelta
import math
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from pufferlib.sac_sequence_replay import SequenceReplay
import project.PPO_SAC_Profiling.sac_variants as sac_variants
from project.PPO_SAC_Profiling.sac_variants import install_variant
from project.PPO_SAC_Profiling.test_sac_variants import (
    _TinyActor,
    _TinyCritic,
    _assert_learner_state_equal,
    _make_learner,
)


FIXED_VARIANTS = ("fixed_targets", "fixed_targets_cached", "fixed_targets_actor_q")
_OBSERVATION_SHAPE = (3,)


def _make_boundary_replay(*, device="cpu", pin_memory=False, observation_offset=0.0):
    """Build rows with terminal, truncation, and explicit gap boundaries."""
    replay = SequenceReplay(
        11,
        _OBSERVATION_SHAPE,
        num_agents=3,
        device=device,
        pin_memory=pin_memory,
    )
    segment_lengths = (4, 4, 3)
    for time_idx in range(max(segment_lengths)):
        if time_idx == 2:
            replay.break_chains(torch.tensor([2]))
        agent_ids = [
            agent_id for agent_id, length in enumerate(segment_lengths)
            if time_idx < length
        ]
        observations = torch.tensor([
            [
                agent_id + 0.17 + observation_offset,
                time_idx + 0.29,
                10 * agent_id + time_idx + 0.13,
            ]
            for agent_id in agent_ids
        ])
        terminals = torch.tensor([
            agent_id == 0 and time_idx == 1 for agent_id in agent_ids
        ])
        truncations = torch.tensor([
            agent_id == 1 and time_idx == 2 for agent_id in agent_ids
        ])
        replay.add_sequence(
            torch.tensor(agent_ids),
            observations,
            torch.tensor([(agent_id + time_idx) % 3 for agent_id in agent_ids]),
            torch.tensor([
                0.11 * (agent_id + 1) + 0.037 * time_idx + observation_offset
                for agent_id in agent_ids
            ]),
            observations * 0.7 + 0.21,
            torch.full((len(agent_ids),), 0.93),
            terminals,
            truncations,
        )
    return replay


def _make_overwritten_replay(*, device="cpu", pin_memory=False):
    """Build one FIFO chain whose oldest two rows were overwritten."""
    replay = SequenceReplay(
        8,
        _OBSERVATION_SHAPE,
        num_agents=1,
        device=device,
        pin_memory=pin_memory,
    )
    for time_idx in range(10):
        observations = torch.tensor([[
            0.17,
            time_idx + 0.29,
            time_idx + 0.13,
        ]])
        replay.add_sequence(
            torch.tensor([0]),
            observations,
            torch.tensor([time_idx % 3]),
            torch.tensor([0.2 + 0.03 * time_idx]),
            observations * 0.7 + 0.21,
            torch.tensor([0.91]),
            torch.tensor([False]),
            torch.tensor([False]),
        )
    return replay


def _make_pql_learner(
        *, seed=123, update_epochs=3, trace_lambda=0.73, device="cpu",
        replay_device=None, pin_memory=False, observation_offset=0.0,
        replay_kind="boundaries", filter_score="none", keep_fraction=1.0):
    torch.manual_seed(seed)
    actor_state = _TinyActor().state_dict()
    critic_state = _TinyCritic().state_dict()
    learner = _make_learner(
        actor_state,
        critic_state,
        update_epochs=update_epochs,
        device=device,
        pin_memory=pin_memory,
    )
    actual_replay_device = device if replay_device is None else replay_device
    if replay_kind == "boundaries":
        replay = _make_boundary_replay(
            device=actual_replay_device,
            pin_memory=pin_memory,
            observation_offset=observation_offset,
        )
    elif replay_kind == "overwritten":
        replay = _make_overwritten_replay(
            device=actual_replay_device,
            pin_memory=pin_memory,
        )
    else:
        raise AssertionError(f"unknown replay fixture {replay_kind!r}")
    learner.replay = replay
    learner.batch_size = len(replay)
    learner.minibatch_size = 4
    learner.max_minibatch_size = 3
    learner.config.update({
        "return_estimator": "pql",
        "pql_horizon": 3,
        "pql_lambda": trace_lambda,
        "profile_filter_score": filter_score,
        "profile_keep_fraction": keep_fraction,
    })
    return learner


def _replay_field(learner, name):
    return getattr(learner.replay._replay, name)


def _manual_pql_targets(learner, replay_indices, segment_offsets):
    """Independent scalar backward recursion over each saved sequence segment."""
    replay_indices = torch.as_tensor(replay_indices, dtype=torch.int64, device="cpu")
    segment_offsets = torch.as_tensor(segment_offsets, dtype=torch.int64, device="cpu")
    device = learner.device
    next_observations = _replay_field(learner, "next_observations")[replay_indices]
    next_observations = next_observations.to(device)
    with torch.no_grad():
        logits = learner.policy(next_observations).float()
        target_q1, target_q2 = learner.target_critic(next_observations)
    rewards = _replay_field(learner, "rewards")[replay_indices].detach().cpu().tolist()
    discounts = _replay_field(learner, "discounts")[replay_indices].detach().cpu().tolist()
    alpha = float(learner.alpha.detach().cpu())
    next_values = []
    for row_logits, row_q1, row_q2 in zip(
            logits.detach().cpu().tolist(),
            target_q1.detach().cpu().tolist(),
            target_q2.detach().cpu().tolist()):
        maximum_logit = max(row_logits)
        exponentials = [math.exp(value - maximum_logit) for value in row_logits]
        normalizer = sum(exponentials)
        probabilities = [value / normalizer for value in exponentials]
        log_probabilities = [
            value - maximum_logit - math.log(normalizer) for value in row_logits
        ]
        soft_value = 0.0
        for action_idx, probability in enumerate(probabilities):
            minimum_q = min(row_q1[action_idx], row_q2[action_idx])
            if learner.config["critic_entropy_in_target"]:
                minimum_q -= alpha * log_probabilities[action_idx]
            soft_value += probability * minimum_q
        next_values.append(soft_value)

    trace_lambda = learner.config["pql_lambda"]
    targets = [0.0] * replay_indices.numel()
    offsets = segment_offsets.tolist()
    for segment_start, segment_end in zip(offsets[:-1], offsets[1:]):
        successor_target = None
        for row_idx in range(segment_end - 1, segment_start - 1, -1):
            if successor_target is None:
                target = rewards[row_idx] + discounts[row_idx] * next_values[row_idx]
            else:
                target = rewards[row_idx] + discounts[row_idx] * (
                    (1.0 - trace_lambda) * next_values[row_idx]
                    + trace_lambda * successor_target
                )
            targets[row_idx] = target
            successor_target = target
    return torch.tensor(targets, dtype=torch.float32, device=device)


def _plan_target_cache(learner, sequence_batches):
    physical_targets = torch.full(
        (learner.replay.capacity,), float("nan"), dtype=torch.float32, device=learner.device,
    )
    coverage = torch.zeros(learner.replay.capacity, dtype=torch.int64)
    for replay_indices, segment_offsets in sequence_batches:
        batch_targets = _manual_pql_targets(learner, replay_indices, segment_offsets)
        physical_targets[replay_indices.to(learner.device)] = batch_targets
        coverage[replay_indices] += 1
    assert torch.equal(coverage, learner.replay._valid.to(torch.int64))
    return physical_targets


def _expected_cycle_start_q(learner, replay_indices=None):
    if replay_indices is None:
        replay_indices = torch.nonzero(learner.replay._valid, as_tuple=False).flatten()
    observations = _replay_field(learner, "observations")[replay_indices].to(learner.device)
    with torch.no_grad():
        q1, q2 = learner.critic(observations)
    expected = torch.zeros(
        (learner.replay.capacity, q1.shape[-1]), dtype=torch.float32, device=learner.device,
    )
    expected[replay_indices.to(learner.device)] = torch.minimum(q1.float(), q2.float())
    return expected


def _expected_filter_scores(learner, score_name, targets, minimum_q):
    replay_indices = torch.nonzero(learner.replay._valid, as_tuple=False).flatten()
    observations = _replay_field(learner, "observations")[replay_indices].to(learner.device)
    actions = _replay_field(learner, "actions")[replay_indices].to(learner.device)
    with torch.no_grad():
        logits = learner.policy(observations).float()
        probabilities = torch.softmax(logits, dim=-1)
    rows = replay_indices.to(learner.device)
    row_q = minimum_q.index_select(0, rows)
    baseline = (probabilities * row_q).sum(-1)
    if score_name == "q_advantage":
        recorded_q = row_q.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        return (recorded_q - baseline).abs()
    if score_name == "target_advantage":
        return (targets.index_select(0, rows) - baseline).abs()
    assert score_name == "expected_abs_advantage"
    return (probabilities * (row_q - baseline.unsqueeze(-1)).abs()).sum(-1)


def _capture_training_rows(learner, batch_sizes=None):
    row_lookup = {
        tuple(row.tolist()): index
        for index, row in enumerate(_replay_field(learner, "observations")[:learner.replay.capacity].cpu())
        if bool(learner.replay._valid[index])
    }
    seen_rows = []
    original_loss = learner._critic_loss

    def capture(observations, actions, targets):
        if batch_sizes is not None:
            batch_sizes.append(int(observations.shape[0]))
        seen_rows.extend(row_lookup[tuple(row.tolist())] for row in observations.detach().cpu())
        return original_loss(observations, actions, targets)

    learner._critic_loss = capture
    return seen_rows


def _copy_sequence_plan(sequence_batches):
    return [
        (indices.clone(), offsets.clone())
        for indices, offsets in sequence_batches
    ]


def _assert_sequence_plan_equal(actual, expected):
    assert len(actual) == len(expected)
    for (actual_indices, actual_offsets), (expected_indices, expected_offsets) in zip(actual, expected):
        torch.testing.assert_close(actual_indices, expected_indices, rtol=0, atol=0)
        torch.testing.assert_close(actual_offsets, expected_offsets, rtol=0, atol=0)


@pytest.mark.parametrize("variant", FIXED_VARIANTS)
@pytest.mark.parametrize("trace_lambda", [0.0, 1.0])
def test_fixed_variants_cache_physical_pql_targets_from_stock_plan(
        variant, trace_lambda, monkeypatch):
    learner = _make_pql_learner(trace_lambda=trace_lambda, update_epochs=3)
    install_variant(learner, variant)

    # Build an independent reference plan and stock preparation on identical cuts.
    torch.manual_seed(811)
    stock_plan = learner.replay.build_epoch_batches(
        learner.config["pql_horizon"], learner.minibatch_size,
    )
    stock_plan = _copy_sequence_plan(stock_plan)
    stock_prepared = [
        learner._prepare_pql_targets(indices, offsets).detach().clone()
        for indices, offsets in stock_plan
    ]
    manual_cache = _plan_target_cache(learner, stock_plan)

    prepare_calls = []
    original_prepare = learner._prepare_pql_targets

    def capture_prepare(indices, offsets):
        targets = original_prepare(indices, offsets)
        prepare_calls.append((indices.clone(), offsets.clone(), targets.detach().clone()))
        return targets

    monkeypatch.setattr(learner, "_prepare_pql_targets", capture_prepare)
    microbatch_sizes = []
    training_rows = _capture_training_rows(learner, microbatch_sizes)
    torch.manual_seed(811)
    updates = learner._train_sac_transition(
        defaultdict(float), lambda *_args, **_kwargs: None, 0,
    )

    assert updates == 9  # Three epochs, with logical batches 4, 4, and a 3-row tail.
    assert learner.learner_updates == 9
    assert learner.sampled_transitions == 33
    assert learner.fixed_target_precompute_rows == len(learner.replay)
    assert learner.fixed_target_precompute_calls == 3
    assert learner.fixed_cycle_targets.requires_grad is False
    assert learner.fixed_cycle_targets.grad_fn is None
    torch.testing.assert_close(learner.fixed_cycle_targets, manual_cache, rtol=1e-6, atol=1e-7)

    _assert_sequence_plan_equal(learner.fixed_pql_sequence_batches, stock_plan)
    assert len(prepare_calls) == len(stock_plan)
    for (actual_indices, actual_offsets, actual_targets), (expected_indices, expected_offsets), expected_targets in zip(
            prepare_calls, stock_plan, stock_prepared):
        torch.testing.assert_close(actual_indices, expected_indices, rtol=0, atol=0)
        torch.testing.assert_close(actual_offsets, expected_offsets, rtol=0, atol=0)
        torch.testing.assert_close(actual_targets, expected_targets, rtol=0, atol=0)

    row_counts = Counter(training_rows)
    assert row_counts == Counter({row: 3 for row in range(len(learner.replay))})
    assert microbatch_sizes == [3, 1, 3, 1, 3] * 3
    assert learner.replay.last_plan_metrics["cut_terminal_count"] == 1
    assert learner.replay.last_plan_metrics["cut_truncation_count"] == 1
    assert learner.replay.last_plan_metrics["cut_gap_count"] == 1
    metadata = learner.replay.get_sequence_metadata(torch.arange(learner.replay.capacity))
    terminal_rows = torch.nonzero(metadata["terminals"], as_tuple=False).flatten()
    assert terminal_rows.numel() == 1
    torch.testing.assert_close(
        learner.fixed_cycle_targets[terminal_rows],
        _replay_field(learner, "rewards")[terminal_rows],
        rtol=0,
        atol=0,
    )


def test_pql_plan_cuts_fifo_overwrite_and_keeps_microbatch_and_epoch_tails():
    learner = _make_pql_learner(
        replay_kind="overwritten", trace_lambda=1.0, update_epochs=3,
    )
    install_variant(learner, "fixed_targets_actor_q")
    rows_by_time = {
        round(float(observation[1]) - 0.29): index
        for index, observation in enumerate(_replay_field(learner, "observations"))
    }
    assert set(rows_by_time) == set(range(2, 10))
    microbatch_sizes = []
    training_rows = _capture_training_rows(learner, microbatch_sizes)
    torch.manual_seed(821)
    updates = learner._train_sac_transition(
        defaultdict(float), lambda *_args, **_kwargs: None, 0,
    )

    assert updates == 6  # Three epochs, two logical updates per epoch.
    assert learner.sampled_transitions == 24
    assert learner.replay.last_plan_metrics["cut_overwrite_count"] >= 1
    assert learner.fixed_target_precompute_rows == 8
    assert learner.fixed_target_precompute_calls == 2
    assert Counter(training_rows) == Counter({row: 3 for row in range(8)})
    assert microbatch_sizes == [3, 1, 3, 1] * 3
    for indices, offsets in learner.fixed_pql_sequence_batches:
        assert indices.numel() <= learner.minibatch_size
        assert int(offsets[0]) == 0
        assert int(offsets[-1]) == indices.numel()
        assert bool(torch.all(offsets[1:] > offsets[:-1]))


@pytest.mark.parametrize(
    "score_name",
    ["q_advantage", "target_advantage", "expected_abs_advantage"],
)
def test_filter_scores_match_cycle_start_q_policy_and_fixed_pql_target_oracles(
        score_name, monkeypatch):
    learner = _make_pql_learner(
        seed=831,
        update_epochs=2,
        filter_score=score_name,
        keep_fraction=0.46,
    )
    install_variant(learner, "fixed_targets_actor_q")

    torch.manual_seed(823)
    plan = learner.replay.build_epoch_batches(
        learner.config["pql_horizon"], learner.minibatch_size,
    )
    plan = _copy_sequence_plan(plan)
    expected_targets = _plan_target_cache(learner, plan)
    expected_q = _expected_cycle_start_q(learner)
    expected_scores = _expected_filter_scores(learner, score_name, expected_targets, expected_q)
    kept_count = math.ceil(len(learner.replay) * learner.config["profile_keep_fraction"])
    expected_indices = torch.argsort(expected_scores, descending=True, stable=True)[:kept_count]
    expected_physical_indices = torch.nonzero(learner.replay._valid, as_tuple=False).flatten()[expected_indices]

    observed_alphas = []
    actor_rows = []
    critic_rows = []
    row_lookup = {
        tuple(observation.tolist()): row_idx
        for row_idx, observation in enumerate(_replay_field(learner, "observations").cpu())
    }
    original_actor_loss = sac_variants._fixed_actor_q_policy_loss
    original_critic_loss = learner._critic_loss

    def capture_actor_loss(current_learner, observations, cached_q):
        assert cached_q.requires_grad is False
        assert cached_q.grad_fn is None
        actor_rows.extend(row_lookup[tuple(row.tolist())] for row in observations.detach().cpu())
        cached_indices = torch.tensor(
            [row_lookup[tuple(row.tolist())] for row in observations.detach().cpu()],
            dtype=torch.int64,
            device=expected_q.device,
        )
        torch.testing.assert_close(cached_q, expected_q[cached_indices], rtol=0, atol=0)
        observed_alphas.append(float(current_learner.alpha.detach().cpu()))
        return original_actor_loss(current_learner, observations, cached_q)

    def capture_critic_loss(observations, actions, targets):
        critic_rows.extend(row_lookup[tuple(row.tolist())] for row in observations.detach().cpu())
        return original_critic_loss(observations, actions, targets)

    monkeypatch.setattr(sac_variants, "_fixed_actor_q_policy_loss", capture_actor_loss)
    learner._critic_loss = capture_critic_loss
    torch.manual_seed(823)
    updates = learner._train_sac_transition(
        defaultdict(float), lambda *_args, **_kwargs: None, 0,
    )

    assert updates == 4  # Six selected rows form two logical batches per epoch.
    assert learner.filter_candidate_rows == len(learner.replay)
    assert learner.filter_retained_rows == kept_count
    assert learner.fixed_filter_candidate_indices.tolist() == list(range(len(learner.replay)))
    assert set(learner.fixed_filter_indices.tolist()) == set(expected_physical_indices.tolist())
    torch.testing.assert_close(
        learner.fixed_filter_scores,
        expected_scores,
        rtol=1e-6,
        atol=1e-7,
    )
    assert learner.fixed_filter_scores.requires_grad is False
    assert learner.fixed_filter_scores.grad_fn is None
    torch.testing.assert_close(learner.fixed_cycle_targets, expected_targets, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(learner.fixed_cycle_actor_q, expected_q, rtol=0, atol=0)
    assert learner.fixed_cycle_targets.grad_fn is None
    assert learner.fixed_cycle_actor_q.grad_fn is None
    assert learner.fixed_actor_loss_rows == kept_count * learner.config["update_epochs"]
    assert learner.sampled_transitions == kept_count * learner.config["update_epochs"]
    retained_counts = Counter(learner.fixed_filter_indices.tolist())
    expected_training_counts = Counter({
        row_idx: learner.config["update_epochs"] for row_idx in retained_counts
    })
    assert Counter(actor_rows) == expected_training_counts
    assert Counter(critic_rows) == expected_training_counts
    assert learner.learner_updates == updates
    assert learner.optimizer.state and learner.critic_optimizer.state and learner.alpha_optimizer.state
    assert not torch.equal(learner.log_alpha.detach(), torch.tensor(-1.2))
    assert max(observed_alphas) - min(observed_alphas) > 1e-7


def test_filter_reduces_real_exposures_and_optimizer_steps_and_fraction_one_preserves_rng():
    unfiltered = _make_pql_learner(seed=841, update_epochs=2)
    full_fraction = _make_pql_learner(
        seed=841,
        update_epochs=2,
        filter_score="q_advantage",
        keep_fraction=1.0,
    )
    reduced = _make_pql_learner(
        seed=841,
        update_epochs=2,
        filter_score="q_advantage",
        keep_fraction=0.46,
    )
    install_variant(unfiltered, "fixed_targets_actor_q")
    install_variant(full_fraction, "fixed_targets_actor_q")
    install_variant(reduced, "fixed_targets_actor_q")

    torch.manual_seed(827)
    unfiltered_losses = defaultdict(float)
    unfiltered_updates = unfiltered._train_sac_transition(
        unfiltered_losses, lambda *_args, **_kwargs: None, 0,
    )
    torch.manual_seed(827)
    full_losses = defaultdict(float)
    full_updates = full_fraction._train_sac_transition(
        full_losses, lambda *_args, **_kwargs: None, 0,
    )
    assert full_updates == unfiltered_updates == 6
    assert dict(full_losses) == dict(unfiltered_losses)
    _assert_learner_state_equal(full_fraction, unfiltered)
    assert full_fraction.filter_candidate_rows == full_fraction.filter_retained_rows == 11

    torch.manual_seed(827)
    reduced_updates = reduced._train_sac_transition(
        defaultdict(float), lambda *_args, **_kwargs: None, 0,
    )
    assert reduced_updates == 4
    assert reduced.filter_candidate_rows == 11
    assert reduced.filter_retained_rows == 6
    assert reduced.sampled_transitions == 12
    assert unfiltered.sampled_transitions == 22
    assert reduced.learner_updates < unfiltered.learner_updates
    assert reduced.sampled_transitions < unfiltered.sampled_transitions


@pytest.mark.parametrize(
    "score_name,keep_fraction,match",
    [
        ("unknown", 1.0, "profile_filter_score"),
        ("none", 0.9, "profile_keep_fraction"),
        ("q_advantage", 0.0, "profile_keep_fraction"),
        ("q_advantage", -0.2, "profile_keep_fraction"),
        ("q_advantage", 1.01, "profile_keep_fraction"),
        ("q_advantage", float("nan"), "profile_keep_fraction"),
        ("q_advantage", float("inf"), "profile_keep_fraction"),
        ("q_advantage", True, "profile_keep_fraction"),
    ],
)
def test_invalid_filter_configuration_fails_before_updates(score_name, keep_fraction, match):
    learner = _make_pql_learner(
        filter_score=score_name,
        keep_fraction=keep_fraction,
    )
    with pytest.raises(ValueError, match=match):
        install_variant(learner, "fixed_targets_actor_q")
    assert learner.learner_updates == 0


@pytest.mark.parametrize("variant", ["fixed_targets", "fixed_targets_cached"])
def test_filtering_is_rejected_for_fixed_target_variants_without_actor_q(variant):
    learner = _make_pql_learner(filter_score="q_advantage", keep_fraction=0.5)
    with pytest.raises(ValueError, match="only by fixed_targets_actor_q"):
        install_variant(learner, variant)


def test_nonfinite_cycle_start_scores_fail_before_optimizer_updates():
    learner = _make_pql_learner(filter_score="q_advantage", keep_fraction=0.5)
    with torch.no_grad():
        learner.critic.q1.weight.fill_(float("nan"))
    install_variant(learner, "fixed_targets_actor_q")

    with pytest.raises(ValueError, match="nonfinite scores"):
        learner._train_sac_transition(
            defaultdict(float), lambda *_args, **_kwargs: None, 0,
        )
    assert learner.learner_updates == 0


def test_filter_ties_keep_a_stable_physical_row_order():
    learner = _make_pql_learner(
        filter_score="q_advantage",
        keep_fraction=0.55,
        update_epochs=1,
    )
    with torch.no_grad():
        for parameter in learner.critic.parameters():
            parameter.zero_()
    install_variant(learner, "fixed_targets_actor_q")
    learner._train_sac_transition(
        defaultdict(float), lambda *_args, **_kwargs: None, 0,
    )

    assert learner.fixed_filter_scores.tolist() == [0.0] * len(learner.replay)
    assert learner.fixed_filter_indices.tolist() == list(range(7))


def test_next_cycle_refreshes_pql_targets_and_actor_q_for_retained_and_overwritten_rows():
    learner = _make_pql_learner(seed=853, update_epochs=1)
    install_variant(learner, "fixed_targets_actor_q")
    torch.manual_seed(829)
    learner._train_sac_transition(defaultdict(float), lambda *_args, **_kwargs: None, 0)
    old_targets = learner.fixed_cycle_targets.detach().clone()
    old_q = learner.fixed_cycle_actor_q.detach().clone()
    old_target_cache = learner.fixed_cycle_targets
    old_q_cache = learner.fixed_cycle_actor_q

    new_observation = torch.tensor([[8.17, -2.29, 87.13]])
    learner.replay.add_sequence(
        torch.tensor([2]),
        new_observation,
        torch.tensor([2]),
        torch.tensor([1.37]),
        new_observation * -0.4,
        torch.tensor([0.89]),
        torch.tensor([False]),
        torch.tensor([False]),
    )
    assert len(learner.replay) == 11
    torch.testing.assert_close(
        _replay_field(learner, "observations")[0], new_observation[0], rtol=0, atol=0,
    )

    torch.manual_seed(839)
    expected_plan = _copy_sequence_plan(learner.replay.build_epoch_batches(3, 4))
    expected_targets = _plan_target_cache(learner, expected_plan)
    expected_q = _expected_cycle_start_q(learner)
    torch.manual_seed(839)
    learner._train_sac_transition(defaultdict(float), lambda *_args, **_kwargs: None, 1)

    assert learner.fixed_cycle_targets is not old_target_cache
    assert learner.fixed_cycle_actor_q is not old_q_cache
    assert learner.fixed_target_precompute_rows == learner.fixed_actor_q_precompute_rows == 11
    torch.testing.assert_close(learner.fixed_cycle_targets, expected_targets, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(learner.fixed_cycle_actor_q, expected_q, rtol=0, atol=0)
    assert torch.max(torch.abs(learner.fixed_cycle_targets[1:] - old_targets[1:])).item() > 1e-6
    assert torch.max(torch.abs(learner.fixed_cycle_actor_q[1:] - old_q[1:])).item() > 1e-6


@pytest.mark.skipif(
    os.environ.get("SAC_PROFILE_CUDA_TEST") != "1" or not torch.cuda.is_available(),
    reason="set SAC_PROFILE_CUDA_TEST=1 on an idle CUDA device for PQL replay cache tests",
)
@pytest.mark.parametrize(
    "replay_device,pin_memory",
    [("cpu", True), ("cuda", False)],
)
def test_cuda_pql_target_and_q_caches_match_oracles_for_pinned_cpu_and_gpu_replay(
        replay_device, pin_memory):
    learner = _make_pql_learner(
        seed=857,
        update_epochs=1,
        device="cuda",
        replay_device=replay_device,
        pin_memory=pin_memory,
    )
    install_variant(learner, "fixed_targets_actor_q")
    torch.manual_seed(853)
    expected_plan = _copy_sequence_plan(learner.replay.build_epoch_batches(3, 4))
    expected_targets = _plan_target_cache(learner, expected_plan)
    expected_q = _expected_cycle_start_q(learner)
    torch.manual_seed(853)
    learner._train_sac_transition(defaultdict(float), lambda *_args, **_kwargs: None, 0)
    torch.cuda.synchronize()

    assert learner.fixed_cycle_targets.device.type == "cuda"
    assert learner.fixed_cycle_actor_q.device.type == "cuda"
    assert learner.fixed_target_precompute_rows == len(learner.replay)
    torch.testing.assert_close(learner.fixed_cycle_targets, expected_targets, rtol=1e-6, atol=1e-7)
    torch.testing.assert_close(learner.fixed_cycle_actor_q, expected_q, rtol=0, atol=0)


def _pql_filter_gloo_worker(rank, init_method, result_dir):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = "2"
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    try:
        learner = _make_pql_learner(
            seed=863,
            update_epochs=1,
            observation_offset=rank * 0.07,
            filter_score="q_advantage",
            keep_fraction=0.46,
        )
        ddp_policy = DistributedDataParallel(learner.uncompiled_policy)
        ddp_critic = DistributedDataParallel(learner.uncompiled_critic)
        learner.ddp_policy = ddp_policy
        learner.ddp_critic = ddp_critic
        learner.policy_train = ddp_policy
        learner.critic_train = ddp_critic
        learner.distributed = True
        learner.world_size = 2
        install_variant(learner, "fixed_targets_actor_q")

        updates = learner._train_sac_transition(
            defaultdict(float), lambda *_args, **_kwargs: None, 0,
        )
        assert updates == 2
        assert learner.filter_candidate_rows == 11
        assert learner.filter_retained_rows == 6
        assert learner.sampled_transitions == 12
        assert learner.learner_updates == 2

        for module in (learner.uncompiled_policy, learner.uncompiled_critic):
            for parameter in module.parameters():
                gathered = [torch.empty_like(parameter) for _ in range(2)]
                dist.all_gather(gathered, parameter.detach())
                torch.testing.assert_close(gathered[0], gathered[1], rtol=1e-6, atol=1e-7)
        gathered_alpha = [torch.empty_like(learner.log_alpha) for _ in range(2)]
        dist.all_gather(gathered_alpha, learner.log_alpha.detach())
        torch.testing.assert_close(gathered_alpha[0], gathered_alpha[1], rtol=1e-6, atol=1e-7)
        dist.barrier()
        torch.save({
            "updates": updates,
            "learner_updates": learner.learner_updates,
            "sampled_transitions": learner.sampled_transitions,
            "candidate_rows": learner.filter_candidate_rows,
            "retained_rows": learner.filter_retained_rows,
            "policy": {
                key: value.detach().cpu().clone()
                for key, value in learner.uncompiled_policy.state_dict().items()
            },
            "critic": {
                key: value.detach().cpu().clone()
                for key, value in learner.uncompiled_critic.state_dict().items()
            },
            "alpha": learner.log_alpha.detach().cpu().clone(),
        }, Path(result_dir) / f"rank-{rank}.pt")
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def _spawn_gloo_with_timeout(worker, args, timeout_seconds=120):
    process_context = mp.spawn(worker, args=args, nprocs=2, join=False)
    for _ in range(timeout_seconds):
        if process_context.join(timeout=1):
            return
    for process in process_context.processes:
        if process.is_alive():
            process.terminate()
    for process in process_context.processes:
        process.join(timeout=5)
    pytest.fail(f"two-rank Gloo worker exceeded {timeout_seconds}s timeout")


@pytest.mark.skipif(
    not dist.is_available() or not getattr(dist, "is_gloo_available", lambda: False)(),
    reason="PyTorch Gloo distributed support is unavailable",
)
def test_two_rank_gloo_filtering_keeps_equal_update_and_backward_counts(tmp_path):
    result_dir = tmp_path / "gloo-results"
    result_dir.mkdir()
    init_path = tmp_path / "pql-filter-gloo-init"
    _spawn_gloo_with_timeout(
        _pql_filter_gloo_worker,
        (f"file://{init_path}", str(result_dir)),
    )
    results = [
        torch.load(result_dir / f"rank-{rank}.pt", map_location="cpu", weights_only=False)
        for rank in range(2)
    ]

    assert [result["updates"] for result in results] == [2, 2]
    assert [result["learner_updates"] for result in results] == [2, 2]
    assert [result["sampled_transitions"] for result in results] == [12, 12]
    assert [result["candidate_rows"] for result in results] == [11, 11]
    assert [result["retained_rows"] for result in results] == [6, 6]
    for model_name in ("policy", "critic"):
        assert results[0][model_name].keys() == results[1][model_name].keys()
        for parameter_name in results[0][model_name]:
            torch.testing.assert_close(
                results[0][model_name][parameter_name],
                results[1][model_name][parameter_name],
                rtol=1e-6,
                atol=1e-7,
            )
    torch.testing.assert_close(results[0]["alpha"], results[1]["alpha"], rtol=1e-6, atol=1e-7)
