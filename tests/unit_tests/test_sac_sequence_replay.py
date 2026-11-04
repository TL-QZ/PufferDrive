import pytest
import torch

from pufferlib.sac_sequence_replay import SequenceReplay


def _batch(agent_ids, values, terminals=None, truncations=None, discounts=None):
    agent_ids = torch.tensor(agent_ids, dtype=torch.int64)
    values = torch.tensor(values, dtype=torch.float32)
    batch_size = values.numel()
    observations = values.reshape(batch_size, 1)
    next_observations = (values + 100).reshape(batch_size, 1)
    actions = torch.arange(batch_size, dtype=torch.int64) + 10
    rewards = values + 0.5
    if discounts is None:
        discounts = torch.full((batch_size,), 0.9, dtype=torch.float32)
    else:
        discounts = torch.tensor(discounts, dtype=torch.float32)
    if terminals is None:
        terminals = torch.zeros(batch_size, dtype=torch.bool)
    else:
        terminals = torch.tensor(terminals, dtype=torch.bool)
    if truncations is None:
        truncations = torch.zeros(batch_size, dtype=torch.bool)
    else:
        truncations = torch.tensor(truncations, dtype=torch.bool)
    return (
        agent_ids,
        observations,
        actions,
        rewards,
        next_observations,
        discounts,
        terminals,
        truncations,
    )


def _insert(replay, *args, **kwargs):
    replay.add_sequence(*_batch(*args, **kwargs))


def _planned_segments(replay, plans):
    segments = []
    for replay_indices, segment_offsets in plans:
        assert replay_indices.device.type == "cpu"
        assert segment_offsets.device.type == "cpu"
        assert int(segment_offsets[0]) == 0
        assert int(segment_offsets[-1]) == replay_indices.numel()
        assert bool(torch.all(segment_offsets[1:] > segment_offsets[:-1]))
        for start, end in zip(segment_offsets[:-1].tolist(), segment_offsets[1:].tolist()):
            segments.append(replay_indices[start:end])
    return segments


def test_interleaved_stable_agent_ids_form_independent_sequences_and_copy_inputs():
    replay = SequenceReplay(capacity=12, observation_shape=(1,), num_agents=2)
    first_batch = _batch([0, 1], [0.0, 10.0])
    replay.add_sequence(*first_batch)
    replay.add_sequence(*_batch([1, 0], [11.0, 1.0]))
    replay.add_sequence(*_batch([0, 1], [2.0, 12.0]))
    first_batch[1].fill_(-99)

    plans = replay.build_epoch_batches(horizon=8, logical_batch_size=16)
    segments = _planned_segments(replay, plans)
    assert len(replay) == 6
    assert sorted(torch.cat([plan[0] for plan in plans]).tolist()) == list(range(6))
    for segment in segments:
        metadata = replay.get_sequence_metadata(segment)
        values = replay.get_batch(segment)[0][:, 0]
        assert torch.unique(metadata["agent_ids"]).numel() == 1
        assert torch.allclose(values[1:] - values[:-1], torch.ones(values.numel() - 1))
    observations = replay.get_batch(torch.arange(6))[0]
    assert not bool(torch.any(observations == -99))


def test_terminal_truncation_and_terminal_precedence_end_chains():
    replay = SequenceReplay(capacity=8, observation_shape=(1,), num_agents=3)
    _insert(
        replay,
        [0, 1, 2],
        [0.0, 10.0, 20.0],
        terminals=[True, False, True],
        truncations=[False, True, True],
    )
    _insert(replay, [0, 1, 2], [1.0, 11.0, 21.0])

    observations, _, _, next_observations, discounts = replay.get_batch(torch.arange(6))
    metadata = replay.get_sequence_metadata(torch.arange(6))
    assert observations[:, 0].tolist() == [0.0, 10.0, 20.0, 1.0, 11.0, 21.0]
    assert next_observations[:, 0].tolist() == [100.0, 110.0, 120.0, 101.0, 111.0, 121.0]
    torch.testing.assert_close(
        discounts, torch.tensor([0.0, 0.9, 0.0, 0.9, 0.9, 0.9])
    )
    assert metadata["terminals"].tolist() == [True, False, True, False, False, False]
    assert metadata["truncations"].tolist() == [False, True, False, False, False, False]

    plans = replay.build_epoch_batches(horizon=8, logical_batch_size=8)
    segments = _planned_segments(replay, plans)
    assert sorted(len(segment) for segment in segments) == [1, 1, 1, 1, 1, 1]


def test_break_chains_splits_resident_suffix_from_future_rows():
    replay = SequenceReplay(capacity=8, observation_shape=(1,), num_agents=1)
    _insert(replay, [0], [0.0])
    _insert(replay, [0], [1.0])
    replay.break_chains(torch.tensor([0]))
    _insert(replay, [0], [2.0])

    plans = replay.build_epoch_batches(horizon=8, logical_batch_size=8)
    segments = _planned_segments(replay, plans)
    values_by_segment = sorted(
        tuple(replay.get_batch(segment)[0][:, 0].tolist()) for segment in segments
    )
    assert values_by_segment == [(0.0, 1.0), (2.0,)]
    assert replay.last_plan_metrics["cut_gap_count"] == 1
    replay.build_epoch_batches(horizon=8, logical_batch_size=8)
    assert replay.last_plan_metrics["cut_gap_count"] == 1
    replay.reset_collection_metrics()
    assert replay.last_plan_metrics["cut_gap_count"] == 0


def test_fifo_overwrite_unlinks_old_generation_and_keeps_exact_coverage():
    replay = SequenceReplay(capacity=3, observation_shape=(1,), num_agents=1)
    for value in range(3):
        _insert(replay, [0], [float(value)])
    _insert(replay, [0], [3.0])

    plans = replay.build_epoch_batches(horizon=8, logical_batch_size=8)
    replay_indices = torch.cat([plan[0] for plan in plans])
    assert sorted(replay_indices.tolist()) == [0, 1, 2]
    assert replay.get_batch(torch.arange(3))[0][:, 0].tolist() == [3.0, 1.0, 2.0]
    segments = _planned_segments(replay, plans)
    segment_values = sorted(
        tuple(replay.get_batch(segment)[0][:, 0].tolist()) for segment in segments
    )
    assert segment_values == [(1.0, 2.0, 3.0)]
    assert replay.last_plan_metrics["cut_overwrite_count"] == 1


def test_horizon_and_logical_boundaries_cut_without_dropping_rows():
    replay = SequenceReplay(capacity=10, observation_shape=(1,), num_agents=1)
    for value in range(7):
        _insert(replay, [0], [float(value)])

    torch.manual_seed(5)
    plans = replay.build_epoch_batches(horizon=3, logical_batch_size=2)
    all_indices = torch.cat([plan[0] for plan in plans])
    assert sorted(all_indices.tolist()) == list(range(7))
    assert all(indices.numel() <= 2 for indices, _ in plans)
    segments = _planned_segments(replay, plans)
    assert all(0 < len(segment) <= 2 for segment in segments)
    for segment in segments:
        values = replay.get_batch(segment)[0][:, 0]
        assert torch.allclose(values[1:] - values[:-1], torch.ones(values.numel() - 1))
    assert replay.last_plan_metrics["cut_horizon_count"] == 2
    assert replay.last_plan_metrics["cut_logical_batch_count"] > 0
    assert replay.last_suffix_length_counts.sum().item() == 7


def test_stale_generation_is_a_cut_and_malformed_reciprocity_fails():
    replay = SequenceReplay(capacity=4, observation_shape=(1,), num_agents=1)
    _insert(replay, [0], [0.0])
    _insert(replay, [0], [1.0])
    replay._generation[1] += 1
    replay._previous_indices[1] = -1
    replay._previous_generations[1] = -1
    plans = replay.build_epoch_batches(horizon=4, logical_batch_size=4)
    assert replay.last_plan_metrics["cut_stale_generation_count"] == 1
    assert sorted(torch.cat([plan[0] for plan in plans]).tolist()) == [0, 1]

    replay._next_indices[0] = 1
    replay._next_generations[0] = replay._generation[1]
    with pytest.raises(RuntimeError, match="successor reciprocity"):
        replay.build_epoch_batches(horizon=4, logical_batch_size=4)


def test_constructor_and_epoch_inputs_require_positive_integers():
    with pytest.raises(ValueError, match="num_agents"):
        SequenceReplay(capacity=4, observation_shape=(1,), num_agents=True)
    replay = SequenceReplay(capacity=4, observation_shape=(1,), num_agents=1)
    with pytest.raises(ValueError, match="horizon"):
        replay.build_epoch_batches(horizon=False, logical_batch_size=2)


def test_logical_batch_larger_than_partial_replay_returns_one_batch():
    replay = SequenceReplay(capacity=8, observation_shape=(1,), num_agents=1)
    _insert(replay, [0], [0.0])
    _insert(replay, [0], [1.0])

    plans = replay.build_epoch_batches(horizon=128, logical_batch_size=4)

    assert len(plans) == 1
    assert plans[0][0].numel() == 2
    assert plans[0][1].tolist() == [0, 2]
