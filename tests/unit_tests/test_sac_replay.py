import pytest
import torch

from pufferlib.sac_replay import TensorReplay


def _transitions(start, count):
    values = torch.arange(start, start + count, dtype=torch.float32)
    observations = torch.stack((values, values + 10), dim=1)
    actions = values.to(torch.int64) + 20
    rewards = values + 30
    next_observations = observations + 100
    discounts = values + 40
    return observations, actions, rewards, next_observations, discounts


def test_add_copies_inserted_data_and_tracks_occupancy():
    replay = TensorReplay(capacity=4, observation_shape=(2,))
    transitions = _transitions(0, 2)
    replay.add(*transitions)
    assert len(replay) == 2

    for source in transitions:
        source.fill_(-1)

    batch = replay.get_batch(torch.tensor([0, 1], dtype=torch.int64))
    expected = _transitions(0, 2)
    for actual, original in zip(batch, expected):
        torch.testing.assert_close(actual, original)


def test_replace_updates_selected_rows_without_changing_occupancy():
    replay = TensorReplay(capacity=4, observation_shape=(2,))
    original = _transitions(0, 3)
    replay.add(*original)
    replacement = _transitions(50, 1)
    expected_replacement = tuple(field.clone() for field in replacement)
    replay.replace(torch.tensor([1], dtype=torch.int64), *replacement)
    for field in replacement:
        field.fill_(-1)
    assert len(replay) == 3

    batch = replay.get_batch(torch.tensor([0, 1, 2], dtype=torch.int64))
    for field_idx, actual in enumerate(batch):
        expected = original[field_idx].clone()
        expected[1] = expected_replacement[field_idx][0]
        torch.testing.assert_close(actual, expected)


def test_get_batch_preserves_field_and_index_order_and_returns_copies():
    replay = TensorReplay(capacity=4, observation_shape=(2,))
    replay.add(*_transitions(0, 3))

    batch = replay.get_batch(torch.tensor([2, 0], dtype=torch.int64))
    expected = _transitions(0, 3)
    for actual, original in zip(batch, expected):
        torch.testing.assert_close(actual, original[[2, 0]])
        actual.fill_(-7)

    stored = replay.get_batch(torch.tensor([2, 0], dtype=torch.int64))
    for actual, original in zip(stored, expected):
        torch.testing.assert_close(actual, original[[2, 0]])


def test_pin_memory_returns_pinned_cpu_storage_and_batches():
    try:
        torch.empty((1,), pin_memory=True)
    except RuntimeError as error:
        pytest.skip(f"pinned-memory allocator unavailable: {error}")

    replay = TensorReplay(capacity=4, observation_shape=(2,), pin_memory=True)
    replay.add(*_transitions(0, 3))

    storage = (
        replay.observations,
        replay.actions,
        replay.rewards,
        replay.next_observations,
        replay.discounts,
    )
    assert all(field.device.type == "cpu" and field.is_pinned() for field in storage)

    batch = replay.get_batch(torch.tensor([2, 0], dtype=torch.int64))
    expected = _transitions(0, 3)
    for actual, original in zip(batch, expected):
        assert actual.device.type == "cpu"
        assert actual.is_pinned()
        torch.testing.assert_close(actual, original[[2, 0]])


def test_add_rejects_append_past_capacity_without_changing_occupancy():
    replay = TensorReplay(capacity=2, observation_shape=(2,))
    replay.add(*_transitions(0, 1))

    with pytest.raises(ValueError, match="capacity exceeded"):
        replay.add(*_transitions(1, 2))

    assert len(replay) == 1
    batch = replay.get_batch(torch.tensor([0], dtype=torch.int64))
    expected = _transitions(0, 1)
    for actual, original in zip(batch, expected):
        torch.testing.assert_close(actual, original)


@pytest.mark.parametrize("capacity", [0, -1, True, 1.5])
def test_constructor_requires_positive_integer_capacity(capacity):
    with pytest.raises(ValueError, match="positive integer"):
        TensorReplay(capacity=capacity, observation_shape=(2,))
