import pytest
import torch

from pufferlib.sac_replay import NStepAccumulator


def _input_batch(agent_ids, timestep):
    observations = torch.tensor(
        [[agent_id, timestep] for agent_id in agent_ids], dtype=torch.float32
    )
    actions = torch.tensor(
        [agent_id * 100 + timestep for agent_id in agent_ids], dtype=torch.int64
    )
    rewards = torch.tensor(
        [agent_id * 10 + timestep + 1 for agent_id in agent_ids], dtype=torch.float32
    )
    next_observations = observations + torch.tensor([0.0, 1.0])
    terminals = torch.tensor(
        [agent_id == 2 and timestep == 4 for agent_id in agent_ids], dtype=torch.bool
    )
    truncations = torch.tensor(
        [agent_id == 0 and timestep == 6 for agent_id in agent_ids], dtype=torch.bool
    )
    return observations, actions, rewards, next_observations, terminals, truncations


def _oracle_step(queues, agent_ids, batch, n_steps, gamma):
    observations, actions, rewards, next_observations, terminals, truncations = batch
    emitted = []
    for row, agent_id in enumerate(agent_ids):
        transition = (
            observations[row].clone(),
            actions[row].clone(),
            rewards[row].clone(),
            next_observations[row].clone(),
        )
        queues[agent_id].append(transition)
        boundary = bool(terminals[row] or truncations[row])
        if boundary:
            horizons = range(len(queues[agent_id]), 0, -1)
        elif len(queues[agent_id]) >= n_steps:
            horizons = (n_steps,)
        else:
            horizons = ()

        for horizon in horizons:
            transitions = queues[agent_id][-horizon:]
            accumulated_reward = sum(
                gamma**offset * float(item[2])
                for offset, item in enumerate(transitions)
            )
            emitted.append(
                (
                    agent_id,
                    int(transitions[0][0][1]),
                    horizon,
                    transitions[0][0].clone(),
                    transitions[0][1].clone(),
                    accumulated_reward,
                    transition[3].clone(),
                    0.0
                    if bool(terminals[row])
                    else gamma**horizon,
                )
            )

        if boundary:
            queues[agent_id].clear()
        elif len(queues[agent_id]) >= n_steps:
            queues[agent_id].pop(0)
    return emitted


def _oracle_flush(queues, agent_ids, gamma):
    emitted = []
    for agent_id in agent_ids:
        pending = queues[agent_id]
        if not pending:
            continue
        successor = pending[-1][3].clone()
        for horizon in range(len(pending), 0, -1):
            transitions = pending[-horizon:]
            accumulated_reward = sum(
                gamma**offset * float(item[2])
                for offset, item in enumerate(transitions)
            )
            emitted.append(
                (
                    agent_id,
                    int(transitions[0][0][1]),
                    horizon,
                    transitions[0][0].clone(),
                    transitions[0][1].clone(),
                    accumulated_reward,
                    successor.clone(),
                    gamma**horizon,
                )
            )
        pending.clear()
    return emitted


def _actual_rows(result):
    fields, horizons = result
    observations, actions, rewards, next_observations, discounts = fields
    return [
        (
            int(observations[row, 0]),
            int(observations[row, 1]),
            int(horizons[row]),
            observations[row].cpu(),
            actions[row].cpu(),
            float(rewards[row]),
            next_observations[row].cpu(),
            float(discounts[row]),
        )
        for row in range(horizons.numel())
    ]


def _cpu_fields(result):
    fields, horizons = result
    return tuple(field.detach().cpu() for field in fields) + (horizons.detach().cpu(),)


def _assert_same_rows(actual, expected):
    actual_by_key = {(row[0], row[1], row[2]): row for row in actual}
    expected_by_key = {(row[0], row[1], row[2]): row for row in expected}
    assert len(actual_by_key) == len(actual)
    assert len(expected_by_key) == len(expected)
    assert len(actual) == len(expected)
    assert actual_by_key.keys() == expected_by_key.keys()
    for key, expected_row in expected_by_key.items():
        actual_row = actual_by_key[key]
        torch.testing.assert_close(actual_row[3], expected_row[3])
        torch.testing.assert_close(actual_row[4], expected_row[4])
        assert actual_row[5] == pytest.approx(expected_row[5])
        torch.testing.assert_close(actual_row[6], expected_row[6])
        assert actual_row[7] == pytest.approx(expected_row[7])


@pytest.mark.parametrize("n_steps", [1, 3, 5])
@pytest.mark.parametrize("gamma", [0.0, 0.9, 1.0])
def test_append_and_flush_match_per_agent_oracle_with_shuffled_ids(n_steps, gamma):
    accumulator = NStepAccumulator(
        num_agents=4,
        observation_shape=(2,),
        n_steps=n_steps,
        gamma=gamma,
        device="cpu",
    )
    queues = {agent_id: [] for agent_id in range(4)}
    all_expected = []

    for timestep in range(8):
        agent_ids = [agent_id for agent_id in range(4) if (agent_id + timestep) % 3 != 0]
        if timestep in (1, 4, 7):
            agent_ids.reverse()
        batch = _input_batch(agent_ids, timestep)
        expected = _oracle_step(queues, agent_ids, batch, n_steps, gamma)
        actual = accumulator.append(
            torch.tensor(agent_ids),
            batch[0],
            batch[1],
            batch[2],
            batch[3],
            batch[4],
            batch[5],
        )
        _assert_same_rows(_actual_rows(actual), expected)
        all_expected.extend(expected)
        assert accumulator.pending_count == sum(map(len, queues.values()))

    expected_flush = _oracle_flush(queues, list(range(4)), gamma)
    actual_flush = accumulator.flush()
    _assert_same_rows(_actual_rows(actual_flush), expected_flush)
    all_expected.extend(expected_flush)
    assert accumulator.pending_count == 0
    assert _actual_rows(accumulator.flush()) == []


def test_boundary_discounts_terminal_zero_and_truncation_bootstraps():
    accumulator = NStepAccumulator(2, (2,), n_steps=3, gamma=0.5)
    batch = _input_batch([0, 1], 0)
    accumulator.append(torch.tensor([0, 1]), *batch)

    batch = _input_batch([1, 0], 4)
    # The chosen boundary rows have distinct stable IDs and retain their own
    # successor observations even though the append order is shuffled.
    terminals = torch.tensor([True, False])
    truncations = torch.tensor([False, True])
    result = accumulator.append(
        torch.tensor([1, 0]),
        batch[0],
        batch[1],
        batch[2],
        batch[3],
        terminals,
        truncations,
    )
    rows = _actual_rows(result)
    assert {row[0] for row in rows} == {0, 1}
    for row in rows:
        if row[0] == 1:
            assert row[7] == 0.0
        else:
            assert row[7] == pytest.approx(0.5**row[2])
    assert accumulator.pending_count == 0


def test_flush_uses_copied_last_successor_and_returns_owned_rows():
    accumulator = NStepAccumulator(1, (2,), n_steps=3, gamma=0.5)
    observations = torch.tensor([[3.0, 0.0]])
    actions = torch.tensor([7])
    rewards = torch.tensor([2.0])
    next_observations = torch.tensor([[3.0, 1.0]])
    accumulator.append(
        torch.tensor([0]),
        observations,
        actions,
        rewards,
        next_observations,
        torch.tensor([False]),
        torch.tensor([False]),
    )
    observations.fill_(-1)
    actions.fill_(-1)
    rewards.fill_(-1)
    next_observations.fill_(-1)

    flushed = accumulator.flush()
    fields, horizons = flushed
    assert horizons.tolist() == [1]
    torch.testing.assert_close(fields[0], torch.tensor([[3.0, 0.0]]))
    torch.testing.assert_close(fields[1], torch.tensor([7]))
    torch.testing.assert_close(fields[2], torch.tensor([2.0]))
    torch.testing.assert_close(fields[3], torch.tensor([[3.0, 1.0]]))
    torch.testing.assert_close(fields[4], torch.tensor([0.5]))

    fields[3].fill_(-8)
    assert accumulator.pending_count == 0
    assert _actual_rows(accumulator.flush()) == []


def test_flush_can_emit_selected_agents_without_clearing_others():
    accumulator = NStepAccumulator(2, (2,), n_steps=3, gamma=0.8)
    batch = _input_batch([0, 1], 0)
    accumulator.append(torch.tensor([0, 1]), *batch)

    selected = accumulator.flush(torch.tensor([1]))
    assert selected[0][0][:, 0].tolist() == [1.0]
    assert accumulator.pending_count == 1

    remaining = accumulator.flush()
    assert remaining[0][0][:, 0].tolist() == [0.0]
    assert accumulator.pending_count == 0


def test_identical_shuffled_input_streams_produce_identical_rows():
    accumulators = [NStepAccumulator(3, (2,), n_steps=3, gamma=0.9) for _ in range(2)]
    for timestep in range(5):
        agent_ids = [2, 0, 1] if timestep % 2 == 0 else [1, 2, 0]
        batch = _input_batch(agent_ids, timestep)
        results = [
            accumulator.append(torch.tensor(agent_ids), *batch)
            for accumulator in accumulators
        ]
        left_fields = results[0][0] + (results[0][1],)
        right_fields = results[1][0] + (results[1][1],)
        for left, right in zip(left_fields, right_fields):
            torch.testing.assert_close(left, right)


def test_clear_discards_pending_rows_and_resets_ring_positions():
    accumulator = NStepAccumulator(2, (2,), n_steps=3, gamma=1.0)
    batch = _input_batch([1], 0)
    accumulator.append(torch.tensor([1]), *batch)
    assert accumulator.pending_count == 1

    accumulator.clear()

    assert accumulator.pending_count == 0
    assert _actual_rows(accumulator.flush()) == []
    assert torch.equal(accumulator._write_positions, torch.zeros(2, dtype=torch.int64))


def test_pin_memory_covers_pending_and_emitted_storage_when_available():
    try:
        torch.empty((1,), pin_memory=True)
    except RuntimeError as error:
        pytest.skip(f"pinned-memory allocator unavailable: {error}")

    accumulator = NStepAccumulator(1, (2,), n_steps=2, gamma=0.9, pin_memory=True)
    assert all(
        storage.is_pinned()
        for storage in (
            accumulator._observations,
            accumulator._actions,
            accumulator._rewards,
            accumulator._last_next_observations,
            accumulator._counts,
            accumulator._write_positions,
        )
    )
    result = accumulator.append(torch.tensor([0]), *_input_batch([0], 0))
    assert result[0][0].shape == (0, 2)
    assert result[1].shape == (0,)
    result = accumulator.append(torch.tensor([0]), *_input_batch([0], 1))
    assert all(field.is_pinned() for field in result[0] + (result[1],))


@pytest.mark.parametrize(
    "device",
    ["cpu"] + (["cuda"] if torch.cuda.is_available() else []),
)
def test_outputs_stay_on_configured_replay_device(device):
    accumulator = NStepAccumulator(1, (2,), n_steps=1, gamma=0.9, device=device)
    batch = tuple(
        field.to(device)
        for field in _input_batch([0], 0)
    )
    result = accumulator.append(torch.tensor([0], device=device), *batch)
    fields, horizons = result
    assert all(field.device.type == torch.device(device).type for field in fields)
    assert horizons.device.type == torch.device(device).type
    assert horizons.tolist() == [1]


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_and_pinned_cpu_accumulators_match_cpu_reference():
    try:
        torch.empty((1,), pin_memory=True)
    except RuntimeError as error:
        pytest.skip(f"pinned-memory allocator unavailable: {error}")

    cuda_accumulator = NStepAccumulator(
        3, (2,), n_steps=3, gamma=0.9, device="cuda"
    )
    pinned_cpu_accumulator = NStepAccumulator(
        3, (2,), n_steps=3, gamma=0.9, device="cpu", pin_memory=True
    )
    cpu_reference = NStepAccumulator(3, (2,), n_steps=3, gamma=0.9, device="cpu")

    def compare_results(cuda_result, pinned_cpu_result, cpu_result):
        cuda_fields = _cpu_fields(cuda_result)
        pinned_cpu_fields = _cpu_fields(pinned_cpu_result)
        cpu_fields = _cpu_fields(cpu_result)
        for cuda_field, pinned_cpu_field, cpu_field in zip(
            cuda_fields, pinned_cpu_fields, cpu_fields
        ):
            torch.testing.assert_close(cuda_field, cpu_field)
            torch.testing.assert_close(pinned_cpu_field, cpu_field)
        assert all(field.device.type == "cuda" for field in cuda_result[0])
        assert cuda_result[1].device.type == "cuda"
        assert all(field.device.type == "cpu" for field in pinned_cpu_result[0])
        assert pinned_cpu_result[1].device.type == "cpu"
        assert all(
            not field.numel() or field.is_pinned()
            for field in pinned_cpu_result[0] + (pinned_cpu_result[1],)
        )
        assert (
            cuda_accumulator.pending_count
            == pinned_cpu_accumulator.pending_count
            == cpu_reference.pending_count
        )

    def append(timestep, agent_ids, terminal_agent=None, truncation_agent=None):
        batch = list(_input_batch(agent_ids, timestep))
        batch[4] = torch.tensor(
            [agent_id == terminal_agent for agent_id in agent_ids], dtype=torch.bool
        )
        batch[5] = torch.tensor(
            [agent_id == truncation_agent for agent_id in agent_ids], dtype=torch.bool
        )
        batch_cuda = tuple(field.cuda() for field in batch)
        agent_ids_cuda = torch.tensor(agent_ids, dtype=torch.int64, device="cuda")
        agent_ids_cpu = torch.tensor(agent_ids, dtype=torch.int64)
        compare_results(
            cuda_accumulator.append(agent_ids_cuda, *batch_cuda),
            pinned_cpu_accumulator.append(agent_ids_cuda, *batch_cuda),
            cpu_reference.append(agent_ids_cpu, *batch),
        )

    append(0, [0, 1, 2])
    append(1, [0, 2])
    compare_results(
        cuda_accumulator.flush(torch.tensor([1], device="cuda")),
        pinned_cpu_accumulator.flush(torch.tensor([1], device="cuda")),
        cpu_reference.flush(torch.tensor([1])),
    )
    append(2, [0, 1, 2], truncation_agent=0)
    append(3, [0, 2])
    compare_results(
        cuda_accumulator.flush(torch.tensor([1], device="cuda")),
        pinned_cpu_accumulator.flush(torch.tensor([1], device="cuda")),
        cpu_reference.flush(torch.tensor([1])),
    )
    append(4, [0, 1, 2], terminal_agent=2)
    compare_results(
        cuda_accumulator.flush(),
        pinned_cpu_accumulator.flush(),
        cpu_reference.flush(),
    )


@pytest.mark.parametrize("num_agents,n_steps", [(0, 1), (1, 0), (True, 1), (1, True)])
def test_constructor_requires_positive_integer_sizes(num_agents, n_steps):
    with pytest.raises(ValueError, match="positive integer"):
        NStepAccumulator(num_agents, (2,), n_steps, gamma=0.99)
