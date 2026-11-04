"""Compare sequence replay to an independent chronological Python FIFO model."""

from collections import Counter, deque
import random

import pytest
import torch

from pufferlib.sac_sequence_replay import SequenceReplay


def test_interleaved_collection_matches_fifo_oracle_across_repeated_wraps():
    randomizer = random.Random(24)
    replay = SequenceReplay(13, (1,), 5)
    resident = deque(maxlen=13)
    tails = {}
    next_label = 0
    for _ in range(200):
        broken_agents = randomizer.sample(range(5), randomizer.randrange(3))
        replay.break_chains(torch.tensor(broken_agents, dtype=torch.long))
        for agent in broken_agents:
            tails.pop(agent, None)
        agents = randomizer.sample(range(5), randomizer.randrange(1, 6))
        labels = list(range(next_label, next_label + len(agents)))
        next_label += len(agents)
        terminals = [randomizer.random() < 0.15 for _ in agents]
        truncations = [randomizer.random() < 0.1 for _ in agents]
        for agent, label, terminal, truncation in zip(agents, labels, terminals, truncations):
            resident.append((label, agent, tails.get(agent)))
            if terminal or truncation:
                tails.pop(agent, None)
            else:
                tails[agent] = label
        observations = torch.tensor(labels, dtype=torch.float32).reshape(-1, 1)
        replay.add_sequence(
            torch.tensor(agents), observations, torch.zeros(len(agents), dtype=torch.long),
            torch.ones(len(agents)), observations + 1, torch.full((len(agents),), 0.9),
            torch.tensor(terminals), torch.tensor(truncations),
        )
        plans = replay.build_epoch_batches(4, 5)
        expected = {label: (agent, previous) for label, agent, previous in resident}
        seen = []
        for indices, offsets in plans:
            sampled = replay.get_batch(indices)[0].flatten().to(torch.long).tolist()
            seen.extend(sampled)
            for start, end in zip(offsets[:-1], offsets[1:]):
                segment = sampled[start:end]
                assert 1 <= len(segment) <= 4
                for previous, current in zip(segment, segment[1:]):
                    assert expected[current][1] == previous
        assert sorted(seen) == sorted(expected)
        roots = {}
        for label, _agent, previous in resident:
            roots[label] = roots.get(previous, label)
        chain_lengths = Counter(roots.values())
        assert replay.last_plan_metrics["sequence_length_max"] == max(chain_lengths.values())
        assert replay.last_plan_metrics["sequence_length_mean"] == pytest.approx(
            len(resident) / len(chain_lengths),
        )
