"""Learner integration checks for PQL target preparation and updates."""

import copy
from collections import defaultdict
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from pufferlib import pufferl_sac
from pufferlib.sac_replay import TensorReplay
from pufferlib.sac_sequence_replay import SequenceReplay


class _TinyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, observations):
        return self.linear(observations)


class _TinyCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.q1 = nn.Linear(3, 2)
        self.q2 = nn.Linear(3, 2)

    def forward(self, observations):
        return self.q1(observations), self.q2(observations)


class _FixedActor(nn.Module):
    def __init__(self, logits):
        super().__init__()
        self.logits = nn.Parameter(torch.tensor(logits, dtype=torch.float32))

    def forward(self, observations):
        return self.logits.expand(observations.shape[0], -1)


class _FixedTwinCritic(nn.Module):
    def __init__(self, q1, q2):
        super().__init__()
        self.q1 = nn.Parameter(torch.tensor(q1, dtype=torch.float32).reshape(1, -1))
        self.q2 = nn.Parameter(torch.tensor(q2, dtype=torch.float32).reshape(1, -1))

    def forward(self, observations):
        return (
            self.q1.expand(observations.shape[0], -1),
            self.q2.expand(observations.shape[0], -1),
        )


def _make_sequence_replay(segment_lengths):
    row_count = sum(segment_lengths)
    replay = SequenceReplay(row_count, (3,), len(segment_lengths))
    for time_idx in range(max(segment_lengths)):
        agent_ids = [agent_id for agent_id, length in enumerate(segment_lengths) if time_idx < length]
        observations = torch.tensor([
            [float(agent_id + 1), float(time_idx + 1), float(agent_id + time_idx)]
            for agent_id in agent_ids
        ])
        replay.add_sequence(
            torch.tensor(agent_ids),
            observations,
            torch.tensor([(agent_id + time_idx) % 2 for agent_id in agent_ids]),
            torch.tensor([0.1 * (agent_id + 1) + 0.03 * time_idx for agent_id in agent_ids]),
            observations + 0.25,
            torch.full((len(agent_ids),), 0.9),
            torch.zeros(len(agent_ids), dtype=torch.bool),
            torch.zeros(len(agent_ids), dtype=torch.bool),
        )
    return replay


def _make_training_learner(replay, actor_state, critic_state, *, max_minibatch_size,
                           trace_lambda=0.95, alpha=0.3):
    learner = object.__new__(pufferl_sac.PuffeSAC)
    learner.device = torch.device("cpu")
    learner.uncompiled_policy = _TinyActor()
    learner.uncompiled_policy.load_state_dict(actor_state)
    learner.policy = learner.uncompiled_policy
    learner.ddp_policy = learner.uncompiled_policy
    learner.policy_train = learner.policy
    learner.uncompiled_critic = _TinyCritic()
    learner.uncompiled_critic.load_state_dict(critic_state)
    learner.critic = learner.uncompiled_critic
    learner.ddp_critic = learner.uncompiled_critic
    learner.critic_train = learner.critic
    learner.uncompiled_target_critic = copy.deepcopy(learner.uncompiled_critic)
    learner.uncompiled_target_critic.requires_grad_(False).eval()
    learner.target_critic = learner.uncompiled_target_critic
    learner.amp_context = nullcontext()
    learner.log_alpha = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(alpha)))))
    learner.optimizer = torch.optim.Adam(learner.policy.parameters(), lr=0.003)
    learner.critic_optimizer = torch.optim.Adam(learner.critic.parameters(), lr=0.004)
    learner.alpha_optimizer = torch.optim.Adam([learner.log_alpha], lr=0.002)
    learner.config = {
        "return_estimator": "pql",
        "pql_horizon": 8,
        "pql_lambda": trace_lambda,
        "critic_entropy_in_target": True,
        "update_epochs": 1,
        "target_entropy": 0.6,
        "learn_alpha": True,
        "tau": 0.15,
        "max_grad_norm": None,
    }
    learner.batch_size = len(replay)
    learner.minibatch_size = len(replay)
    learner.max_minibatch_size = max_minibatch_size
    learner.replay = replay
    learner.distributed = False
    learner.world_size = 1
    learner.target_entropy_controller = None
    learner.learner_updates = 0
    learner.sampled_transitions = 0
    return learner


def _make_target_learner(replay, *, logits, q1, q2, alpha, entropy_in_target):
    learner = object.__new__(pufferl_sac.PuffeSAC)
    learner.device = torch.device("cpu")
    learner.amp_context = nullcontext()
    learner.config = {
        "return_estimator": "pql",
        "pql_lambda": 0.0,
        "critic_entropy_in_target": entropy_in_target,
    }
    learner.log_alpha = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(alpha)))))
    learner.policy = _FixedActor(logits)
    learner.target_critic = _FixedTwinCritic(q1, q2)
    learner.replay = replay
    learner.max_minibatch_size = 2
    return learner


def _single_row_replay():
    replay = SequenceReplay(1, (1,), num_agents=1)
    replay.add_sequence(
        torch.tensor([0]), torch.tensor([[0.0]]), torch.tensor([0]),
        torch.tensor([0.0]), torch.tensor([[0.0]]), torch.tensor([1.0]),
        torch.tensor([False]), torch.tensor([False]),
    )
    return replay


def _parameter_tensors(module):
    return [parameter.detach().clone() for parameter in module.parameters()]


def test_pql_targets_are_invariant_to_microbatch_cut_through_a_segment():
    torch.manual_seed(37)
    actor_state = copy.deepcopy(_TinyActor().state_dict())
    critic_state = copy.deepcopy(_TinyCritic().state_dict())
    full = _make_training_learner(
        _make_sequence_replay([4]), actor_state, critic_state, max_minibatch_size=4,
    )
    micro = _make_training_learner(
        _make_sequence_replay([4]), actor_state, critic_state, max_minibatch_size=2,
    )
    segment_plan = micro.replay.build_epoch_batches(horizon=8, logical_batch_size=4)
    assert len(segment_plan) == 1
    assert segment_plan[0][1].tolist() == [0, 4]
    full_targets = []
    micro_targets = []
    for learner, captured_targets in ((full, full_targets), (micro, micro_targets)):
        original_critic_loss = learner._critic_loss

        def capture_critic_targets(observations, actions, targets, *, _original=original_critic_loss,
                                   _captured=captured_targets):
            _captured.append(targets.detach().clone())
            return _original(observations, actions, targets)

        learner._critic_loss = capture_critic_targets
    torch.manual_seed(101)
    full_losses = defaultdict(float)
    full_updates = full._train_sac_transition(
        full_losses, lambda *_args, **_kwargs: None, 0,
    )
    torch.manual_seed(101)
    micro_losses = defaultdict(float)
    micro_updates = micro._train_sac_transition(
        micro_losses, lambda *_args, **_kwargs: None, 0,
    )

    assert full_updates == micro_updates == 1
    assert full.learner_updates == micro.learner_updates == 1
    assert full.sampled_transitions == micro.sampled_transitions == 4
    assert len(micro_targets) == 2
    assert [targets.numel() for targets in micro_targets] == [2, 2]
    torch.testing.assert_close(torch.cat(full_targets), torch.cat(micro_targets))
    for full_module, micro_module in (
        (full.policy, micro.policy),
        (full.critic, micro.critic),
        (full.target_critic, micro.target_critic),
    ):
        for full_value, micro_value in zip(
            _parameter_tensors(full_module), _parameter_tensors(micro_module),
        ):
            torch.testing.assert_close(full_value, micro_value, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(full.log_alpha, micro.log_alpha, rtol=1e-5, atol=1e-6)
    assert full_losses.keys() == micro_losses.keys()
    for key in full_losses:
        assert micro_losses[key] == pytest.approx(full_losses[key], rel=1e-5, abs=1e-6)


def test_lambda_zero_matches_one_step_soft_sac_targets_on_identical_ordered_rows():
    sequence_replay = _make_sequence_replay([4])
    rows = sequence_replay.get_batch(torch.arange(4))
    one_step_replay = TensorReplay(4, (3,))
    one_step_replay.add(*rows)
    actor = _TinyActor()
    critic = _TinyCritic()
    lambda_zero = _make_training_learner(
        sequence_replay,
        copy.deepcopy(actor.state_dict()),
        copy.deepcopy(critic.state_dict()),
        max_minibatch_size=2,
        trace_lambda=0.0,
    )
    one_step = _make_training_learner(
        one_step_replay,
        copy.deepcopy(actor.state_dict()),
        copy.deepcopy(critic.state_dict()),
        max_minibatch_size=2,
        trace_lambda=0.0,
    )
    one_step.config["return_estimator"] = "n_step"
    ordered_indices = torch.arange(4)
    sequence_plan = sequence_replay.build_epoch_batches(horizon=8, logical_batch_size=4)
    assert len(sequence_plan) == 1
    planned_indices, offsets = sequence_plan[0]
    assert planned_indices.tolist() == ordered_indices.tolist()

    pql_targets = lambda_zero._prepare_pql_targets(planned_indices, offsets)
    one_step_batch = one_step.replay.get_batch(planned_indices)
    one_step_targets = one_step._compute_targets(
        one_step_batch[2], one_step_batch[3], one_step_batch[4],
    )

    torch.testing.assert_close(pql_targets, one_step_targets)


def test_pql_bootstrap_uses_current_alpha_and_detaches_target_pass():
    learner = _make_target_learner(
        _single_row_replay(),
        logits=[torch.log(torch.tensor(0.75)).item(), torch.log(torch.tensor(0.25)).item()],
        q1=[2.0, 0.0],
        q2=[2.0, 0.0],
        alpha=0.5,
        entropy_in_target=True,
    )
    first = learner._prepare_pql_targets(torch.tensor([0]), torch.tensor([0, 1]))
    learner.log_alpha.data.fill_(torch.log(torch.tensor(2.0)))
    second = learner._prepare_pql_targets(torch.tensor([0]), torch.tensor([0, 1]))

    entropy = -(0.75 * torch.log(torch.tensor(0.75)) + 0.25 * torch.log(torch.tensor(0.25)))
    torch.testing.assert_close(first, torch.tensor([1.5 + 0.5 * entropy]))
    torch.testing.assert_close(second, torch.tensor([1.5 + 2.0 * entropy]))
    assert not first.requires_grad and not second.requires_grad
    assert learner.log_alpha.grad is None
    assert learner.policy.logits.grad is None
    assert learner.target_critic.q1.grad is None
    assert learner.target_critic.q2.grad is None


def test_pql_bootstrap_takes_twin_critic_minimum_before_action_mixture():
    learner = _make_target_learner(
        _single_row_replay(),
        logits=[torch.log(torch.tensor(0.75)).item(), torch.log(torch.tensor(0.25)).item()],
        q1=[0.0, 10.0],
        q2=[10.0, 0.0],
        alpha=1.0,
        entropy_in_target=False,
    )

    target = learner._prepare_pql_targets(torch.tensor([0]), torch.tensor([0, 1]))

    torch.testing.assert_close(target, torch.tensor([0.0]))
    mixed_then_minimum = min(0.75 * 0.0 + 0.25 * 10.0, 0.75 * 10.0 + 0.25 * 0.0)
    assert mixed_then_minimum == pytest.approx(2.5)
