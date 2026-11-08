"""CPU semantic parity tests for isolated SAC replay gather candidates."""

import copy
from collections import defaultdict
from contextlib import nullcontext
import math
import os

import pytest
import torch
from torch import nn

from pufferlib import pufferl_sac
from pufferlib.sac_replay import TensorReplay
from project.PPO_SAC_Profiling.sac_variants import install_variant


class _TinyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 3)

    def forward(self, observations):
        return self.linear(observations)


class _TinyCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.q1 = nn.Linear(3, 3)
        self.q2 = nn.Linear(3, 3)

    def forward(self, observations):
        return self.q1(observations), self.q2(observations)


class _RecordingReplay(TensorReplay):
    def __init__(self, *, device="cpu", pin_memory=False):
        super().__init__(7, (3,), device=device, pin_memory=pin_memory)
        self.sampled_indices = []
        observations = torch.arange(21, dtype=torch.float32).reshape(7, 3) / 10
        self.add(
            observations,
            torch.tensor([0, 1, 2, 0, 1, 2, 0]),
            torch.tensor([0.2, -0.1, 0.5, 0.3, -0.2, 0.4, 0.1]),
            observations * 0.7 + 0.15,
            torch.tensor([0.99, 0.99, 0.0, 0.99, 0.0, 0.99, 0.99]),
        )

    def get_batch(self, indices):
        self.sampled_indices.append(indices.clone())
        return super().get_batch(indices)


def _make_learner(actor_state, critic_state, *, update_epochs=2, device="cpu", pin_memory=False):
    device = torch.device(device)
    learner = object.__new__(pufferl_sac.PuffeSAC)
    learner.device = device
    learner.uncompiled_policy = _TinyActor().to(device)
    learner.uncompiled_policy.load_state_dict(actor_state)
    learner.policy = learner.uncompiled_policy
    learner.ddp_policy = learner.uncompiled_policy
    learner.policy_train = learner.policy
    learner.uncompiled_critic = _TinyCritic().to(device)
    learner.uncompiled_critic.load_state_dict(critic_state)
    learner.critic = learner.uncompiled_critic
    learner.ddp_critic = learner.uncompiled_critic
    learner.critic_train = learner.critic
    learner.uncompiled_target_critic = copy.deepcopy(learner.uncompiled_critic).requires_grad_(False).eval()
    learner.target_critic = learner.uncompiled_target_critic
    learner.amp_context = nullcontext()
    learner.log_alpha = nn.Parameter(torch.tensor(-1.2, device=device))
    learner.optimizer = torch.optim.Adam(learner.policy.parameters(), lr=0.003)
    learner.critic_optimizer = torch.optim.Adam(learner.critic.parameters(), lr=0.004)
    learner.alpha_optimizer = torch.optim.Adam([learner.log_alpha], lr=0.002)
    learner.config = {
        "return_estimator": "n_step",
        "update_epochs": update_epochs,
        "target_entropy": 0.2,
        "tau": 0.15,
        "max_grad_norm": 1e-4,
        "critic_entropy_in_target": True,
        "learn_alpha": True,
        "log_collapse_diagnostics": False,
    }
    learner.batch_size = 7
    learner.minibatch_size = 4
    learner.max_minibatch_size = 3
    learner.replay = _RecordingReplay(device="cpu", pin_memory=pin_memory)
    learner.distributed = False
    learner.world_size = 1
    learner.learner_updates = 0
    learner.sampled_transitions = 0
    learner.agent_steps = 0
    learner.target_entropy_controller = pufferl_sac.AdaptiveTargetEntropyController(
        0.2,
        ema_decay=0.8,
        mean_threshold=2.0,
        std_threshold=2.0,
        decay_factor=0.9,
        required_count=2,
    )
    return learner


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


def _assert_learner_state_equal(actual, expected):
    for actual_model, expected_model in (
        (actual.uncompiled_policy, expected.uncompiled_policy),
        (actual.uncompiled_critic, expected.uncompiled_critic),
        (actual.uncompiled_target_critic, expected.uncompiled_target_critic),
    ):
        _assert_nested_equal(actual_model.state_dict(), expected_model.state_dict())
    torch.testing.assert_close(actual.log_alpha, expected.log_alpha, rtol=0, atol=0)
    for actual_optimizer, expected_optimizer in (
        (actual.optimizer, expected.optimizer),
        (actual.critic_optimizer, expected.critic_optimizer),
        (actual.alpha_optimizer, expected.alpha_optimizer),
    ):
        _assert_nested_equal(actual_optimizer.state_dict(), expected_optimizer.state_dict())
    assert actual.target_entropy_controller.state_dict() == expected.target_entropy_controller.state_dict()
    assert actual.learner_updates == expected.learner_updates
    assert actual.sampled_transitions == expected.sampled_transitions


@pytest.mark.parametrize("variant", ["obs_only", "cached"])
def test_candidates_match_stock_over_multiple_cycles_with_tails_clipping_and_adaptive_tes(variant):
    torch.manual_seed(37)
    actor_state = copy.deepcopy(_TinyActor().state_dict())
    critic_state = copy.deepcopy(_TinyCritic().state_dict())
    stock = _make_learner(actor_state, critic_state)
    candidate = _make_learner(actor_state, critic_state)
    assert install_variant(stock, "stock") is stock
    assert install_variant(candidate, variant) is candidate
    target_policy = _TinyActor()
    target_policy.load_state_dict(actor_state)
    target_policy.requires_grad_(False).eval()

    torch.manual_seed(101)
    stock_cycle_losses = []
    for cycle in range(3):
        stock.agent_steps = candidate.agent_steps = (cycle + 1) * 100
        losses = defaultdict(float)
        updates = stock._train_sac_transition(
            losses, lambda *_args, **_kwargs: None, cycle,
            target_policy=target_policy if cycle == 1 else None,
        )
        assert updates == 4  # Two shuffled passes, each with logical minibatches of 4 and 3.
        stock_cycle_losses.append(dict(losses))

    torch.manual_seed(101)
    candidate_cycle_losses = []
    for cycle in range(3):
        candidate.agent_steps = (cycle + 1) * 100
        losses = defaultdict(float)
        updates = candidate._train_sac_transition(
            losses, lambda *_args, **_kwargs: None, cycle,
            target_policy=target_policy if cycle == 1 else None,
        )
        assert updates == 4
        candidate_cycle_losses.append(dict(losses))

    assert stock_cycle_losses == candidate_cycle_losses
    _assert_learner_state_equal(candidate, stock)
    assert stock.learner_updates == candidate.learner_updates == 12
    assert stock.sampled_transitions == candidate.sampled_transitions == 42
    assert stock.target_entropy_controller.decay_count > 0
    assert candidate.target_entropy_controller.decay_count == stock.target_entropy_controller.decay_count

    # The candidates keep every critic gather and skip the actor's former five-field gather.
    assert len(stock.replay.sampled_indices) == 36
    assert len(candidate.replay.sampled_indices) == 18
    stock_cursor = 0
    candidate_cursor = 0
    for _cycle in range(3):
        for logical_batch_count in (4, 3, 4, 3):
            microbatch_count = math.ceil(logical_batch_count / 3)
            critic_indices = candidate.replay.sampled_indices[
                candidate_cursor:candidate_cursor + microbatch_count
            ]
            stock_critic_indices = stock.replay.sampled_indices[
                stock_cursor:stock_cursor + microbatch_count
            ]
            stock_actor_indices = stock.replay.sampled_indices[
                stock_cursor + microbatch_count:stock_cursor + 2 * microbatch_count
            ]
            assert all(
                torch.equal(stock_index, candidate_index)
                for stock_index, candidate_index in zip(stock_critic_indices, critic_indices)
            )
            assert all(
                torch.equal(critic_index, actor_index)
                for critic_index, actor_index in zip(stock_critic_indices, stock_actor_indices)
            )
            stock_cursor += 2 * microbatch_count
            candidate_cursor += microbatch_count


@pytest.mark.parametrize("variant", ["stock", "obs_only", "cached"])
def test_candidates_match_stock_when_policy_updates_are_disabled(variant):
    torch.manual_seed(43)
    actor_state = copy.deepcopy(_TinyActor().state_dict())
    critic_state = copy.deepcopy(_TinyCritic().state_dict())
    stock = _make_learner(actor_state, critic_state, update_epochs=1)
    candidate = _make_learner(actor_state, critic_state, update_epochs=1)
    install_variant(stock, "stock")
    install_variant(candidate, variant)
    initial_policy_state = copy.deepcopy(candidate.policy.state_dict())
    initial_alpha = candidate.log_alpha.detach().clone()

    torch.manual_seed(109)
    stock_losses = defaultdict(float)
    stock_updates = stock._train_sac_transition(
        stock_losses, lambda *_args, **_kwargs: None, 0, update_policy=False,
    )
    torch.manual_seed(109)
    candidate_losses = defaultdict(float)
    candidate_updates = candidate._train_sac_transition(
        candidate_losses, lambda *_args, **_kwargs: None, 0, update_policy=False,
    )

    assert stock_updates == candidate_updates == 2
    assert dict(stock_losses) == dict(candidate_losses)
    _assert_learner_state_equal(candidate, stock)
    assert candidate.policy.state_dict().keys() == initial_policy_state.keys()
    _assert_nested_equal(candidate.policy.state_dict(), initial_policy_state)
    torch.testing.assert_close(candidate.log_alpha, initial_alpha, rtol=0, atol=0)


def test_candidates_reject_pql_without_replacing_the_original_method():
    learner = _make_learner(_TinyActor().state_dict(), _TinyCritic().state_dict())
    original_method = learner._train_sac_transition
    learner.config["return_estimator"] = "pql"

    with pytest.raises(ValueError, match="n_step"):
        install_variant(learner, "obs_only")

    assert learner._train_sac_transition == original_method


def test_installing_stock_restores_original_bound_method():
    learner = _make_learner(_TinyActor().state_dict(), _TinyCritic().state_dict())
    original_method = learner._train_sac_transition

    install_variant(learner, "cached")
    install_variant(learner, "stock")

    assert learner._train_sac_transition == original_method


@pytest.mark.skipif(
    os.environ.get("SAC_PROFILE_CUDA_TEST") != "1" or not torch.cuda.is_available(),
    reason="set SAC_PROFILE_CUDA_TEST=1 on an idle CUDA device to run pinned-copy parity",
)
@pytest.mark.parametrize("variant", ["obs_only", "cached"])
def test_cuda_pinned_replay_candidates_match_stock(variant):
    torch.manual_seed(47)
    torch.cuda.manual_seed_all(47)
    actor_state = copy.deepcopy(_TinyActor().state_dict())
    critic_state = copy.deepcopy(_TinyCritic().state_dict())
    stock = _make_learner(
        actor_state, critic_state, device="cuda", pin_memory=True,
    )
    candidate = _make_learner(
        actor_state, critic_state, device="cuda", pin_memory=True,
    )
    install_variant(stock, "stock")
    install_variant(candidate, variant)

    torch.manual_seed(113)
    stock_losses = defaultdict(float)
    stock_updates = stock._train_sac_transition(stock_losses, lambda *_args, **_kwargs: None, 0)
    torch.cuda.synchronize()
    torch.manual_seed(113)
    candidate_losses = defaultdict(float)
    candidate_updates = candidate._train_sac_transition(
        candidate_losses, lambda *_args, **_kwargs: None, 0,
    )
    torch.cuda.synchronize()

    assert stock_updates == candidate_updates == 4
    assert dict(stock_losses) == dict(candidate_losses)
    _assert_learner_state_equal(candidate, stock)
    assert sum(indices.numel() for indices in stock.replay.sampled_indices) == 28
    assert sum(indices.numel() for indices in candidate.replay.sampled_indices) == 14
