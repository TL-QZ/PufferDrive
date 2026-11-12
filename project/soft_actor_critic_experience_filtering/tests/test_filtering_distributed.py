"""Two-rank CPU/Gloo checks for globally shared TD-error thresholds."""

from datetime import timedelta
import copy
from collections import defaultdict
from contextlib import nullcontext
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from pufferlib import pufferl_sac
from pufferlib.sac_replay import TensorReplay
from project.soft_actor_critic_experience_filtering.filtering import (
    ExperienceFilter,
    normalize_experience_filter_config,
)


_WORLD_SIZE = 2


class _TinyTwinCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.q1 = nn.Linear(3, 2)
        self.q2 = nn.Linear(3, 2)

    def forward(self, observations):
        return self.q1(observations), self.q2(observations)


class _TinyActor(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward(self, observations):
        return self.linear(observations)


def _make_filter():
    config = normalize_experience_filter_config({
        "enabled": True,
        "ewma_beta": 0.25,
        "threshold_scale": 0.5,
    })
    return ExperienceFilter(config, device="cpu")


def _assert_rank_agreement(tensors):
    for tensor in tensors:
        tensor = tensor.detach().contiguous()
        gathered = [torch.empty_like(tensor) for _ in range(_WORLD_SIZE)]
        dist.all_gather(gathered, tensor)
        for other in gathered[1:]:
            torch.testing.assert_close(other, gathered[0], rtol=1e-6, atol=1e-7)


def _run_distributed_gradient_oracle(rank, scenario):
    torch.manual_seed(771)
    initial_critic = _TinyTwinCritic()
    oracle_critic = copy.deepcopy(initial_critic)
    distributed_critic = copy.deepcopy(initial_critic)
    ddp_critic = torch.nn.parallel.DistributedDataParallel(distributed_critic)

    observations = (
        torch.tensor([
            [2.0, 0.0, 0.0], [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0], [2.0, 2.0, 2.0],
        ]) if rank == 0 else torch.tensor([
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 1.0, 1.0],
        ])
    )
    actions = torch.tensor([0, 1, 0, 1])
    targets = (
        torch.tensor([1.0, -1.0, 2.0, 0.5]) if rank == 0
        else torch.tensor([0.7, 1.1, -0.2, 4.0])
    )
    replay = TensorReplay(4, (3,))
    replay.add(observations, actions, torch.zeros(4), observations + 0.1, torch.ones(4))

    experience_filter = _make_filter()
    if scenario == "uneven":
        local_scores = (
            torch.tensor([10.0, 2.0, 1.0, 0.0]) if rank == 0
            else torch.tensor([8.0, 5.0, 3.0, 2.0])
        )
        expected_local_mask = (
            torch.tensor([True, False, False, False]) if rank == 0
            else torch.tensor([True, True, False, False])
        )
    else:
        local_scores = (
            torch.tensor([1.0, 2.0, 3.0, 4.0]) if rank == 0
            else torch.tensor([10.0, 6.0, 5.0, 1.0])
        )
        expected_local_mask = (
            torch.tensor([False, False, False, False]) if rank == 0
            else torch.tensor([True, True, True, False])
        )
    keep_mask, selection = experience_filter.select(local_scores)
    torch.testing.assert_close(keep_mask.cpu(), expected_local_mask)
    gathered_local_count = [torch.zeros((), dtype=torch.int64) for _ in range(_WORLD_SIZE)]
    dist.all_gather(gathered_local_count, torch.tensor(int(expected_local_mask.sum())))
    global_retained_count = sum(int(value.item()) for value in gathered_local_count)
    assert selection["retained_count"] == global_retained_count

    learner = object.__new__(pufferl_sac.PuffeSAC)
    learner.device = torch.device("cpu")
    learner.distributed = True
    learner.world_size = _WORLD_SIZE
    learner.rank = rank
    learner.replay = replay
    learner.uncompiled_critic = distributed_critic
    learner.critic = distributed_critic
    learner.ddp_critic = ddp_critic
    learner.critic_train = ddp_critic
    learner.critic_optimizer = torch.optim.SGD(distributed_critic.parameters(), lr=0.003)
    learner.max_minibatch_size = 2
    learner.amp_context = nullcontext()
    learner.config = {"max_grad_norm": None}
    learner.experience_filter = experience_filter
    learner._filtered_critic_metric_sums = torch.zeros(5, dtype=torch.float64)

    learner._train_filtered_critic(
        torch.arange(4), targets, keep_mask, selection["retained_count"],
        lambda *_args, **_kwargs: None, 0, None,
    )

    oracle_optimizer = torch.optim.SGD(oracle_critic.parameters(), lr=0.003)
    rank_observations = [
        torch.tensor([
            [2.0, 0.0, 0.0], [0.0, 2.0, 0.0],
            [0.0, 0.0, 2.0], [2.0, 2.0, 2.0],
        ]),
        torch.tensor([
            [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 1.0, 1.0],
        ]),
    ]
    rank_actions = [torch.tensor([0, 1, 0, 1])] * _WORLD_SIZE
    rank_targets = [
        torch.tensor([1.0, -1.0, 2.0, 0.5]),
        torch.tensor([0.7, 1.1, -0.2, 4.0]),
    ]
    rank_masks = (
        [
            torch.tensor([True, False, False, False]),
            torch.tensor([True, True, False, False]),
        ] if scenario == "uneven" else [
            torch.tensor([False, False, False, False]),
            torch.tensor([True, True, True, False]),
        ]
    )
    oracle_observations = torch.cat([
        rank_observations[index][rank_masks[index]] for index in range(_WORLD_SIZE)
    ])
    oracle_actions = torch.cat([
        rank_actions[index][rank_masks[index]] for index in range(_WORLD_SIZE)
    ])
    oracle_targets = torch.cat([
        rank_targets[index][rank_masks[index]] for index in range(_WORLD_SIZE)
    ])
    oracle_q1, oracle_q2 = oracle_critic(oracle_observations)
    oracle_loss_rows = (
        (oracle_q1.gather(1, oracle_actions[:, None]).squeeze(1) - oracle_targets).square()
        + (oracle_q2.gather(1, oracle_actions[:, None]).squeeze(1) - oracle_targets).square()
    )
    oracle_optimizer.zero_grad(set_to_none=True)
    (oracle_loss_rows.sum() / global_retained_count).backward()
    oracle_optimizer.step()

    distributed_state = {
        key: value.detach().cpu().clone()
        for key, value in distributed_critic.state_dict().items()
    }
    oracle_state = {
        key: value.detach().cpu().clone()
        for key, value in oracle_critic.state_dict().items()
    }
    for key in oracle_state:
        torch.testing.assert_close(distributed_state[key], oracle_state[key], rtol=1e-5, atol=1e-7)
    for distributed_parameter, oracle_parameter in zip(
        distributed_critic.parameters(), oracle_critic.parameters(),
    ):
        torch.testing.assert_close(
            distributed_parameter.grad, oracle_parameter.grad, rtol=1e-5, atol=1e-7,
        )
    gathered_states = [None] * _WORLD_SIZE
    dist.all_gather_object(gathered_states, distributed_state)
    for key in distributed_state:
        torch.testing.assert_close(gathered_states[0][key], gathered_states[1][key])
    assert experience_filter.metrics()["filter/update_steps"] == 1

    # Actor gradients use the same global selected-row denominator as the critic.
    torch.manual_seed(772)
    initial_actor = _TinyActor()
    oracle_actor = copy.deepcopy(initial_actor)
    distributed_actor = copy.deepcopy(initial_actor)
    ddp_actor = torch.nn.parallel.DistributedDataParallel(distributed_actor)
    local_mask = rank_masks[rank]
    selected_observations = rank_observations[rank][local_mask]
    actor_observations = (
        selected_observations
        if selected_observations.numel()
        else torch.zeros((1, 3), dtype=torch.float32)
    )
    actor_logits = ddp_actor(actor_observations)
    if selected_observations.numel():
        local_actor_loss_sum = actor_logits.square().sum(-1).sum()
        actor_loss = local_actor_loss_sum * (_WORLD_SIZE / global_retained_count)
    else:
        actor_loss = actor_logits.sum() * 0.0
    distributed_actor_optimizer = torch.optim.SGD(distributed_actor.parameters(), lr=0.003)
    distributed_actor_optimizer.zero_grad(set_to_none=True)
    actor_loss.backward()

    oracle_actor_optimizer = torch.optim.SGD(oracle_actor.parameters(), lr=0.003)
    oracle_actor_observations = torch.cat([
        rank_observations[index][rank_masks[index]] for index in range(_WORLD_SIZE)
    ])
    oracle_actor_loss = oracle_actor(oracle_actor_observations).square().sum(-1).mean()
    oracle_actor_optimizer.zero_grad(set_to_none=True)
    oracle_actor_loss.backward()
    for distributed_parameter, oracle_parameter in zip(
        distributed_actor.parameters(), oracle_actor.parameters(),
    ):
        torch.testing.assert_close(
            distributed_parameter.grad, oracle_parameter.grad, rtol=1e-5, atol=1e-7,
        )
    distributed_actor_optimizer.step()
    oracle_actor_optimizer.step()
    for distributed_parameter, oracle_parameter in zip(
        distributed_actor.parameters(), oracle_actor.parameters(),
    ):
        torch.testing.assert_close(
            distributed_parameter, oracle_parameter, rtol=1e-5, atol=1e-7,
        )
    return distributed_state


def _run_enabled_transition_with_empty_rank(rank):
    torch.manual_seed(903)
    policy = _TinyActor()
    critic = _TinyTwinCritic()
    initial_policy_state = copy.deepcopy(policy.state_dict())
    for parameter in critic.parameters():
        nn.init.zeros_(parameter)
    initial_critic_state = copy.deepcopy(critic.state_dict())
    ddp_policy = torch.nn.parallel.DistributedDataParallel(policy)
    ddp_critic = torch.nn.parallel.DistributedDataParallel(critic)

    observations = torch.tensor([
        [0.1, 0.2, 0.3], [0.4, -0.2, 0.1],
    ]) + float(rank)
    rewards = torch.tensor([1.0, 2.0] if rank == 0 else [10.0, 6.0])
    replay = TensorReplay(2, (3,))
    replay.add(
        observations,
        torch.tensor([0, 1]),
        rewards,
        observations + 0.1,
        torch.zeros(2),
    )

    learner = object.__new__(pufferl_sac.PuffeSAC)
    learner.device = torch.device("cpu")
    learner.distributed = True
    learner.world_size = _WORLD_SIZE
    learner.rank = rank
    learner.replay = replay
    learner.batch_size = 2
    learner.minibatch_size = 2
    learner.max_minibatch_size = 1
    learner.config = {
        "return_estimator": "n_step",
        "update_epochs": 1,
        "pql_horizon": 4,
        "pql_lambda": 0.6,
        "critic_entropy_in_target": True,
        "learn_alpha": True,
        "target_entropy": 0.5,
        "tau": 0.2,
        "max_grad_norm": None,
        "log_collapse_diagnostics": False,
    }
    learner.uncompiled_policy = policy
    learner.policy = policy
    learner.ddp_policy = ddp_policy
    learner.policy_train = ddp_policy
    learner.uncompiled_critic = critic
    learner.critic = critic
    learner.ddp_critic = ddp_critic
    learner.critic_train = ddp_critic
    learner.uncompiled_target_critic = copy.deepcopy(critic).requires_grad_(False).eval()
    learner.target_critic = learner.uncompiled_target_critic
    learner.amp_context = nullcontext()
    learner.log_alpha = nn.Parameter(torch.tensor(-1.0))
    learner.optimizer = torch.optim.SGD(policy.parameters(), lr=0.01)
    learner.critic_optimizer = torch.optim.SGD(critic.parameters(), lr=0.01)
    learner.alpha_optimizer = torch.optim.SGD([learner.log_alpha], lr=0.01)
    learner.experience_filter = ExperienceFilter({"threshold_scale": 0.5}, "cpu")
    learner.target_entropy_controller = pufferl_sac.AdaptiveTargetEntropyController(
        0.5,
        ema_decay=0.0,
        mean_threshold=10.0,
        std_threshold=10.0,
        decay_factor=0.5,
        required_count=10,
    )
    learner.agent_steps = 4
    learner.learner_updates = 0
    learner.sampled_transitions = 0
    learner._filtered_critic_metric_sums = torch.zeros(5, dtype=torch.float64)
    learner._filtered_actor_metric_sums = torch.zeros(5, dtype=torch.float64)

    actor_observation_batches = []
    original_policy_loss_rows = learner._policy_loss_rows

    def record_actor_rows(policy_observations, record_diagnostics=True):
        actor_observation_batches.append((
            policy_observations.detach().cpu().clone(), record_diagnostics,
        ))
        return original_policy_loss_rows(
            policy_observations, record_diagnostics=record_diagnostics,
        )

    learner._policy_loss_rows = record_actor_rows
    candidate_index_batches = []
    original_score = learner._score_experience_filter_candidates

    def record_candidate_indices(replay_indices, targets):
        candidate_index_batches.append(replay_indices.detach().cpu().clone())
        return original_score(replay_indices, targets)

    learner._score_experience_filter_candidates = record_candidate_indices
    selected_masks = []
    original_select = learner.experience_filter.select

    def record_selected_mask(scores, **kwargs):
        mask, metrics = original_select(scores, **kwargs)
        selected_masks.append(mask.detach().cpu().clone())
        return mask, metrics

    learner.experience_filter.select = record_selected_mask
    observed_entropies = []
    original_observe = learner.target_entropy_controller.observe

    def record_entropy(entropy, *, agent_steps):
        observed_entropies.append(entropy)
        return original_observe(entropy, agent_steps=agent_steps)

    learner.target_entropy_controller.observe = record_entropy
    updates = learner._train_sac_transition(
        defaultdict(float), lambda *_args, **_kwargs: None, 0,
    )

    assert updates == 1
    assert learner.learner_updates == 1
    assert learner.sampled_transitions == 4
    assert len(candidate_index_batches) == 1
    assert len(selected_masks) == 1
    expected_local_retained = 0 if rank == 0 else 2
    assert selected_masks[0].sum().item() == expected_local_retained
    expected_actor_observations = replay.get_batch(
        candidate_index_batches[0][selected_masks[0]],
    )[0]
    recorded_actor_observations = (
        torch.cat([observations for observations, _recording in actor_observation_batches])
        if actor_observation_batches else torch.empty((0, 3))
    )
    if expected_local_retained == 0 and actor_observation_batches:
        # The first candidate may be forwarded to keep DDP rounds aligned, but
        # it must be marked as a non-training sentinel and receive zero weight.
        assert len(actor_observation_batches) == 2
        expected_sentinel = replay.get_batch(candidate_index_batches[0][:1])[0]
        for sentinel_observations, record_diagnostics in actor_observation_batches:
            assert record_diagnostics is False
            torch.testing.assert_close(sentinel_observations, expected_sentinel)
    else:
        torch.testing.assert_close(recorded_actor_observations, expected_actor_observations)
    assert expected_actor_observations.shape[0] == expected_local_retained
    filter_metrics = learner.experience_filter.metrics()
    assert filter_metrics["filter/candidate_count"] == 4
    assert filter_metrics["filter/retained_count"] == 2
    assert filter_metrics["filter/actor_rows"] == 2
    assert learner.experience_filter.state_dict()["cycle"]["actor_steps"] == 1
    assert learner.experience_filter.state_dict()["cycle"]["alpha_steps"] == 1
    assert learner.experience_filter.state_dict()["cycle"]["target_steps"] == 1
    assert filter_metrics["filter/local_empty_ranks"] == 1
    assert filter_metrics["filter/update_steps"] == 1
    assert learner.log_alpha.item() != pytest.approx(-1.0)
    assert any(parameter.detach().abs().sum().item() > 0 for parameter in policy.parameters())
    assert len(observed_entropies) == 1

    # Single-process oracle over the globally selected rows: rank 1 retains both
    # rows because their rewards (10 and 6) exceed the shared threshold of 5.
    oracle_policy = _TinyActor()
    oracle_policy.load_state_dict(initial_policy_state)
    oracle_critic = _TinyTwinCritic()
    oracle_critic.load_state_dict(initial_critic_state)
    oracle_target_critic = copy.deepcopy(oracle_critic).requires_grad_(False).eval()
    oracle_observations = torch.tensor([
        [0.1, 0.2, 0.3], [0.4, -0.2, 0.1],
        [1.1, 1.2, 1.3], [1.4, 0.8, 1.1],
    ])[2:]
    oracle_actions = torch.tensor([0, 1])
    oracle_targets = torch.tensor([10.0, 6.0])
    oracle_critic_optimizer = torch.optim.SGD(oracle_critic.parameters(), lr=0.01)
    oracle_q1, oracle_q2 = oracle_critic(oracle_observations)
    oracle_critic_rows = (
        (oracle_q1.gather(1, oracle_actions[:, None]).squeeze(1) - oracle_targets).square()
        + (oracle_q2.gather(1, oracle_actions[:, None]).squeeze(1) - oracle_targets).square()
    )
    oracle_critic_optimizer.zero_grad(set_to_none=True)
    oracle_critic_rows.mean().backward()
    oracle_critic_optimizer.step()

    oracle_logits = oracle_policy(oracle_observations)
    oracle_logprobs = torch.nn.functional.log_softmax(oracle_logits, dim=-1)
    oracle_probs = oracle_logprobs.exp()
    with torch.no_grad():
        oracle_q1, oracle_q2 = oracle_critic(oracle_observations)
        oracle_minimum_q = torch.minimum(oracle_q1, oracle_q2)
    oracle_policy_loss = (
        oracle_probs * (torch.tensor(-1.0).exp() * oracle_logprobs - oracle_minimum_q)
    ).sum(-1).mean()
    oracle_entropy = -(oracle_probs * oracle_logprobs).sum(-1).mean()
    oracle_policy_optimizer = torch.optim.SGD(oracle_policy.parameters(), lr=0.01)
    oracle_policy_optimizer.zero_grad(set_to_none=True)
    oracle_policy_loss.backward()
    oracle_policy_optimizer.step()

    oracle_log_alpha = nn.Parameter(torch.tensor(-1.0))
    oracle_alpha_optimizer = torch.optim.SGD([oracle_log_alpha], lr=0.01)
    oracle_alpha_loss = oracle_log_alpha * (oracle_entropy.detach() - 0.5)
    oracle_alpha_optimizer.zero_grad(set_to_none=True)
    oracle_alpha_loss.backward()
    oracle_alpha_optimizer.step()
    with torch.no_grad():
        for target_parameter, online_parameter in zip(
            oracle_target_critic.parameters(), oracle_critic.parameters(),
        ):
            target_parameter.lerp_(online_parameter, 0.2)

    torch.testing.assert_close(policy.state_dict()["linear.weight"], oracle_policy.state_dict()["linear.weight"], rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(policy.state_dict()["linear.bias"], oracle_policy.state_dict()["linear.bias"], rtol=1e-5, atol=1e-7)
    for actual_tensor, expected_tensor in zip(critic.state_dict().values(), oracle_critic.state_dict().values()):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-5, atol=1e-7)
    for actual_tensor, expected_tensor in zip(learner.target_critic.state_dict().values(), oracle_target_critic.state_dict().values()):
        torch.testing.assert_close(actual_tensor, expected_tensor, rtol=1e-5, atol=1e-7)
    torch.testing.assert_close(learner.log_alpha, oracle_log_alpha, rtol=1e-5, atol=1e-7)
    assert observed_entropies == pytest.approx([oracle_entropy.item()], rel=1e-5, abs=1e-7)
    _assert_rank_agreement([
        *policy.parameters(), *critic.parameters(), *learner.target_critic.parameters(),
        learner.log_alpha,
    ])


def _worker(rank, init_method, result_dir):
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=_WORLD_SIZE,
        timeout=timedelta(seconds=60),
    )
    try:
        selector = _make_filter()

        # Uneven retained counts: global max 10 gives threshold 5, so rank 0
        # keeps one row while rank 1 keeps two, including the equality tie.
        uneven_scores = (
            torch.tensor([10.0, 2.0, 1.0, 0.0]) if rank == 0
            else torch.tensor([8.0, 5.0, 3.0, 2.0])
        )
        uneven_mask, uneven_metrics = selector.select(uneven_scores)
        uneven_logs = selector.metrics()
        selector.begin_cycle()

        # Rank 0 has no retained rows, but rank 1's two rows still count.
        locally_empty_scores = (
            torch.tensor([1.0, 2.0, 3.0, 4.0]) if rank == 0
            else torch.tensor([10.0, 5.0, 4.0, 3.0])
        )
        locally_empty_mask, locally_empty_metrics = selector.select(locally_empty_scores)
        locally_empty_logs = selector.metrics()
        selector.begin_cycle()

        # A later all-zero batch stays globally empty because the positive EMA
        # threshold is not silently relaxed.
        globally_empty_mask, globally_empty_metrics = selector.select(torch.zeros(3))
        globally_empty_logs = selector.metrics()

        _run_distributed_gradient_oracle(rank, "uneven")
        _run_distributed_gradient_oracle(rank, "empty")
        _run_enabled_transition_with_empty_rank(rank)

        invalid_selector = _make_filter()
        invalid_scores = (
            torch.tensor([float("nan"), 1.0]) if rank == 0
            else torch.tensor([2.0, 1.0])
        )
        try:
            invalid_selector.select(invalid_scores)
        except FloatingPointError as error:
            finite_error = str(error)
        else:
            finite_error = None
        gathered_finite_errors = [None] * _WORLD_SIZE
        dist.all_gather_object(gathered_finite_errors, finite_error)
        assert gathered_finite_errors == [
            "Nonfinite or negative SAC critic TD-error score detected on at least one rank; "
            "scores must be finite and nonnegative",
        ] * _WORLD_SIZE, repr(gathered_finite_errors)

        gathered = [None] * _WORLD_SIZE
        dist.all_gather_object(
            gathered,
            {
                "state": selector.state_dict(),
                "uneven_logs": uneven_logs,
                "locally_empty_logs": locally_empty_logs,
                "globally_empty_logs": globally_empty_logs,
                "uneven_metrics": uneven_metrics,
                "locally_empty_metrics": locally_empty_metrics,
                "globally_empty_metrics": globally_empty_metrics,
            },
        )
        torch.save({
            "uneven_mask": uneven_mask,
            "locally_empty_mask": locally_empty_mask,
            "globally_empty_mask": globally_empty_mask,
            "uneven_metrics": uneven_metrics,
            "locally_empty_metrics": locally_empty_metrics,
            "globally_empty_metrics": globally_empty_metrics,
            "uneven_logs": uneven_logs,
            "locally_empty_logs": locally_empty_logs,
            "globally_empty_logs": globally_empty_logs,
            "finite_errors": gathered_finite_errors,
            "gathered": gathered,
        }, Path(result_dir) / f"rank-{rank}.pt")
        dist.barrier()
    finally:
        dist.destroy_process_group()


def _spawn(worker, args, timeout_seconds=75):
    context = mp.spawn(worker, args=args, nprocs=_WORLD_SIZE, join=False)
    for _ in range(timeout_seconds):
        if context.join(timeout=1):
            return
    for process in context.processes:
        if process.is_alive():
            process.terminate()
    for process in context.processes:
        process.join(timeout=5)
    pytest.fail(f"two-rank Gloo worker exceeded {timeout_seconds}s timeout")


@pytest.mark.skipif(
    not dist.is_available() or not getattr(dist, "is_gloo_available", lambda: False)(),
    reason="PyTorch Gloo distributed support is unavailable",
)
def test_gloo_selection_matches_concatenated_oracle_with_empty_ranks(tmp_path):
    result_dir = tmp_path / "results"
    result_dir.mkdir()
    _spawn(_worker, (f"file://{tmp_path / 'gloo-init'}", str(result_dir)))

    results = [
        torch.load(result_dir / f"rank-{rank}.pt", map_location="cpu", weights_only=False)
        for rank in range(_WORLD_SIZE)
    ]

    # Concatenated single-process oracle: beta=.25, scale=.5, initial max=10.
    uneven_all_scores = torch.tensor([10.0, 2.0, 1.0, 0.0, 8.0, 5.0, 3.0, 2.0])
    uneven_oracle_mask = uneven_all_scores >= 5.0
    assert results[0]["uneven_mask"].tolist() == uneven_oracle_mask[:4].tolist()
    assert results[1]["uneven_mask"].tolist() == uneven_oracle_mask[4:].tolist()
    assert all(result["uneven_metrics"]["retained_count"] == 3 for result in results)
    assert all(result["uneven_metrics"]["candidate_count"] == 8 for result in results)
    assert all(result["uneven_metrics"]["current_max"] == pytest.approx(10.0) for result in results)
    assert all(result["uneven_metrics"]["ema_max"] == pytest.approx(10.0) for result in results)

    local_empty_all_scores = torch.tensor([1.0, 2.0, 3.0, 4.0, 10.0, 5.0, 4.0, 3.0])
    # EMA remains 10, making the threshold 5.
    local_empty_oracle_mask = local_empty_all_scores >= 5.0
    assert results[0]["locally_empty_mask"].tolist() == local_empty_oracle_mask[:4].tolist()
    assert results[1]["locally_empty_mask"].tolist() == local_empty_oracle_mask[4:].tolist()
    assert results[0]["locally_empty_mask"].sum().item() == 0
    assert results[1]["locally_empty_mask"].sum().item() == 2
    assert all(result["locally_empty_metrics"]["retained_count"] == 2 for result in results)
    assert all(result["locally_empty_metrics"]["candidate_count"] == 8 for result in results)
    assert all(result["locally_empty_metrics"]["local_empty_ranks"] == 1 for result in results)
    assert all(result["locally_empty_logs"]["filter/candidate_count"] == 8 for result in results)
    assert all(result["locally_empty_logs"]["filter/retained_count"] == 2 for result in results)
    assert all(
        result["locally_empty_logs"]["filter/retained_fraction"] == pytest.approx(2 / 8)
        for result in results
    )
    assert all(
        result["locally_empty_logs"]["filter/filtered_fraction"] == pytest.approx(6 / 8)
        for result in results
    )

    # Current max=0 updates EMA to 7.5; positive threshold leaves no rows.
    assert all(not result["globally_empty_mask"].any() for result in results)
    assert all(result["globally_empty_metrics"]["retained_count"] == 0 for result in results)
    assert all(result["globally_empty_metrics"]["candidate_count"] == 6 for result in results)
    assert all(result["globally_empty_logs"]["filter/empty_batches"] == 1 for result in results)
    assert all(result["globally_empty_metrics"]["ema_max"] == pytest.approx(7.5) for result in results)
    assert all(result["globally_empty_metrics"]["threshold"] == pytest.approx(3.75) for result in results)
    assert all(
        result["globally_empty_logs"]["filter/cumulative_candidate_count"] == 22
        for result in results
    )
    assert all(
        result["globally_empty_logs"]["filter/cumulative_retained_count"] == 5
        for result in results
    )
    assert all(
        result["globally_empty_logs"]["filter/cumulative_retained_fraction"] == pytest.approx(5 / 22)
        for result in results
    )

    for result in results:
        assert result["finite_errors"] == [
            "Nonfinite or negative SAC critic TD-error score detected on at least one rank; "
            "scores must be finite and nonnegative",
        ] * _WORLD_SIZE
        gathered = result["gathered"]
        assert gathered[0]["state"] == gathered[1]["state"]
        assert gathered[0]["uneven_logs"] == gathered[1]["uneven_logs"]
        assert gathered[0]["locally_empty_logs"] == gathered[1]["locally_empty_logs"]
        assert gathered[0]["globally_empty_logs"] == gathered[1]["globally_empty_logs"]
        assert gathered[0]["uneven_metrics"] == gathered[1]["uneven_metrics"]
        assert gathered[0]["locally_empty_metrics"] == gathered[1]["locally_empty_metrics"]
        assert gathered[0]["globally_empty_metrics"] == gathered[1]["globally_empty_metrics"]
