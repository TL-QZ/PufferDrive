"""CPU integration checks for experience filtering in the SAC learner."""

import copy
from collections import defaultdict
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from pufferlib import pufferl, pufferl_sac
from pufferlib.sac_replay import TensorReplay
from pufferlib.sac_sequence_replay import SequenceReplay
from project.soft_actor_critic_experience_filtering.filtering import ExperienceFilter


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


def _make_tensor_replay():
    replay = TensorReplay(4, (3,))
    observations = torch.tensor([
        [0.1, 0.2, 0.3],
        [0.4, -0.2, 0.1],
        [-0.3, 0.5, 0.2],
        [0.7, 0.1, -0.4],
    ])
    replay.add(
        observations,
        torch.tensor([0, 1, 0, 1]),
        torch.tensor([0.2, -0.1, 0.5, 0.3]),
        observations * 0.7 + 0.15,
        torch.tensor([0.9, 0.9, 0.0, 0.9]),
    )
    return replay


def _make_sequence_replay():
    replay = SequenceReplay(4, (3,), num_agents=1)
    for time_idx in range(4):
        observation = torch.tensor([[0.2 + time_idx, 0.3 - 0.1 * time_idx, 0.1 * time_idx]])
        replay.add_sequence(
            torch.tensor([0]), observation,
            torch.tensor([time_idx % 2]),
            torch.tensor([0.1 * (time_idx + 1)]), observation + 0.25,
            torch.tensor([0.9 if time_idx < 3 else 0.0]),
            torch.tensor([False]), torch.tensor([False]),
        )
    return replay


def _make_training_learner(replay, actor_state, critic_state, *, return_estimator,
                           experience_filter=None, threshold_scale=0.0, max_minibatch_size=2,
                           apply_to="actor_critic"):
    learner = object.__new__(pufferl_sac.PuffeSAC)
    learner.device = torch.device("cpu")
    learner.uncompiled_policy = _TinyActor()
    learner.uncompiled_policy.load_state_dict(actor_state)
    learner.policy = learner.uncompiled_policy
    learner.policy_train = learner.policy
    learner.ddp_policy = learner.policy
    learner.uncompiled_critic = _TinyCritic()
    learner.uncompiled_critic.load_state_dict(critic_state)
    learner.critic = learner.uncompiled_critic
    learner.critic_train = learner.critic
    learner.ddp_critic = learner.critic
    learner.uncompiled_target_critic = copy.deepcopy(learner.uncompiled_critic)
    learner.uncompiled_target_critic.requires_grad_(False).eval()
    learner.target_critic = learner.uncompiled_target_critic
    learner.amp_context = nullcontext()
    learner.log_alpha = nn.Parameter(torch.tensor(-1.2))
    learner.optimizer = torch.optim.Adam(learner.policy.parameters(), lr=0.003)
    learner.critic_optimizer = torch.optim.Adam(learner.critic.parameters(), lr=0.004)
    learner.alpha_optimizer = torch.optim.Adam([learner.log_alpha], lr=0.002)
    learner.config = {
        "return_estimator": return_estimator,
        "pql_horizon": 8,
        "pql_lambda": 0.7,
        "update_epochs": 1,
        "target_entropy": 0.7,
        "tau": 0.15,
        "max_grad_norm": None,
        "critic_entropy_in_target": True,
        "learn_alpha": True,
        "log_collapse_diagnostics": False,
    }
    learner.target_entropy_controller = pufferl_sac.AdaptiveTargetEntropyController(
        0.7,
        ema_decay=0.0,
        mean_threshold=10.0,
        std_threshold=10.0,
        decay_factor=0.5,
        required_count=1,
    )
    learner.batch_size = len(replay)
    learner.minibatch_size = len(replay)
    learner.max_minibatch_size = max_minibatch_size
    learner.replay = replay
    learner.distributed = False
    learner.world_size = 1
    learner.learner_updates = 0
    learner.sampled_transitions = 0
    learner.agent_steps = 4
    learner.experience_filter = (
        ExperienceFilter(
            {"threshold_scale": threshold_scale, "apply_to": apply_to}, torch.device("cpu"),
        )
        if experience_filter else None
    )
    learner._filtered_critic_metric_sums = torch.zeros(5, dtype=torch.float64)
    learner._filtered_actor_metric_sums = torch.zeros(5, dtype=torch.float64)
    return learner


def _effective_losses(learner, losses):
    effective = dict(losses)
    metric_sums = learner._filtered_critic_metric_sums
    retained_count = int(metric_sums[4].item())
    if retained_count:
        effective.update({
            "critic_loss": (metric_sums[0] / retained_count).item(),
            "q1": (metric_sums[1] / retained_count).item(),
            "q2": (metric_sums[2] / retained_count).item(),
            "target": (metric_sums[3] / retained_count).item(),
        })
    return effective


def _parameters(module):
    return [parameter.detach().clone() for parameter in module.parameters()]


def _gradients(module):
    return [None if parameter.grad is None else parameter.grad.detach().clone() for parameter in module.parameters()]


def _assert_controller_state_close(first_state, second_state):
    assert first_state.keys() == second_state.keys()
    for key, first_value in first_state.items():
        second_value = second_state[key]
        if isinstance(first_value, float):
            assert second_value == pytest.approx(first_value, rel=1e-6, abs=1e-7)
        else:
            assert second_value == first_value


@pytest.mark.parametrize("return_estimator", ["n_step", "pql"])
def test_zero_scale_matches_unfiltered_adaptive_entropy_sac_update(return_estimator):
    torch.manual_seed(23)
    replay_factory = _make_sequence_replay if return_estimator == "pql" else _make_tensor_replay
    actor_state = copy.deepcopy(_TinyActor().state_dict())
    critic_state = copy.deepcopy(_TinyCritic().state_dict())
    unfiltered = _make_training_learner(
        replay_factory(), actor_state, critic_state, return_estimator=return_estimator,
    )
    filtered = _make_training_learner(
        replay_factory(), actor_state, critic_state, return_estimator=return_estimator,
        experience_filter=True,
    )

    torch.manual_seed(101)
    unfiltered_losses = defaultdict(float)
    unfiltered_updates = unfiltered._train_sac_transition(
        unfiltered_losses, lambda *_args, **_kwargs: None, 0,
    )
    torch.manual_seed(101)
    filtered_losses = defaultdict(float)
    filtered_updates = filtered._train_sac_transition(
        filtered_losses, lambda *_args, **_kwargs: None, 0,
    )

    assert unfiltered_updates == filtered_updates == 1
    assert unfiltered.learner_updates == filtered.learner_updates == 1
    assert unfiltered.sampled_transitions == filtered.sampled_transitions == 4
    for unfiltered_module, filtered_module in (
        (unfiltered.policy, filtered.policy),
        (unfiltered.critic, filtered.critic),
        (unfiltered.target_critic, filtered.target_critic),
    ):
        for unfiltered_parameter, filtered_parameter in zip(
            _parameters(unfiltered_module), _parameters(filtered_module),
        ):
            torch.testing.assert_close(unfiltered_parameter, filtered_parameter, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(unfiltered.log_alpha, filtered.log_alpha, rtol=1e-5, atol=1e-6)
    _assert_controller_state_close(
        unfiltered.target_entropy_controller.state_dict(),
        filtered.target_entropy_controller.state_dict(),
    )

    unfiltered_effective = _effective_losses(unfiltered, unfiltered_losses)
    filtered_effective = _effective_losses(filtered, filtered_losses)
    assert unfiltered_effective.keys() == filtered_effective.keys()
    for key, value in unfiltered_effective.items():
        assert filtered_effective[key] == pytest.approx(value, rel=1e-5, abs=1e-6)

    filter_metrics = filtered.experience_filter.metrics()
    assert filter_metrics["filter/candidate_count"] == 4
    assert filter_metrics["filter/retained_count"] == 4
    assert filter_metrics["filter/update_steps"] == 1
    assert filtered.target_entropy_controller.decay_count == 1


def test_pql_filter_scores_masks_gradients_and_updates_are_microbatch_invariant(monkeypatch):
    torch.manual_seed(41)
    actor_state = copy.deepcopy(_TinyActor().state_dict())
    critic_state = copy.deepcopy(_TinyCritic().state_dict())
    learners = [
        _make_training_learner(
            _make_sequence_replay(), actor_state, critic_state,
            return_estimator="pql", experience_filter=True, threshold_scale=0.5,
            max_minibatch_size=max_minibatch_size,
        )
        for max_minibatch_size in (1, 4)
    ]
    target_batches = [[], []]
    selection_batches = [[], []]
    for index, learner in enumerate(learners):
        original_prepare = learner._prepare_pql_targets
        original_select = learner.experience_filter.select

        def record_targets(replay_indices, segment_offsets, *, _original=original_prepare,
                           _captured=target_batches[index]):
            targets = _original(replay_indices, segment_offsets)
            _captured.append(targets.detach().clone())
            return targets

        def record_selection(scores, *args, _original=original_select,
                             _captured=selection_batches[index], **kwargs):
            keep_mask, metrics = _original(scores, *args, **kwargs)
            _captured.append((scores.detach().clone(), keep_mask.clone(), metrics))
            return keep_mask, metrics

        monkeypatch.setattr(learner, "_prepare_pql_targets", record_targets)
        monkeypatch.setattr(learner.experience_filter, "select", record_selection)

    recorded_losses = []
    for learner in learners:
        torch.manual_seed(109)
        losses = defaultdict(float)
        updates = learner._train_sac_transition(losses, lambda *_args, **_kwargs: None, 0)
        assert updates == 1
        recorded_losses.append(losses)

    assert [len(targets) for targets in target_batches] == [1, 1]
    torch.testing.assert_close(target_batches[0][0], target_batches[1][0], rtol=1e-6, atol=1e-7)
    assert [len(selections) for selections in selection_batches] == [1, 1]
    score_a, mask_a, metrics_a = selection_batches[0][0]
    score_b, mask_b, metrics_b = selection_batches[1][0]
    torch.testing.assert_close(score_a, score_b, rtol=1e-5, atol=1e-6)
    assert torch.equal(mask_a, mask_b)
    assert 0 < int(mask_a.sum()) < mask_a.numel()
    for metric_name in ("candidate_count", "retained_count", "filtered_count", "empty_batches"):
        assert metrics_a[metric_name] == metrics_b[metric_name]
    for metric_name in ("current_max", "ema_max", "threshold"):
        assert metrics_a[metric_name] == pytest.approx(metrics_b[metric_name], rel=1e-5, abs=1e-6)

    for first, second in zip(learners, learners[1:]):
        for first_module, second_module in (
            (first.policy, second.policy),
            (first.critic, second.critic),
            (first.target_critic, second.target_critic),
        ):
            for first_parameter, second_parameter in zip(
                _parameters(first_module), _parameters(second_module),
            ):
                torch.testing.assert_close(first_parameter, second_parameter, rtol=1e-5, atol=1e-6)
            first_gradients = _gradients(first_module)
            second_gradients = _gradients(second_module)
            for first_gradient, second_gradient in zip(first_gradients, second_gradients):
                if first_gradient is None or second_gradient is None:
                    assert first_gradient is second_gradient
                else:
                    torch.testing.assert_close(first_gradient, second_gradient, rtol=1e-5, atol=1e-6)
        torch.testing.assert_close(first.log_alpha, second.log_alpha, rtol=1e-5, atol=1e-6)
        assert first.experience_filter.state_dict()["scoring_batches"] == 1
        assert second.experience_filter.state_dict()["scoring_batches"] == 1
        assert first.experience_filter.state_dict()["cumulative"] == second.experience_filter.state_dict()["cumulative"]
        _assert_controller_state_close(
            first.target_entropy_controller.state_dict(),
            second.target_entropy_controller.state_dict(),
        )
        first_losses = _effective_losses(first, recorded_losses[0])
        second_losses = _effective_losses(second, recorded_losses[1])
        assert first_losses.keys() == second_losses.keys()
        for key, value in first_losses.items():
            assert second_losses[key] == pytest.approx(value, rel=1e-5, abs=1e-6)


def test_filtering_prepares_full_pql_targets_and_keeps_original_target_alignment(monkeypatch):
    torch.manual_seed(31)
    replay = SequenceReplay(6, (3,), num_agents=1)
    for time_idx in range(6):
        observation = torch.tensor([[float(time_idx), 0.1 * time_idx, 1.0]])
        replay.add_sequence(
            torch.tensor([0]), observation, torch.tensor([time_idx % 2]),
            torch.tensor([0.2 + time_idx * 0.03]), observation + 0.2,
            torch.tensor([0.9 if time_idx < 5 else 0.0]),
            torch.tensor([False]), torch.tensor([False]),
        )
    learner = _make_training_learner(
        replay, copy.deepcopy(_TinyActor().state_dict()),
        copy.deepcopy(_TinyCritic().state_dict()), return_estimator="pql",
        experience_filter=True, threshold_scale=0.5,
    )
    prepared_target_batches = []
    critic_rows = []
    actor_observation_batches = []
    original_prepare = learner._prepare_pql_targets
    original_critic_loss_sum = learner._critic_loss_sum
    original_policy_loss_rows = learner._policy_loss_rows

    def record_prepared_targets(replay_indices, segment_offsets):
        targets = original_prepare(replay_indices, segment_offsets)
        prepared_target_batches.append((replay_indices.clone(), targets.clone()))
        return targets

    def record_critic_rows(observations, actions, targets):
        critic_rows.append((observations[:, 0].detach().clone(), targets.detach().clone()))
        return original_critic_loss_sum(observations, actions, targets)

    def record_actor_rows(observations, record_diagnostics=True):
        actor_observation_batches.append(observations.detach().clone())
        return original_policy_loss_rows(observations, record_diagnostics=record_diagnostics)

    monkeypatch.setattr(learner, "_prepare_pql_targets", record_prepared_targets)
    monkeypatch.setattr(learner, "_critic_loss_sum", record_critic_rows)
    monkeypatch.setattr(learner, "_policy_loss_rows", record_actor_rows)

    # Fixed candidate scores isolate position mapping while the production PQL
    # target preparation and filtered critic update run unchanged.
    def index_scores(replay_indices, targets):
        score_by_index = torch.tensor([10.0, 1.0, 9.0, 2.0, 3.0, 8.0])
        return score_by_index[replay_indices]

    monkeypatch.setattr(learner, "_score_experience_filter_candidates", index_scores)
    losses = defaultdict(float)
    learner._train_sac_transition(losses, lambda *_args, **_kwargs: None, 0)

    assert len(prepared_target_batches) == 1
    candidate_indices, all_targets = prepared_target_batches[0]
    assert candidate_indices.numel() == 6
    assert all_targets.shape == (6,)
    assert torch.unique(candidate_indices).numel() == 6
    score_by_index = torch.tensor([10.0, 1.0, 9.0, 2.0, 3.0, 8.0])
    candidate_scores = score_by_index[candidate_indices]
    retained_positions = torch.nonzero(
        candidate_scores >= candidate_scores.max() * 0.5, as_tuple=False,
    ).flatten()
    retained_indices = candidate_indices[retained_positions]
    observed_indices = torch.cat([rows for rows, _targets in critic_rows]).to(torch.int64)
    observed_targets = torch.cat([targets for _rows, targets in critic_rows])
    torch.testing.assert_close(observed_indices, retained_indices)
    torch.testing.assert_close(observed_targets, all_targets[retained_positions])
    expected_actor_observations = learner.replay.get_batch(retained_indices)[0]
    torch.testing.assert_close(torch.cat(actor_observation_batches), expected_actor_observations)
    metrics = learner.experience_filter.metrics()
    assert metrics["filter/candidate_count"] == 6
    assert metrics["filter/retained_count"] == 3
    assert metrics["filter/filtered_count"] == 3
    assert metrics["filter/update_steps"] == 1
    assert metrics["filter/actor_rows"] == 3
    assert learner.experience_filter.state_dict()["cycle"]["actor_steps"] == 1


def test_actor_critic_mode_weights_actor_alpha_and_entropy_to_retained_rows(monkeypatch):
    learner = _make_training_learner(
        _make_tensor_replay(), copy.deepcopy(_TinyActor().state_dict()),
        copy.deepcopy(_TinyCritic().state_dict()), return_estimator="n_step",
        experience_filter=True, threshold_scale=0.5, max_minibatch_size=2,
    )
    learner.optimizer = torch.optim.SGD(learner.policy.parameters(), lr=0.001)
    learner.critic_optimizer = torch.optim.SGD(learner.critic.parameters(), lr=0.001)
    learner.alpha_optimizer = torch.optim.SGD([learner.log_alpha], lr=0.01)
    learner.target_entropy_controller = pufferl_sac.AdaptiveTargetEntropyController(
        0.7,
        ema_decay=0.0,
        mean_threshold=10.0,
        std_threshold=10.0,
        decay_factor=0.5,
        required_count=1,
    )
    candidate_index_batches = []

    def fixed_scores(replay_indices, targets):
        candidate_index_batches.append(replay_indices.detach().clone())
        return torch.tensor([10.0, 8.0, 5.0, 2.0])

    monkeypatch.setattr(
        learner, "_score_experience_filter_candidates", fixed_scores,
    )
    actor_observation_batches = []

    def selected_policy_loss_rows(observations, record_diagnostics=True):
        actor_observation_batches.append(observations.detach().clone())
        logits = learner.policy_train(observations)
        retained_entropy_rows = observations[:, 0] + 0.5
        return logits.sum(dim=-1), retained_entropy_rows.detach()

    monkeypatch.setattr(learner, "_policy_loss_rows", selected_policy_loss_rows)
    observed_entropies = []
    original_observe = learner.target_entropy_controller.observe

    def record_observe(entropy, *, agent_steps):
        observed_entropies.append(entropy)
        return original_observe(entropy, agent_steps=agent_steps)

    monkeypatch.setattr(learner.target_entropy_controller, "observe", record_observe)
    alpha_before = learner.log_alpha.detach().item()
    losses = defaultdict(float)

    updates = learner._train_sac_transition(losses, lambda *_args, **_kwargs: None, 0)

    assert updates == 1
    expected_indices = candidate_index_batches[0][:3]
    expected_observations = learner.replay.get_batch(expected_indices)[0]
    expected_entropy = (expected_observations[:, 0] + 0.5).mean().item()
    expected_log_alpha = alpha_before - 0.01 * (expected_entropy - 0.7)
    torch.testing.assert_close(torch.cat(actor_observation_batches), expected_observations)
    assert observed_entropies == pytest.approx([expected_entropy])
    assert learner.log_alpha.item() == pytest.approx(expected_log_alpha)
    metrics = learner.experience_filter.metrics()
    assert metrics["filter/candidate_count"] == 4
    assert metrics["filter/retained_count"] == 3
    assert metrics["filter/actor_rows"] == 3
    assert learner.experience_filter.state_dict()["cycle"]["actor_steps"] == 1
    assert learner.experience_filter.state_dict()["cycle"]["alpha_steps"] == 1
    assert learner.experience_filter.state_dict()["cycle"]["target_steps"] == 1
    assert learner.sampled_transitions == 4


def test_legacy_critic_only_empty_selection_still_updates_actor_alpha_entropy_and_target():
    torch.manual_seed(37)
    learner = _make_training_learner(
        _make_tensor_replay(), copy.deepcopy(_TinyActor().state_dict()),
        copy.deepcopy(_TinyCritic().state_dict()), return_estimator="n_step",
        experience_filter=True, threshold_scale=2.0, apply_to="critic",
    )
    learner.target_entropy_controller = pufferl_sac.AdaptiveTargetEntropyController(
        0.1,
        ema_decay=0.0,
        mean_threshold=10.0,
        std_threshold=10.0,
        decay_factor=0.5,
        required_count=1,
    )
    with torch.no_grad():
        for parameter in learner.target_critic.parameters():
            parameter.add_(0.25)
    actor_before = _parameters(learner.policy)
    critic_before = _parameters(learner.critic)
    target_before = _parameters(learner.target_critic)
    alpha_before = learner.log_alpha.detach().clone()
    losses = defaultdict(float)

    updates = learner._train_sac_transition(losses, lambda *_args, **_kwargs: None, 0)

    assert updates == learner.learner_updates == 1
    assert learner.experience_filter.metrics()["filter/retained_count"] == 0
    assert learner.experience_filter.metrics()["filter/empty_batches"] == 1
    assert learner.experience_filter.metrics()["filter/update_steps"] == 0
    assert any(not torch.equal(before, after) for before, after in zip(actor_before, _parameters(learner.policy)))
    assert all(torch.equal(before, after) for before, after in zip(critic_before, _parameters(learner.critic)))
    assert any(not torch.equal(before, after) for before, after in zip(target_before, _parameters(learner.target_critic)))
    assert not torch.equal(alpha_before, learner.log_alpha.detach())
    assert learner.target_entropy_controller.decay_count == 1
    assert learner.target_entropy_controller.target_entropy == pytest.approx(0.05)


def test_actor_critic_global_empty_selection_skips_all_updates():
    torch.manual_seed(39)
    learner = _make_training_learner(
        _make_tensor_replay(), copy.deepcopy(_TinyActor().state_dict()),
        copy.deepcopy(_TinyCritic().state_dict()), return_estimator="n_step",
        experience_filter=True, threshold_scale=2.0,
    )
    learner.target_entropy_controller = pufferl_sac.AdaptiveTargetEntropyController(
        0.1,
        ema_decay=0.0,
        mean_threshold=10.0,
        std_threshold=10.0,
        decay_factor=0.5,
        required_count=1,
    )
    actor_before = _parameters(learner.policy)
    critic_before = _parameters(learner.critic)
    target_before = _parameters(learner.target_critic)
    alpha_before = learner.log_alpha.detach().clone()
    controller_before = learner.target_entropy_controller.state_dict()
    optimizer_states_before = [
        copy.deepcopy(optimizer.state_dict())
        for optimizer in (learner.optimizer, learner.critic_optimizer, learner.alpha_optimizer)
    ]

    updates = learner._train_sac_transition(
        defaultdict(float), lambda *_args, **_kwargs: None, 0,
    )

    assert updates == learner.learner_updates == 0
    assert learner.sampled_transitions == 4
    for before, after in zip(actor_before, _parameters(learner.policy)):
        torch.testing.assert_close(before, after)
    for before, after in zip(critic_before, _parameters(learner.critic)):
        torch.testing.assert_close(before, after)
    for before, after in zip(target_before, _parameters(learner.target_critic)):
        torch.testing.assert_close(before, after)
    torch.testing.assert_close(alpha_before, learner.log_alpha.detach())
    assert learner.target_entropy_controller.state_dict() == controller_before
    for before, optimizer in zip(
        optimizer_states_before,
        (learner.optimizer, learner.critic_optimizer, learner.alpha_optimizer),
    ):
        assert optimizer.state_dict() == before
    metrics = learner.experience_filter.metrics()
    assert metrics["filter/candidate_count"] == 4
    assert metrics["filter/retained_count"] == 0
    assert metrics["filter/empty_batches"] == 1
    for counter in ("critic_steps", "actor_rows", "actor_steps", "alpha_steps", "target_steps"):
        assert learner.experience_filter.state_dict()["cycle"][counter] == 0


def test_actor_critic_update_policy_false_keeps_actor_alpha_and_entropy_frozen():
    torch.manual_seed(43)
    learner = _make_training_learner(
        _make_tensor_replay(), copy.deepcopy(_TinyActor().state_dict()),
        copy.deepcopy(_TinyCritic().state_dict()), return_estimator="n_step",
        experience_filter=True, threshold_scale=0.5,
    )
    actor_before = _parameters(learner.policy)
    critic_before = _parameters(learner.critic)
    target_before = _parameters(learner.target_critic)
    alpha_before = learner.log_alpha.detach().clone()
    controller_before = learner.target_entropy_controller.state_dict()

    updates = learner._train_sac_transition(
        defaultdict(float), lambda *_args, **_kwargs: None, 0, update_policy=False,
    )

    assert updates == learner.learner_updates == 1
    assert learner.experience_filter.metrics()["filter/retained_count"] > 0
    for before, after in zip(actor_before, _parameters(learner.policy)):
        torch.testing.assert_close(before, after)
    assert any(
        not torch.equal(before, after)
        for before, after in zip(critic_before, _parameters(learner.critic))
    )
    assert any(
        not torch.equal(before, after)
        for before, after in zip(target_before, _parameters(learner.target_critic))
    )
    torch.testing.assert_close(alpha_before, learner.log_alpha.detach())
    assert learner.target_entropy_controller.state_dict() == controller_before
    metrics = learner.experience_filter.metrics()
    assert metrics["filter/update_steps"] == 1
    assert metrics["filter/actor_rows"] == 0
    assert learner.experience_filter.state_dict()["cycle"]["actor_steps"] == 0
    assert learner.experience_filter.state_dict()["cycle"]["alpha_steps"] == 0
    assert learner.experience_filter.state_dict()["cycle"]["target_steps"] == 1


class _TinyVecEnv:
    single_observation_space = SimpleNamespace(shape=(3,))
    single_action_space = SimpleNamespace(n=2)

    def __init__(self):
        self.num_agents = 4
        self.agents_per_batch = 4

    def async_reset(self, seed):
        self.reset_seed = seed


class _NoopUtilization:
    def stop(self):
        pass

    def join(self):
        pass


class _NoopLogger:
    run_id = "filter-test"


def _make_checkpoint_learner(monkeypatch, data_dir, *, threshold_scale=0.2,
                             apply_to="actor_critic"):
    monkeypatch.setattr(pufferl, "Utilization", _NoopUtilization)
    monkeypatch.setattr(pufferl.PuffeRL, "print_dashboard", lambda self, clear=False: None)
    config = {
        "data_dir": str(data_dir),
        "action_count": 2,
        "run_name": "filter-checkpoint-test",
        "total_timesteps": 20,
        "batch_size": 4,
        "minibatch_size": 2,
        "max_minibatch_size": 2,
        "checkpoint_interval": None,
        "experience_filter": {
            "threshold_scale": threshold_scale,
            "apply_to": apply_to,
        },
    }
    return pufferl_sac.PuffeSAC(
        config,
        _TinyVecEnv(),
        _TinyActor(),
        _TinyCritic(),
        TensorReplay(4, (3,)),
        logger=_NoopLogger(),
        reset_env=False,
    )


def test_production_checkpoint_round_trip_and_filter_state_validation(monkeypatch, tmp_path):
    source = _make_checkpoint_learner(monkeypatch, tmp_path / "source")
    source.experience_filter.select(torch.tensor([1.0, 4.0, 2.0, 3.0]))
    source.experience_filter.mark_critic_step()
    expected_filter_state = source.experience_filter.state_dict()
    source.save_checkpoint()
    checkpoint_path = tmp_path / "source" / "trainer_state.pt"
    saved_state = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    assert saved_state["experience_filter_state_dict"] == expected_filter_state

    restored = _make_checkpoint_learner(monkeypatch, tmp_path / "restored")
    restored.load_training_state(checkpoint_path)
    assert restored.experience_filter.state_dict() == expected_filter_state

    missing_state_path = tmp_path / "missing_filter_state.pt"
    missing_state = dict(saved_state)
    missing_state.pop("experience_filter_state_dict")
    torch.save(missing_state, missing_state_path)
    missing_state_target = _make_checkpoint_learner(monkeypatch, tmp_path / "missing_target")
    with pytest.raises(ValueError, match="Enabled SAC experience-filter checkpoint is missing its filter state"):
        missing_state_target.load_training_state(missing_state_path)

    mismatched_target = _make_checkpoint_learner(
        monkeypatch, tmp_path / "mismatched_target", apply_to="critic",
    )
    with pytest.raises(ValueError, match="SAC experience-filter checkpoint settings do not match configuration"):
        mismatched_target.load_training_state(checkpoint_path)
