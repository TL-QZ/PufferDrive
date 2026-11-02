import copy
from collections import defaultdict
from pathlib import Path

import pytest
import torch
import torch.distributed as dist

from pufferlib import pufferl_sac
from pufferlib.sac_diagnostics import SACCollapseDiagnostics
from test_sac_learner import (
    _FakeVecEnv,
    _TinyActor,
    _TinyCritic,
    _make_checkpoint_learner,
    _make_replay,
    _make_training_learner,
)


def _assert_nested_equal(left, right):
    if torch.is_tensor(left):
        torch.testing.assert_close(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_nested_equal(left[key], right[key])
    elif isinstance(left, (tuple, list)):
        assert len(left) == len(right)
        for left_value, right_value in zip(left, right):
            _assert_nested_equal(left_value, right_value)
    else:
        assert left == right


def _parity_learner(actor_state, critic_state, diagnostics_enabled):
    learner = _make_training_learner(
        _make_replay(), actor_state, critic_state, max_minibatch_size=3, max_grad_norm=0.01,
    )
    learner.config.update(
        update_epochs=1,
        log_collapse_diagnostics=diagnostics_enabled,
    )
    learner.minibatch_size = 7
    forward_counts = {"actor": 0, "critic": 0, "target_critic": 0}
    for name, module in (
        ("actor", learner.policy),
        ("critic", learner.critic),
        ("target_critic", learner.target_critic),
    ):
        module.register_forward_hook(
            lambda _module, _inputs, _output, name=name: forward_counts.__setitem__(
                name, forward_counts[name] + 1,
            )
        )
    return learner, forward_counts


def test_diagnostics_do_not_change_training_parameters_optimizers_rng_or_forwards():
    torch.manual_seed(34)
    actor_state = copy.deepcopy(_TinyActor().state_dict())
    critic_state = copy.deepcopy(_TinyCritic().state_dict())
    disabled, disabled_forwards = _parity_learner(actor_state, critic_state, False)
    enabled, enabled_forwards = _parity_learner(actor_state, critic_state, True)

    torch.manual_seed(917)
    disabled._train_sac_transition(defaultdict(float), lambda *_args, **_kwargs: None, 0)
    disabled_rng = torch.get_rng_state()
    torch.manual_seed(917)
    enabled._train_sac_transition(defaultdict(float), lambda *_args, **_kwargs: None, 0)
    enabled_rng = torch.get_rng_state()

    for disabled_module, enabled_module in (
        (disabled.policy, enabled.policy),
        (disabled.critic, enabled.critic),
        (disabled.target_critic, enabled.target_critic),
    ):
        _assert_nested_equal(disabled_module.state_dict(), enabled_module.state_dict())
    torch.testing.assert_close(disabled.log_alpha, enabled.log_alpha)
    _assert_nested_equal(disabled.optimizer.state_dict(), enabled.optimizer.state_dict())
    _assert_nested_equal(disabled.critic_optimizer.state_dict(), enabled.critic_optimizer.state_dict())
    _assert_nested_equal(disabled.alpha_optimizer.state_dict(), enabled.alpha_optimizer.state_dict())
    torch.testing.assert_close(disabled_rng, enabled_rng)
    assert disabled_forwards == enabled_forwards


def test_hand_computed_row_metrics_entropy_min_and_single_action_policy():
    tracker = SACCollapseDiagnostics("cpu")
    logits = torch.tensor([[10.0, -10.0], [0.0, 0.0]])
    probabilities = torch.softmax(logits, dim=-1)
    minimum_q = torch.tensor([[4.0, 3.0], [5.0, 1.0]])
    tracker.add_policy(logits, probabilities, minimum_q, torch.tensor(0.5))
    tracker.add_td_errors(
        torch.tensor([2.0, -1.0]),
        torch.tensor([-3.0, 0.0]),
        torch.tensor([1.0, 1.0]),
    )
    tracker.add_grad_norm("actor", torch.tensor(5.0), max_grad_norm=2.0)
    tracker.add_grad_norm("actor", torch.tensor(1.0), max_grad_norm=2.0)
    tracker.add_entropy_update_mean(torch.tensor(1.25))
    tracker.add_entropy_update_mean(torch.tensor(0.25))
    metrics = tracker.reduce_and_reset()

    assert metrics["diagnostics/policy/max_action_probability_mean"] == pytest.approx(0.75)
    assert metrics["diagnostics/policy/near_deterministic_fraction"] == pytest.approx(0.5)
    assert metrics["diagnostics/policy/logit_range_mean"] == pytest.approx(10.0)
    assert metrics["diagnostics/policy/logit_range_max"] == pytest.approx(20.0)
    assert metrics["diagnostics/policy/q_action_gap_over_alpha_mean"] == pytest.approx(5.0)
    assert metrics["diagnostics/policy/q_action_gap_over_alpha_max"] == pytest.approx(8.0)
    assert metrics["diagnostics/critic/abs_td_error_mean"] == pytest.approx(2.0)
    assert metrics["diagnostics/critic/abs_td_error_max"] == pytest.approx(4.0)
    assert metrics["diagnostics/actor/grad_norm_mean"] == pytest.approx(3.0)
    assert metrics["diagnostics/actor/grad_norm_max"] == pytest.approx(5.0)
    assert metrics["diagnostics/actor/grad_clip_fraction"] == pytest.approx(0.5)
    assert metrics["diagnostics/entropy_update_mean_min"] == pytest.approx(0.25)

    single_action = SACCollapseDiagnostics("cpu")
    single_action.add_policy(
        torch.tensor([[3.0]]), torch.tensor([[1.0]]), torch.tensor([[2.0]]), torch.tensor(0.2),
    )
    one_action_metrics = single_action.reduce_and_reset()
    assert one_action_metrics["diagnostics/policy/q_action_gap_over_alpha_mean"] == 0.0


def test_row_means_weight_unequal_microbatches_and_grad_metrics_need_clipping():
    tracker = SACCollapseDiagnostics("cpu")
    tracker.add_policy(
        torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
        torch.tensor([[0.0, 0.0], [0.0, 0.0]]),
        torch.tensor(1.0),
    )
    tracker.add_policy(
        torch.tensor([[10.0, -10.0]]),
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[2.0, 0.0]]),
        torch.tensor(1.0),
    )
    metrics = tracker.reduce_and_reset()
    assert metrics["diagnostics/policy/max_action_probability_mean"] == pytest.approx(2.0 / 3.0)
    assert metrics["diagnostics/policy/logit_range_mean"] == pytest.approx(20.0 / 3.0)
    assert metrics["diagnostics/policy/logit_range_max"] == pytest.approx(20.0)

    learner = _make_training_learner(
        _make_replay(), _TinyActor().state_dict(), _TinyCritic().state_dict(),
        max_minibatch_size=3, max_grad_norm=None,
    )
    learner.config["update_epochs"] = 1
    learner.config["log_collapse_diagnostics"] = True
    learner.minibatch_size = 7
    learner._train_sac_transition(defaultdict(float), lambda *_args, **_kwargs: None, 0)
    disabled_clip_metrics = learner._collapse_diagnostics_tracker.reduce_and_reset()
    assert not any("grad_norm" in key or "grad_clip" in key for key in disabled_clip_metrics)


def test_training_records_clip_grad_norm_return_before_clipping(monkeypatch):
    learner = _make_training_learner(
        _make_replay(), _TinyActor().state_dict(), _TinyCritic().state_dict(),
        max_minibatch_size=3, max_grad_norm=1e-8,
    )
    learner.config["update_epochs"] = 1
    learner.config["log_collapse_diagnostics"] = True
    learner.minibatch_size = 7
    actual_clip = torch.nn.utils.clip_grad_norm_
    # The critic clip happens before the actor clip for each logical update.
    clip_order = []

    def record_ordered_norm(parameters, max_norm, *args, **kwargs):
        result = actual_clip(parameters, max_norm, *args, **kwargs)
        clip_order.append(result.detach().clone())
        return result

    monkeypatch.setattr(torch.nn.utils, "clip_grad_norm_", record_ordered_norm)
    learner._train_sac_transition(defaultdict(float), lambda *_args, **_kwargs: None, 0)
    assert len(clip_order) == 2
    metrics = learner._collapse_diagnostics_tracker.reduce_and_reset()
    assert metrics["diagnostics/critic/grad_norm_mean"] == pytest.approx(float(clip_order[0]), rel=1e-6)
    assert metrics["diagnostics/critic/grad_norm_max"] == pytest.approx(float(clip_order[0]), rel=1e-6)
    assert metrics["diagnostics/actor/grad_norm_mean"] == pytest.approx(float(clip_order[1]), rel=1e-6)
    assert metrics["diagnostics/actor/grad_norm_max"] == pytest.approx(float(clip_order[1]), rel=1e-6)
    assert metrics["diagnostics/critic/grad_clip_fraction"] == 1.0
    assert metrics["diagnostics/actor/grad_clip_fraction"] == 1.0


def test_diagnostics_flag_validates_and_is_operational_on_checkpoint_resume(monkeypatch, tmp_path):
    with pytest.raises(ValueError, match="log_collapse_diagnostics must be a boolean"):
        _make_checkpoint_learner(monkeypatch, tmp_path, log_collapse_diagnostics=1)

    enabled = _make_checkpoint_learner(
        monkeypatch, tmp_path / "enabled", log_collapse_diagnostics=True,
    )
    enabled._collapse_diagnostics_tracker = SACCollapseDiagnostics("cpu")
    enabled._collapse_diagnostics_tracker.add_entropy_update_mean(torch.tensor(0.1))
    enabled.save_checkpoint()
    checkpoint_path = Path(enabled.config["data_dir"]) / "trainer_state.pt"

    disabled = _make_checkpoint_learner(
        monkeypatch, tmp_path / "disabled", log_collapse_diagnostics=False,
    )
    disabled._collapse_diagnostics_tracker = SACCollapseDiagnostics("cpu")
    disabled._collapse_diagnostics_tracker.add_entropy_update_mean(torch.tensor(0.9))
    disabled.load_training_state(checkpoint_path)
    assert disabled.config["log_collapse_diagnostics"] is False
    assert disabled._collapse_diagnostics_tracker.reduce_and_reset() == {}

    default_off = _make_checkpoint_learner(
        monkeypatch, tmp_path / "default-off", log_collapse_diagnostics=False,
    )
    default_off.save_checkpoint()
    legacy_path = Path(default_off.config["data_dir"]) / "trainer_state.pt"
    legacy_state = torch.load(legacy_path, map_location="cpu", weights_only=False)
    legacy_state["config"].pop("log_collapse_diagnostics")
    legacy_path = tmp_path / "legacy-default-off.pt"
    torch.save(legacy_state, legacy_path)
    enabled_after_resume = _make_checkpoint_learner(
        monkeypatch, tmp_path / "enabled-after-resume", log_collapse_diagnostics=True,
    )
    enabled_after_resume.load_training_state(legacy_path)
    assert enabled_after_resume.config["log_collapse_diagnostics"] is True


def test_mean_and_log_emits_then_resets_only_enabled_diagnostics(monkeypatch, tmp_path):
    class _RecordingLogger:
        def __init__(self):
            self.records = []

        def log(self, metrics, step):
            self.records.append((metrics, step))

    learner = _make_checkpoint_learner(
        monkeypatch, tmp_path, log_collapse_diagnostics=True,
    )
    learner.logger = _RecordingLogger()
    learner._collapse_diagnostics_tracker = SACCollapseDiagnostics("cpu")
    learner._collapse_diagnostics_tracker.add_entropy_update_mean(torch.tensor(0.3))
    first_logs = learner.mean_and_log()
    assert first_logs["diagnostics/entropy_update_mean_min"] == pytest.approx(0.3)
    assert learner.logger.records[-1][0] is first_logs

    second_logs = learner.mean_and_log()
    assert not any(key.startswith("diagnostics/") for key in second_logs)

    disabled = _make_checkpoint_learner(
        monkeypatch, tmp_path / "mean-log-disabled", log_collapse_diagnostics=False,
    )
    disabled.logger = _RecordingLogger()
    disabled_logs = disabled.mean_and_log()
    assert not any(key.startswith("diagnostics/") for key in disabled_logs)


def test_diagnostics_work_with_compiled_forward_wrappers():
    learner = _make_training_learner(
        _make_replay(), _TinyActor().state_dict(), _TinyCritic().state_dict(),
        max_minibatch_size=3,
    )
    learner.config["update_epochs"] = 1
    learner.config["log_collapse_diagnostics"] = True
    learner.minibatch_size = 7
    learner.policy_train = torch.compile(learner.policy, backend="eager")
    learner.critic_train = torch.compile(learner.critic, backend="eager")
    updates = learner._train_sac_transition(defaultdict(float), lambda *_args, **_kwargs: None, 0)
    metrics = learner._collapse_diagnostics_tracker.reduce_and_reset()
    assert updates > 0
    assert "diagnostics/policy/logit_range_mean" in metrics
    assert "diagnostics/critic/abs_td_error_mean" in metrics


def test_mocked_distributed_reduction_weights_rows_and_reduces_extrema(monkeypatch):
    local = SACCollapseDiagnostics("cpu")
    remote = SACCollapseDiagnostics("cpu")
    local.add_td_errors(torch.tensor([1.0, 3.0]), torch.tensor([2.0, 4.0]), torch.zeros(2))
    local.add_grad_norm("actor", torch.tensor(2.0), max_grad_norm=3.0)
    local.add_grad_norm("actor", torch.tensor(4.0), max_grad_norm=3.0)
    local.add_entropy_update_mean(torch.tensor(0.4))
    local.add_entropy_update_mean(torch.tensor(0.2))
    remote.add_td_errors(torch.tensor([10.0]), torch.tensor([0.0]), torch.zeros(1))
    remote.add_grad_norm("actor", torch.tensor(6.0), max_grad_norm=3.0)
    remote.add_entropy_update_mean(torch.tensor(0.1))
    remote_packed = torch.stack((remote.sums, remote.counts, remote.maxima, remote.minima))
    observed_ops = []

    def reduce_with_remote_rank(local_tensor, op):
        observed_ops.append(op)
        if op == dist.ReduceOp.SUM:
            local_tensor.add_(remote_packed[:2])
        elif op == dist.ReduceOp.MAX:
            local_tensor.copy_(torch.maximum(local_tensor, remote_packed[2]))
        elif op == dist.ReduceOp.MIN:
            local_tensor.copy_(torch.minimum(local_tensor, remote_packed[3]))
        else:
            raise AssertionError(f"Unexpected reduction operator: {op}")

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "all_reduce", reduce_with_remote_rank)
    metrics = local.reduce_and_reset(distributed=True)

    assert observed_ops == [dist.ReduceOp.SUM, dist.ReduceOp.MAX, dist.ReduceOp.MIN]
    assert metrics["diagnostics/critic/abs_td_error_mean"] == pytest.approx(20.0 / 6.0)
    assert metrics["diagnostics/critic/abs_td_error_max"] == pytest.approx(10.0)
    assert metrics["diagnostics/actor/grad_norm_mean"] == pytest.approx(4.0)
    assert metrics["diagnostics/actor/grad_norm_max"] == pytest.approx(6.0)
    assert metrics["diagnostics/actor/grad_clip_fraction"] == pytest.approx(2.0 / 3.0)
    assert metrics["diagnostics/entropy_update_mean_min"] == pytest.approx(0.1)
