"""CPU tests for SAC TD-error experience filtering."""

import json
import pytest
import torch

from project.soft_actor_critic_experience_filtering.filtering import (
    ExperienceFilter,
    normalize_experience_filter_config,
)
from project.soft_actor_critic_tuning.scripts.run_trial import TrialMonitor


def _config(*, beta=0.25, scale=0.5, apply_to="actor_critic"):
    return normalize_experience_filter_config({
        "enabled": True,
        "score": "twin_max_abs_td_error",
        "apply_to": apply_to,
        "threshold_mode": "ewma_max",
        "ewma_beta": beta,
        "threshold_scale": scale,
    })


def test_config_normalization_keeps_filter_opt_in_and_validates_modes_and_numbers():
    assert normalize_experience_filter_config(None) is None
    assert normalize_experience_filter_config({"enabled": False}) is None
    normalized = normalize_experience_filter_config({"enabled": True})
    assert normalized == _config(beta=0.25, scale=0.01)
    assert normalize_experience_filter_config({"enabled": True, "apply_to": "critic"}) == _config(
        beta=0.25, scale=0.01, apply_to="critic",
    )

    for invalid in (
        {"enabled": True, "score": "mean_abs_td_error"},
        {"enabled": True, "apply_to": "actor"},
        {"enabled": True, "apply_to": "actor-critic"},
        {"enabled": True, "threshold_mode": "fixed"},
        {"enabled": True, "ewma_beta": 0},
        {"enabled": True, "ewma_beta": 1.01},
        {"enabled": True, "ewma_beta": True},
        {"enabled": True, "threshold_scale": -0.1},
        {"enabled": True, "threshold_scale": float("nan")},
        {"enabled": True, "threshold_scale": False},
    ):
        with pytest.raises((TypeError, ValueError)):
            normalize_experience_filter_config(invalid)


def test_ema_uses_new_max_weight_and_equality_is_retained():
    selector = ExperienceFilter(_config(beta=0.25, scale=0.5), device="cpu")

    first_mask, first_metrics = selector.select(torch.tensor([4.0, 8.0, 2.0]))
    assert first_metrics["current_max"] == pytest.approx(8.0)
    assert first_metrics["ema_max"] == pytest.approx(8.0)
    assert first_metrics["threshold"] == pytest.approx(4.0)
    assert first_mask.tolist() == [True, True, False]

    second_mask, second_metrics = selector.select(torch.tensor([2.0, 3.5, 4.0]))
    assert second_metrics["current_max"] == pytest.approx(4.0)
    assert second_metrics["ema_max"] == pytest.approx(7.0)
    assert second_metrics["threshold"] == pytest.approx(3.5)
    assert second_mask.tolist() == [False, True, True]


def test_scale_zero_and_all_zero_scores_keep_every_row():
    selector = ExperienceFilter(_config(scale=0), device="cpu")
    mask, metrics = selector.select(torch.tensor([0.0, 0.0, 0.0]))
    assert mask.tolist() == [True, True, True]
    assert metrics["retained_count"] == 3
    assert metrics["filtered_count"] == 0


@pytest.mark.parametrize("scores", [
    torch.tensor([float("nan")]),
    torch.tensor([float("inf")]),
])
def test_nonfinite_scores_fail_fast(scores):
    selector = ExperienceFilter(_config(), device="cpu")
    with pytest.raises(FloatingPointError, match="Nonfinite"):
        selector.select(scores)


def test_checkpoint_state_round_trip_and_filter_config_mismatch_rejection():
    original = ExperienceFilter(_config(beta=0.25, scale=0.5), device="cpu")
    original.select(torch.tensor([2.0, 10.0, 4.0]))
    original.select(torch.tensor([1.0, 5.0, 3.0]))
    original.mark_critic_step()
    saved = original.state_dict()

    restored = ExperienceFilter(_config(beta=0.25, scale=0.5), device="cpu")
    checked = restored.validate_state_dict(saved)
    restored.load_state_dict(checked)
    assert restored.state_dict() == saved
    metrics = restored.metrics()
    assert metrics["filter/update_steps"] == 1
    assert metrics["filter/cumulative_update_steps"] == 1
    for prefix in ("", "cumulative_"):
        for subsystem in ("actor", "critic", "alpha", "target"):
            assert f"filter/{prefix}{subsystem}_steps" not in metrics

    mismatched = ExperienceFilter(_config(beta=0.5, scale=0.5), device="cpu")
    with pytest.raises(ValueError, match="config|match|filter"):
        mismatched.validate_state_dict(saved)
    with pytest.raises(ValueError, match="state|checkpoint|filter"):
        restored.validate_state_dict(None)

    legacy_filter = ExperienceFilter(_config(apply_to="critic"), device="cpu")
    with pytest.raises(ValueError, match="config|match|filter"):
        legacy_filter.validate_state_dict(saved)

    legacy_filter.select(torch.tensor([4.0, 2.0, 1.0]))
    legacy_filter.mark_critic_step()
    legacy_state = legacy_filter.state_dict()
    legacy_restored = ExperienceFilter(_config(apply_to="critic"), device="cpu")
    legacy_restored.load_state_dict(legacy_restored.validate_state_dict(legacy_state))
    assert legacy_restored.state_dict() == legacy_state

    legacy_v1_state = dict(legacy_state)
    legacy_v1_state["version"] = 1
    legacy_v1_state.pop("actor_metric_history_complete")
    for counter_name in ("cumulative", "cycle"):
        legacy_v1_state[counter_name] = {
            name: legacy_state[counter_name][name]
            for name in ("candidate_count", "retained_count", "filtered_count", "empty_batches",
                         "local_empty_ranks", "critic_steps")
        }
    v1_restored = ExperienceFilter(_config(apply_to="critic"), device="cpu")
    v1_restored.load_state_dict(v1_restored.validate_state_dict(legacy_v1_state))
    assert v1_restored.state_dict()["actor_metric_history_complete"] is False
    with pytest.raises(ValueError, match="config|Legacy|apply_to='critic'"):
        ExperienceFilter(_config(), device="cpu").validate_state_dict(legacy_v1_state)


def test_trial_monitor_records_filter_metrics_and_ratio_of_cumulative_counts(tmp_path):
    metrics_path = tmp_path / "metrics.jsonl"
    monitor = TrialMonitor(metrics_path, tmp_path / "stop_reason.json", None)
    selector = ExperienceFilter(_config(scale=1.0), device="cpu")
    first_mask, _ = selector.select(torch.tensor([10.0, 2.0, 1.0, 0.0]))
    selector.begin_cycle()
    second_mask, _ = selector.select(torch.tensor([4.0, 2.0, 1.0, 0.0]))
    assert first_mask.sum().item() == 1
    assert second_mask.sum().item() == 0
    logs = {"epoch": 3, **selector.metrics()}

    assert monitor(logs) is False
    record = json.loads(metrics_path.read_text(encoding="utf-8").splitlines()[0])
    assert record["filter/retained_count"] == 0
    assert record["filter/candidate_count"] == 4
    assert record["filter/filtered_count"] == 4
    assert record["filter/retained_fraction"] == pytest.approx(0 / 4)
    assert record["filter/filtered_fraction"] == pytest.approx(4 / 4)
    assert record["filter/cumulative_retained_count"] == 1
    assert record["filter/cumulative_candidate_count"] == 8
    assert record["filter/cumulative_retained_fraction"] == pytest.approx(1 / 8)
    assert record["filter/cumulative_filtered_fraction"] == pytest.approx(7 / 8)
