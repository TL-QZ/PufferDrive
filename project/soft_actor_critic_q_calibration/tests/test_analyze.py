"""Synthetic checks for the Q-calibration analysis outputs."""

import csv
import json

import numpy as np
import pytest

from project.soft_actor_critic_q_calibration.analyze import (
    DISTANCE_BIN_LABELS,
    INVALID_DISTANCE_LABEL,
    _distance_bins,
    _plot_mean_interval,
    _summarize_values,
    analyze_runs,
)
import matplotlib.pyplot as plt


def _write_synthetic_run(run_directory):
    run_directory.mkdir(parents=True)
    row_count = 5
    arrays = {
        "q1": np.asarray([10, 18, 32, 2, 3], dtype=np.float32),
        "q2": np.asarray([12, 22, 28, 4, 5], dtype=np.float32),
        "target_q1": np.asarray([9, 17, 30, 2, 2], dtype=np.float32),
        "target_q2": np.asarray([11, 20, 29, 3, 4], dtype=np.float32),
        "log_prob": np.full(row_count, -1.0, dtype=np.float32),
        "entropy": np.full(row_count, 0.5, dtype=np.float32),
        "reward": np.asarray([1, 2, 3, -1, 0], dtype=np.float32),
        "graph_distance_m": np.asarray([0.0, 5.0, 40.0, np.nan, 100.0], dtype=np.float32),
        "graph_status": np.asarray([5, 5, 5, 4, 4], dtype=np.float32),
        "euclidean_distance_m": np.asarray([4.0, 6.0, 12.0, np.inf, 100.0], dtype=np.float32),
        "speed_mps": np.asarray([1, 6, 11, 3, 4], dtype=np.float32),
        "goal_index": np.asarray([0, 1, 2, 0, 1], dtype=np.float32),
        "goal_count": np.full(row_count, 3, dtype=np.float32),
        "goal_mismatch": np.asarray([False, True, False, False, False]),
        "scene_id": np.asarray([1, 1, 2, 3, 4], dtype=np.int32),
        "episode_id": np.ones(row_count, dtype=np.int32),
        "step": np.asarray([0, 8, 0, 0, 0], dtype=np.int32),
        "agent_slot": np.asarray([0, 1, 0, 0, 0], dtype=np.int32),
        "action": np.asarray([0, 1, 2, 0, 1], dtype=np.int32),
        "terminal": np.asarray([False, True, True, False, False]),
        "truncated": np.asarray([False, False, False, True, False]),
        "goal_reward_coeff": np.ones(row_count, dtype=np.float32),
        "g_reward": np.asarray([8, 16, 30, -2, 0], dtype=np.float32),
        "g_soft": np.asarray([10, 20, 35, -1, 1], dtype=np.float32),
        "censored": np.asarray([False, False, False, True, True]),
        "horizon": np.asarray([3, 2, 1, 5, 4], dtype=np.float32),
        "tail_discount": np.asarray([0, 0, 0, 0.99, 0.98], dtype=np.float32),
        "g_soft_bootstrap": np.asarray([10, 20, 35, 3, 4], dtype=np.float32),
        "bootstrap_tail": np.asarray([0, 0, 0, 4, 3], dtype=np.float32),
        "collision_coeff": np.asarray([2, 2, 4, 4, 4], dtype=np.float32),
        "goal_hit_next": np.asarray([0, 1, 0, np.nan, np.nan], dtype=np.float32),
    }
    np.savez_compressed(run_directory / "transitions.npz", **arrays)
    (run_directory / "run.json").write_text(
        json.dumps(
            {
                "run_id": run_directory.name,
                "status": "completed",
                "checkpoint_sha256": "checkpoint-abc",
                "source_config_sha256": "config-abc",
                "gamma": 0.999,
                "alpha": 0.001,
                "action_selection": "sample",
                "env_config": {"num_agents": row_count},
            }
        ),
        encoding="utf-8",
    )


def test_distance_bins_keep_invalid_values_separate():
    distances = np.asarray([-1.0, 0.0, 4.999, 5.0, 159.9, 160.0, np.inf, np.nan])
    bins, valid = _distance_bins(distances)

    assert valid.tolist() == [False, True, True, True, True, True, False, False]
    assert bins.tolist() == [-1, 0, 0, 1, 5, 6, -1, -1]
    assert DISTANCE_BIN_LABELS[bins[5]] == "[160,infinity)"
    assert INVALID_DISTANCE_LABEL == "invalid_or_unreachable"


def test_cluster_summary_preserves_scene_balanced_sensitivity():
    values = np.asarray([0.0, 0.0, 10.0])
    clusters = np.asarray([0, 0, 1])

    summary = _summarize_values(
        values,
        clusters,
        bootstrap_samples=100,
        rng=np.random.default_rng(7),
    )

    assert summary["mean"] == 10 / 3
    assert summary["scene_balanced_mean"] == 5.0
    assert summary["row_count"] == 3
    assert summary["cluster_count"] == 2
    assert summary["ci_low"] <= summary["mean"] <= summary["ci_high"]


def test_cluster_interval_draws_percentiles_even_if_they_exclude_the_mean():
    figure, axis = plt.subplots()
    try:
        assert _plot_mean_interval(
            axis,
            np.asarray([0.0]),
            np.asarray([5.0]),
            np.asarray([10.0]),
            np.asarray([20.0]),
            label="test",
            color="black",
        )
        figure.canvas.draw()
        assert axis.lines[0].get_ydata().tolist() == [5.0]
        interval = axis.collections[0].get_segments()[0]
        assert interval[:, 1].tolist() == [10.0, 20.0]
    finally:
        plt.close(figure)


def test_multiple_runs_require_matching_checkpoint_and_task_contract(tmp_path):
    first_run = tmp_path / "run_a"
    second_run = tmp_path / "run_b"
    _write_synthetic_run(first_run)
    _write_synthetic_run(second_run)
    second_metadata = json.loads((second_run / "run.json").read_text(encoding="utf-8"))
    second_metadata["source_config_sha256"] = "different-config"
    (second_run / "run.json").write_text(json.dumps(second_metadata), encoding="utf-8")

    with pytest.raises(ValueError, match="different source_config_sha256"):
        analyze_runs([first_run, second_run], tmp_path / "pooled", bootstrap_samples=0)


def test_analysis_separates_complete_censored_and_bootstrap_targets(tmp_path):
    run_directory = tmp_path / "run_a"
    output_directory = tmp_path / "analysis"
    _write_synthetic_run(run_directory)

    evidence = analyze_runs([run_directory], output_directory, bootstrap_samples=20)

    assert evidence["overall"]["complete_true_terminal_target_rows"] == 3
    assert evidence["overall"]["censored_rows"] == 2
    assert evidence["overall"]["censored_rows_with_bootstrap_estimate"] == 2
    assert evidence["overall"]["complete_cluster_count"] == 2
    assert evidence["overall"]["behavior_fields_available"] == ["goal_hit_next"]
    assert evidence["overall"]["graph_distance_status_counts"]["reachable"] == 3
    assert evidence["overall"]["graph_distance_status_counts"]["unreachable_or_invalid_graph_entry"] == 2

    with (output_directory / "distance_summary.csv").open(newline="", encoding="utf-8") as handle:
        distance_rows = list(csv.DictReader(handle))
    graph_rows = {
        (row["distance_bin"], row["goal_mismatch_subset"]): row
        for row in distance_rows
        if row["distance_source"] == "graph"
    }
    assert int(graph_rows[("[0,5)", "all_rows")]["row_count"]) == 1
    assert int(graph_rows[("[0,5)", "exclude_goal_mismatch")]["row_count"]) == 1
    assert int(graph_rows[("[5,10)", "all_rows")]["row_count"]) == 1
    assert int(graph_rows[("[5,10)", "exclude_goal_mismatch")]["row_count"]) == 0
    assert float(graph_rows[("[0,5)", "all_rows")]["mean_g_soft"]) == 10.0

    with (output_directory / "coverage.csv").open(newline="", encoding="utf-8") as handle:
        coverage_rows = list(csv.DictReader(handle))
    invalid_graph = next(
        row for row in coverage_rows
        if row["distance_source"] == "graph" and row["distance_bin"] == INVALID_DISTANCE_LABEL
    )
    assert int(invalid_graph["row_count"]) == 2
    assert int(invalid_graph["censored_rows"]) == 2
    assert int(invalid_graph["bootstrap_tail_rows"]) == 2

    expected_outputs = {
        "analysis.json",
        "distance_summary.csv",
        "residual_summary.csv",
        "coverage.csv",
        "reliability.csv",
        "confounder_summary.csv",
        "censoring_summary.csv",
        "horizon_sensitivity.csv",
        "distance_calibration.png",
        "residuals_by_distance.png",
        "value_reliability.png",
        "behavior_confounders.png",
        "tail_sensitivity.png",
    }
    assert expected_outputs <= {path.name for path in output_directory.iterdir()}

    saved_evidence = json.loads((output_directory / "analysis.json").read_text(encoding="utf-8"))
    assert saved_evidence["overall"]["interpretation"]["flat_q"].startswith("Flat Q")
