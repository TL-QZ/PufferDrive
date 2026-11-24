import math

from project.soft_actor_critic_q_calibration.development.analyze_round_four import (
    action_mean_rows,
    score_actions,
)


def test_rank_coefficients_keep_raw_direction_while_pair_accuracy_reverses_infractions():
    scores = score_actions([1, 2, 3], [3, 2, 1], lower_is_better=True)

    assert scores["pearson_r"] == -1.0
    assert scores["spearman_rho"] == -1.0
    assert scores["kendall_tau_b"] == -1.0
    assert scores["pairwise_ranking_accuracy"] == 1.0
    assert scores["comparable_pair_count"] == 3


def test_rank_metrics_handle_target_and_outcome_ties():
    scores = score_actions([1, 1, 3, 4], [2, 2, 2, 1])

    assert scores["action_count"] == 4
    assert scores["possible_pair_count"] == 6
    assert scores["tied_outcome_pair_count"] == 3
    assert scores["comparable_pair_count"] == 3
    assert math.isfinite(scores["kendall_tau_b"])
    assert math.isfinite(scores["spearman_rho"])


def test_action_means_pair_target_and_outcome_over_only_repeats_reaching_horizon():
    branches = [
        {
            "action_idx": action,
            "repeat_idx": repeat,
            "actual_critic_target": float(2 * repeat),
            "metrics_mean": {"episode_return": float(10 * repeat)},
            "horizon_metrics": ({"8": {"metrics_mean": {"episode_return": float(10 * repeat)}}}
                               if repeat < 2 else {}),
        }
        for action in range(2)
        for repeat in range(4)
    ]
    rows = action_mean_rows(
        branches,
        model_name="pql",
        horizon="8",
        metrics={"episode_return": "Episode return"},
        state_id="map/episode/frame/agent",
        assigned_bin={
            "bin_id": "distance:00",
            "axis": "nearest_visible_goal_distance_m",
            "lower_m": 0.0,
            "upper_m": 2.0,
        },
    )

    assert len(rows) == 2
    for row in rows:
        assert row["repeat_count"] == 2
        assert row["repeat_indices"] == "0,1"
        assert row["target_mean"] == 1.0
        assert row["outcome_mean"] == 5.0


def test_no_rank_score_when_outcome_is_constant():
    scores = score_actions([1, 2, 3], [5, 5, 5])

    assert math.isnan(scores["pearson_r"])
    assert math.isnan(scores["spearman_rho"])
    assert math.isnan(scores["kendall_tau_b"])
    assert math.isnan(scores["pairwise_ranking_accuracy"])
    assert scores["tied_outcome_pair_count"] == 3
