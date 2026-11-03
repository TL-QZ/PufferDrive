"""Summarize frozen-policy Q calibration rollouts without treating tails as truth.

Example::

    python project/soft_actor_critic_q_calibration/analyze.py \
        --runs 'project/soft_actor_critic_q_calibration/runs/*' \
        --output-dir project/soft_actor_critic_q_calibration/analysis

Each run directory must contain ``transitions.npz`` and ``run.json``. NPZ
columns are flattened transition rows; returns were computed from the complete
per-agent rollout before any transition subsampling.
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import os
from pathlib import Path
from typing import Any, Iterable

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(__file__).resolve().parent / ".matplotlib-cache"),
)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DISTANCE_UPPER_EDGES_M = np.asarray([5, 10, 20, 40, 80, 160], dtype=np.float64)
DISTANCE_BIN_LABELS = (
    "[0,5)",
    "[5,10)",
    "[10,20)",
    "[20,40)",
    "[40,80)",
    "[80,160)",
    "[160,infinity)",
)
HORIZON_UPPER_EDGES = np.asarray([32, 128, 512, 1024, 2560], dtype=np.int64)
HORIZON_BIN_LABELS = (
    "[1,32]",
    "[33,128]",
    "[129,512]",
    "[513,1024]",
    "[1025,2560]",
    "[2561,infinity)",
)
INVALID_DISTANCE_LABEL = "invalid_or_unreachable"
BOOTSTRAP_SEED = 20261102
DEFAULT_BOOTSTRAP_SAMPLES = 250
MIN_INTERPRETABLE_CLUSTERS = 30

REQUIRED_COLUMNS = {
    "q1",
    "q2",
    "target_q1",
    "target_q2",
    "log_prob",
    "entropy",
    "reward",
    "graph_distance_m",
    "euclidean_distance_m",
    "speed_mps",
    "goal_index",
    "goal_count",
    "goal_mismatch",
    "scene_id",
    "episode_id",
    "step",
    "agent_slot",
    "action",
    "terminal",
    "truncated",
    "goal_reward_coeff",
    "g_reward",
    "g_soft",
    "censored",
    "horizon",
    "tail_discount",
    "g_soft_bootstrap",
}
OPTIONAL_COLUMNS = {
    "reward_goal_component",
    "collision_coeff",
    "offroad_coeff",
    "comfort_coeff",
    "goal_radius_m",
    "goal_speed_mps",
    "goal_hit_next",
    "graph_status",
    "waypoint_success",
    "collision",
    "offroad",
    "truncation_endpoint",
    "bootstrap_tail",
}

PREDICTION_FIELDS = (
    "q1",
    "q2",
    "q_min",
    "q_mean",
    "target_q1",
    "target_q2",
    "target_q_min",
    "target_q_mean",
)
RETURN_FIELDS = ("g_soft", "g_reward")
RESIDUAL_FIELDS = (
    "q1_residual",
    "q2_residual",
    "q_min_residual",
    "q_mean_residual",
    "target_q1_residual",
    "target_q2_residual",
    "target_q_min_residual",
    "target_q_mean_residual",
)
METRIC_FIELDS = PREDICTION_FIELDS + RETURN_FIELDS + RESIDUAL_FIELDS + (
    "q1_abs_error",
    "q2_abs_error",
    "q_min_abs_error",
    "q_mean_abs_error",
    "q1_squared_error",
    "q2_squared_error",
    "q_min_squared_error",
    "q_mean_squared_error",
    "twin_disagreement",
)

COEFFICIENT_FIELDS = (
    "goal_reward_coeff",
    "collision_coeff",
    "offroad_coeff",
    "comfort_coeff",
    "goal_radius_m",
    "goal_speed_mps",
)


def _as_bool(values: np.ndarray, name: str) -> np.ndarray:
    array = np.asarray(values)
    if array.dtype == np.bool_:
        return array
    if not np.issubdtype(array.dtype, np.number):
        raise ValueError(f"{name} must contain boolean or numeric 0/1 values")
    if not np.all(np.isfinite(array)) or np.any((array != 0) & (array != 1)):
        raise ValueError(f"{name} must contain only boolean or numeric 0/1 values")
    return array.astype(np.bool_, copy=False)


def _resolve_run_directories(run_patterns: Iterable[str | Path]) -> list[Path]:
    directories: list[Path] = []
    for pattern in run_patterns:
        matches = glob.glob(str(pattern), recursive=True)
        if not matches and Path(pattern).exists():
            matches = [str(pattern)]
        for match in matches:
            path = Path(match).expanduser().resolve()
            if path.is_dir() and (path / "transitions.npz").is_file():
                directories.append(path)
    unique_paths = sorted(set(directories))
    if not unique_paths:
        raise FileNotFoundError("--runs matched no directories containing transitions.npz")
    return unique_paths


def _read_run(directory: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    metadata_path = directory / "run.json"
    transitions_path = directory / "transitions.npz"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing run metadata: {metadata_path}")
    if not transitions_path.is_file():
        raise FileNotFoundError(f"Missing transition rows: {transitions_path}")

    with metadata_path.open("r", encoding="utf-8") as handle:
        metadata = json.load(handle)
    if not isinstance(metadata, dict):
        raise ValueError(f"{metadata_path} must contain a JSON object")
    if metadata.get("status") != "completed":
        raise ValueError(
            f"{metadata_path} has status {metadata.get('status')!r}; only completed runs can be analyzed"
        )

    with np.load(transitions_path, allow_pickle=False) as archive:
        available = set(archive.files)
        missing = sorted(REQUIRED_COLUMNS - available)
        if missing:
            raise ValueError(f"{transitions_path} is missing columns: {missing}")
        columns: dict[str, np.ndarray] = {}
        for name in sorted(REQUIRED_COLUMNS | (OPTIONAL_COLUMNS & available)):
            array = np.asarray(archive[name]).reshape(-1)
            if array.dtype.hasobject:
                raise ValueError(f"{transitions_path}:{name} must not use object dtype")
            columns[name] = array

    row_counts = {len(values) for values in columns.values()}
    if len(row_counts) != 1:
        raise ValueError(f"{transitions_path} columns have inconsistent row counts")
    row_count = row_counts.pop()
    if row_count == 0:
        raise ValueError(f"{transitions_path} contains no transition rows")

    for name in REQUIRED_COLUMNS - {
        "scene_id",
        "episode_id",
        "goal_mismatch",
        "terminal",
        "truncated",
        "censored",
    }:
        if not np.issubdtype(columns[name].dtype, np.number):
            raise ValueError(f"{transitions_path}:{name} must be numeric")

    for name in ("goal_mismatch", "terminal", "truncated", "censored"):
        columns[name] = _as_bool(columns[name], name)
    if "truncation_endpoint" in columns:
        columns["truncation_endpoint"] = _as_bool(
            columns["truncation_endpoint"], "truncation_endpoint"
        )
    for name in ("waypoint_success", "collision", "offroad"):
        if name in columns:
            columns[name] = _as_bool(columns[name], name)
    for name in (
        "q1",
        "q2",
        "target_q1",
        "target_q2",
        "log_prob",
        "entropy",
        "reward",
        "graph_distance_m",
        "euclidean_distance_m",
        "speed_mps",
        "goal_reward_coeff",
        "g_reward",
        "g_soft",
        "horizon",
        "tail_discount",
        "g_soft_bootstrap",
        *OPTIONAL_COLUMNS,
    ):
        if name in columns and np.issubdtype(columns[name].dtype, np.number):
            columns[name] = columns[name].astype(np.float32, copy=False)
    return columns, metadata


def _load_runs(run_directories: list[Path]) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    column_lists: dict[str, list[np.ndarray]] = {}
    run_index_parts: list[np.ndarray] = []
    run_summaries: list[dict[str, Any]] = []
    all_columns: set[str] = set()

    loaded_runs = []
    for run_index, directory in enumerate(run_directories):
        columns, metadata = _read_run(directory)
        loaded_runs.append((run_index, directory, columns, metadata))
        all_columns.update(columns)

    for run_index, directory, columns, metadata in loaded_runs:
        row_count = len(columns["q1"])
        run_index_parts.append(np.full(row_count, run_index, dtype=np.int32))
        for name in all_columns:
            if name in columns:
                value = columns[name]
            elif name in OPTIONAL_COLUMNS:
                value = np.full(row_count, np.nan, dtype=np.float32)
            else:
                raise ValueError(f"Run {directory} is missing required column {name}")
            column_lists.setdefault(name, []).append(value)
        env_config = metadata.get("env_config", {})
        if not isinstance(env_config, dict):
            env_config = {}
        run_summaries.append(
            {
                "run_id": str(metadata.get("run_id", directory.name)),
                "run_directory": str(directory),
                "row_count": row_count,
                "metadata": {
                    key: metadata[key]
                    for key in (
                        "checkpoint_sha256",
                        "source_config_sha256",
                        "checkpoint_hash",
                        "config_hash",
                        "gamma",
                        "alpha",
                        "distance_definition",
                        "map_set",
                        "map_ids",
                        "env_seed",
                        "action_seed",
                        "action_selection",
                        "training_git_commit",
                    )
                    if key in metadata
                },
                "num_agents": env_config.get("num_agents", metadata.get("num_agents")),
            }
        )

    result = {
        name: np.concatenate(arrays)
        for name, arrays in column_lists.items()
    }
    result["run_index"] = np.concatenate(run_index_parts)
    return result, run_summaries


def _run_contract(metadata: dict[str, Any]) -> dict[str, Any]:
    env_config = metadata.get("env_config", {})
    if not isinstance(env_config, dict):
        env_config = {}
    return {
        "checkpoint_sha256": metadata.get(
            "checkpoint_sha256", metadata.get("checkpoint_hash")
        ),
        "source_config_sha256": metadata.get(
            "source_config_sha256", metadata.get("config_hash")
        ),
        "alpha": metadata.get("alpha"),
        "gamma": metadata.get("gamma"),
        "action_selection": metadata.get("action_selection"),
        "num_agents": env_config.get("num_agents", metadata.get("num_agents")),
    }


def _validate_run_compatibility(run_directories: list[Path]) -> list[dict[str, Any]]:
    contracts = []
    for directory in run_directories:
        with (directory / "run.json").open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
        contracts.append(_run_contract(metadata))
    if len(contracts) <= 1:
        return contracts

    for field in contracts[0]:
        values = [contract[field] for contract in contracts]
        if any(value is None for value in values):
            raise ValueError(
                f"Cannot pool multiple runs: run.json must provide {field} for every run"
            )
        first = values[0]
        if field in {"alpha", "gamma"}:
            compatible = all(
                np.isclose(float(value), float(first), rtol=1e-12, atol=0.0)
                for value in values[1:]
            )
        else:
            compatible = all(value == first for value in values[1:])
        if not compatible:
            raise ValueError(f"Cannot pool runs with different {field}: {values}")
    return contracts


def _cluster_indices(data: dict[str, np.ndarray]) -> np.ndarray:
    """Group same run/scene/episode rows for cluster-aware uncertainty."""
    run = data["run_index"]
    scene = data["scene_id"]
    episode = data["episode_id"]
    order = np.lexsort((episode, scene, run))
    sorted_run = run[order]
    sorted_scene = scene[order]
    sorted_episode = episode[order]
    starts = np.ones(len(order), dtype=np.bool_)
    if len(order) > 1:
        starts[1:] = (
            (sorted_run[1:] != sorted_run[:-1])
            | (sorted_scene[1:] != sorted_scene[:-1])
            | (sorted_episode[1:] != sorted_episode[:-1])
        )
    sorted_cluster = np.cumsum(starts, dtype=np.int64) - 1
    cluster_indices = np.empty(len(order), dtype=np.int64)
    cluster_indices[order] = sorted_cluster
    return cluster_indices


def _distance_bins(distance: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    distance = np.asarray(distance, dtype=np.float64)
    valid = np.isfinite(distance) & (distance >= 0.0)
    bins = np.full(distance.shape, -1, dtype=np.int8)
    bins[valid] = np.searchsorted(
        DISTANCE_UPPER_EDGES_M, distance[valid], side="right"
    ).astype(np.int8)
    return bins, valid


def _bin_label(bin_index: int) -> str:
    if bin_index < 0:
        return INVALID_DISTANCE_LABEL
    return DISTANCE_BIN_LABELS[bin_index]


def _derive_fields(data: dict[str, np.ndarray]) -> None:
    data["q_min"] = np.minimum(data["q1"], data["q2"])
    data["q_mean"] = 0.5 * (data["q1"] + data["q2"])
    data["target_q_min"] = np.minimum(data["target_q1"], data["target_q2"])
    data["target_q_mean"] = 0.5 * (data["target_q1"] + data["target_q2"])
    for prediction in PREDICTION_FIELDS:
        data[f"{prediction}_residual"] = data[prediction] - data["g_soft"]
    for prediction in ("q1", "q2", "q_min", "q_mean"):
        residual = data[f"{prediction}_residual"]
        data[f"{prediction}_abs_error"] = np.abs(residual)
        data[f"{prediction}_squared_error"] = np.square(residual)
    data["twin_disagreement"] = np.abs(data["q1"] - data["q2"])


def _summarize_values(
    values: np.ndarray,
    cluster_indices: np.ndarray,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    return _summarize_matrix(
        np.asarray(values, dtype=np.float64).reshape(-1, 1),
        cluster_indices,
        bootstrap_samples,
        rng,
    )[0]


def _summarize_matrix(
    values: np.ndarray,
    cluster_indices: np.ndarray,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> list[dict[str, float | int]]:
    """Summarize multiple row metrics with one shared cluster resampling pass."""
    values = np.asarray(values, dtype=np.float64)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    if values.shape[0] != len(cluster_indices):
        raise ValueError("values and cluster_indices must have the same row count")
    metric_count = values.shape[1]
    finite = np.isfinite(values)
    rows_with_values = np.any(finite, axis=1)
    if not np.any(rows_with_values):
        return [
            {
                "mean": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "scene_balanced_mean": float("nan"),
                "row_count": 0,
                "cluster_count": 0,
            }
            for _ in range(metric_count)
        ]

    selected_values = values[rows_with_values]
    selected_finite = finite[rows_with_values]
    _, inverse = np.unique(cluster_indices[rows_with_values], return_inverse=True)
    cluster_count = int(np.max(inverse)) + 1
    cluster_sums = np.zeros((cluster_count, metric_count), dtype=np.float64)
    cluster_counts = np.zeros((cluster_count, metric_count), dtype=np.float64)
    for metric_index in range(metric_count):
        metric_finite = selected_finite[:, metric_index]
        cluster_sums[:, metric_index] = np.bincount(
            inverse[metric_finite],
            weights=selected_values[metric_finite, metric_index],
            minlength=cluster_count,
        )
        cluster_counts[:, metric_index] = np.bincount(
            inverse[metric_finite], minlength=cluster_count
        )

    totals = np.sum(cluster_sums, axis=0)
    denominators = np.sum(cluster_counts, axis=0)
    means = np.divide(
        totals,
        denominators,
        out=np.full(metric_count, np.nan, dtype=np.float64),
        where=denominators > 0,
    )
    cluster_means = np.divide(
        cluster_sums,
        cluster_counts,
        out=np.full_like(cluster_sums, np.nan),
        where=cluster_counts > 0,
    )
    scene_counts = np.count_nonzero(cluster_counts > 0, axis=0)
    scene_balanced = np.divide(
        np.nansum(cluster_means, axis=0),
        scene_counts,
        out=np.full(metric_count, np.nan, dtype=np.float64),
        where=scene_counts > 0,
    )
    metric_cluster_counts = np.count_nonzero(cluster_counts > 0, axis=0)
    ci_low = np.full(metric_count, np.nan, dtype=np.float64)
    ci_high = np.full(metric_count, np.nan, dtype=np.float64)
    if cluster_count >= 2 and bootstrap_samples > 0:
        estimates = np.empty((bootstrap_samples, metric_count), dtype=np.float64)
        for replicate in range(bootstrap_samples):
            sampled_clusters = rng.integers(0, cluster_count, size=cluster_count)
            sampled_sums = np.sum(cluster_sums[sampled_clusters], axis=0)
            sampled_counts = np.sum(cluster_counts[sampled_clusters], axis=0)
            estimates[replicate] = np.divide(
                sampled_sums,
                sampled_counts,
                out=np.full(metric_count, np.nan, dtype=np.float64),
                where=sampled_counts > 0,
            )
        for metric_index in range(metric_count):
            finite_estimates = estimates[:, metric_index]
            finite_estimates = finite_estimates[np.isfinite(finite_estimates)]
            if len(finite_estimates):
                ci_low[metric_index], ci_high[metric_index] = np.quantile(
                    finite_estimates, [0.025, 0.975]
                )

    result: list[dict[str, float | int]] = []
    for metric_index in range(metric_count):
        result.append(
            {
                "mean": float(means[metric_index]),
                "ci_low": float(ci_low[metric_index]),
                "ci_high": float(ci_high[metric_index]),
                "scene_balanced_mean": float(scene_balanced[metric_index]),
                "row_count": int(denominators[metric_index]),
                "cluster_count": int(metric_cluster_counts[metric_index]),
            }
        )
    return result


def _count_unique_trajectory_keys(
    cluster_indices: np.ndarray, agent_slot: np.ndarray
) -> int:
    if len(cluster_indices) == 0:
        return 0
    keys = np.rec.fromarrays(
        [cluster_indices.astype(np.int64, copy=False), np.asarray(agent_slot)],
        names="cluster,agent",
    )
    return int(len(np.unique(keys)))


def _count_unique_timestep_keys(
    cluster_indices: np.ndarray, step: np.ndarray
) -> int:
    if len(cluster_indices) == 0:
        return 0
    keys = np.rec.fromarrays(
        [cluster_indices.astype(np.int64, copy=False), np.asarray(step)],
        names="cluster,step",
    )
    return int(len(np.unique(keys)))


def _support_status(cluster_count: int) -> str:
    if cluster_count >= MIN_INTERPRETABLE_CLUSTERS:
        return "at_least_30_clusters"
    return "sparse_under_30_clusters"


def _aggregate_metrics(
    data: dict[str, np.ndarray],
    cluster_indices: np.ndarray,
    mask: np.ndarray,
    fields: Iterable[str],
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    field_list = list(fields)
    values = np.column_stack([data[field][mask] for field in field_list])
    summaries = _summarize_matrix(
        values,
        cluster_indices[mask],
        bootstrap_samples,
        rng,
    )
    for field, summary in zip(field_list, summaries, strict=True):
        prefix = field
        output[f"mean_{prefix}"] = summary["mean"]
        output[f"{prefix}_ci_low"] = summary["ci_low"]
        output[f"{prefix}_ci_high"] = summary["ci_high"]
        output[f"scene_balanced_{prefix}"] = summary["scene_balanced_mean"]
    return output


def _distance_summary_rows(
    data: dict[str, np.ndarray],
    cluster_indices: np.ndarray,
    complete: np.ndarray,
    mismatch: np.ndarray,
    graph_bins: np.ndarray,
    euclidean_bins: np.ndarray,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    distance_rows: list[dict[str, Any]] = []
    residual_rows: list[dict[str, Any]] = []
    residual_metrics = (
        "q1_residual",
        "q2_residual",
        "q_min_residual",
        "q_mean_residual",
        "target_q1_residual",
        "target_q2_residual",
        "target_q_min_residual",
        "target_q_mean_residual",
        "q1_abs_error",
        "q2_abs_error",
        "q_min_abs_error",
        "q_mean_abs_error",
        "q1_squared_error",
        "q2_squared_error",
        "q_min_squared_error",
        "q_mean_squared_error",
        "twin_disagreement",
    )
    distance_metrics = PREDICTION_FIELDS + RETURN_FIELDS

    for source, distance_bins in (
        ("graph", graph_bins),
        ("euclidean", euclidean_bins),
    ):
        valid_bin_values = range(len(DISTANCE_BIN_LABELS))
        for bin_index in valid_bin_values:
            distance_mask = distance_bins == bin_index
            for subset_name, subset_mask in (
                ("all_rows", complete),
                ("exclude_goal_mismatch", complete & ~mismatch),
            ):
                mask = distance_mask & subset_mask
                cluster_count = int(len(np.unique(cluster_indices[mask])))
                row = {
                    "distance_source": source,
                    "distance_bin": _bin_label(bin_index),
                    "goal_mismatch_subset": subset_name,
                    "row_count": int(np.count_nonzero(mask)),
                    "cluster_count": cluster_count,
                    "trajectory_count": _count_unique_trajectory_keys(
                        cluster_indices[mask], data["agent_slot"][mask]
                    ),
                    "timestep_count": _count_unique_timestep_keys(
                        cluster_indices[mask], data["step"][mask]
                    ),
                    "support_status": _support_status(cluster_count),
                }
                row.update(
                    _aggregate_metrics(
                        data,
                        cluster_indices,
                        mask,
                        distance_metrics,
                        bootstrap_samples,
                        rng,
                    )
                )
                distance_rows.append(row)

                residual_row = {
                    "distance_source": source,
                    "distance_bin": _bin_label(bin_index),
                    "goal_mismatch_subset": subset_name,
                    "row_count": int(np.count_nonzero(mask)),
                    "cluster_count": cluster_count,
                    "support_status": _support_status(cluster_count),
                }
                residual_row.update(
                    _aggregate_metrics(
                        data,
                        cluster_indices,
                        mask,
                        residual_metrics,
                        bootstrap_samples,
                        rng,
                    )
                )
                for prediction in ("q1", "q2", "q_min", "q_mean"):
                    mse = residual_row[f"mean_{prediction}_squared_error"]
                    residual_row[f"{prediction}_rmse"] = (
                        float(np.sqrt(mse)) if np.isfinite(mse) else float("nan")
                    )
                    mse_ci_low = residual_row[f"{prediction}_squared_error_ci_low"]
                    mse_ci_high = residual_row[f"{prediction}_squared_error_ci_high"]
                    residual_row[f"{prediction}_rmse_ci_low"] = (
                        float(np.sqrt(max(0.0, mse_ci_low)))
                        if np.isfinite(mse_ci_low)
                        else float("nan")
                    )
                    residual_row[f"{prediction}_rmse_ci_high"] = (
                        float(np.sqrt(max(0.0, mse_ci_high)))
                        if np.isfinite(mse_ci_high)
                        else float("nan")
                    )
                residual_rows.append(residual_row)
    return distance_rows, residual_rows


def _coverage_rows(
    data: dict[str, np.ndarray],
    cluster_indices: np.ndarray,
    complete: np.ndarray,
    censored: np.ndarray,
    bootstrap_available: np.ndarray,
    graph_bins: np.ndarray,
    euclidean_bins: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, distance_bins in (
        ("graph", graph_bins),
        ("euclidean", euclidean_bins),
    ):
        for bin_index in range(-1, len(DISTANCE_BIN_LABELS)):
            mask = distance_bins == bin_index
            complete_mask = mask & complete
            censored_mask = mask & censored
            bootstrap_mask = mask & bootstrap_available
            row_count = int(np.count_nonzero(mask))
            cluster_count = int(len(np.unique(cluster_indices[mask])))
            rows.append(
                {
                    "distance_source": source,
                    "distance_bin": _bin_label(bin_index),
                    "row_count": row_count,
                    "timestep_count": _count_unique_timestep_keys(
                        cluster_indices[mask], data["step"][mask]
                    ),
                    "cluster_count": cluster_count,
                    "trajectory_count": _count_unique_trajectory_keys(
                        cluster_indices[mask], data["agent_slot"][mask]
                    ),
                    "complete_rows": int(np.count_nonzero(complete_mask)),
                    "complete_clusters": int(
                        len(np.unique(cluster_indices[complete_mask]))
                    ),
                    "censored_rows": int(np.count_nonzero(censored_mask)),
                    "censored_clusters": int(
                        len(np.unique(cluster_indices[censored_mask]))
                    ),
                    "bootstrap_tail_rows": int(np.count_nonzero(bootstrap_mask)),
                    "bootstrap_tail_clusters": int(
                        len(np.unique(cluster_indices[bootstrap_mask]))
                    ),
                    "goal_mismatch_rows": int(np.count_nonzero(mask & data["goal_mismatch"])),
                    "terminal_transition_rows": int(np.count_nonzero(mask & data["terminal"])),
                    "truncated_transition_rows": int(np.count_nonzero(mask & data["truncated"])),
                    "complete_fraction": (
                        float(np.count_nonzero(complete_mask) / row_count)
                        if row_count
                        else float("nan")
                    ),
                    "censored_fraction": (
                        float(np.count_nonzero(censored_mask) / row_count)
                        if row_count
                        else float("nan")
                    ),
                    "bootstrap_tail_fraction_of_censored": (
                        float(
                            np.count_nonzero(bootstrap_mask)
                            / np.count_nonzero(censored_mask)
                        )
                        if np.any(censored_mask)
                        else float("nan")
                    ),
                    "support_status": _support_status(cluster_count),
                }
            )
    return rows


def _censoring_rows(
    data: dict[str, np.ndarray],
    cluster_indices: np.ndarray,
    complete: np.ndarray,
    censored: np.ndarray,
    bootstrap_available: np.ndarray,
    graph_bins: np.ndarray,
    euclidean_bins: np.ndarray,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for source, distance_bins in (
        ("graph", graph_bins),
        ("euclidean", euclidean_bins),
    ):
        for bin_index in range(-1, len(DISTANCE_BIN_LABELS)):
            in_bin = distance_bins == bin_index
            censored_mask = in_bin & censored
            bootstrap_mask = in_bin & bootstrap_available
            horizon = data["horizon"][censored_mask]
            tail_discount = data["tail_discount"][censored_mask]
            partial_soft = data["g_soft"][censored_mask]
            bootstrapped_soft = data["g_soft_bootstrap"][bootstrap_mask]
            bootstrap_tail = data.get("bootstrap_tail", np.full(len(complete), np.nan))[
                bootstrap_mask
            ]

            def finite_mean(values: np.ndarray) -> float:
                finite = values[np.isfinite(values)]
                return float(np.mean(finite)) if len(finite) else float("nan")

            finite_horizon = horizon[np.isfinite(horizon)]
            rows.append(
                {
                    "distance_source": source,
                    "distance_bin": _bin_label(bin_index),
                    "all_rows": int(np.count_nonzero(in_bin)),
                    "complete_rows": int(np.count_nonzero(in_bin & complete)),
                    "complete_clusters": int(
                        len(np.unique(cluster_indices[in_bin & complete]))
                    ),
                    "censored_rows": int(np.count_nonzero(censored_mask)),
                    "censored_clusters": int(
                        len(np.unique(cluster_indices[censored_mask]))
                    ),
                    "bootstrap_tail_rows": int(np.count_nonzero(bootstrap_mask)),
                    "bootstrap_fraction_of_censored": (
                        float(np.count_nonzero(bootstrap_mask) / np.count_nonzero(censored_mask))
                        if np.any(censored_mask)
                        else float("nan")
                    ),
                    "mean_observed_horizon_censored": finite_mean(horizon),
                    "median_observed_horizon_censored": (
                        float(np.median(finite_horizon)) if len(finite_horizon) else float("nan")
                    ),
                    "mean_tail_discount_censored": finite_mean(tail_discount),
                    "mean_partial_observed_g_soft_censored": finite_mean(partial_soft),
                    "mean_bootstrapped_g_soft_censored": finite_mean(bootstrapped_soft),
                    "mean_discounted_bootstrap_contribution": finite_mean(bootstrap_tail),
                    "mean_complete_g_soft": finite_mean(data["g_soft"][in_bin & complete]),
                    "support_status": _support_status(
                        len(np.unique(cluster_indices[in_bin & complete]))
                    ),
                }
            )
    return rows


def _horizon_summary_rows(
    data: dict[str, np.ndarray],
    cluster_indices: np.ndarray,
    complete: np.ndarray,
    censored: np.ndarray,
    bootstrap_available: np.ndarray,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    horizon = np.asarray(data["horizon"], dtype=np.float64)
    valid_horizon = np.isfinite(horizon) & (horizon >= 1)
    horizon_bins = np.full(len(horizon), -1, dtype=np.int8)
    horizon_bins[valid_horizon] = np.searchsorted(
        HORIZON_UPPER_EDGES, horizon[valid_horizon], side="left"
    ).astype(np.int8)
    bootstrap_tail = data.get("bootstrap_tail", np.full(len(horizon), np.nan))
    rows = []
    fields = (
        "complete_g_soft",
        "partial_censored_g_soft",
        "bootstrapped_censored_g_soft",
        "discounted_bootstrap_contribution",
    )
    all_values = np.column_stack(
        (
            np.where(complete, data["g_soft"], np.nan),
            np.where(censored, data["g_soft"], np.nan),
            np.where(bootstrap_available, data["g_soft_bootstrap"], np.nan),
            np.where(bootstrap_available, bootstrap_tail, np.nan),
        )
    )
    for bin_index, label in enumerate(HORIZON_BIN_LABELS):
        mask = valid_horizon & (horizon_bins == bin_index)
        indices = np.flatnonzero(mask)
        summaries = _summarize_matrix(
            all_values[indices],
            cluster_indices[indices],
            bootstrap_samples,
            rng,
        )
        row: dict[str, Any] = {
            "horizon_bin_steps": label,
            "row_count": int(len(indices)),
            "cluster_count": int(len(np.unique(cluster_indices[indices]))),
            "complete_rows": int(np.count_nonzero(mask & complete)),
            "censored_rows": int(np.count_nonzero(mask & censored)),
            "bootstrap_tail_rows": int(np.count_nonzero(mask & bootstrap_available)),
            "complete_clusters": int(len(np.unique(cluster_indices[mask & complete]))),
            "censored_clusters": int(len(np.unique(cluster_indices[mask & censored]))),
            "support_status": _support_status(
                len(np.unique(cluster_indices[mask & complete]))
            ),
        }
        for field, summary in zip(fields, summaries, strict=True):
            row[f"mean_{field}"] = summary["mean"]
            row[f"{field}_ci_low"] = summary["ci_low"]
            row[f"{field}_ci_high"] = summary["ci_high"]
        rows.append(row)
    invalid = ~valid_horizon
    rows.append(
        {
            "horizon_bin_steps": INVALID_DISTANCE_LABEL,
            "row_count": int(np.count_nonzero(invalid)),
            "cluster_count": int(len(np.unique(cluster_indices[invalid]))),
            "complete_rows": int(np.count_nonzero(invalid & complete)),
            "censored_rows": int(np.count_nonzero(invalid & censored)),
            "bootstrap_tail_rows": int(np.count_nonzero(invalid & bootstrap_available)),
            "complete_clusters": int(len(np.unique(cluster_indices[invalid & complete]))),
            "censored_clusters": int(len(np.unique(cluster_indices[invalid & censored]))),
            "support_status": _support_status(
                len(np.unique(cluster_indices[invalid & complete]))
            ),
            **{
                f"mean_{field}": float("nan")
                for field in fields
            },
        }
    )
    return rows


def _quantile_groups(values: np.ndarray, requested_groups: int = 4) -> tuple[np.ndarray, list[str]]:
    finite = np.isfinite(values)
    labels = np.full(len(values), -1, dtype=np.int8)
    finite_values = values[finite]
    if not len(finite_values):
        return labels, []
    quantiles = np.quantile(finite_values, np.linspace(0.0, 1.0, requested_groups + 1))
    edges = np.unique(quantiles)
    if len(edges) <= 1:
        labels[finite] = 0
        return labels, ["constant"]
    labels[finite] = np.searchsorted(edges[1:-1], finite_values, side="right").astype(np.int8)
    return labels, [f"Q{index + 1}" for index in range(len(edges) - 1)]


def _reliability_rows(
    data: dict[str, np.ndarray],
    cluster_indices: np.ndarray,
    complete: np.ndarray,
    mismatch: np.ndarray,
    graph_bins: np.ndarray,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for bin_index in range(len(DISTANCE_BIN_LABELS)):
        for subset_name, subset_mask in (
            ("all_rows", complete),
            ("exclude_goal_mismatch", complete & ~mismatch),
        ):
            mask = (graph_bins == bin_index) & subset_mask
            finite_mask = mask & np.isfinite(data["q_min"]) & np.isfinite(data["g_soft"])
            if not np.any(finite_mask):
                continue
            row_indices = np.flatnonzero(finite_mask)
            groups, labels = _quantile_groups(data["q_min"][row_indices], requested_groups=5)
            for group_index, group_label in enumerate(labels):
                group_rows = row_indices[groups == group_index]
                if not len(group_rows):
                    continue
                target_summary = _summarize_values(
                    data["g_soft"][group_rows],
                    cluster_indices[group_rows],
                    bootstrap_samples,
                    rng,
                )
                prediction_summary = _summarize_values(
                    data["q_min"][group_rows],
                    cluster_indices[group_rows],
                    0,
                    rng,
                )
                rows.append(
                    {
                        "distance_source": "graph",
                        "distance_bin": _bin_label(bin_index),
                        "goal_mismatch_subset": subset_name,
                        "prediction": "q_min",
                        "prediction_group": group_label,
                        "row_count": int(len(group_rows)),
                        "cluster_count": int(target_summary["cluster_count"]),
                        "support_status": _support_status(int(target_summary["cluster_count"])),
                        "mean_predicted_q_min": prediction_summary["mean"],
                        "scene_balanced_predicted_q_min": prediction_summary[
                            "scene_balanced_mean"
                        ],
                        "mean_empirical_g_soft": target_summary["mean"],
                        "g_soft_ci_low": target_summary["ci_low"],
                        "g_soft_ci_high": target_summary["ci_high"],
                        "mean_empirical_minus_predicted": (
                            target_summary["mean"] - prediction_summary["mean"]
                        ),
                    }
                )
    return rows


def _speed_label(speed_mps: float) -> str:
    if speed_mps < 5:
        return "[0,5) m/s"
    if speed_mps < 10:
        return "[5,10) m/s"
    if speed_mps < 20:
        return "[10,20) m/s"
    return "[20,infinity) m/s"


def _confounder_rows(
    data: dict[str, np.ndarray],
    cluster_indices: np.ndarray,
    complete: np.ndarray,
    graph_bins: np.ndarray,
    bootstrap_samples: int,
    rng: np.random.Generator,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    factors: list[tuple[str, np.ndarray, list[str]]] = []
    speed_groups = np.full(len(complete), -1, dtype=np.int8)
    finite_speed = np.isfinite(data["speed_mps"]) & (data["speed_mps"] >= 0.0)
    speed_groups[finite_speed] = np.select(
        [data["speed_mps"][finite_speed] < 5, data["speed_mps"][finite_speed] < 10,
         data["speed_mps"][finite_speed] < 20],
        [0, 1, 2],
        default=3,
    )
    factors.append(("speed_mps", speed_groups, ["[0,5) m/s", "[5,10) m/s", "[10,20) m/s", "[20,infinity) m/s"]))

    varying_coefficients: dict[str, list[str]] = {}
    for field in COEFFICIENT_FIELDS:
        if field not in data:
            continue
        finite_values = data[field][complete & np.isfinite(data[field])]
        if len(finite_values) == 0 or len(np.unique(finite_values)) <= 1:
            continue
        groups, labels = _quantile_groups(data[field], requested_groups=4)
        factors.append((field, groups, labels))
        varying_coefficients[field] = labels

    for field, factor_groups, factor_labels in factors:
        for bin_index in range(len(DISTANCE_BIN_LABELS)):
            in_distance = complete & (graph_bins == bin_index)
            for factor_index, label in enumerate(factor_labels):
                mask = in_distance & (factor_groups == factor_index)
                cluster_count = int(len(np.unique(cluster_indices[mask])))
                row = {
                    "distance_source": "graph",
                    "distance_bin": _bin_label(bin_index),
                    "factor": field,
                    "stratum": label,
                    "row_count": int(np.count_nonzero(mask)),
                    "cluster_count": cluster_count,
                    "support_status": _support_status(cluster_count),
                }
                row.update(
                    _aggregate_metrics(
                        data,
                        cluster_indices,
                        mask,
                        ("q_min", "g_soft", "q_min_residual", "q_min_abs_error"),
                        bootstrap_samples,
                        rng,
                    )
                )
                rows.append(row)
    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]], preferred_fields: list[str]) -> None:
    fields = list(preferred_fields)
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def _csv_value(value: Any) -> Any:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return ""
    return value


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _distance_metric_map(rows: list[dict[str, Any]], source: str, subset: str) -> dict[str, dict[str, Any]]:
    return {
        row["distance_bin"]: row
        for row in rows
        if row["distance_source"] == source
        and row["goal_mismatch_subset"] == subset
    }


def _draw_empty(axis: plt.Axes, message: str) -> None:
    axis.text(0.5, 0.5, message, ha="center", va="center", transform=axis.transAxes)
    axis.set_axis_off()


def _plot_mean_interval(
    axis: plt.Axes,
    x: np.ndarray,
    means: np.ndarray,
    low: np.ndarray,
    high: np.ndarray,
    *,
    label: str,
    color: str,
    marker: str = "o",
    linestyle: str = "-",
    linewidth: float = 1.4,
    alpha: float = 1.0,
) -> bool:
    """Draw bootstrap interval endpoints independently of the point estimate."""
    means = np.asarray(means, dtype=np.float64)
    low = np.asarray(low, dtype=np.float64)
    high = np.asarray(high, dtype=np.float64)
    valid_mean = np.isfinite(means)
    if not np.any(valid_mean):
        return False
    valid_interval = valid_mean & np.isfinite(low) & np.isfinite(high) & (low <= high)
    if np.any(valid_interval):
        axis.vlines(
            x[valid_interval],
            low[valid_interval],
            high[valid_interval],
            color=color,
            alpha=alpha,
            linewidth=1.0,
        )
        cap_width = 0.045
        axis.hlines(
            low[valid_interval],
            x[valid_interval] - cap_width,
            x[valid_interval] + cap_width,
            color=color,
            alpha=alpha,
            linewidth=1.0,
        )
        axis.hlines(
            high[valid_interval],
            x[valid_interval] - cap_width,
            x[valid_interval] + cap_width,
            color=color,
            alpha=alpha,
            linewidth=1.0,
        )
    axis.plot(
        x[valid_mean],
        means[valid_mean],
        marker=marker,
        linestyle=linestyle,
        linewidth=linewidth,
        alpha=alpha,
        color=color,
        label=label,
    )
    return True


def _make_distance_calibration_plot(
    output_path: Path,
    summary_rows: list[dict[str, Any]],
    coverage_rows: list[dict[str, Any]],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(14, 5.2), sharey=False)
    colors = {
        "q1": "#4C78A8",
        "q2": "#F58518",
        "q_min": "#54A24B",
        "q_mean": "#B279A2",
        "g_soft": "#E45756",
        "g_reward": "#777777",
        "target_q_min": "#72B7B2",
    }
    labels = {
        "q1": "Q1",
        "q2": "Q2",
        "q_min": "min(Q1,Q2)",
        "q_mean": "mean(Q1,Q2)",
        "g_soft": "complete soft return",
        "g_reward": "complete reward-only return",
        "target_q_min": "target min-Q (secondary)",
    }
    for axis, source in zip(axes, ("graph", "euclidean"), strict=True):
        rows = _distance_metric_map(summary_rows, source, "all_rows")
        x = np.arange(len(DISTANCE_BIN_LABELS))
        plotted = False
        for metric in ("q1", "q2", "q_min", "q_mean", "g_soft", "g_reward", "target_q_min"):
            means = np.asarray([rows[label][f"mean_{metric}"] for label in DISTANCE_BIN_LABELS], dtype=float)
            lows = np.asarray([rows[label][f"{metric}_ci_low"] for label in DISTANCE_BIN_LABELS], dtype=float)
            highs = np.asarray([rows[label][f"{metric}_ci_high"] for label in DISTANCE_BIN_LABELS], dtype=float)
            if _plot_mean_interval(
                axis,
                x,
                means,
                lows,
                highs,
                marker="o",
                linewidth=1.4 if metric != "target_q_min" else 1.0,
                linestyle="--" if metric in ("g_reward", "target_q_min") else "-",
                alpha=0.9 if metric != "target_q_min" else 0.7,
                color=colors[metric],
                label=labels[metric],
            ):
                plotted = True
        if not plotted:
            _draw_empty(axis, "No complete true-terminal return rows")
            continue
        source_coverage = {
            row["distance_bin"]: row
            for row in coverage_rows
            if row["distance_source"] == source
        }
        complete_bin_rows = _distance_metric_map(summary_rows, source, "all_rows")
        tick_labels = [
            (
                f"{label}\nN={complete_bin_rows[label]['row_count']:,}, "
                f"C={complete_bin_rows[label]['cluster_count']:,}"
            )
            for label in DISTANCE_BIN_LABELS
        ]
        censored_fraction = np.asarray(
            [source_coverage[label]["censored_fraction"] for label in DISTANCE_BIN_LABELS],
            dtype=float,
        )
        secondary = axis.twinx()
        valid_fraction = np.isfinite(censored_fraction)
        secondary.plot(
            x[valid_fraction],
            censored_fraction[valid_fraction],
            color="#222222",
            marker="x",
            linestyle=":",
            linewidth=1.1,
            label="censored fraction (all sampled rows)",
        )
        secondary.set_ylim(0, 1)
        secondary.set_ylabel("Censored fraction")
        axis.set_xticks(x, tick_labels, rotation=25, ha="right", fontsize=7)
        axis.set_title(f"{source.capitalize()} distance")
        axis.set_xlabel("Distance bin (m)")
        axis.grid(alpha=0.2)
        handles, labels_for_legend = axis.get_legend_handles_labels()
        secondary_handles, secondary_labels = secondary.get_legend_handles_labels()
        axis.legend(handles + secondary_handles, labels_for_legend + secondary_labels, fontsize=7, loc="best")
    axes[0].set_ylabel("Q or realized return")
    figure.suptitle(
        "Distance calibration: realized returns are noisy samples; error bars resample scene/episode clusters"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _make_residual_plot(output_path: Path, rows: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex="col")
    x = np.arange(len(DISTANCE_BIN_LABELS))
    for column, source in enumerate(("graph", "euclidean")):
        source_rows = _distance_metric_map(rows, source, "all_rows")
        signed_axis = axes[0, column]
        absolute_axis = axes[1, column]
        plotted_signed = False
        plotted_absolute = False
        for metric, label, color in (
            ("q1_residual", "Q1 - G_soft", "#4C78A8"),
            ("q2_residual", "Q2 - G_soft", "#F58518"),
            ("q_min_residual", "min-Q - G_soft", "#54A24B"),
        ):
            means = np.asarray([source_rows[bin_label][f"mean_{metric}"] for bin_label in DISTANCE_BIN_LABELS], dtype=float)
            lows = np.asarray([source_rows[bin_label][f"{metric}_ci_low"] for bin_label in DISTANCE_BIN_LABELS], dtype=float)
            highs = np.asarray([source_rows[bin_label][f"{metric}_ci_high"] for bin_label in DISTANCE_BIN_LABELS], dtype=float)
            if _plot_mean_interval(
                signed_axis,
                x,
                means,
                lows,
                highs,
                label=label,
                color=color,
            ):
                plotted_signed = True
        min_mae = np.asarray([source_rows[bin_label]["mean_q_min_abs_error"] for bin_label in DISTANCE_BIN_LABELS], dtype=float)
        min_mae_low = np.asarray([source_rows[bin_label]["q_min_abs_error_ci_low"] for bin_label in DISTANCE_BIN_LABELS], dtype=float)
        min_mae_high = np.asarray([source_rows[bin_label]["q_min_abs_error_ci_high"] for bin_label in DISTANCE_BIN_LABELS], dtype=float)
        min_rmse = np.asarray([source_rows[bin_label]["q_min_rmse"] for bin_label in DISTANCE_BIN_LABELS], dtype=float)
        min_rmse_low = np.asarray([source_rows[bin_label]["q_min_rmse_ci_low"] for bin_label in DISTANCE_BIN_LABELS], dtype=float)
        min_rmse_high = np.asarray([source_rows[bin_label]["q_min_rmse_ci_high"] for bin_label in DISTANCE_BIN_LABELS], dtype=float)
        if np.any(np.isfinite(min_mae)):
            plotted_absolute = True
            _plot_mean_interval(
                absolute_axis,
                x,
                min_mae,
                min_mae_low,
                min_mae_high,
                marker="o",
                label="min-Q MAE",
                color="#E45756",
            )
            _plot_mean_interval(
                absolute_axis,
                x,
                min_rmse,
                min_rmse_low,
                min_rmse_high,
                marker="s",
                linestyle="--",
                label="min-Q RMSE",
                color="#B279A2",
            )
        if plotted_signed:
            signed_axis.axhline(0, color="black", linewidth=0.8)
            signed_axis.legend(fontsize=8)
            signed_axis.grid(alpha=0.2)
            signed_axis.set_title(f"{source.capitalize()}: signed error")
            signed_axis.set_ylabel("Mean residual (Q - G_soft)")
        else:
            _draw_empty(signed_axis, "No complete residuals")
        if plotted_absolute:
            absolute_axis.legend(fontsize=8)
            absolute_axis.grid(alpha=0.2)
            absolute_axis.set_title(f"{source.capitalize()}: absolute error")
            absolute_axis.set_ylabel("Return error")
            absolute_axis.set_xticks(x, DISTANCE_BIN_LABELS, rotation=25, ha="right")
            absolute_axis.set_xlabel("Distance bin (m)")
        else:
            _draw_empty(absolute_axis, "No complete residuals")
    figure.suptitle(
        "Residuals to one realized soft return; MAE/RMSE include policy and environment return variance"
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _make_reliability_plot(output_path: Path, rows: list[dict[str, Any]]) -> None:
    figure, axis = plt.subplots(figsize=(8, 7))
    colors = plt.get_cmap("viridis", len(DISTANCE_BIN_LABELS))
    plotted = False
    finite_values: list[float] = []
    for bin_index, bin_label in enumerate(DISTANCE_BIN_LABELS):
        selected = [
            row for row in rows
            if row["distance_bin"] == bin_label and row["goal_mismatch_subset"] == "all_rows"
        ]
        if not selected:
            continue
        predicted = np.asarray([row["mean_predicted_q_min"] for row in selected], dtype=float)
        empirical = np.asarray([row["mean_empirical_g_soft"] for row in selected], dtype=float)
        low = np.asarray([row["g_soft_ci_low"] for row in selected], dtype=float)
        high = np.asarray([row["g_soft_ci_high"] for row in selected], dtype=float)
        valid = np.isfinite(predicted) & np.isfinite(empirical)
        if not np.any(valid):
            continue
        plotted = True
        _plot_mean_interval(
            axis,
            predicted[valid],
            empirical[valid],
            low[valid],
            high[valid],
            marker="o",
            linestyle="-",
            color=colors(bin_index),
            label=f"{bin_label} m",
        )
        finite_values.extend(predicted[valid].tolist())
        finite_values.extend(empirical[valid].tolist())
    if not plotted:
        _draw_empty(axis, "No complete returns for reliability analysis")
    else:
        extent = max(1.0, max(abs(value) for value in finite_values))
        axis.plot([-extent, extent], [-extent, extent], color="black", linestyle="--", linewidth=1, label="identity")
        axis.set_xlim(-extent, extent)
        axis.set_ylim(-extent, extent)
        axis.set_aspect("equal", adjustable="box")
        axis.legend(fontsize=8, ncol=2)
        axis.grid(alpha=0.2)
        axis.set_xlabel("Mean predicted min(Q1,Q2) within prediction quantile")
        axis.set_ylabel("Mean empirical complete G_soft")
        axis.set_title("Prediction reliability within graph-distance strata")
    figure.suptitle("Distance-only averages can hide action-specific errors; points use min-Q prediction bins")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _make_behavior_plot(
    output_path: Path,
    data: dict[str, np.ndarray],
    complete: np.ndarray,
    censored: np.ndarray,
    graph_bins: np.ndarray,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    x = np.arange(len(DISTANCE_BIN_LABELS))
    metrics = (
        ("speed_mps", "Mean speed (m/s)", complete),
        ("goal_index", "Mean native active goal index", complete),
        ("goal_count", "Mean native active goal count", complete),
    )
    for axis, (field, ylabel, population) in zip(axes.flat[:3], metrics, strict=True):
        means = []
        for bin_index in range(len(DISTANCE_BIN_LABELS)):
            mask = population & (graph_bins == bin_index) & np.isfinite(data[field])
            means.append(float(np.mean(data[field][mask])) if np.any(mask) else float("nan"))
        axis.plot(x, means, marker="o", label=field)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
        axis.legend(fontsize=8)
        axis.set_xticks(x, DISTANCE_BIN_LABELS, rotation=25, ha="right")
        axis.set_xlabel("Graph-distance proxy bin (m)")

    axis = axes.flat[3]
    mismatch_rate = []
    goal_component = []
    goal_hit_rate = []
    has_goal_component = "reward_goal_component" in data and np.any(np.isfinite(data["reward_goal_component"]))
    has_goal_hit = "goal_hit_next" in data and np.any(np.isfinite(data["goal_hit_next"]))
    for bin_index in range(len(DISTANCE_BIN_LABELS)):
        mask = (graph_bins == bin_index) & (complete | censored)
        mismatch_rate.append(float(np.mean(data["goal_mismatch"][mask])) if np.any(mask) else float("nan"))
        if has_goal_component:
            goal_mask = complete & (graph_bins == bin_index) & np.isfinite(data["reward_goal_component"])
            goal_component.append(
                float(np.mean(data["reward_goal_component"][goal_mask])) if np.any(goal_mask) else float("nan")
            )
        if has_goal_hit:
            hit_mask = mask & np.isfinite(data["goal_hit_next"])
            goal_hit_rate.append(
                float(np.mean(data["goal_hit_next"][hit_mask])) if np.any(hit_mask) else float("nan")
            )
    axis.plot(x, mismatch_rate, marker="x", color="#E45756", label="goal mismatch rate")
    if has_goal_component:
        axis.plot(x, goal_component, marker="o", color="#4C78A8", label="mean goal reward component")
    if has_goal_hit:
        axis.plot(
            x,
            goal_hit_rate,
            marker="s",
            color="#54A24B",
            label="post-step reached-goal flag (not future success)",
        )
    axis.set_ylabel("Mismatch / available goal metrics")
    axis.set_xlabel("Graph-distance proxy bin (m)")
    axis.set_xticks(x, DISTANCE_BIN_LABELS, rotation=25, ha="right")
    axis.grid(alpha=0.2)
    axis.legend(fontsize=8)
    figure.suptitle("Behavior and observed-state confounders; see confounder_summary.csv for matched strata")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _make_tail_plot(
    output_path: Path,
    rows: list[dict[str, Any]],
    horizon_rows: list[dict[str, Any]],
) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(16, 5))
    graph_rows = {
        row["distance_bin"]: row
        for row in rows
        if row["distance_source"] == "graph"
    }
    x = np.arange(len(DISTANCE_BIN_LABELS))
    complete_rows = np.asarray([graph_rows[label]["complete_rows"] for label in DISTANCE_BIN_LABELS], dtype=float)
    censored_rows = np.asarray([graph_rows[label]["censored_rows"] for label in DISTANCE_BIN_LABELS], dtype=float)
    bootstrap_rows = np.asarray([graph_rows[label]["bootstrap_tail_rows"] for label in DISTANCE_BIN_LABELS], dtype=float)
    axes[0].bar(x - 0.25, complete_rows, width=0.25, label="complete true-terminal")
    axes[0].bar(x, censored_rows, width=0.25, label="censored (partial observed)")
    axes[0].bar(x + 0.25, bootstrap_rows, width=0.25, label="censored with bootstrap estimate")
    axes[0].set_title("Rows by endpoint category")
    axes[0].set_ylabel("Sampled transition rows")
    axes[0].legend(fontsize=7)
    axes[0].grid(axis="y", alpha=0.2)

    means_horizon = np.asarray(
        [graph_rows[label]["mean_observed_horizon_censored"] for label in DISTANCE_BIN_LABELS], dtype=float
    )
    medians_horizon = np.asarray(
        [graph_rows[label]["median_observed_horizon_censored"] for label in DISTANCE_BIN_LABELS], dtype=float
    )
    axes[1].plot(x, means_horizon, marker="o", label="mean censored horizon")
    axes[1].plot(x, medians_horizon, marker="s", linestyle="--", label="median censored horizon")
    axes[1].set_title("Observed suffix horizon")
    axes[1].set_ylabel("Steps")
    axes[1].legend(fontsize=8)
    axes[1].grid(alpha=0.2)

    horizon_data = [
        row for row in horizon_rows if row["horizon_bin_steps"] in HORIZON_BIN_LABELS
    ]
    horizon_x = np.arange(len(horizon_data))
    for field, label, color, marker in (
        ("complete_g_soft", "complete G_soft (sampled MC)", "#4C78A8", "o"),
        ("partial_censored_g_soft", "censored partial G_soft", "#F58518", "x"),
        ("bootstrapped_censored_g_soft", "censored bootstrapped G_soft", "#54A24B", "s"),
    ):
        means = np.asarray([row[f"mean_{field}"] for row in horizon_data], dtype=float)
        lows = np.asarray([row.get(f"{field}_ci_low", np.nan) for row in horizon_data], dtype=float)
        highs = np.asarray([row.get(f"{field}_ci_high", np.nan) for row in horizon_data], dtype=float)
        _plot_mean_interval(
            axes[2],
            horizon_x,
            means,
            lows,
            highs,
            marker=marker,
            linestyle="--" if field == "partial_censored_g_soft" else "-",
            color=color,
            label=label,
        )
    axes[2].set_title("Returns by available observed suffix horizon")
    axes[2].set_ylabel("Return estimate (not interchangeable)")
    axes[2].set_xticks(
        horizon_x,
        [f"{row['horizon_bin_steps']}\nN={row['row_count']:,}" for row in horizon_data],
        rotation=25,
        ha="right",
        fontsize=7,
    )
    axes[2].legend(fontsize=7)
    axes[2].grid(alpha=0.2)
    for axis in axes[:2]:
        axis.set_xticks(x, DISTANCE_BIN_LABELS, rotation=25, ha="right")
        axis.set_xlabel("Graph-distance proxy bin (m)")
    axes[2].set_xlabel("Observed suffix length (steps; bins, not re-truncated trajectories)")
    figure.suptitle("Bootstrapped tails are secondary estimates, never Monte Carlo ground truth")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _overall_evidence(
    data: dict[str, np.ndarray],
    cluster_indices: np.ndarray,
    complete: np.ndarray,
    mismatch: np.ndarray,
    graph_bins: np.ndarray,
) -> dict[str, Any]:
    complete_soft = data["g_soft"][complete]
    variance = float(np.var(complete_soft)) if len(complete_soft) else float("nan")
    complete_cluster_count = int(len(np.unique(cluster_indices[complete])))
    mismatch_rows = []
    for subset_name, mask in (
        ("all_rows", complete),
        ("exclude_goal_mismatch", complete & ~mismatch),
    ):
        residual = data["q_min_residual"][mask]
        residual = residual[np.isfinite(residual)]
        mismatch_rows.append(
            {
                "subset": subset_name,
                "row_count": int(len(residual)),
                "cluster_count": int(len(np.unique(cluster_indices[mask]))),
                "mean_min_q_residual": float(np.mean(residual)) if len(residual) else float("nan"),
                "min_q_mae": float(np.mean(np.abs(residual))) if len(residual) else float("nan"),
                "min_q_rmse": float(np.sqrt(np.mean(np.square(residual)))) if len(residual) else float("nan"),
            }
        )
    distance_counts = []
    for bin_index, label in enumerate(DISTANCE_BIN_LABELS):
        mask = (graph_bins == bin_index) & complete
        clusters = int(len(np.unique(cluster_indices[mask])))
        distance_counts.append(
            {
                "distance_bin": label,
                "complete_rows": int(np.count_nonzero(mask)),
                "complete_clusters": clusters,
                "support_status": _support_status(clusters),
            }
        )
    graph_status_names = {
        0: "no_active_goal",
        1: "missing_current_or_goal_lane",
        2: "lane_graph_unavailable",
        3: "current_or_goal_lane_absent_from_graph",
        4: "unreachable_or_invalid_graph_entry",
        5: "reachable",
        6: "inactive_or_padded_slot",
    }
    graph_status_counts: dict[str, int] = {}
    if "graph_status" in data:
        status_values = data["graph_status"]
        finite_status = status_values[np.isfinite(status_values)]
        for status_value, count in zip(*np.unique(finite_status, return_counts=True), strict=True):
            status_code = int(status_value)
            status_name = graph_status_names.get(status_code, f"unknown_status_{status_code}")
            graph_status_counts[status_name] = int(count)
    optional_behavior = [
        field
        for field in (
            "waypoint_success",
            "collision",
            "offroad",
            "reward_goal_component",
            "goal_hit_next",
        )
        if field in data and np.any(np.isfinite(data[field]))
    ]
    coefficient_summary = {}
    for field in COEFFICIENT_FIELDS:
        if field not in data:
            continue
        values = data[field][np.isfinite(data[field])]
        if len(values):
            unique_values = np.unique(values)
            coefficient_summary[field] = {
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
                "unique_sampled_values": int(len(unique_values)),
                "stratified": bool(len(unique_values) > 1),
            }
    return {
        "row_count": int(len(data["q1"])),
        "complete_true_terminal_target_rows": int(np.count_nonzero(complete)),
        "censored_rows": int(np.count_nonzero(data["censored"])),
        "censored_rows_with_bootstrap_estimate": int(
            np.count_nonzero(data["censored"] & np.isfinite(data["g_soft_bootstrap"]))
        ),
        "complete_cluster_count": complete_cluster_count,
        "independent_cluster_definition": "run_id + scene_id + episode_id; all agents, sampled steps, and shared rollout rows remain together",
        "minimum_interpretable_clusters_per_distance_bin": MIN_INTERPRETABLE_CLUSTERS,
        "complete_g_soft_mean": float(np.mean(complete_soft)) if len(complete_soft) else float("nan"),
        "complete_g_soft_variance_across_realized_rows": variance,
        "complete_g_soft_standard_deviation_across_realized_rows": float(np.sqrt(variance)) if np.isfinite(variance) else float("nan"),
        "min_q_mismatch_sensitivity": mismatch_rows,
        "graph_distance_coverage": distance_counts,
        "graph_distance_status_counts": graph_status_counts,
        "coefficient_summary": coefficient_summary,
        "behavior_fields_available": optional_behavior,
        "behavior_fields_missing": [
            field
            for field in (
                "waypoint_success",
                "collision",
                "offroad",
                "reward_goal_component",
                "goal_hit_next",
            )
            if field not in optional_behavior
        ],
        "interpretation": {
            "complete_target": "Rows with censored=false and finite g_soft; this suffix reaches a true terminal even when the selected row itself is not terminal.",
            "censored_rows": "g_soft is an observed partial return and is not treated as zero-tail truth or a general lower bound.",
            "bootstrap_tail": "g_soft_bootstrap is a separate critic-dependent estimate and is never included as Monte Carlo ground truth.",
            "realized_return_noise": "One realized return is a noisy sample of conditional Q; reported RMSE contains policy and environment return variance.",
            "flat_q": "Flat Q by distance alone is not evidence of miscalibration; assess Q against returns within distance and confounder strata.",
            "distance": "graph_distance_m is a raw lane-graph proxy, not exact route length; Euclidean distance is reported in parallel.",
        },
    }


def analyze_runs(
    run_patterns: Iterable[str | Path],
    output_directory: str | Path,
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
) -> dict[str, Any]:
    if bootstrap_samples < 0:
        raise ValueError("bootstrap_samples must be nonnegative")
    run_directories = _resolve_run_directories(run_patterns)
    run_contracts = _validate_run_compatibility(run_directories)
    data, run_summaries = _load_runs(run_directories)
    _derive_fields(data)
    cluster_indices = _cluster_indices(data)
    graph_bins, graph_valid = _distance_bins(data["graph_distance_m"])
    euclidean_bins, euclidean_valid = _distance_bins(data["euclidean_distance_m"])
    if "graph_status" in data:
        graph_valid &= data["graph_status"] == 5
    graph_bins[~graph_valid] = -1
    euclidean_bins[~euclidean_valid] = -1

    complete = ~data["censored"] & np.isfinite(data["g_soft"])
    censored = data["censored"]
    bootstrap_available = censored & np.isfinite(data["g_soft_bootstrap"])
    mismatch = data["goal_mismatch"]
    rng = np.random.default_rng(BOOTSTRAP_SEED)

    distance_rows, residual_rows = _distance_summary_rows(
        data,
        cluster_indices,
        complete,
        mismatch,
        graph_bins,
        euclidean_bins,
        bootstrap_samples,
        rng,
    )
    coverage_rows = _coverage_rows(
        data,
        cluster_indices,
        complete,
        censored,
        bootstrap_available,
        graph_bins,
        euclidean_bins,
    )
    censoring_rows = _censoring_rows(
        data,
        cluster_indices,
        complete,
        censored,
        bootstrap_available,
        graph_bins,
        euclidean_bins,
    )
    reliability_rows = _reliability_rows(
        data,
        cluster_indices,
        complete,
        mismatch,
        graph_bins,
        bootstrap_samples,
        rng,
    )
    confounder_rows = _confounder_rows(
        data,
        cluster_indices,
        complete,
        graph_bins,
        0,
        rng,
    )
    horizon_rows = _horizon_summary_rows(
        data,
        cluster_indices,
        complete,
        censored,
        bootstrap_available,
        bootstrap_samples,
        rng,
    )

    output_path = Path(output_directory).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    _write_csv(
        output_path / "distance_summary.csv",
        distance_rows,
        ["distance_source", "distance_bin", "goal_mismatch_subset", "row_count", "cluster_count", "trajectory_count", "timestep_count", "support_status"],
    )
    _write_csv(
        output_path / "residual_summary.csv",
        residual_rows,
        ["distance_source", "distance_bin", "goal_mismatch_subset", "row_count", "cluster_count", "support_status"],
    )
    _write_csv(
        output_path / "coverage.csv",
        coverage_rows,
        ["distance_source", "distance_bin", "row_count", "timestep_count", "cluster_count", "trajectory_count", "complete_rows", "complete_clusters", "censored_rows", "censored_clusters", "bootstrap_tail_rows", "bootstrap_tail_clusters"],
    )
    _write_csv(
        output_path / "reliability.csv",
        reliability_rows,
        ["distance_source", "distance_bin", "goal_mismatch_subset", "prediction", "prediction_group", "row_count", "cluster_count", "support_status"],
    )
    _write_csv(
        output_path / "confounder_summary.csv",
        confounder_rows,
        ["distance_source", "distance_bin", "factor", "stratum", "row_count", "cluster_count", "support_status"],
    )
    _write_csv(
        output_path / "censoring_summary.csv",
        censoring_rows,
        ["distance_source", "distance_bin", "all_rows", "complete_rows", "censored_rows", "bootstrap_tail_rows", "support_status"],
    )
    _write_csv(
        output_path / "horizon_sensitivity.csv",
        horizon_rows,
        ["horizon_bin_steps", "row_count", "cluster_count", "complete_rows", "censored_rows", "bootstrap_tail_rows", "support_status"],
    )

    _make_distance_calibration_plot(
        output_path / "distance_calibration.png", distance_rows, coverage_rows
    )
    _make_residual_plot(output_path / "residuals_by_distance.png", residual_rows)
    _make_reliability_plot(output_path / "value_reliability.png", reliability_rows)
    _make_behavior_plot(output_path / "behavior_confounders.png", data, complete, censored, graph_bins)

    complete_soft_mean = float(np.mean(data["g_soft"][complete])) if np.any(complete) else float("nan")
    _make_tail_plot(output_path / "tail_sensitivity.png", censoring_rows, horizon_rows)

    evidence = {
        "analysis_version": 1,
        "run_count": len(run_summaries),
        "runs": run_summaries,
        "pooling_contracts": run_contracts,
        "row_count": int(len(data["q1"])),
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "distance_bins_m": DISTANCE_BIN_LABELS,
        "invalid_distance_category": INVALID_DISTANCE_LABEL,
        "optional_columns_present": sorted(OPTIONAL_COLUMNS & set(data)),
        "overall": _overall_evidence(data, cluster_indices, complete, mismatch, graph_bins),
        "outputs": [
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
        ],
    }
    evidence["overall"]["complete_g_soft_mean"] = complete_soft_mean
    evidence_path = output_path / "analysis.json"
    with evidence_path.open("w", encoding="utf-8") as handle:
        json.dump(_json_safe(evidence), handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    return evidence


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--runs",
        nargs="+",
        required=True,
        help="Run directories or glob patterns containing transitions.npz and run.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Local directory for CSV summaries, PNG figures, and analysis.json",
    )
    parser.add_argument(
        "--bootstrap-samples",
        type=int,
        default=DEFAULT_BOOTSTRAP_SAMPLES,
        help=f"Scene/episode cluster bootstrap replicates (default: {DEFAULT_BOOTSTRAP_SAMPLES})",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    evidence = analyze_runs(args.runs, args.output_dir, args.bootstrap_samples)
    print(
        f"Analyzed {evidence['row_count']:,} transition rows from {evidence['run_count']} runs; "
        f"{evidence['overall']['complete_true_terminal_target_rows']:,} rows have complete true-terminal G_soft."
    )
    print(f"Wrote analysis artifacts to {Path(args.output_dir).resolve()}")


if __name__ == "__main__":
    main()
