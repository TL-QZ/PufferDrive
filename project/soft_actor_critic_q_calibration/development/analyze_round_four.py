"""Analyze round-four action outcomes, correlations, and rankings."""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import kendalltau, spearmanr

from .plot_binned_driving_quality import METRICS as BASE_METRICS
from .run_layout import model_directory, records_directory

HORIZONS = ("full", "8", "32", "128", "1280", "2560")
STATISTICS = ("pearson_r", "pairwise_ranking_accuracy", "spearman_rho", "kendall_tau_b")
ALL_METRIC_LABELS = dict(BASE_METRICS, discounted_reward_return="Discounted reward return")
STATISTIC_LABELS = {
    "pearson_r": "Pearson r (raw outcome)",
    "pairwise_ranking_accuracy": "Pairwise ranking accuracy",
    "spearman_rho": "Spearman rho (raw outcome)",
    "kendall_tau_b": "Kendall tau-b (raw outcome)",
}
BIN_AXES = (
    "nearest_observed_partner_center_distance_m",
    "nearest_observed_road_boundary_distance_m",
    "nearest_visible_goal_distance_m",
)
BIN_AXIS_LABELS = {
    BIN_AXES[0]: "Nearest observed agent",
    BIN_AXES[1]: "Nearest observed road boundary",
    BIN_AXES[2]: "Next unfulfilled goal (route order)",
}


def metric_labels(model_name):
    labels = dict(BASE_METRICS)
    labels["discounted_reward_return"] = "Discounted reward return"
    if model_name == "ppo":
        labels.pop("standard_soft_return", None)
    return labels


def branch_target(branch, model_name):
    key = "ppo_value_target" if model_name == "ppo" else "actual_critic_target"
    return optional_float(branch.get(key))


def optional_float(value):
    if value is None:
        return math.nan
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def branch_outcome(branch, horizon, metric):
    if horizon == "full":
        if metric in ("discounted_reward_return", "standard_soft_return"):
            return optional_float(branch.get(metric))
        return optional_float(branch.get("metrics_mean", {}).get(metric))
    snapshot = branch.get("horizon_metrics", {}).get(horizon)
    if snapshot is None:
        return math.nan
    if metric in ("discounted_reward_return", "standard_soft_return"):
        value = snapshot.get(metric)
        return optional_float(value)
    return optional_float(snapshot.get("metrics_mean", {}).get(metric))


def action_mean_rows(branches, *, model_name, horizon, metrics, state_id, assigned_bin):
    """Pair target and outcome on the exact same repeats for each action."""
    by_action = defaultdict(list)
    for branch in branches:
        if horizon == "full" or horizon in branch.get("horizon_metrics", {}):
            by_action[int(branch["action_idx"])].append(branch)
    output = []
    for metric, label in metrics.items():
        for action_idx in sorted(by_action):
            pairs = []
            for branch in by_action[action_idx]:
                target = branch_target(branch, model_name)
                outcome = branch_outcome(branch, horizon, metric)
                if math.isfinite(target) and math.isfinite(outcome):
                    pairs.append((target, outcome, int(branch.get("repeat_idx", 0))))
            if not pairs:
                continue
            output.append({
                "state_id": state_id,
                "bin_id": assigned_bin["bin_id"],
                "bin_axis": assigned_bin["axis"],
                "bin_lower_m": assigned_bin["lower_m"],
                "bin_upper_m": assigned_bin["upper_m"],
                "horizon": horizon,
                "metric": metric,
                "metric_label": label,
                "action_idx": action_idx,
                "target_mean": float(np.mean([item[0] for item in pairs])),
                "outcome_mean": float(np.mean([item[1] for item in pairs])),
                "repeat_count": len(pairs),
                "repeat_indices": ",".join(str(item[2]) for item in pairs),
            })
    return output


def score_actions(targets, outcomes, *, lower_is_better=False):
    """Return raw-direction correlations and desirability-oriented pair accuracy."""
    target = np.asarray(targets, dtype=float)
    outcome = np.asarray(outcomes, dtype=float)
    valid = np.isfinite(target) & np.isfinite(outcome)
    target, outcome = target[valid], outcome[valid]
    possible_pairs = int(target.size * (target.size - 1) // 2)
    empty = {
        "pearson_r": math.nan,
        "pairwise_ranking_accuracy": math.nan,
        "spearman_rho": math.nan,
        "kendall_tau_b": math.nan,
        "action_count": int(target.size),
        "possible_pair_count": possible_pairs,
        "comparable_pair_count": 0,
        "tied_outcome_pair_count": possible_pairs,
    }
    if target.size < 2:
        return empty
    pearson = (
        float(np.corrcoef(target, outcome)[0, 1])
        if np.ptp(target) > 0 and np.ptp(outcome) > 0 else math.nan
    )
    # Correlation coefficients retain the recorded outcome direction, as in round 3.
    rank_outcome = -outcome if lower_is_better else outcome
    left, right = np.triu_indices(target.size, k=1)
    outcome_order = np.sign(rank_outcome[left] - rank_outcome[right])
    target_order = np.sign(target[left] - target[right])
    comparable = outcome_order != 0
    accuracy = (
        float(np.mean((1 + outcome_order[comparable] * target_order[comparable]) / 2))
        if comparable.any() else math.nan
    )
    rho = tau = math.nan
    if np.ptp(target) > 0 and np.ptp(outcome) > 0:
        rho = float(spearmanr(target, outcome).statistic)
        tau = float(kendalltau(target, outcome, variant="b").statistic)
    return {
        "pearson_r": pearson,
        "pairwise_ranking_accuracy": accuracy,
        "spearman_rho": rho,
        "kendall_tau_b": tau,
        "action_count": int(target.size),
        "possible_pair_count": possible_pairs,
        "comparable_pair_count": int(comparable.sum()),
        "tied_outcome_pair_count": int((~comparable).sum()),
    }


def state_score_rows(action_rows):
    grouped = defaultdict(list)
    for row in action_rows:
        grouped[(row["state_id"], row["horizon"], row["metric"])].append(row)
    output = []
    for (state_id, horizon, metric), selected in sorted(grouped.items()):
        selected.sort(key=lambda row: row["action_idx"])
        first = selected[0]
        values = score_actions(
            [row["target_mean"] for row in selected],
            [row["outcome_mean"] for row in selected],
            lower_is_better=(metric == "total_infractions"),
        )
        output.append({
            "state_id": state_id,
            "bin_id": first["bin_id"],
            "bin_axis": first["bin_axis"],
            "bin_lower_m": first["bin_lower_m"],
            "bin_upper_m": first["bin_upper_m"],
            "horizon": horizon,
            "metric": metric,
            "metric_label": first["metric_label"],
            "mean_action_outcome": float(np.mean([row["outcome_mean"] for row in selected])),
            "mean_repeats_per_action": float(np.mean([row["repeat_count"] for row in selected])),
            **values,
        })
    return output


def summarize(action_rows, score_rows, group_columns, expected_groups, selected_state_counts):
    action_groups = defaultdict(list)
    score_groups = defaultdict(list)
    for row in action_rows:
        key = tuple(row[column] for column in group_columns) + (row["metric"],)
        action_groups[key].append(row)
    for row in score_rows:
        key = tuple(row[column] for column in group_columns) + (row["metric"],)
        score_groups[key].append(row)
    output = []
    for key in expected_groups:
        action_groups.setdefault(key, [])
    for key, group in sorted(action_groups.items(), key=lambda item: tuple(map(str, item[0]))):
        statistics = score_groups[key]
        summary = dict(zip((*group_columns, "metric"), key))
        group_key = key[:-1]
        selected_state_count = selected_state_counts[group_key]
        summary.update({
            "metric_label": group[0]["metric_label"] if group else ALL_METRIC_LABELS.get(key[-1], key[-1]),
            "selected_state_count": selected_state_count,
            "state_count": len({row["state_id"] for row in group}),
            "action_mean_count": len(group),
            "eligible_repeat_count": int(sum(row["repeat_count"] for row in group)),
            "states_with_two_actions": sum(row["action_count"] >= 2 for row in statistics),
            "mean_repeats_per_action": float(np.mean([row["repeat_count"] for row in group])) if group else math.nan,
            "raw_mean_outcome": float(np.mean([row["outcome_mean"] for row in group])) if group else math.nan,
        })
        for statistic in STATISTICS:
            values = np.asarray([row[statistic] for row in statistics], dtype=float)
            values = values[np.isfinite(values)]
            summary[f"mean_{statistic}"] = float(values.mean()) if values.size else math.nan
            summary[f"states_scored_{statistic}"] = int(values.size)
        summary["comparable_action_pairs"] = int(sum(row["comparable_pair_count"] for row in statistics))
        summary["possible_action_pairs"] = int(sum(row["possible_pair_count"] for row in statistics))
        output.append(summary)
    return output


def read_model(model_dir):
    records_dir = records_directory(model_dir)
    selection = json.loads((records_dir / "selection.json").read_text())
    name = model_dir.name.lower()
    metrics = metric_labels(name)
    action_rows = []
    for state in selection["states"]:
        stem = state["state_id"].replace("/", "__")
        probe = json.loads((records_dir / "probes" / f"{stem}.json").read_text())
        branches = probe["branches"]
        if probe["state_id"] != state["state_id"]:
            raise ValueError(f"Probe identity mismatch: {stem}")
        expected = {(repeat, action) for repeat in range(4) for action in range(12)}
        actual = {(int(row["repeat_idx"]), int(row["action_idx"])) for row in branches}
        if actual != expected or len(actual) != len(branches):
            raise ValueError(f"{stem} must have four repeats for all 12 actions")
        for horizon in HORIZONS:
            action_rows.extend(action_mean_rows(
                branches, model_name=name, horizon=horizon, metrics=metrics,
                state_id=state["state_id"], assigned_bin=state["assigned_bin"],
            ))
    scores = state_score_rows(action_rows)
    bins = make_bins(selection)
    metric_names = list(metrics)
    expected_pooled = [
        (horizon, metric)
        for horizon in HORIZONS
        for metric in metric_names
    ]
    selected_pooled = {(horizon,): len(selection["states"]) for horizon in HORIZONS}
    expected_binned = [
        (horizon, item["axis"], item["bin_id"], metric)
        for horizon in HORIZONS
        for item in bins
        for metric in metric_names
    ]
    selected_binned = {
        (horizon, item["axis"], item["bin_id"]): item["state_count"]
        for horizon in HORIZONS
        for item in bins
    }
    pooled = summarize(action_rows, scores, ("horizon",), expected_pooled, selected_pooled)
    binned = summarize(
        action_rows, scores, ("horizon", "bin_axis", "bin_id"),
        expected_binned, selected_binned,
    )
    return selection, metrics, action_rows, scores, pooled, binned


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = []
    for row in rows:
        fields.extend(key for key in row if key not in fields)
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({
                key: "" if isinstance(value, float) and not math.isfinite(value) else value
                for key, value in row.items()
            })


def index_rows(rows):
    """Index by horizon and metric to keep repeated plot lookup linear."""
    index = defaultdict(list)
    for row in rows:
        base_key = (row["horizon"], row["metric"])
        index[base_key].append(row)
        index[(*base_key, row["bin_id"])].append(row)
    return index


def plot_raw_pooled(action_index, model, horizon, metrics, destination):
    fig, panels = plt.subplots(4, 3, figsize=(16, 14), constrained_layout=True)
    for index, (metric, label) in enumerate(metrics.items()):
        panel = panels.flat[index]
        state_groups = defaultdict(list)
        for row in action_index[(horizon, metric)]:
            state_groups[row["state_id"]].append(row)
        for state_rows in state_groups.values():
            state_rows.sort(key=lambda row: row["target_mean"])
            x = [row["target_mean"] for row in state_rows]
            y = [row["outcome_mean"] for row in state_rows]
            panel.plot(x, y, color="#176b87", lw=.55, alpha=.24)
            panel.scatter(x, y, color="#176b87", s=7, alpha=.32, linewidths=0)
        panel.set_title(label, fontsize=9)
        panel.set_xlabel("Target mean over eligible repeats", fontsize=8)
        panel.grid(axis="y", alpha=.17)
        panel.spines[["top", "right"]].set_visible(False)
        if metric in ("num_goals_reached", "total_infractions"):
            panel.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    for panel in panels.flat[len(metrics):]:
        panel.set_visible(False)
    fig.suptitle(f"{model.upper()} · {horizon} · raw outcomes by action", fontsize=16)
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def plot_pooled_statistics(score_index, model, horizon, metrics, destination):
    fig, panels = plt.subplots(4, 3, figsize=(16, 14), constrained_layout=True)
    for index, (metric, label) in enumerate(metrics.items()):
        panel = panels.flat[index]
        selected = score_index[(horizon, metric)]
        for stat_idx, statistic in enumerate(STATISTICS):
            values = np.asarray([row[statistic] for row in selected], dtype=float)
            finite = values[np.isfinite(values)]
            if not finite.size:
                continue
            jitter = np.linspace(-.18, .18, len(finite)) if len(finite) > 1 else np.zeros(1)
            panel.scatter(stat_idx + jitter, finite, s=7, alpha=.20, color="#146882", linewidths=0)
            panel.plot([stat_idx - .24, stat_idx + .24], [finite.mean(), finite.mean()],
                       color="#a44116", lw=1.5)
        counts = [sum(math.isfinite(row[stat]) for row in selected) for stat in STATISTICS]
        panel.set_title(f"{label}\nscored states " + " / ".join(map(str, counts)), fontsize=8)
        panel.set_xticks(range(4), ["Pearson", "Pairwise", "Spearman", "Kendall"], rotation=24, fontsize=7)
        panel.set_ylim(-1.05, 1.05)
        panel.axhline(0, color="#777777", linestyle="--", lw=.7)
        panel.plot([.76, 1.24], [.5, .5], color="#777777", linestyle=":", lw=.9)
        panel.grid(axis="y", alpha=.17)
        panel.spines[["top", "right"]].set_visible(False)
    for panel in panels.flat[len(metrics):]:
        panel.set_visible(False)
    fig.suptitle(
        f"{model.upper()} · {horizon} · pooled state-level statistics\n"
        "Orange marks the equal-state mean. Dashed baselines are zero and 50% chance.",
        fontsize=15,
    )
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def make_bins(selection):
    grouped = {}
    for state in selection["states"]:
        bounds = state["assigned_bin"]
        item = grouped.setdefault(bounds["bin_id"], {
            "bin_id": bounds["bin_id"],
            "axis": bounds["axis"],
            "lower_m": float(bounds["lower_m"]),
            "upper_m": None if bounds["upper_m"] is None else float(bounds["upper_m"]),
            "state_count": 0,
        })
        item["state_count"] += 1
    for item in grouped.values():
        upper = "∞" if item["upper_m"] is None else f'{item["upper_m"]:g}'
        item["label"] = f'{item["lower_m"]:g}–{upper} m'
    return list(grouped.values())


def plot_binned_raw(action_index, model, horizon, metric, label, bins, destination):
    fig, panels = plt.subplots(3, 7, figsize=(23, 10), squeeze=False)
    for row_idx, axis in enumerate(BIN_AXES):
        bin_group = sorted(
            [item for item in bins if item["axis"] == axis],
            key=lambda item: item["lower_m"],
        )
        for col_idx, panel in enumerate(panels[row_idx]):
            if col_idx >= len(bin_group):
                panel.set_visible(False)
                continue
            item = bin_group[col_idx]
            grouped = defaultdict(list)
            for record in action_index[(horizon, metric, item["bin_id"])]:
                grouped[record["state_id"]].append(record)
            for state_rows in grouped.values():
                state_rows.sort(key=lambda record: record["target_mean"])
                x = [record["target_mean"] for record in state_rows]
                y = [record["outcome_mean"] for record in state_rows]
                panel.plot(x, y, color="#176b87", lw=.65, alpha=.24)
                panel.scatter(x, y, color="#176b87", s=7, alpha=.32, linewidths=0)
            panel.set_title(f'{item["label"]}\n{len(grouped)} states', fontsize=8)
            panel.grid(axis="y", alpha=.16)
            panel.spines[["top", "right"]].set_visible(False)
            panel.tick_params(labelsize=7)
        panels[row_idx, 0].set_ylabel(BIN_AXIS_LABELS[axis] + "\n" + label, fontsize=8)
    fig.suptitle(f"{model.upper()} · {horizon} · {label} by distance bin", fontsize=15)
    fig.supxlabel("Target mean over eligible repeats")
    fig.tight_layout()
    fig.savefig(destination, dpi=135)
    plt.close(fig)


def plot_binned_statistic(score_index, model, horizon, metrics, bins, statistic, destination):
    fig, panels = plt.subplots(3, 7, figsize=(23, 11), squeeze=False)
    metric_names = list(metrics)
    for row_idx, axis in enumerate(BIN_AXES):
        bin_group = sorted(
            [item for item in bins if item["axis"] == axis],
            key=lambda item: item["lower_m"],
        )
        for col_idx, panel in enumerate(panels[row_idx]):
            if col_idx >= len(bin_group):
                panel.set_visible(False)
                continue
            item = bin_group[col_idx]
            for metric_idx, metric in enumerate(metric_names):
                values = [
                    record[statistic] for record in score_index[(horizon, metric, item["bin_id"])]
                    if math.isfinite(record[statistic])
                ]
                if not values:
                    continue
                jitter = np.linspace(-.18, .18, len(values)) if len(values) > 1 else np.zeros(1)
                color = plt.get_cmap("tab20")(metric_idx % 20)
                panel.scatter(metric_idx + jitter, values, color=color, s=8, alpha=.40, linewidths=0)
                panel.plot([metric_idx - .22, metric_idx + .22], [np.mean(values)] * 2,
                           color="#222222", lw=1)
            panel.set_title(f'{item["label"]} · {item["state_count"]} states', fontsize=8)
            tick_labels = []
            for name in metric_names:
                count = sum(
                    math.isfinite(row[statistic])
                    for row in score_index[(horizon, name, item["bin_id"])]
                )
                tick_labels.append(f"{metrics[name]}\n{count}")
            panel.set_xticks(range(len(metric_names)), tick_labels, rotation=90, fontsize=5.5)
            panel.set_xlim(-.55, len(metric_names) - .45)
            panel.grid(axis="y", alpha=.16)
            panel.spines[["top", "right"]].set_visible(False)
            panel.tick_params(axis="y", labelsize=7)
        panels[row_idx, 0].set_ylabel(BIN_AXIS_LABELS[axis], fontsize=8)
    if statistic == "pairwise_ranking_accuracy":
        low, high, baseline = 0, 1, .5
    else:
        low, high, baseline = -1, 1, 0
    for panel in panels.flat:
        if panel.get_visible():
            panel.set_ylim(low - .04, high + .04)
            panel.axhline(baseline, color="#777777", linestyle="--", lw=.7)
    fig.suptitle(
        f"{model.upper()} · {horizon} · {STATISTIC_LABELS[statistic]} by bin\n"
        "Each dot is a state-level result. Dark bars show the bin mean.",
        fontsize=14,
    )
    fig.tight_layout()
    fig.savefig(destination, dpi=140)
    plt.close(fig)


def analyze_model(model_dir):
    model_dir = model_directory(model_dir)
    selection, metrics, action_rows, score_rows, pooled_rows, binned_rows = read_model(model_dir)
    analysis_dir = records_directory(model_dir) / "analysis" / "round_four"
    write_csv(analysis_dir / "state_action_horizon_means.csv", action_rows)
    write_csv(analysis_dir / "state_horizon_scores.csv", score_rows)
    write_csv(analysis_dir / "pooled_summary.csv", pooled_rows)
    write_csv(analysis_dir / "binned_summary.csv", binned_rows)

    plot_root = model_dir / "plots" / "round_four"
    bins = make_bins(selection)
    action_index = index_rows(action_rows)
    score_index = index_rows(score_rows)
    pooled_lines = [
        f"# {model_dir.name.upper()} round-four pooled results", "",
        "Raw plots connect action means for one initial state. Each point averages target and outcome over the same repeats for that action. State-level scores average equally across actions available at the horizon, then score averages weight each state equally. Exact horizons include only rollouts with a stored snapshot at that step. Full outcomes use every rollout through its endpoint.", "",
        "| Horizon | Raw outcomes | State-level statistics |", "|---|---|---|",
    ]
    bin_lines = [
        f"# {model_dir.name.upper()} round-four results by initial-state bin", "",
        "Raw plots show action means connected within each initial state. Statistical plots show each state's result for the four metrics in a bin, with a dark line for the bin mean. The number of selected starting states is in each bin title. Exact-horizon plots condition on reaching that horizon.", "",
        "| Horizon | Raw values | Pearson | Pairwise accuracy | Spearman | Kendall tau-b |", "|---|---|---|---|---|---|",
    ]
    for horizon in HORIZONS:
        horizon_dir = plot_root / "pooled" / f"horizon_{horizon}"
        horizon_dir.mkdir(parents=True, exist_ok=True)
        plot_raw_pooled(action_index, model_dir.name, horizon, metrics, horizon_dir / "raw_action_means.png")
        plot_pooled_statistics(score_index, model_dir.name, horizon, metrics, horizon_dir / "state_statistics.png")
        (horizon_dir / "README.md").write_text(
            f"# {model_dir.name.upper()} · {horizon} pooled plots\n\n"
            "[Raw action means](raw_action_means.png) · [State-level statistics](state_statistics.png)\n\n"
            "Each point averages a target and outcome over the same eligible repeats for that action. "
            "The statistical chart shows one value per initial state, with orange bars for equal-state means.\n"
        )
        pooled_lines.append(
            f"| {horizon} | [Raw outcomes](horizon_{horizon}/raw_action_means.png) | "
            f"[Statistics](horizon_{horizon}/state_statistics.png) |"
        )

        binned_dir = plot_root / "by_bin" / f"horizon_{horizon}"
        binned_dir.mkdir(parents=True, exist_ok=True)
        for metric, label in metrics.items():
            plot_binned_raw(
                action_index, model_dir.name, horizon, metric, label, bins,
                binned_dir / f"raw_{metric.replace('/', '_')}.png",
            )
        stat_links = {}
        for statistic in STATISTICS:
            stat_path = binned_dir / f"{statistic}.png"
            plot_binned_statistic(score_index, model_dir.name, horizon, metrics, bins, statistic, stat_path)
            stat_links[statistic] = f"[{statistic.replace('_', ' ')}](horizon_{horizon}/{statistic}.png)"
        bin_lines.append(
            f"| {horizon} | [All metric plots](horizon_{horizon}/README.md) | "
            + " | ".join(stat_links[statistic] for statistic in STATISTICS) + " |"
        )
        metric_readme = [
            f"# {model_dir.name.upper()} · {horizon} · raw outcomes by distance bin", "",
            "| Outcome | Plot |", "|---|---|",
        ]
        for metric, label in metrics.items():
            metric_readme.append(f'| {label} | [By-bin plot](raw_{metric.replace("/", "_")}.png) |')
        (binned_dir / "README.md").write_text("\n".join(metric_readme) + "\n")

    notes = [
        "",
        "Pearson r, Spearman rho and Kendall tau-b retain the recorded metric direction. A negative coefficient for total infractions means higher targets accompany fewer infractions. Pairwise accuracy is desirability-oriented, so it reverses infractions and values above 50% mean more desirable action ordering. Kendall tau-b accounts for ties in both target and outcome. Equal-outcome pairs are excluded from pairwise accuracy; equal target values receive half credit.",
        "The goal-distance row uses the next unfulfilled goal in route order for bin assignment.",
        "",
        "The exact eligible repeat counts, action counts, state counts, and pair counts are available in the CSV summaries under records/analysis/round_four.",
    ]
    (plot_root / "README.md").write_text(
        "# Round-four plots\n\n[Pooled plots](pooled/README.md) · [Plots by bin](by_bin/README.md)\n"
    )
    pooled_root = plot_root / "pooled"
    pooled_root.mkdir(exist_ok=True)
    (pooled_root / "README.md").write_text("\n".join(pooled_lines + notes) + "\n")
    binned_root = plot_root / "by_bin"
    binned_root.mkdir(exist_ok=True)
    (binned_root / "README.md").write_text("\n".join(bin_lines + notes) + "\n")
    branch_count = sum(
        len(json.loads(path.read_text())["branches"])
        for path in (records_directory(model_dir) / "probes").glob("*.json")
    )
    return {
        "model": model_dir.name,
        "state_count": len(selection["states"]),
        "branch_count": branch_count,
        "action_mean_rows": len(action_rows),
        "score_rows": len(score_rows),
        "pooled_summary": pooled_rows,
        "binned_summary": binned_rows,
    }


def write_comparison(run_dir, results):
    analysis_dir = run_dir / "records" / "analysis" / "round_four"
    pooled = [
        {"model": model, **entry}
        for model, result in results.items()
        for entry in result["pooled_summary"]
    ]
    binned = [
        {"model": model, **entry}
        for model, result in results.items()
        for entry in result["binned_summary"]
    ]
    write_csv(analysis_dir / "three_checkpoint_comparison_pooled.csv", pooled)
    write_csv(analysis_dir / "three_checkpoint_comparison_binned.csv", binned)
    comparison = [
        "# Round-four checkpoint comparison", "",
        "Each point summarizes one action after averaging the target and driving outcome over the same repeat indices for that action that reached the plotted horizon. Different actions and states may therefore have different repeat counts. The full outcome includes every rollout through its endpoint. Exact-horizon comparisons include only rollouts that reached the requested step, so they are conditional on surviving to that step. The accompanying CSV includes state, action, repeat, and pair coverage for every model and bin.", "",
        "| Horizon | Outcome | Checkpoint | Raw mean | Pearson r | Pairwise accuracy | Spearman rho | Kendall tau-b | Selected states | Pearson states | Accuracy states | Spearman states | Kendall states | Comparable / possible pairs | Eligible repeats |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    common_metrics = list(metric_labels("pql"))
    for horizon in HORIZONS:
        for metric in common_metrics:
            for model, result in results.items():
                matching = [row for row in result["pooled_summary"]
                            if row["horizon"] == horizon and row["metric"] == metric]
                if not matching:
                    continue
                row = matching[0]
                numbers = [row["raw_mean_outcome"], row["mean_pearson_r"],
                           row["mean_pairwise_ranking_accuracy"], row["mean_spearman_rho"],
                           row["mean_kendall_tau_b"]]
                formatted = [
                    f"{value:.4f}" if isinstance(value, (float, np.floating)) and math.isfinite(value) else "—"
                    for value in numbers
                ]
                comparison.append(
                    f'| {horizon} | {row["metric_label"]} | {model.upper()} | '
                    + " | ".join(formatted)
                    + f' | {row["selected_state_count"]} | '
                    + " | ".join(
                        f'{row[f"states_scored_{statistic}"]}/{row["selected_state_count"]}'
                        for statistic in STATISTICS
                    ) + " | "
                    + f'{row["comparable_action_pairs"]}/{row["possible_action_pairs"]} | '
                    + f'{row["eligible_repeat_count"]} |'
                )
    comparison.extend([
        "",
        "Correlation coefficients retain raw outcome direction. For total infractions, negative correlation coefficients mean larger targets accompany fewer infractions. Pairwise accuracy is oriented so larger values mean the target ranks more desirable action outcomes higher. State-level correlation and accuracy means weight states equally. Use the per-model CSVs for each statistic's exact scored-state coverage.",
        "",
        "[Pooled CSV](records/analysis/round_four/three_checkpoint_comparison_pooled.csv) · [Binned CSV](records/analysis/round_four/three_checkpoint_comparison_binned.csv)",
        "",
        "[PQL plots](pql/plots/round_four/README.md) · [TES plots](tes/plots/round_four/README.md) · [PPO plots](ppo/plots/round_four/README.md)",
    ])
    (run_dir / "ROUND_FOUR_COMPARISON.md").write_text("\n".join(comparison) + "\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args(argv)
    run_dir = args.run_dir.expanduser().resolve()
    results = {
        name: analyze_model(run_dir / name)
        for name in ("pql", "tes", "ppo")
    }
    write_comparison(run_dir, results)
    print(json.dumps({
        name: {
            "state_count": result["state_count"],
            "branch_count": result["branch_count"],
            "action_mean_rows": result["action_mean_rows"],
            "score_rows": result["score_rows"],
        }
        for name, result in results.items()
    }, indent=2))


if __name__ == "__main__":
    main()
