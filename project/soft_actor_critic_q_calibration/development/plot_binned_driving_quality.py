"""Plot each forced action's training target against its recorded driving outcomes."""
import argparse
import json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np

from .plot_paths import plot_output_dir
from .run_layout import model_directory

ROWS = {
    "nearest_observed_partner_center_distance_m": "Nearest agent",
    "nearest_observed_road_boundary_distance_m": "Road boundary",
    "nearest_visible_goal_distance_m": "Visible goal",
}
METRICS = {
    "episode_return": "Episode return",
    "standard_soft_return": "Sampled soft return",
    "puffer_score": "Puffer score",
    "num_goals_reached": "Number of goals reached",
    "total_distance_travelled": "Total distance traveled (m)",
    "total_infractions": "Total infractions (lower is better)",
    "avg_distance_per_infraction": "Average distance per infraction (m)",
    "avg_speed_per_agent": "Average speed (m/s)",
    "reward_components/collision": "Collision reward component",
    "reward_components/offroad": "Off-road reward component",
}

def outcome_metrics(model_name, *, include_common_return=False):
    metrics = dict(METRICS)
    if model_name == "ppo":
        metrics = {}
        for key, label in METRICS.items():
            if key == "standard_soft_return":
                metrics["discounted_reward_return"] = "Discounted reward return"
            else:
                metrics[key] = label
    elif include_common_return:
        metrics["discounted_reward_return"] = "Discounted reward return"
    return metrics


def branch_outcome(branch, key):
    if key in ("standard_soft_return", "discounted_reward_return"):
        return branch[key]
    return branch["metrics_mean"][key]


def state_scores(target, outcome, *, lower_is_better=False):
    """Return Pearson r and ordering agreement for one state's action outcomes."""
    if np.ptp(outcome) == 0:
        return np.nan, np.nan
    r = np.corrcoef(target, outcome)[0, 1] if np.ptp(target) > 0 else np.nan
    left, right = np.triu_indices(len(target), k=1)
    outcome_order = np.sign(outcome[left] - outcome[right])
    if lower_is_better:
        outcome_order = -outcome_order
    target_order = np.sign(target[left] - target[right])
    comparable = outcome_order != 0
    agreement = (1 + outcome_order[comparable] * target_order[comparable]) / 2
    return r, agreement.mean()


def plot_scores(groups, model_name, output, metric_idx, metric, label, target_label):
    filename_metric = metric.replace("/", "_")
    correlations = {key: np.array([state_scores(v[:, 0], v[:, metric_idx], lower_is_better=(metric == "total_infractions")) for v in states])
                    for key, states in groups.items()}
    for column, (folder, statistic) in enumerate((
        ("pearson_r", "Pearson r"), ("pairwise_ranking_accuracy", "Pairwise ranking accuracy"),
    )):
        fig, panels = plt.subplots(3, 6, figsize=(20, 9), sharey=True)
        for row, (axis, axis_label) in enumerate(ROWS.items()):
            keys = sorted((key for key in groups if key[0] == axis), key=lambda key: key[1])
            for col, panel in enumerate(panels[row]):
                if col >= len(keys):
                    panel.set_visible(False)
                    continue
                key = keys[col]
                _, low, high = key
                distance = f"{low:g}–{high:g} m" if high is not None else f"{low:g}+ m"
                scores = correlations[key][:, column]
                valid = np.isfinite(scores)
                positions = np.arange(1, len(scores) + 1)
                panel.set_title(f"{distance}    ·    {valid.sum()}/{len(scores)} scored", fontsize=10)
                panel.axhline(0 if column == 0 else 50, color="#777777", ls="--", lw=0.9)
                panel.scatter(positions[valid], scores[valid] * (1 if column == 0 else 100),
                              s=23, color="#146882", zorder=3)
                panel.set_xlim(.4, len(scores) + .6)
                panel.set_xticks([1, 5, 10, 15])
                panel.set_yticks([-1, -.5, 0, .5, 1] if column == 0 else [0, 25, 50, 75, 100])
                panel.set_ylim((-1.08, 1.08) if column == 0 else (-4, 104))
                panel.grid(axis="y", alpha=.17)
                panel.spines[["top", "right"]].set_visible(False)
                if col == 0:
                    panel.set_ylabel(axis_label + "\n" + ("Pearson r" if column == 0 else "Accuracy (%)"))
        fig.suptitle(f"{model_name.upper()}  |  {statistic}: {target_label.lower()} vs {label.lower()}", fontsize=19)
        note = ("Dashed line = zero correlation. Constant target or outcome gives no score."
                if column == 0 else
                "Dashed line = chance (50%). Equal outcomes are excluded; equal targets receive half credit.")
        fig.text(.5, .925, f"One dot per starting state, calculated across its 12 actions.  {note}",
                 ha="center", fontsize=12)
        fig.supxlabel("Starting state within bin (selection order)")
        fig.subplots_adjust(left=.065, right=.99, top=.85, bottom=.10, hspace=.43, wspace=.18)
        fig.savefig(output / folder / f"critic_target_vs_{filename_metric}_{folder}.png", dpi=150)
        plt.close(fig)


def plot_pooled(states, name, output, metrics, target_label):
    views = [
        ("raw_values", "Raw rollout outcomes"),
        ("pearson_r", "Pearson r"),
        ("pairwise_ranking_accuracy", "Pairwise ranking accuracy"),
    ]
    scores = np.array([[state_scores(v[:, 0], v[:, i], lower_is_better=(metric == "total_infractions"))
                        for i, metric in enumerate(metrics, start=1)] for v in states])
    for view, (folder, title) in enumerate(views):
        rows = (len(metrics) + 2) // 3
        fig, panels = plt.subplots(rows, 3, figsize=(16, 4.3 * rows), constrained_layout=True)
        for unused in panels.flat[len(metrics):]:
            unused.set_visible(False)
        fig.suptitle(f"{name.upper()}  |  {title}", fontsize=18)
        for i, (panel, label) in enumerate(zip(panels.flat, metrics.values())):
            panel.set_title(label)
            panel.grid(axis="y", alpha=.17)
            panel.spines[["top", "right"]].set_visible(False)
            if view == 0:
                for index, values in enumerate(states):
                    color = plt.get_cmap("tab20")(index % 20)
                    panel.plot(values[:, 0], values[:, i + 1], color=color, lw=.55, alpha=.47)
                    panel.scatter(values[:, 0], values[:, i + 1], color=color, s=8, alpha=.65, linewidths=0)
                panel.set_xlabel(target_label)
                if label in (METRICS["total_infractions"], METRICS["num_goals_reached"]):
                    panel.yaxis.set_major_locator(MaxNLocator(integer=True))
                continue
            values = scores[:, i, view - 1]
            valid = np.isfinite(values)
            scale = 1 if view == 1 else 100
            panel.scatter(np.flatnonzero(valid) + 1, scale * values[valid],
                          color="#146882", s=17, alpha=.78, linewidths=0)
            panel.axhline(0 if view == 1 else 50, color="#777777", ls="--", lw=1)
            mean_text = "undefined"
            if valid.any():
                mean_score = values[valid].mean() * scale
                panel.axhline(mean_score, color="#a44116", lw=1.3)
                mean_text = f"{mean_score:.4f}" if view == 1 else f"{mean_score:.2f}%"
            panel.set_title(f"{label}\nMean {mean_text}  ·  {valid.sum()}/{len(states)} scored")
            panel.set_xlim(0, len(states) + 1)
            panel.set_ylim((-1.06, 1.06) if view == 1 else (-3, 103))
            panel.set_xlabel("Starting state (selection order)")
            if i % 3 == 0:
                panel.set_ylabel("Pearson r" if view == 1 else "Pairwise ranking accuracy (%)")
        destination = output / folder
        destination.mkdir(exist_ok=True)
        fig.savefig(destination / f"critic_target_vs_driving_outcomes_{folder}.png", dpi=160)
        plt.close(fig)


def plot_model(directory, *, by_bin=False):
    model = model_directory(directory)
    records = model / "records"
    ppo = model.name == "ppo"
    target_field = "ppo_value_target" if ppo else "actual_critic_target"
    target_label = "PPO value target (V + GAE)" if ppo else "Critic training target"
    metrics = outcome_metrics(model.name)
    states = []
    groups = defaultdict(list)
    selection = json.loads((records / "selection.json").read_text())
    for state in selection["states"]:
        stem = state["state_id"].replace("/", "__")
        probe = json.loads((records / "probes" / f"{stem}.json").read_text())
        branches = probe["branches"]
        if (probe["state_id"] != state["state_id"] or probe["assigned_bin"] != state["assigned_bin"]
                or sorted(b["action_idx"] for b in branches) != list(range(12))):
            raise ValueError(f"Invalid forced-action group in {stem}")
        values = np.asarray([[b[target_field], *(branch_outcome(b, key) for key in metrics)]
                             for b in branches], dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"Missing target or outcome in {stem}")
        bounds = state["assigned_bin"]
        key = bounds["axis"], bounds["lower_m"], bounds["upper_m"]
        ordered = values[np.argsort(values[:, 0], kind="stable")]
        states.append(ordered)
        groups[key].append(ordered)

    output = plot_output_dir(model, "driving_quality_plots")
    if not by_bin:
        plot_pooled(states, model.name, output, metrics, target_label)
        return
    output = output / "by_bin"
    output.mkdir(exist_ok=True)
    for folder in ("raw_values", "pearson_r", "pairwise_ranking_accuracy"):
        (output / folder).mkdir(exist_ok=True)
    for metric_idx, (metric, label) in enumerate(metrics.items(), start=1):
        fig, panels = plt.subplots(3, 6, figsize=(21, 10), sharex=True, sharey=True)
        for row_idx, (axis, row_label) in enumerate(ROWS.items()):
            keys = sorted((key for key in groups if key[0] == axis), key=lambda key: key[1])
            for col_idx, panel in enumerate(panels[row_idx]):
                if col_idx >= len(keys):
                    panel.set_visible(False)
                    continue
                key = keys[col_idx]
                _, lower, upper = key
                panel.set_title(f"{lower:g}–{upper:g} m" if upper is not None else f"{lower:g}+ m")
                for i, values in enumerate(groups[key]):
                    color = plt.get_cmap("tab20")(i % 20)
                    x, y = values[:, 0], values[:, metric_idx]
                    panel.plot(x, y, color=color, linewidth=0.75, alpha=0.76)
                    panel.scatter(x, y, s=11, color=color, alpha=0.86, linewidths=0)
                panel.grid(True, linewidth=0.5, alpha=0.23)
                panel.set_axisbelow(True)
                panel.spines[["top", "right"]].set_visible(False)
                panel.tick_params(labelbottom=True)
                panel.xaxis.set_major_locator(MaxNLocator(nbins=4))
                if metric in ("num_goals_reached", "total_infractions"):
                    panel.yaxis.set_major_locator(MaxNLocator(nbins=6, integer=True))
                if col_idx == 0:
                    panel.set_ylabel(row_label + "\n" + label)
        fig.suptitle(f"{model.name.upper()}  |  {target_label} vs {label.lower()}", fontsize=20, y=0.98)
        fig.text(.5, .925, "Each dot is one forced-action rollout. Each line connects actions from the same starting state in target order.",
                 ha="center", fontsize=12)
        fig.supxlabel(target_label, y=.025)
        fig.subplots_adjust(left=.065, right=.99, top=.865, bottom=.095, wspace=.12, hspace=.39)
        filename_metric = metric.replace("/", "_")
        fig.savefig(output / "raw_values" / f"critic_target_vs_{filename_metric}.png", dpi=150, facecolor="white")
        plt.close(fig)
        plot_scores(groups, model.name, output, metric_idx, metric, label, target_label)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model_dirs", nargs="+")
    parser.add_argument("--by-bin", action="store_true", help="Regenerate the archived bin panels")
    args = parser.parse_args()
    for directory in args.model_dirs:
        plot_model(directory, by_bin=args.by_bin)
