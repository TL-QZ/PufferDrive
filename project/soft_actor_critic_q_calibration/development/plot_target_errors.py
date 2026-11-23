"""Plot individual branch critic-minus-target residuals against actor probability."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
import numpy as np

if __package__:
    from .plot_paths import plot_output_dir, model_directory, records_directory
else:
    from plot_paths import plot_output_dir, model_directory, records_directory


def plot_target_errors(run_dir, raw_q=False):
    run_dir = model_directory(run_dir)
    output = plot_output_dir(run_dir, "raw_q_plots" if raw_q else "target_error_plots")
    analysis_output = plot_output_dir(run_dir, "raw_q_plots" if raw_q else "target_error_plots", artifact_type="analysis")
    rows = []
    for path in sorted((records_directory(run_dir) / "probes").glob("*.json")):
        probe = json.loads(path.read_text())
        if probe.get("status") != "complete":
            continue
        for branch in probe["branches"]:
            row = {
                "probe_id": path.stem, "action_idx": branch["action_idx"],
                "repeat_idx": branch["repeat_idx"],
                "actor_probability": float(np.exp(branch["initial_log_prob"])),
                "actual_critic_target": branch["actual_critic_target"],
            }
            for critic in ("q1", "q2", "q_min"):
                row[critic] = branch[critic]
                target = branch["actual_critic_target"]
                row[f"{critic}_minus_target"] = (
                    branch[critic] - target if target is not None else None
                )
            rows.append(row)
    if not rows:
        raise ValueError("No completed branch records found")
    for probe_id in sorted({row["probe_id"] for row in rows}):
        probe_rows = [row for row in rows if row["probe_id"] == probe_id]
        probabilities_by_action = {}
        for row in probe_rows:
            probabilities_by_action.setdefault(row["action_idx"], []).append(row["actor_probability"])
        ordered_actions = sorted(probabilities_by_action, key=lambda action: (
            -float(np.mean(probabilities_by_action[action])), action))
        rank_by_action = {action: rank for rank, action in enumerate(ordered_actions, start=1)}
        for row in probe_rows:
            row["actor_probability_rank"] = rank_by_action[row["action_idx"]]
    with (analysis_output / "individual_target_errors.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    all_errors = [row[critic if raw_q else f"{critic}_minus_target"] for row in rows
                  for critic in ("q1", "q2", "q_min")
                  if row[critic if raw_q else f"{critic}_minus_target"] is not None]
    if raw_q:
        all_errors.extend(row["actual_critic_target"] for row in rows
                          if row["actual_critic_target"] is not None)
    finite_errors = np.asarray(all_errors, dtype=float)
    finite_errors = finite_errors[np.isfinite(finite_errors)]
    if not len(finite_errors):
        raise ValueError("No finite critic-minus-target errors")
    margin = max(float(np.ptp(finite_errors)) * 0.05, 0.01)
    limits = (min(0, finite_errors.min()) - margin, max(0, finite_errors.max()) + margin)
    counts = {}
    fig, axes = plt.subplots(3, 2, figsize=(14, 13), layout="constrained", sharey=True)
    color_map = plt.get_cmap("viridis", 12)
    normalization = BoundaryNorm(np.arange(0.5, 13.5), 12)
    for row_idx, (critic, label) in enumerate((("q1", "Q1"), ("q2", "Q2"), ("q_min", "min(Q1, Q2)"))):
        selected = [row for row in rows if row[critic if raw_q else f"{critic}_minus_target"] is not None
                    and np.isfinite(row[critic if raw_q else f"{critic}_minus_target"])
                    and np.isfinite(row["actor_probability"]) and row["actor_probability"] > 0]
        counts[critic] = {"plotted": len(selected), "excluded": len(rows) - len(selected)}
        for column_idx, scale in enumerate(("linear", "log")):
            axis = axes[row_idx, column_idx]
            continuation_groups = {}
            for row in selected:
                key = (row["probe_id"], row["repeat_idx"])
                continuation_groups.setdefault(key, []).append(row)
            for records in continuation_groups.values():
                ordered = sorted(records, key=lambda row: (row["actor_probability"], row["action_idx"]))
                axis.plot([row["actor_probability"] for row in ordered],
                          [row[critic if raw_q else f"{critic}_minus_target"] for row in ordered],
                          linestyle=":", linewidth=0.65, alpha=0.22,
                          color="#2563a6", zorder=1)
            axis.scatter([row["actor_probability"] for row in selected],
                         [row[critic if raw_q else f"{critic}_minus_target"] for row in selected],
                         s=19, alpha=0.65,
                         c=[row["actor_probability_rank"] for row in selected],
                         cmap=color_map, norm=normalization, edgecolors="none", zorder=2)
            if raw_q:
                paired = [row for row in selected if row["actual_critic_target"] is not None
                          and np.isfinite(row["actual_critic_target"])]
                axis.vlines([row["actor_probability"] for row in paired],
                            [row[critic] for row in paired],
                            [row["actual_critic_target"] for row in paired],
                            colors="#777777", linestyles="--", linewidths=0.7, alpha=0.35)
                axis.scatter([row["actor_probability"] for row in paired],
                             [row["actual_critic_target"] for row in paired],
                             marker="x", s=23, linewidths=0.8, alpha=0.7,
                             c=[row["actor_probability_rank"] for row in paired],
                             cmap=color_map, norm=normalization, zorder=3)
            axis.axhline(0, color="#333333", linewidth=1, linestyle="--")
            axis.set_xscale(scale)
            if scale == "linear":
                axis.set_xlim(-0.02, 1.02)
            axis.set_ylim(*limits)
            axis.set_xlabel(f"Initial actor probability of forced action ({scale} scale)")
            axis.set_ylabel(f"{label} and training target" if raw_q else f"{label} − implemented training target")
            axis.set_title(f"{run_dir.name.upper()}: {label} raw output" if raw_q else f"{run_dir.name.upper()}: {label} target residual per rollout")
            axis.grid(alpha=0.18)
    fig.suptitle(f"{run_dir.name.upper()}: raw critic outputs" if raw_q else f"{run_dir.name.upper()}: critic errors against training targets", fontsize=17)
    fig.colorbar(plt.cm.ScalarMappable(norm=normalization, cmap=color_map),
                 ax=axes.ravel().tolist(), location="top", shrink=0.75, aspect=55,
                 fraction=0.025, pad=0.015, ticks=range(1, 13),
                 label="Actor-probability rank within each initial condition (1 = most likely)")
    fig.supxlabel("Dots = Q; × = rollout target; dashed vertical lines pair the same action and rollout.\n"
                  "Dotted lines connect Q across actions within a situation and repeat. Repeated Q points overlap." if raw_q else "Dots = rollouts; dotted lines connect actions within the same starting situation and repeat.\n"
                  "Lines follow increasing actor probability. Positive = Q exceeds target.\n"
                  "Target is bootstrapped, not independent ground truth.", fontsize=10)
    fig.savefig(output / ("critic_raw_q.png" if raw_q else "critic_target_errors.png"), dpi=180)
    plt.close(fig)
    if not raw_q:
        target_figure, target_axes = plt.subplots(1, 3, figsize=(17, 6),
                                                layout="constrained", sharex=True, sharey=True)
        target_counts = {}
        for axis, (critic, label) in zip(target_axes, (("q1", "Q1"), ("q2", "Q2"), ("q_min", "min(Q1, Q2)"))):
            selected = [row for row in rows if row["actual_critic_target"] is not None
                        and np.isfinite(row["actual_critic_target"])
                        and row[f"{critic}_minus_target"] is not None
                        and np.isfinite(row[f"{critic}_minus_target"])]
            target_counts[critic] = {"plotted": len(selected), "excluded": len(rows) - len(selected)}
            axis.scatter([row["actual_critic_target"] for row in selected],
                         [row[f"{critic}_minus_target"] for row in selected],
                         c=[row["actor_probability_rank"] for row in selected],
                         cmap=color_map, norm=normalization, s=22, alpha=0.65, edgecolors="none")
            axis.axhline(0, color="#333333", linestyle="--", linewidth=1)
            axis.set_ylim(*limits)
            axis.set_xlabel("Implemented training target")
            axis.set_ylabel(f"{label} − training target")
            axis.set_title(f"{label} | {len(selected)} rollout dots")
            axis.grid(alpha=0.18)
        target_figure.suptitle(f"{run_dir.name.upper()}: critic error versus training target", fontsize=17)
        target_figure.colorbar(plt.cm.ScalarMappable(norm=normalization, cmap=color_map),
                              ax=target_axes.tolist(), location="top", shrink=0.65, aspect=55,
                              ticks=range(1, 13), label="Actor-probability rank (1 = most likely)")
        target_figure.supxlabel(
            "Individual rollouts; no averaging. Positive = Q exceeds target.\n"
            "Target also enters Y with a minus sign; a negative trend alone does not establish critic bias.",
            fontsize=10)
        target_figure.savefig(output / "critic_errors_vs_target.png", dpi=180)
        plt.close(target_figure)
        (analysis_output / "target_axis_plot_counts.json").write_text(json.dumps(target_counts, indent=2) + "\n")
    (analysis_output / "plot_counts.json").write_text(json.dumps(counts, indent=2) + "\n")
    (output / "README.md").write_text(
        ("# Raw critic outputs\n\nEach dot uses the initial online critic output; no target is subtracted. "
         "Repeated rollouts share initial Q and probability, so dots overlap. "
         "Color indicates within-situation actor probability rank. Dotted lines connect actions "
         "within each situation and continuation repeat. Targets are overlaid as × markers; "
         "dashed vertical lines connect Q to the target from the same action and rollout.\n" if raw_q else
        "# Individual rollout critic errors against training targets\n\n"
        "Each dot is one forced-action continuation, not an eight-repeat mean. "
        "X is exp(initial_log_prob); Y is Q minus actual_critic_target. "
        "The target uses the checkpoint's implemented SAC/PQL formula.\n\n"
        "`critic_target_errors.png` combines six panels: rows are Q1, Q2, and min-Q; "
        "columns are linear and log probability axes. "
        "Linear plots show probabilities directly; log plots separate very small probabilities. "
        "All plots share Y limits and retain all finite errors. Counts/exclusions are in the matching records/analysis directory: plot_counts.json.\n\n"
        "`critic_errors_vs_target.png` shows individual Q1/Q2/min-Q errors against the training "
        "target on X, with shared axes and probability-rank colors. Counts are in "
        "the matching records/analysis directory: target_axis_plot_counts.json. Because error = Q - target, both axes contain the target; "
        "a negative trend alone is not evidence of bias. For a fixed situation/action with "
        "constant Q, changing the sampled target produces a line of slope -1 exactly.\n\n"
        "The eight continuations of a situation/action share initial probability and Q, "
        "but can have different sampled targets, producing vertical groups or overlapping dots. "
        "Dotted lines connect actions in increasing probability order within the same starting "
        "situation and continuation repeat; they are visual guides, not temporal trajectories or fits. "
        "Dot color is action-probability rank within its starting situation: 1 is most likely, "
        "12 least likely; exact ties are ordered by action ID. "
        "A residual includes target variability; it alone does not establish true Q estimation bias. "
        "Current action probability does not establish historical replay coverage.\n")
    )
    print(json.dumps({"output": str(output), "counts": counts}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--raw-q", action="store_true")
    args = parser.parse_args()
    plot_target_errors(args.run_dir, raw_q=args.raw_q)
