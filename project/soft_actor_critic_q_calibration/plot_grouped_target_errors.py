"""Plot min-Q residuals grouped by actor probability rank across starting situations."""

import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def plot_grouped_target_errors(run_dir, raw_q=False):
    run_dir = Path(run_dir)
    output = run_dir / ("raw_q_plots" if raw_q else "target_error_plots")
    output.mkdir(exist_ok=True)
    probes = []
    ranks = []
    for path in sorted((run_dir / "probes").glob("*.json")):
        probe = json.loads(path.read_text())
        if probe.get("status") != "complete":
            continue
        groups = {}
        for branch in probe["branches"]:
            target = branch["actual_critic_target"]
            branch["plot_value"] = branch["q_min"] if raw_q else (
                branch["q_min"] - target if target is not None else None)
            groups.setdefault(branch["action_idx"], []).append(branch)
        ordered = sorted(groups, key=lambda action: (
            -float(np.mean([record["initial_log_prob"] for record in groups[action]])), action))
        for rank, action in enumerate(ordered, start=1):
            records = groups[action]
            ranks.append({"probe_id": path.stem, "rank": rank, "action_idx": action,
                          "actor_probability": float(np.exp(records[0]["initial_log_prob"]))})
        probes.append((path.stem, groups, ordered))
    if not probes:
        raise ValueError("No completed probes")
    errors = [record["plot_value"]
              for _, groups, _ in probes for records in groups.values() for record in records
              if record["plot_value"] is not None
              and np.isfinite(record["plot_value"])]
    if not errors:
        raise ValueError("No finite min-Q residuals")
    if raw_q:
        errors.extend(record["actual_critic_target"] for _, groups, _ in probes
                      for records in groups.values() for record in records
                      if record["actual_critic_target"] is not None
                      and np.isfinite(record["actual_critic_target"]))
    margin = max(float(np.ptp(errors)) * 0.06, 0.01)
    limits = (min(0, min(errors)) - margin, max(0, max(errors)) + margin)
    probabilities = [
        np.exp(record["initial_log_prob"])
        for _, groups, _ in probes for records in groups.values() for record in records
        if np.isfinite(record["initial_log_prob"])
        and np.exp(record["initial_log_prob"]) > 0
        and record["plot_value"] is not None
        and np.isfinite(record["plot_value"])
    ]
    # Match the pooled min-Q panel's Matplotlib autoscaling, including log-space padding.
    reference_figure, reference_axis = plt.subplots()
    reference_axis.scatter(probabilities, np.zeros(len(probabilities)))
    reference_axis.set_xscale("log")
    log_probability_limits = reference_axis.get_xlim()
    plt.close(reference_figure)
    figure, axes = plt.subplots(12, 2, figsize=(16, 42),
                                squeeze=False, layout="constrained", sharey=True)
    plotted = 0
    excluded = 0
    rank_counts = {}
    for row_idx in range(12):
        rank = row_idx + 1
        rank_counts[rank] = 0
        for column_idx, scale in enumerate(("linear", "log")):
            axis = axes[row_idx, column_idx]
            for probe_id, groups, ordered in probes:
                records = groups[ordered[row_idx]]
                usable = [record for record in records if record["plot_value"] is not None
                          and np.isfinite(record["plot_value"])
                          and np.isfinite(record["initial_log_prob"])
                          and np.exp(record["initial_log_prob"]) > 0]
                if column_idx == 0:
                    plotted += len(usable)
                    rank_counts[rank] += len(usable)
                    excluded += len(records) - len(usable)
                if not usable:
                    continue
                probabilities = [np.exp(record["initial_log_prob"]) for record in usable]
                residuals = [record["plot_value"] for record in usable]
                color = "#2563a6"
                axis.scatter(probabilities, residuals, color=color, alpha=0.65,
                             s=25, edgecolors="none")
                axis.scatter([np.mean(probabilities)], [np.mean(residuals)],
                             color=color, marker="D", s=26, edgecolors="black", linewidths=0.6)
                if raw_q:
                    paired = [record for record in usable if record["actual_critic_target"] is not None
                              and np.isfinite(record["actual_critic_target"])]
                    target_probabilities = [np.exp(record["initial_log_prob"]) for record in paired]
                    targets = [record["actual_critic_target"] for record in paired]
                    axis.vlines(target_probabilities, [record["q_min"] for record in paired], targets,
                                colors="#777777", linestyles="--", linewidths=0.7, alpha=0.4)
                    axis.scatter(target_probabilities, targets, marker="x", color="#c65d14",
                                 s=25, linewidths=0.9, alpha=0.7, zorder=3)
            axis.axhline(0, color="#333333", linestyle="--", linewidth=0.8)
            axis.set_xscale(scale)
            if scale == "linear":
                axis.set_xlim(-0.02, 1.02)
            else:
                axis.set_xlim(*log_probability_limits)
            axis.set_ylim(*limits)
            axis.set_title(f"Action probability rank {rank} | {scale} probability", fontsize=10)
            axis.set_xlabel("Initial actor probability", fontsize=9)
            axis.set_ylabel("min(Q1, Q2) and training target" if raw_q else "min(Q1, Q2) − training target", fontsize=9)
            axis.grid(alpha=0.15)
            axis.tick_params(labelsize=9)
    figure.suptitle(
        (f"{run_dir.name.upper()}: raw min-Q grouped by action probability rank\n" if raw_q else
         f"{run_dir.name.upper()}: min-Q target error grouped by action probability rank\n") +
        f"Each row pools the same rank across {len(probes)} starting situations\n"
        "Rank 1 = most likely; rank 12 = least likely. Dots = rollouts; diamonds = eight-repeat means.",
        fontsize=15)
    figure.supxlabel("Rank is determined separately within each starting situation. Shared Y limits. " +
                      ("Dots/diamonds = Q/mean Q; orange × = targets; dashed lines pair Q and target for the same rollout." if raw_q else
                       "Bootstrapped target disagreement is not independent true-Q error."), fontsize=10)
    figure.savefig(output / ("min_q_raw_by_rank.png" if raw_q else "min_q_target_errors_by_rank.png"), dpi=150)
    plt.close(figure)
    with (output / "action_probability_ranks.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(ranks[0]))
        writer.writeheader()
        writer.writerows(ranks)
    print(json.dumps({"situations": len(probes), "rollouts_per_column": plotted,
                      "excluded": excluded, "rollouts_per_rank": rank_counts,
                      "output": str(output / ("min_q_raw_by_rank.png" if raw_q else "min_q_target_errors_by_rank.png"))}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--raw-q", action="store_true")
    args = parser.parse_args()
    plot_grouped_target_errors(args.run_dir, raw_q=args.raw_q)
