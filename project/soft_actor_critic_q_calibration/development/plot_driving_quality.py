"""Compare implemented critic targets with individual rollout outcomes."""

import argparse
from collections import Counter
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

if __package__:
    from .plot_paths import plot_output_dir, model_directory, records_directory
else:
    from plot_paths import plot_output_dir, model_directory, records_directory


ENDING_STYLES = {
    "agent_fault": ("Agent terminal + final-step fault/violation", "#d73027", "^"),
    "agent_other": ("Other agent terminal (fault not established)", "#984ea3", "s"),
    "scene_cutoff": ("Scene cutoff", "#2166ac", "o"),
    "validity_gap": ("Validity gap", "#e08214", "x"),
    "other_cutoff": ("Other cutoff", "#666666", "D"),
}


def plot_driving_quality(run_dir, center_target=False, x_limits=None):
    run_dir = model_directory(run_dir)
    if x_limits is not None and (not np.all(np.isfinite(x_limits)) or x_limits[0] >= x_limits[1]):
        raise ValueError("X limits must be finite and increasing")
    output = plot_output_dir(run_dir, "driving_quality_plots")
    analysis_output = plot_output_dir(run_dir, "driving_quality_plots", artifact_type="analysis")
    records = []
    for path in sorted((records_directory(run_dir) / "probes").glob("*.json")):
        probe = json.loads(path.read_text())
        if probe.get("status") != "complete":
            continue
        with np.load(path.with_suffix(".npz")) as arrays:
            for branch_idx, branch in enumerate(probe["branches"]):
                if str(arrays["task_id"][branch_idx]) != branch["task_id"]:
                    raise ValueError(f"Branch ordering mismatch: {path}")
                final_idx = int(branch["observed_horizon"]) - 1
                if final_idx < 0 or not arrays["valid"][final_idx, branch_idx]:
                    raise ValueError(f"Invalid final step: {path}")
                final_fault = any(bool(arrays[name][final_idx, branch_idx]) for name in (
                    "event_at_fault_collision", "event_offroad", "event_red_light"))
                if branch["terminal"]:
                    ending = "agent_fault" if final_fault else "agent_other"
                elif branch["scene_boundary"]:
                    ending = "scene_cutoff"
                elif branch["validity_gap"]:
                    ending = "validity_gap"
                else:
                    ending = "other_cutoff"
                records.append({**branch, "probe_id": path.stem, "ending_type": ending})
    if not records:
        raise ValueError("No completed rollouts")
    target_medians = {}
    for probe_id in sorted({record["probe_id"] for record in records}):
        targets = [record["actual_critic_target"] for record in records
                   if record["probe_id"] == probe_id
                   and record["actual_critic_target"] is not None
                   and np.isfinite(record["actual_critic_target"])]
        if not targets:
            raise ValueError(f"No finite targets for {probe_id}")
        target_medians[probe_id] = {"median": float(np.median(targets)), "count": len(targets)}
    for record in records:
        target = record["actual_critic_target"]
        record["plot_target"] = (target - target_medians[record["probe_id"]]["median"]
                                 if center_target and target is not None else target)
    x_label = "Training target − median for same initial state" if center_target else "Implemented training target"
    metrics = (
        ("collision", "Any collision", True),
        ("at_fault_collision", "Any at-fault collision", True),
        ("offroad", "Any off-road event", True),
        ("red_light", "Any red-light violation", True),
        ("goal_reached", "Reached at least one goal", True),
        ("observed_horizon", "Observed horizon (steps)", False),
        ("discounted_reward_return", "Discounted reward return", False),
    )
    figure, axes = plt.subplots(4, 2, figsize=(15, 17), layout="constrained", sharex=True)
    counts = {}
    for axis, (metric, label, binary) in zip(axes.flat, metrics):
        selected = [record for record in records if record["actual_critic_target"] is not None
                    and np.isfinite(record["actual_critic_target"])
                    and record.get(metric) is not None and np.isfinite(record[metric])]
        counts[metric] = {"plotted": len(selected), "excluded": len(records) - len(selected)}
        visible = selected if x_limits is None else [record for record in selected
                  if x_limits[0] <= record["plot_target"] <= x_limits[1]]
        counts[metric]["visible"] = len(visible)
        for ending, (_, color, marker) in ENDING_STYLES.items():
            subset = [record for record in selected if record["ending_type"] == ending]
            axis.scatter([record["plot_target"] for record in subset],
                         [record[metric] for record in subset], color=color, marker=marker,
                         s=23, alpha=0.42, linewidths=0.5)
        axis.set_title(f"{label} | {len(visible)} visible / {len(selected)} rollouts")
        axis.set_xlabel(x_label)
        if x_limits is not None:
            axis.set_xlim(*x_limits)
        if center_target:
            axis.axvline(0, color="#555555", linestyle="--", linewidth=0.8)
        axis.set_ylabel(label)
        axis.tick_params(labelbottom=True)
        if binary:
            axis.set_ylim(-0.15, 1.15)
            axis.set_yticks([0, 1], ["0 — no", "1 — yes"])
            counts[metric]["positive"] = sum(bool(record[metric]) for record in selected)
            axis.text(0.02, 0.92, f"Visible yes: {sum(bool(record[metric]) for record in visible)} / {len(visible)}",
                      transform=axis.transAxes, fontsize=10)
        axis.grid(alpha=0.18)
    legend_axis = axes.flat[-1]
    legend_axis.axis("off")
    ending_counts = Counter(record["ending_type"] for record in records)
    handles = [Line2D([], [], marker=marker, color=color, linestyle="none",
                      label=f"{label}: {ending_counts[ending]}")
               for ending, (label, color, marker) in ENDING_STYLES.items()]
    legend_axis.legend(handles=handles, loc="upper left", frameon=False, fontsize=11)
    legend_axis.text(0.02, 0.50,
                     "Ending types use the final measured step.\n"
                     "Fault/violation = at-fault collision, off-road, or red-light.\n"
                     "A collision alone does not establish fault.\n"
                     "Agent terminal takes precedence if the scene also ends.\n\n"
                     "Each dot is one rollout; binary dots overlap (no jitter).\n"
                     "Safety and goal flags cover the entire measured rollout.\n"
                     "Cutoff returns are partial; durations differ.\n"
                     "These are training targets, not online Q outputs.",
                     transform=legend_axis.transAxes, va="top", fontsize=10, linespacing=1.5)
    title = "median-centered target" if center_target else "training target"
    figure.suptitle(f"{run_dir.name.upper()}: {title} versus driving outcomes", fontsize=17)
    if center_target:
        figure.supxlabel("Each state's median uses all 12 actions × 8 repeats. X = 0 is that state's median target.\n"
                          "Individual rollouts retained; outcomes on Y are unchanged.", fontsize=11)
    destination = output / ("centered_target_vs_driving_quality.png" if center_target else "target_vs_driving_quality.png")
    if x_limits is not None:
        destination = destination.with_stem(destination.stem + "_zoom")
    figure.savefig(destination, dpi=170)
    plt.close(figure)
    summary = {"metrics": counts, "ending_types": dict(ending_counts), "center_target": center_target,
               "x_limits": x_limits}
    if center_target:
        summary["target_medians_by_initial_state"] = target_medians
    counts_name = "centered_plot_counts" if center_target else "plot_counts"
    suffix = "_zoom" if x_limits is not None else ""
    (analysis_output / f"{counts_name}{suffix}.json").write_text(json.dumps(summary, indent=2) + "\n")
    (analysis_output / (("CENTERED_README" if center_target else "README") + suffix + ".md")).write_text(
        (f"X-axis display restricted to {x_limits}; data are unchanged. "
         "Panel counts refer to visible points; ending-type legend counts cover the full dataset.\n\n"
         if x_limits is not None else "") +
        ("# Median-centered training target\n\n"
         "X = each rollout's target minus the median target across all actions and repeats "
         "sharing its initial state (reference episode, frame, and probing agent). "
         "This removes state-specific target offsets without changing within-state ordering "
         "or gaps. No scaling or averaging of dots is performed. All Y values are unchanged. "
         "Centering does not remove continuation randomness or make cutoffs complete. "
         "Per-state medians and sample counts are saved in centered_plot_counts.json.\n\n"
         if center_target else "") +
        "# Training target versus driving outcomes\n\n"
        "Seven panels show individual rollout flags, horizon, and observed discounted reward return. "
        "The underlying X measurement is actual_critic_target. goal_reached means at least one goal, not goal count.\n\n"
        "Ending classifications check the final valid step in each probe NPZ. An agent terminal "
        "with at-fault collision, off-road, or red-light recorded on that step is labeled "
        "agent terminal plus fault/violation. Other terminals do not establish fault. "
        "Scene cutoff denotes a nonterminal probe ending at the scene boundary; time-limit "
        "and early-scene reset are not separated. Terminal takes precedence over a simultaneous "
        "scene boundary. This is a recorded-event classification, not a counterfactual causal proof.\n\n"
        "Colors indicate ending type on all panels. No jitter or averaging is used. "
        "Binary points overlap; counts accompany each panel. Cutoff reward returns are partial, "
        "and outcomes have differing observation durations. Probes share two reference scenes, "
        "so points are not independent population samples. No correlation fit is inferred.\n"
    )
    (output / "README.md").write_text(
        "# Driving-quality figures\n\n"
        "target_vs_driving_quality.png uses the training target on X. "
        "centered_target_vs_driving_quality.png subtracts each initial state’s median target; "
        "the _zoom version shows X from -0.4 to 0.2. Y values remain unchanged.\n\n"
        "Each dot is one rollout. Colors and markers identify how the rollout ended. "
        "Scene-cutoff returns are incomplete. Counts, per-state medians, and detailed "
        "notes are in the corresponding records/analysis/driving_quality_plots/ folder.\n"
    )
    print(json.dumps({"output": str(destination), **summary}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--center-target", action="store_true",
                        help="Subtract the median target within each initial state")
    parser.add_argument("--x-limits", type=float, nargs=2, metavar=("MIN", "MAX"))
    args = parser.parse_args()
    plot_driving_quality(args.run_dir, center_target=args.center_target, x_limits=args.x_limits)
