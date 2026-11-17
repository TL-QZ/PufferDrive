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


ENDING_STYLES = {
    "agent_fault": ("Agent terminal + final-step fault/violation", "#d73027", "^"),
    "agent_other": ("Other agent terminal (fault not established)", "#984ea3", "s"),
    "scene_cutoff": ("Scene cutoff", "#2166ac", "o"),
    "validity_gap": ("Validity gap", "#e08214", "x"),
    "other_cutoff": ("Other cutoff", "#666666", "D"),
}


def plot_driving_quality(run_dir):
    run_dir = Path(run_dir)
    output = run_dir / "driving_quality_plots"
    output.mkdir(exist_ok=True)
    records = []
    for path in sorted((run_dir / "probes").glob("*.json")):
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
        for ending, (_, color, marker) in ENDING_STYLES.items():
            subset = [record for record in selected if record["ending_type"] == ending]
            axis.scatter([record["actual_critic_target"] for record in subset],
                         [record[metric] for record in subset], color=color, marker=marker,
                         s=23, alpha=0.42, linewidths=0.5)
        axis.set_title(f"{label} | {len(selected)} rollouts")
        axis.set_xlabel("Implemented training target")
        axis.set_ylabel(label)
        axis.tick_params(labelbottom=True)
        if binary:
            axis.set_ylim(-0.15, 1.15)
            axis.set_yticks([0, 1], ["0 — no", "1 — yes"])
            counts[metric]["positive"] = sum(bool(record[metric]) for record in selected)
            axis.text(0.02, 0.92, f"Yes: {counts[metric]['positive']} / {len(selected)}",
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
    figure.suptitle(f"{run_dir.name.upper()}: training target versus driving outcomes", fontsize=17)
    destination = output / "target_vs_driving_quality.png"
    figure.savefig(destination, dpi=170)
    plt.close(figure)
    summary = {"metrics": counts, "ending_types": dict(ending_counts)}
    (output / "plot_counts.json").write_text(json.dumps(summary, indent=2) + "\n")
    (output / "README.md").write_text(
        "# Training target versus driving outcomes\n\n"
        "Seven panels show individual rollout flags, horizon, and observed discounted reward return. "
        "X is actual_critic_target. goal_reached means at least one goal, not goal count.\n\n"
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
    print(json.dumps({"output": str(destination), **summary}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    plot_driving_quality(parser.parse_args().run_dir)
