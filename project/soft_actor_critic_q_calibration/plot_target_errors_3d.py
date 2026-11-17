"""Plot target, actor probability, and critic residual in three dimensions."""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


def plot_target_errors_3d(run_dir):
    run_dir = Path(run_dir)
    output = run_dir / "target_error_plots"
    output.mkdir(exist_ok=True)
    records = []
    for path in sorted((run_dir / "probes").glob("*.json")):
        probe = json.loads(path.read_text())
        if probe.get("status") != "complete":
            continue
        for branch in probe["branches"]:
            target = branch["actual_critic_target"]
            if target is None or not np.isfinite(target) or not np.isfinite(branch["initial_log_prob"]):
                continue
            records.append({**branch, "probe_id": path.stem})
    if not records:
        raise ValueError("No finite targets and action probabilities")
    targets = np.asarray([record["actual_critic_target"] for record in records])
    log_probabilities = np.asarray([record["initial_log_prob"] for record in records])
    probabilities = np.exp(log_probabilities)
    continuation_groups = {}
    for record_idx, record in enumerate(records):
        key = (record["probe_id"], record["repeat_idx"])
        continuation_groups.setdefault(key, []).append(record_idx)
    for indices in continuation_groups.values():
        indices.sort(key=lambda idx: (log_probabilities[idx], records[idx]["action_idx"]))
    residuals = np.asarray([[record[critic] - record["actual_critic_target"] for record in records]
                            for critic in ("q1", "q2", "q_min")])
    if not np.all(np.isfinite(residuals)):
        raise ValueError("Non-finite critic residuals")
    margin = max(float(np.ptp(residuals)) * 0.05, 0.01)
    error_limits = (min(0, residuals.min()) - margin, max(0, residuals.max()) + margin)
    figure = plt.figure(figsize=(19, 12), layout="constrained")
    sign_styles = (("Positive: Q > target", "#d73027", "^"),
                   ("Negative: Q < target", "#2166ac", "v"),
                   ("Exactly zero", "#555555", "o"))
    for scale_idx, scale in enumerate(("linear", "log10")):
        y_values = probabilities if scale == "linear" else log_probabilities / np.log(10)
        for critic_idx, label in enumerate(("Q1", "Q2", "min(Q1, Q2)")):
            axis = figure.add_subplot(2, 3, scale_idx * 3 + critic_idx + 1, projection="3d")
            y_limits = (0, 1) if scale == "linear" else (np.floor(y_values.min()), 0)
            plane_x, plane_y = np.meshgrid([targets.min(), targets.max()], y_limits)
            axis.plot_surface(plane_x, plane_y, np.zeros_like(plane_x),
                              color="#999999", alpha=0.12, shade=False)
            for indices in continuation_groups.values():
                axis.plot(targets[indices], y_values[indices], residuals[critic_idx, indices],
                          linestyle=":", color="#555555", linewidth=0.65, alpha=0.25)
            masks = (residuals[critic_idx] > 0, residuals[critic_idx] < 0, residuals[critic_idx] == 0)
            for mask, (_, color, marker) in zip(masks, sign_styles):
                axis.scatter(targets[mask], y_values[mask], residuals[critic_idx, mask],
                             color=color, marker=marker, s=22, alpha=0.8, depthshade=False,
                             edgecolors="white", linewidths=0.2)
            axis.set_xlabel("Training target", labelpad=9)
            axis.set_ylabel("Action probability" if scale == "linear" else "log10(action probability)", labelpad=10)
            axis.set_zlabel(f"{label} − target", labelpad=9)
            axis.set_title(f"{label} | {scale} probability")
            axis.set_xlim(targets.min(), targets.max())
            axis.set_ylim(*y_limits)
            axis.set_zlim(*error_limits)
            axis.view_init(elev=24, azim=-55)
            axis.tick_params(labelsize=8)
    legend_handles = [Line2D([], [], marker=marker, color=color, linestyle="none", label=label)
                      for label, color, marker in sign_styles]
    legend_handles.append(Line2D([], [], color="#555555", linestyle=":", label="Same initial state + repeat"))
    figure.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.96),
                  ncol=4, frameon=False)
    figure.get_layout_engine().set(rect=(0, 0, 1, 0.93))
    figure.suptitle(f"{run_dir.name.upper()}: training target × action probability × critic error", fontsize=17)
    figure.supxlabel(
        f"{len(records)} individual rollout points per panel. Top: probability; bottom: log10(probability).\n"
        "Gray plane: error = 0. Dotted lines follow increasing action probability, not time.\n"
        "Error = Q − target: target appears on both X and Z, so correlation alone does not establish bias.", fontsize=11)
    destination = output / "critic_errors_3d.png"
    figure.savefig(destination, dpi=180)
    plt.close(figure)
    print(json.dumps({"output": str(destination), "points_per_panel": len(records)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run_dir", type=Path)
    plot_target_errors_3d(parser.parse_args().run_dir)
