"""Snapshot local TES-98 histories and plot entropy against driving performance.

Run from the activated repository venv. Reads existing artifacts only; never
loads models, starts training, or contacts W&B. Outputs stay beside this script.
"""

import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

OUTPUT = Path(__file__).resolve().parent
RUNS = OUTPUT.parent / "runs"
REPO = OUTPUT.parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(OUTPUT / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


TRAIN_FIELDS = [
    "epoch", "agent_steps", "losses/entropy", "losses/target_entropy",
    "losses/alpha", "losses/q1", "losses/q2", "losses/critic_loss",
    "learning_rate", "alpha_learning_rate", "target_entropy/decay_count",
    "target_entropy/forced_decay_count", "target_entropy/mean", "target_entropy/std",
]
EVAL_FIELDS = [
    "puffer_score", "progress_ratio", "avg_speed_per_agent", "offroad_rate",
    "collision_rate", "num_goals_reached", "episode_return", "making_progress_rate",
]


def write_csv(path, rows, fields):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main():
    data_dir = OUTPUT / "evidence"
    data_dir.mkdir(exist_ok=True)
    provenance = []
    summaries = []
    training_rows = []
    evaluation_rows = []
    histories = []
    for metrics_path in sorted(RUNS.glob("*/trial_98/metrics.jsonl")):
        raw = metrics_path.read_bytes()
        rows = [json.loads(line) for line in raw.splitlines() if line.strip()]
        rows = [row for row in rows if "losses/entropy" in row]
        if not rows:
            continue
        run = metrics_path.parent.parent.name
        provenance.append({"path": str(metrics_path.relative_to(REPO)),
                           "sha256": hashlib.sha256(raw).hexdigest(), "rows": len(rows)})
        training_rows.extend({"run": run, **row} for row in rows)
        evals = []
        for path in sorted((metrics_path.parent / "learner/eval/training/carla_fast").glob(
                "epoch_*_step_*/evaluation_summary.json")):
            match = re.fullmatch(r"epoch_(\d+)_step_(\d+)", path.parent.name)
            if match is None:
                raise ValueError(f"Unexpected evaluation directory: {path}")
            eval_raw = path.read_bytes()
            report = json.loads(eval_raw)
            row = {"run": run, "epoch": int(match[1]), "agent_steps": int(match[2]),
                   "source": str(path.relative_to(REPO)), **report["metrics_mean"]}
            evals.append(row)
            provenance.append({"path": row["source"],
                               "sha256": hashlib.sha256(eval_raw).hexdigest()})
        evals.sort(key=lambda row: row["agent_steps"])
        evaluation_rows.extend(evals)
        milestones = {"first": rows[0], "latest": rows[-1],
                      "minimum_entropy": min(rows, key=lambda row: row["losses/entropy"]),
                      "minimum_alpha": min(rows, key=lambda row: row["losses/alpha"])}
        for threshold in (1.0, 0.1, 0.001):
            milestones[f"first_entropy_below_{threshold}"] = next(
                (row for row in rows if row["losses/entropy"] < threshold), None)
        milestones["first_exact_zero"] = next(
            (row for row in rows if row["losses/entropy"] == 0), None)
        summaries.append({
            "run": run, "training_rows": len(rows), "evaluation_rows": len(evals),
            "exact_zero_rows": sum(row["losses/entropy"] == 0 for row in rows),
            "milestones": {name: {key: row.get(key) for key in TRAIN_FIELDS}
                           if row is not None else None for name, row in milestones.items()},
            "peak_evaluation": max(evals, key=lambda row: row["puffer_score"]) if evals else None,
            "latest_evaluation": evals[-1] if evals else None,
        })
        histories.append((run, rows, evals))

    write_csv(data_dir / "training.csv", training_rows, ["run", *TRAIN_FIELDS])
    write_csv(data_dir / "evaluations.csv", evaluation_rows,
              ["run", "epoch", "agent_steps", *EVAL_FIELDS, "source"])
    (data_dir / "summary.json").write_text(json.dumps(summaries, indent=2) + "\n")
    (data_dir / "provenance.json").write_text(json.dumps({
        "extracted_at_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Nonempty standard TES trial_98 histories; excludes PPO behavior variants",
        "sources": provenance,
    }, indent=2) + "\n")

    figure, axes = plt.subplots(len(histories), 3, figsize=(15, 3.2 * len(histories)), squeeze=False)
    for index, (run, rows, evals) in enumerate(histories):
        steps = [row["agent_steps"] / 1e9 for row in rows]
        label = run.split("_2026")[0] + " / " + run.split("_2026")[1][:13]
        axes[index, 0].plot(steps, [row["losses/entropy"] for row in rows], label="Observed H")
        axes[index, 0].plot(steps, [row["losses/target_entropy"] for row in rows], label="Target H")
        axes[index, 0].set_ylabel(label + "\nEntropy (nats)")
        axes[index, 0].set_ylim(bottom=0)
        axes[index, 0].legend()
        axes[index, 1].semilogy(steps, [row["losses/alpha"] for row in rows])
        axes[index, 1].set_ylabel("Alpha (log scale)")
        axes[index, 2].plot([row["agent_steps"] / 1e9 for row in evals],
                            [row["puffer_score"] for row in evals])
        axes[index, 2].set_ylabel("CARLA puffer score")
        for axis in axes[index]:
            axis.set_xlabel("Global received rows (billions)")
            axis.grid(alpha=0.25)
    figure.suptitle("TES trial 98: raw local histories; goal spacing differs between rounds")
    figure.tight_layout()
    figure.savefig(data_dir / "entropy_and_performance.png", dpi=150)
    svg_path = data_dir / "entropy_and_performance.svg"
    figure.savefig(svg_path)
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    plt.close(figure)
    print(f"Extracted {len(histories)} runs, {len(training_rows)} learner rows, "
          f"{len(evaluation_rows)} evaluations into {data_dir}")


if __name__ == "__main__":
    main()
