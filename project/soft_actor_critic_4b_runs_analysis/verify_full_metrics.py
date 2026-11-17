"""Independently recompute summary metrics from every native episode CSV."""

import csv
from datetime import datetime, timezone
import json
import math

from summarize_full_evaluations import OUTPUT, MANIFEST_PATH, result_files


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())
    checks = []
    for job in manifest["jobs"]:
        _, summary_path, _ = result_files(job)
        if summary_path is None:
            raise RuntimeError(f"Missing completed evaluation: {job['case']}")
        summary = json.loads(summary_path.read_text())
        with (summary_path.parent / "episode_metrics.csv").open() as stream:
            episodes = list(csv.DictReader(stream))
        distance = math.fsum(float(row["total_distance_travelled_sum"]) for row in episodes)
        infractions = math.fsum(float(row["total_infraction_count"]) for row in episodes)
        pooled = {"total_distance_travelled": distance, "total_infractions": infractions,
                  "avg_distance_per_infraction": distance / max(infractions, 1.0)}
        mismatches = []
        for metric, expected in summary["metrics_mean"].items():
            if metric in pooled:
                actual = pooled[metric]
            else:
                values = [float(row[metric]) for row in episodes if row.get(metric) not in (None, "")]
                actual = math.fsum(values) / len(values)
            if not math.isfinite(actual) or not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-9):
                mismatches.append(metric)
        checks.append(dict(case=job["case"], goal_spacing_m=job["goal_spacing_m"],
                           episode_count=len(episodes), metrics_checked=len(summary["metrics_mean"]),
                           total_distance_m=distance, total_infractions=infractions,
                           distance_per_infraction_m=pooled["avg_distance_per_infraction"],
                           verified=len(episodes) == 1000 and not mismatches, mismatches=mismatches))
    report = dict(checked_at=datetime.now(timezone.utc).isoformat(),
                  all_passed=all(check["verified"] for check in checks), jobs=checks)
    (OUTPUT / "distance_verification.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"all_passed": report["all_passed"], "jobs": len(checks)}))
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
