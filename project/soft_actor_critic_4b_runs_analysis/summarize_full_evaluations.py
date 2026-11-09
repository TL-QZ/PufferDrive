"""Validate and report the native-protocol CARLA checkpoint evaluations."""

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import yaml


OUTPUT = Path(__file__).resolve().parent
MANIFEST_PATH = OUTPUT / "native_manifest.json"
RATE_METRICS = {
    "making_progress_rate", "dnf_rate", "collision_rate", "at_fault_collision_rate",
    "offroad_rate", "red_light_violation_rate",
}
TABLE_METRICS = (
    "puffer_score", "progress_ratio", "num_goals_reached", "making_progress_rate",
    "dnf_rate", "collision_rate", "at_fault_collision_rate", "offroad_rate",
    "red_light_violation_rate", "total_distance_travelled_km", "avg_distance_per_infraction",
)
LOWER_IS_BETTER = {
    'dnf_rate', 'collision_rate', 'at_fault_collision_rate',
    'offroad_rate', 'red_light_violation_rate',
}
FALLBACK_METRICS = (
    "puffer_score", "progress_ratio", "num_goals_reached", "making_progress_rate",
    "dnf_rate", "collision_rate", "at_fault_collision_rate", "offroad_rate",
    "red_light_violation_rate", "avg_speed_per_agent", "episode_return",
    "total_distance_travelled", "total_infractions", "avg_distance_per_infraction",
)
RUN_LABELS = {
    'ppo': '**PPO — BASELINE**', 'tes': 'SAC: TES round 2',
    'nstep4_native10': 'SAC: n-step 4 round 2',
    'nstep2_native10': 'SAC: n-step 2 TES round 4.1',
    'pql_round4_1_native10': 'SAC: PQL round 4.1',
    'pql_official1_native10': 'SAC: PQL official round 1',
    'pql_official2': 'SAC: PQL official round 2',
}


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def result_case(job):
    return job.get("base_case") if job.get("reuse_previous_result") else job["case"]


def result_files(job):
    root = OUTPUT / "results" / result_case(job)
    summaries = sorted(root.glob("carla/*/evaluation_summary.json"), key=lambda path: path.stat().st_mtime)
    if len(summaries) > 1:
        raise RuntimeError(f"Ambiguous evaluation summaries for {job['case']}: {summaries}")
    if summaries:
        summary = summaries[0]
        return root, summary, summary.parent / "resolved_benchmark.yaml"
    resolved = sorted(root.glob("carla/*/resolved_benchmark.yaml"), key=lambda path: path.stat().st_mtime)
    return root, None, resolved[-1] if resolved else None


def catalog_checks(job):
    catalog_path = Path(job["catalog"])
    expected_spacing = float(job["goal_spacing_m"])
    check = {"catalog_hash_matches": False, "catalog_protocol_matches": False,
             "catalog_goal_spacing_m": None, "expected_goal_spacing_m": expected_spacing}
    if not catalog_path.is_file():
        return check
    check["catalog_hash_matches"] = sha256(catalog_path) == job["catalog_sha256"]
    catalog = yaml.safe_load(catalog_path.read_text())
    env = catalog.get("env", {})
    benchmark = next((item for item in catalog.get("benchmarks", []) if item.get("name") == "carla"), {})
    min_spacing = env.get("min_goal_spacing")
    max_spacing = env.get("max_goal_spacing")
    check["catalog_goal_spacing_m"] = min_spacing
    check["catalog_protocol_matches"] = (
        min_spacing is not None and max_spacing is not None
        and abs(float(min_spacing) - expected_spacing) < 1e-6
        and abs(float(max_spacing) - expected_spacing) < 1e-6
        and int(benchmark.get("seed", -1)) == 42
        and int(benchmark.get("num_scenarios", -1)) == 1000
        and int(benchmark.get("env", {}).get("scenario_length", -1)) == 6000
        and abs(float(env.get("dt", -1)) - 0.1) < 1e-6
    )
    return check


def resolved_checks(resolved_path, expected_spacing):
    if not resolved_path or not resolved_path.is_file():
        return {"resolved_benchmark_path": None, "resolved_goal_spacing_m": None,
                "resolved_protocol_matches": False}
    resolved = yaml.safe_load(resolved_path.read_text())
    args_env = resolved.get("args", {}).get("env", {})
    benchmark = resolved.get("benchmark", {})
    min_spacing = args_env.get("min_goal_spacing")
    max_spacing = args_env.get("max_goal_spacing")
    protocol_matches = (
        min_spacing is not None and max_spacing is not None
        and abs(float(min_spacing) - expected_spacing) < 1e-6
        and abs(float(max_spacing) - expected_spacing) < 1e-6
        and int(benchmark.get("seed", -1)) == 42
        and int(benchmark.get("num_scenarios", -1)) == 1000
        and int(benchmark.get("env", {}).get("scenario_length", -1)) == 6000
        and abs(float(args_env.get("dt", -1)) - 0.1) < 1e-6
    )
    return {"resolved_benchmark_path": str(resolved_path.relative_to(OUTPUT)),
            "resolved_goal_spacing_m": min_spacing, "resolved_protocol_matches": protocol_matches}


def metric_text(row, key):
    value = row.get(key)
    if value in (None, ""):
        return "—"
    number = float(value)
    if key in RATE_METRICS:
        return f"{100 * number:.2f}%"
    if key == "total_distance_travelled_km":
        return f"{number:,.1f}"
    return f"{number:.5g}"


def report_table(rows):
    headings = [
        "Run", "Goal spacing (m)", "Status", "Episodes", "Puffer score", "Progress ratio",
        "Goals reached", "Making progress", "DNF", "Collision", "At-fault collision",
        "Off-road", "Red-light", "Total distance (km)", "Distance/infraction (m)",
    ]
    ranked_values = {
        metric: sorted({float(row[metric]) for row in rows
                        if row.get(metric) not in (None, '') and row['status'] == 'finished'},
                       reverse=metric not in LOWER_IS_BETTER)[:2]
        for metric in TABLE_METRICS
    }
    headings[4:] = [heading + (' ↓' if metric in LOWER_IS_BETTER else ' ↑')
                    for heading, metric in zip(headings[4:], TABLE_METRICS)]
    output = ["**Bold = best; <u>underline = second-best</u> in each metric column.** ↑ higher is better; ↓ lower is better. Rankings use unrounded values across the displayed runs; ties share rank. Goal counts reflect each run's 10 m or 30 m spacing.", "",
              "| " + " | ".join(headings) + " |",
              "|" + "|".join(["---"] * 3 + ["---:"] * (len(headings) - 3)) + "|"]
    for row in rows:
        values = [RUN_LABELS.get(row["case"], row["case"]), metric_text(row, "goal_spacing_m"), row["status"],
                  str(row.get("num_episodes") or "—")]
        for metric in TABLE_METRICS:
            text = metric_text(row, metric)
            eligible = row.get(metric) not in (None, '') and row['status'] == 'finished'
            ranks = ranked_values[metric]
            if eligible and ranks and float(row[metric]) == ranks[0]:
                text = f'**{text}**'
            elif eligible and len(ranks) > 1 and float(row[metric]) == ranks[1]:
                text = f'<u>{text}</u>'
            values.append(text)
        output.append("| " + " | ".join(values) + " |")
    return output


def build_resource_summary(manifest):
    created_at = datetime.fromisoformat(manifest["created_at"])
    usage_path = OUTPUT / "resource_usage.jsonl"
    samples = []
    if usage_path.exists():
        for line in usage_path.read_text().splitlines():
            if not line.strip():
                continue
            sample = json.loads(line)
            if datetime.fromisoformat(sample["timestamp"]) >= created_at:
                samples.append(sample)
    peak_gpu = {}
    peak_rss = {}
    observed_cases = set()
    for sample in samples:
        for gpu in sample.get("gpus", []):
            key = str(gpu["gpu"])
            peak_gpu[key] = max(peak_gpu.get(key, 0), int(gpu["memory_used_mib"]))
        for job in sample.get("jobs", []):
            case = job["case"]
            observed_cases.add(case)
            peak_rss[case] = max(peak_rss.get(case, 0), int(job.get("rss_bytes", 0)))
    available = [int(sample["host_mem_available_bytes"]) for sample in samples]
    result = {
        "batch_start": manifest["created_at"],
        "batch_end": samples[-1]["timestamp"] if samples else None,
        "sample_count": len(samples),
        "minimum_host_available_gib": round(min(available) / (1024**3), 3) if available else None,
        "peak_gpu_memory_mib": peak_gpu,
        "peak_process_group_rss_gib": {case: round(value / (1024**3), 3) for case, value in sorted(peak_rss.items())},
        "observed_cases": sorted(observed_cases),
        "history_source": "resource_usage.jsonl filtered at native_manifest.created_at",
    }
    (OUTPUT / "resource_summary.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def main():
    manifest = json.loads(MANIFEST_PATH.read_text())
    rows, checks, scenario_sets = [], [], {}
    resource_summary = build_resource_summary(manifest)
    for job in manifest["jobs"]:
        case = job["case"]
        status_path = OUTPUT / f"{case}.status.json"
        status = json.loads(status_path.read_text()) if status_path.exists() else {"state": "pending"}
        result_root, summary_path, resolved_path = result_files(job)
        catalog_check = catalog_checks(job)
        resolved_check = resolved_checks(resolved_path, float(job["goal_spacing_m"]))
        row = {
            "case": case, "run_id": job["run_id"], "algorithm": job["algorithm"],
            "status": status["state"], "goal_spacing_m": float(job["goal_spacing_m"]),
            "result_reused": bool(job.get("reuse_previous_result")),
            "result_source_case": result_case(job), "num_scenarios": None, "num_episodes": None,
            "summary_path": None,
        }
        check = {
            "case": case, "status": status["state"], "complete": False,
            "result_reused": bool(job.get("reuse_previous_result")),
            **catalog_check, **resolved_check,
        }
        if summary_path:
            summary = json.loads(summary_path.read_text())
            episode_path = summary_path.parent / "episode_metrics.csv"
            with episode_path.open(newline="") as stream:
                episodes = list(csv.DictReader(stream))
            identities = {(episode["map_name"], episode["seed"]) for episode in episodes}
            scenario_sets[case] = identities
            metrics = summary.get("metrics_mean", {})
            row.update(num_scenarios=summary.get("num_scenarios"), num_episodes=summary.get("num_episodes"),
                       summary_path=str(summary_path.relative_to(OUTPUT)), **metrics)
            if metrics.get("total_distance_travelled") is not None:
                row["total_distance_travelled_km"] = float(metrics["total_distance_travelled"]) / 1000.0
            render_files = list(summary_path.parent.rglob("*.html"))
            checkpoint_path = Path(job["checkpoint"])
            checkpoint_matches = sha256(checkpoint_path) == job["checkpoint_sha256"]
            config_path = checkpoint_path.with_name("config.yaml")
            config_matches = sha256(config_path) == job["config_sha256"]
            complete = (
                status["state"] == "finished" and status.get("exitcode") == 0
                and len(episodes) == 1000 and summary.get("num_scenarios") == 1000
                and summary.get("num_episodes") == 1000 and len(identities) == 1000
                and not render_files and checkpoint_matches and config_matches
                and catalog_check["catalog_hash_matches"] and catalog_check["catalog_protocol_matches"]
                and resolved_check["resolved_protocol_matches"]
            )
            check.update(episode_rows=len(episodes), num_scenarios=summary.get("num_scenarios"),
                         num_episodes=summary.get("num_episodes"), html_files=len(render_files),
                         unique_map_seed_pairs=len(identities), checkpoint_hash_matches=checkpoint_matches,
                         config_hash_matches=config_matches, scenario_pairs=sorted([list(item) for item in identities]))
            check["complete"] = complete
        else:
            check.update(checkpoint_hash_matches=sha256(Path(job["checkpoint"])) == job["checkpoint_sha256"],
                         config_hash_matches=sha256(Path(job["checkpoint"]).with_name("config.yaml")) == job["config_sha256"])
        rows.append(row)
        checks.append(check)

    rows.sort(key=lambda row: row['algorithm'] != 'PPO')
    available_sets_match = bool(scenario_sets) and all(
        values == next(iter(scenario_sets.values())) for values in scenario_sets.values()
    )
    same_scenarios = len(scenario_sets) == len(rows) and available_sets_match
    all_complete = all(check["complete"] for check in checks) and same_scenarios
    report_status = {
        "complete": all_complete,
        "identical_scenario_sets": same_scenarios,
        "available_scenario_sets_match": available_sets_match,
        "scenario_sets_available": len(scenario_sets),
        "expected_jobs": len(rows),
        "protocol": manifest["protocol"],
        "jobs": checks,
        "resource_summary": "resource_summary.json",
    }
    (OUTPUT / "verification.json").write_text(json.dumps(report_status, indent=2) + "\n")

    metric_fields = list(dict.fromkeys(key for row in rows for key in row if key not in {
        "case", "run_id", "algorithm", "status", "goal_spacing_m", "result_reused",
        "result_source_case", "num_scenarios", "num_episodes", "summary_path",
    }))
    for metric in FALLBACK_METRICS:
        if metric not in metric_fields:
            metric_fields.append(metric)
    if "total_distance_travelled_km" not in metric_fields:
        metric_fields.append("total_distance_travelled_km")
    columns = ["case", "run_id", "algorithm", "status", "goal_spacing_m", "result_reused",
               "result_source_case", "num_scenarios", "num_episodes", *metric_fields, "summary_path"]
    with (OUTPUT / "full_carla_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)

    finished = sum(check["complete"] for check in checks)
    table = report_table(rows)
    validation = (
        f"All seven evaluated checkpoints (PPO plus six SAC runs) passed: 7,000 episodes, matched map/seed pairs across goal spacings, "
        "verified checkpoint/config/catalog hashes, and no rendered HTML."
        if all_complete else
        f"Pending or incomplete: {finished}/{len(rows)} jobs verified; "
        f"{len(scenario_sets)}/{len(rows)} scenario sets are available. Pending jobs show no inherited scores."
    )
    notes = [
        "The two 30 m jobs marked reused read their previously completed full-CARLA output; their resolved benchmark files are checked against the native 30 m catalog. Other jobs use their own native catalog outputs.",
        "Rate columns are displayed as percentages. In the CSV, rate fields remain fractions. `total_distance_travelled` is the summed distance across all evaluated episode agents, not an episode mean; the table converts it to km. `avg_distance_per_infraction` is summed distance divided by max(summed `total_infraction_count`, 1), not a mean of episode ratios. The count is affected agent-episodes (an agent is counted for a collision, off-road event, or red-light violation), not a raw event count; see [`pufferlib/utils.py`](../../pufferlib/utils.py).",
        "Every run uses seed 42, 1,000 scenarios, 6,000 steps and dt 0.1. Scenario identity is checked as (map_name, seed) pairs across all available results, including across 10 m and 30 m goal spacing.",
        "Each row represents one training checkpoint and one evaluation seed. The recovered n-step-2, PQL round-4.1, and PQL official round-2 checkpoints came from learner-only recovery with replay and environments restarted.",
    ]
    report = ["# Full CARLA native-protocol final-checkpoint evaluation", "", manifest["protocol"], "",
              "The current comparison uses all seven eligible final checkpoints in [native_manifest.json](native_manifest.json). The earlier six-job common-30 m comparison is retained in [historical_common30m](historical_common30m/).", "",
              *table, "", "All summary metrics and episode counts: [full_carla_metrics.csv](full_carla_metrics.csv).", "",
              "Validation: " + validation, "", *[f"- {note}" for note in notes], ""]
    (OUTPUT / "full_carla_report.md").write_text("\n".join(report))

    techniques = (OUTPUT / 'sac_techniques.md').read_text()
    results = ["# PPO baseline and SAC technique comparison", "",
               "## Full-CARLA results — PPO baseline first", "",
               "**PPO is the baseline in the first row; all SAC runs follow in the same table.** All full evaluations use 1,000 scenarios, seed 42, 6,000 steps and dt=0.1; goal spacing uses each run's native evaluation catalog (10 m or 30 m).", "",
               *table, "", validation, "",
               techniques, "",
               "Goal spacing follows each run's evaluation protocol: 10 m for n-step-4, n-step-2, PQL round 4.1, and PQL official round 1; 30 m for TES round 2, PPO, and PQL official round 2. Catalog hashes and resolved min/max spacing are checked per job.", "",
               "The total-distance column is the sum over all agents and episodes, shown in km. Distance per infraction divides the global distance sum by max(summed `total_infraction_count`, 1); that count records affected agent-episodes (collision, off-road, or red-light), not raw events. Rate metrics are percentages here; the CSV stores their original fractions. The aggregation follows [`pufferlib/utils.py`](../../pufferlib/utils.py).", "",
               "See [full_carla_report.md](full_carla_report.md) for the protocol notes and [full_carla_metrics.csv](full_carla_metrics.csv) for every metric. [verification.json](verification.json) tracks job status, hash checks, resolved spacing and scenario identity. [resource_summary.json](resource_summary.json) covers samples from this native batch only.", "",
               "## Full-evaluation distance comparison", "",
               "![Full CARLA distance metrics](full_carla_distances.png)", "",
               "[PDF](full_carla_distances.pdf). Rebuild with `python project/soft_actor_critic_4b_runs_analysis/plot_full_distances.py` after regenerating the tables. [Independent distance recomputation](distance_verification.json) checks both values against every episode CSV.", "",
               "## Training comparison context", "",
               "Learning curves remain historical `carla_fast` evaluations and are matched by received-agent steps in [learning_curve_report.md](learning_curve_report.md). They are not the full-CARLA final-checkpoint results above. The unfinished run remains excluded from final-checkpoint evaluation.", "",
               "The n-step-2, PQL round-4.1, and PQL official round-2 policies include learner-only recovery; replay and environments restarted. These are individual runs, not multi-seed algorithm estimates.", ""]
    if (OUTPUT / "render_manifest.json").exists():
        results.extend(["## Failure renders", "",
                        "[Failure-render galleries and verification](failure_render_report.md): 10 selected failure scenarios each for PPO 4B, PQL official round 2, and n-step 4 round 2, using their native evaluation settings.", ""])
    (OUTPUT / "RESULTS.md").write_text("\n".join(results))
    print(json.dumps({"complete": all_complete, "verified_jobs": finished,
                      "expected_jobs": len(rows), "scenario_sets_available": len(scenario_sets),
                      "identical_scenario_sets": same_scenarios}, indent=2))


if __name__ == "__main__":
    main()
