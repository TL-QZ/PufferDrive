#!/usr/bin/env python3
"""Verify prepared failure replay artifacts and refresh their Markdown report."""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import pandas as pd
import yaml

from pufferlib.ocean.evaluation_utils.evaluation_utils import select_render_rows


PROJECT_DIR = Path(__file__).resolve().parent
MANIFEST_PATH = PROJECT_DIR / "render_manifest.json"
NATIVE_MANIFEST_PATH = PROJECT_DIR / "native_manifest.json"
REPORT_PATH = PROJECT_DIR / "failure_render_report.md"
CASES = ("render_ppo", "render_pql_official2", "render_nstep4")
LABELS = {"render_ppo": "PPO 4B baseline", "render_pql_official2": "PQL official round 2",
          "render_nstep4": "n-step 4 round 2"}
COMPARE_METRICS = (
    "episode_return", "score", "collision_rate", "offroad_rate", "red_light_violation_rate",
    "num_goals_reached", "total_distance_travelled_sum", "total_infraction_count",
    "avg_distance_per_infraction",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rel(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_DIR).as_posix()


def map_key(value) -> str:
    return os.path.splitext(os.path.basename(str(value)))[0]


def identities(frame: pd.DataFrame) -> list[tuple[str, int]]:
    return [(map_key(row.map_name), int(row.seed)) for row in frame.itertuples()]


def benchmark_spacing(path: Path) -> tuple[float, float]:
    resolved = yaml.safe_load(path.read_text(encoding="utf-8"))
    environment = resolved["args"]["env"]
    return float(environment["min_goal_spacing"]), float(environment["max_goal_spacing"])


def verify_job(job: dict, native_jobs: dict) -> dict:
    case = job["case"]
    checks: list[tuple[str, str, str]] = []
    hard_failures: list[str] = []
    pending: list[str] = []
    metric_mismatches: list[str] = []
    compared_metric_count = 0

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append(("PASS" if ok else "FAIL", name, detail))
        if not ok:
            hard_failures.append(f"{name}: {detail}")

    def waiting(name: str, detail: str) -> None:
        checks.append(("WAIT", name, detail))
        pending.append(f"{name}: {detail}")

    def count_check(name: str, found: int, expected: int, detail: str) -> None:
        if found == expected:
            check(name, True, detail)
        elif status.get("state") in {"running", "pending"} and found < expected:
            waiting(name, detail)
        else:
            check(name, False, detail)

    status_path = PROJECT_DIR / f"{case}.status.json"
    if status_path.is_file():
        status = json.loads(status_path.read_text(encoding="utf-8"))
        state = status.get("state", "unknown")
        if state == "finished" and status.get("exitcode") == 0:
            check("process status", True, "finished with exit code 0")
        elif state in {"running", "pending"}:
            waiting("process status", f"{state}; exit code {status.get('exitcode')}")
        else:
            check("process status", False, f"state={state}, exit code={status.get('exitcode')}")
    else:
        status = {}
        waiting("process status", "status file has not been written")

    source_csv = Path(job["source_episode_metrics_csv"])
    source_resolved = Path(job["source_resolved_benchmark"])
    replay_csv = Path(job["failure_replay_csv"])
    copied_resolved = replay_csv.with_name("resolved_benchmark.yaml")
    native_job = native_jobs[job["source_case"]]
    config_path = Path(job["checkpoint"]).with_name("config.yaml")
    hash_checks = (
        ("checkpoint hash", Path(job["checkpoint"]), native_job["checkpoint_sha256"]),
        ("catalog hash", Path(job["catalog"]), native_job["catalog_sha256"]),
        ("config hash", config_path, native_job["config_sha256"]),
        ("source CSV hash", source_csv, job["source_episode_metrics_csv_sha256"]),
        ("replay CSV hash", replay_csv, job["failure_replay_csv_sha256"]),
        ("source resolved-config hash", source_resolved, job["source_resolved_benchmark_sha256"]),
        ("copied resolved-config hash", copied_resolved, job["resolved_benchmark_sha256"]),
    )
    for name, path, expected in hash_checks:
        if path.is_file():
            check(name, sha256(path) == expected, f"{rel(path)}; expected {expected}")
        elif status.get("state") in {"running", "pending"}:
            waiting(name, f"file not ready: {rel(path)}")
        else:
            check(name, False, f"file missing: {rel(path)}")

    expected_spacing = {"render_ppo": (30.0, 30.0), "render_pql_official2": (30.0, 30.0), "render_nstep4": (10.0, 10.0)}[case]
    replay_resolved = replay_csv.parent / "failures" / "resolved_benchmark.yaml"
    source_spacing = None
    replay_spacing = None
    if source_resolved.is_file() and replay_resolved.is_file():
        source_spacing = benchmark_spacing(source_resolved)
        replay_spacing = benchmark_spacing(replay_resolved)
        check(
            "resolved goal spacing",
            source_spacing == replay_spacing == expected_spacing,
            f"source {source_spacing[0]:g}/{source_spacing[1]:g} m; replay {replay_spacing[0]:g}/{replay_spacing[1]:g} m",
        )
    elif status.get("state") in {"running", "pending"}:
        waiting("resolved goal spacing", "source or generated replay config not ready")
    else:
        check("resolved goal spacing", False, "source or generated replay config missing")

    failure_dir = replay_csv.parent / "failures"
    selected_path = failure_dir / "selected_failures.csv"
    replay_metrics_path = failure_dir / "episode_metrics.csv"
    replay_dir = failure_dir / "replays"
    html_dir = failure_dir / "rendered_replays"
    source_rows = None
    selected_rows = None
    replay_rows = None
    if source_csv.is_file():
        try:
            source_rows = select_render_rows(str(source_csv), job["render_filter"]).head(10).copy()
            selected_count = len(source_rows)
            check("source selection", selected_count == 10, f"exact source selector produced {selected_count} first rows")
        except Exception as error:
            check("source selection", False, str(error))
    else:
        waiting("source selection", f"source CSV missing: {rel(source_csv)}")

    if selected_path.is_file():
        try:
            selected_rows = pd.read_csv(selected_path)
            if source_rows is not None:
                expected_ids = identities(source_rows)
                selected_ids = identities(selected_rows)
                detail = f"{len(selected_rows)} selected; identities match source first-10: {selected_ids == expected_ids}"
                count_check("selected row count", len(selected_rows), 10, detail)
                if len(selected_rows) == 10:
                    check("selected identity match", selected_ids == expected_ids, detail)
                batch_values = sorted(pd.to_numeric(selected_rows["agents_per_batch"], errors="raise").astype(int).unique())
                check("recorded batch size", batch_values == [6000], f"selected agents_per_batch values: {batch_values}")
            else:
                count_check("selected rows", len(selected_rows), 10, f"found {len(selected_rows)}")
        except Exception as error:
            if status.get("state") in {"running", "pending"}:
                waiting("selected rows", f"file may still be written: {error}")
            else:
                check("selected rows", False, str(error))
    elif status.get("state") in {"running", "pending"}:
        waiting("selected rows", "selected_failures.csv not ready")
    else:
        check("selected rows", False, "selected_failures.csv missing")

    if replay_metrics_path.is_file():
        try:
            replay_rows = pd.read_csv(replay_metrics_path)
            count_check("replayed episode rows", len(replay_rows), 10, f"found {len(replay_rows)} rows")
            if source_rows is not None:
                source_ids = identities(source_rows)
                replay_ids = identities(replay_rows)
                same_replayed_ids = Counter(replay_ids) == Counter(source_ids)
                detail = f"{len(replay_ids)} rows; identities match selected source: {same_replayed_ids}"
                if len(replay_rows) == 10:
                    check("replayed identities", same_replayed_ids, detail)
                    check("unique scenario identities", len(set(source_ids)) == len(set(replay_ids)) == 10,
                          "source and replay must each contain ten distinct map/seed pairs")
                elif status.get("state") in {"running", "pending"}:
                    waiting("replayed identities", detail)
                else:
                    check("replayed identities", False, detail)
                source_by_id = source_rows.assign(_map=source_rows["map_name"].map(map_key)).set_index(["_map", "seed"])
                replay_by_id = replay_rows.assign(_map=replay_rows["map_name"].map(map_key)).set_index(["_map", "seed"])
                joined = source_by_id.join(replay_by_id, lsuffix="_source", rsuffix="_replay", how="inner")
                missing_metrics = set(COMPARE_METRICS) - (set(source_rows.columns) & set(replay_rows.columns))
                check("required metric columns", not missing_metrics, f"missing: {sorted(missing_metrics)}")
                for metric in COMPARE_METRICS if len(replay_rows) == 10 else ():
                    left_name = f"{metric}_source" if f"{metric}_source" in joined else metric
                    right_name = f"{metric}_replay" if f"{metric}_replay" in joined else metric
                    if left_name not in joined or right_name not in joined or left_name == right_name:
                        continue
                    for identity, row in joined.iterrows():
                        compared_metric_count += 1
                        left = pd.to_numeric(pd.Series([row[left_name]]), errors="coerce").iloc[0]
                        right = pd.to_numeric(pd.Series([row[right_name]]), errors="coerce").iloc[0]
                        if pd.isna(left) or pd.isna(right) or not math.isclose(float(left), float(right), rel_tol=1e-5, abs_tol=1e-4):
                            metric_mismatches.append(f"{identity[0]} seed {identity[1]} {metric}: source={left}, replay={right}")
                if len(replay_rows) == 10:
                    check("metric comparison coverage", compared_metric_count == 10 * len(COMPARE_METRICS),
                          f"compared {compared_metric_count} values; expected {10 * len(COMPARE_METRICS)}")
            else:
                waiting("metric comparison", "source selection is not available")
        except Exception as error:
            if status.get("state") in {"running", "pending"}:
                waiting("replayed metrics", f"file may still be written: {error}")
            else:
                check("replayed metrics", False, str(error))
    elif status.get("state") in {"running", "pending"}:
        waiting("replayed episode rows", "failure episode_metrics.csv not ready")
    else:
        check("replayed episode rows", False, "failure episode_metrics.csv missing")

    if replay_dir.is_dir():
        replay_files = sorted(replay_dir.glob("*.replay.zlib"))
        count_check("compressed replay files", len(replay_files), 10, f"found {len(replay_files)} .replay.zlib files")
    elif status.get("state") in {"running", "pending"}:
        waiting("compressed replay files", "replay directory not ready")
    else:
        check("compressed replay files", False, "replay directory missing")

    if html_dir.is_dir():
        html_files = sorted(path for path in html_dir.glob("*.html") if path.name != "index.html")
        count_check("HTML pages", len(html_files), 10, f"found {len(html_files)} episode pages excluding index.html")
        if (html_dir / "index.html").is_file():
            check("gallery index", True, "index.html exists")
        elif status.get("state") in {"running", "pending"}:
            waiting("gallery index", "index.html not ready")
        else:
            check("gallery index", False, "index.html missing")
    elif status.get("state") in {"running", "pending"}:
        waiting("HTML gallery", "rendered_replays directory not ready")
    else:
        check("HTML gallery", False, "rendered_replays directory missing")

    if replay_rows is not None and len(replay_rows) == 10:
        check("replay path column", "replay_path" in replay_rows, "episode metrics must identify their replay files")
        if "replay_path" in replay_rows:
            names_match_rows = all(
                Path(row.replay_path).name.startswith(f"{map_key(row.map_name)}__seed_{int(row.seed)}__episode_")
                for row in replay_rows.itertuples()
            )
            check("artifact map/seed identities", names_match_rows,
                  "each replay filename must encode its episode map and seed")
            expected_replays = {Path(value).name for value in replay_rows["replay_path"]}
            expected_pages = {name.removesuffix(".replay.zlib") + ".html" for name in expected_replays}
            actual_replays = {path.name for path in replay_dir.glob("*.replay.zlib")}
            actual_pages = {path.name for path in html_dir.glob("*.html") if path.name != "index.html"}
            check("replay artifact identities", len(expected_replays) == 10 and actual_replays == expected_replays,
                  "compressed filenames must match all ten replayed episodes")
            if len(actual_pages) == 10 or status.get("state") == "finished":
                check("HTML artifact identities", actual_pages == expected_pages,
                      "HTML filenames must match all ten replayed episodes")
            if (html_dir / "index.html").is_file():
                index_text = (html_dir / "index.html").read_text()
                check("gallery page links", all(name in index_text for name in expected_pages),
                      "index must reference all ten episode pages")

    log_path = Path(status["log"]) if status.get("log") else None
    if log_path and log_path.is_file():
        log_text = log_path.read_text(encoding="utf-8", errors="replace")
        padding_ok = "Padding policy inference from 500 to 6000 agents" in log_text
        if padding_ok:
            checks.append(("PASS", "batch padding", "policy inference padded to recorded 6000-agent batch"))
        elif status.get("state") in {"running", "pending"}:
            waiting("batch padding", "6000-agent padding confirmation not in log yet")
        else:
            check("batch padding", False, "6000-agent padding confirmation absent")
    else:
        waiting("batch padding", "runner log not available")

    return {
        "case": case,
        "source_case": job["source_case"],
        "gallery": rel(html_dir / "index.html"),
        "status": status.get("state", "pending"),
        "checks": checks,
        "failures": hard_failures,
        "pending": pending,
        "metric_mismatches": metric_mismatches,
        "compared_metric_count": compared_metric_count,
        "source_spacing": source_spacing if source_resolved.is_file() else None,
        "replay_spacing": replay_spacing if replay_resolved.is_file() else None,
        "selected_count": len(selected_rows) if selected_rows is not None else 0,
        "replayed_count": len(replay_rows) if replay_rows is not None else 0,
        "replay_count": len(list(replay_dir.glob("*.replay.zlib"))) if replay_dir.is_dir() else 0,
        "html_count": len([path for path in html_dir.glob("*.html") if path.name != "index.html"]) if html_dir.is_dir() else 0,
        "index_exists": (html_dir / "index.html").is_file(),
    }


def render_report(results: list[dict]) -> str:
    failures = sum(len(result["failures"]) for result in results)
    pending = sum(len(result["pending"]) for result in results)
    mismatches = [(result["case"], item) for result in results for item in result["metric_mismatches"]]
    if failures:
        overall = "FAILED"
    elif pending:
        overall = "INCOMPLETE — render jobs or artifacts are still pending"
    elif mismatches:
        overall = "COMPLETE — metric mismatches are reported below"
    else:
        overall = "VERIFIED"

    lines = [
        "# Failure replay verification",
        "",
        f"**Overall: {overall}.**",
        "",
        "The renderer selected the first 10 rows returned by the source `all_infractions` selector. "
        "Replay identities are checked by map and seed; metric differences are listed explicitly. "
        "Recorded `agents_per_batch=6000` is checked in the selected rows and against the runner log.",
        "",
        "| Case | Gallery | Process | Goal spacing (source → replay) | Selected | Replayed | Zlib | HTML | Index |",
        "|---|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for result in results:
        source_spacing = result["source_spacing"]
        replay_spacing = result["replay_spacing"]
        spacing = "pending" if source_spacing is None or replay_spacing is None else (
            f"{source_spacing[0]:g}/{source_spacing[1]:g} → {replay_spacing[0]:g}/{replay_spacing[1]:g} m"
        )
        gallery = f"[open]({result['gallery']})" if result["index_exists"] else "pending"
        lines.append(
            f"| {LABELS[result['case']]} | {gallery} | {result['status']} | {spacing} | "
            f"{result['selected_count']}/10 | {result['replayed_count']}/10 | "
            f"{result['replay_count']}/10 | {result['html_count']}/10 | "
            f"{'yes' if result['index_exists'] else 'pending'} |"
        )

    lines.extend(["", "## Gallery links", "", "[Execution notes and allocation fix](render_execution_notes.md).", ""])
    for result in results:
        lines.append(f"- [{LABELS[result['case']]} gallery]({result['gallery']})")
    for result in results:
        lines.extend(["", f"## {result['case']}", ""])
        lines.append(f"Runner state: `{result['status']}`. Source case: `{result['source_case']}`.")
        lines.append("")
        for state, name, detail in result["checks"]:
            lines.append(f"- **{state}** {name}: {detail}")
        if result["failures"]:
            lines.extend(["", "Verification failures:"])
            lines.extend(f"- {failure}" for failure in result["failures"])
        if result["pending"]:
            lines.extend(["", "Pending artifacts/checks:"])
            lines.extend(f"- {item}" for item in result["pending"])
        if result["metric_mismatches"]:
            lines.extend(["", f"Metric mismatches ({len(result['metric_mismatches'])} values):"])
            lines.extend(f"- {item}" for item in result["metric_mismatches"])
        elif result["compared_metric_count"] == 10 * len(COMPARE_METRICS) and not result["failures"]:
            lines.extend(["", "Compared metrics match within `rel_tol=1e-5`, `abs_tol=1e-4`."])
    lines.extend(["", "## Reproduce", "", "```sh", "source .venv/bin/activate", "python -B project/soft_actor_critic_4b_runs_analysis/verify_failure_renders.py", "```", ""])
    return "\n".join(lines)


def main() -> None:
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    native = json.loads(NATIVE_MANIFEST_PATH.read_text(encoding="utf-8"))
    native_jobs = {job["case"]: job for job in native["jobs"]}
    jobs = {job["case"]: job for job in manifest["jobs"]}
    results = [verify_job(jobs[case], native_jobs) for case in CASES]
    report = render_report(results)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=PROJECT_DIR, delete=False) as output:
        temporary_path = Path(output.name)
        output.write(report)
        output.flush()
        os.fsync(output.fileno())
    temporary_path.replace(REPORT_PATH)
    verification = {"complete": all(not result["failures"] and not result["pending"]
                                      and not result["metric_mismatches"] for result in results),
                    "jobs": results}
    (PROJECT_DIR / "failure_render_verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(report.splitlines()[2])
    print(f"Wrote {REPORT_PATH}")


if __name__ == "__main__":
    main()
