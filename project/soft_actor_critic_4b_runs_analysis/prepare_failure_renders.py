#!/usr/bin/env python3
"""Prepare isolated replay CSVs and a render manifest from native evaluations."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import shutil
import tempfile

from pufferlib.ocean.evaluation_utils.evaluation_utils import select_render_rows


PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parents[1]
NATIVE_MANIFEST = PROJECT_DIR / "native_manifest.json"
OUTPUT_DIR = PROJECT_DIR / "failure_renders"
OUTPUT_MANIFEST = PROJECT_DIR / "render_manifest.json"
RENDER_JOBS = (
    ("ppo", "render_ppo", "0"),
    ("pql_official2", "render_pql_official2", "1"),
    ("nstep4_native10", "render_nstep4", "4"),
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    if OUTPUT_MANIFEST.exists() or OUTPUT_DIR.exists():
        raise FileExistsError(f"Refusing to overwrite {OUTPUT_MANIFEST} or {OUTPUT_DIR}")
    with NATIVE_MANIFEST.open(encoding="utf-8") as source:
        native_manifest = json.load(source)
    jobs_by_case = {job["case"]: job for job in native_manifest["jobs"]}

    prepared = []
    for source_case, render_case, gpu in RENDER_JOBS:
        job = jobs_by_case[source_case]
        source_directory = PROJECT_DIR / "results" / source_case
        source_csvs = sorted(source_directory.glob("carla/*/episode_metrics.csv"))
        if len(source_csvs) != 1:
            raise RuntimeError(f"Expected exactly one episode_metrics.csv for {source_case}, found {len(source_csvs)}")
        source_csv = source_csvs[0].resolve()
        source_resolved = source_csv.with_name("resolved_benchmark.yaml")
        if not source_resolved.is_file():
            raise FileNotFoundError(source_resolved)
        selected_count = len(select_render_rows(str(source_csv), "all_infractions"))
        if selected_count < 10:
            raise ValueError(f"{source_case} has only {selected_count} all-infractions rows; need at least 10")

        checkpoint = Path(job["checkpoint"]).resolve()
        catalog = Path(job["catalog"]).resolve()
        if not checkpoint.is_file() or not catalog.is_file():
            raise FileNotFoundError(f"Missing checkpoint or catalog for {source_case}")
        checkpoint_hash = sha256(checkpoint)
        catalog_hash = sha256(catalog)
        if checkpoint_hash != job["checkpoint_sha256"] or catalog_hash != job["catalog_sha256"]:
            raise ValueError(f"Native manifest hash mismatch for {source_case}")
        prepared.append({
            "source_case": source_case,
            "render_case": render_case,
            "gpu": gpu,
            "job": job,
            "source_csv": source_csv,
            "source_resolved": source_resolved.resolve(),
            "source_csv_sha256": sha256(source_csv),
            "source_resolved_sha256": sha256(source_resolved),
            "selected_count": selected_count,
            "checkpoint": checkpoint,
            "catalog": catalog,
            "checkpoint_sha256": checkpoint_hash,
            "catalog_sha256": catalog_hash,
        })

    with tempfile.TemporaryDirectory(prefix=".failure_renders.", dir=PROJECT_DIR) as temporary:
        staged_root = Path(temporary) / "failure_renders"
        staged_root.mkdir()
        manifest_jobs = []
        for item in prepared:
            source_case = item["source_case"]
            case_directory = staged_root / source_case
            case_directory.mkdir()
            copied_csv = case_directory / "episode_metrics.csv"
            copied_resolved = case_directory / "resolved_benchmark.yaml"
            shutil.copy2(item["source_csv"], copied_csv)
            shutil.copy2(item["source_resolved"], copied_resolved)
            job = item["job"]
            manifest_jobs.append({
                "case": item["render_case"],
                "algorithm": job["algorithm"],
                "checkpoint": str(item["checkpoint"]),
                "checkpoint_sha256": item["checkpoint_sha256"],
                "catalog": str(item["catalog"]),
                "catalog_sha256": item["catalog_sha256"],
                "gpu": item["gpu"],
                "failure_replay_csv": str((OUTPUT_DIR / source_case / copied_csv.name).resolve()),
                "failure_replay_csv_sha256": sha256(copied_csv),
                "render_filter": "all_infractions",
                "max_rendered_failures": 10,
                "capture_observations": True,
                "source_case": source_case,
                "source_episode_metrics_csv": str(item["source_csv"]),
                "source_episode_metrics_csv_sha256": item["source_csv_sha256"],
                "source_resolved_benchmark": str(item["source_resolved"]),
                "source_resolved_benchmark_sha256": item["source_resolved_sha256"],
                "resolved_benchmark_sha256": sha256(copied_resolved),
                "resolved_benchmark_provenance": "byte-for-byte copy of source evaluation's resolved_benchmark.yaml",
                "all_infractions_selected_rows": item["selected_count"],
            })
        manifest = {
            "created_from": str(NATIVE_MANIFEST),
            "render_filter": "all_infractions",
            "jobs": manifest_jobs,
        }
        staged_manifest = Path(temporary) / "render_manifest.json"
        staged_manifest.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        staged_root.rename(OUTPUT_DIR)
        with OUTPUT_MANIFEST.open("x", encoding="utf-8") as output:
            output.write(staged_manifest.read_text(encoding="utf-8"))

    for item in prepared:
        print(f"{item['render_case']}: {item['selected_count']} selected episodes; GPU {item['gpu']}")
    print(f"Wrote {OUTPUT_MANIFEST}")


if __name__ == "__main__":
    main()
