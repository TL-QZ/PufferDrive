"""Freeze the four completed GPU-replay/filter runs into the native comparison."""

from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

import yaml

OUTPUT = Path(__file__).resolve().parent
ROOT = OUTPUT.parents[1]
FILTER_BATCH = "actor_critic_filter_20261112T163346Z"
RUNS = (
    ("tes_gpu4b", "sac_gpu_4b_comparison/runs/tes_gpu_4b_20261110", 0),
    ("pql_gpu4b", "sac_gpu_4b_comparison/runs/pql_gpu_4b_20261110", 1),
    ("tes_experience_filter", f"soft_actor_critic_experience_filtering/runs/{FILTER_BATCH}/tes_actor_critic_20261112T163346Z", 2),
    ("pql_experience_filter", f"soft_actor_critic_experience_filtering/runs/{FILTER_BATCH}/pql_actor_critic_20261112T163346Z", 3),
)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    manifest_path = OUTPUT / "native_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    existing = {job["case"] for job in manifest["jobs"]}
    prepared = []
    for case, relative_run, gpu in RUNS:
        if case in existing:
            raise ValueError(f"Already registered: {case}")
        trial = ROOT / "project" / relative_run / "trial_98"
        summary = json.loads((trial / "summary.json").read_text())
        if summary["status"] != "completed" or summary["exit_code"] != 0 or summary["agent_steps"] < 4_000_000_000:
            raise ValueError(f"Incomplete training: {trial}")
        source = trial / "learner"
        config = yaml.safe_load((source / "config.yaml").read_text())
        catalog_source = ROOT / config["eval"]["benchmark_config"]
        catalog = yaml.safe_load(catalog_source.read_text())
        benchmark = next(item for item in catalog["benchmarks"] if item["name"] == "carla")
        environment = {**catalog["env"], **benchmark["env"]}
        if (environment["min_goal_spacing"], environment["max_goal_spacing"], environment["dt"], environment["scenario_length"], benchmark["seed"], benchmark["num_scenarios"]) != (30, 30, 0.1, 6000, 42, 1000):
            raise ValueError(f"Unexpected evaluation protocol: {catalog_source}")
        filtering = config["train"].get("experience_filter", {})
        if "experience_filter" in case and (not filtering.get("enabled") or filtering.get("apply_to") != "actor_critic"):
            raise ValueError(f"Expected shared actor/critic filtering: {case}")
        prepared.append((case, gpu, trial, source, catalog_source, summary, filtering))

    added_at = datetime.now(timezone.utc).isoformat()
    for case, gpu, trial, source, catalog_source, summary, filtering in prepared:
        destination = OUTPUT / "checkpoints" / case
        destination.mkdir(parents=True, exist_ok=False)
        for filename in ("final_model.pt", "config.yaml"):
            shutil.copy2(source / filename, destination / filename)
        catalog_path = OUTPUT / "native_catalogs" / f"{case}.yaml"
        if catalog_path.exists():
            raise FileExistsError(catalog_path)
        shutil.copy2(catalog_source, catalog_path)
        manifest["jobs"].append(dict(
            case=case, run_id=f"{trial.parent.name}_trial_98", algorithm="SAC", gpu=gpu,
            source_checkpoint=str(source / "final_model.pt"), checkpoint=str(destination / "final_model.pt"),
            checkpoint_sha256=digest(destination / "final_model.pt"), config_sha256=digest(destination / "config.yaml"),
            catalog=str(catalog_path), catalog_source=str(catalog_source), catalog_sha256=digest(catalog_path),
            goal_spacing_m=30.0, selection="final_model", recovery=False, reuse_previous_result=False,
            training_agent_steps=summary["agent_steps"], training_summary=str(trial / "summary.json"),
            experience_filter=filtering, added_at=added_at,
        ))
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Added {len(prepared)} completed checkpoints; {len(manifest['jobs'])} total jobs.")


if __name__ == "__main__":
    main()
