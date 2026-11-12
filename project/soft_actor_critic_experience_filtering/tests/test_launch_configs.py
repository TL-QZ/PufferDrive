from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import tempfile

import pytest
import yaml


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_DIR = REPO_ROOT / "project" / "soft_actor_critic_experience_filtering"
SOURCE_CONFIGS = REPO_ROOT / "project" / "sac_gpu_4b_comparison" / "configs"
EXPECTED_FILTER = {
    "enabled": True,
    "score": "twin_max_abs_td_error",
    "apply_to": "critic",
    "threshold_mode": "ewma_max",
    "ewma_beta": 0.25,
    "threshold_scale": 0.01,
}


def _load_yaml(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        value = yaml.safe_load(stream)
    assert isinstance(value, dict), path
    return value


def _remove_path(config: dict, *keys: str) -> None:
    value = config
    for key in keys[:-1]:
        value = value[key]
    value.pop(keys[-1])


def test_configs_match_source_except_isolated_identity_and_filter() -> None:
    for algorithm in ("tes", "pql"):
        source_dir = SOURCE_CONFIGS / algorithm
        local_dir = PROJECT_DIR / "configs" / algorithm

        source_tunable = _load_yaml(source_dir / "SAC_tunable_config.yaml")
        local_tunable = _load_yaml(local_dir / "SAC_tunable_config.yaml")
        source_filter = source_tunable["train"].pop("experience_filter", None)
        assert source_filter is None
        assert local_tunable["train"].pop("experience_filter") == EXPECTED_FILTER
        for config, is_local in ((source_tunable, False), (local_tunable, True)):
            config.pop("run_name")
            config.pop("wandb_group")
            _remove_path(config, "train", "data_dir")
        assert local_tunable == source_tunable

        source_fixed = _load_yaml(source_dir / "SAC_fixed_config.yaml")
        local_fixed = _load_yaml(local_dir / "SAC_fixed_config.yaml")
        source_fixed["eval"]["benchmark_config"] = "<local-catalog>"
        local_fixed["eval"]["benchmark_config"] = "<local-catalog>"
        assert local_fixed == source_fixed

        assert (local_dir / "SAC_specific_config.yaml").read_bytes() == (
            source_dir / "SAC_specific_config.yaml"
        ).read_bytes()
        assert (local_dir / "evaluation_benchmarks.yaml").read_bytes() == (
            source_dir / "evaluation_benchmarks.yaml"
        ).read_bytes()

        source_run = _load_yaml(source_dir / "SAC_run_config.yaml")
        local_run = _load_yaml(local_dir / "SAC_run_config.yaml")
        source_run["launcher"]["gpu_pairs"] = local_run["launcher"]["gpu_pairs"]
        assert local_run == source_run
        expected_gpu_group = [0, 1] if algorithm == "tes" else [2, 3]
        assert local_run["launcher"]["gpu_pairs"] == [expected_gpu_group]


def _run_dry_run(algorithm: str, output_dir: Path, *extra_args: str) -> tuple[dict, dict]:
    launcher = PROJECT_DIR / f"launch_{algorithm}.sh"
    result = subprocess.run(
        [str(launcher), "--dry-run", *extra_args, "--output-dir", str(output_dir)],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "WANDB_MODE": "online"},
    )
    assert "Dry run wrote 1 resolved config" in result.stdout
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    config_path = Path(manifest["trials"][97]["config_path"])
    resolved = _load_yaml(config_path)
    return manifest, resolved


@pytest.mark.parametrize(
    ("algorithm", "gpu_pair", "wandb_group"),
    [
        ("tes", [0, 1], "tes_experience_filtering_trial98"),
        ("pql", [2, 3], "pql_experience_filtering_trial98"),
    ],
)
def test_trial_98_dry_run_resolves_filter_budget_and_default_gpu_group(
    algorithm: str, gpu_pair: list[int], wandb_group: str
) -> None:
    with tempfile.TemporaryDirectory(prefix=f"experience-filter-{algorithm}-", dir="/tmp") as temp_dir:
        output_dir = Path(temp_dir) / "dry_run"
        manifest, resolved = _run_dry_run(algorithm, output_dir)

    assert manifest["dry_run"] is True
    assert manifest["selected_trial_ids"] == [98]
    assert manifest["launcher"]["gpu_pairs"] == [gpu_pair]
    assert manifest["launcher"]["no_wandb"] is False
    assert resolved["train"]["experience_filter"] == EXPECTED_FILTER
    assert resolved["train"]["total_timesteps"] == 4_000_000_000
    assert resolved["train"]["evaluation_interval_epochs"] == 2
    assert resolved["wandb"] is True
    assert resolved["wandb_group"] == wandb_group
    assert resolved["eval"]["benchmark_config"].endswith(
        f"soft_actor_critic_experience_filtering/configs/{algorithm}/evaluation_benchmarks.yaml"
    )
    if algorithm == "pql":
        assert resolved["train"]["return_estimator"] == "pql"
        assert resolved["train"]["pql_horizon"] == 128
        assert resolved["train"]["n_steps"] == 1


def test_launcher_gpu_override_and_no_wandb_are_forwarded() -> None:
    with tempfile.TemporaryDirectory(prefix="experience-filter-override-", dir="/tmp") as temp_dir:
        output_dir = Path(temp_dir) / "dry_run"
        manifest, resolved = _run_dry_run(
            "tes", output_dir, "--gpu-groups", "4,5", "--no-wandb"
        )

    assert manifest["launcher"]["gpu_pairs"] == [[4, 5]]
    assert manifest["launcher"]["no_wandb"] is True
    assert resolved["wandb"] is False
    assert resolved["neptune"] is False
    assert resolved["tb"] is False
