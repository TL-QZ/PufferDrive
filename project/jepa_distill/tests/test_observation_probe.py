"""Focused safety and dry-run checks for the observation probe scaffold."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from project.jepa_distill.observation_probe.config import (
    DEFAULT_CONFIG,
    load_probe_config,
    validate_probe_config,
)
from project.jepa_distill.observation_probe.data import preflight_probe_data
from project.jepa_distill.observation_probe.train import preflight_probe, train_probe


_LAYOUT = {
    "boundary_features": 9,
    "context_dim": 26,
    "dtype": "float32",
    "ego_features": 10,
    "goal_dim": 9,
    "goal_features": 3,
    "lane_features": 9,
    "num_reward_coefs": 17,
    "obs_slots_boundary_kept": 50,
    "obs_slots_lane_kept": 70,
    "obs_slots_partners_n": 16,
    "obs_slots_traffic_controls_n": 4,
    "obs_valid_count_features": 4,
    "observation_dim": 1292,
    "partner_features": 9,
    "traffic_control_features": 7,
}
_ACTION_TABLE = [
    [-1.0, -1.0], [-1.0, 0.0], [-1.0, 1.0],
    [-0.2666666805744171, -1.0], [-0.2666666805744171, 0.0],
    [-0.2666666805744171, 1.0], [0.0, -1.0], [0.0, 0.0],
    [0.0, 1.0], [1.0, -1.0], [1.0, 0.0], [1.0, 1.0],
]
_ARRAY_LAYOUTS = {
    "observations": (np.dtype("float32"), lambda steps, slots, dim, classes: (steps + 1, slots, dim)),
    "controls": (np.dtype("float32"), lambda steps, slots, dim, classes: (steps, slots, 2)),
    "logits": (np.dtype("float32"), lambda steps, slots, dim, classes: (steps, slots, classes)),
    "transition_valid": (np.dtype("bool"), lambda steps, slots, dim, classes: (steps, slots)),
    "eligibility_mask": (np.dtype("bool"), lambda steps, slots, dim, classes: (steps, slots)),
    "terminated": (np.dtype("bool"), lambda steps, slots, dim, classes: (steps, slots)),
    "truncated": (np.dtype("bool"), lambda steps, slots, dim, classes: (steps, slots)),
    "endpoint_valid": (np.dtype("bool"), lambda steps, slots, dim, classes: (steps + 1, slots)),
    "generation": (np.dtype("int64"), lambda steps, slots, dim, classes: (steps + 1, slots)),
}


def _write_header(path: Path, dtype: np.dtype, shape: tuple[int, ...]) -> None:
    array = np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=shape)
    array.flush()
    del array


def _write_manifest(
    manifest_path: Path,
    *,
    round_idx: int,
    valid_window_count: int,
    observation_layout: dict | None = None,
    observation_dim: int = 1292,
) -> None:
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    steps, slots, action_classes = 20, 2, len(_ACTION_TABLE)
    layout = copy.deepcopy(observation_layout or _LAYOUT)
    layout["observation_dim"] = observation_dim
    descriptors = {}
    for name, (dtype, shape_fn) in _ARRAY_LAYOUTS.items():
        shape = shape_fn(steps, slots, observation_dim, action_classes)
        filename = f"{name}.npy"
        _write_header(manifest_path.parent / filename, dtype, shape)
        descriptors[name] = {
            "dtype": dtype.name,
            "path": filename,
            "shape": list(shape),
        }

    window_indices = np.arange(valid_window_count, dtype=np.int64)
    np.save(manifest_path.parent / "window_indices.npy", window_indices)
    candidate_count = steps * slots
    manifest = {
        "schema_version": 1,
        "dataset_format": "condition_b_streaming_npy_v1",
        "complete": True,
        "collection_round_idx": round_idx,
        "split": "train",
        "collection_mode": "teacher_driven_streaming",
        "action_selection": "sample",
        "chunk_length": 4,
        "num_action_classes": action_classes,
        "slots": slots,
        "observation_dim": observation_dim,
        "simulator_steps": steps,
        "observation_layout": layout,
        "action_layout": {
            "control_order": ["longitudinal", "lateral"],
            "normalized_controls": _ACTION_TABLE,
            "normalized_range": [-1.0, 1.0],
        },
        "teacher_checkpoint": "fixture-checkpoint.pt",
        "teacher_checkpoint_sha256": "fixture-sha256",
        "split_seed": 0,
        "effective_config": {
            "teacher_config": {
                "env": {
                    "action_type": "continuous",
                    "dt": 0.3,
                    "dynamics_model": "jerk",
                    "reward_conditioning": True,
                    "obs_norm_xy_offset_m": 200.0,
                    "obs_norm_goal_offset_m": 200.0,
                }
            }
        },
        "code_revision": None,
        "arrays": descriptors,
        "window_indices": {
            "dtype": "int64",
            "path": "window_indices.npy",
            "shape": [valid_window_count],
        },
        "collection_transition_count": steps * slots,
        "transition_count": steps * slots,
        "valid_window_count": valid_window_count,
        "candidate_window_count": candidate_count,
        "rejected_window_count": candidate_count - valid_window_count,
        "window_index_flattening": "time_index * slots + slot_index",
    }
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")


def _write_collection(root: Path) -> None:
    counts = {
        (0, 0): 3,
        (0, 1): 2,
        (1, 0): 4,
        (1, 1): 5,
    }
    for (round_idx, rank), count in counts.items():
        manifest_path = (
            root / f"rank_{rank:03d}" / "train" / f"round_{round_idx:04d}" / "manifest.json"
        )
        _write_manifest(
            manifest_path,
            round_idx=round_idx,
            valid_window_count=count,
        )


def _probe_config(root: Path, extra_overrides: list[str] | None = None) -> dict:
    overrides = [
        "checkpoint=null",
        "data.mode=cached",
        f"data.collection_root={json.dumps(str(root))}",
        "data.collection_rounds=[0,1]",
        "data.source_ranks=[0,1]",
        "training.batch_size=null",
        "training.epochs_per_collection=null",
        "training.microbatch_size=null",
        "training.world_size=null",
    ]
    overrides.extend(extra_overrides or [])
    return load_probe_config(DEFAULT_CONFIG, overrides)


def test_preflight_checks_real_format_headers_and_pools_two_ranks_two_rounds(tmp_path):
    collection_root = tmp_path / "collections"
    _write_collection(collection_root)

    report = preflight_probe_data(_probe_config(collection_root))

    assert len(report["selected_manifests"]) == 4
    assert report["selected_rounds"] == [0, 1]
    assert report["source_ranks"] == [0, 1]
    assert [entry["pooled_valid_window_count"] for entry in report["per_round"]] == [5, 9]
    assert [entry["rank_window_counts"] for entry in report["per_round"]] == [
        {"rank_000": 3, "rank_001": 2},
        {"rank_000": 4, "rank_001": 5},
    ]
    assert report["compatibility"]["observation_dim"] == 1292
    assert report["array_audit"].startswith("NPY headers")
    assert report["observation_recipe_compatibility"] == "matched_across_selected_manifests"


@pytest.mark.parametrize(
    "invalid_case",
    [
        "missing_rank",
        "incomplete",
        "layout_mismatch",
        "path_traversal",
        "header_shape_mismatch",
        "unsupported_schema",
    ],
)
def test_preflight_rejects_invalid_selected_manifest_data(tmp_path, invalid_case):
    collection_root = tmp_path / "collections"
    _write_collection(collection_root)
    bad_manifest = (
        collection_root / "rank_001" / "train" / "round_0000" / "manifest.json"
    )

    if invalid_case == "missing_rank":
        bad_manifest.unlink()
    else:
        manifest = json.loads(bad_manifest.read_text(encoding="utf-8"))
        if invalid_case == "incomplete":
            manifest["complete"] = False
        elif invalid_case == "unsupported_schema":
            manifest["schema_version"] = 999
        elif invalid_case == "layout_mismatch":
            observation_dim = manifest["observation_dim"] + 1
            manifest["observation_dim"] = observation_dim
            manifest["observation_layout"]["observation_dim"] = observation_dim
            descriptor = manifest["arrays"]["observations"]
            shape = list(descriptor["shape"])
            shape[-1] = observation_dim
            descriptor["shape"] = shape
            _write_header(
                bad_manifest.parent / descriptor["path"], np.dtype("float32"), tuple(shape)
            )
        elif invalid_case == "path_traversal":
            escape_path = tmp_path / "outside.npy"
            np.save(escape_path, np.zeros((1,), dtype=np.float32))
            manifest["arrays"]["observations"]["path"] = "../../../../outside.npy"
        elif invalid_case == "header_shape_mismatch":
            manifest["arrays"]["observations"]["shape"][0] += 1
        bad_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises((FileNotFoundError, ValueError)):
        preflight_probe_data(_probe_config(collection_root))


def test_dry_run_accepts_unset_budgets_and_reports_nominal_steps_when_resolved(tmp_path):
    collection_root = tmp_path / "collections"
    _write_collection(collection_root)
    config = _probe_config(collection_root)

    unresolved_report = preflight_probe(config)

    assert unresolved_report["nominal_optimizer_steps"] is None
    assert unresolved_report["capped_optimizer_steps"] is None
    assert "training.batch_size" in unresolved_report["unresolved_settings"]
    assert "training.epochs_per_collection" in unresolved_report["unresolved_settings"]

    budgeted_config = _probe_config(
        collection_root,
        [
            "training.batch_size=4",
            "training.epochs_per_collection=3",
            "training.max_optimizer_steps=8",
        ],
    )
    budgeted_report = preflight_probe(budgeted_config)
    # ceil(5 / 4) + ceil(9 / 4), then three epochs; the cap applies globally.
    assert budgeted_report["nominal_optimizer_steps"] == 15
    assert budgeted_report["capped_optimizer_steps"] == 8


@pytest.mark.parametrize(
    ("recipe_key", "changed_value"),
    [("dt", 0.1), ("dynamics_model", "constant_accel"), ("obs_norm_xy_offset_m", 100.0)],
)
def test_preflight_rejects_mismatched_observation_recipe_metadata(
    tmp_path, recipe_key, changed_value
):
    collection_root = tmp_path / "collections"
    _write_collection(collection_root)
    bad_manifest = (
        collection_root / "rank_001" / "train" / "round_0000" / "manifest.json"
    )
    manifest = json.loads(bad_manifest.read_text(encoding="utf-8"))
    manifest["effective_config"]["teacher_config"]["env"][recipe_key] = changed_value
    bad_manifest.write_text(json.dumps(manifest), encoding="utf-8")

    with pytest.raises(ValueError, match="incompatible dt/dynamics/observation normalization"):
        preflight_probe_data(_probe_config(collection_root))


def test_missing_observation_recipe_metadata_is_reported_as_unverified(tmp_path):
    collection_root = tmp_path / "collections"
    _write_collection(collection_root)
    manifest_path = (
        collection_root / "rank_001" / "train" / "round_0000" / "manifest.json"
    )
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["effective_config"].pop("teacher_config")
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    report = preflight_probe_data(_probe_config(collection_root))

    assert report["observation_recipe_compatibility"] == "unverified_missing_manifest_metadata"


@pytest.mark.parametrize(
    "override",
    [
        "training.epochs_per_collection=0",
        "training.batch_size=-1",
        "training.max_optimizer_steps=0",
        "data.collection_rounds=[]",
        "rendering.presence_threshold=1.0",
    ],
)
def test_invalid_supplied_parameters_are_rejected(tmp_path, override):
    with pytest.raises(ValueError):
        _probe_config(tmp_path / "unused", [override])


def test_normal_launch_rejects_unresolved_config_without_creating_artifacts(tmp_path):
    collection_root = tmp_path / "collections"
    _write_collection(collection_root)
    config = _probe_config(collection_root)
    report = preflight_probe(config)
    before = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}

    with pytest.raises(ValueError, match="Unresolved required observation probe settings"):
        train_probe(config, report)

    after = {path.relative_to(tmp_path) for path in tmp_path.rglob("*")}
    assert after == before


def test_resolved_training_entrypoint_rejects_missing_checkpoint():
    config = load_probe_config(DEFAULT_CONFIG)
    config["checkpoint"] = "missing-probe-checkpoint.pt"
    config["data"]["validation_manifest"] = "validation-manifest"
    config["data"]["test_manifest"] = "test-manifest"
    config["training"].update(
        {
            "epochs_per_collection": 1,
            "batch_size": 4,
            "microbatch_size": 2,
            "learning_rate": 0.001,
            "world_size": 1,
            "device": "cpu",
            "run_id": "missing-checkpoint-test",
        }
    )
    config["matching"]["costs"] = {"position": 1.0}
    config["loss"]["group_weights"] = {"ego": 1.0}
    config["rendering"]["presence_threshold"] = 0.5
    assert validate_probe_config(config, require_resolved=True) == []

    with pytest.raises(FileNotFoundError):
        train_probe(config, {})


def test_cli_help_and_preflight_imports_do_not_initialize_runtime_resources():
    help_result = subprocess.run(
        [sys.executable, "-m", "project.jepa_distill.observation_probe.train", "--help"],
        check=False,
        capture_output=True,
        text=True,
    )
    assert help_result.returncode == 0, help_result.stderr
    assert "--dry-run" in help_result.stdout

    import_result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import sys; "
            "import project.jepa_distill.observation_probe.config; "
            "import project.jepa_distill.observation_probe.data; "
            "import project.jepa_distill.observation_probe.train; "
            "import torch; "
            "assert not torch.cuda.is_initialized(); "
            "assert not any(name == 'wandb' or name.startswith('wandb.') for name in sys.modules); "
            "assert not any(name == 'pufferlib' or name.startswith('pufferlib.') for name in sys.modules); "
            "assert not any(name == 'carla' or name.startswith('carla.') for name in sys.modules); "
            "assert not any(name == 'nuplan' or name.startswith('nuplan.') for name in sys.modules)",
        ],
        check=False,
        capture_output=True,
        text=True,
    )
    assert import_result.returncode == 0, import_result.stderr
