"""Condition B teacher resolution and strict checkpoint loading."""

from copy import deepcopy
from pathlib import Path

import pytest
import torch
import yaml

from project.jepa_distill.teacher import load_teacher, observation_layout_from_config, resolve_teacher_config

torch.set_num_threads(1)


REPO_ROOT = Path(__file__).resolve().parents[3]
CONDITION_CONFIG = REPO_ROOT / "project/jepa_distill/config/condition_b.yaml"
CHECKPOINT = (
    REPO_ROOT
    / "experiments/baseline_run_sync_2026-08-24/"
    / "baseline_run_sync_2026-08-24_2026-08-26_16-23-18_seed0/final_model.pt"
)


def load_condition_config():
    with CONDITION_CONFIG.open(encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def test_resolve_returns_full_saved_args_with_clean_effective_layout():
    config = load_condition_config()
    original = deepcopy(config)
    resolved = resolve_teacher_config(config)

    assert "env" in resolved and "policy" in resolved and "train" in resolved
    assert resolved["env"]["obs_dropout_lane"] == 0.0
    assert resolved["env"]["obs_dropout_boundary"] == 0.0
    layout = observation_layout_from_config(resolved)
    assert layout["obs_slots_lane_kept"] == resolved["env"]["obs_slots_lane_n"]
    assert layout["obs_slots_boundary_kept"] == resolved["env"]["obs_slots_boundary_n"]
    assert config == original
    assert "obs_slots_lane_kept" not in resolved["env"]


@pytest.mark.skipif(not CHECKPOINT.is_file(), reason="baseline checkpoint is not available")
def test_load_teacher_strict_checkpoint_without_live_environment():
    config = load_condition_config()
    resolved = resolve_teacher_config(config)
    teacher = load_teacher(config, resolved_config=resolved, device="cpu")

    assert teacher.training is False
    assert all(not parameter.requires_grad for parameter in teacher.parameters())
    assert teacher.actor_backbone.out_dim == 1024
    assert teacher.condition_b_observation_layout["observation_dim"] == 1292
    # Resume validates manifests before any new collection can populate caches.
    import hashlib
    with CHECKPOINT.open('rb') as checkpoint_stream:
        expected_hash = hashlib.file_digest(checkpoint_stream, 'sha256').hexdigest()
    assert teacher.condition_b_checkpoint_sha256 == expected_hash
    assert Path(teacher.condition_b_checkpoint_path) == CHECKPOINT
