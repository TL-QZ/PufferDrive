"""Check the launcher's mean-action contract after restoring checkpoint settings."""

import os
from pathlib import Path
import shlex
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

from omegaconf import OmegaConf
import pytest
import torch

from pufferlib import pufferl
from pufferlib.pytorch import sample_logits
from pufferlib.ocean.evaluation_utils import evaluation_utils as evaluation

REPO = Path(__file__).resolve().parents[3]
PROJECT = Path(__file__).resolve().parents[1]


def test_mean_actions_after_checkpoint_merge():
    run_root = REPO / "experiments/baseline_run_sync_2026-08-24/nuplan_selfplay_dt03"
    if len(list(run_root.glob("*_seed0/final_model.pt"))) != 1:
        pytest.skip("requires exactly one completed local self-play seed 0")
    preview = subprocess.run(
        [str(PROJECT / "eval/evaluate_nuplan_selfplay.sh"), "0"], cwd=REPO,
        env=dict(os.environ, CUDA_VISIBLE_DEVICES="1", DRY_RUN="1",
                 BENCHMARKS="carla,nuplan_single,nuplan_multi"),
        text=True, capture_output=True, check=True,
    )
    command = shlex.split(next(line.removeprefix("Command:") for line in preview.stdout.splitlines()
                              if line.startswith("Command:")))
    overrides = command[4:]
    with patch.object(sys, "argv", ["test", *overrides]):
        args = pufferl.load_config("puffer_drive")
    base, _ = evaluation.load_checkpoint_architecture(args)
    assert base["env"]["action_type"] == "discrete", "Fixture must exercise checkpoint restoration"
    common, benchmarks = evaluation.load_benchmark_config(args["eval"]["benchmark_config"], command[3])
    action_table = torch.tensor([[-1.0, 0.0], [1.0, 0.0]])
    for benchmark in benchmarks:
        resolved = evaluation.build_benchmark_args(base, benchmark, common)
        resolved = OmegaConf.to_container(OmegaConf.merge(
            OmegaConf.create(dict(resolved)), OmegaConf.from_dotlist(overrides)), resolve=True)
        assert resolved["policy"]["action_type"] == "discrete"
        assert resolved["eval"]["action_selection"] == "mean"
        policy = SimpleNamespace(is_continuous=False,
                                 discrete_probs_to_continuous_mean=lambda probabilities: probabilities @ action_table)
        _, _, _, controls = sample_logits(
            torch.zeros(2, 2), action_selection=resolved["eval"]["action_selection"],
            env_continuous=resolved["env"]["action_type"] == "continuous", policy=policy)
        torch.testing.assert_close(controls, torch.zeros(2, 2))
