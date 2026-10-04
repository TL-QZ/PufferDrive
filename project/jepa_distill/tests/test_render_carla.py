"""Preflight and no-match behavior for Condition B CARLA failure rendering."""

from __future__ import annotations

import csv
import hashlib
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml

from project.jepa_distill import render_carla


FAILURE_COLUMNS = (
    "collision_rate",
    "at_fault_collision_rate",
    "offroad_rate",
    "red_light_violation_rate",
)


def _episode_rows():
    rows = []
    for row_idx, failure_column in enumerate((*FAILURE_COLUMNS, None)):
        row = {
            "map_name": f"Town{row_idx + 1}",
            "seed": str(100 + row_idx),
            "agents_per_batch": "6000",
        }
        row.update({column: "0.0" for column in FAILURE_COLUMNS})
        if failure_column is not None:
            row[failure_column] = "1.0"
        rows.append(row)
    return rows


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(
            output_file,
            fieldnames=("map_name", "seed", "agents_per_batch", *FAILURE_COLUMNS),
        )
        writer.writeheader()
        writer.writerows(rows)


def _make_eval_input(
    root: Path,
    *,
    execution_k: int = 1,
    layout: str = "sequential",
    protocol_k: int | None = None,
    rows=None,
):
    episode_rows = _episode_rows() if rows is None else rows
    if layout == "sequential":
        source_run_dir = root / f"k{execution_k}"
    elif layout == "parallel":
        source_run_dir = root / f"run_k{execution_k}" / f"k{execution_k}"
    else:
        raise AssertionError(f"Unknown fixture layout: {layout}")
    evaluation_dir = (
        source_run_dir
        / "eval"
        / f"carla_act_k{execution_k}"
        / "standalone"
    )
    evaluation_dir.mkdir(parents=True)
    checkpoint_path = root / "student.pt"
    checkpoint_path.write_bytes(b"synthetic student checkpoint")
    map_dir = root / "maps"
    map_dir.mkdir()
    map_names = sorted({Path(row["map_name"]).stem for row in episode_rows})
    for map_name in map_names:
        (map_dir / f"{map_name}.bin").write_bytes(b"fixture map")

    recorded_k = execution_k if protocol_k is None else protocol_k
    protocol = {
        "checkpoint": str(checkpoint_path),
        "training_dt": 0.3,
        "evaluation_dt": 0.1,
        "model_chunk_length": 4,
        "execution_k": recorded_k,
        "hold_steps_per_slot": 1 if recorded_k == 1 else 3,
        "replan_interval_steps": 1 if recorded_k == 1 else 3 * recorded_k,
        "slot_schedule": (
            "slot 0 every evaluation step"
            if recorded_k == 1
            else "each slot held for 3 evaluation steps"
        ),
        "seed": 42,
        "num_scenarios": len(episode_rows),
        "scenario_length_steps": 6000,
        "num_agents": 300,
        "num_workers": 20,
        "benchmark": "carla",
        "action_selection": "mean",
        "action_type": "continuous",
    }
    (source_run_dir / "protocol.json").write_text(
        json.dumps(protocol), encoding="utf-8"
    )

    resolved_benchmark = {
        "benchmark_config": "/unused/shared/catalog.yaml",
        "checkpoint_config": None,
        "benchmark": {
            "name": "carla",
            "seed": 42,
            "num_scenarios": len(episode_rows),
            "env": {
                "dt": 0.1,
                "scenario_length": 6000,
                "map_dir": str(map_dir),
                "num_maps": len(map_names),
                "action_type": "continuous",
            },
        },
        "args": {
            "env_name": "drive",
            "env": {
                "dt": 0.1,
                "scenario_length": 6000,
                "map_dir": str(map_dir),
                "num_maps": len(map_names),
                "action_type": "continuous",
                "max_agents_per_env": 300,
            },
            "train": {"amp": True, "precision": "bfloat16", "compile": False},
            "eval": {
                "action_selection": "mean",
                "benchmarks": ["carla"],
                "num_agents": 300,
                "render_scenarios": False,
            },
            "vec": {"num_envs": 20},
        },
    }
    (evaluation_dir / "resolved_benchmark.yaml").write_text(
        yaml.safe_dump(resolved_benchmark, sort_keys=False), encoding="utf-8"
    )
    metrics_csv = evaluation_dir / "episode_metrics.csv"
    _write_csv(metrics_csv, episode_rows)
    return source_run_dir, metrics_csv, evaluation_dir / "resolved_benchmark.yaml", checkpoint_path


@pytest.mark.parametrize("layout", ["sequential", "parallel"])
def test_preflight_finds_expected_k_inputs_in_both_evaluation_layouts(tmp_path, layout):
    source_run_dir, metrics_csv, resolved_path, checkpoint_path = _make_eval_input(
        tmp_path / "evaluation", execution_k=3, layout=layout
    )
    output_dir = tmp_path / "renders"

    [plan] = render_carla.preflight_render(
        tmp_path / "evaluation",
        output_dir,
        execution_ks=(3,),
        failure_mode="all_infractions",
        max_failures=10,
    )

    assert Path(plan["source_run_dir"]) == source_run_dir
    assert Path(plan["metrics_csv"]) == metrics_csv
    assert Path(plan["resolved_benchmark_path"]) == resolved_path
    assert Path(plan["checkpoint_path"]) == checkpoint_path
    assert plan["execution_k"] == 3
    assert plan["recorded_agents_per_batch"] == 6000
    assert plan["match_count"] == 4
    assert Path(plan["output_dir"]) == output_dir / "k3"
    assert not output_dir.exists()


@pytest.mark.parametrize(
    ("failure_mode", "expected_map"),
    [
        ("collision", "Town1"),
        ("at_fault_collision", "Town2"),
        ("offroad", "Town3"),
        ("red_light", "Town4"),
    ],
)
def test_preflight_selects_each_baseline_failure_filter(tmp_path, failure_mode, expected_map):
    _make_eval_input(tmp_path / "evaluation")

    [plan] = render_carla.preflight_render(
        tmp_path / "evaluation",
        tmp_path / "renders",
        execution_ks=(1,),
        failure_mode=failure_mode,
        max_failures=10,
    )

    assert plan["match_count"] == 1
    assert plan["selected_rows"][0]["map_name"] == expected_map


def test_preflight_applies_limit_and_unlimited_failure_selection(tmp_path):
    _make_eval_input(tmp_path / "evaluation")

    [limited] = render_carla.preflight_render(
        tmp_path / "evaluation",
        tmp_path / "limited",
        execution_ks=(1,),
        failure_mode="all_infractions",
        max_failures=2,
    )
    [unlimited] = render_carla.preflight_render(
        tmp_path / "evaluation",
        tmp_path / "unlimited",
        execution_ks=(1,),
        failure_mode="all_infractions",
        max_failures=None,
    )

    assert [row["map_name"] for row in limited["selected_rows"]] == ["Town1", "Town2"]
    assert limited["match_count"] == 2
    assert [row["map_name"] for row in unlimited["selected_rows"]] == [
        "Town1",
        "Town2",
        "Town3",
        "Town4",
    ]
    assert unlimited["match_count"] == 4


def test_preflight_accepts_zero_matches_without_creating_outputs(tmp_path):
    no_failure_rows = _episode_rows()[-1:]
    _make_eval_input(tmp_path / "evaluation", rows=no_failure_rows)
    output_dir = tmp_path / "renders"

    [plan] = render_carla.preflight_render(
        tmp_path / "evaluation",
        output_dir,
        execution_ks=(1,),
        failure_mode="all_infractions",
        max_failures=10,
    )

    assert plan["match_count"] == 0
    assert plan["selected_rows"] == []
    assert not output_dir.exists()


def test_zero_match_run_copies_provenance_without_loading_student(tmp_path, monkeypatch):
    source_run_dir, metrics_csv, resolved_path, checkpoint_path = _make_eval_input(
        tmp_path / "evaluation", rows=_episode_rows()[-1:]
    )
    protocol_path = source_run_dir / "protocol.json"
    source_files = [protocol_path, metrics_csv, resolved_path, checkpoint_path]
    before_hashes = {path: _sha256(path) for path in source_files}
    output_dir = tmp_path / "renders"

    def unexpected_model_load(*_args, **_kwargs):
        raise AssertionError("a no-match render must not load student weights or initialize CUDA")

    monkeypatch.setattr(render_carla, "_load_student_checkpoint", unexpected_model_load)
    assert render_carla.main(
        [
            "--eval-root",
            str(tmp_path / "evaluation"),
            "--output-dir",
            str(output_dir),
            "--k",
            "1",
        ]
    ) == 0

    render_output = output_dir / "k1"
    assert (render_output / "source_protocol.json").read_bytes() == protocol_path.read_bytes()
    assert (render_output / "source_episode_metrics.csv").read_bytes() == metrics_csv.read_bytes()
    assert (render_output / "source_resolved_benchmark.yaml").read_bytes() == resolved_path.read_bytes()
    assert (render_output / "selected_failures.csv").is_file()
    report = json.loads((render_output / "render_report.json").read_text(encoding="utf-8"))
    assert report["status"] == "no_matching_failures"
    assert report["selected_scenarios"] == 0
    assert report["html_index"] is None
    assert not (render_output / "failures").exists()
    assert {path: _sha256(path) for path in source_files} == before_hashes


def test_preflight_rejects_missing_and_ambiguous_metrics(tmp_path):
    source_run_dir, metrics_csv, _, _ = _make_eval_input(tmp_path / "evaluation")
    metrics_csv.unlink()
    with pytest.raises((FileNotFoundError, ValueError), match="CSV|metrics|episode"):
        render_carla.preflight_render(
            tmp_path / "evaluation",
            tmp_path / "missing_output",
            execution_ks=(1,),
            failure_mode="all_infractions",
            max_failures=10,
        )

    _write_csv(metrics_csv, _episode_rows())
    duplicate_csv = source_run_dir / "extra" / "episode_metrics.csv"
    _write_csv(duplicate_csv, _episode_rows())
    with pytest.raises((FileNotFoundError, ValueError), match="multiple|ambiguous|exactly one|CSV"):
        render_carla.preflight_render(
            tmp_path / "evaluation",
            tmp_path / "ambiguous_output",
            execution_ks=(1,),
            failure_mode="all_infractions",
            max_failures=10,
        )


def test_preflight_rejects_protocol_execution_k_mismatch(tmp_path):
    _make_eval_input(tmp_path / "evaluation", execution_k=1, protocol_k=4)

    with pytest.raises(ValueError, match="k|execution_k|protocol"):
        render_carla.preflight_render(
            tmp_path / "evaluation",
            tmp_path / "renders",
            execution_ks=(1,),
            failure_mode="all_infractions",
            max_failures=10,
        )


def test_preflight_does_not_load_checkpoint_or_modify_sources(tmp_path, monkeypatch):
    _, metrics_csv, resolved_path, checkpoint_path = _make_eval_input(
        tmp_path / "evaluation", execution_k=2
    )
    sources = [metrics_csv, resolved_path, checkpoint_path, metrics_csv.parents[3] / "protocol.json"]
    before_hashes = {path: _sha256(path) for path in sources}
    output_dir = tmp_path / "renders"

    def unexpected_model_load(*_args, **_kwargs):
        raise AssertionError("preflight must not load model weights or initialize CUDA")

    monkeypatch.setattr(render_carla, "_load_student_checkpoint", unexpected_model_load, raising=False)
    plans = render_carla.preflight_render(
        tmp_path / "evaluation",
        output_dir,
        execution_ks=(2,),
        failure_mode="all_infractions",
        max_failures=10,
        device="cuda:0",
    )

    assert len(plans) == 1
    assert not output_dir.exists()
    assert {path: _sha256(path) for path in sources} == before_hashes


def test_cli_dry_run_prints_preflight_without_loading_or_writing(tmp_path, monkeypatch, capsys):
    import torch

    _, metrics_csv, resolved_path, checkpoint_path = _make_eval_input(
        tmp_path / "evaluation", execution_k=4
    )
    output_dir = tmp_path / "renders"
    before_hashes = {
        path: _sha256(path)
        for path in (
            metrics_csv,
            resolved_path,
            checkpoint_path,
            metrics_csv.parents[3] / "protocol.json",
        )
    }

    def unexpected_model_load(*_args, **_kwargs):
        raise AssertionError("CLI dry-run must not load student weights or initialize CUDA")

    def unexpected_cuda_init(*_args, **_kwargs):
        raise AssertionError("CLI dry-run must not initialize CUDA")

    monkeypatch.setattr(render_carla, "_load_student_checkpoint", unexpected_model_load)
    monkeypatch.setattr(torch.cuda, "_lazy_init", unexpected_cuda_init)
    assert render_carla.main(
        [
            "--eval-root",
            str(tmp_path / "evaluation"),
            "--output-dir",
            str(output_dir),
            "--k",
            "4",
            "--dry-run",
        ]
    ) == 0

    dry_run_report = json.loads(capsys.readouterr().out)
    assert dry_run_report["dry_run"] is True
    assert dry_run_report["runs"][0]["execution_k"] == 4
    assert dry_run_report["runs"][0]["selected_scenarios"] == 4
    assert dry_run_report["runs"][0]["output_dir"] == str(output_dir / "k4")
    assert not output_dir.exists()
    assert {path: _sha256(path) for path in before_hashes} == before_hashes


def test_failure_replay_waves_pad_recorded_batch_and_reset_chunk_cache(tmp_path, monkeypatch):
    import numpy as np
    import torch
    from torch import nn

    from pufferlib import pufferl, spaces
    from project.jepa_distill.evaluate_carla import ChunkPolicyAdapter

    _, metrics_csv, _, _ = _make_eval_input(tmp_path / "evaluation")
    map_dir = metrics_csv.parents[4] / "maps"
    vector_instances = []

    class RecordingStudent(nn.Module):
        def __init__(self):
            super().__init__()
            self.inference_inputs = []

        def forward(self, observations):
            self.inference_inputs.append(observations.detach().clone())
            logits = torch.zeros(
                (observations.shape[0], 4, 12),
                dtype=observations.dtype,
                device=observations.device,
            )
            logits[:, :, 3] = 1.0
            return logits

    class FakeReplayCapture:
        captures = []
        replay_offsets = []

        def __init__(self, *args, **kwargs):
            self.pending_count = 0
            self.capture_shapes = []
            self.captures.append(self)

        def capture_frame(self, obs, policy_obs_tensor, *_args):
            self.capture_shapes.append(tuple(policy_obs_tensor.shape))

        def queue_replay(self, _summary, episode_id):
            self.replay_offsets.append(episode_id)

        def write_pending(self):
            return None

    class FakeVecEnv:
        def __init__(self, wave_index):
            self.agents_per_batch = 2
            self.single_action_space = spaces.Box(-1.0, 1.0, shape=(2,), dtype=np.float32)
            self.action_space = spaces.Box(
                low=np.full((2, 2), -1.0, dtype=np.float32),
                high=np.full((2, 2), 1.0, dtype=np.float32),
                dtype=np.float32,
            )
            self.initial_value = float(1 + 10 * wave_index)
            self.step_count = 0
            self.closed = False

        def reset(self, _seed):
            observations = np.full((2, 3), self.initial_value, dtype=np.float32)
            return observations, {}

        def step(self, actions):
            assert actions.shape == (2, 2)
            self.step_count += 1
            observations = np.full(
                (2, 3), self.initial_value + self.step_count, dtype=np.float32
            )
            if self.step_count == 1:
                terminals = np.asarray([True, False])
                infos = [[], []]
            else:
                terminals = np.asarray([False, False])
                infos = [
                    [{"summary_type": "evaluation_episode", "scenario_id": self.initial_value}],
                    [{"summary_type": "evaluation_episode", "scenario_id": self.initial_value + 1}],
                ]
            truncations = np.asarray([False, False])
            rewards = np.zeros(2, dtype=np.float32)
            return observations, rewards, terminals, truncations, infos

        def close(self):
            self.closed = True

    def fake_make(*_args, **_kwargs):
        vecenv = FakeVecEnv(len(vector_instances))
        vector_instances.append(vecenv)
        return vecenv

    original_import_module = importlib.import_module

    def fake_import_module(module_name, *args, **kwargs):
        if module_name == "pufferlib.ocean":
            return SimpleNamespace(env_creator=lambda _env_name: object())
        return original_import_module(module_name, *args, **kwargs)

    monkeypatch.setattr(pufferl.importlib, "import_module", fake_import_module)
    monkeypatch.setattr(pufferl.pufferlib.vector, "make", fake_make)
    monkeypatch.setattr(pufferl.drive_eval_replay, "EvalReplayCapture", FakeReplayCapture)
    monkeypatch.setattr(
        pufferl.drive_eval_replay,
        "_render_eval_replays",
        lambda _summaries, _output_dir: None,
    )
    monkeypatch.setattr(
        pufferl.drive_benchmark,
        "_write_eval_reports",
        lambda summaries, _output_dir, num_scenarios: {
            "episodes": len(summaries),
            "num_scenarios": num_scenarios,
        },
    )

    run_args = {
        "package": "ocean",
        "train": {
            "seed": 42,
            "device": "cpu",
            "amp": False,
            "compile": False,
            "use_rnn": False,
        },
        "vec": {"num_envs": 2, "seed": 42},
        "env": {
            "map_dir": str(map_dir),
            "max_agents_per_env": 300,
            "num_agents": 300,
            "scenario_length": 2,
        },
        "eval": {
            "render_filter": "all_infractions",
            "action_selection": "mean",
            "observation_replay_wave_size": 2,
            "observation_replay_writer_count": 2,
        },
    }
    student = RecordingStudent()
    policy = ChunkPolicyAdapter(student, execution_k=4)
    output_dir = tmp_path / "failure_analysis"

    result = pufferl._render_eval_failures(
        "drive",
        run_args,
        {"name": "carla"},
        str(metrics_csv),
        str(output_dir),
        policy,
        True,
        None,
        evaluation_policy_cache={"policy": policy},
    )

    assert result["summary"] == {"episodes": 4, "num_scenarios": 4}
    assert len(vector_instances) == 2
    assert all(vecenv.closed for vecenv in vector_instances)
    assert [tuple(batch.shape) for batch in student.inference_inputs] == [
        (6000, 3),
        (1, 3),
        (6000, 3),
        (1, 3),
    ]
    # The active vector batch occupies the front rows; replay inference retains
    # the original 6000-row policy shape and zero-pads unused rows.
    torch.testing.assert_close(
        student.inference_inputs[0][2:], torch.zeros((5998, 3), dtype=torch.float32)
    )
    torch.testing.assert_close(
        student.inference_inputs[2][2:], torch.zeros((5998, 3), dtype=torch.float32)
    )
    assert student.inference_inputs[0][0, 0].item() == 1.0
    assert student.inference_inputs[1][0, 0].item() == 2.0
    assert student.inference_inputs[2][0, 0].item() == 11.0
    assert student.inference_inputs[3][0, 0].item() == 12.0
    assert [capture.capture_shapes for capture in FakeReplayCapture.captures] == [
        [(6000, 3), (6000, 3)],
        [(6000, 3), (6000, 3)],
    ]
    assert [summary["agents_per_batch"] for summary in result["episodes"]] == [
        6000,
        6000,
        6000,
        6000,
    ]


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
