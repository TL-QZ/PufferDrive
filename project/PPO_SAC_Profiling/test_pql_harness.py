"""Focused CLI, config, and target-audit checks for the PQL profile harness."""

from types import SimpleNamespace
from pathlib import Path
import json

import pytest
import torch

from project.PPO_SAC_Profiling import profile_training as harness
from project.PPO_SAC_Profiling import analyze_runs


def test_pql_cli_overrides_and_forces_raw_one_step_rows(tmp_path):
    args = harness.parse_args([
        "--algorithm", "sac",
        "--output-dir", str(tmp_path / "pql-run"),
        "--device", "cpu",
        "--return-estimator", "pql",
        "--pql-horizon", "16",
        "--pql-lambda", "0.91",
    ])

    config, _source = harness.prepare_config(args, args.output_dir)

    assert config["train"]["return_estimator"] == "pql"
    assert config["train"]["pql_horizon"] == 16
    assert config["train"]["pql_lambda"] == 0.91
    assert config["train"]["n_steps"] == 1


def test_omitted_estimator_preserves_saved_source_selection(tmp_path):
    source, _source_path = harness.load_source_config("sac")
    args = harness.parse_args([
        "--algorithm", "sac",
        "--output-dir", str(tmp_path / "source-run"),
        "--device", "cpu",
    ])

    config, _resolved_source = harness.prepare_config(args, args.output_dir)

    assert config["train"]["return_estimator"] == source["train"].get("return_estimator", "n_step")
    assert config["train"].get("n_steps") == source["train"].get("n_steps")


@pytest.mark.parametrize("preset", ["pql_cpu_smoke.yaml", "pql_gpu.yaml"])
def test_pql_presets_parse_with_expected_profiles(tmp_path, preset):
    preset_path = Path(__file__).resolve().parent / "configs" / preset
    args = harness.parse_args([
        "--profile-config", str(preset_path),
        "--output-dir", str(tmp_path / preset.removesuffix(".yaml")),
    ])

    assert args.algorithm == "sac"
    assert args.return_estimator == "pql"
    assert args.pql_lambda == 0.95


@pytest.mark.parametrize("extra, message", [
    (("--algorithm", "ppo", "--return-estimator", "pql"), "apply to SAC only"),
    (("--variant", "obs_only", "--return-estimator", "pql"), "does not support PQL"),
    (("--variant", "fixed_targets_actor_q", "--sac-keep-fraction", "0.5"), "requires --sac-filter-score"),
    (("--variant", "fixed_targets", "--sac-filter-score", "q_advantage"), "requires --variant fixed_targets_actor_q"),
    (("--pql-lambda", "nan", "--return-estimator", "pql"), "finite and in [0, 1]"),
    (("--sac-keep-fraction", "0", "--variant", "fixed_targets_actor_q"), "finite and in (0, 1]"),
])
def test_invalid_pql_and_filter_combinations_fail_before_setup(tmp_path, extra, message):
    base = ["--algorithm", "sac", "--output-dir", str(tmp_path / "invalid-run"), "--device", "cpu"]
    with pytest.raises(SystemExit):
        harness.parse_args(base + list(extra))


def test_pql_target_audit_reuses_complete_saved_segments_and_physical_rows():
    first_indices = torch.tensor([3, 2], dtype=torch.int64)
    first_offsets = torch.tensor([0, 1], dtype=torch.int64)
    second_indices = torch.tensor([1, 0], dtype=torch.int64)
    second_offsets = torch.tensor([0, 1], dtype=torch.int64)
    calls = []

    def prepare_pql_targets(indices, offsets):
        calls.append((indices.clone(), offsets.clone()))
        return indices.float() + offsets.float() * 10

    class ReplayStub:
        def __len__(self):
            return 4

    trainer = SimpleNamespace(
        config={"return_estimator": "pql"},
        replay=ReplayStub(),
        fixed_pql_sequence_batches=[
            (first_indices, first_offsets),
            (second_indices, second_offsets),
        ],
        fixed_cycle_targets=torch.arange(4, dtype=torch.float32),
        max_minibatch_size=2,
        device=torch.device("cpu"),
        _prepare_pql_targets=prepare_pql_targets,
    )

    audit = harness.audit_fixed_targets(trainer, requested_rows=2, torch=torch)

    assert len(calls) == 2
    torch.testing.assert_close(calls[0][0], first_indices, rtol=0, atol=0)
    torch.testing.assert_close(calls[0][1], first_offsets, rtol=0, atol=0)
    torch.testing.assert_close(calls[1][0], second_indices, rtol=0, atol=0)
    torch.testing.assert_close(calls[1][1], second_offsets, rtol=0, atol=0)
    assert audit["target_path"] == "saved_pql_sequence_batches_and_current_policy"
    assert audit["rows"] == 2
    assert audit["forward_rows_for_audit"] == 4
    assert audit["segment_batches_for_audit"] == 2
    assert audit["current_mean"] == 6.5
    assert audit["mean_absolute_difference"] == 5.0


def test_analyzer_accepts_pre_filter_summary_json(tmp_path):
    (tmp_path / "cycle_001.json").write_text(
        '{"cycle_index":1,"measurement_included":true,"global":'
        '{"collection_seconds":1.0,"optimization_seconds":2.0,"received_rows":8,'
        '"sampled_transitions":4,"loss_row_exposures":8,"peak_gpu_allocated_bytes":0},'
        '"ranks":[{"critic_loss_rows_local":4,"loss_row_exposures_local":8}]}'
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        '{"algorithm":"sac","variant":"stock","world_size":1,'
        '"profile_settings":{},"cycle_files":["cycle_001.json"]}'
    )

    row = analyze_runs.summarize_run(summary_path)

    assert row["return_estimator"] == "n_step"
    assert row["profile_filter_score"] == "none"
    assert row["filter_candidate_rows"] == 0
    assert row["filter_score_min"] is None
    assert row["sampled_transitions"] == 4
    assert row["loss_row_exposures"] == 8


def test_analyzer_preserves_filter_score_extrema_and_retention(tmp_path):
    cycle_records = []
    for cycle_index, score_min, score_max, score_mean in (
            (1, 0.5, 4.0, 2.0), (2, 0.25, 3.0, 1.0)):
        cycle_name = f"cycle_{cycle_index:03d}.json"
        (tmp_path / cycle_name).write_text(
            '{"cycle_index":' + str(cycle_index) + ',"measurement_included":true,"global":'
            '{"collection_seconds":1.0,"optimization_seconds":2.0,"received_rows":8,'
            '"sampled_transitions":4,"loss_row_exposures":8,"peak_gpu_allocated_bytes":0,'
            '"filter_candidate_rows":10,"filter_retained_rows":5,"filter_score_min":'
            + str(score_min) + ',"filter_score_max":' + str(score_max)
            + ',"filter_score_mean":' + str(score_mean) + '},"ranks":[]} '
        )
        cycle_records.append(cycle_name)
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        '{"algorithm":"sac","variant":"fixed_targets_actor_q","world_size":1,'
        '"profile_settings":{},"sac_training":{"return_estimator":"pql",'
        '"profile_filter_score":"target_advantage","profile_keep_fraction":0.5},'
        '"cycle_files":' + json.dumps(cycle_records) + '}'
    )

    row = analyze_runs.summarize_run(summary_path)

    assert row["return_estimator"] == "pql"
    assert row["filter_score_min"] == 0.25
    assert row["filter_score_max"] == 4.0
    assert row["filter_score_mean"] == 1.5
    assert row["filter_retention_fraction"] == 0.5


def test_analyzer_leaves_ppo_estimator_unlabeled(tmp_path):
    (tmp_path / "cycle_001.json").write_text(
        '{"cycle_index":1,"measurement_included":true,"global":'
        '{"collection_seconds":1.0,"optimization_seconds":2.0,"received_rows":8,'
        '"sampled_transitions":4,"loss_row_exposures":4,"peak_gpu_allocated_bytes":0},'
        '"ranks":[{"loss_row_exposures_local":4}]}'
    )
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(
        '{"algorithm":"ppo","variant":"stock","world_size":1,'
        '"profile_settings":{},"cycle_files":["cycle_001.json"]}'
    )

    row = analyze_runs.summarize_run(summary_path)

    assert row["return_estimator"] is None
