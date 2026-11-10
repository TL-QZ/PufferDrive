"""Semantic checks for cycle-local EWMA SAC threshold filtering."""

import copy
from collections import defaultdict
from datetime import timedelta
import json
import os
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel

from project.PPO_SAC_Profiling import analyze_fixed_targets, analyze_runs, profile_training as harness
import project.PPO_SAC_Profiling.sac_variants as sac_variants
from project.PPO_SAC_Profiling.sac_variants import install_variant
from project.PPO_SAC_Profiling.test_pql_filter import (
    _assert_learner_state_equal,
    _copy_sequence_plan,
    _make_pql_learner,
    _plan_target_cache,
    _spawn_gloo_with_timeout,
)


def _make_ewma_learner(*, beta=0.25, threshold_scale=0.5, update_epochs=1):
    learner = _make_pql_learner(
        seed=1701,
        update_epochs=update_epochs,
        filter_score="target_advantage",
        keep_fraction=1.0,
    )
    learner.config.update({
        "profile_filter_mode": "ewma_threshold",
        "profile_filter_ewma_beta": beta,
        "profile_filter_threshold_scale": threshold_scale,
    })
    install_variant(learner, "fixed_targets_actor_q")
    return learner


def _install_controlled_scores(monkeypatch, scores_by_cycle):
    """Replace only cycle scoring, leaving cache construction and learner updates real."""
    def controlled_scores(
            learner, replay_indices, fixed_targets, fixed_actor_q, score_name, profile, epoch):
        assert score_name == "target_advantage"
        scores = torch.as_tensor(scores_by_cycle[epoch], dtype=torch.float32)
        assert scores.numel() == replay_indices.numel()
        return scores.to(learner.device).clone()

    monkeypatch.setattr(sac_variants, "_precompute_fixed_filter_scores", controlled_scores)


def _train_cycle(learner, cycle_index):
    torch.manual_seed(1801 + cycle_index)
    return learner._train_sac_transition(
        defaultdict(float), lambda *_args, **_kwargs: None, cycle_index,
    )


def test_ewma_threshold_uses_inclusive_boundary_and_updates_once_per_cycle(monkeypatch):
    first_scores = [4, 2, 1, 0, 2, 3, 0, 0, 1, 2, 1]
    second_scores = [2, 1.75, 1.74, 0, 0, 0, 0, 0, 0, 0, 0]
    _install_controlled_scores(monkeypatch, [first_scores, second_scores])
    learner = _make_ewma_learner(threshold_scale=0.5)

    first_updates = _train_cycle(learner, 0)

    assert first_updates == 2
    assert learner.filter_current_max == 4.0
    assert learner.filter_ema_max == 4.0
    assert learner.filter_threshold == 2.0
    assert set(learner.fixed_filter_indices.tolist()) == {0, 1, 4, 5, 9}
    assert learner.filter_retained_rows == 5

    # Harness counter reset must not restart the filter's cycle-spanning state.
    harness.reset_cycle_counters(learner)
    assert learner.filter_ema_max == 4.0
    second_updates = _train_cycle(learner, 1)

    assert second_updates == 1
    assert learner.filter_current_max == 2.0
    assert learner.filter_ema_max == pytest.approx(3.5)  # .25*2 + .75*4
    assert learner.filter_threshold == pytest.approx(1.75)
    assert set(learner.fixed_filter_indices.tolist()) == {0, 1}
    assert learner.filter_retained_rows == 2


def test_zero_maximum_keeps_zero_scores_at_the_inclusive_zero_threshold(monkeypatch):
    _install_controlled_scores(monkeypatch, [[0.0] * 11])
    learner = _make_ewma_learner(threshold_scale=0.5)

    updates = _train_cycle(learner, 0)

    assert updates == 3
    assert learner.filter_current_max == 0.0
    assert learner.filter_ema_max == 0.0
    assert learner.filter_threshold == 0.0
    assert learner.filter_retained_rows == 11
    assert learner.fixed_filter_indices.tolist() == list(range(11))


def test_historical_max_drop_skips_all_updates_then_recovers(monkeypatch):
    _install_controlled_scores(monkeypatch, [
        list(range(10, -1, -1)),
        [0.0] * 11,
        list(range(11)),
    ])
    learner = _make_ewma_learner(threshold_scale=0.5)

    first_updates = _train_cycle(learner, 0)
    assert first_updates == 2
    assert learner.filter_ema_max == 10.0
    assert learner.filter_retained_rows == 6

    before_zero_cycle = copy.deepcopy(learner)
    old_targets = learner.fixed_cycle_targets
    zero_updates = _train_cycle(learner, 1)

    assert zero_updates == 0
    assert learner.filter_current_max == 0.0
    assert learner.filter_ema_max == pytest.approx(7.5)
    assert learner.filter_threshold == pytest.approx(3.75)
    assert learner.filter_retained_rows == 0
    assert learner.fixed_filter_indices.numel() == 0
    assert learner.fixed_cycle_targets is not old_targets
    _assert_learner_state_equal(learner, before_zero_cycle)

    recovered_updates = _train_cycle(learner, 2)

    assert recovered_updates == 2
    assert learner.filter_current_max == 10.0
    assert learner.filter_ema_max == pytest.approx(8.125)
    assert learner.filter_threshold == pytest.approx(4.0625)
    assert learner.filter_retained_rows == 6
    assert learner.learner_updates == first_updates + recovered_updates


def test_zero_retained_rows_still_precompute_full_pql_target_cache(monkeypatch):
    learner = _make_ewma_learner(threshold_scale=2.0)
    torch.manual_seed(1801)
    plan = _copy_sequence_plan(learner.replay.build_epoch_batches(
        learner.config["pql_horizon"], learner.minibatch_size,
    ))
    expected_targets = _plan_target_cache(learner, plan)
    observed_filter_inputs = []

    def controlled_scores(
            current, replay_indices, fixed_targets, fixed_actor_q, score_name, profile, epoch):
        assert fixed_targets is current.fixed_cycle_targets
        torch.testing.assert_close(fixed_targets, expected_targets, rtol=1e-6, atol=1e-7)
        assert fixed_targets.shape == (current.replay.capacity,)
        assert fixed_actor_q.shape == (current.replay.capacity, 3)
        observed_filter_inputs.append((fixed_targets.clone(), fixed_actor_q.clone()))
        return torch.arange(1, replay_indices.numel() + 1, dtype=torch.float32)

    monkeypatch.setattr(sac_variants, "_precompute_fixed_filter_scores", controlled_scores)
    before_updates = copy.deepcopy(learner)

    updates = _train_cycle(learner, 0)

    assert updates == 0
    assert len(observed_filter_inputs) == 1
    assert learner.fixed_target_precompute_rows == len(learner.replay)
    assert learner.fixed_actor_q_precompute_rows == len(learner.replay)
    assert learner.filter_retained_rows == 0
    assert learner.filter_candidate_rows == len(learner.replay)
    torch.testing.assert_close(learner.fixed_cycle_targets, expected_targets, rtol=1e-6, atol=1e-7)
    _assert_learner_state_equal(learner, before_updates)


def test_explicit_fraction_mode_preserves_existing_filter_updates(monkeypatch):
    implicit_fraction = _make_pql_learner(
        seed=1711,
        update_epochs=2,
        filter_score="q_advantage",
        keep_fraction=0.46,
    )
    explicit_fraction = _make_pql_learner(
        seed=1711,
        update_epochs=2,
        filter_score="q_advantage",
        keep_fraction=0.46,
    )
    explicit_fraction.config.update({
        "profile_filter_mode": "fraction",
        "profile_filter_ewma_beta": 0.25,
        "profile_filter_threshold_scale": 0.01,
    })
    install_variant(implicit_fraction, "fixed_targets_actor_q")
    install_variant(explicit_fraction, "fixed_targets_actor_q")

    torch.manual_seed(1811)
    implicit_updates = implicit_fraction._train_sac_transition(
        defaultdict(float), lambda *_args, **_kwargs: None, 0,
    )
    torch.manual_seed(1811)
    explicit_updates = explicit_fraction._train_sac_transition(
        defaultdict(float), lambda *_args, **_kwargs: None, 0,
    )

    assert explicit_updates == implicit_updates == 4
    assert explicit_fraction.filter_retained_rows == implicit_fraction.filter_retained_rows == 6
    assert explicit_fraction.fixed_filter_indices.tolist() == implicit_fraction.fixed_filter_indices.tolist()
    _assert_learner_state_equal(explicit_fraction, implicit_fraction)


def _sac_cli_args(tmp_path, *extra):
    return [
        "--algorithm", "sac",
        "--output-dir", str(tmp_path / "ewma-profile"),
        "--device", "cpu",
        *extra,
    ]


def test_ewma_cli_values_reach_resolved_config_and_provenance(tmp_path):
    args = harness.parse_args(_sac_cli_args(
        tmp_path,
        "--variant", "fixed_targets_actor_q",
        "--sac-filter-score", "target_advantage",
        "--sac-filter-mode", "ewma_threshold",
        "--sac-filter-ewma-beta", "0.4",
        "--sac-filter-threshold-scale", "0.03",
    ))
    config, _source = harness.prepare_config(args, args.output_dir)
    provenance = harness.sac_training_provenance(args, config["train"])

    assert provenance["profile_filter_mode"] == "ewma_threshold"
    assert provenance["profile_filter_ewma_beta"] == pytest.approx(0.4)
    assert provenance["profile_filter_threshold_scale"] == pytest.approx(0.03)


def test_fraction_cli_defaults_and_provenance_remain_unchanged(tmp_path):
    args = harness.parse_args(_sac_cli_args(tmp_path))
    config, _source = harness.prepare_config(args, args.output_dir)
    provenance = harness.sac_training_provenance(args, config["train"])

    assert args.sac_filter_mode == "fraction"
    assert args.sac_filter_ewma_beta == pytest.approx(0.25)
    assert args.sac_filter_threshold_scale == pytest.approx(0.01)
    assert provenance["profile_filter_mode"] == "fraction"


@pytest.mark.parametrize("extra", [
    ("--sac-filter-mode", "ewma_threshold"),
    ("--sac-filter-mode", "ewma_threshold", "--sac-filter-score", "q_advantage",
     "--sac-keep-fraction", "0.5"),
    ("--sac-filter-mode", "ewma_threshold", "--sac-filter-score", "q_advantage",
     "--variant", "fixed_targets"),
    ("--sac-filter-mode", "other"),
    ("--sac-filter-ewma-beta", "nan"),
    ("--sac-filter-ewma-beta", "inf"),
    ("--sac-filter-ewma-beta", "-0.01"),
    ("--sac-filter-ewma-beta", "1.01"),
    ("--sac-filter-threshold-scale", "nan"),
    ("--sac-filter-threshold-scale", "inf"),
    ("--sac-filter-threshold-scale", "-0.01"),
])
def test_invalid_ewma_cli_config_fails_before_profile_setup(tmp_path, extra):
    with pytest.raises(SystemExit):
        harness.parse_args(_sac_cli_args(tmp_path, *extra))


@pytest.mark.parametrize("key,value", [
    ("profile_filter_mode", "unknown"),
    ("profile_filter_mode", "ewma_threshold"),  # score defaults to none
    ("profile_filter_ewma_beta", float("nan")),
    ("profile_filter_ewma_beta", float("inf")),
    ("profile_filter_ewma_beta", -0.01),
    ("profile_filter_ewma_beta", 1.01),
    ("profile_filter_threshold_scale", float("nan")),
    ("profile_filter_threshold_scale", float("inf")),
    ("profile_filter_threshold_scale", -0.01),
])
def test_invalid_ewma_learner_config_fails_before_optimizer_updates(key, value):
    learner = _make_pql_learner(filter_score="q_advantage", keep_fraction=1.0)
    learner.config.update({
        "profile_filter_mode": "ewma_threshold",
        "profile_filter_ewma_beta": 0.25,
        "profile_filter_threshold_scale": 0.5,
    })
    if key == "profile_filter_mode" and value == "ewma_threshold":
        learner.config["profile_filter_score"] = "none"
    learner.config[key] = value

    with pytest.raises(ValueError):
        install_variant(learner, "fixed_targets_actor_q")

    assert learner.learner_updates == 0
    assert learner.sampled_transitions == 0


@pytest.mark.parametrize("key,value", [
    ("profile_filter_score", "none"),
    ("profile_keep_fraction", 0.5),
])
def test_ewma_requires_scored_rows_and_full_keep_fraction(key, value):
    learner = _make_pql_learner(filter_score="target_advantage", keep_fraction=1.0)
    learner.config.update({
        "profile_filter_mode": "ewma_threshold",
        "profile_filter_ewma_beta": 0.25,
        "profile_filter_threshold_scale": 0.5,
        key: value,
    })

    with pytest.raises(ValueError):
        install_variant(learner, "fixed_targets_actor_q")

    assert learner.learner_updates == 0


def _ewma_gloo_worker(rank, init_method, result_dir):
    os.environ["RANK"] = str(rank)
    os.environ["WORLD_SIZE"] = "2"
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    torch.set_num_threads(1)
    dist.init_process_group(
        "gloo",
        init_method=init_method,
        rank=rank,
        world_size=2,
        timeout=timedelta(seconds=120),
    )
    try:
        learner = _make_pql_learner(
            seed=1901,
            update_epochs=1,
            observation_offset=rank * 0.07,
            filter_score="target_advantage",
            keep_fraction=1.0,
        )
        learner.config.update({
            "profile_filter_mode": "ewma_threshold",
            "profile_filter_ewma_beta": 0.25,
            "profile_filter_threshold_scale": 0.5,
        })
        install_variant(learner, "fixed_targets_actor_q")
        learner.ddp_policy = DistributedDataParallel(learner.uncompiled_policy)
        learner.ddp_critic = DistributedDataParallel(learner.uncompiled_critic)
        learner.policy_train = learner.ddp_policy
        learner.critic_train = learner.ddp_critic
        learner.distributed = True
        learner.world_size = 2

        rank_scores = {
            0: [list(range(10, -1, -1)), [0.0] * 11, list(range(12, 1, -1))],
            1: [[4, 3.5, 3, 2.5, 2, 1.5, 1, 0.5, 0, 0, 0],
                [4, 3.5, 3, 2.5, 2, 1.5, 1, 0.5, 0, 0, 0],
                [4, 3.5, 3, 2.5, 2, 1.5, 1, 0.5, 0, 0, 0]],
        }

        def controlled_scores(
                current, replay_indices, fixed_targets, fixed_actor_q,
                score_name, profile, epoch):
            assert score_name == "target_advantage"
            return torch.tensor(
                rank_scores[rank][epoch], dtype=torch.float32, device=current.device,
            )

        sac_variants._precompute_fixed_filter_scores = controlled_scores
        cycles = []
        for cycle_index in range(3):
            updates = _train_cycle(learner, cycle_index)
            cycles.append({
                "updates": updates,
                "current_max": learner.filter_current_max,
                "ema_max": learner.filter_ema_max,
                "threshold": learner.filter_threshold,
                "retained_before_sync": learner.filter_retained_before_sync,
                "retained_rows": learner.filter_retained_rows,
            })
            ema_before_reset = learner.filter_ema_max
            harness.reset_cycle_counters(learner)
            assert learner.filter_ema_max == ema_before_reset

        torch.save({
            "cycles": cycles,
            "learner_updates": learner.learner_updates,
            "sampled_transitions": learner.sampled_transitions,
            "policy": {
                key: value.detach().cpu().clone()
                for key, value in learner.uncompiled_policy.state_dict().items()
            },
            "critic": {
                key: value.detach().cpu().clone()
                for key, value in learner.uncompiled_critic.state_dict().items()
            },
            "alpha": learner.log_alpha.detach().cpu().clone(),
        }, Path(result_dir) / f"rank-{rank}.pt")
        dist.barrier()
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


@pytest.mark.skipif(
    not dist.is_available() or not getattr(dist, "is_gloo_available", lambda: False)(),
    reason="PyTorch Gloo distributed support is unavailable",
)
def test_two_rank_gloo_ewma_uses_local_history_and_skips_globally_empty_cycle(tmp_path):
    result_dir = tmp_path / "ewma-gloo-results"
    result_dir.mkdir()
    init_path = tmp_path / "ewma-gloo-init"
    _spawn_gloo_with_timeout(
        _ewma_gloo_worker,
        (f"file://{init_path}", str(result_dir)),
    )
    results = [
        torch.load(result_dir / f"rank-{rank}.pt", map_location="cpu", weights_only=False)
        for rank in range(2)
    ]

    assert [[cycle["updates"] for cycle in result["cycles"]] for result in results] == [
        [2, 0, 2], [2, 0, 2],
    ]
    assert [[cycle["retained_before_sync"] for cycle in result["cycles"]]
            for result in results] == [[6, 0, 8], [5, 5, 5]]
    assert [[cycle["retained_rows"] for cycle in result["cycles"]]
            for result in results] == [[5, 0, 5], [5, 0, 5]]
    assert [result["cycles"][1]["ema_max"] for result in results] == pytest.approx([7.5, 4.0])
    assert [result["cycles"][2]["ema_max"] for result in results] == pytest.approx([8.625, 4.0])
    assert [result["learner_updates"] for result in results] == [4, 4]
    # This counter includes both ranks: five selected rows in each of two trained cycles.
    assert [result["sampled_transitions"] for result in results] == [20, 20]
    for model_name in ("policy", "critic"):
        for parameter_name in results[0][model_name]:
            torch.testing.assert_close(
                results[0][model_name][parameter_name],
                results[1][model_name][parameter_name],
                rtol=1e-6,
                atol=1e-7,
            )
    torch.testing.assert_close(results[0]["alpha"], results[1]["alpha"], rtol=1e-6, atol=1e-7)


def _write_zero_exposure_run(run_dir):
    run_dir.mkdir(parents=True, exist_ok=True)
    cycle_name = "cycle_001.json"
    (run_dir / cycle_name).write_text(json.dumps({
        "cycle_index": 1,
        "measurement_included": True,
        "global": {
            "collection_seconds": 1.0,
            "optimization_seconds": 2.0,
            "received_rows": 8,
            "sampled_transitions": 0,
            "loss_row_exposures": 0,
            "peak_gpu_allocated_bytes": 0,
            "filter_candidate_rows": 11,
            "filter_retained_rows": 0,
        },
        "ranks": [{
            "critic_loss_rows_local": 0,
            "loss_row_exposures_local": 0,
            "target_forward_rows_local": 11,
            "fixed_target_precompute_rows_local": 11,
            "fixed_actor_q_precompute_rows_local": 11,
            "fixed_actor_q_cache_bytes_local": 132,
            "policy_loss_rows_local": 0,
        }],
    }) + "\n")
    summary = {
        "algorithm": "sac",
        "variant": "fixed_targets_actor_q",
        "world_size": 1,
        "profile_settings": {"sac_filter_score": "target_advantage", "sac_keep_fraction": 1.0},
        "sac_training": {
            "return_estimator": "pql",
            "profile_filter_score": "target_advantage",
            "profile_keep_fraction": 1.0,
        },
        "cycle_files": [cycle_name],
        "final_model_states": [{"sha256": "fixture", "fixed_target_audit": None}],
        "all_parameters_finite": True,
    }
    summary_path = run_dir / "summary.json"
    summary_path.write_text(json.dumps(summary) + "\n")
    return summary_path


def test_both_analyzers_report_zero_exposures_with_dash_per_sample(tmp_path, monkeypatch):
    run_dir = tmp_path / "zero-exposure-run"
    summary_path = _write_zero_exposure_run(run_dir)
    run_row = analyze_runs.summarize_run(summary_path)

    assert run_row["sampled_transitions"] == 0
    assert run_row["learner_microseconds_per_sample"] is None

    run_output = tmp_path / "run-analysis" / "report.json"
    monkeypatch.setattr("sys.argv", ["analyze_runs", str(run_dir), "--output", str(run_output)])
    analyze_runs.main()
    run_markdown = run_output.with_suffix(".md").read_text()
    assert "| 0.000 | 0.000 | — |" in run_markdown

    # The fixed-target CLI imports its sibling analyzer as a top-level module.
    monkeypatch.syspath_prepend(str(Path(analyze_runs.__file__).resolve().parent))
    from project.PPO_SAC_Profiling import analyze_fixed_targets

    fixed_output = tmp_path / "fixed-analysis" / "report.json"
    fixed_output.parent.mkdir(parents=True)
    monkeypatch.setattr(
        "sys.argv", ["analyze_fixed_targets", str(run_dir), "--output", str(fixed_output)],
    )
    analyze_fixed_targets.main()
    fixed_markdown = fixed_output.with_suffix(".md").read_text()
    assert "| — |" in fixed_markdown
    fixed_rows = json.loads(fixed_output.with_suffix(".json").read_text())
    assert fixed_rows[0]["target_rows_per_learner_sample"] is None
    assert fixed_rows[0]["learner_microseconds_per_sample"] is None
