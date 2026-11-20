import json
import importlib.util
from pathlib import Path

import pytest

launcher_path = Path(__file__).resolve().parents[2] / "project/soft_actor_critic_tuning/scripts/run_sac16_grid.py"
launcher_spec = importlib.util.spec_from_file_location("run_sac16_grid", launcher_path)
run_sac16_grid = importlib.util.module_from_spec(launcher_spec)
launcher_spec.loader.exec_module(run_sac16_grid)


@pytest.mark.parametrize("summary_present", [False, True])
def test_step_cadence_requires_final_benchmark_artifacts(tmp_path, summary_present):
    learner = tmp_path / "learner"
    learner.mkdir()
    (learner / "final.pt").touch()
    (learner / "trainer_state.pt").touch()
    (tmp_path / "rank_0_runtime.json").write_text("{}")
    (tmp_path / "metrics.jsonl").write_text(json.dumps({
        "epoch": 3, "agent_steps": 35, "sac/learner_updates": 3,
    }) + "\n")
    if summary_present:
        evaluation_dir = learner / "eval/training/example/epoch_000003_step_35"
        evaluation_dir.mkdir(parents=True)
        (evaluation_dir / "evaluation_summary.json").write_text(json.dumps({
            "num_scenarios": 1, "num_episodes": 1, "metrics_mean": {"score": 0.5},
        }))
    trial = {
        "trial_id": 98, "parameters": {}, "trial_dir": str(tmp_path), "gpu_pair": [0],
        "_resolved_config": {"train": {
            "final_model_name": "final.pt", "evaluation_interval_epochs": None,
            "evaluation_interval_agent_steps": 10, "evaluation_benchmarks": "example",
        }},
    }
    run_sac16_grid._finish_trial(trial, 0, False, 1.0)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["status"] == ("completed" if summary_present else "failed_missing_artifacts")
    benchmark = summary["training_evaluation"]["benchmarks"]["example"]
    if summary_present:
        assert benchmark["metrics_mean"] == {"score": 0.5}
    else:
        assert any("missing summary" in item for item in summary["missing_artifacts"])
