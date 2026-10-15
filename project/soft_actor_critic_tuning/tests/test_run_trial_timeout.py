import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from project.soft_actor_critic_tuning.scripts import run_trial


class RunTrialTimeoutTests(unittest.TestCase):
    def test_trial_hours_argument_is_optional_and_numeric_values_are_validated(self):
        parser = run_trial._build_parser()
        uncapped = parser.parse_args(["--config", "config.yaml", "--artifact-dir", "trial"])
        capped = parser.parse_args([
            "--config", "config.yaml", "--artifact-dir", "trial", "--trial-hours", "2.5",
        ])

        self.assertIsNone(uncapped.trial_hours)
        self.assertEqual(capped.trial_hours, 2.5)
        for value in ("true", "0", "-1", "nan", "inf", "1e308"):
            with self.subTest(value=value), self.assertRaises(SystemExit):
                parser.parse_args([
                    "--config", "config.yaml", "--artifact-dir", "trial", "--trial-hours", value,
                ])

    def test_uncapped_deadline_ignores_stale_inherited_environment(self):
        with patch.dict("os.environ", {"SAC16_TRIAL_DEADLINE_MONOTONIC": "0"}):
            self.assertIsNone(run_trial._resolve_deadline_monotonic(None))

    def test_uncapped_monitor_logs_late_cycle_without_time_stopping(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            metrics_path = Path(temporary_dir) / "metrics.jsonl"
            monitor = run_trial.TrialMonitor(metrics_path, Path(temporary_dir) / "stop.json", None)
            with patch.object(run_trial.time, "monotonic", side_effect=[0, 10**9]) as monotonic:
                should_stop = monitor({"loss": 1.25, "agent_steps": 42})

            self.assertFalse(should_stop)
            monotonic.assert_not_called()
            self.assertEqual(json.loads(metrics_path.read_text(encoding="utf-8")), {
                "agent_steps": 42,
                "loss": 1.25,
            })
            self.assertIsNone(monitor.stop_reason)

    def test_uncapped_monitor_still_stops_on_nonfinite_loss(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            stop_path = Path(temporary_dir) / "stop.json"
            monitor = run_trial.TrialMonitor(
                Path(temporary_dir) / "metrics.jsonl", stop_path, None,
            )
            self.assertTrue(monitor({"critic_loss": float("inf")}))

            self.assertEqual(monitor.stop_reason, "nonfinite_learner_loss")
            self.assertEqual(json.loads(stop_path.read_text(encoding="utf-8")), {
                "metric": "critic_loss",
                "stop_reason": "nonfinite_learner_loss",
            })

    def test_capped_monitor_keeps_deadline_before_next_cycle_stop(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            stop_path = Path(temporary_dir) / "stop.json"
            with patch.object(run_trial.time, "monotonic", side_effect=[0, 95]):
                monitor = run_trial.TrialMonitor(
                    Path(temporary_dir) / "metrics.jsonl", stop_path, 100,
                )
                self.assertTrue(monitor({"loss": 1.25}))

            self.assertEqual(monitor.stop_reason, "budget_before_next_cycle")
            self.assertEqual(json.loads(stop_path.read_text(encoding="utf-8"))["stop_reason"], "budget_before_next_cycle")


if __name__ == "__main__":
    unittest.main()
