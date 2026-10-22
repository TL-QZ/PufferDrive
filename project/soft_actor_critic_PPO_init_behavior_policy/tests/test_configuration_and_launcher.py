import itertools
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from project.soft_actor_critic_PPO_init_behavior_policy import configuration
from project.soft_actor_critic_tuning.scripts import run_sac16_grid


REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = REPO_ROOT / "project/soft_actor_critic_PPO_init_behavior_policy/scripts/launch.sh"


class BehaviorPolicyConfigurationTests(unittest.TestCase):
    def _expected_config(self, trial_id):
        _, expected, launcher = run_sac16_grid._load_configs(configuration.SOURCE_CONFIG_DIR)
        axes = run_sac16_grid._grid_axes(expected)
        combination = list(itertools.product(*(values for _, values in axes)))[trial_id - 1]
        for (path, _), value in zip(axes, combination):
            run_sac16_grid._set_dotted(expected, path, value)
        expected["train"]["evaluation_interval_epochs"] = launcher[
            "trial_evaluation_interval_epochs"
        ][trial_id]
        expected["train"]["learn_alpha"] = True
        expected["train"]["total_timesteps"] = configuration.TOTAL_STEPS
        expected["wandb_group"] = configuration.WANDB_GROUP
        expected["behavior_policy"] = {
            **configuration.BEHAVIOR_POLICY_SETTINGS,
            "source_trial": trial_id,
        }
        return expected

    def test_trial_snapshots_match_source_except_declared_experiment_overrides(self):
        for trial_id in configuration.TRIAL_IDS:
            with self.subTest(trial_id=trial_id):
                with (configuration.CONFIG_DIR / f"trial_{trial_id}.yaml").open(
                    encoding="utf-8"
                ) as source:
                    snapshot = configuration.yaml.safe_load(source)
                self.assertEqual(snapshot, self._expected_config(trial_id))
                self.assertEqual(snapshot["train"]["learn_alpha"], True)
                self.assertEqual(snapshot["train"]["total_timesteps"], 1_000_000_000)
                self.assertIsNone(snapshot["load_model_path"])
                self.assertIsNone(snapshot["load_id"])
                self.assertIsNone(snapshot["train"]["resume_state_path"])
                configuration._load_trial_snapshot(trial_id)

    def test_snapshot_generator_writes_trials_to_selected_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "generated-configs"
            with patch.object(configuration, "CONFIG_DIR", config_dir):
                paths = configuration.generate_trial_snapshots()

            self.assertEqual(
                [path.name for path in paths], ["trial_97.yaml", "trial_98.yaml"]
            )
            for trial_id, path in zip(configuration.TRIAL_IDS, paths):
                with self.subTest(trial_id=trial_id):
                    with path.open(encoding="utf-8") as source:
                        snapshot = configuration.yaml.safe_load(source)
                    self.assertEqual(snapshot, self._expected_config(trial_id))

    def test_prepare_resolves_teacher_paths_and_sets_fresh_unique_run_fields(self):
        for trial_id in configuration.TRIAL_IDS:
            with self.subTest(trial_id=trial_id), tempfile.TemporaryDirectory() as temp_dir:
                run_dir = Path(temp_dir) / f"run_trial_{trial_id}"
                config_path, resolved = configuration.prepare_run(
                    trial_id, run_dir, no_wandb=True
                )
                self.assertEqual(config_path, run_dir / "config.yaml")
                self.assertTrue(resolved["train"]["fresh_run"])
                self.assertTrue(resolved["run_name"].startswith(f"{run_dir.name}_trial_{trial_id}_"))
                self.assertEqual(resolved["train"]["data_dir"], str(run_dir / "learner"))
                self.assertEqual(
                    resolved["behavior_policy"]["checkpoint"],
                    str((REPO_ROOT / configuration.BEHAVIOR_POLICY_SETTINGS["checkpoint"]).resolve()),
                )
                self.assertEqual(
                    resolved["behavior_policy"]["teacher_config"],
                    str((REPO_ROOT / configuration.BEHAVIOR_POLICY_SETTINGS["teacher_config"]).resolve()),
                )
                self.assertFalse(resolved["wandb"])
                self.assertFalse(resolved["neptune"])
                self.assertFalse(resolved["tb"])
                self.assertTrue(resolved["no_model_upload"])

    def test_local_teacher_paths_can_be_changed_without_rebuilding_from_source(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config_dir = Path(temp_dir) / "config"
            config_dir.mkdir()
            snapshot_path = configuration.CONFIG_DIR / "trial_97.yaml"
            source = configuration.yaml.safe_load(snapshot_path.read_text(encoding="utf-8"))
            source["behavior_policy"]["checkpoint"] = str(
                (REPO_ROOT / configuration.BEHAVIOR_POLICY_SETTINGS["checkpoint"]).resolve()
            )
            source["behavior_policy"]["teacher_config"] = str(
                (REPO_ROOT / configuration.BEHAVIOR_POLICY_SETTINGS["teacher_config"]).resolve()
            )
            with (config_dir / "trial_97.yaml").open("w", encoding="utf-8") as output:
                configuration.yaml.safe_dump(source, output, sort_keys=False)

            with patch.object(configuration, "CONFIG_DIR", config_dir):
                run_dir = Path(temp_dir) / "custom_paths"
                config_path, resolved = configuration.prepare_run(97, run_dir)

            self.assertTrue(config_path.is_file())
            self.assertEqual(
                resolved["behavior_policy"]["teacher_config"],
                str((REPO_ROOT / configuration.BEHAVIOR_POLICY_SETTINGS["teacher_config"]).resolve()),
            )

    def test_launcher_dry_run_writes_resolved_config_and_prints_two_rank_command(self):
        for trial_id, expected_ratio, expected_interval in ((97, 0.1, 10), (98, 0.5, 2)):
            with self.subTest(trial_id=trial_id), tempfile.TemporaryDirectory() as temp_dir:
                run_dir = Path(temp_dir) / f"trial_{trial_id}"
                result = subprocess.run(
                    [
                        "bash", str(LAUNCHER), "--trial", str(trial_id), "--gpus", "2,3",
                        "--dry-run", "--output-dir", str(run_dir), "--no-wandb",
                    ],
                    cwd=REPO_ROOT,
                    env=os.environ.copy(),
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn("CUDA_VISIBLE_DEVICES=2,3", result.stdout)
                self.assertIn("torchrun --standalone --nnodes=1 --nproc-per-node=2", result.stdout)
                self.assertIn("-m pufferlib.pufferl_sac_ppo_init_behavior_policy", result.stdout)
                self.assertIn(f"--config {run_dir}/config.yaml", result.stdout)
                self.assertTrue((run_dir / "config.yaml").is_file())
                self.assertFalse((run_dir / "metrics.jsonl").exists())
                with (run_dir / "config.yaml").open(encoding="utf-8") as source:
                    resolved = configuration.yaml.safe_load(source)
                self.assertEqual(resolved["train"]["replay_update_ratio"], expected_ratio)
                self.assertEqual(resolved["train"]["evaluation_interval_epochs"], expected_interval)
                self.assertEqual(resolved["train"]["total_timesteps"], 1_000_000_000)
                self.assertTrue(resolved["train"]["fresh_run"])
                self.assertEqual(resolved["wandb_group"], configuration.WANDB_GROUP)
                self.assertFalse(resolved["wandb"])

    def test_invalid_launch_arguments_and_existing_run_directory_fail_without_training(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            invalid = subprocess.run(
                ["bash", str(LAUNCHER), "--trial", "99", "--output-dir", str(Path(temp_dir) / "bad")],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(invalid.returncode, 0)
            self.assertFalse((Path(temp_dir) / "bad").exists())

            existing_dir = Path(temp_dir) / "existing"
            existing_dir.mkdir()
            result = subprocess.run(
                [
                    "bash", str(LAUNCHER), "--trial", "97", "--output-dir", str(existing_dir),
                    "--dry-run",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("refusing existing run directory", result.stderr)
            self.assertEqual(list(existing_dir.iterdir()), [])


if __name__ == "__main__":
    unittest.main()
