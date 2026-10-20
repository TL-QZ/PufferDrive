import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from project.soft_actor_critic_tuning.scripts import run_sac16_grid


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "project/soft_actor_critic_tuning/scripts/launch_round4_1.sh"
CONFIG_ROOT = REPO_ROOT / "project/soft_actor_critic_tuning/configs"
RUNNER_MODULE = "project.soft_actor_critic_tuning.scripts.run_sac16_grid"


class Round4_1LauncherTests(unittest.TestCase):
    def _run_launcher(self, arguments):
        return subprocess.run(
            ["bash", str(SCRIPT_PATH), *map(str, arguments)],
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=False,
        )

    def _run_grid(self, arguments):
        return subprocess.run(
            [sys.executable, "-m", RUNNER_MODULE, *map(str, arguments)],
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=False,
        )

    def test_round4_1_configs_preserve_round4_grid_with_selected_run_changes(self):
        round4_dir = CONFIG_ROOT / "round4"
        round4_1_dir = CONFIG_ROOT / "round4_1"
        config_names = (
            "SAC_fixed_config.yaml",
            "SAC_specific_config.yaml",
            "SAC_tunable_config.yaml",
            "SAC_run_config.yaml",
        )
        round4_configs = {
            name: run_sac16_grid._load_yaml(round4_dir / name) for name in config_names
        }
        round4_1_configs = {
            name: run_sac16_grid._load_yaml(round4_1_dir / name) for name in config_names
        }

        self.assertEqual(
            round4_1_configs["SAC_fixed_config.yaml"],
            round4_configs["SAC_fixed_config.yaml"],
        )
        self.assertEqual(
            round4_1_configs["SAC_specific_config.yaml"],
            round4_configs["SAC_specific_config.yaml"],
        )

        expected_tunable = copy.deepcopy(round4_configs["SAC_tunable_config.yaml"])
        expected_tunable["wandb_group"] = "vanilla_sac_round4_1"
        expected_tunable["train"]["total_timesteps"] = 2000000000
        expected_tunable["train"]["evaluation_interval_epochs"] = 10
        self.assertEqual(round4_1_configs["SAC_tunable_config.yaml"], expected_tunable)

        expected_run_profile = copy.deepcopy(
            round4_configs["SAC_run_config.yaml"]["launcher"]
        )
        expected_run_profile["trial_ids"] = [97, 98]
        expected_run_profile["trial_evaluation_interval_epochs"] = {97: 10, 98: 2}
        self.assertEqual(
            round4_1_configs["SAC_run_config.yaml"]["launcher"], expected_run_profile
        )

        _, resolved, _ = run_sac16_grid._load_configs(round4_1_dir)
        self.assertEqual(resolved["wandb_group"], "vanilla_sac_round4_1")
        self.assertEqual(resolved["train"]["total_timesteps"], 2000000000)
        self.assertEqual(resolved["train"]["evaluation_interval_epochs"], 10)
        self.assertEqual(resolved["train"]["minibatch_size"], 128000)
        self.assertEqual(resolved["train"]["max_minibatch_size"], 32000)
        self.assertEqual(resolved["env"]["num_agents"], 3200)
        self.assertEqual(resolved["vec"]["num_envs"], 20)

    def test_each_selected_trial_dry_runs_with_its_grid_ratio_and_evaluation_interval(self):
        for trial_id, expected_ratio, expected_interval in (
            (97, 0.1, 10),
            (98, 0.5, 2),
        ):
            with self.subTest(trial_id=trial_id), tempfile.TemporaryDirectory() as temporary_dir:
                output_dir = Path(temporary_dir) / f"round4-1-trial-{trial_id}"
                result = self._run_launcher(
                    [
                        "--trial",
                        str(trial_id),
                        "--dry-run",
                        "--output-dir",
                        str(output_dir),
                    ]
                )
                self.assertEqual(result.returncode, 0, result.stderr)

                with (output_dir / "manifest.json").open(encoding="utf-8") as manifest_file:
                    manifest = json.load(manifest_file)
                planned_trials = [
                    trial for trial in manifest["trials"] if trial["status"] == "planned"
                ]
                skipped_trials = [
                    trial for trial in manifest["trials"] if trial["status"] == "skipped"
                ]
                resolved_configs = list((output_dir / "configs").glob("*.yaml"))

                self.assertEqual(manifest["selected_trial_ids"], [trial_id])
                self.assertEqual(len(manifest["trials"]), 108)
                self.assertEqual(len(planned_trials), 1)
                self.assertEqual(len(skipped_trials), 107)
                self.assertEqual(len(resolved_configs), 1)
                self.assertIsNone(manifest["launcher"]["trial_hours"])
                self.assertEqual(manifest["launcher"]["gpu_pairs"], [[0, 1]])
                self.assertEqual(manifest["launcher"]["max_concurrent"], 1)

                trial = planned_trials[0]
                self.assertEqual(trial["gpu_pair"], [0, 1])
                self.assertEqual(trial["parameters"]["train.gamma"], 0.999)
                self.assertEqual(trial["parameters"]["train.target_entropy"], 1.863680)
                self.assertEqual(trial["parameters"]["train.update_epochs"], 3)
                self.assertEqual(
                    trial["parameters"]["train.replay_update_ratio"], expected_ratio
                )
                self.assertEqual(trial["evaluation_interval_epochs"], expected_interval)
                self.assertNotIn("--trial-hours", trial["command"])
                self.assertIn("--nproc-per-node=2", trial["command"])

                config = run_sac16_grid._load_yaml(resolved_configs[0])
                self.assertEqual(config["train"]["batch_size"], 8192000)
                self.assertEqual(config["train"]["minibatch_size"], 128000)
                self.assertEqual(config["train"]["max_minibatch_size"], 32000)
                self.assertEqual(config["train"]["total_timesteps"], 2000000000)
                self.assertEqual(config["train"]["evaluation_interval_epochs"], expected_interval)
                self.assertEqual(config["train"]["tau"], 0.005)
                self.assertEqual(config["wandb_group"], "vanilla_sac_round4_1")
                self.assertTrue(config["wandb"])

    def test_invalid_trial_gpu_and_override_arguments_fail_before_output_creation(self):
        invalid_arguments = (
            [],
            ["--trial", "1"],
            ["--trial", "99"],
            ["--trial", "97,98"],
            ["--trial", "all"],
            ["--trial", "97", "--trial", "98"],
            ["--trial", "97", "--gpus", "0"],
            ["--trial", "97", "--gpus", "0,0"],
            ["--trial", "97", "--gpus", "00,0"],
            ["--trial", "97", "--gpus", "-1,1"],
            ["--trial", "97", "--gpus", "0,1,2"],
            ["--trial", "97", "--max-concurrent", "2"],
            ["--trial", "97", "--trial-hours", "3"],
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            for index, invalid_args in enumerate(invalid_arguments):
                with self.subTest(arguments=invalid_args):
                    output_dir = Path(temporary_dir) / f"invalid-{index}"
                    result = self._run_launcher(
                        [*invalid_args, "--output-dir", str(output_dir)]
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(output_dir.exists())

    def test_malformed_evaluation_interval_mappings_fail_before_artifacts(self):
        invalid_mappings = (
            ("not-a-mapping", "invalid"),
            ("out-of-grid-id", {109: 2}),
            ("string-id", {"97": 2}),
            ("boolean-id", {True: 2}),
            ("zero-interval", {97: 0}),
            ("negative-interval", {97: -1}),
            ("boolean-interval", {97: True}),
            ("fractional-interval", {97: 1.5}),
        )
        source_config_dir = CONFIG_ROOT / "round4_1"
        with tempfile.TemporaryDirectory() as temporary_dir:
            for name, interval_mapping in invalid_mappings:
                with self.subTest(mapping=name):
                    case_dir = Path(temporary_dir) / name
                    config_dir = case_dir / "config"
                    shutil.copytree(source_config_dir, config_dir)
                    run_config_path = config_dir / "SAC_run_config.yaml"
                    run_config = run_sac16_grid._load_yaml(run_config_path)
                    run_config["launcher"]["trial_evaluation_interval_epochs"] = interval_mapping
                    with run_config_path.open("w", encoding="utf-8") as run_config_file:
                        run_sac16_grid.yaml.safe_dump(
                            run_config, run_config_file, sort_keys=False
                        )

                    output_dir = case_dir / "output"
                    result = self._run_grid(
                        [
                            "--config-dir",
                            str(config_dir),
                            "--trial-ids",
                            "97",
                            "--dry-run",
                            "--output-dir",
                            str(output_dir),
                        ]
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertIn("trial_evaluation_interval_epochs", result.stderr)
                    self.assertFalse(output_dir.exists())

    def test_help_shows_supported_single_trial_interface(self):
        result = self._run_launcher(["--help"])
        self.assertEqual(result.returncode, 0)
        self.assertIn("--trial 97|98", result.stdout)
        self.assertIn("--gpus ID,ID", result.stdout)


if __name__ == "__main__":
    unittest.main()
