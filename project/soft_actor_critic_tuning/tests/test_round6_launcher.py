import copy
import json
import math
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from project.soft_actor_critic_tuning.scripts import run_sac16_grid


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "project/soft_actor_critic_tuning/scripts/launch_round6.sh"
CONFIG_ROOT = REPO_ROOT / "project/soft_actor_critic_tuning/configs"
RUNNER_MODULE = "project.soft_actor_critic_tuning.scripts.run_sac16_grid"


class Round6LauncherTests(unittest.TestCase):
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

    def test_round6_preserves_round4_1_grid_with_only_expected_config_changes(self):
        round4_1_dir = CONFIG_ROOT / "round4_1"
        round6_dir = CONFIG_ROOT / "round6"
        config_names = (
            "SAC_fixed_config.yaml",
            "SAC_specific_config.yaml",
            "SAC_tunable_config.yaml",
            "SAC_run_config.yaml",
        )
        round4_1_configs = {
            name: run_sac16_grid._load_yaml(round4_1_dir / name)
            for name in config_names
        }
        round6_configs = {
            name: run_sac16_grid._load_yaml(round6_dir / name)
            for name in config_names
        }

        self.assertEqual(
            round6_configs["SAC_fixed_config.yaml"],
            round4_1_configs["SAC_fixed_config.yaml"],
        )
        self.assertEqual(
            round6_configs["SAC_specific_config.yaml"],
            round4_1_configs["SAC_specific_config.yaml"],
        )

        expected_tunable = copy.deepcopy(round4_1_configs["SAC_tunable_config.yaml"])
        expected_tunable["wandb_group"] = "vanilla_sac_round6"
        expected_tunable["train"]["anneal_lr"] = True
        expected_tunable["train"]["total_timesteps"] = 2000000000
        self.assertEqual(round6_configs["SAC_tunable_config.yaml"], expected_tunable)

        expected_launcher = copy.deepcopy(
            round4_1_configs["SAC_run_config.yaml"]["launcher"]
        )
        expected_launcher["trial_total_timesteps"] = {
            97: 2000000000,
            98: 4000000000,
        }
        self.assertEqual(
            round6_configs["SAC_run_config.yaml"]["launcher"], expected_launcher
        )

        _, round4_1_resolved, _ = run_sac16_grid._load_configs(round4_1_dir)
        _, round6_resolved, _ = run_sac16_grid._load_configs(round6_dir)
        expected_resolved = copy.deepcopy(round4_1_resolved)
        expected_resolved["wandb_group"] = "vanilla_sac_round6"
        expected_resolved["train"]["anneal_lr"] = True
        self.assertEqual(round6_resolved, expected_resolved)
        axes = run_sac16_grid._grid_axes(round6_resolved)
        self.assertEqual(math.prod(len(values) for _, values in axes), 108)
        self.assertEqual(round6_resolved["train"]["total_timesteps"], 2000000000)

    def test_each_candidate_dry_runs_with_its_budget_refresh_and_evaluation(self):
        for trial_id, expected_ratio, expected_interval, expected_budget in (
            (97, 0.1, 10, 2000000000),
            (98, 0.5, 2, 4000000000),
        ):
            with self.subTest(trial_id=trial_id), tempfile.TemporaryDirectory() as temporary_dir:
                output_dir = Path(temporary_dir) / f"round6-trial-{trial_id}"
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

                manifest = json.loads(
                    (output_dir / "manifest.json").read_text(encoding="utf-8")
                )
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
                self.assertEqual(trial["total_timesteps"], expected_budget)
                self.assertNotIn("--trial-hours", trial["command"])
                self.assertIn("--nproc-per-node=2", trial["command"])

                config = run_sac16_grid._load_yaml(resolved_configs[0])
                self.assertEqual(config["train"]["batch_size"], 8192000)
                self.assertEqual(config["train"]["minibatch_size"], 128000)
                self.assertEqual(config["train"]["max_minibatch_size"], 32000)
                self.assertEqual(config["train"]["total_timesteps"], expected_budget)
                self.assertEqual(config["train"]["evaluation_interval_epochs"], expected_interval)
                self.assertEqual(config["train"]["tau"], 0.005)
                self.assertTrue(config["train"]["anneal_lr"])
                self.assertIsNone(config["train"]["resume_state_path"])
                self.assertIsNone(config["load_model_path"])
                self.assertTrue(config["train"]["fresh_run"])
                self.assertEqual(config["wandb_group"], "vanilla_sac_round6")
                self.assertTrue(config["wandb"])

    def test_invalid_trial_and_gpu_arguments_fail_before_output_creation(self):
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

    def test_smoke_manifest_budget_matches_resolved_trial_config(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "round6-smoke-trial-98"
            result = self._run_grid(
                [
                    "--config-dir",
                    str(CONFIG_ROOT / "round6"),
                    "--trial-ids",
                    "98",
                    "--smoke",
                    "--dry-run",
                    "--output-dir",
                    str(output_dir),
                ]
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            config = run_sac16_grid._load_yaml(
                Path(manifest["trials"][97]["config_path"])
            )

        self.assertEqual(manifest["selected_trial_ids"], [98])
        self.assertEqual(manifest["trials"][97]["total_timesteps"], 4000000000)
        self.assertEqual(config["train"]["total_timesteps"], 4000000000)
        self.assertEqual(manifest["launcher"]["trial_hours"], 60 / 3600)

    def test_invalid_trial_budget_mappings_fail_before_artifacts(self):
        invalid_mappings = (
            ("not-a-mapping", "invalid"),
            ("null-mapping", None),
            ("out-of-grid-id", {109: 2}),
            ("zero-id", {0: 2}),
            ("string-id", {"97": 2}),
            ("boolean-id", {True: 2}),
            ("string-budget", {97: "2000000000"}),
            ("boolean-budget", {97: True}),
            ("fractional-budget", {97: 2000000000.5}),
            ("nan-budget", {97: float("nan")}),
            ("zero-budget", {97: 0}),
            ("negative-budget", {97: -1}),
        )
        source_config_dir = CONFIG_ROOT / "round6"
        with tempfile.TemporaryDirectory() as temporary_dir:
            for name, budget_mapping in invalid_mappings:
                with self.subTest(mapping=name):
                    case_dir = Path(temporary_dir) / name
                    config_dir = case_dir / "config"
                    shutil.copytree(source_config_dir, config_dir)
                    run_config_path = config_dir / "SAC_run_config.yaml"
                    run_config = run_sac16_grid._load_yaml(run_config_path)
                    run_config["launcher"]["trial_total_timesteps"] = budget_mapping
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
                    self.assertIn("trial_total_timesteps", result.stderr)
                    self.assertFalse(output_dir.exists())

    def test_help_documents_single_trial_interface(self):
        result = self._run_launcher(["--help"])
        self.assertEqual(result.returncode, 0)
        self.assertIn("--trial 97|98", result.stdout)
        self.assertIn("--gpus ID,ID", result.stdout)


if __name__ == "__main__":
    unittest.main()
