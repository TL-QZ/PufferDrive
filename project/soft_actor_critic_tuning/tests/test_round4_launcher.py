import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from project.soft_actor_critic_tuning.scripts import run_sac16_grid


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "project/soft_actor_critic_tuning/scripts/launch_round4.sh"
CONFIG_ROOT = REPO_ROOT / "project/soft_actor_critic_tuning/configs"


class Round4LauncherTests(unittest.TestCase):
    def _run_launcher(self, arguments):
        return subprocess.run(
            ["bash", str(SCRIPT_PATH), *map(str, arguments)],
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=False,
        )

    def test_round4_configs_only_change_the_selected_run_settings(self):
        round3_dir = CONFIG_ROOT / "round3"
        round4_dir = CONFIG_ROOT / "round4"
        config_names = (
            "SAC_fixed_config.yaml",
            "SAC_specific_config.yaml",
            "SAC_tunable_config.yaml",
            "SAC_run_config.yaml",
        )
        round3_configs = {name: run_sac16_grid._load_yaml(round3_dir / name) for name in config_names}
        round4_configs = {name: run_sac16_grid._load_yaml(round4_dir / name) for name in config_names}

        self.assertEqual(round4_configs["SAC_fixed_config.yaml"], round3_configs["SAC_fixed_config.yaml"])

        expected_specific = round3_configs["SAC_specific_config.yaml"]
        expected_specific["train"]["batch_size"] = 8192000
        expected_specific["train"]["target_entropy"] = 1.863680
        self.assertEqual(round4_configs["SAC_specific_config.yaml"], expected_specific)

        expected_tunable = round3_configs["SAC_tunable_config.yaml"]
        expected_tunable["wandb_group"] = "vanilla_sac_round4"
        expected_tunable["train"]["gamma"] = 0.999
        self.assertEqual(round4_configs["SAC_tunable_config.yaml"], expected_tunable)

        expected_run = round3_configs["SAC_run_config.yaml"]["launcher"]
        expected_run["trial_ids"] = [97, 99]
        expected_run["trial_hours"] = None
        expected_run["gpu_pairs"] = [[0, 1]]
        expected_run["max_concurrent"] = 1
        self.assertEqual(round4_configs["SAC_run_config.yaml"]["launcher"], expected_run)

        _, resolved, _ = run_sac16_grid._load_configs(round4_dir)
        self.assertEqual(resolved["train"]["total_timesteps"], 500000000)
        self.assertEqual(resolved["train"]["minibatch_size"], 128000)
        self.assertEqual(resolved["train"]["max_minibatch_size"], 32000)
        self.assertEqual(resolved["env"]["num_agents"], 3200)
        self.assertEqual(resolved["vec"]["num_envs"], 20)

    def test_each_selected_trial_dry_runs_one_two_gpu_job_without_deadline(self):
        for trial_id, expected_ratio, gpu_args, no_wandb in (
            (97, 0.1, [], True),
            (99, 0.75, ["--gpus", "4,5"], False),
        ):
            with self.subTest(trial_id=trial_id), tempfile.TemporaryDirectory() as temporary_dir:
                output_dir = Path(temporary_dir) / f"round4-trial-{trial_id}"
                arguments = ["--trial", str(trial_id), "--dry-run", *gpu_args]
                if no_wandb:
                    arguments.append("--no-wandb")
                arguments.extend(("--output-dir", str(output_dir)))

                result = self._run_launcher(arguments)
                self.assertEqual(result.returncode, 0, result.stderr)

                with (output_dir / "manifest.json").open(encoding="utf-8") as manifest_file:
                    manifest = json.load(manifest_file)
                planned_trials = [trial for trial in manifest["trials"] if trial["status"] == "planned"]
                skipped_trials = [trial for trial in manifest["trials"] if trial["status"] == "skipped"]
                resolved_configs = list((output_dir / "configs").glob("*.yaml"))

                self.assertEqual(manifest["selected_trial_ids"], [trial_id])
                self.assertEqual(len(manifest["trials"]), 108)
                self.assertEqual(len(planned_trials), 1)
                self.assertEqual(len(skipped_trials), 107)
                self.assertEqual(len(resolved_configs), 1)
                self.assertEqual(manifest["launcher"]["trial_hours"], None)
                expected_gpu_pair = [0, 1] if trial_id == 97 else [4, 5]
                self.assertEqual(manifest["launcher"]["gpu_pairs"], [expected_gpu_pair])
                self.assertEqual(manifest["launcher"]["max_concurrent"], 1)

                trial = planned_trials[0]
                self.assertEqual(trial["gpu_pair"], expected_gpu_pair)
                self.assertEqual(trial["parameters"]["train.gamma"], 0.999)
                self.assertEqual(trial["parameters"]["train.target_entropy"], 1.863680)
                self.assertEqual(trial["parameters"]["train.update_epochs"], 3)
                self.assertEqual(trial["parameters"]["train.replay_update_ratio"], expected_ratio)
                self.assertNotIn("--trial-hours", trial["command"])
                self.assertIn("--nproc-per-node=2", trial["command"])

                config = run_sac16_grid._load_yaml(resolved_configs[0])
                self.assertEqual(config["train"]["batch_size"], 8192000)
                self.assertEqual(config["train"]["minibatch_size"], 128000)
                self.assertEqual(config["train"]["max_minibatch_size"], 32000)
                self.assertEqual(config["train"]["total_timesteps"], 500000000)
                self.assertEqual(config["train"]["tau"], 0.005)
                self.assertEqual(config["wandb_group"], "vanilla_sac_round4")
                self.assertEqual(config["wandb"], not no_wandb)

    def test_invalid_trial_gpu_and_override_arguments_fail_before_output_creation(self):
        invalid_arguments = (
            [],
            ["--trial", "1"],
            ["--trial", "97,99"],
            ["--trial", "all"],
            ["--trial", "97", "--trial", "99"],
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
                        [*invalid_args, "--output-dir", str(output_dir)],
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(output_dir.exists())

    def test_help_shows_supported_single_trial_interface(self):
        result = self._run_launcher(["--help"])
        self.assertEqual(result.returncode, 0)
        self.assertIn("--trial 97|99", result.stdout)
        self.assertIn("--gpus ID,ID", result.stdout)


if __name__ == "__main__":
    unittest.main()
