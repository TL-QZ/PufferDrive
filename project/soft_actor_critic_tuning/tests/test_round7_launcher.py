import copy
import json
import math
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from project.soft_actor_critic_tuning.scripts import run_sac16_grid


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "project/soft_actor_critic_tuning/scripts/launch_round7.sh"
CONFIG_ROOT = REPO_ROOT / "project/soft_actor_critic_tuning/configs"


class Round7LauncherTests(unittest.TestCase):
    def _run_launcher(self, arguments):
        return subprocess.run(
            ["bash", str(SCRIPT_PATH), *map(str, arguments)],
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=False,
        )

    def test_round7_differs_from_round6_only_by_capacity_and_group(self):
        round6_dir = CONFIG_ROOT / "round6"
        round7_dir = CONFIG_ROOT / "round7"
        config_names = (
            "SAC_fixed_config.yaml",
            "SAC_specific_config.yaml",
            "SAC_tunable_config.yaml",
            "SAC_run_config.yaml",
        )
        round6_configs = {
            name: run_sac16_grid._load_yaml(round6_dir / name)
            for name in config_names
        }
        round7_configs = {
            name: run_sac16_grid._load_yaml(round7_dir / name)
            for name in config_names
        }

        self.assertEqual(
            round7_configs["SAC_fixed_config.yaml"],
            round6_configs["SAC_fixed_config.yaml"],
        )
        expected_specific = copy.deepcopy(round6_configs["SAC_specific_config.yaml"])
        expected_specific["train"]["batch_size"] = 2048000
        self.assertEqual(round7_configs["SAC_specific_config.yaml"], expected_specific)

        expected_tunable = copy.deepcopy(round6_configs["SAC_tunable_config.yaml"])
        expected_tunable["wandb_group"] = "vanilla_sac_round7"
        self.assertEqual(round7_configs["SAC_tunable_config.yaml"], expected_tunable)
        self.assertEqual(
            round7_configs["SAC_run_config.yaml"],
            round6_configs["SAC_run_config.yaml"],
        )

        _, round6_resolved, round6_launcher = run_sac16_grid._load_configs(round6_dir)
        _, round7_resolved, round7_launcher = run_sac16_grid._load_configs(round7_dir)
        expected_resolved = copy.deepcopy(round6_resolved)
        expected_resolved["train"]["batch_size"] = 2048000
        expected_resolved["wandb_group"] = "vanilla_sac_round7"
        self.assertEqual(round7_resolved, expected_resolved)
        self.assertEqual(round7_launcher, round6_launcher)
        self.assertEqual(
            math.prod(len(values) for _, values in run_sac16_grid._grid_axes(round7_resolved)),
            108,
        )

    def test_each_candidate_dry_runs_with_isolated_round7_paths(self):
        for trial_id, ratio, interval, budget, refresh_count, horizon in (
            (97, 0.1, 10, 2000000000, 204800, 4883),
            (98, 0.5, 2, 4000000000, 1024000, 1954),
        ):
            with self.subTest(trial_id=trial_id), tempfile.TemporaryDirectory() as temporary_dir:
                output_dir = Path(temporary_dir) / f"round7-trial-{trial_id}"
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
                planned = [trial for trial in manifest["trials"] if trial["status"] == "planned"]
                self.assertEqual(manifest["selected_trial_ids"], [trial_id])
                self.assertEqual(manifest["config_dir"], str((CONFIG_ROOT / "round7").resolve()))
                self.assertEqual(manifest["output_dir"], str(output_dir.resolve()))
                self.assertEqual(manifest["launcher"]["gpu_pairs"], [[0, 1]])
                self.assertEqual(manifest["launcher"]["max_concurrent"], 1)
                self.assertIsNone(manifest["launcher"]["trial_hours"])
                self.assertEqual(len(planned), 1)

                trial = planned[0]
                trial_dir = output_dir / f"trial_{trial_id:02d}"
                self.assertEqual(Path(trial["trial_dir"]), trial_dir.resolve())
                self.assertEqual(Path(trial["config_path"]), (output_dir / "configs" / f"trial_{trial_id:02d}.yaml").resolve())
                self.assertEqual(trial["parameters"]["train.replay_update_ratio"], ratio)
                self.assertEqual(trial["evaluation_interval_epochs"], interval)
                self.assertEqual(trial["total_timesteps"], budget)
                self.assertEqual(trial["parameters"]["train.gamma"], 0.999)
                self.assertEqual(trial["parameters"]["train.target_entropy"], 1.863680)
                self.assertEqual(trial["parameters"]["train.update_epochs"], 3)
                self.assertIn("--nproc-per-node=2", trial["command"])

                config = run_sac16_grid._load_yaml(Path(trial["config_path"]))
                self.assertEqual(config["train"]["batch_size"], 2048000)
                self.assertEqual(config["train"]["minibatch_size"], 128000)
                self.assertEqual(config["train"]["max_minibatch_size"], 32000)
                self.assertEqual(config["train"]["total_timesteps"], budget)
                self.assertEqual(config["train"]["evaluation_interval_epochs"], interval)
                self.assertTrue(config["train"]["anneal_lr"])
                self.assertIsNone(config["train"]["resume_state_path"])
                self.assertIsNone(config["load_model_path"])
                self.assertTrue(config["train"]["fresh_run"])
                self.assertEqual(config["train"]["data_dir"], str((trial_dir / "learner").resolve()))
                self.assertTrue(config["wandb"])
                self.assertEqual(config["wandb_group"], "vanilla_sac_round7")
                self.assertEqual(
                    math.ceil(budget / (2 * refresh_count)),
                    horizon,
                )

    def test_invalid_trial_and_gpu_arguments_fail_before_output_creation(self):
        invalid_arguments = (
            [],
            ["--trial", "96"],
            ["--trial", "97", "--trial", "98"],
            ["--trial", "97", "--gpus", "0"],
            ["--trial", "97", "--gpus", "0,0"],
        )
        with tempfile.TemporaryDirectory() as temporary_dir:
            for index, arguments in enumerate(invalid_arguments):
                with self.subTest(arguments=arguments):
                    output_dir = Path(temporary_dir) / f"invalid-{index}"
                    result = self._run_launcher(
                        [*arguments, "--output-dir", str(output_dir)]
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
