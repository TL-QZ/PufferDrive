import copy
import itertools
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
import unittest

from project.soft_actor_critic_tuning.scripts import run_sac16_grid


REPO_ROOT = Path(__file__).resolve().parents[3]
EXPERIMENT_DIR = REPO_ROOT / "project/soft_actor_critic_target_entropy_scheduling"
CONFIG_DIR = EXPERIMENT_DIR / "configs"
ROUND6_CONFIG_DIR = REPO_ROOT / "project/soft_actor_critic_tuning/configs/round6"
LAUNCHER = EXPERIMENT_DIR / "scripts/launch.sh"
CONFIG_NAMES = (
    "SAC_fixed_config.yaml",
    "SAC_specific_config.yaml",
    "SAC_tunable_config.yaml",
    "SAC_run_config.yaml",
)


class TargetEntropySchedulingLauncherTests(unittest.TestCase):
    def _run_launcher(self, arguments):
        return subprocess.run(
            ["bash", str(LAUNCHER), *map(str, arguments)],
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=False,
        )

    def test_configs_inherit_round6_except_for_entropy_schedule_identity(self):
        round6 = {
            name: run_sac16_grid._load_yaml(ROUND6_CONFIG_DIR / name)
            for name in CONFIG_NAMES
        }
        configured = {
            name: run_sac16_grid._load_yaml(CONFIG_DIR / name)
            for name in CONFIG_NAMES
        }

        self.assertEqual(configured["SAC_fixed_config.yaml"], round6["SAC_fixed_config.yaml"])
        self.assertEqual(configured["SAC_run_config.yaml"], round6["SAC_run_config.yaml"])

        expected_specific = copy.deepcopy(round6["SAC_specific_config.yaml"])
        expected_specific["train"].update(
            learn_alpha=True,
            target_entropy=1.863680,
            target_entropy_schedule="adaptive",
            target_entropy_ema_decay=0.999,
            target_entropy_mean_threshold=0.01,
            target_entropy_std_threshold=0.05,
            target_entropy_decay_factor=0.9,
            target_entropy_required_count=1,
        )
        expected_specific["controlled_exp"]["train"]["target_entropy"]["values"] = [
            0.621227,
            1.242453,
            1.863680,
        ]
        expected_specific["sweep"]["train"]["target_entropy"]["values"] = [
            0.621227,
            1.242453,
            1.863680,
        ]
        self.assertEqual(configured["SAC_specific_config.yaml"], expected_specific)

        expected_tunable = copy.deepcopy(round6["SAC_tunable_config.yaml"])
        expected_tunable["wandb_group"] = "target_entropy_scheduling"
        expected_tunable["train"]["data_dir"] = "experiments/pufferdrive_target_entropy_scheduling"
        self.assertEqual(configured["SAC_tunable_config.yaml"], expected_tunable)

        _, resolved, _ = run_sac16_grid._load_configs(CONFIG_DIR)
        _, round6_resolved, _ = run_sac16_grid._load_configs(ROUND6_CONFIG_DIR)
        expected_resolved = copy.deepcopy(round6_resolved)
        expected_resolved["train"].update(
            learn_alpha=True,
            target_entropy=1.863680,
            target_entropy_schedule="adaptive",
            target_entropy_ema_decay=0.999,
            target_entropy_mean_threshold=0.01,
            target_entropy_std_threshold=0.05,
            target_entropy_decay_factor=0.9,
            target_entropy_required_count=1,
        )
        expected_resolved["controlled_exp"]["train"]["target_entropy"]["values"] = [
            0.621227,
            1.242453,
            1.863680,
        ]
        expected_resolved["sweep"]["train"]["target_entropy"]["values"] = [
            0.621227,
            1.242453,
            1.863680,
        ]
        expected_resolved["wandb_group"] = "target_entropy_scheduling"
        expected_resolved["train"]["data_dir"] = "experiments/pufferdrive_target_entropy_scheduling"
        self.assertEqual(resolved, expected_resolved)

        axes = run_sac16_grid._grid_axes(resolved)
        self.assertEqual(math.prod(len(values) for _, values in axes), 108)
        for trial_id, refresh_ratio in ((97, 0.1), (98, 0.5)):
            combination = next(itertools.islice(itertools.product(*(values for _, values in axes)), trial_id - 1, None))
            parameters = {path: value for (path, _), value in zip(axes, combination)}
            self.assertEqual(parameters["train.gamma"], 0.999)
            self.assertAlmostEqual(parameters["train.target_entropy"], 0.75 * math.log(12), places=6)
            self.assertEqual(parameters["train.update_epochs"], 3)
            self.assertEqual(parameters["train.replay_update_ratio"], refresh_ratio)

    def test_selected_trials_dry_run_with_round6_budgets_and_adaptive_entropy(self):
        for trial_id, refresh_ratio, interval, budget in (
            (97, 0.1, 10, 2000000000),
            (98, 0.5, 2, 4000000000),
        ):
            with self.subTest(trial_id=trial_id), tempfile.TemporaryDirectory() as temporary_dir:
                output_dir = Path(temporary_dir) / f"target-entropy-trial-{trial_id}"
                result = self._run_launcher(
                    [
                        "--trial",
                        str(trial_id),
                        "--dry-run",
                        "--output-dir",
                        output_dir,
                    ]
                )
                self.assertEqual(result.returncode, 0, result.stderr)

                manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
                planned = [trial for trial in manifest["trials"] if trial["status"] == "planned"]
                skipped = [trial for trial in manifest["trials"] if trial["status"] == "skipped"]
                self.assertEqual(manifest["selected_trial_ids"], [trial_id])
                self.assertEqual(manifest["config_dir"], str(CONFIG_DIR.resolve()))
                self.assertEqual(manifest["output_dir"], str(output_dir.resolve()))
                self.assertEqual(len(manifest["trials"]), 108)
                self.assertEqual(len(planned), 1)
                self.assertEqual(len(skipped), 107)
                self.assertEqual(manifest["launcher"]["gpu_pairs"], [[0, 1]])
                self.assertEqual(manifest["launcher"]["max_concurrent"], 1)
                self.assertIsNone(manifest["launcher"]["trial_hours"])

                trial = planned[0]
                self.assertEqual(trial["trial_id"], trial_id)
                self.assertEqual(trial["parameters"]["train.replay_update_ratio"], refresh_ratio)
                self.assertAlmostEqual(trial["parameters"]["train.target_entropy"], 0.75 * math.log(12), places=6)
                self.assertEqual(trial["evaluation_interval_epochs"], interval)
                self.assertEqual(trial["total_timesteps"], budget)
                config = run_sac16_grid._load_yaml(Path(trial["config_path"]))
                self.assertAlmostEqual(config["train"]["target_entropy"], 0.75 * math.log(12), places=6)
                self.assertEqual(config["train"]["target_entropy_schedule"], "adaptive")
                self.assertEqual(config["train"]["target_entropy_ema_decay"], 0.999)
                self.assertEqual(config["train"]["target_entropy_mean_threshold"], 0.01)
                self.assertEqual(config["train"]["target_entropy_std_threshold"], 0.05)
                self.assertEqual(config["train"]["target_entropy_decay_factor"], 0.9)
                self.assertEqual(config["train"]["target_entropy_required_count"], 1)
                self.assertTrue(config["train"]["learn_alpha"])
                self.assertTrue(config["train"]["anneal_lr"])
                self.assertEqual(config["train"]["total_timesteps"], budget)
                self.assertEqual(config["train"]["evaluation_interval_epochs"], interval)
                self.assertEqual(config["train"]["replay_update_ratio"], refresh_ratio)
                self.assertEqual(config["wandb_group"], "target_entropy_scheduling")
                self.assertTrue(config["wandb"])
                self.assertEqual(
                    config["wandb_project"],
                    run_sac16_grid._load_yaml(ROUND6_CONFIG_DIR / "SAC_tunable_config.yaml")[
                        "wandb_project"
                    ],
                )
                trial_dir = output_dir / f"trial_{trial_id:02d}"
                self.assertEqual(config["train"]["data_dir"], str(trial_dir.resolve() / "learner"))
                self.assertTrue(config["train"]["fresh_run"])
                self.assertIsNone(config["train"]["resume_state_path"])
                self.assertIsNone(config["load_model_path"])

    def test_default_output_paths_are_unique_and_inside_experiment_runs(self):
        output_dirs = []
        try:
            for _ in range(2):
                result = self._run_launcher(["--trial", "97", "--dry-run", "--no-wandb"])
                self.assertEqual(result.returncode, 0, result.stderr)
                dry_run_line = next(
                    line for line in result.stdout.splitlines() if line.startswith("Dry run wrote ")
                )
                manifest_path = Path(dry_run_line.rsplit(" ", 1)[-1])
                self.assertTrue(manifest_path.is_file())
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                output_dir = Path(manifest["output_dir"])
                self.assertEqual(output_dir.parent, (EXPERIMENT_DIR / "runs").resolve())
                self.assertRegex(output_dir.name, re.compile(r"^target_entropy_scheduling_\d{8}T\d{6}\.\d+_\d+$"))
                output_dirs.append(output_dir)

            self.assertNotEqual(output_dirs[0], output_dirs[1])
        finally:
            for output_dir in output_dirs:
                shutil.rmtree(output_dir, ignore_errors=True)

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
                        [*arguments, "--output-dir", output_dir]
                    )
                    self.assertNotEqual(result.returncode, 0)
                    self.assertFalse(output_dir.exists())


if __name__ == "__main__":
    unittest.main()
