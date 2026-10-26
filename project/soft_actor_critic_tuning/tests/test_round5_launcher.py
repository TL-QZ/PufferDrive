import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from project.soft_actor_critic_tuning.scripts import run_sac16_grid


REPO_ROOT = Path(__file__).resolve().parents[3]
SCRIPT_PATH = REPO_ROOT / "project/soft_actor_critic_tuning/scripts/launch_round5.sh"
CONFIG_ROOT = REPO_ROOT / "project/soft_actor_critic_tuning/configs"


class Round5LauncherTests(unittest.TestCase):
    def _run_launcher(self, arguments):
        return subprocess.run(
            ["bash", str(SCRIPT_PATH), *map(str, arguments)],
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=False,
        )

    def test_round5_launcher_dry_runs_selected_rate_pairs_on_six_gpus(self):
        expected_trial_settings = {
            1: (0.1, 0.0005, 0.0005, 0.0003, 10, [0]),
            14: (0.1, 0.0001, 0.0001, 0.0001, 10, [1]),
            27: (0.1, 0.00005, 0.00005, 0.00003, 10, [2]),
            28: (0.5, 0.0005, 0.0005, 0.0003, 2, [3]),
            41: (0.5, 0.0001, 0.0001, 0.0001, 2, [4]),
            54: (0.5, 0.00005, 0.00005, 0.00003, 2, [5]),
        }
        expected_ids = list(expected_trial_settings)

        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "round5-dry-run"
            result = self._run_launcher(
                ["--dry-run", "--output-dir", str(output_dir)]
            )
            self.assertEqual(result.returncode, 0, result.stderr)

            manifest = json.loads(
                (output_dir / "manifest.json").read_text(encoding="utf-8")
            )
            planned_trials = [
                trial for trial in manifest["trials"] if trial["status"] == "planned"
            ]
            resolved_configs = sorted((output_dir / "configs").glob("*.yaml"))

            self.assertTrue(manifest["dry_run"])
            self.assertEqual(manifest["selected_trial_ids"], expected_ids)
            self.assertEqual(len(manifest["trials"]), 54)
            self.assertEqual(len(planned_trials), 6)
            self.assertEqual(len(resolved_configs), 6)
            self.assertEqual(
                manifest["memory_preflight"], {"status": "not_run_dry_run"}
            )

            expected_axes = [
                {"path": "train.replay_update_ratio", "values": [0.1, 0.5]},
                {"path": "train.learning_rate", "values": [0.0005, 0.0001, 0.00005]},
                {"path": "train.critic_learning_rate", "values": [0.0005, 0.0001, 0.00005]},
                {"path": "train.alpha_learning_rate", "values": [0.0003, 0.0001, 0.00003]},
            ]
            self.assertEqual(manifest["grid_axes"], expected_axes)

            launcher = manifest["launcher"]
            self.assertIsNone(launcher["trial_hours"])
            self.assertEqual(launcher["gpu_pairs"], [[0], [1], [2], [3], [4], [5]])
            self.assertEqual(launcher["max_concurrent"], 6)

            planned_by_id = {trial["trial_id"]: trial for trial in planned_trials}
            configs_by_name = {
                path.name: run_sac16_grid._load_yaml(path) for path in resolved_configs
            }
            for trial_id, expected in expected_trial_settings.items():
                ratio, actor_lr, critic_lr, alpha_lr, eval_interval, gpu_pair = expected
                with self.subTest(trial_id=trial_id):
                    trial = planned_by_id[trial_id]
                    self.assertEqual(trial["gpu_pair"], gpu_pair)
                    self.assertEqual(trial["evaluation_interval_epochs"], eval_interval)
                    self.assertNotIn("--trial-hours", trial["command"])
                    self.assertEqual(
                        trial["parameters"],
                        {
                            "train.replay_update_ratio": ratio,
                            "train.learning_rate": actor_lr,
                            "train.critic_learning_rate": critic_lr,
                            "train.alpha_learning_rate": alpha_lr,
                        },
                    )

                    config_name = f"trial_{trial_id:02d}.yaml"
                    config = configs_by_name[config_name]
                    self.assertEqual(config["train"]["batch_size"], 512000)
                    self.assertEqual(config["train"]["minibatch_size"], 128000)
                    self.assertEqual(config["train"]["max_minibatch_size"], 32000)
                    self.assertEqual(config["train"]["total_timesteps"], 2000000000)
                    self.assertFalse(config["train"]["anneal_lr"])
                    self.assertTrue(config["train"]["fresh_run"])
                    self.assertEqual(
                        config["train"]["evaluation_interval_epochs"], eval_interval
                    )

    def test_round5_fixed_environment_matches_round4_1(self):
        round4_1_fixed = run_sac16_grid._load_yaml(
            CONFIG_ROOT / "round4_1" / "SAC_fixed_config.yaml"
        )
        round5_fixed = run_sac16_grid._load_yaml(
            CONFIG_ROOT / "round5" / "SAC_fixed_config.yaml"
        )

        self.assertEqual(round5_fixed["env"], round4_1_fixed["env"])


if __name__ == "__main__":
    unittest.main()
