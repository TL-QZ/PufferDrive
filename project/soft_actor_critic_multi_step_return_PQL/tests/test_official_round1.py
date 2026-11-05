import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from project.soft_actor_critic_tuning.scripts import run_sac16_grid


REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = REPO_ROOT / "project/soft_actor_critic_multi_step_return_PQL"
CONFIG_ROOT = PROJECT_ROOT / "configs/PQL_official_round1"
LAUNCHER = PROJECT_ROOT / "scripts/launch_PQL_official_round1.sh"


class PqlOfficialRoundOneTests(unittest.TestCase):
    def _run_launcher(self, arguments):
        return subprocess.run(
            ["bash", str(LAUNCHER), *map(str, arguments)],
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=False,
        )

    def _dry_run(self, selector, output_dir):
        result = self._run_launcher(
            ["--trial", selector, "--dry-run", "--output-dir", output_dir]
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        with (output_dir / "manifest.json").open(encoding="utf-8") as manifest_file:
            manifest = json.load(manifest_file)
        self.assertEqual(manifest["selected_trial_ids"], [98 if selector == "98" else 1])
        self.assertEqual(
            sum(trial["status"] == "planned" for trial in manifest["trials"]), 1
        )
        trial_id = manifest["selected_trial_ids"][0]
        config = run_sac16_grid._load_yaml(
            output_dir / "configs" / f"trial_{trial_id:02d}.yaml"
        )
        return config

    def test_dry_runs_resolve_to_same_recipe_except_replacement_and_identity(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            temporary_root = Path(temporary_dir)
            config_98 = self._dry_run("98", temporary_root / "run_98")
            config_01 = self._dry_run("01", temporary_root / "run_01")

        for config, expected_ratio in ((config_98, 0.5), (config_01, 0.25)):
            self.assertEqual(config["train"]["replay_update_ratio"], expected_ratio)
            self.assertEqual(config["env"]["num_agents"], 800)
            self.assertEqual(config["vec"]["num_envs"], 20)
            self.assertEqual(config["train"]["batch_size"], 2048000)
            self.assertEqual(config["train"]["n_steps"], 1)
            self.assertEqual(config["train"]["return_estimator"], "pql")
            self.assertEqual(config["train"]["pql_horizon"], 128)
            self.assertEqual(config["train"]["pql_lambda"], 0.95)
            self.assertEqual(config["train"]["learning_rate"], 0.0001)
            self.assertEqual(config["train"]["critic_learning_rate"], 0.0001)
            self.assertEqual(config["train"]["alpha_learning_rate"], 0.00007)
            self.assertEqual(config["train"]["target_entropy_std_threshold"], 0.025)
            self.assertEqual(config["train"]["target_entropy_required_count"], 2000)
            self.assertEqual(config["train"]["evaluation_interval_epochs"], 2)
            self.assertEqual(config["train"]["total_timesteps"], 4000000000)
            self.assertTrue(config["train"]["anneal_lr"])
            self.assertTrue(config["train"]["log_collapse_diagnostics"])
            self.assertEqual(config["wandb_group"], "PQL_official_round1")
            self.assertEqual(
                config["eval"]["benchmark_config"],
                "project/soft_actor_critic_multi_step_return_PQL/configs/PQL_official_round1/evaluation_benchmarks.yaml",
            )

        common_98 = copy.deepcopy(config_98)
        common_01 = copy.deepcopy(config_01)
        for config in (common_98, common_01):
            config.pop("controlled_exp", None)
            config.pop("run_name", None)
            config["train"].pop("replay_update_ratio")
            config["train"].pop("data_dir")
        self.assertEqual(common_98, common_01)

    def test_both_trials_use_the_same_round4_1_benchmark_catalog(self):
        source_catalog = PROJECT_ROOT / "configs/round4_1_pql/evaluation_benchmarks.yaml"
        official_catalog = CONFIG_ROOT / "evaluation_benchmarks.yaml"
        self.assertEqual(official_catalog.read_bytes(), source_catalog.read_bytes())

    def test_launcher_accepts_code_name_aliases_and_documents_dry_run(self):
        for selector, expected_id in (("98", "98"), ("01", "1"), ("1", "1")):
            with self.subTest(selector=selector), tempfile.TemporaryDirectory() as temporary_dir:
                output_dir = Path(temporary_dir) / "dry_run"
                result = self._run_launcher(
                    ["--trial", selector, "--dry-run", "--output-dir", output_dir]
                )
                self.assertEqual(result.returncode, 0, result.stderr)
                with (output_dir / "manifest.json").open(encoding="utf-8") as manifest_file:
                    manifest = json.load(manifest_file)
                self.assertEqual(manifest["selected_trial_ids"], [int(expected_id)])

        help_result = self._run_launcher(["--help"])
        self.assertEqual(help_result.returncode, 0, help_result.stderr)
        self.assertIn("--trial 98|01", help_result.stdout)
        self.assertIn("--dry-run", help_result.stdout)
        self.assertIn("--gpus ID,ID", help_result.stdout)


if __name__ == "__main__":
    unittest.main()
