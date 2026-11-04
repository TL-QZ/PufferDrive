import copy
import json
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

from project.soft_actor_critic_tuning.scripts import run_sac16_grid


REPO_ROOT = Path(__file__).resolve().parents[3]
SOURCE_CONFIG_ROOT = REPO_ROOT / "project/soft_actor_critic_multi_step_return/configs"
PQL_PROJECT_ROOT = REPO_ROOT / "project/soft_actor_critic_multi_step_return_PQL"
VARIANTS = (
    {
        "source": "round4_1_nstep2_tes_sac",
        "pql": "round4_1_pql",
        "source_launcher": REPO_ROOT / "project/soft_actor_critic_multi_step_return/scripts/launch_round4_1.sh",
        "pql_launcher": PQL_PROJECT_ROOT / "scripts/launch_round4_1_pql.sh",
        "trial_selectors": (("97", 97), ("98", 98)),
        "wandb_group": "round4_1_pql",
        "data_dir": "experiments/pufferdrive_round4_1_pql",
        "evaluation_intervals": {97: 10, 98: 2},
        "total_timesteps": {97: 2000000000, 98: 4000000000},
    },
    {
        "source": "round2_enhance",
        "pql": "round2_enhance_pql",
        "source_launcher": REPO_ROOT / "project/soft_actor_critic_multi_step_return/scripts/launch_round2_enhance.sh",
        "pql_launcher": PQL_PROJECT_ROOT / "scripts/launch_round2_enhance_pql.sh",
        "trial_selectors": (("01", 1), ("1", 1)),
        "wandb_group": "sac_multi_step_return_round2_enhance_pql",
        "data_dir": "experiments/pufferdrive_round2_enhance_pql",
        "evaluation_intervals": {1: 2},
        "total_timesteps": {1: 4000000000},
    },
)


class PqlConfigAndLauncherTests(unittest.TestCase):
    def _run_launcher(self, launcher_path, arguments):
        return subprocess.run(
            ["bash", str(launcher_path), *map(str, arguments)],
            cwd=REPO_ROOT,
            env=os.environ.copy(),
            capture_output=True,
            text=True,
            check=False,
        )

    def test_merged_configs_preserve_source_recipe_except_pql_and_isolated_paths(self):
        for variant in VARIANTS:
            with self.subTest(variant=variant["pql"]):
                source_dir = SOURCE_CONFIG_ROOT / variant["source"]
                pql_dir = PQL_PROJECT_ROOT / "configs" / variant["pql"]
                _, source_resolved, source_launcher = run_sac16_grid._load_configs(source_dir)
                _, pql_resolved, pql_launcher = run_sac16_grid._load_configs(pql_dir)

                expected = copy.deepcopy(source_resolved)
                expected["train"].update(
                    {
                        "n_steps": 1,
                        "return_estimator": "pql",
                        "pql_horizon": 128,
                        "pql_lambda": 0.95,
                        "pql_entropy_convention": "soft_each_return_endpoint",
                        "critic_entropy_in_target": True,
                        "data_dir": variant["data_dir"],
                    }
                )
                expected["wandb_group"] = variant["wandb_group"]
                expected["eval"]["benchmark_config"] = (
                    f"project/soft_actor_critic_multi_step_return_PQL/configs/{variant['pql']}/evaluation_benchmarks.yaml"
                )
                self.assertEqual(pql_resolved, expected)
                self.assertEqual(pql_launcher, source_launcher)
                self.assertTrue(pql_resolved["wandb"])

                source_catalog = source_dir / "evaluation_benchmarks.yaml"
                pql_catalog = pql_dir / "evaluation_benchmarks.yaml"
                self.assertEqual(source_catalog.read_bytes(), pql_catalog.read_bytes())

    def test_source_and_pql_launchers_dry_run_every_supported_trial_selector(self):
        for variant in VARIANTS:
            for selector, trial_id in variant["trial_selectors"]:
                with self.subTest(variant=variant["pql"], selector=selector), tempfile.TemporaryDirectory() as temporary_dir:
                    temporary_root = Path(temporary_dir)
                    source_output = temporary_root / "source_output"
                    pql_output = temporary_root / "pql_output"
                    common_arguments = ["--trial", selector, "--dry-run"]

                    source_result = self._run_launcher(
                        variant["source_launcher"],
                        [*common_arguments, "--output-dir", source_output],
                    )
                    self.assertEqual(source_result.returncode, 0, source_result.stderr)
                    pql_result = self._run_launcher(
                        variant["pql_launcher"],
                        [*common_arguments, "--output-dir", pql_output],
                    )
                    self.assertEqual(pql_result.returncode, 0, pql_result.stderr)

                    for output_dir in (source_output, pql_output):
                        with (output_dir / "manifest.json").open(encoding="utf-8") as manifest_file:
                            manifest = json.load(manifest_file)
                        self.assertEqual(manifest["selected_trial_ids"], [trial_id])
                        self.assertEqual(
                            sum(trial["status"] == "planned" for trial in manifest["trials"]),
                            1,
                        )

                    trial_config_name = f"trial_{trial_id:02d}.yaml"
                    source_trial = run_sac16_grid._load_yaml(
                        source_output / "configs" / trial_config_name
                    )
                    pql_trial = run_sac16_grid._load_yaml(
                        pql_output / "configs" / trial_config_name
                    )
                    expected = copy.deepcopy(source_trial)
                    expected["train"].update(
                        {
                            "n_steps": 1,
                            "return_estimator": "pql",
                            "pql_horizon": 128,
                            "pql_lambda": 0.95,
                            "pql_entropy_convention": "soft_each_return_endpoint",
                            "critic_entropy_in_target": True,
                            "data_dir": str(pql_output / f"trial_{trial_id:02d}" / "learner"),
                        }
                    )
                    expected["run_name"] = f"{pql_output.name}_trial_{trial_id:02d}"
                    expected["wandb_group"] = variant["wandb_group"]
                    expected["eval"]["benchmark_config"] = (
                        f"project/soft_actor_critic_multi_step_return_PQL/configs/{variant['pql']}/evaluation_benchmarks.yaml"
                    )
                    self.assertEqual(pql_trial, expected)
                    self.assertEqual(
                        pql_trial["train"]["evaluation_interval_epochs"],
                        variant["evaluation_intervals"][trial_id],
                    )
                    self.assertEqual(
                        pql_trial["train"]["total_timesteps"],
                        variant["total_timesteps"][trial_id],
                    )

    def test_launcher_help_lists_source_trial_selectors(self):
        for variant, expected_usage in (
            (VARIANTS[0], "--trial 97|98"),
            (VARIANTS[1], "--trial 01"),
        ):
            with self.subTest(variant=variant["pql"]):
                result = self._run_launcher(variant["pql_launcher"], ["--help"])
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn(expected_usage, result.stdout)
                self.assertIn("--dry-run", result.stdout)
                self.assertIn("--gpus ID,ID", result.stdout)


if __name__ == "__main__":
    unittest.main()
