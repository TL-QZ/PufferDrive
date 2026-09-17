"""CLI compatibility and fail-fast checks for selectable baseline plots."""

import itertools
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import patch

from project.metric_analysis import plot_finetune_comparison as comparison


class ModelComparisonTests(unittest.TestCase):
    def setUp(self):
        self.scratch = tempfile.TemporaryDirectory(prefix="model-comparison-")
        self.addCleanup(self.scratch.cleanup)
        self.root = Path(self.scratch.name)
        metrics = {metric: 0.5 for group in comparison.METRIC_GROUPS.values() for metric in group}
        metrics["total_infractions"] = 100
        self.summary = self.root / "evaluation_summary.json"
        self.summary.write_text(json.dumps({
            "num_scenarios": 1000, "num_episodes": 1000, "metrics_mean": metrics,
        }))

    def summary_args(self, models, legacy=False):
        arguments = []
        for model in models:
            prefix = comparison.LEGACY_STAGES[model] if legacy else model.replace("_", "-")
            for benchmark in comparison.BENCHMARKS:
                arguments.extend([
                    f"--{prefix}-{benchmark.replace('_', '-')}-json", str(self.summary),
                ])
        return arguments

    def parse(self, arguments):
        with patch("sys.argv", ["plot_comparison", *arguments]):
            return comparison.parse_args()

    def test_legacy_aliases_and_defaults(self):
        arguments = self.parse(self.summary_args(comparison.LEGACY_STAGES, legacy=True))
        self.assertTrue(arguments.legacy_comparison)
        paths = comparison.resolve_summary_paths(arguments)
        self.assertEqual(list(paths), list(comparison.LEGACY_STAGES))
        self.assertEqual(len(comparison.load_comparison_metrics(paths)), 6)

    def test_all_seven_subsets_and_reverse_order(self):
        models = list(comparison.MODEL_STAGES)
        selections = [selection for count in range(1, 4)
                      for selection in itertools.combinations(models, count)]
        selections.append(tuple(reversed(models)))
        for selection in selections:
            with self.subTest(selection=selection):
                args = self.parse(["--models", *selection, *self.summary_args(selection)])
                paths = comparison.resolve_summary_paths(args)
                self.assertEqual(list(paths), list(selection))
                records = comparison.load_comparison_metrics(paths)
                self.assertEqual(len(records), 3 * len(selection))
                self.assertEqual(records[0]["model_stage"], selection[0])

    def test_invalid_model_selection(self):
        for models in ([], ["unknown"], ["carla_trained", "carla_trained"]):
            with self.subTest(models=models), patch("sys.stderr"):
                with self.assertRaises(SystemExit):
                    self.parse(["--models", *models])

    def test_missing_and_unselected_paths(self):
        with self.assertRaisesRegex(ValueError, "carla_trained / carla"):
            comparison.resolve_summary_paths(self.parse(["--models", "carla_trained"]))
        args = self.parse([
            "--models", "carla_trained",
            *self.summary_args(["carla_trained", "nuplan_self_play"]),
        ])
        with self.assertRaisesRegex(ValueError, "unselected model nuplan_self_play"):
            comparison.resolve_summary_paths(args)

    def test_bad_summary_never_creates_output(self):
        valid = self.summary.read_text()
        missing_metric = json.loads(valid)
        del missing_metric["metrics_mean"]["score"]
        incomplete = json.loads(valid)
        incomplete["num_episodes"] = 999
        nonfinite = json.loads(valid)
        nonfinite["metrics_mean"]["score"] = float("nan")
        contents = ["invalid json", json.dumps(missing_metric), json.dumps(incomplete),
                    json.dumps(nonfinite), None]
        output = self.root / "output"
        arguments = ["--models", "carla_trained", *self.summary_args(["carla_trained"]),
                     "--output-dir", str(output)]
        for content in contents:
            if content is None:
                self.summary.unlink()
            else:
                self.summary.write_text(content)
            with self.subTest(content=content), patch("sys.argv", ["plot", *arguments]):
                with self.assertRaisesRegex(SystemExit, "carla_trained / carla"):
                    comparison.main()
                self.assertFalse(output.exists())

    def test_launcher_discovery_and_ambiguity(self):
        project = self.root / "project/baseline_run_sync_2026-08-24"
        script = project / "plot/plot_self_play_comparison.sh"
        script.parent.mkdir(parents=True)
        shutil.copy(comparison.REPO_ROOT / script.relative_to(self.root), script)
        activate = self.root / ".venv/bin/activate"
        activate.parent.mkdir(parents=True)
        # Capture the final Python arguments without rendering or requiring a GPU.
        activate.write_text('python() { printf "%s\\n" "$@"; }\n')
        run = self.root / (
            "experiments/baseline_run_sync_2026-08-24/nuplan_selfplay_dt03/"
            "baseline_run_sync_2026-08-24_nuplan_selfplay_dt03_fixture_seed0"
        )
        for benchmark in comparison.BENCHMARKS:
            target = run / f"eval/{benchmark}_final_model_mean_metrics/first/evaluation_summary.json"
            target.parent.mkdir(parents=True)
            shutil.copy(self.summary, target)
        command = ["bash", str(script), "0", "--models", "nuplan_self_play"]
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("--nuplan-self-play-carla-json", result.stdout)
        self.assertIn("self_play_comparison/seed0/nuplan_self_play", result.stdout)
        # An unavailable unselected CARLA model must not block self-play-only plots.
        result = subprocess.run(["bash", str(script), "0"], capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("carla_trained", result.stderr)
        duplicate = run / "eval/carla_final_model_mean_metrics/second/evaluation_summary.json"
        duplicate.parent.mkdir()
        shutil.copy(self.summary, duplicate)
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("found 2", result.stderr)
        self.assertIn(str(duplicate), result.stderr)
        duplicate.unlink()
        original = run / "eval/carla_final_model_mean_metrics/first/evaluation_summary.json"
        original.unlink()
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("nuplan_self_play / carla summary; found 0", result.stderr)
        shutil.copy(self.summary, original)
        second_run = run.with_name(run.name.replace("fixture", "duplicate"))
        second_run.mkdir()
        result = subprocess.run(command, capture_output=True, text=True)
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("run for seed 0; found 2", result.stderr)
        self.assertIn(str(second_run), result.stderr)
        self.assertFalse((project / "output").exists())


if __name__ == "__main__":
    unittest.main()
