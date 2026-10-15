from contextlib import redirect_stdout
import io
import itertools
import json
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

from project.soft_actor_critic_tuning.scripts import run_sac16_grid


_UNSET_TRIAL_HOURS = object()


class RunSAC16GridTests(unittest.TestCase):
    def _copy_round3_config_with_trial_ids(
        self, temporary_dir, trial_ids, trial_hours=_UNSET_TRIAL_HOURS,
    ):
        source_dir = run_sac16_grid.CONFIG_DIR.parent / "round3"
        config_dir = Path(temporary_dir) / "round3"
        config_dir.mkdir()
        for config_path in (
            run_sac16_grid.RUN_CONFIG_PATH,
            run_sac16_grid.FIXED_CONFIG_PATH,
            run_sac16_grid.TUNABLE_CONFIG_PATH,
            run_sac16_grid.SPECIFIC_CONFIG_PATH,
        ):
            shutil.copyfile(source_dir / config_path.name, config_dir / config_path.name)

        run_config_path = config_dir / run_sac16_grid.RUN_CONFIG_PATH.name
        run_config = run_sac16_grid._load_yaml(run_config_path)
        run_config["launcher"]["trial_ids"] = trial_ids
        if trial_hours is not _UNSET_TRIAL_HOURS:
            run_config["launcher"]["trial_hours"] = trial_hours
        with run_config_path.open("w", encoding="utf-8") as output:
            run_sac16_grid.yaml.safe_dump(run_config, output, sort_keys=False)
        return config_dir

    def test_round_configs_keep_distinct_resource_and_launcher_settings(self):
        round2_config_dir = run_sac16_grid.CONFIG_DIR.parent / "round2"
        _, original_config, original_launcher = run_sac16_grid._load_configs()
        _, round2_config, round2_launcher = run_sac16_grid._load_configs(round2_config_dir)

        self.assertEqual(original_config["train"]["batch_size"], 8192000)
        self.assertEqual(original_config["train"]["total_timesteps"], 10000000000)
        self.assertEqual(original_config["wandb_group"], "vanilla_sac_grid")
        original_axes = run_sac16_grid._grid_axes(original_config)
        self.assertEqual(len(list(itertools.product(*(values for _, values in original_axes)))), 16)
        self.assertEqual([len(group) for group in original_launcher["gpu_pairs"]], [2, 2, 2, 2])
        self.assertEqual(original_launcher["max_concurrent"], 2)

        self.assertEqual(round2_config["train"]["batch_size"], 512000)
        self.assertEqual(round2_config["train"]["total_timesteps"], 1000000000)
        self.assertEqual(round2_config["wandb_group"], "vanilla_sac_round2")
        self.assertEqual([len(group) for group in round2_launcher["gpu_pairs"]], [1] * 8)
        self.assertEqual(round2_launcher["max_concurrent"], 8)

    def test_profiles_without_trial_ids_keep_all_trials_and_smoke_trial_one(self):
        for config_dir in (
            run_sac16_grid.CONFIG_DIR,
            run_sac16_grid.CONFIG_DIR.parent / "round2",
        ):
            with self.subTest(config_dir=config_dir):
                _, config, launcher = run_sac16_grid._load_configs(config_dir)
                trial_count = len(list(itertools.product(*(values for _, values in run_sac16_grid._grid_axes(config)))))
                self.assertNotIn("trial_ids", launcher)
                self.assertEqual(
                    run_sac16_grid._parse_trial_ids(None, trial_count, smoke=False),
                    list(range(1, trial_count + 1)),
                )
                self.assertEqual(run_sac16_grid._parse_trial_ids(None, trial_count, smoke=True), [1])

    def test_round3_config_selects_three_candidates_across_108_trial_grid(self):
        round3_config_dir = run_sac16_grid.CONFIG_DIR.parent / "round3"
        _, round3_config, round3_launcher = run_sac16_grid._load_configs(round3_config_dir)
        axes = run_sac16_grid._grid_axes(round3_config)
        expected_axes = [
            ("train.gamma", [0.95, 0.99, 0.999]),
            ("train.target_entropy", [0.621227, 1.242453, 1.863680]),
            ("train.update_epochs", [3, 15, 50, 100]),
            ("train.replay_update_ratio", [0.1, 0.5, 0.75]),
        ]
        self.assertEqual(axes, expected_axes)
        combinations = list(itertools.product(*(values for _, values in axes)))
        self.assertEqual(len(combinations), 108)
        self.assertEqual(len(set(combinations)), 108)

        selected_ids = list(range(1, 13)) + list(range(49, 61)) + list(range(97, 109))
        self.assertEqual(round3_launcher["trial_ids"], selected_ids)
        self.assertEqual(round3_config["train"]["batch_size"], 512000)
        self.assertEqual(round3_config["train"]["total_timesteps"], 500000000)
        self.assertEqual(round3_config["train"]["initial_alpha"], 0.1)
        self.assertEqual(round3_config["train"]["tau"], 0.005)
        self.assertEqual(round3_config["train"]["gamma"], 0.95)
        self.assertEqual(round3_config["train"]["target_entropy"], 0.621227)
        self.assertEqual(round3_config["train"]["update_epochs"], 3)
        self.assertEqual(round3_config["train"]["replay_update_ratio"], 0.1)
        self.assertEqual(round3_config["train"]["minibatch_size"], 128000)
        self.assertEqual(round3_config["train"]["max_minibatch_size"], 32000)
        self.assertEqual(round3_config["wandb_group"], "vanilla_sac_round3")
        self.assertEqual(round3_launcher["gpu_pairs"], [[gpu_id] for gpu_id in range(8)])
        self.assertEqual(round3_launcher["max_concurrent"], 8)
        self.assertEqual(round3_launcher["trial_hours"], 3)

        sweep_train = round3_config["sweep"]["train"]
        self.assertNotIn("tau", sweep_train)
        self.assertEqual(
            list(sweep_train),
            ["gamma", "target_entropy", "update_epochs", "replay_update_ratio"],
        )
        self.assertEqual(
            [sweep_train[name]["values"] for name in sweep_train],
            [values for _, values in expected_axes],
        )
        expected_parameters = {}
        candidate_values = (
            (0.95, 0.621227),
            (0.99, 1.242453),
            (0.999, 1.863680),
        )
        update_settings = list(itertools.product([3, 15, 50, 100], [0.1, 0.5, 0.75]))
        candidate_trial_ranges = (range(1, 13), range(49, 61), range(97, 109))
        for (gamma, entropy), candidate_ids in zip(candidate_values, candidate_trial_ranges):
            for trial_id, (update_epochs, replay_update_ratio) in zip(candidate_ids, update_settings):
                expected_parameters[trial_id] = {
                    "train.gamma": gamma,
                    "train.target_entropy": entropy,
                    "train.update_epochs": update_epochs,
                    "train.replay_update_ratio": replay_update_ratio,
                }

        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "round3-grid"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    run_sac16_grid.main([
                        "--dry-run",
                        "--config-dir",
                        str(round3_config_dir),
                        "--output-dir",
                        str(output_dir),
                    ]),
                    0,
                )

            with (output_dir / "manifest.json").open(encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)
            config_count = len(list((output_dir / "configs").glob("*.yaml")))
            candidate_taus = [
                run_sac16_grid._load_yaml(
                    Path(manifest["trials"][trial_id - 1]["config_path"]),
                )["train"]["tau"]
                for trial_id in (1, 49, 97)
            ]

        self.assertEqual(manifest["selected_trial_ids"], selected_ids)
        self.assertEqual(len(manifest["trials"]), 108)
        self.assertEqual(
            [trial["trial_id"] for trial in manifest["trials"] if trial["status"] == "planned"],
            selected_ids,
        )
        self.assertEqual(
            [trial["trial_id"] for trial in manifest["trials"] if trial["status"] == "skipped"],
            [trial_id for trial_id in range(1, 109) if trial_id not in set(selected_ids)],
        )
        self.assertEqual(config_count, 36)
        self.assertEqual(manifest["launcher"]["trial_hours"], 3)
        for trial_id in selected_ids:
            trial = manifest["trials"][trial_id - 1]
            self.assertEqual(trial["parameters"], expected_parameters[trial_id])
            self.assertIn("--trial-hours", trial["command"])
            flag_index = trial["command"].index("--trial-hours")
            self.assertEqual(trial["command"][flag_index + 1], "3")

        self.assertEqual(candidate_taus, [0.005, 0.005, 0.005])

    def test_cli_trial_ids_override_configured_selection(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            config_dir = self._copy_round3_config_with_trial_ids(temporary_dir, [1, 49, 97])
            output_dir = Path(temporary_dir) / "cli-override"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    run_sac16_grid.main([
                        "--dry-run",
                        "--config-dir",
                        str(config_dir),
                        "--output-dir",
                        str(output_dir),
                        "--trial-ids",
                        "108",
                    ]),
                    0,
                )

            with (output_dir / "manifest.json").open(encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)

        self.assertEqual(manifest["selected_trial_ids"], [108])
        self.assertEqual(len(manifest["trials"]), 108)
        self.assertEqual(manifest["trials"][107]["status"], "planned")

    def test_smoke_uses_first_configured_trial_id(self):
        round3_config_dir = run_sac16_grid.CONFIG_DIR.parent / "round3"
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "smoke"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    run_sac16_grid.main([
                        "--dry-run",
                        "--smoke",
                        "--config-dir",
                        str(round3_config_dir),
                        "--output-dir",
                        str(output_dir),
                    ]),
                    0,
                )

            with (output_dir / "manifest.json").open(encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)
            smoke_config = run_sac16_grid._load_yaml(Path(manifest["trials"][0]["config_path"]))
            config_count = len(list((output_dir / "configs").glob("*.yaml")))

        self.assertEqual(manifest["selected_trial_ids"], [1])
        self.assertEqual(len(manifest["trials"]), 108)
        self.assertEqual(config_count, 1)
        self.assertEqual(smoke_config["env"]["num_agents"], 3200)
        self.assertEqual(smoke_config["train"]["batch_size"], 128000)
        self.assertEqual(smoke_config["train"]["gamma"], 0.95)
        self.assertEqual(smoke_config["train"]["target_entropy"], 0.621227)
        self.assertEqual(manifest["launcher"]["trial_hours"], 60 / 3600)
        self.assertIn("--trial-hours", manifest["trials"][0]["command"])

    def test_numeric_trial_hours_override_is_exported_and_passed_to_child(self):
        round3_config_dir = run_sac16_grid.CONFIG_DIR.parent / "round3"
        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "numeric-override"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    run_sac16_grid.main([
                        "--dry-run",
                        "--config-dir",
                        str(round3_config_dir),
                        "--output-dir",
                        str(output_dir),
                        "--trial-hours",
                        "2.5",
                    ]),
                    0,
                )
            manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))

        self.assertEqual(manifest["launcher"]["trial_hours"], 2.5)
        self.assertIn("--trial-hours", manifest["trials"][0]["command"])
        flag_index = manifest["trials"][0]["command"].index("--trial-hours")
        self.assertEqual(manifest["trials"][0]["command"][flag_index + 1], "2.5")

    def test_launcher_timeout_validation_accepts_null_and_rejects_invalid_values(self):
        profile = {"grace_seconds": 0}
        self.assertEqual(run_sac16_grid._validate_launcher(profile, [(0,)], 1, None), 0)
        for value in (True, 0, -1, float("nan"), float("inf"), "2.5"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                run_sac16_grid._validate_launcher(profile, [(0,)], 1, value)

    def test_uncapped_parent_launch_ignores_stale_deadline_and_never_times_out(self):
        class ProcessStub:
            pid = 12345

            def __init__(self):
                self.poll_count = 0

            def poll(self):
                self.poll_count += 1
                return None if self.poll_count == 1 else 0

        process = ProcessStub()
        with tempfile.TemporaryDirectory() as temporary_dir:
            config_dir = self._copy_round3_config_with_trial_ids(
                temporary_dir, [1], trial_hours=None,
            )
            output_dir = Path(temporary_dir) / "uncapped-parent"
            with (
                patch.dict(os.environ, {"SAC16_TRIAL_DEADLINE_MONOTONIC": "0"}),
                patch.object(run_sac16_grid, "_memory_preflight", return_value={"fits": True}),
                patch.object(run_sac16_grid, "_track_owned_descendants"),
                patch.object(run_sac16_grid, "_terminate_owned_group"),
                patch.object(run_sac16_grid, "_finish_trial", return_value={"status": "completed"}) as finish_trial,
                patch.object(run_sac16_grid.subprocess, "Popen", return_value=process) as popen,
                patch.object(run_sac16_grid.time, "sleep"),
                patch.object(
                    run_sac16_grid.time,
                    "monotonic",
                    side_effect=[10, 10**6, 10**12, 10**12],
                ) as monotonic,
                redirect_stdout(io.StringIO()),
            ):
                self.assertEqual(
                    run_sac16_grid.main([
                        "--config-dir",
                        str(config_dir),
                        "--output-dir",
                        str(output_dir),
                        "--trial-ids",
                        "1",
                        "--gpu-groups",
                        "0",
                        "--max-concurrent",
                        "1",
                    ]),
                    0,
                )

        command = popen.call_args.args[0]
        child_environment = popen.call_args.kwargs["env"]
        self.assertNotIn("--trial-hours", command)
        self.assertNotIn("SAC16_TRIAL_DEADLINE_MONOTONIC", child_environment)
        finish_trial.assert_called_once()
        self.assertFalse(finish_trial.call_args.args[2])
        self.assertEqual(monotonic.call_count, 4)

    def test_invalid_configured_trial_ids_fail_before_output_creation(self):
        invalid_trial_ids = (
            None,
            "17,26,31",
            [],
            [17, True],
            [17, "26"],
            [17, 26.0],
            [17, 17],
            [0],
            [109],
        )
        for trial_ids in invalid_trial_ids:
            with self.subTest(trial_ids=trial_ids), tempfile.TemporaryDirectory() as temporary_dir:
                config_dir = self._copy_round3_config_with_trial_ids(temporary_dir, trial_ids)
                output_dir = Path(temporary_dir) / "invalid"
                with redirect_stdout(io.StringIO()), self.assertRaises(ValueError):
                    run_sac16_grid.main([
                        "--dry-run",
                        "--config-dir",
                        str(config_dir),
                        "--output-dir",
                        str(output_dir),
                    ])
                self.assertFalse(output_dir.exists())

    def test_config_dir_controls_default_output_root_and_manifest(self):
        round2_config_dir = run_sac16_grid.CONFIG_DIR.parent / "round2"
        cases = (
            (None, run_sac16_grid.PROJECT_DIR / "runs", 16),
            (round2_config_dir, run_sac16_grid.PROJECT_DIR / "runs" / "round2", 64),
            (run_sac16_grid.CONFIG_DIR.parent / "round3", run_sac16_grid.PROJECT_DIR / "runs" / "round3", 108),
        )
        for config_dir, expected_output_root, expected_trial_count in cases:
            with self.subTest(config_dir=config_dir), tempfile.TemporaryDirectory() as temporary_dir:
                output_dir = Path(temporary_dir) / "dry-run"
                arguments = ["--dry-run", "--trial-ids", "1", "--gpu-groups", "0"]
                if config_dir is not None:
                    arguments.extend(("--config-dir", str(config_dir)))

                with (
                    patch.object(run_sac16_grid, "_make_run_dir", return_value=output_dir) as make_run_dir,
                    redirect_stdout(io.StringIO()),
                ):
                    self.assertEqual(run_sac16_grid.main(arguments), 0)

                make_run_dir.assert_called_once_with(expected_output_root, expected_trial_count)
                with (output_dir / "manifest.json").open(encoding="utf-8") as manifest_file:
                    manifest = json.load(manifest_file)
                expected_config_dir = (config_dir or run_sac16_grid.CONFIG_DIR).resolve()
                self.assertEqual(manifest["config_dir"], str(expected_config_dir))
                self.assertEqual(manifest["experiment"], f"SAC{expected_trial_count}-grid")

    def test_round2_grid_dry_run_has_64_unique_trials_and_single_gpu_resources(self):
        round2_config_dir = run_sac16_grid.CONFIG_DIR.parent / "round2"
        _, base_config, _ = run_sac16_grid._load_configs(round2_config_dir)
        axes = run_sac16_grid._grid_axes(base_config)
        combinations = list(itertools.product(*(values for _, values in axes)))

        self.assertEqual(
            [path for path, _ in axes],
            ["train.tau", "train.gamma", "train.target_entropy"],
        )
        self.assertEqual(
            [values for _, values in axes],
            [
                [0.001, 0.005, 0.01, 0.05],
                [0.95, 0.98, 0.99, 0.999],
                [0.621227, 1.242453, 1.863680, 2.435209],
            ],
        )
        self.assertEqual(len(combinations), 64)
        self.assertEqual(len(set(combinations)), 64)
        self.assertEqual(run_sac16_grid._parse_trial_ids("64", 64, smoke=False), [64])
        with self.assertRaisesRegex(ValueError, r"\[1, 64\]"):
            run_sac16_grid._parse_trial_ids("65", 64, smoke=False)

        baseline = base_config["train"]
        self.assertEqual((baseline["tau"], baseline["gamma"], baseline["target_entropy"]), combinations[47])
        self.assertEqual(baseline["initial_alpha"], 0.1)
        self.assertEqual(baseline["update_epochs"], 3)
        self.assertEqual(baseline["replay_update_ratio"], 0.1)

        with tempfile.TemporaryDirectory() as temporary_dir:
            output_dir = Path(temporary_dir) / "round2-grid"
            with redirect_stdout(io.StringIO()):
                self.assertEqual(
                    run_sac16_grid.main([
                        "--dry-run",
                        "--config-dir",
                        str(round2_config_dir),
                        "--output-dir",
                        str(output_dir),
                    ]),
                    0,
                )

            with (output_dir / "manifest.json").open(encoding="utf-8") as manifest_file:
                manifest = json.load(manifest_file)

        self.assertEqual(manifest["experiment"], "SAC64-grid")
        self.assertEqual(manifest["selected_trial_ids"], list(range(1, 65)))
        self.assertEqual(len(manifest["trials"]), 64)
        self.assertTrue(all(len(gpu_group) == 1 for gpu_group in manifest["launcher"]["gpu_pairs"]))
        manifest_parameters = [
            tuple(trial["parameters"][path] for path, _ in axes)
            for trial in manifest["trials"]
        ]
        self.assertEqual(len(set(manifest_parameters)), 64)
        trial_48 = manifest["trials"][47]
        self.assertEqual(trial_48["trial_id"], 48)
        self.assertEqual(
            tuple(trial_48["parameters"][path] for path, _ in axes),
            combinations[47],
        )
        self.assertEqual(len(trial_48["gpu_pair"]), 1)
        self.assertIn("--nproc-per-node=1", trial_48["command"])

    def test_run_directory_names_preserve_16_and_follow_grid_size(self):
        output_root = Path("/tmp/sac-grid-runs")
        self.assertTrue(run_sac16_grid._make_run_dir(output_root).name.startswith("sac16_grid_"))
        self.assertTrue(run_sac16_grid._make_run_dir(output_root, 64).name.startswith("sac64_grid_"))
        self.assertTrue(run_sac16_grid._make_run_dir(output_root, 108).name.startswith("sac108_grid_"))

    def test_missing_config_dir_fails_without_falling_back(self):
        with tempfile.TemporaryDirectory() as temporary_dir:
            missing_config_dir = Path(temporary_dir) / "round2"
            with self.assertRaises(FileNotFoundError):
                run_sac16_grid._load_configs(missing_config_dir)

    def test_gpu_group_parser_accepts_singletons_and_pairs(self):
        self.assertEqual(
            run_sac16_grid._parse_gpu_pairs("0;1;2,3"),
            [(0,), (1,), (2, 3)],
        )

    def test_gpu_group_parser_rejects_invalid_or_reused_ids(self):
        for value in ("", "0,,1", "0,1,2", "-1", "0,0", "0;1,0"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                run_sac16_grid._parse_gpu_pairs(value)

    def test_cli_retains_gpu_pairs_alias(self):
        parser = run_sac16_grid._build_parser()
        self.assertEqual(parser.parse_args(["--gpu-groups", "0"]).gpu_pairs, "0")
        self.assertEqual(parser.parse_args(["--gpu-pairs", "0,1"]).gpu_pairs, "0,1")
        with self.assertRaisesRegex(ValueError, "--gpu-groups"):
            run_sac16_grid.main(["--dry-run", "--trial-ids", "1", "--gpu-groups", ""])

    def test_commands_use_group_size_as_torchrun_rank_count(self):
        singleton = run_sac16_grid._build_command("config.yaml", "trial", (7,), 3)
        pair = run_sac16_grid._build_command("config.yaml", "trial", (2, 3), 3)
        self.assertIn("--nproc-per-node=1", singleton)
        self.assertIn("--nproc-per-node=2", pair)

    def test_memory_preflight_scales_with_actual_ranks(self):
        config = {"train": {"batch_size": 100}, "env": {}}
        with (
            patch.object(run_sac16_grid, "_estimate_observation_size", return_value=923),
            patch.object(run_sac16_grid, "_available_memory_bytes", return_value=10**15),
        ):
            single = run_sac16_grid._memory_preflight(config, [(0,)], 1)
            pair = run_sac16_grid._memory_preflight(config, [(0, 1)], 1)
            mixed = run_sac16_grid._memory_preflight(config, [(0,), (1, 2)], 2)

        self.assertAlmostEqual(
            pair["required_host_memory_bytes"],
            2 * single["required_host_memory_bytes"],
            delta=1,
        )
        self.assertEqual(single["rank_counts_by_trial"], [1])
        self.assertEqual(pair["ranks_per_trial"], 2)
        self.assertEqual(mixed["total_ranks"], 3)
        self.assertEqual(mixed["rank_counts_by_trial"], [1, 2])
        self.assertIsNone(mixed["ranks_per_trial"])


if __name__ == "__main__":
    unittest.main()
