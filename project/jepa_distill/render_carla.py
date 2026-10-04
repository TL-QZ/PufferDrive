"""Render selected CARLA failures from a completed Condition B evaluation."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import shutil
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

from .evaluate_carla import (
    ChunkPolicyAdapter,
    _load_student_checkpoint,
    parse_execution_ks,
    validate_checkpoint_metadata,
)


EXPECTED_FAILURE_COLUMNS = {
    "all_infractions": (
        "collision_rate",
        "at_fault_collision_rate",
        "offroad_rate",
        "red_light_violation_rate",
    ),
    "collision": ("collision_rate",),
    "at_fault_collision": ("at_fault_collision_rate",),
    "offroad": ("offroad_rate",),
    "red_light": ("red_light_violation_rate",),
}
FAILURE_MODE_COLUMNS = {
    "all_infractions": "all_infractions",
    "collision": "collision_rate",
    "at_fault_collision": "at_fault_collision_rate",
    "offroad": "offroad_rate",
    "red_light": "red_light_violation_rate",
}
EXPECTED_TRAINING_DT = 0.3
EXPECTED_EVALUATION_DT = 0.1
EXPECTED_CHUNK_LENGTH = 4
EXPECTED_SLOT_HOLD_STEPS = 3


def parse_max_failures(value: str | int | None) -> Optional[int]:
    """Parse a positive failure limit, or an explicit unlimited value."""

    if value is None:
        return None
    if isinstance(value, bool):
        raise argparse.ArgumentTypeError("--max-failures must be a positive integer, all, or null")
    if isinstance(value, int):
        parsed_value = value
    elif isinstance(value, str) and value.lower() in ("all", "null"):
        return None
    else:
        try:
            parsed_value = int(value)
        except (TypeError, ValueError) as exc:
            raise argparse.ArgumentTypeError(
                "--max-failures must be a positive integer, all, or null"
            ) from exc
    if parsed_value <= 0:
        raise argparse.ArgumentTypeError("--max-failures must be a positive integer, all, or null")
    return parsed_value


def parse_workers(value: str | int) -> int:
    if isinstance(value, bool):
        raise argparse.ArgumentTypeError("--workers must be a positive integer")
    try:
        worker_count = int(value)
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("--workers must be a positive integer") from exc
    if worker_count <= 0:
        raise argparse.ArgumentTypeError("--workers must be a positive integer")
    return worker_count


def _read_json_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Could not read {label} JSON: {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} JSON must contain an object: {path}")
    return dict(value)


def _read_yaml_mapping(path: Path, label: str) -> dict[str, Any]:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - project dependency
        raise RuntimeError("PyYAML is required to inspect saved evaluation configuration") from exc
    try:
        with path.open("r", encoding="utf-8") as config_file:
            value = yaml.safe_load(config_file)
    except (OSError, yaml.YAMLError) as exc:
        raise ValueError(f"Could not read {label} YAML: {path}: {exc}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} YAML must contain a mapping: {path}")
    return dict(value)


def _number(value: Any, label: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a finite number")
    try:
        number_value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a finite number") from exc
    if not math.isfinite(number_value):
        raise ValueError(f"{label} must be a finite number")
    return number_value


def _positive_integer(value: Any, label: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    try:
        integer_value = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be a positive integer") from exc
    if integer_value <= 0 or str(value).strip() not in (str(integer_value), f"{integer_value}.0"):
        raise ValueError(f"{label} must be a positive integer")
    return integer_value


def _validate_scenario_rows(
    rows: Sequence[Mapping[str, str]],
    benchmark_environment: Mapping[str, Any],
    execution_k: int,
) -> None:
    map_dir_value = benchmark_environment.get("map_dir")
    if not isinstance(map_dir_value, str) or not map_dir_value.strip():
        raise ValueError("resolved benchmark env.map_dir must be a path")
    map_dir = Path(map_dir_value).expanduser().resolve()
    if map_dir.is_file() and map_dir.suffix == ".bin":
        available_map_stems = {map_dir.stem}
        available_map_count = 1
    elif map_dir.is_dir():
        map_paths = tuple(map_dir.glob("*.bin"))
        available_map_stems = {map_path.stem for map_path in map_paths}
        available_map_count = len(map_paths)
    else:
        raise FileNotFoundError(f"Resolved CARLA map path does not exist: {map_dir}")
    configured_map_count = _positive_integer(
        benchmark_environment.get("num_maps"), "resolved benchmark env.num_maps"
    )
    if configured_map_count > available_map_count:
        raise ValueError(
            f"Resolved benchmark requests {configured_map_count} CARLA maps, "
            f"but {map_dir} contains {available_map_count}"
        )

    for row_number, row in enumerate(rows, start=2):
        map_name = Path(row["map_name"]).stem
        if map_name not in available_map_stems:
            raise ValueError(
                f"k={execution_k} CSV row {row_number} references unknown CARLA map "
                f"{row['map_name']!r} in {map_dir}"
            )
        seed_value = row.get("seed", "")
        try:
            episode_seed = int(seed_value)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"CSV row {row_number} seed must be an integer") from exc
        if episode_seed < -(2**63) or episode_seed >= 2**63:
            raise ValueError(f"CSV row {row_number} seed is outside the supported signed 64-bit range")


def _validate_protocol(protocol: Mapping[str, Any], execution_k: int) -> None:
    expected_hold = 1 if execution_k == 1 else EXPECTED_SLOT_HOLD_STEPS
    expected_replan = 1 if execution_k == 1 else EXPECTED_SLOT_HOLD_STEPS * execution_k
    protocol_k = protocol.get("execution_k")
    if isinstance(protocol_k, bool) or protocol_k != execution_k:
        raise ValueError(
            f"k={execution_k} input has mismatched protocol.execution_k="
            f"{protocol.get('execution_k')!r}"
        )
    if protocol.get("model_chunk_length") != EXPECTED_CHUNK_LENGTH:
        raise ValueError("protocol model_chunk_length must be 4")
    if not math.isclose(
        _number(protocol.get("training_dt"), "protocol.training_dt"),
        EXPECTED_TRAINING_DT,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(f"protocol training_dt must be {EXPECTED_TRAINING_DT}")
    if not math.isclose(
        _number(protocol.get("evaluation_dt"), "protocol.evaluation_dt"),
        EXPECTED_EVALUATION_DT,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(f"protocol evaluation_dt must be {EXPECTED_EVALUATION_DT}")
    if isinstance(protocol.get("hold_steps_per_slot"), bool) or protocol.get("hold_steps_per_slot") != expected_hold:
        raise ValueError(f"k={execution_k} protocol has an unexpected slot hold")
    if isinstance(protocol.get("replan_interval_steps"), bool) or protocol.get("replan_interval_steps") != expected_replan:
        raise ValueError(f"k={execution_k} protocol has an unexpected replan interval")
    if protocol.get("benchmark") != "carla":
        raise ValueError("rendering input protocol must name the carla benchmark")
    if protocol.get("action_selection") != "mean" or protocol.get("action_type") != "continuous":
        raise ValueError("rendering input must use mean continuous controls")
    _positive_integer(protocol.get("num_scenarios"), "protocol.num_scenarios")
    _positive_integer(protocol.get("scenario_length_steps"), "protocol.scenario_length_steps")
    _positive_integer(protocol.get("num_agents"), "protocol.num_agents")
    _positive_integer(protocol.get("num_workers"), "protocol.num_workers")
    checkpoint_value = protocol.get("checkpoint")
    if not isinstance(checkpoint_value, str) or not checkpoint_value.strip():
        raise ValueError("protocol.checkpoint must contain a checkpoint path")


def _candidate_run_dirs(eval_root: Path, execution_k: int) -> list[Path]:
    candidates = (
        eval_root / f"k{execution_k}",
        eval_root / f"run_k{execution_k}" / f"k{execution_k}",
    )
    return [candidate for candidate in candidates if (candidate / "protocol.json").is_file()]


def _read_metrics_csv(metrics_path: Path) -> tuple[list[str], list[dict[str, str]], int]:
    try:
        with metrics_path.open("r", newline="", encoding="utf-8") as metrics_file:
            reader = csv.DictReader(metrics_file)
            fieldnames = reader.fieldnames
            if not fieldnames:
                raise ValueError(f"Metrics CSV has no header: {metrics_path}")
            required_columns = ("map_name", "seed", "agents_per_batch")
            missing_columns = [column for column in required_columns if column not in fieldnames]
            if missing_columns:
                raise ValueError(
                    f"Metrics CSV is missing required columns: {', '.join(missing_columns)}"
                )
            rows = list(reader)
    except OSError as exc:
        raise ValueError(f"Could not read metrics CSV: {metrics_path}: {exc}") from exc
    if not rows:
        raise ValueError(f"Metrics CSV has no episode rows: {metrics_path}")

    batch_sizes: set[int] = set()
    for row_number, row in enumerate(rows, start=2):
        if not row.get("map_name"):
            raise ValueError(f"Metrics CSV row {row_number} has an empty map_name")
        batch_sizes.add(
            _positive_integer(row.get("agents_per_batch"), f"CSV row {row_number} agents_per_batch")
        )
    if len(batch_sizes) != 1:
        raise ValueError("Metrics CSV must contain one recorded agents_per_batch value")
    recorded_batch_size = next(iter(batch_sizes))
    return list(fieldnames), rows, recorded_batch_size


def _discover_source(eval_root: Path, execution_k: int) -> dict[str, Any]:
    candidates = _candidate_run_dirs(eval_root, execution_k)
    if not candidates:
        raise FileNotFoundError(
            f"No completed k={execution_k} evaluation found under {eval_root}; "
            f"expected {eval_root}/k{execution_k} or "
            f"{eval_root}/run_k{execution_k}/k{execution_k}"
        )
    if len(candidates) != 1:
        raise ValueError(
            f"Ambiguous k={execution_k} evaluation input under {eval_root}: "
            + ", ".join(str(candidate) for candidate in candidates)
        )
    source_run_dir = candidates[0].resolve()
    protocol_path = source_run_dir / "protocol.json"
    protocol = _read_json_mapping(protocol_path, "per-k protocol")
    _validate_protocol(protocol, execution_k)

    metrics_files = sorted(source_run_dir.rglob("episode_metrics.csv"))
    if len(metrics_files) != 1:
        raise ValueError(
            f"Expected exactly one episode_metrics.csv for k={execution_k} under "
            f"{source_run_dir}; found {len(metrics_files)}"
        )
    resolved_files = sorted(source_run_dir.rglob("resolved_benchmark.yaml"))
    if len(resolved_files) != 1:
        raise ValueError(
            f"Expected exactly one resolved_benchmark.yaml for k={execution_k} under "
            f"{source_run_dir}; found {len(resolved_files)}"
        )

    metrics_path = metrics_files[0].resolve()
    resolved_path = resolved_files[0].resolve()
    resolved = _read_yaml_mapping(resolved_path, "resolved benchmark")
    benchmark = resolved.get("benchmark")
    runtime_args = resolved.get("args")
    if not isinstance(benchmark, Mapping) or not isinstance(runtime_args, Mapping):
        raise ValueError(
            "resolved_benchmark.yaml must contain saved `benchmark` and `args` mappings"
        )
    _validate_resolved_config(protocol, benchmark, runtime_args)

    checkpoint_path = Path(str(protocol["checkpoint"])).expanduser()
    if not checkpoint_path.is_absolute():
        checkpoint_path = checkpoint_path.resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Student checkpoint from protocol does not exist: {checkpoint_path}")
    csv_fieldnames, source_rows, recorded_batch_size = _read_metrics_csv(metrics_path)
    if len(source_rows) != protocol["num_scenarios"]:
        raise ValueError(
            f"k={execution_k} protocol records {protocol['num_scenarios']} scenarios, "
            f"but its metrics CSV contains {len(source_rows)} rows"
        )
    _validate_scenario_rows(source_rows, runtime_args["env"], execution_k)

    return {
        "execution_k": execution_k,
        "source_run_dir": source_run_dir,
        "protocol_path": protocol_path.resolve(),
        "metrics_csv": metrics_path,
        "resolved_benchmark_path": resolved_path,
        "checkpoint_path": checkpoint_path,
        "protocol": protocol,
        "resolved_benchmark": resolved,
        "benchmark": dict(benchmark),
        "runtime_args": copy.deepcopy(dict(runtime_args)),
        "source_rows": len(source_rows),
        "csv_fieldnames": csv_fieldnames,
        "_source_rows": source_rows,
        "recorded_agents_per_batch": recorded_batch_size,
    }


def _validate_resolved_config(
    protocol: Mapping[str, Any],
    benchmark: Mapping[str, Any],
    runtime_args: Mapping[str, Any],
) -> None:
    if benchmark.get("name") != "carla":
        raise ValueError("resolved benchmark must be named carla")
    if benchmark.get("seed") != protocol.get("seed"):
        raise ValueError("protocol and resolved benchmark seeds do not match")
    if benchmark.get("num_scenarios") != protocol.get("num_scenarios"):
        raise ValueError("protocol and resolved benchmark scenario counts do not match")
    benchmark_env = benchmark.get("env")
    if not isinstance(benchmark_env, Mapping):
        raise ValueError("resolved benchmark env must be a mapping")
    if not math.isclose(
        _number(benchmark_env.get("dt"), "resolved benchmark env.dt"),
        EXPECTED_EVALUATION_DT,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("resolved benchmark env.dt must be 0.1")
    if benchmark_env.get("scenario_length") != protocol.get("scenario_length_steps"):
        raise ValueError("protocol and resolved benchmark scenario lengths do not match")
    if benchmark_env.get("action_type") != "continuous":
        raise ValueError("resolved benchmark env.action_type must be continuous")

    runtime_env = runtime_args.get("env")
    runtime_eval = runtime_args.get("eval")
    runtime_train = runtime_args.get("train")
    runtime_vec = runtime_args.get("vec")
    if not all(isinstance(section, Mapping) for section in (runtime_env, runtime_eval, runtime_train, runtime_vec)):
        raise ValueError("resolved args must contain env, eval, train, and vec mappings")
    if not math.isclose(
        _number(runtime_env.get("dt"), "resolved args env.dt"),
        EXPECTED_EVALUATION_DT,
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError("resolved args env.dt must be 0.1")
    if runtime_eval.get("action_selection") != "mean":
        raise ValueError("resolved args eval.action_selection must be mean")
    benchmarks = runtime_eval.get("benchmarks")
    if not isinstance(benchmarks, Sequence) or isinstance(benchmarks, (str, bytes)) or list(benchmarks) != ["carla"]:
        raise ValueError("resolved args eval.benchmarks must select only carla")
    if runtime_eval.get("num_agents") != protocol.get("num_agents"):
        raise ValueError("protocol and resolved args eval.num_agents do not match")
    for key in ("dt", "scenario_length", "action_type", "map_dir", "num_maps"):
        if key in benchmark_env and key in runtime_env and benchmark_env[key] != runtime_env[key]:
            raise ValueError(f"resolved benchmark and args env.{key} do not match")
    if runtime_env.get("scenario_length") != protocol.get("scenario_length_steps"):
        raise ValueError("protocol and resolved args env.scenario_length do not match")
    if runtime_env.get("action_type") != "continuous":
        raise ValueError("resolved args env.action_type must be continuous")
    if runtime_train.get("amp") is not True or runtime_train.get("precision") != "bfloat16":
        raise ValueError("resolved args must preserve BF16 inference (amp=true, precision=bfloat16)")
    if runtime_train.get("compile") is not False:
        raise ValueError("resolved args train.compile must be false for this evaluator")
    if not isinstance(runtime_args.get("env_name"), str) or not runtime_args.get("env_name"):
        raise ValueError("resolved args env_name is missing")
    _positive_integer(runtime_vec.get("num_envs"), "resolved args vec.num_envs")


def preflight_render(
    eval_root: str | Path,
    output_dir: str | Path,
    execution_ks: Sequence[int],
    failure_mode: str,
    max_failures: Optional[int],
    *,
    device: str = "cuda:0",
    workers: int = 2,
) -> list[dict[str, Any]]:
    """Validate all source inputs and compute selected rows without writing files."""

    source_root = Path(eval_root).expanduser().resolve()
    output_root = Path(output_dir).expanduser().resolve()
    if not source_root.is_dir():
        raise FileNotFoundError(f"Evaluation root does not exist or is not a directory: {source_root}")
    if not isinstance(device, str) or not device.strip():
        raise ValueError("device must be a non-empty device string")
    worker_count = parse_workers(workers)
    if failure_mode not in EXPECTED_FAILURE_COLUMNS:
        raise ValueError("failure_mode must be all_infractions, collision, at_fault_collision, offroad, or red_light")
    if max_failures is not None:
        max_failures = parse_max_failures(max_failures)
    if (
        not execution_ks
        or len(set(execution_ks)) != len(execution_ks)
        or any(isinstance(k, bool) or not isinstance(k, int) or k not in (1, 2, 3, 4) for k in execution_ks)
    ):
        raise ValueError("execution_ks must contain unique integers from 1 through 4")
    if source_root == output_root or source_root in output_root.parents or output_root in source_root.parents:
        raise ValueError("Evaluation input and render output directories must not overlap")
    if output_root.exists():
        raise FileExistsError(f"Render output root already exists: {output_root}")

    plans = []
    output_paths = set()
    for execution_k in execution_ks:
        source = _discover_source(source_root, execution_k)
        output_path = output_root / f"k{execution_k}"
        if output_path in output_paths:
            raise ValueError(f"Duplicate k={execution_k} output path")
        output_paths.add(output_path)
        if output_path.exists():
            raise FileExistsError(f"Render output already exists: {output_path}")
        all_source_rows = source.pop("_source_rows")
        fieldnames = source["csv_fieldnames"]
        missing_columns = [
            column for column in EXPECTED_FAILURE_COLUMNS[failure_mode] if column not in fieldnames
        ]
        if missing_columns:
            raise ValueError(
                f"Metrics CSV is missing required failure columns: {', '.join(missing_columns)}"
            )
        filtered_rows = []
        for row_number, row in enumerate(all_source_rows, start=2):
            selected = False
            for column in EXPECTED_FAILURE_COLUMNS[failure_mode]:
                metric_value = _number(row.get(column), f"CSV row {row_number} {column}")
                selected = selected or metric_value > 0
            if selected:
                filtered_rows.append(row)
        if max_failures is not None:
            filtered_rows = filtered_rows[:max_failures]
        source.update(
            {
                "output_dir": output_path,
                "selected_rows": filtered_rows,
                "match_count": len(filtered_rows),
                "failure_mode": failure_mode,
                "max_failures": max_failures,
                "device": device,
                "workers": worker_count,
                "recorded_agents_per_batch": source["recorded_agents_per_batch"],
                "csv_fieldnames": fieldnames,
            }
        )
        plans.append(source)
    return plans


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as output_file:
        json.dump(payload, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def _write_selected_csv(path: Path, fieldnames: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("x", newline="", encoding="utf-8") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _write_local_catalog(
    path: Path,
    benchmark: Mapping[str, Any],
    runtime_environment: Mapping[str, Any],
) -> None:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - project dependency
        raise RuntimeError("PyYAML is required to write the local CARLA catalog") from exc
    saved_benchmark = copy.deepcopy(dict(benchmark))
    saved_benchmark["env"] = copy.deepcopy(dict(runtime_environment))
    local_catalog = {"env": {}, "benchmarks": [saved_benchmark]}
    with path.open("x", encoding="utf-8") as catalog_file:
        yaml.safe_dump(local_catalog, catalog_file, sort_keys=False)


def _write_output_provenance(plan: Mapping[str, Any]) -> None:
    output_dir = Path(plan["output_dir"])
    output_dir.mkdir(parents=True)
    shutil.copy2(plan["protocol_path"], output_dir / "source_protocol.json")
    shutil.copy2(plan["metrics_csv"], output_dir / "source_episode_metrics.csv")
    shutil.copy2(plan["resolved_benchmark_path"], output_dir / "source_resolved_benchmark.yaml")
    _write_local_catalog(
        output_dir / "benchmark_catalog.yaml",
        plan["benchmark"],
        plan["runtime_args"]["env"],
    )
    _write_selected_csv(
        output_dir / "selected_failures.csv",
        plan["csv_fieldnames"],
        plan["selected_rows"],
    )


def _render_one(plan: Mapping[str, Any], student: Any) -> dict[str, Any]:
    execution_k = int(plan["execution_k"])
    output_dir = Path(plan["output_dir"])
    selected_csv = output_dir / "selected_failures.csv"
    report: dict[str, Any] = {
        "execution_k": execution_k,
        "source_protocol": str(plan["protocol_path"]),
        "source_episode_metrics": str(plan["metrics_csv"]),
        "checkpoint": str(plan["checkpoint_path"]),
        "failure_mode": plan["failure_mode"],
        "max_failures": plan["max_failures"],
        "selected_scenarios": plan["match_count"],
        "recorded_agents_per_batch": plan["recorded_agents_per_batch"],
        "replay_workers": plan["workers"],
        "device": plan["device"],
        "html_index": None,
        "status": "no_matching_failures" if not plan["selected_rows"] else "rendering",
    }
    if not plan["selected_rows"]:
        report["message"] = "No scenarios matched this failure mode; no replay HTML index was generated."
        _write_json(output_dir / "render_report.json", report)
        print(f"k={execution_k}: 0 scenarios matched; wrote {output_dir / 'render_report.json'}")
        return report
    if student is None:
        raise ValueError(f"k={execution_k} has selected failures but no loaded student model")

    runtime_args = copy.deepcopy(plan["runtime_args"])
    runtime_args["wandb"] = False
    runtime_args["neptune"] = False
    runtime_args["tb"] = False
    runtime_args["load_id"] = None
    runtime_args["load_model_path"] = None
    runtime_args["train"]["device"] = plan["device"]
    runtime_args["eval"].update(
        {
            "benchmark_config": str(output_dir / "benchmark_catalog.yaml"),
            "benchmarks": ["carla"],
            "action_selection": "mean",
            "capture_observations": True,
            "failure_replay_csv": str(selected_csv),
            "render_filter": FAILURE_MODE_COLUMNS[plan["failure_mode"]],
            "max_rendered_failures": None,
        }
    )
    runtime_args["vec"]["num_envs"] = plan["workers"]
    adapter = ChunkPolicyAdapter(student, execution_k=execution_k)
    adapter.eval()

    from pufferlib import pufferl

    benchmark_results = pufferl.eval(
        env_name=runtime_args["env_name"],
        args=runtime_args,
        policy=adapter,
        eval_output_dir=str(output_dir),
        eval_output_subdir="failure_replay",
        use_training_config=True,
        benchmark_names=["carla"],
    )
    benchmark_result = benchmark_results.get("carla") if isinstance(benchmark_results, Mapping) else None
    replay_summaries = benchmark_result.get("episodes") if isinstance(benchmark_result, Mapping) else None
    if not isinstance(replay_summaries, Sequence) or isinstance(replay_summaries, (str, bytes)):
        replay_summaries = []
    if len(replay_summaries) != plan["match_count"]:
        raise RuntimeError(
            f"k={execution_k} replayed {len(replay_summaries)} scenarios; "
            f"expected {plan['match_count']}"
        )
    replay_paths = [
        summary.get("replay_path")
        for summary in replay_summaries
        if isinstance(summary, Mapping)
    ]
    if len(replay_paths) != plan["match_count"] or any(
        not isinstance(replay_path, str)
        or not Path(replay_path).is_file()
        or Path(replay_path).stat().st_size <= 0
        for replay_path in replay_paths
    ):
        raise RuntimeError(f"k={execution_k} did not produce every expected replay file")
    rendered_dir = output_dir / "failures" / "rendered_replays"
    html_index_path = rendered_dir / "index.html"
    html_pages = (
        [page for page in rendered_dir.glob("*.html") if page.name != "index.html"]
        if rendered_dir.is_dir()
        else []
    )
    if not html_index_path.is_file() or html_index_path.stat().st_size <= 0:
        raise RuntimeError(f"k={execution_k} replay HTML index was not produced: {html_index_path}")
    if len(html_pages) != plan["match_count"]:
        raise RuntimeError(
            f"k={execution_k} produced {len(html_pages)} HTML pages; "
            f"expected {plan['match_count']}"
        )
    if any(page.stat().st_size <= 0 for page in html_pages):
        raise RuntimeError(f"k={execution_k} produced an empty replay HTML page")
    report["html_index"] = str(html_index_path)
    report["status"] = "completed"
    _write_json(output_dir / "render_report.json", report)
    print(f"k={execution_k}: rendered {plan['match_count']} selected scenarios")
    return report


def run_render(args: argparse.Namespace) -> list[Mapping[str, Any]]:
    plans = preflight_render(
        args.eval_root,
        args.output_dir,
        args.k,
        args.failure_mode,
        args.max_failures,
        device=args.device,
        workers=args.workers,
    )
    if args.dry_run:
        print(
            json.dumps(
                {
                    "eval_root": str(Path(args.eval_root).expanduser().resolve()),
                    "output_dir": str(Path(args.output_dir).expanduser().resolve()),
                    "failure_mode": args.failure_mode,
                    "max_failures": args.max_failures,
                    "dry_run": True,
                    "runs": [
                        {
                            "execution_k": plan["execution_k"],
                            "source_run_dir": str(plan["source_run_dir"]),
                            "checkpoint": str(plan["checkpoint_path"]),
                            "source_csv": str(plan["metrics_csv"]),
                            "selected_scenarios": plan["match_count"],
                            "output_dir": str(plan["output_dir"]),
                        }
                        for plan in plans
                    ],
                },
                indent=2,
            )
        )
        return []

    reports = []
    for plan in plans:
        student = None
        if plan["match_count"] > 0:
            student, checkpoint = _load_student_checkpoint(
                plan["checkpoint_path"],
                model_config=None,
                device=args.device,
            )
            validate_checkpoint_metadata(checkpoint)
        _write_output_provenance(plan)
        reports.append(_render_one(plan, student))
    return reports


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Render selected CARLA failure replays from completed Condition B evaluations."
    )
    parser.add_argument("--eval-root", required=True, help="Completed evaluation root with k1-k4 results")
    parser.add_argument("--output-dir", required=True, help="Fresh directory for copied inputs and rendered output")
    parser.add_argument(
        "--k",
        type=parse_execution_ks,
        default=(1, 2, 3, 4),
        help="Render all k values, or one of 1, 2, 3, or 4 (default: all)",
    )
    parser.add_argument(
        "--failure-mode",
        choices=tuple(EXPECTED_FAILURE_COLUMNS),
        default="all_infractions",
        help="Failure metric filter (default: all_infractions)",
    )
    parser.add_argument(
        "--max-failures",
        type=parse_max_failures,
        default=10,
        help="Maximum selected scenarios per k; use all or null for no limit (default: 10)",
    )
    parser.add_argument("--device", default="cuda:0", help="Student inference device (default: cuda:0)")
    parser.add_argument("--workers", type=parse_workers, default=2, help="Replay workers per k (default: 2)")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs and print selections without loading the checkpoint or writing files",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    try:
        run_render(args)
    except (FileExistsError, FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
