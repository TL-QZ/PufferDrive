"""Standalone CARLA evaluation of Condition B action chunks."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

from .evaluate import (
    StudentPolicyAdapter,
    _load_student_checkpoint,
    _runtime_args,
    evaluate_student,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCHMARK_CONFIG = (
    REPO_ROOT
    / "project/baseline_run_sync_2026-08-24/override_config/evaluation_benchmarks.yaml"
)
PRODUCTION_EVAL_DT = 0.1
TRAINING_DT = 0.3
CHUNK_LENGTH = 4
HOLD_STEPS = 3
PRODUCTION_NUM_AGENTS = 300
PRODUCTION_NUM_WORKERS = 20
SMOKE_SCENARIOS = 2
SMOKE_EPISODE_TIMESTEPS = 13
SMOKE_NUM_AGENTS = 50
BENCHMARK_NAMES = ("carla", "nuplan_single", "nuplan_multi")


class ChunkPolicyAdapter(StudentPolicyAdapter):
    """Select and execute one action slot from each four-step model chunk."""

    def __init__(
        self,
        student: torch.nn.Module,
        *,
        execution_k: int,
        action_table: Optional[torch.Tensor] = None,
        action_table_physical: Optional[torch.Tensor] = None,
    ) -> None:
        if (
            isinstance(execution_k, bool)
            or not isinstance(execution_k, int)
            or execution_k not in (1, 2, 3, 4)
        ):
            raise ValueError("execution_k must be one of 1, 2, 3, or 4")
        super().__init__(
            student,
            action_table=action_table,
            action_table_physical=action_table_physical,
        )
        self.execution_k = int(execution_k)
        self._chunk_cache: Optional[torch.Tensor] = None
        self._cache_valid: Optional[torch.Tensor] = None
        self._steps_since_plan: Optional[torch.Tensor] = None

    def reset_eval_state(self, mask: Any = None) -> None:
        """Clear all cached plans, or only rows reported done by the vector env."""

        if self._cache_valid is None or self._steps_since_plan is None:
            return
        if mask is None:
            self._cache_valid.zero_()
            self._steps_since_plan.zero_()
            return

        reset_mask = torch.as_tensor(mask, dtype=torch.bool, device=self._cache_valid.device).reshape(-1)
        if reset_mask.numel() > self._cache_valid.numel():
            raise ValueError("eval reset mask is longer than the policy batch")
        if not reset_mask.any():
            return
        valid_rows = self._cache_valid[: reset_mask.numel()]
        age_rows = self._steps_since_plan[: reset_mask.numel()]
        valid_rows[reset_mask] = False
        age_rows[reset_mask] = 0

    def forward_eval(
        self,
        observations: torch.Tensor,
        state: Optional[Mapping[str, torch.Tensor]] = None,
    ) -> tuple[tuple[torch.Tensor], torch.Tensor]:
        if state is not None:
            raise ValueError("ChunkPolicyAdapter is feed-forward and does not accept recurrent state")
        if not isinstance(observations, torch.Tensor) or observations.ndim < 2:
            raise ValueError("observations must be a batched torch tensor")
        batch_count = observations.shape[0]
        if batch_count <= 0:
            raise ValueError("observations must contain at least one policy row")
        if (
            self._cache_valid is None
            or self._cache_valid.numel() != batch_count
            or self._cache_valid.device != observations.device
        ):
            self._chunk_cache = None
            self._cache_valid = torch.zeros(batch_count, dtype=torch.bool, device=observations.device)
            self._steps_since_plan = torch.zeros(batch_count, dtype=torch.long, device=observations.device)

        assert self._cache_valid is not None
        assert self._steps_since_plan is not None
        plan_interval = 1 if self.execution_k == 1 else HOLD_STEPS * self.execution_k
        replan_rows = torch.nonzero(
            ~self._cache_valid | (self._steps_since_plan >= plan_interval), as_tuple=False
        ).flatten()
        if replan_rows.numel():
            chunk_logits = self._student_chunk_logits(observations.index_select(0, replan_rows))
            if chunk_logits.ndim != 3 or chunk_logits.shape[1] != CHUNK_LENGTH:
                raise ValueError(
                    "chunk evaluation requires student logits with shape [B, 4, classes], "
                    f"got {tuple(chunk_logits.shape)}"
                )
            if chunk_logits.shape[-1] != self.num_action_classes:
                raise ValueError(
                    "student chunk class count does not match the action table: "
                    f"{chunk_logits.shape[-1]} != {self.num_action_classes}"
                )
            if self._chunk_cache is None:
                self._chunk_cache = torch.empty(
                    (batch_count, CHUNK_LENGTH, self.num_action_classes),
                    dtype=chunk_logits.dtype,
                    device=chunk_logits.device,
                )
            elif (
                self._chunk_cache.dtype != chunk_logits.dtype
                or self._chunk_cache.device != chunk_logits.device
            ):
                raise ValueError("student chunk logits changed dtype or device during evaluation")
            self._chunk_cache.index_copy_(0, replan_rows, chunk_logits)
            self._cache_valid[replan_rows] = True
            self._steps_since_plan[replan_rows] = 0

        assert self._chunk_cache is not None
        row_indices = torch.arange(batch_count, device=observations.device)
        if self.execution_k == 1:
            slot_indices = torch.zeros_like(row_indices)
        else:
            slot_indices = torch.div(self._steps_since_plan, HOLD_STEPS, rounding_mode="floor")
        selected_logits = self._chunk_cache[row_indices, slot_indices]
        self._steps_since_plan.add_(1)
        unused_value = torch.zeros(
            (batch_count, 1), dtype=selected_logits.dtype, device=selected_logits.device
        )
        return (selected_logits,), unused_value


def validate_checkpoint_metadata(checkpoint: Mapping[str, Any]) -> dict[str, float | int]:
    """Require the expected four-slot student trained with dt=0.3."""

    model_config = checkpoint.get("model_config")
    if not isinstance(model_config, Mapping):
        raise ValueError("checkpoint must include model_config metadata")
    model_section = model_config.get("model", model_config)
    if not isinstance(model_section, Mapping):
        raise ValueError("checkpoint model_config.model must be a mapping")
    chunk_length = model_section.get("chunk_length")
    if (
        isinstance(chunk_length, bool)
        or not isinstance(chunk_length, int)
        or chunk_length != CHUNK_LENGTH
    ):
        raise ValueError(f"checkpoint chunk_length must be {CHUNK_LENGTH}, got {chunk_length!r}")

    checkpoint_config = checkpoint.get("config")
    if not isinstance(checkpoint_config, Mapping):
        raise ValueError("checkpoint must include saved training config")
    overrides = checkpoint_config.get("env_overrides")
    if not isinstance(overrides, Mapping) or "dt" not in overrides:
        teacher_config = checkpoint_config.get("teacher_config")
        teacher_env = teacher_config.get("env") if isinstance(teacher_config, Mapping) else None
        training_dt = teacher_env.get("dt") if isinstance(teacher_env, Mapping) else None
    else:
        training_dt = overrides.get("dt")
    try:
        numeric_dt = float(training_dt)
    except (TypeError, ValueError) as exc:
        raise ValueError("checkpoint must record training dt=0.3") from exc
    if not math.isfinite(numeric_dt) or not math.isclose(numeric_dt, TRAINING_DT, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError(f"checkpoint training dt must be {TRAINING_DT}, got {training_dt!r}")
    return {"chunk_length": CHUNK_LENGTH, "training_dt": numeric_dt}


def parse_execution_ks(value: str) -> tuple[int, ...]:
    if value == "all":
        return (1, 2, 3, 4)
    try:
        execution_k = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("--k must be all, 1, 2, 3, or 4") from exc
    if execution_k not in (1, 2, 3, 4):
        raise argparse.ArgumentTypeError("--k must be all, 1, 2, 3, or 4")
    return (execution_k,)


def _load_benchmark_entry(benchmark_name: str) -> Mapping[str, Any]:
    if benchmark_name not in BENCHMARK_NAMES:
        raise ValueError(f"benchmark must be one of {', '.join(BENCHMARK_NAMES)}")
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover - project dependency
        raise RuntimeError("PyYAML is required to resolve the baseline benchmark catalog") from exc
    if not BENCHMARK_CONFIG.is_file():
        raise FileNotFoundError(f"baseline benchmark catalog not found: {BENCHMARK_CONFIG}")
    with BENCHMARK_CONFIG.open("r", encoding="utf-8") as config_file:
        catalog = yaml.safe_load(config_file)
    benchmarks = catalog.get("benchmarks") if isinstance(catalog, Mapping) else None
    if not isinstance(benchmarks, list):
        raise ValueError("baseline benchmark catalog must contain a benchmarks list")
    for entry in benchmarks:
        if isinstance(entry, Mapping) and entry.get("name") == benchmark_name:
            environment = entry.get("env")
            if not isinstance(environment, Mapping):
                raise ValueError(f"baseline benchmark {benchmark_name} must contain an env mapping")
            return entry
    raise ValueError(f"baseline benchmark catalog does not contain {benchmark_name!r}")


def build_protocol_config(
    checkpoint: Mapping[str, Any],
    execution_k: int,
    *,
    checkpoint_path: str | Path,
    output_dir: str | Path,
    device: str,
    smoke: bool = False,
    benchmark: str = "carla",
) -> dict[str, Any]:
    """Build the single-k configuration from saved teacher args and the baseline catalog."""

    if (
        isinstance(execution_k, bool)
        or not isinstance(execution_k, int)
        or execution_k not in (1, 2, 3, 4)
    ):
        raise ValueError("execution_k must be one of 1, 2, 3, or 4")
    validate_checkpoint_metadata(checkpoint)
    benchmark_entry = _load_benchmark_entry(benchmark)
    benchmark_environment = benchmark_entry["env"]
    benchmark_scenarios = benchmark_entry.get("num_scenarios")
    benchmark_episode_timesteps = benchmark_environment.get("scenario_length")
    if (
        isinstance(benchmark_scenarios, bool)
        or not isinstance(benchmark_scenarios, int)
        or benchmark_scenarios <= 0
    ):
        raise ValueError(f"baseline benchmark {benchmark} must have a positive num_scenarios")
    if (
        isinstance(benchmark_episode_timesteps, bool)
        or not isinstance(benchmark_episode_timesteps, int)
        or benchmark_episode_timesteps <= 0
    ):
        raise ValueError(f"baseline benchmark {benchmark} must have a positive scenario_length")
    runtime_args = _runtime_args({}, {"device": device}, checkpoint)
    if runtime_args is None:
        raise ValueError("checkpoint does not contain saved teacher PufferLib arguments")

    runtime_args["env"] = dict(runtime_args.get("env", {}))
    for name in (
        "obs_dropout_lane",
        "obs_dropout_boundary",
        "partner_blindness_prob",
        "partner_blindness_trigger_prob",
        "phantom_braking_prob",
        "phantom_braking_trigger_prob",
    ):
        runtime_args["env"][name] = 0.0
    runtime_args["eval"] = dict(runtime_args.get("eval", {}))
    runtime_args["eval"].update(
        {
            "benchmark_config": str(BENCHMARK_CONFIG),
            "benchmarks": [benchmark],
            "action_selection": "mean",
            "num_agents": (
                SMOKE_NUM_AGENTS
                if smoke and benchmark == "carla"
                else PRODUCTION_NUM_AGENTS
            ),
            "render_scenarios": False,
            "render_filter": None,
            "failure_replay_csv": None,
            "capture_observations": False,
            "max_rendered_failures": None,
        }
    )
    runtime_args["vec"] = dict(runtime_args.get("vec", {}))
    runtime_args["vec"]["num_envs"] = 1 if smoke else PRODUCTION_NUM_WORKERS

    run_id = f"smoke_k{execution_k}" if smoke else f"k{execution_k}"
    evaluation = {
        "student_checkpoint": str(Path(checkpoint_path).resolve()),
        "benchmark_config": str(BENCHMARK_CONFIG),
        "benchmarks": [benchmark],
        "population": "student_self_play",
        "action_selection": "mean",
        "execution_horizon": 1,
        "execution_k": execution_k,
        "seed": 42,
        "num_scenarios": SMOKE_SCENARIOS if smoke else benchmark_scenarios,
        "episode_timesteps": SMOKE_EPISODE_TIMESTEPS if smoke else benchmark_episode_timesteps,
        "output_root": str(Path(output_dir).resolve()),
        "run_id": run_id,
        "output_name": f"act_k{execution_k}",
        "output_subdir": "standalone",
        "num_agents": (
            SMOKE_NUM_AGENTS
            if smoke and benchmark == "carla"
            else PRODUCTION_NUM_AGENTS
        ),
        "vec_overrides": {"num_envs": 1 if smoke else PRODUCTION_NUM_WORKERS},
        "device": device,
        "amp": True,
        "precision": "bfloat16",
        "compile": False,
        "env_overrides": {
            "action_type": "continuous",
            "dt": PRODUCTION_EVAL_DT,
            "obs_dropout_lane": 0.0,
            "obs_dropout_boundary": 0.0,
            "partner_blindness_prob": 0.0,
            "partner_blindness_trigger_prob": 0.0,
            "phantom_braking_prob": 0.0,
            "phantom_braking_trigger_prob": 0.0,
        },
        "wandb": {"enabled": False, "mode": "disabled"},
    }
    return {"puffer_args": runtime_args, "evaluation": evaluation}


def _protocol_record(
    config: Mapping[str, Any], metadata: Mapping[str, Any], *, smoke: bool
) -> dict[str, Any]:
    evaluation = config["evaluation"]
    execution_k = int(evaluation["execution_k"])
    return {
        "checkpoint": evaluation["student_checkpoint"],
        "training_dt": metadata["training_dt"],
        "evaluation_dt": PRODUCTION_EVAL_DT,
        "model_chunk_length": metadata["chunk_length"],
        "execution_k": execution_k,
        "hold_steps_per_slot": 1 if execution_k == 1 else HOLD_STEPS,
        "replan_interval_steps": 1 if execution_k == 1 else HOLD_STEPS * execution_k,
        "slot_schedule": "slot 0 every evaluation step" if execution_k == 1 else "each slot held for 3 evaluation steps",
        "benchmark_config": str(BENCHMARK_CONFIG),
        "benchmark": evaluation["benchmarks"][0],
        "seed": 42,
        "num_scenarios": evaluation["num_scenarios"],
        "scenario_length_steps": evaluation["episode_timesteps"],
        "num_agents": evaluation["num_agents"],
        "num_workers": evaluation["vec_overrides"]["num_envs"],
        "action_selection": "mean",
        "action_type": "continuous",
        "smoke": smoke,
    }


def _write_json(path: Path, value: Any) -> None:
    with path.open("x", encoding="utf-8") as output_file:
        json.dump(value, output_file, indent=2, sort_keys=True)
        output_file.write("\n")


def run_evaluation(
    checkpoint_path: str | Path,
    *,
    execution_ks: Sequence[int],
    output_dir: str | Path,
    device: str = "cuda:0",
    smoke: bool = False,
    dry_run: bool = False,
    benchmark: str = "carla",
) -> list[Mapping[str, Any]]:
    output_path = Path(output_dir).expanduser().resolve()
    checkpoint_path = Path(checkpoint_path).expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"output directory already exists: {output_path}")
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"student checkpoint not found: {checkpoint_path}")
    if (
        not execution_ks
        or any(isinstance(k, bool) or not isinstance(k, int) or k not in (1, 2, 3, 4) for k in execution_ks)
        or len(set(execution_ks)) != len(execution_ks)
    ):
        raise ValueError("execution_ks must contain unique integer values from 1, 2, 3, and 4")

    student, checkpoint = _load_student_checkpoint(checkpoint_path, model_config=None, device=device)
    metadata = validate_checkpoint_metadata(checkpoint)
    plans = [
        build_protocol_config(
            checkpoint,
            execution_k,
            checkpoint_path=checkpoint_path,
            output_dir=output_path,
            device=device,
            smoke=smoke,
            benchmark=benchmark,
        )
        for execution_k in execution_ks
    ]
    protocol_records = [
        _protocol_record(config, metadata, smoke=smoke) for config in plans
    ]
    if dry_run:
        print(json.dumps({"output_dir": str(output_path), "runs": protocol_records}, indent=2))
        return protocol_records

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.mkdir()
    _write_json(output_path / "protocol.json", {"runs": protocol_records})
    results = []
    for execution_k, config, protocol in zip(execution_ks, plans, protocol_records):
        run_root = output_path / config["evaluation"]["run_id"]
        run_root.mkdir()
        _write_json(run_root / "protocol.json", protocol)
        adapter = ChunkPolicyAdapter(student, execution_k=execution_k)
        adapter.eval()
        results.append(evaluate_student(config, student=adapter))
    return results


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate Condition B action chunks on baseline benchmarks.")
    parser.add_argument("--checkpoint", required=True, help="Condition B student checkpoint path")
    parser.add_argument("--output-dir", required=True, help="New output directory; existing paths are rejected")
    parser.add_argument(
        "--k",
        type=parse_execution_ks,
        default=(1, 2, 3, 4),
        help="Execution horizon: all, 1, 2, 3, or 4 (default: all)",
    )
    parser.add_argument("--benchmark", choices=BENCHMARK_NAMES, default="carla", help="Baseline benchmark (default: carla)")
    parser.add_argument("--device", default="cuda:0", help="Torch device for student inference (default: cuda:0)")
    parser.add_argument("--dry-run", action="store_true", help="Validate and print the protocol without creating output")
    parser.add_argument("--smoke", action="store_true", help="Run 2 scenarios × 13 steps on one worker (50 agents for CARLA, 300 for nuPlan)")
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _argument_parser()
    args = parser.parse_args(argv)
    try:
        run_evaluation(
            args.checkpoint,
            execution_ks=args.k,
            output_dir=args.output_dir,
            device=args.device,
            smoke=args.smoke,
            dry_run=args.dry_run,
            benchmark=args.benchmark,
        )
    except (FileExistsError, FileNotFoundError, ValueError, OSError, RuntimeError) as exc:
        parser.error(str(exc))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
