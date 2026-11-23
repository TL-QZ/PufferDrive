"""Launch the authorized two-model experiment in durable, isolated tmux sessions."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys

from .run_layout import initialize_model_directory, model_directory, records_directory


PROJECT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT.parents[1]
CONFIG_PATH = Path(__file__).resolve().parent / "configs" / "first_experiment.json"


def _write_run_readme(run_directory, model_names):
    readme = run_directory / "README.md"
    if readme.exists():
        return
    model_links = "\n".join(f"- [{name}]({name}/README.md)" for name in model_names)
    readme.write_text(
        f"# {run_directory.name}\n\n"
        "Each model directory contains its collection scripts, records, plots, and renders.\n\n"
        "## Models\n\n"
        f"{model_links}\n"
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True,
                        help="run directory under runs/<run>; per-model folders are initialized here")
    parser.add_argument("--session-prefix", default="q_probe_first")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    output = args.output.expanduser().resolve()
    run_items = []
    for name, model in plan["models"].items():
        model_output = model_directory(output / name)
        records_output = records_directory(model_output)
        scripts_output = model_output / "scripts"
        session = f"{args.session_prefix}_{name}"
        command = [
            sys.executable, "-u", "-m", "project.soft_actor_critic_q_calibration.development.run_probes",
            "--checkpoint", str(REPOSITORY_ROOT / model["checkpoint"]),
            "--output", str(model_output), "--device", "cuda:0",
            "--episodes", str(plan["episodes"]), "--repeats", str(plan["repeats"]),
            "--branch-batch-size", str(plan["branch_batch_size"]),
            "--seed", str(plan["seed"]), "--cpu-threads", "2",
        ]
        run_items.append({
            "model": name,
            "session": session,
            "gpu": model["gpu"],
            "output": str(model_output),
            "records": str(records_output),
            "scripts": str(scripts_output),
            "command": command,
        })

    print(json.dumps(run_items, indent=2), flush=True)
    if args.dry_run:
        return

    for item in run_items:
        check = subprocess.run(["tmux", "has-session", "-t", item["session"]], capture_output=True)
        if check.returncode == 0:
            raise RuntimeError(f"Refusing to replace existing session {item['session']}")
        model_output = Path(item["output"])
        if model_output.exists():
            raise FileExistsError(f"Use a new experiment output directory: {model_output}")

    for item in run_items:
        initialize_model_directory(item["output"])
    _write_run_readme(output, [item["model"] for item in run_items])

    for item in run_items:
        records_output = Path(item["records"])
        scripts_output = Path(item["scripts"])
        collect_script = scripts_output / "collect.sh"
        environment = {
            "CUDA_VISIBLE_DEVICES": str(item["gpu"]), "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1", "NUMPY_MADVISE_HUGEPAGE": "0",
        }
        prefix = "env " + " ".join(shlex.quote(f"{key}={value}") for key, value in environment.items())
        command = prefix + " " + shlex.join(item["command"])
        summary = shlex.join([
            sys.executable, "-m", "project.soft_actor_critic_q_calibration.development.summarize_probes",
            item["output"],
        ])
        collect_script.write_text(
            "#!/usr/bin/env bash\nset -uo pipefail\n"
            + f"cd {shlex.quote(str(REPOSITORY_ROOT))}\n"
            + f"{command} > {shlex.quote(str(records_output / 'console.log'))} 2>&1\n"
            + "probe_exit=$?\n"
            + f"printf '%s\\n' \"$probe_exit\" > {shlex.quote(str(records_output / 'exit_code'))}\n"
            + f"if [ \"$probe_exit\" -eq 0 ]; then {summary} >> {shlex.quote(str(records_output / 'console.log'))} 2>&1; fi\n"
            + "exit \"$probe_exit\"\n"
        )
        os.chmod(collect_script, 0o755)
        subprocess.run(
            ["tmux", "new-session", "-d", "-s", item["session"], shlex.join(["bash", str(collect_script)])],
            check=True, cwd=REPOSITORY_ROOT,
        )
        item["started_at"] = datetime.now(timezone.utc).isoformat()

    (output / "launch.json").write_text(json.dumps(run_items, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
