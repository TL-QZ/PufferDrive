"""Launch the authorized two-model experiment in durable, isolated tmux sessions."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shlex
import subprocess
import sys


PROJECT = Path(__file__).resolve().parent
ROOT = PROJECT.parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--session-prefix", default="q_probe_first")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    plan = json.loads((PROJECT / "first_experiment.json").read_text())
    output = args.output.resolve()
    output.relative_to(PROJECT)
    launches = []
    for name, model in plan["models"].items():
        model_output = output / name
        session = f"{args.session_prefix}_{name}"
        command = [
            sys.executable, "-u", "-m", "project.soft_actor_critic_q_calibration.run_probes",
            "--checkpoint", str(ROOT / model["checkpoint"]),
            "--output", str(model_output), "--device", "cuda:0",
            "--episodes", str(plan["episodes"]), "--repeats", str(plan["repeats"]),
            "--branch-batch-size", str(plan["branch_batch_size"]),
            "--seed", str(plan["seed"]), "--cpu-threads", "2",
        ]
        launches.append({"model": name, "session": session, "gpu": model["gpu"],
                         "output": str(model_output), "command": command})
    print(json.dumps(launches, indent=2), flush=True)
    if args.dry_run:
        return
    for launch in launches:
        check = subprocess.run(["tmux", "has-session", "-t", launch["session"]], capture_output=True)
        if check.returncode == 0:
            raise RuntimeError(f"Refusing to replace existing session {launch['session']}")
        model_output = Path(launch["output"])
        if model_output.exists():
            raise FileExistsError(f"Use a new experiment output directory: {model_output}")
    output.mkdir(parents=True, exist_ok=True)
    for launch in launches:
        model_output = Path(launch["output"])
        model_output.mkdir()
        shell_path = model_output / "run.sh"
        environment = {
            "CUDA_VISIBLE_DEVICES": str(launch["gpu"]), "OMP_NUM_THREADS": "2",
            "MKL_NUM_THREADS": "2", "OPENBLAS_NUM_THREADS": "2", "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1", "NUMPY_MADVISE_HUGEPAGE": "0",
        }
        prefix = "env " + " ".join(shlex.quote(f"{key}={value}") for key, value in environment.items())
        command = prefix + " " + shlex.join(launch["command"])
        summary = shlex.join([sys.executable, "-m", "project.soft_actor_critic_q_calibration.summarize_probes", str(model_output)])
        shell_path.write_text(
            "#!/usr/bin/env bash\nset -uo pipefail\n"
            + f"cd {shlex.quote(str(ROOT))}\n"
            + f"{command} > {shlex.quote(str(model_output / 'console.log'))} 2>&1\n"
            + "probe_exit=$?\n"
            + f"printf '%s\\n' \"$probe_exit\" > {shlex.quote(str(model_output / 'exit_code'))}\n"
            + f"if [ \"$probe_exit\" -eq 0 ]; then {summary} >> {shlex.quote(str(model_output / 'console.log'))} 2>&1; fi\n"
            + "exit \"$probe_exit\"\n"
        )
        subprocess.run(["tmux", "new-session", "-d", "-s", launch["session"],
                        shlex.join(["bash", str(shell_path)])], check=True, cwd=ROOT)
        launch["started_at"] = datetime.now(timezone.utc).isoformat()
    (output / "launch.json").write_text(json.dumps(launches, indent=2) + "\n")


if __name__ == "__main__":
    main()
