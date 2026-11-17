"""Wait for the four added evaluations, then verify and rebuild their reports."""

from datetime import datetime, timezone
import json
import subprocess
import sys
import time

from prepare_additional_evaluations import OUTPUT, ROOT, RUNS

POLL_SECONDS = 30
MAX_POLLS = 7 * 24 * 60 * 2


def main():
    completion_path = OUTPUT / "additional_4runs_completion.status.json"
    try:
        for poll_idx in range(MAX_POLLS):
            statuses = {case: json.loads((OUTPUT / f"{case}.status.json").read_text())
                        for case, _, _ in RUNS}
            failed = {case: status["state"] for case, status in statuses.items()
                      if status["state"] not in {"pending", "running", "finished"}}
            if failed:
                raise RuntimeError(f"Evaluation failed: {failed}")
            if all(status["state"] == "finished" and status.get("exitcode") == 0
                   for status in statuses.values()):
                break
            time.sleep(POLL_SECONDS)
        else:
            raise TimeoutError("Evaluation completion wait exceeded seven days")
        for script in ("summarize_full_evaluations.py", "verify_full_metrics.py", "plot_full_distances.py"):
            subprocess.run([sys.executable, "-B", str(OUTPUT / script)], cwd=ROOT, check=True)
            if script == "summarize_full_evaluations.py":
                verification = json.loads((OUTPUT / "verification.json").read_text())
                if not verification["complete"]:
                    raise RuntimeError("Full comparison verification failed")
        status = {"state": "finished", "verified_jobs": verification["expected_jobs"]}
    except Exception as error:
        status = {"state": "failed", "error": str(error)}
        raise
    finally:
        status["updated_at"] = datetime.now(timezone.utc).isoformat()
        completion_path.write_text(json.dumps(status, indent=2) + "\n")


if __name__ == "__main__":
    main()
