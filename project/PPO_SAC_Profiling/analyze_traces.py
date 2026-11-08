"""Summarize raw CUDA activities without double-counting profiler annotations."""
import argparse
from collections import defaultdict
import hashlib
import json
from pathlib import Path


def summarize_trace(path):
    raw = path.read_bytes()
    events = json.loads(raw)["traceEvents"]
    scopes = {event["name"]: (event["ts"], event["ts"] + event["dur"])
              for event in events if event.get("ph") == "X"
              and event.get("cat") == "user_annotation"
              and event.get("name") in ("profile_collection", "profile_training")}
    activities = defaultdict(lambda: {"count": 0, "duration_us": 0.0, "bytes": 0})
    kernels = defaultdict(float)
    for event in events:
        category = event.get("cat")
        if event.get("ph") != "X" or category not in ("kernel", "gpu_memcpy", "gpu_memset"):
            continue
        phase = next((name for name, (start, end) in scopes.items()
                      if start <= event["ts"] < end), "outside_cpu_scopes")
        kind = event["name"] if category == "gpu_memcpy" else category
        row = activities[(phase, kind)]
        row["count"] += 1
        row["duration_us"] += event["dur"]
        row["bytes"] += event.get("args", {}).get("bytes", 0)
        if category == "kernel" and phase == "profile_training":
            kernels[event["name"]] += event["dur"]
    return {"trace": str(path), "sha256": hashlib.sha256(raw).hexdigest(),
            "scope_assignment": "GPU event start falls within the CPU annotation; late launches remain outside",
            "duration_semantics": "Sum of activity durations, not wall time; streams may overlap; annotations excluded",
            "activities": [{"phase": phase, "kind": kind, **row}
                           for (phase, kind), row in sorted(activities.items())],
            "top_training_kernels_us": sorted(kernels.items(), key=lambda item: item[1], reverse=True)[:15]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    paths = sorted({path for root in args.inputs for path in root.rglob("trace_cycle_*_rank[0-9][0-9].json")})
    rows = [summarize_trace(path) for path in paths]
    args.output.with_suffix(".json").write_text(json.dumps(rows, indent=2) + "\n")
    lines = ["# Raw CUDA activity evidence", "",
             "Activity durations are summed, not wall time; streams may overlap. CPU-scope assignment uses event start. "
             "User annotations and host/device operator duplicates are excluded. Trace cycles are excluded from throughput.", ""]
    for row in rows:
        lines += [f"## {row['trace']}", "", "| Phase | Activity | Count | Activity seconds | GiB copied |",
                  "|---|---|---:|---:|---:|"]
        for event in row["activities"]:
            lines.append(f"| {event['phase']} | {event['kind']} | {event['count']} | "
                         f"{event['duration_us']/1e6:.4f} | {event['bytes']/2**30:.3f} |")
        lines.append("")
    args.output.with_suffix(".md").write_text("\n".join(lines) + "\n")


if __name__ == "__main__":
    main()
