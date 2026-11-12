"""Print the latest filtering rates from one or more trial metrics files."""

import argparse
import json
from pathlib import Path


def latest_filter_record(path):
    latest = None
    with path.open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            # A live writer may not yet have completed its final JSONL record.
            if not line.endswith("\n"):
                break
            try:
                record = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"{path}:{line_number}: malformed JSON record") from error
            if "filter/retained_fraction" in record:
                latest = record
    return latest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("metrics", nargs="+", type=Path, help="trial_98/metrics.jsonl paths")
    args = parser.parse_args(argv)
    for path in args.metrics:
        try:
            record = latest_filter_record(path)
        except (OSError, ValueError) as error:
            parser.exit(1, f"{error}\n")
        print(path)
        if record is None:
            print("  Waiting for the first filtering metric record.")
            continue
        retained = record["filter/retained_fraction"]
        filtered = record["filter/filtered_fraction"]
        print(
            f"  epoch={record.get('epoch', '?')} steps={record.get('agent_steps', '?')} "
            f"retained={retained:.2%} filtered={filtered:.2%}"
        )
        details = {key: value for key, value in record.items() if key.startswith("filter/")}
        print("  " + json.dumps(details, sort_keys=True))


if __name__ == "__main__":
    main()
