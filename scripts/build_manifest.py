#!/usr/bin/env python3
import argparse
from datetime import date
import json
from pathlib import Path
import re


_DATE = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)")


def date_from_trajectory_id(trajectory_id: str) -> str:
    match = _DATE.search(trajectory_id)
    if match is None:
        raise ValueError(f"trajectory {trajectory_id!r} does not contain YYYY-MM-DD")
    value = match.group(1)
    try:
        date.fromisoformat(value)
    except ValueError as error:
        raise ValueError(
            f"trajectory {trajectory_id!r} contains invalid date {value!r}"
        ) from error
    return value


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a RegNav JSONL manifest")
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--robot", required=True)
    parser.add_argument("--environment", required=True)
    date_source = parser.add_mutually_exclusive_group(required=True)
    date_source.add_argument("--date")
    date_source.add_argument("--date-from-trajectory", action="store_true")
    parser.add_argument("--sample-period", required=True, type=float)
    args = parser.parse_args()

    rows = []
    for trajectory_file in sorted(args.root.glob("*/traj_data.pkl")):
        trajectory = trajectory_file.parent
        rows.append(
            {
                "trajectory_id": trajectory.name,
                "dataset": args.dataset,
                "robot": args.robot,
                "environment": args.environment,
                "date": (
                    date_from_trajectory_id(trajectory.name)
                    if args.date_from_trajectory
                    else args.date
                ),
                "path": str(trajectory.resolve()),
                "sample_period": args.sample_period,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))


if __name__ == "__main__":
    main()
