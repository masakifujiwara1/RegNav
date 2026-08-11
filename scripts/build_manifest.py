#!/usr/bin/env python3
import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a RegNav JSONL manifest")
    parser.add_argument("root", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--robot", required=True)
    parser.add_argument("--environment", required=True)
    parser.add_argument("--date", required=True)
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
                "date": args.date,
                "path": str(trajectory.resolve()),
                "sample_period": args.sample_period,
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("".join(json.dumps(row) + "\n" for row in rows))


if __name__ == "__main__":
    main()
