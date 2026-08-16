from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path
from typing import Sequence


@dataclass(frozen=True)
class TrajectoryRecord:
    trajectory_id: str
    dataset: str
    robot: str
    environment: str
    date: str
    path: str
    sample_period: float
    obstacle_dir: str | None = None

    def __post_init__(self) -> None:
        if not all(
            (self.trajectory_id, self.dataset, self.robot, self.environment, self.date, self.path)
        ):
            raise ValueError("manifest text fields cannot be empty")
        if self.sample_period <= 0:
            raise ValueError("sample_period must be positive")


def record_domain(record: TrajectoryRecord) -> str:
    """Return a stable dataset/robot label for balancing and reports."""
    return f"{record.dataset}/{record.robot}"


def load_manifest(path: Path) -> list[TrajectoryRecord]:
    required = {
        "trajectory_id",
        "dataset",
        "robot",
        "environment",
        "date",
        "path",
        "sample_period",
    }
    allowed = required | {"obstacle_dir"}
    records = []
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        missing = required - row.keys()
        extra = row.keys() - allowed
        if missing:
            raise ValueError(f"line {line_number} missing {sorted(missing)[0]}")
        if extra:
            raise ValueError(f"line {line_number} has unknown field {sorted(extra)[0]}")
        records.append(TrajectoryRecord(**row))
    return records


def group_split(
    records: Sequence[TrajectoryRecord],
    seed: int,
    ratios: tuple[float, float, float],
) -> dict[str, str]:
    if len(ratios) != 3 or any(ratio < 0 for ratio in ratios) or abs(sum(ratios) - 1) > 1e-9:
        raise ValueError("ratios must be non-negative and sum to one")
    boundaries = (ratios[0], ratios[0] + ratios[1])
    result = {}
    for record in records:
        group = f"{seed}|{record.dataset}|{record.environment}|{record.date}"
        value = int.from_bytes(sha256(group.encode()).digest()[:8], "big") / 2**64
        result[record.trajectory_id] = (
            "train" if value < boundaries[0] else "validation" if value < boundaries[1] else "test"
        )
    return result
