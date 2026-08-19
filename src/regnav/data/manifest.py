from collections import defaultdict
from dataclasses import dataclass
from hashlib import sha256
from math import floor
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

    grouped: dict[tuple[str, str, str], list[TrajectoryRecord]] = defaultdict(list)
    for record in records:
        grouped[(record.dataset, record.environment, record.date)].append(record)
    if not grouped:
        return {}

    def group_hash(group: tuple[str, str, str]) -> int:
        value = "|".join((str(seed), *group))
        return int.from_bytes(sha256(value.encode()).digest()[:8], "big")

    groups = sorted(grouped, key=group_hash)
    group_order = {group: index for index, group in enumerate(groups)}
    split_names = ("train", "validation", "test")
    active = [index for index, ratio in enumerate(ratios) if ratio > 0]
    counts = [floor(len(groups) * ratio) for ratio in ratios]
    if len(groups) <= len(active):
        counts = [0, 0, 0]
        for index in active[: len(groups)]:
            counts[index] = 1
    else:
        for index in active:
            if counts[index]:
                continue
            donor = max(
                (candidate for candidate in active if counts[candidate] > 1),
                key=lambda candidate: (counts[candidate], -candidate),
            )
            counts[donor] -= 1
            counts[index] = 1
        remaining = len(groups) - sum(counts)
        fractional = [
            len(groups) * ratios[index] - floor(len(groups) * ratios[index])
            for index in range(3)
        ]
        order = sorted(active, key=lambda index: (-fractional[index], index))
        for index in order[:remaining]:
            counts[index] += 1

    assignment: dict[tuple[str, str, str], str] = {}
    offset = 0
    for split, count in zip(split_names, counts):
        for group in groups[offset : offset + count]:
            assignment[group] = split
        offset += count

    domains = {
        group: {record_domain(record) for record in group_records}
        for group, group_records in grouped.items()
    }
    all_domains = set().union(*domains.values())

    # Keep date groups atomic while covering available robot/dataset domains in held-out splits.
    for target in ("validation", "test"):
        target_groups = [group for group in groups if assignment[group] == target]
        if not target_groups:
            continue
        for domain in sorted(all_domains):
            target_domains = {
                current_domain
                for group in target_groups
                for current_domain in domains[group]
            }
            if domain in target_domains:
                continue
            candidates = []
            for source in groups:
                if assignment[source] != "train" or domain not in domains[source]:
                    continue
                remaining_train_domains = {
                    current_domain
                    for group in groups
                    if assignment[group] == "train" and group != source
                    for current_domain in domains[group]
                }
                if not domains[source] <= remaining_train_domains:
                    continue
                coverage = len(domains[source] & (target_domains | {domain}))
                candidates.append((coverage, -group_order[source], source))
            if not candidates:
                continue
            _, _, source = max(candidates)
            destination_candidates = []
            for destination in target_groups:
                other_domains = {
                    current_domain
                    for group in target_groups
                    if group != destination
                    for current_domain in domains[group]
                }
                new_domains = other_domains | domains[source]
                if target_domains <= new_domains:
                    destination_candidates.append(
                        (len(domains[destination]), group_order[destination], destination)
                    )
            if not destination_candidates:
                continue
            _, _, destination = min(destination_candidates)
            assignment[source], assignment[destination] = target, "train"
            target_groups = [group for group in groups if assignment[group] == target]

    return {
        record.trajectory_id: assignment[group]
        for group, group_records in grouped.items()
        for record in group_records
    }
