#!/usr/bin/env python3
from __future__ import annotations

import argparse
from bisect import bisect_right
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import pickle
import re
from statistics import median
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


_DATE = re.compile(r"(?<!\d)(\d{4}-\d{2}-\d{2})(?!\d)")


@dataclass(frozen=True)
class ControlSample:
    timestamp: float
    linear: float
    angular: float


def _validate_timestamps(timestamps: Sequence[float], name: str) -> None:
    if len(timestamps) < 2:
        raise ValueError(f"{name} must contain at least two timestamps")
    if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
        raise ValueError(f"{name} timestamps must be strictly increasing")


def _wrap_angle(angle: float) -> float:
    return float(np.arctan2(np.sin(angle), np.cos(angle)))


def integrate_controls(
    image_timestamps: Sequence[float], controls: Sequence[ControlSample]
) -> tuple[np.ndarray, np.ndarray]:
    """Integrate zero-order-held unicycle commands, splitting at command events."""
    _validate_timestamps(image_timestamps, "image")
    ordered_controls = sorted(controls, key=lambda control: control.timestamp)
    if not ordered_controls:
        raise ValueError("controls must not be empty")
    control_timestamps = [control.timestamp for control in ordered_controls]
    positions = np.zeros((len(image_timestamps), 2), dtype=np.float32)
    yaws = np.zeros(len(image_timestamps), dtype=np.float32)
    control_index = bisect_right(control_timestamps, image_timestamps[0]) - 1
    for index, (start, end) in enumerate(zip(image_timestamps, image_timestamps[1:]), 1):
        segment_start = start
        position = positions[index - 1].astype(np.float64)
        yaw = float(yaws[index - 1])
        while segment_start < end:
            while (
                control_index + 1 < len(ordered_controls)
                and ordered_controls[control_index + 1].timestamp <= segment_start
            ):
                control_index += 1
            if control_index < 0:
                linear = angular = 0.0
            else:
                control = ordered_controls[control_index]
                linear, angular = control.linear, control.angular
            next_control = (
                ordered_controls[control_index + 1].timestamp
                if control_index + 1 < len(ordered_controls)
                else end
            )
            segment_end = min(end, next_control)
            dt = segment_end - segment_start
            if abs(angular) < 1e-8:
                position += (linear * np.cos(yaw) * dt, linear * np.sin(yaw) * dt)
            else:
                next_yaw = yaw + angular * dt
                radius = linear / angular
                position += (
                    radius * (np.sin(next_yaw) - np.sin(yaw)),
                    radius * (-np.cos(next_yaw) + np.cos(yaw)),
                )
                yaw = _wrap_angle(next_yaw)
            segment_start = segment_end
        positions[index] = position.astype(np.float32)
        yaws[index] = yaw
    return positions, yaws


def _write_frames(output_dir: Path, payloads: Iterable[bytes], positions: np.ndarray, yaws: np.ndarray) -> int:
    output_dir.mkdir(parents=True, exist_ok=True)
    count = 0
    for count, payload in enumerate(payloads, 1):
        try:
            with Image.open(BytesIO(payload)) as image:
                image.verify()
        except Exception as error:
            raise ValueError(f"image frame {count - 1} is not a valid image") from error
        (output_dir / f"{count - 1}.jpg").write_bytes(payload)
    if count != len(positions):
        raise ValueError(f"image count changed during conversion: expected {len(positions)}, got {count}")
    with (output_dir / "traj_data.pkl").open("wb") as file:
        pickle.dump({"position": positions, "yaw": yaws}, file, protocol=pickle.HIGHEST_PROTOCOL)
    return count


def write_vint_trajectory(
    images: Sequence[tuple[float, bytes]],
    controls: Sequence[ControlSample],
    output_dir: Path,
) -> float:
    """Write timestamped JPEGs and integrated trajectory in ViNT layout."""
    timestamps = [timestamp for timestamp, _ in images]
    _validate_timestamps(timestamps, "image")
    positions, yaws = integrate_controls(timestamps, controls)
    _write_frames(output_dir, (payload for _, payload in images), positions, yaws)
    return float(median(np.diff(np.asarray(timestamps, dtype=np.float64))))


def _trajectory_date(trajectory_id: str, explicit_date: str | None) -> str:
    value = explicit_date or (_DATE.search(trajectory_id).group(1) if _DATE.search(trajectory_id) else None)
    if value is None:
        raise ValueError("trajectory id has no YYYY-MM-DD; pass --date")
    date.fromisoformat(value)
    return value


def _read_bag_controls(bag_path: Path, control_topic: str) -> list[ControlSample]:
    from rosbags.highlevel import AnyReader

    controls = []
    with AnyReader([bag_path]) as reader:
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic != control_topic:
                continue
            message = reader.deserialize(rawdata, connection.msgtype)
            controls.append(
                ControlSample(
                    timestamp=timestamp / 1e9,
                    linear=float(message.linear.x),
                    angular=float(message.angular.z),
                )
            )
    return controls


def _read_bag_image_timestamps(bag_path: Path, image_topic: str) -> list[float]:
    from rosbags.highlevel import AnyReader

    timestamps = []
    with AnyReader([bag_path]) as reader:
        for connection, timestamp, _ in reader.messages():
            if connection.topic == image_topic:
                timestamps.append(timestamp / 1e9)
    return timestamps


def _iter_bag_images(bag_path: Path, image_topic: str) -> Iterable[bytes]:
    from rosbags.highlevel import AnyReader

    with AnyReader([bag_path]) as reader:
        for connection, _, rawdata in reader.messages():
            if connection.topic != image_topic:
                continue
            message = reader.deserialize(rawdata, connection.msgtype)
            yield bytes(message.data)


def convert_beonav_bag(
    bag_path: Path,
    output_root: Path,
    manifest_path: Path,
    *,
    dataset: str = "beonav",
    robot: str = "wheeled",
    environment: str = "usc-campus",
    trajectory_id: str | None = None,
    trajectory_date: str | None = None,
    image_topic: str = "/cam1/color/image_raw/compressed",
    control_topic: str = "/cmd_vel",
    source_url: str | None = None,
) -> dict[str, object]:
    if not bag_path.is_file():
        raise FileNotFoundError(bag_path)
    if manifest_path.exists():
        raise FileExistsError(f"manifest already exists: {manifest_path}")
    trajectory_id = trajectory_id or bag_path.stem
    output_dir = output_root / trajectory_id
    if output_dir.exists():
        raise FileExistsError(f"trajectory output already exists: {output_dir}")
    controls = _read_bag_controls(bag_path, control_topic)
    image_timestamps = _read_bag_image_timestamps(bag_path, image_topic)
    if not controls:
        raise ValueError(f"no control messages found on {control_topic}")
    _validate_timestamps(image_timestamps, "image")
    positions, yaws = integrate_controls(image_timestamps, controls)
    image_count = _write_frames(output_dir, _iter_bag_images(bag_path, image_topic), positions, yaws)
    sample_period = float(median(np.diff(np.asarray(image_timestamps, dtype=np.float64))))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "trajectory_id": trajectory_id,
                "dataset": dataset,
                "robot": robot,
                "environment": environment,
                "date": _trajectory_date(trajectory_id, trajectory_date),
                "path": str(output_dir.resolve()),
                "sample_period": sample_period,
            }
        )
        + "\n"
    )
    digest = sha256()
    with bag_path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    provenance = {
        "source_bag": str(bag_path.resolve()),
        "source_sha256": digest.hexdigest(),
        "source_url": source_url,
        "image_topic": image_topic,
        "control_topic": control_topic,
        "control_model": "latest /cmd_vel linear.x and angular.z, unicycle integration",
        "image_count": image_count,
        "sample_period": sample_period,
    }
    (manifest_path.parent / f"{manifest_path.stem}-provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    return {"trajectory_id": trajectory_id, "image_count": image_count, "sample_period": sample_period}


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert one BeoNav ROS bag to ViNT format")
    parser.add_argument("bag", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--dataset", default="beonav")
    parser.add_argument("--robot", default="wheeled")
    parser.add_argument("--environment", default="usc-campus")
    parser.add_argument("--trajectory-id")
    parser.add_argument("--date")
    parser.add_argument("--image-topic", default="/cam1/color/image_raw/compressed")
    parser.add_argument("--control-topic", default="/cmd_vel")
    parser.add_argument("--source-url")
    args = parser.parse_args()
    print(json.dumps(convert_beonav_bag(
        args.bag,
        args.output_root,
        args.manifest,
        dataset=args.dataset,
        robot=args.robot,
        environment=args.environment,
        trajectory_id=args.trajectory_id,
        trajectory_date=args.date,
        image_topic=args.image_topic,
        control_topic=args.control_topic,
        source_url=args.source_url,
    ), sort_keys=True))


if __name__ == "__main__":
    main()
