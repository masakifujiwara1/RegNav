#!/usr/bin/env python3
from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import date
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path
import pickle
from statistics import median
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


SCAND_DOI = "doi:10.18738/T8/0PRYRH"
_ROBOT_TOPICS = {
    "jackal": (
        "/camera/rgb/image_raw/compressed",
        "/jackal_velocity_controller/odom",
    ),
    "spot": ("/image_raw/compressed", "/odom"),
}


@dataclass(frozen=True)
class OdomSample:
    timestamp: float
    x: float
    y: float
    yaw: float


def default_topics(robot: str) -> tuple[str, str]:
    try:
        return _ROBOT_TOPICS[robot]
    except KeyError as error:
        raise ValueError(f"unsupported robot: {robot!r}; choose jackal or spot") from error


def _validate_timestamps(timestamps: Sequence[float], name: str) -> None:
    if len(timestamps) < 2:
        raise ValueError(f"{name} must contain at least two timestamps")
    if any(current <= previous for previous, current in zip(timestamps, timestamps[1:])):
        raise ValueError(f"{name} timestamps must be strictly increasing")


def _wrap_angle(angle: np.ndarray | float) -> np.ndarray | float:
    return np.arctan2(np.sin(angle), np.cos(angle))


def interpolate_odometry(
    image_timestamps: Sequence[float], odometry: Sequence[OdomSample]
) -> tuple[np.ndarray, np.ndarray]:
    """Interpolate measured odometry poses at each image timestamp."""
    _validate_timestamps(image_timestamps, "image")
    odometry = sorted(odometry, key=lambda sample: sample.timestamp)
    odometry_timestamps = [sample.timestamp for sample in odometry]
    _validate_timestamps(odometry_timestamps, "odometry")
    source_times = np.asarray(odometry_timestamps, dtype=np.float64)
    image_times = np.asarray(image_timestamps, dtype=np.float64)
    edge_tolerance = 0.1
    if (
        image_times[0] < source_times[0] - edge_tolerance
        or image_times[-1] > source_times[-1] + edge_tolerance
    ):
        raise ValueError("image timestamps must be inside the odometry range")
    positions = np.column_stack(
        (
            np.interp(image_times, source_times, [sample.x for sample in odometry]),
            np.interp(image_times, source_times, [sample.y for sample in odometry]),
        )
    ).astype(np.float32)
    yaw = np.interp(
        image_times,
        source_times,
        np.unwrap(np.asarray([sample.yaw for sample in odometry], dtype=np.float64)),
    )
    return positions, np.asarray(_wrap_angle(yaw), dtype=np.float32)


def _write_frames(
    output_dir: Path,
    payloads: Iterable[bytes],
    positions: np.ndarray,
    yaws: np.ndarray,
) -> int:
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
        raise ValueError(
            f"image count changed during conversion: expected {len(positions)}, got {count}"
        )
    with (output_dir / "traj_data.pkl").open("wb") as file:
        pickle.dump(
            {"position": positions, "yaw": yaws},
            file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    return count


def write_scand_trajectory(
    images: Sequence[tuple[float, bytes]],
    odometry: Sequence[OdomSample],
    output_dir: Path,
) -> float:
    """Write timestamped JPEGs and measured trajectory in ViNT layout."""
    timestamps = [timestamp for timestamp, _ in images]
    _validate_timestamps(timestamps, "image")
    positions, yaws = interpolate_odometry(timestamps, odometry)
    _write_frames(output_dir, (payload for _, payload in images), positions, yaws)
    return float(median(np.diff(np.asarray(timestamps, dtype=np.float64))))


def _quaternion_yaw(quaternion: object) -> float:
    x, y, z, w = (
        float(quaternion.x),
        float(quaternion.y),
        float(quaternion.z),
        float(quaternion.w),
    )
    return float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))


def _read_bag_odometry(bag_path: Path, odometry_topic: str) -> list[OdomSample]:
    from rosbags.highlevel import AnyReader

    odometry = []
    with AnyReader([bag_path]) as reader:
        for connection, timestamp, rawdata in reader.messages():
            if connection.topic != odometry_topic:
                continue
            message = reader.deserialize(rawdata, connection.msgtype)
            pose = message.pose.pose
            odometry.append(
                OdomSample(
                    timestamp=timestamp / 1e9,
                    x=float(pose.position.x),
                    y=float(pose.position.y),
                    yaw=_quaternion_yaw(pose.orientation),
                )
            )
    return odometry


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


def _sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def convert_scand_bag(
    bag_path: Path,
    output_root: Path,
    manifest_path: Path,
    *,
    robot: str,
    trajectory_date: str,
    dataset: str = "scand",
    environment: str = "ut-campus",
    trajectory_id: str | None = None,
    image_topic: str | None = None,
    odometry_topic: str | None = None,
    source_url: str | None = None,
) -> dict[str, object]:
    if not bag_path.is_file():
        raise FileNotFoundError(bag_path)
    if manifest_path.exists():
        raise FileExistsError(f"manifest already exists: {manifest_path}")
    default_image_topic, default_odometry_topic = default_topics(robot)
    image_topic = image_topic or default_image_topic
    odometry_topic = odometry_topic or default_odometry_topic
    date.fromisoformat(trajectory_date)
    trajectory_id = trajectory_id or f"scand-{robot}-{bag_path.stem}"
    output_dir = output_root / trajectory_id
    if output_dir.exists():
        raise FileExistsError(f"trajectory output already exists: {output_dir}")

    image_timestamps = _read_bag_image_timestamps(bag_path, image_topic)
    odometry = _read_bag_odometry(bag_path, odometry_topic)
    if not image_timestamps:
        raise ValueError(f"no image messages found on {image_topic}")
    if not odometry:
        raise ValueError(f"no odometry messages found on {odometry_topic}")
    _validate_timestamps(image_timestamps, "image")
    positions, yaws = interpolate_odometry(image_timestamps, odometry)
    image_count = _write_frames(
        output_dir,
        _iter_bag_images(bag_path, image_topic),
        positions,
        yaws,
    )
    sample_period = float(
        median(np.diff(np.asarray(image_timestamps, dtype=np.float64)))
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(
            {
                "trajectory_id": trajectory_id,
                "dataset": dataset,
                "robot": robot,
                "environment": environment,
                "date": trajectory_date,
                "path": str(output_dir.resolve()),
                "sample_period": sample_period,
            },
            sort_keys=True,
        )
        + "\n"
    )
    provenance = {
        "source_bag": str(bag_path.resolve()),
        "source_sha256": _sha256_file(bag_path),
        "source_url": source_url,
        "source_dataset_doi": SCAND_DOI,
        "robot": robot,
        "image_topic": image_topic,
        "odometry_topic": odometry_topic,
        "pose_source": "nav_msgs/Odometry.pose.pose interpolated at image timestamps",
        "image_count": image_count,
        "odometry_count": len(odometry),
        "sample_period": sample_period,
    }
    (manifest_path.parent / f"{manifest_path.stem}-provenance.json").write_text(
        json.dumps(provenance, indent=2, sort_keys=True) + "\n"
    )
    return {
        "trajectory_id": trajectory_id,
        "robot": robot,
        "image_count": image_count,
        "odometry_count": len(odometry),
        "sample_period": sample_period,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert one SCAND ROS bag to ViNT format"
    )
    parser.add_argument("bag", type=Path)
    parser.add_argument("output_root", type=Path)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--robot", choices=tuple(_ROBOT_TOPICS), required=True)
    parser.add_argument("--date", required=True, help="recording date in YYYY-MM-DD")
    parser.add_argument("--dataset", default="scand")
    parser.add_argument("--environment", default="ut-campus")
    parser.add_argument("--trajectory-id")
    parser.add_argument("--image-topic")
    parser.add_argument("--odometry-topic")
    parser.add_argument("--source-url")
    args = parser.parse_args()
    print(
        json.dumps(
            convert_scand_bag(
                args.bag,
                args.output_root,
                args.manifest,
                robot=args.robot,
                trajectory_date=args.date,
                dataset=args.dataset,
                environment=args.environment,
                trajectory_id=args.trajectory_id,
                image_topic=args.image_topic,
                odometry_topic=args.odometry_topic,
                source_url=args.source_url,
            ),
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
