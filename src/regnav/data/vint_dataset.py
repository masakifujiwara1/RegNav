from math import ceil
from pathlib import Path
import pickle
from collections.abc import Sequence

import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

from regnav.config import ModelConfig
from regnav.data.geometry import poses_to_local, resample_future, route_goal_from_trajectory
from regnav.data.manifest import TrajectoryRecord


_IMAGE_MEAN = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float32).view(3, 1, 1)
_IMAGE_STD = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float32).view(3, 1, 1)


def preprocess_image(image: Image.Image, image_size: tuple[int, int]) -> torch.Tensor:
    image = image.convert("RGB").resize(image_size, Image.Resampling.BILINEAR)
    tensor = torch.from_numpy(np.asarray(image, dtype=np.float32).copy()).permute(2, 0, 1) / 255
    return (tensor - _IMAGE_MEAN) / _IMAGE_STD


class VintTrajectoryDataset(Dataset[dict[str, torch.Tensor]]):
    def __init__(self, records: Sequence[TrajectoryRecord], config: ModelConfig):
        self.records = records
        self.config = config
        self.index: list[tuple[int, int]] = []
        for record_index, record in enumerate(records):
            last_frame = max(
                int(path.stem)
                for path in Path(record.path).glob("*.jpg")
                if path.stem.isdigit()
            )
            final_offset = ceil(config.horizon / record.sample_period)
            self.index.extend(
                (record_index, frame) for frame in range(last_frame - final_offset + 1)
            )

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        record_index, frame = self.index[index]
        record = self.records[record_index]
        root = Path(record.path)
        with (root / "traj_data.pkl").open("rb") as file:
            trajectory = pickle.load(file)
        positions = np.asarray(trajectory["position"], dtype=np.float32)
        yaws = np.asarray(trajectory["yaw"], dtype=np.float32)
        final_offset = ceil(self.config.horizon / record.sample_period)
        stop = frame + final_offset + 1
        local = poses_to_local(positions[frame:stop], yaws[frame:stop], anchor=0)
        timestamps = np.arange(len(local), dtype=np.float32) * record.sample_period
        query_times = np.arange(1, self.config.num_poses + 1, dtype=np.float32) * self.config.interval
        target = resample_future(timestamps, local, query_times)

        image = Image.open(root / f"{frame}.jpg")
        image_tensor = preprocess_image(image, self.config.image_size)
        sample = {
            "image": image_tensor,
            "ego": torch.from_numpy(self._ego(positions, yaws, frame, record.sample_period)),
            "route_goal": torch.from_numpy(route_goal_from_trajectory(target)),
            "target_trajectory": torch.from_numpy(target),
            "valid_mask": torch.ones(self.config.num_poses, dtype=torch.bool),
        }
        if record.obstacle_dir:
            obstacle_path = Path(record.obstacle_dir) / f"{frame}.npy"
            if obstacle_path.exists():
                sample["obstacle_points"] = torch.from_numpy(
                    np.asarray(np.load(obstacle_path), dtype=np.float32)
                )
        return sample

    @staticmethod
    def _ego(
        positions: np.ndarray, yaws: np.ndarray, frame: int, period: float
    ) -> np.ndarray:
        if frame == 0:
            return np.zeros(4, np.float32)
        velocity = np.linalg.norm(positions[frame] - positions[frame - 1]) / period
        angular = np.arctan2(
            np.sin(yaws[frame] - yaws[frame - 1]),
            np.cos(yaws[frame] - yaws[frame - 1]),
        ) / period
        if frame == 1:
            return np.array([velocity, angular, 0, 0], np.float32)
        previous_velocity = np.linalg.norm(positions[frame - 1] - positions[frame - 2]) / period
        previous_angular = np.arctan2(
            np.sin(yaws[frame - 1] - yaws[frame - 2]),
            np.cos(yaws[frame - 1] - yaws[frame - 2]),
        ) / period
        return np.array(
            [velocity, angular, (velocity - previous_velocity) / period, (angular - previous_angular) / period],
            np.float32,
        )
