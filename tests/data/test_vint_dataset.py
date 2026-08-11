import pickle

import numpy as np
import torch
from PIL import Image

from regnav.config import ModelConfig
from regnav.data.collate import collate_regnav
from regnav.data.manifest import TrajectoryRecord
from regnav.data.vint_dataset import VintTrajectoryDataset


def _trajectory(tmp_path, frames=10, obstacles=False):
    root = tmp_path / "trajectory"
    root.mkdir()
    for frame in range(frames):
        Image.new("RGB", (16, 12), (frame, 0, 0)).save(root / f"{frame}.jpg")
    with (root / "traj_data.pkl").open("wb") as file:
        pickle.dump(
            {
                "position": np.column_stack(
                    (np.arange(frames, dtype=np.float32) * 0.5, np.zeros(frames))
                ),
                "yaw": np.zeros(frames, np.float32),
            },
            file,
        )
    obstacle_dir = None
    if obstacles:
        obstacle_dir = tmp_path / "obstacles"
        obstacle_dir.mkdir()
        np.save(obstacle_dir / "0.npy", np.array([[1.0, 0.0]], np.float32))
    return TrajectoryRecord(
        trajectory_id="route-a",
        dataset="recon",
        robot="jackal",
        environment="park",
        date="2026-01-01",
        path=str(root),
        sample_period=0.5,
        obstacle_dir=str(obstacle_dir) if obstacle_dir else None,
    )


def _config():
    return ModelConfig(image_size=(32, 18), backbone="mobilenet_v3_small", patch_size=1)


def test_dataset_emits_only_samples_with_complete_future(tmp_path):
    dataset = VintTrajectoryDataset([_trajectory(tmp_path)], _config())

    assert len(dataset) == 2
    sample = dataset[0]
    assert sample["image"].shape == (3, 18, 32)
    assert sample["ego"].shape == (4,)
    assert sample["route_goal"].shape == (6,)
    assert sample["target_trajectory"].shape == (8, 3)
    assert sample["valid_mask"].shape == (8,)
    torch.testing.assert_close(sample["target_trajectory"][-1, 0], torch.tensor(4.0))


def test_collate_pads_optional_obstacle_points(tmp_path):
    sample = VintTrajectoryDataset([_trajectory(tmp_path, obstacles=True)], _config())[0]
    without_points = dict(sample)
    without_points.pop("obstacle_points")

    batch = collate_regnav([sample, without_points])

    assert batch["obstacle_points"].shape == (2, 1, 2)
    torch.testing.assert_close(
        batch["obstacle_points_mask"], torch.tensor([[True], [False]])
    )
