import pickle
from io import BytesIO
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
from PIL import Image
import pytest

from scripts.convert_beonav import ControlSample, integrate_controls, write_vint_trajectory


def _jpeg(color: tuple[int, int, int]) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 6), color).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_integrate_controls_uses_latest_command_between_images():
    controls = [
        ControlSample(timestamp=0.0, linear=1.0, angular=0.0),
        ControlSample(timestamp=1.2, linear=0.0, angular=0.0),
    ]

    positions, yaws = integrate_controls([0.0, 0.5, 1.0, 1.5], controls)

    np.testing.assert_allclose(positions[:, 0], [0.0, 0.5, 1.0, 1.2], atol=1e-6)
    np.testing.assert_allclose(positions[:, 1], 0.0, atol=1e-6)
    np.testing.assert_allclose(yaws, 0.0, atol=1e-6)


def test_write_vint_trajectory_writes_jpegs_and_trajectory_pickle(tmp_path):
    images = [(10.0, _jpeg((255, 0, 0))), (10.5, _jpeg((0, 255, 0))), (11.0, _jpeg((0, 0, 255)))]
    controls = [ControlSample(timestamp=9.0, linear=0.5, angular=0.0)]

    sample_period = write_vint_trajectory(images, controls, tmp_path / "beonav-route")

    assert sample_period == pytest.approx(0.5)
    assert sorted(path.name for path in (tmp_path / "beonav-route").glob("*.jpg")) == [
        "0.jpg",
        "1.jpg",
        "2.jpg",
    ]
    with (tmp_path / "beonav-route" / "traj_data.pkl").open("rb") as file:
        trajectory = pickle.load(file)
    np.testing.assert_allclose(trajectory["position"][:, 0], [0.0, 0.25, 0.5], atol=1e-6)
    assert trajectory["position"].shape == (3, 2)
    assert trajectory["yaw"].shape == (3,)
    with Image.open(tmp_path / "beonav-route" / "1.jpg") as image:
        assert image.size == (8, 6)


def test_write_vint_trajectory_rejects_non_monotonic_images(tmp_path):
    images = [(1.0, _jpeg((0, 0, 0))), (0.5, _jpeg((0, 0, 0)))]

    with pytest.raises(ValueError, match="strictly increasing"):
        write_vint_trajectory(images, [ControlSample(0.0, 0.0, 0.0)], tmp_path / "route")
