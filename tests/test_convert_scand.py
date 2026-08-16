from io import BytesIO
from pathlib import Path
import pickle
import sys

sys.path.insert(0, str(Path(__file__).parents[1]))

import numpy as np
from PIL import Image
import pytest

from scripts.convert_scand import (
    OdomSample,
    default_topics,
    interpolate_odometry,
    write_scand_trajectory,
)


def _jpeg(color: tuple[int, int, int]) -> bytes:
    buffer = BytesIO()
    Image.new("RGB", (8, 6), color).save(buffer, format="JPEG")
    return buffer.getvalue()


def test_default_topics_keep_jackal_and_spot_distinct():
    assert default_topics("jackal") == (
        "/camera/rgb/image_raw/compressed",
        "/jackal_velocity_controller/odom",
    )
    assert default_topics("spot") == ("/image_raw/compressed", "/odom")


def test_interpolate_odometry_handles_image_times_and_yaw_wrap():
    odometry = [
        OdomSample(0.0, 10.0, 20.0, 3.0),
        OdomSample(1.0, 12.0, 24.0, -3.0),
    ]

    positions, yaws = interpolate_odometry([0.0, 0.5, 1.0], odometry)

    np.testing.assert_allclose(positions, [[10.0, 20.0], [11.0, 22.0], [12.0, 24.0]], atol=1e-6)
    np.testing.assert_allclose(yaws, [3.0, np.pi, -3.0], atol=1e-6)


def test_write_scand_trajectory_writes_images_and_odom_pickle(tmp_path):
    images = [(10.0, _jpeg((255, 0, 0))), (10.5, _jpeg((0, 255, 0))), (11.0, _jpeg((0, 0, 255)))]
    odometry = [OdomSample(9.0, 4.0, 5.0, 0.0), OdomSample(12.0, 5.5, 5.0, 0.3)]

    sample_period = write_scand_trajectory(images, odometry, tmp_path / "scand-route")

    assert sample_period == pytest.approx(0.5)
    with (tmp_path / "scand-route" / "traj_data.pkl").open("rb") as file:
        trajectory = pickle.load(file)
    np.testing.assert_allclose(trajectory["position"][:, 0], [4.5, 4.75, 5.0], atol=1e-6)
    np.testing.assert_allclose(trajectory["position"][:, 1], 5.0, atol=1e-6)
    np.testing.assert_allclose(trajectory["yaw"], [0.1, 0.15, 0.2], atol=1e-6)
    assert sorted(path.name for path in (tmp_path / "scand-route").glob("*.jpg")) == [
        "0.jpg",
        "1.jpg",
        "2.jpg",
    ]


def test_interpolate_odometry_rejects_images_outside_odom_range():
    with pytest.raises(ValueError, match="odometry range"):
        interpolate_odometry(
            [0.0, 2.0],
            [OdomSample(0.5, 0.0, 0.0, 0.0), OdomSample(1.0, 1.0, 0.0, 0.0)],
        )

def test_interpolate_odometry_clamps_small_end_sync_gap():
    positions, _ = interpolate_odometry(
        [0.0, 1.05],
        [OdomSample(0.0, 0.0, 0.0, 0.0), OdomSample(1.0, 1.0, 0.0, 0.0)],
    )

    np.testing.assert_allclose(positions[:, 0], [0.0, 1.0], atol=1e-6)
