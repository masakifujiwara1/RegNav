import numpy as np
import pytest

from regnav.data.geometry import (
    poses_to_local,
    resample_future,
    route_goal_from_trajectory,
)


def test_poses_to_local_rotates_about_anchor():
    positions = np.array([[1, 2], [1, 3]], np.float32)
    yaws = np.array([np.pi / 2, np.pi / 2], np.float32)

    local = poses_to_local(positions, yaws, anchor=0)

    np.testing.assert_allclose(local, [[0, 0, 0], [1, 0, 0]], atol=1e-5)


def test_resample_future_unwraps_yaw():
    poses = np.array([[0, 0, 3.10], [1, 0, -3.10]], np.float32)

    out = resample_future(np.array([0.0, 1.0]), poses, np.array([0.5]))

    assert abs(abs(out[0, 2]) - np.pi) < 0.05


def test_resample_future_rejects_invalid_time_ranges():
    poses = np.zeros((2, 3), np.float32)

    with pytest.raises(ValueError, match="strictly increasing"):
        resample_future(np.array([0.0, 0.0]), poses, np.array([0.0]))
    with pytest.raises(ValueError, match="available range"):
        resample_future(np.array([0.0, 1.0]), poses, np.array([1.1]))


@pytest.mark.parametrize(
    ("yaw", "expected"),
    [
        (0.8, [1, 0, 0]),
        (0.1, [0, 1, 0]),
        (-0.8, [0, 0, 1]),
    ],
)
def test_route_goal_classifies_final_heading(yaw, expected):
    trajectory = np.array([[0.5, 0.1, 0.1], [1.0, 1.0, yaw]], np.float32)

    goal = route_goal_from_trajectory(trajectory, turn_threshold=0.35)

    np.testing.assert_allclose(goal[:3], trajectory[-1])
    np.testing.assert_allclose(goal[3:], expected)
