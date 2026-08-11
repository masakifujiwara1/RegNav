import pytest
import torch

from regnav.baselines import constant_velocity, route_only
from regnav.metrics import ade, fde, heading_error


def test_ade_and_fde_respect_valid_mask():
    prediction = torch.tensor([[[1.0, 0.0, 0.0], [9.0, 0.0, 0.0]]])
    target = torch.zeros_like(prediction)
    mask = torch.tensor([[True, False]])

    assert ade(prediction, target, mask).item() == pytest.approx(1.0)
    assert fde(prediction, target, mask).item() == pytest.approx(1.0)


def test_heading_error_wraps_at_pi():
    prediction = torch.tensor([[[0.0, 0.0, 3.13]]])
    target = torch.tensor([[[0.0, 0.0, -3.13]]])

    error = heading_error(prediction, target, torch.tensor([[True]]))

    assert error.item() < 0.03


def test_route_only_baseline_reaches_goal():
    route = torch.tensor([[4.0, 2.0, 0.5, 0.0, 1.0, 0.0]])

    trajectory = route_only(route, num_poses=8)

    torch.testing.assert_close(trajectory[0, -1], route[0, :3])


def test_constant_velocity_integrates_straight_motion():
    ego = torch.tensor([[1.0, 0.0, 0.0, 0.0]])

    trajectory = constant_velocity(ego, num_poses=2, interval=0.5)

    torch.testing.assert_close(
        trajectory[0, :, 0], torch.tensor([0.5, 1.0])
    )
