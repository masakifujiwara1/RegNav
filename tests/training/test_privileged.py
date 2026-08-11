import torch

from regnav.training.privileged import collision_free_labels, component_targets


def test_point_inside_oriented_footprint_marks_collision():
    proposals = torch.zeros(1, 2, 8, 3)
    proposals[:, :, :, 0] = torch.arange(1, 9)
    points = torch.tensor([[[1.0, 0.0], [20.0, 20.0]]])
    mask = torch.tensor([[True, True]])

    labels = collision_free_labels(proposals, points, mask, footprint=(1.0, 0.7))

    assert labels.shape == (1, 2)
    assert labels[0, 0].item() == 0.0


def test_component_targets_follow_fixed_order():
    proposals = torch.zeros(1, 1, 8, 3)
    proposals[0, 0, -1, 0] = 4.0
    route = torch.tensor([[4.0, 0.0, 0.0, 0.0, 1.0, 0.0]])

    targets = component_targets(proposals, route, torch.ones(1, 1))

    torch.testing.assert_close(targets[0, 0, :3], torch.ones(3))
    assert 0.0 < targets[0, 0, 3] <= 1.0
