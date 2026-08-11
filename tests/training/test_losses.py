import torch

from regnav.config import TrainingConfig
from regnav.training.losses import compute_loss


def _batch():
    target = torch.zeros(1, 8, 3)
    target[0, :, 0] = torch.arange(1, 9)
    return {
        "image": torch.zeros(1, 3, 4, 4),
        "ego": torch.zeros(1, 4),
        "route_goal": torch.tensor([[8.0, 0.0, 0.0, 0.0, 1.0, 0.0]]),
        "target_trajectory": target,
        "valid_mask": torch.tensor([[True] * 7 + [False]]),
    }


def _output():
    batch = _batch()
    exact = batch["target_trajectory"].clone()
    exact[0, -1, 0] = 1000.0
    proposals = torch.stack((torch.zeros_like(exact), exact), dim=1).requires_grad_()
    return {
        "trajectory": exact,
        "proposals": proposals,
        "scores": torch.zeros(1, 2, requires_grad=True),
        "score_components": torch.zeros(1, 2, 4, requires_grad=True),
        "refinements": [proposals, proposals],
    }


def test_loss_selects_best_proposal_and_ignores_invalid_pose():
    losses = compute_loss(_output(), _batch(), TrainingConfig())

    assert losses["trajectory"].item() == 0.0
    assert torch.isfinite(losses["total"])
    losses["total"].backward()


def test_missing_privileged_points_skips_collision_loss():
    losses = compute_loss(_output(), _batch(), TrainingConfig())

    assert losses["collision"].item() == 0.0
