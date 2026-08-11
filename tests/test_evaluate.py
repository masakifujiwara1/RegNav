import torch

from regnav.baselines import route_only
from regnav.evaluate import summarize_batch


def test_summary_contains_model_and_baseline_metrics():
    route = torch.tensor([[4.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    target = route_only(route, num_poses=8)
    batch = {
        "image": torch.zeros(1, 3, 4, 4),
        "ego": torch.zeros(1, 4),
        "route_goal": route,
        "target_trajectory": target,
        "valid_mask": torch.ones(1, 8, dtype=torch.bool),
    }

    summary = summarize_batch(target, batch, interval=0.5)

    assert summary["model"]["ade"] == 0.0
    assert summary["constant_velocity"]["ade"] > 0.0
    assert summary["route_only"]["fde"] == 0.0
    assert summary["beats_both_baselines"] is False
