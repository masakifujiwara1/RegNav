import torch

from regnav.baselines import route_only
from regnav.evaluate import build_parser, summarize_batch


def test_summary_accepts_route_aware_model_without_beating_zero_route_fde():
    route = torch.tensor([[4.0, 0.0, 0.0, 0.0, 1.0, 0.0]])
    target = route_only(route, num_poses=8)
    target[:, 1:-1, 1] = 0.5
    prediction = target.clone()
    prediction[:, :-1, 0] += 0.01
    batch = {
        "image": torch.zeros(1, 3, 4, 4),
        "ego": torch.zeros(1, 4),
        "route_goal": route,
        "target_trajectory": target,
        "valid_mask": torch.ones(1, 8, dtype=torch.bool),
    }

    summary = summarize_batch(prediction, batch, interval=0.5)

    assert summary["route_only"]["fde"] == 0.0
    assert summary["meets_baseline_acceptance"] is True
    assert "beats_both_baselines" not in summary


def test_summary_reports_target_constraint_rates():
    target = torch.tensor([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]])
    batch = {
        "image": torch.zeros(1, 3, 4, 4),
        "ego": torch.zeros(1, 4),
        "route_goal": torch.tensor([[2.0, 0.0, 0.0, 0.0, 1.0, 0.0]]),
        "target_trajectory": target,
        "valid_mask": torch.ones(1, 2, dtype=torch.bool),
    }

    summary = summarize_batch(target, batch, interval=0.5)

    assert summary["target"]["velocity_violation_rate"] == 1.0


def test_parser_accepts_trajectory_visualization_options():
    args = build_parser().parse_args(
        [
            "--checkpoint",
            "model.pt",
            "--manifest",
            "manifest.jsonl",
            "--split",
            "test",
            "--output",
            "report.json",
            "--visualize-dir",
            "viz",
            "--visualize-count",
            "4",
            "--visualize-worst-k",
            "2",
        ]
    )

    assert str(args.visualize_dir) == "viz"
    assert args.visualize_count == 4
    assert args.visualize_worst_k == 2
