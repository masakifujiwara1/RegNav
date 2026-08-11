import torch


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1)


def ade(
    prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    return _masked_mean((prediction[..., :2] - target[..., :2]).norm(dim=-1), valid_mask)


def fde(
    prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    if not valid_mask.any(dim=-1).all():
        raise ValueError("each trajectory must contain a valid pose")
    index = valid_mask.sum(dim=-1) - 1
    batch = torch.arange(len(prediction), device=prediction.device)
    return (prediction[batch, index, :2] - target[batch, index, :2]).norm(dim=-1).mean()


def heading_error(
    prediction: torch.Tensor, target: torch.Tensor, valid_mask: torch.Tensor
) -> torch.Tensor:
    difference = prediction[..., 2] - target[..., 2]
    wrapped = torch.atan2(torch.sin(difference), torch.cos(difference)).abs()
    return _masked_mean(wrapped, valid_mask)


def constraint_violation_rates(
    trajectory: torch.Tensor,
    valid_mask: torch.Tensor,
    interval: float,
    max_velocity: float,
    max_acceleration: float,
    max_curvature: float,
) -> dict[str, torch.Tensor]:
    origin = trajectory.new_zeros((len(trajectory), 1, 3))
    poses = torch.cat((origin, trajectory), dim=1)
    distance = (poses[:, 1:, :2] - poses[:, :-1, :2]).norm(dim=-1)
    velocity = distance / interval
    acceleration = (velocity[:, 1:] - velocity[:, :-1]).abs() / interval
    acceleration_mask = valid_mask[:, 1:] & valid_mask[:, :-1]
    yaw_delta = poses[:, 1:, 2] - poses[:, :-1, 2]
    yaw_delta = torch.atan2(torch.sin(yaw_delta), torch.cos(yaw_delta)).abs()
    curvature = yaw_delta / distance.clamp_min(1e-6)
    return {
        "velocity_violation_rate": _masked_mean(
            (velocity > max_velocity).float(), valid_mask
        ),
        "acceleration_violation_rate": _masked_mean(
            (acceleration > max_acceleration).float(), acceleration_mask
        ),
        "curvature_violation_rate": _masked_mean(
            (curvature > max_curvature).float(), valid_mask
        ),
    }


def trajectory_metrics(
    prediction: torch.Tensor,
    target: torch.Tensor,
    valid_mask: torch.Tensor,
    interval: float,
    max_velocity: float = 0.8,
    max_acceleration: float = 1.0,
    max_curvature: float = 2.0,
) -> dict[str, float]:
    values = {
        "ade": ade(prediction, target, valid_mask),
        "fde": fde(prediction, target, valid_mask),
        "heading_error": heading_error(prediction, target, valid_mask),
        **constraint_violation_rates(
            prediction,
            valid_mask,
            interval,
            max_velocity,
            max_acceleration,
            max_curvature,
        ),
    }
    return {name: float(value) for name, value in values.items()}
