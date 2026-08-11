import torch


def route_only(route_goal: torch.Tensor, num_poses: int) -> torch.Tensor:
    fraction = torch.linspace(
        1 / num_poses,
        1,
        num_poses,
        device=route_goal.device,
        dtype=route_goal.dtype,
    )
    goal = route_goal[:, :3]
    yaw = torch.atan2(torch.sin(goal[:, 2]), torch.cos(goal[:, 2]))
    return torch.stack(
        (
            goal[:, 0, None] * fraction,
            goal[:, 1, None] * fraction,
            yaw[:, None] * fraction,
        ),
        dim=-1,
    )


def constant_velocity(
    ego: torch.Tensor, num_poses: int, interval: float
) -> torch.Tensor:
    times = torch.arange(
        1, num_poses + 1, device=ego.device, dtype=ego.dtype
    ) * interval
    velocity = ego[:, 0, None]
    angular = ego[:, 1, None]
    turning = angular.abs() > 1e-6
    safe_angular = torch.where(turning, angular, torch.ones_like(angular))
    yaw = angular * times
    x = torch.where(
        turning,
        velocity / safe_angular * torch.sin(yaw),
        velocity * times,
    )
    y = torch.where(
        turning,
        velocity / safe_angular * (1 - torch.cos(yaw)),
        torch.zeros_like(x),
    )
    yaw = torch.atan2(torch.sin(yaw), torch.cos(yaw))
    return torch.stack((x, y, yaw), dim=-1)
