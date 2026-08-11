import torch


def collision_free_labels(
    proposals: torch.Tensor,
    obstacle_points: torch.Tensor,
    obstacle_mask: torch.Tensor,
    footprint: tuple[float, float],
) -> torch.Tensor:
    delta = obstacle_points[:, None, None] - proposals[..., None, :2]
    yaw = proposals[..., 2, None]
    cosine, sine = yaw.cos(), yaw.sin()
    local_x = cosine * delta[..., 0] + sine * delta[..., 1]
    local_y = -sine * delta[..., 0] + cosine * delta[..., 1]
    inside = (
        (local_x.abs() <= footprint[0] / 2)
        & (local_y.abs() <= footprint[1] / 2)
        & obstacle_mask[:, None, None]
    )
    return (~inside.any(dim=(-1, -2))).float().detach()


def component_targets(
    proposals: torch.Tensor,
    route_goal: torch.Tensor,
    collision_labels: torch.Tensor,
) -> torch.Tensor:
    endpoint = proposals[..., -1, :2]
    goal = route_goal[:, None, :2]
    distance = goal.norm(dim=-1).clamp_min(1e-6)
    progress = (endpoint * goal).sum(dim=-1).div(distance.square()).clamp(0, 1)
    route = torch.exp(-(endpoint - goal).norm(dim=-1))
    acceleration = proposals[..., 2:, :2] - 2 * proposals[..., 1:-1, :2] + proposals[..., :-2, :2]
    comfort = torch.exp(-acceleration.square().mean(dim=(-1, -2)))
    return torch.stack((collision_labels, progress, route, comfort), dim=-1).detach()
