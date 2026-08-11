import torch
from torch.nn import functional as F

from regnav.config import TrainingConfig
from regnav.contracts import RegNavBatch, RegNavOutput
from regnav.training.privileged import collision_free_labels, component_targets


def _proposal_errors(
    proposals: torch.Tensor, target: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    distances = (proposals[..., :2] - target[:, None, :, :2]).norm(dim=-1)
    return (distances * mask[:, None]).sum(dim=-1) / mask.sum(dim=-1, keepdim=True).clamp_min(1)


def _best(proposals: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    index = _proposal_errors(proposals, target, mask).argmin(dim=-1)
    return proposals[torch.arange(len(proposals), device=proposals.device), index]


def _masked_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return (values * mask).sum() / mask.sum().clamp_min(1)


def compute_loss(
    output: RegNavOutput,
    batch: RegNavBatch,
    config: TrainingConfig,
    footprint: tuple[float, float] = (1.0, 0.7),
    diversity_margin: float = 0.5,
) -> dict[str, torch.Tensor]:
    proposals = output["proposals"]
    target = batch["target_trajectory"]
    mask = batch["valid_mask"]
    selected = _best(proposals, target, mask)

    trajectory = _masked_mean(
        F.smooth_l1_loss(selected[..., :2], target[..., :2], reduction="none").sum(-1),
        mask,
    )
    heading = _masked_mean(1 - torch.cos(selected[..., 2] - target[..., 2]), mask)
    acceleration = selected[:, 2:, :2] - 2 * selected[:, 1:-1, :2] + selected[:, :-2, :2]
    acceleration_mask = mask[:, 2:] & mask[:, 1:-1] & mask[:, :-2]
    smoothness = _masked_mean(acceleration.square().sum(-1), acceleration_mask)

    if proposals.shape[1] > 1:
        distances = torch.cdist(proposals[..., -1, :2], proposals[..., -1, :2])
        pairs = ~torch.eye(proposals.shape[1], dtype=torch.bool, device=proposals.device)
        diversity = F.relu(diversity_margin - distances[:, pairs]).mean()
    else:
        diversity = proposals.sum() * 0

    refinement = proposals.sum() * 0
    for intermediate in output.get("refinements", [])[:-1]:
        prediction = _best(intermediate, target, mask)
        refinement = refinement + _masked_mean(
            F.smooth_l1_loss(prediction[..., :2], target[..., :2], reduction="none").sum(-1),
            mask,
        )

    errors = _proposal_errors(proposals, target, mask)
    ranking = F.kl_div(
        F.log_softmax(output["scores"], dim=-1),
        F.softmax(-errors.detach(), dim=-1),
        reduction="batchmean",
    )
    collision = proposals.sum() * 0
    if "obstacle_points" in batch:
        collision_labels = collision_free_labels(
            proposals,
            batch["obstacle_points"],
            batch["obstacle_points_mask"],
            footprint,
        )
        collision = F.binary_cross_entropy(
            output["score_components"][..., 0].clamp(1e-6, 1 - 1e-6),
            collision_labels,
        )
    else:
        collision_labels = torch.ones_like(output["scores"])
    targets = component_targets(proposals, batch["route_goal"], collision_labels)
    components = F.mse_loss(output["score_components"][..., 1:], targets[..., 1:])
    scorer = ranking + collision + components

    total = (
        config.trajectory_weight * trajectory
        + config.heading_weight * heading
        + config.smoothness_weight * smoothness
        + config.diversity_weight * diversity
        + config.refinement_weight * refinement
        + config.scorer_weight * scorer
    )
    return {
        "trajectory": trajectory,
        "heading": heading,
        "smoothness": smoothness,
        "diversity": diversity,
        "refinement": refinement,
        "ranking": ranking,
        "collision": collision,
        "components": components,
        "total": total,
    }
