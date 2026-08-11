from typing import NotRequired, TypedDict

from torch import Tensor


class RegNavBatch(TypedDict):
    image: Tensor
    ego: Tensor
    route_goal: Tensor
    target_trajectory: Tensor
    valid_mask: Tensor
    obstacle_points: NotRequired[Tensor]
    obstacle_points_mask: NotRequired[Tensor]


class RegNavOutput(TypedDict):
    trajectory: Tensor
    proposals: Tensor
    scores: Tensor
    score_components: Tensor
    refinements: NotRequired[list[Tensor]]
