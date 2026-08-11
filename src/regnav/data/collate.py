from collections.abc import Sequence

import torch

from regnav.contracts import RegNavBatch


def collate_regnav(samples: Sequence[dict[str, torch.Tensor]]) -> RegNavBatch:
    fixed = ("image", "ego", "route_goal", "target_trajectory", "valid_mask")
    batch: RegNavBatch = {key: torch.stack([sample[key] for sample in samples]) for key in fixed}  # type: ignore[typeddict-item]
    point_counts = [len(sample.get("obstacle_points", ())) for sample in samples]
    if not any(point_counts):
        return batch

    points = torch.zeros((len(samples), max(point_counts), 2), dtype=torch.float32)
    mask = torch.zeros((len(samples), max(point_counts)), dtype=torch.bool)
    for index, (sample, count) in enumerate(zip(samples, point_counts)):
        if count:
            points[index, :count] = sample["obstacle_points"]
            mask[index, :count] = True
    batch["obstacle_points"] = points
    batch["obstacle_points_mask"] = mask
    return batch
