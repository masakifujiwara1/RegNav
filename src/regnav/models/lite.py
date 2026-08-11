import torch
from torch import nn

from regnav.config import ModelConfig
from regnav.contracts import RegNavBatch, RegNavOutput


class RegNavLite(nn.Module):
    def __init__(self, config: ModelConfig, backbone: nn.Module):
        super().__init__()
        self.config = config
        self.backbone = backbone
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.image_projection = nn.Sequential(
            nn.Linear(config.backbone_dim, 128), nn.ReLU()
        )
        self.ego_projection = nn.Sequential(nn.Linear(config.ego_features, 32), nn.ReLU())
        self.route_projection = nn.Sequential(
            nn.Linear(config.route_features, 32), nn.ReLU()
        )
        self.trajectory_head = nn.Sequential(
            nn.Linear(192, 256),
            nn.ReLU(),
            nn.Linear(256, config.num_poses * 3),
        )

    def forward(self, batch: RegNavBatch) -> RegNavOutput:
        image = self.pool(self.backbone(batch["image"])).flatten(1)
        features = torch.cat(
            (
                self.image_projection(image),
                self.ego_projection(batch["ego"]),
                self.route_projection(batch["route_goal"]),
            ),
            dim=1,
        )
        trajectory = self.trajectory_head(features).view(-1, self.config.num_poses, 3)
        trajectory = torch.cumsum(trajectory, dim=1)
        trajectory = torch.cat(
            (
                trajectory[..., :2],
                torch.atan2(
                    torch.sin(trajectory[..., 2:]), torch.cos(trajectory[..., 2:])
                ),
            ),
            dim=-1,
        )
        batch_size = trajectory.shape[0]
        return {
            "trajectory": trajectory,
            "proposals": trajectory[:, None],
            "scores": trajectory.new_zeros((batch_size, 1)),
            "score_components": trajectory.new_zeros((batch_size, 1, 4)),
        }
